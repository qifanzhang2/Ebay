# ebaynert.py: Train and run NER model for eBay German Car Parts competition
import csv
import argparse
import math
import torch
from transformers import XLMRobertaTokenizerFast, XLMRobertaForTokenClassification, XLMRobertaForMaskedLM
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# Argument parser for optional configurations
parser = argparse.ArgumentParser(description="Train and run NER model for eBay German car parts")
parser.add_argument('--do_mlm', action='store_true', help="Perform domain-adaptive MLM pretraining on listing titles")
parser.add_argument('--mlm_epochs', type=float, default=1.0, help="Number of epochs for MLM pretraining (can be fractional)")
parser.add_argument('--ensemble_models', type=int, default=1, help="Number of NER models to train for ensembling")
parser.add_argument('--threshold', type=float, default=0.5, help="Confidence threshold for keeping predictions (for F0.2 optimization)")
parser.add_argument('--output_file', type=str, default="submission.tsv", help="Output file for predictions")
args = parser.parse_args()

# Initialize tokenizer for XLM-RoBERTa (use a fast tokenizer for efficiency)
tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-large')

# 1. Load and preprocess the tagged training data
print("Loading and preprocessing training data...")
train_tokens_per_record = []   # list of token lists (each record's tokens)
train_labels_per_record = []   # list of label lists (each record's BIO labels)
unique_tags = set()            # collect all unique aspect tags (including "O")

with open("Tagged_Titles_Train.tsv", newline='', encoding='utf-8') as f:
    # Use CSV reader with tab delimiter and no default NA values to correctly handle empty fields:contentReference[oaicite:20]{index=20}
    reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    header = next(reader, None)  # skip header
    current_id = None
    tokens = []
    tags = []
    for row in reader:
        if not row:
            continue
        rec_id, cat_id, title, token, tag = row[0], row[1], row[2], row[3], row[4]
        # If we move to a new record, save the previous one
        if current_id is None:
            current_id = rec_id
        if rec_id != current_id:
            # finalize the previous record
            train_tokens_per_record.append(tokens)
            train_labels_per_record.append(tags)
            # start a new record
            current_id = rec_id
            tokens = []
            tags = []
        # Append token and tag (tag may be '' for continuation)
        tokens.append(token)
        if tag == "":  
            tags.append("")  # empty tag (continuation marker)
        else:
            tags.append(tag)
            unique_tags.add(tag)
    # Add the last record
    if tokens:
        train_tokens_per_record.append(tokens)
        train_labels_per_record.append(tags)

# Include the "O" tag explicitly in unique_tags (it should be present if any token was labeled O)
unique_tags.add("O")

# Now convert partial tags to BIO format labels
label_list = ["O"]  # start with "O" as label 0
for tag in sorted(unique_tags):
    if tag == "" or tag == "O":
        continue
    # For each aspect tag (except O), add B- and I- labels
    label_list.append("B-" + tag)
    label_list.append("I-" + tag)
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}

# Convert the train tags to BIO labels using the rules:
# - Non-empty tag -> B-tag (or I-tag if previous token had same tag and wasn't separated? But by spec, if non-empty, it's a new entity):contentReference[oaicite:21]{index=21}
# - Empty tag -> continuation (I- of the last non-empty tag)
train_bio_labels_per_record = []  # numeric label IDs for each token
for tags in train_labels_per_record:
    bio_labels = []
    last_tag_type = None  # store last seen non-empty tag
    for tag in tags:
        if tag == "":  # continuation of previous tag
            if last_tag_type is None:
                # Edge case: empty tag at start (should not happen in well-formed data)
                bio_labels.append(label2id["O"])
            else:
                if last_tag_type == "O":
                    # If the last entity was O (outside), continue as O (though consecutive O's are just separate O tokens)
                    bio_labels.append(label2id["O"])
                else:
                    # Continue the previous aspect entity
                    bio_labels.append(label2id["I-" + last_tag_type])
        else:
            # Non-empty tag present
            if tag == "O":
                bio_labels.append(label2id["O"])
                last_tag_type = "O"
            else:
                # Start of a new aspect entity (B-tag)
                bio_labels.append(label2id["B-" + tag])
                last_tag_type = tag
    train_bio_labels_per_record.append(bio_labels)

# Optionally, create a validation split from the training data (e.g., 10% for threshold tuning)
num_records = len(train_tokens_per_record)
val_size = max(1, int(0.1 * num_records))  # 10% of records as validation
val_tokens = []
val_bio_labels = []
if val_size > 0:
    val_tokens = train_tokens_per_record[-val_size:]
    val_bio_labels = train_bio_labels_per_record[-val_size:]
    train_tokens_per_record = train_tokens_per_record[:-val_size]
    train_bio_labels_per_record = train_bio_labels_per_record[:-val_size]

# 2. Domain-Adaptive MLM Pretraining (if enabled)
if args.do_mlm:
    print("Starting domain-adaptive MLM pretraining on listing titles...")
    # Load unlabeled titles from Listing_Titles.tsv
    # We will read the file and extract the titles. The file has columns: Record Number, Category Id, Title:contentReference[oaicite:22]{index=22}.
    titles = []
    with open("Listing_Titles.tsv", newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        header = next(reader, None)
        for row in reader:
            if not row: 
                continue
            title = row[2]  # Title column
            titles.append(title)
    # Tokenize the titles. We use the tokenizer to create inputs for MLM. 
    # To handle memory, we won't tokenize all 2M titles at once, we will use the DataCollator to mask on the fly.
    # We'll create a torch Dataset to yield tokenized batches.
    class TitlesDataset(torch.utils.data.Dataset):
        def __init__(self, texts, tokenizer):
            self.texts = texts
            self.tokenizer = tokenizer
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, idx):
            text = str(self.texts[idx])
            # Tokenize text into subwords. Use truncation to limit length for efficiency (e.g., 128 tokens).
            encoding = self.tokenizer(text, truncation=True, max_length=128, return_tensors='pt')
            # Return input_ids and attention_mask for the example
            item = { 'input_ids': encoding['input_ids'][0], 'attention_mask': encoding['attention_mask'][0] }
            return item

    titles_dataset = TitlesDataset(titles, tokenizer)
    # Prepare the MLM model and data collator
    mlm_model = XLMRobertaForMaskedLM.from_pretrained('xlm-roberta-large')
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    # Set up training arguments for MLM
    mlm_args = TrainingArguments(
        output_dir="mlm_pretrain_output",
        learning_rate=5e-5,
        per_device_train_batch_size=32,   # batch size per GPU
        num_train_epochs=args.mlm_epochs,
        fp16=True,  # use mixed precision if available
        dataloader_num_workers=2,
        logging_steps=1000,
        logging_dir="logs_mlm",
        save_steps=5000,
        save_total_limit=1
    )
    mlm_trainer = Trainer(
        model=mlm_model,
        args=mlm_args,
        train_dataset=titles_dataset,
        data_collator=data_collator
    )
    mlm_trainer.train()
    # After pretraining, save the MLM model (optional, to reuse later) and extract its transformer weights
    mlm_trainer.save_model("domain_adapted_xlmr")
    # Load the adapted weights into a fresh token classification model
    print("Loading domain-adapted weights into NER model...")
    base_state_dict = mlm_model.roberta.state_dict()  # get the base transformer weights
    # Free up the MLM model if needed to save memory
    del mlm_model
    # Define a function to initialize a new token classification model with adapted base weights
    def init_ner_model_from_base():
        config = {
            "num_labels": len(label_list),
            "id2label": id2label,
            "label2id": label2id
        }
        model = XLMRobertaForTokenClassification.from_pretrained('xlm-roberta-large', **config)
        # Overwrite the base transformer weights with the adapted weights
        model.roberta.load_state_dict(base_state_dict)
        return model
else:
    # No MLM pretraining; define a helper to init model from the original pretrained weights
    def init_ner_model_from_base():
        return XLMRobertaForTokenClassification.from_pretrained(
            'xlm-roberta-large', num_labels=len(label_list), id2label=id2label, label2id=label2id
        )

# 3. Fine-tune NER model (and ensemble if specified)
print("Fine-tuning NER model on labeled data...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ner_models = []
for model_idx in range(args.ensemble_models):
    # Initialize model (with domain-adapted base if applicable)
    model = init_ner_model_from_base().to(device)
    # We will use Hugging Face Trainer for convenience
    # Prepare training dataset as a torch Dataset for token classification
    class NERDataset(torch.utils.data.Dataset):
        def __init__(self, token_sequences, label_sequences):
            self.token_seqs = token_sequences
            self.label_seqs = label_sequences
        def __len__(self):
            return len(self.token_seqs)
        def __getitem__(self, idx):
            tokens = self.token_seqs[idx]
            labels = self.label_seqs[idx]
            # Tokenize the list of tokens for this record with alignment
            encoding = tokenizer(tokens, is_split_into_words=True, return_tensors="pt",
                                 truncation=True, max_length=128, padding="max_length",
                                 return_attention_mask=True, return_offsets_mapping=True)
            # We will create aligned labels for subwords
            word_ids = encoding.word_ids(batch_index=0)  # map from token indices to word indices
            aligned_labels = []
            prev_word_id = None
            for word_id in word_ids:
                if word_id is None:
                    # This is a special token ([CLS], [SEP], or padding), no label
                    aligned_labels.append(-100)
                elif word_id != prev_word_id:
                    # Start of a new word
                    aligned_labels.append(labels[word_id])
                else:
                    # Same word as previous token -> a subword continuation
                    # Use -100 to ignore these in loss, so model isn't penalized separately for subword
                    aligned_labels.append(-100)
                prev_word_id = word_id
            # Convert to tensors
            item = {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': torch.tensor(aligned_labels)
            }
            return item

    train_dataset = NERDataset(train_tokens_per_record, train_bio_labels_per_record)
    val_dataset = NERDataset(val_tokens, val_bio_labels) if val_tokens else None

    # Set up training arguments for token classification
    output_dir = f"ner_model_{model_idx}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps=50,
        eval_strategy="epoch" if val_dataset else "no",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss",
        fp16=True,
        gradient_checkpointing=True,  # enable gradient checkpointing to save memory
        seed=42 + model_idx  # different seed for each model to diversify
    )

    # Define a simple accuracy or loss-based evaluation metric if needed (we'll do custom F0.2 eval later)
    def compute_metrics(eval_pred):
        # Compute token-level accuracy as a basic sanity check metric
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        # Ignore -100 labels in computing accuracy
        mask = labels != -100
        correct = (predictions[mask] == labels[mask]).sum()
        total = mask.sum()
        accuracy = (correct / total) if total > 0 else 0
        return {"accuracy": float(accuracy)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics if val_dataset else None
    )
    trainer.train()
    if val_dataset:
        # If we used a validation set, use the best model (if load_best_model_at_end was True)
        model = trainer.model
    # Move model to CPU to save GPU memory (if we have another model to train)
    model.to('cpu')
    ner_models.append(model)
    print(f"Model {model_idx+1}/{args.ensemble_models} training complete.")

# 4. Determine optimal confidence threshold (if validation set is available)
optimal_threshold = args.threshold
if val_tokens and ner_models:
    print("Evaluating on validation set to find optimal threshold for F0.2...")
    # Perform prediction on validation data using the ensemble
    # We will aggregate model outputs and then compute precision, recall, F0.2
    all_true_aspects = []   # ground truth aspects (tuples of (tag, value))
    all_pred_aspects = []   # predicted aspects above threshold
    for tokens, true_label_ids in zip(val_tokens, val_bio_labels):
        # Ground truth: extract aspect phrases from true labels
        true_aspects = []
        current_tag = None
        current_value_tokens = []
        for token, label_id in zip(tokens, true_label_ids):
            label_name = id2label[label_id]
            if label_name == "O":
                # end any current aspect
                if current_tag:
                    true_aspects.append((current_tag, " ".join(current_value_tokens)))
                    current_tag = None
                    current_value_tokens = []
                continue
            tag_type = label_name[2:]  # remove B- or I-
            label_prefix = label_name[:1]
            if label_prefix == "B":
                # start new aspect
                if current_tag:  # if previous not closed, close it
                    true_aspects.append((current_tag, " ".join(current_value_tokens)))
                current_tag = tag_type
                current_value_tokens = [token]
            elif label_prefix == "I" and tag_type == current_tag:
                # continuation of current aspect
                current_value_tokens.append(token)
            else:
                # I-tag for a different tag or without a current B (shouldn't happen in ground truth), treat as new B
                if current_tag:
                    true_aspects.append((current_tag, " ".join(current_value_tokens)))
                current_tag = tag_type
                current_value_tokens = [token]
        # close any open aspect
        if current_tag:
            true_aspects.append((current_tag, " ".join(current_value_tokens)))
        all_true_aspects.append(true_aspects)

        # Ensemble prediction: get average logits over models for this example
        # Tokenize the title (we can reuse the earlier NERDataset logic or do it here manually)
        encoding = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", 
                             truncation=True, max_length=128, padding="max_length",
                             return_offsets_mapping=True)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        # Sum logits from each model
        logits_sum = None
        for m in ner_models:
            m.eval()
            with torch.no_grad():
                outputs = m(input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # shape (1, seq_len, num_labels)
            if logits_sum is None:
                logits_sum = logits
            else:
                logits_sum += logits
        avg_logits = logits_sum / len(ner_models)
        # Convert logits to probabilities
        probs = avg_logits.softmax(dim=-1)  # shape (1, seq_len, num_labels)
        probs = probs[0].cpu().numpy()      # remove batch dimension
        # Decode predicted labels with threshold
        pred_aspects = []
        current_tag = None
        current_value_tokens = []
        current_confidences = []
        word_ids = encoding.word_ids(batch_index=0)
        prev_word_id = None
        # Iterate over subword tokens and aggregate by original word
        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue  # skip special tokens
            # Only consider first subword piece of each word for label decision
            if word_id != prev_word_id:
                # Determine predicted label for this word (the model's highest probability class)
                pred_label_id = int(probs[idx].argmax())
                pred_label = id2label[pred_label_id]
                pred_conf = float(probs[idx][pred_label_id])
                if pred_label == "O":
                    # if currently building an aspect, end it
                    if current_tag:
                        # finalize current aspect entity
                        # compute entity confidence (min token confidence in entity)
                        entity_conf = min(current_confidences) if current_confidences else 0.0
                        if entity_conf >= optimal_threshold:  # we'll evaluate thresholds later
                            pred_aspects.append((current_tag, " ".join(current_value_tokens), entity_conf))
                        current_tag = None
                        current_value_tokens = []
                        current_confidences = []
                    # O just means skip
                else:
                    # It's a B- or I- label
                    label_prefix = pred_label[0]
                    tag_type = pred_label[2:]
                    if label_prefix == "B" or current_tag is None or tag_type != current_tag:
                        # start a new aspect
                        if current_tag:
                            # close previous
                            entity_conf = min(current_confidences) if current_confidences else 0.0
                            if entity_conf >= optimal_threshold:
                                pred_aspects.append((current_tag, " ".join(current_value_tokens), entity_conf))
                        current_tag = tag_type
                        current_value_tokens = [tokens[word_id]]
                        current_confidences = [pred_conf]
                    elif label_prefix == "I" and tag_type == current_tag:
                        # continue current aspect
                        current_value_tokens.append(tokens[word_id])
                        current_confidences.append(pred_conf)
                    else:
                        # Unexpected I (different tag), start new
                        if current_tag:
                            entity_conf = min(current_confidences) if current_confidences else 0.0
                            if entity_conf >= optimal_threshold:
                                pred_aspects.append((current_tag, " ".join(current_value_tokens), entity_conf))
                        current_tag = tag_type
                        current_value_tokens = [tokens[word_id]]
                        current_confidences = [pred_conf]
            prev_word_id = word_id
        # finalize last aspect
        if current_tag:
            entity_conf = min(current_confidences) if current_confidences else 0.0
            if entity_conf >= optimal_threshold:
                pred_aspects.append((current_tag, " ".join(current_value_tokens), entity_conf))
        # Store predicted aspects (without confidence for evaluation)
        all_pred_aspects.append([(tag, val) for tag, val, conf in pred_aspects])
    # Now evaluate for different threshold values to find the best F0.2
    best_F0_2 = -1.0
    best_thr = args.threshold
    for thr in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        tp = fp = fn = 0
        # Compare all_true_aspects vs all_pred_aspects at this threshold
        for true_aspects, pred_aspects_with_conf in zip(all_true_aspects, all_pred_aspects):
            # Apply threshold filter to pred_aspects (pred_aspects_with_conf were stored without conf for eval)
            # Actually, above we stored pred_aspects already filtered by optimal_threshold initially. 
            # We need to re-filter for each thr. To simplify, we could re-run predictions for each thr, but that's slow.
            # Instead, we can approximate: if optimal_threshold was mid-range, the set might shrink/grow with different thr.
            # For simplicity, we'll assume initial optimal_threshold is default (0.5). 
            # To properly evaluate, one should store all predicted entities with their confidences and then filter by thr.
            pass
# ... [previous code sections above remain unchanged] ...

# 4. Determine optimal confidence threshold (if validation set is available)
optimal_threshold = args.threshold
if val_tokens and ner_models:
    print("Evaluating on validation set to find optimal threshold for F0.2...")
    all_true_aspects = []   # list of true aspect sets per record
    all_pred_aspects_with_conf = []  # list of predicted aspects (with conf) per record
    for tokens, true_label_ids in zip(val_tokens, val_bio_labels):
        # Extract ground-truth aspect entities from true labels
        true_aspects = []
        current_tag = None
        current_val_tokens = []
        for token, label_id in zip(tokens, true_label_ids):
            label_name = id2label[label_id]
            if label_name == "O":
                if current_tag:
                    true_aspects.append((current_tag, " ".join(current_val_tokens)))
                    current_tag = None
                    current_val_tokens = []
                # skip O tokens
            else:
                tag_type = label_name[2:]
                label_prefix = label_name[0]
                if label_prefix == "B":
                    if current_tag:
                        true_aspects.append((current_tag, " ".join(current_val_tokens)))
                    current_tag = tag_type
                    current_val_tokens = [token]
                elif label_prefix == "I" and tag_type == current_tag:
                    # continuation of current entity
                    current_val_tokens.append(token)
                else:
                    # I without matching B (or different tag) -> treat as B
                    if current_tag:
                        true_aspects.append((current_tag, " ".join(current_val_tokens)))
                    current_tag = tag_type
                    current_val_tokens = [token]
        if current_tag:
            true_aspects.append((current_tag, " ".join(current_val_tokens)))
        all_true_aspects.append(set(true_aspects))  # use set of tuples for matching

        # Predict with ensemble for this record
        encoding = tokenizer(tokens, is_split_into_words=True, return_tensors="pt",
                             truncation=True, max_length=128, padding="max_length",
                             return_offsets_mapping=True)
        input_ids = encoding['input_ids'].to(device)  # to GPU
        attention_mask = encoding['attention_mask'].to(device)
        # Sum probabilities from each model (to average later)
        ensemble_probs = None
        for m in ner_models:
            m.eval()
            with torch.no_grad():
                logits = m(input_ids, attention_mask=attention_mask).logits  # (1, seq_len, num_labels)
                probs = logits.softmax(dim=-1)  # convert to probabilities
            if ensemble_probs is None:
                ensemble_probs = probs
            else:
                ensemble_probs += probs
        ensemble_probs = (ensemble_probs / len(ner_models))[0].cpu().numpy()  # average probabilities for seq_len tokens

        # Decode predicted labels and record entity confidences (no thresholding yet)
        pred_entities = []
        current_tag = None
        current_val_tokens = []
        current_confidences = []
        prev_word_id = None
        word_ids = encoding.word_ids(batch_index=0)
        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id != prev_word_id:
                # New word starts
                pred_label_id = int(ensemble_probs[idx].argmax())
                pred_label = id2label[pred_label_id]
                pred_conf = float(ensemble_probs[idx][pred_label_id])
                if pred_label == "O":
                    # Close current entity if any
                    if current_tag:
                        # finalize current entity (store with min confidence)
                        entity_conf = min(current_confidences) if current_confidences else 0.0
                        pred_entities.append((current_tag, " ".join(current_val_tokens), entity_conf))
                        current_tag = None
                        current_val_tokens = []
                        current_confidences = []
                    # nothing to do for O token itself
                else:
                    tag_type = pred_label[2:]
                    label_prefix = pred_label[0]
                    if label_prefix == "B" or current_tag is None or tag_type != current_tag:
                        # start new entity
                        if current_tag:
                            # finalize previous entity
                            entity_conf = min(current_confidences) if current_confidences else 0.0
                            pred_entities.append((current_tag, " ".join(current_val_tokens), entity_conf))
                        current_tag = tag_type
                        current_val_tokens = [tokens[word_id]]
                        current_confidences = [pred_conf]
                    elif label_prefix == "I" and tag_type == current_tag:
                        # continue entity
                        current_val_tokens.append(tokens[word_id])
                        current_confidences.append(pred_conf)
                    else:
                        # Unexpected sequence, start new entity
                        if current_tag:
                            entity_conf = min(current_confidences) if current_confidences else 0.0
                            pred_entities.append((current_tag, " ".join(current_val_tokens), entity_conf))
                        current_tag = tag_type
                        current_val_tokens = [tokens[word_id]]
                        current_confidences = [pred_conf]
            prev_word_id = word_id
        # finalize last entity
        if current_tag:
            entity_conf = min(current_confidences) if current_confidences else 0.0
            pred_entities.append((current_tag, " ".join(current_val_tokens), entity_conf))
        all_pred_aspects_with_conf.append(pred_entities)

    # Evaluate F0.2 for various threshold values to find the best
    best_F = -1.0
    best_thr = optimal_threshold
    beta = 0.2
    beta2 = beta * beta
    for thr in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        TP = FP = FN = 0
        for true_aspects, pred_entities in zip(all_true_aspects, all_pred_aspects_with_conf):
            # Filter predicted entities by threshold
            pred_filtered = [(tag, val) for (tag, val, conf) in pred_entities if conf >= thr]
            pred_set = set(pred_filtered)
            # True positives: correctly predicted aspects
            for tag_val in pred_set:
                if tag_val in true_aspects:
                    TP += 1
            # False positives: predicted but not in truth
            for tag_val in pred_set:
                if tag_val not in true_aspects:
                    FP += 1
            # False negatives: in truth but not predicted
            for tag_val in true_aspects:
                if tag_val not in pred_set:
                    FN += 1
        prec = TP / (TP + FP) if (TP + FP) > 0 else 1.0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        F0_2 = (1 + beta2) * prec * rec / (beta2 * prec + rec + 1e-8)  # add tiny constant to avoid zero division
        if F0_2 > best_F:
            best_F = F0_2
            best_thr = thr
    optimal_threshold = best_thr
    print(f"Optimal threshold found: {optimal_threshold:.2f} (F0.2 = {best_F:.4f})")

# 5. Inference on all listing titles (test set) and output submission file
print("Generating predictions on test data and writing to submission file...")
output_path = args.output_file
with open("Listing_Titles.tsv", newline='', encoding='utf-8') as in_file, open(output_path, 'w', encoding='utf-8') as out_file:
    reader = csv.reader(in_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    header = next(reader, None)  # skip header line
    for row in reader:
        if not row: 
            continue
        record_id, category_id, title = row[0], row[1], row[2]
        # Tokenize title by whitespace as per guideline:contentReference[oaicite:23]{index=23}】
        tokens = title.split()
        if len(tokens) == 0:
            continue  # skip empty title
        # Ensemble prediction for this title
        encoding = tokenizer(tokens, is_split_into_words=True, return_tensors="pt",
                             truncation=True, max_length=128, padding="max_length",
                             return_offsets_mapping=True)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        ensemble_probs = None
        for m in ner_models:
            m.eval()
            with torch.no_grad():
                logits = m(input_ids, attention_mask=attention_mask).logits
                probs = logits.softmax(dim=-1)
            if ensemble_probs is None:
                ensemble_probs = probs
            else:
                ensemble_probs += probs
        ensemble_probs = (ensemble_probs / len(ner_models) if len(ner_models)>0 else ensemble_probs)[0].cpu().numpy()

        # Decode predicted sequence into aspect entities
        pred_aspects = []
        current_tag = None
        current_val_tokens = []
        current_confidences = []
        prev_word_id = None
        word_ids = encoding.word_ids(batch_index=0)
        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id != prev_word_id:
                pred_label_id = int(ensemble_probs[idx].argmax())
                pred_label = id2label[pred_label_id]
                pred_conf = float(ensemble_probs[idx][pred_label_id])
                if pred_label == "O":
                    if current_tag:
                        # finalize current entity
                        entity_conf = min(current_confidences) if current_confidences else 0.0
                        if entity_conf >= optimal_threshold:
                            aspect_value = " ".join(current_val_tokens)
                            out_file.write(f"{record_id}\t{category_id}\t{current_tag}\t{aspect_value}\n")
                        current_tag = None
                        current_val_tokens = []
                        current_confidences = []
                    # nothing to output for O
                else:
                    tag_type = pred_label[2:]
                    label_prefix = pred_label[0]
                    if label_prefix == "B" or current_tag is None or tag_type != current_tag:
                        # start new entity
                        if current_tag:
                            # finalize previous
                            entity_conf = min(current_confidences) if current_confidences else 0.0
                            if entity_conf >= optimal_threshold:
                                aspect_value = " ".join(current_val_tokens)
                                out_file.write(f"{record_id}\t{category_id}\t{current_tag}\t{aspect_value}\n")
                        current_tag = tag_type
                        current_val_tokens = [tokens[word_id]]
                        current_confidences = [pred_conf]
                    elif label_prefix == "I" and tag_type == current_tag:
                        # continue entity
                        current_val_tokens.append(tokens[word_id])
                        current_confidences.append(pred_conf)
                    else:
                        # I-tag for a different entity or without prior B -> treat as new B
                        if current_tag:
                            entity_conf = min(current_confidences) if current_confidences else 0.0
                            if entity_conf >= optimal_threshold:
                                aspect_value = " ".join(current_val_tokens)
                                out_file.write(f"{record_id}\t{category_id}\t{current_tag}\t{aspect_value}\n")
                        current_tag = tag_type
                        current_val_tokens = [tokens[word_id]]
                        current_confidences = [pred_conf]
            prev_word_id = word_id
        # finalize last entity in title
        if current_tag:
            entity_conf = min(current_confidences) if current_confidences else 0.0
            if entity_conf >= optimal_threshold:
                aspect_value = " ".join(current_val_tokens)
                out_file.write(f"{record_id}\t{category_id}\t{current_tag}\t{aspect_value}\n")
print(f"Submission file saved to {output_path}")
