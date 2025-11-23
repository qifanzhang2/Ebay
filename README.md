EBAY-NER: High-Precision Entity Extraction for German E-Commerce

Objective
Build a named-entity recognition (NER) system over millions of German e-commerce titles—extracting brands, product types, and attributes—optimized for the competition’s F0.2 metric, where false positives are far more costly than missed entities.

My Contribution
Designed and implemented the entire modeling pipeline as a solo undergraduate competitor—from raw title preprocessing through sequence tagging, second-stage verification, and leaderboard submissions—using modern NLP and metric-driven evaluation.

Key Outcomes
4th place out of ~100 teams in the eBay University Machine Learning Challenge.


Final system achieved F0.2 ≈ 0.93, up from a baseline around 0.88.


Raised precision to ~0.94 with only minimal recall loss via a second-stage verifier.


Developed a reusable framework for sequence labeling under asymmetric error costs.



Technical Snapshot
Stack
Python 3.x


PyTorch, Hugging Face Transformers, sklearn


pandas, NumPy


Weights & Biases (experiment tracking), Git


Machine Learning
Sequence Tagger: xlm-roberta-large with a CRF layer for token-level labeling.


Ensembling: 5-fold cross-validated models with category-aware calibration.


Verifier: HistGradientBoostingClassifier as a second-stage “precision gate” on predicted entities.


Data Engineering
Title normalization (lowercasing, Unicode cleanup, punctuation handling).


Subword/word alignment between XLM-R tokens and BIO labels.


Gazetteer features for brands and product types; character-level features for misspellings.


Config-driven training scripts for rapid leaderboard iterations.


Evaluation
Primary metric: F0.2 to heavily penalize false positives.


Held-out validation by product category to avoid leakage and overfitting.


Detailed error analysis dashboards for per-entity and per-category performance.



Layout
ebay-ner/
├── Ebay_save.py              # end-to-end NER pipeline: tagger, calibration, verifier
├── Tagged_Titles_Train.tsv   # labeled German titles for training/validation
├── Listing_Titles.tsv        # unlabeled titles used for final predictions
├── gazetteer_hard.json       # curated brand & product lexicons
└── README.md                 # notes on experiments, parameters, and usage

Dataset preview
item_id | title_de                                              | tokens                             | labels
100123  | Bremsscheiben Satz vorne für VW Golf 7 GTI            | [Bremsscheiben, Satz, vorne, ...]  | B-Produktart I-Produktart B-Position ...
100987  | Original BMW Wasserpumpe 11518635089                   | [Original, BMW, Wasserpumpe, ...]  | B-Qualität B-Marke B-Produktart ...
101432  | Zahnriemensatz Conti CT1139K1 1.6 TDI                  | [Zahnriemensatz, Conti, ...]       | B-Produktart B-Marke ...

Skills Demonstrated
Sequence labeling at scale – multilingual transformer (xlm-roberta-large) with a CRF head for token-level NER.


Feature engineering for noisy titles – character-CNN word features plus per-aspect gazetteer hits to stabilize predictions on misspellings and long-tail brands.


Ensembling & calibration – 5-fold ensemble with category-aware calibration tuned directly on F0.2.


Asymmetric loss & decision theory – treating the model as a noisy measurement device and explicitly shaping errors to avoid costly false positives.


Second-stage verification – HistGradientBoosting “precision gate” using entity-level features (confidence stats, length, gazetteer flags) to filter borderline spans.


Systematic error analysis – iterative breakdown by category, aspect, and surface form to understand where the model still hallucinated entities.
Future Roadmap
Domain adaptation to other marketplaces and languages (e.g., French or Italian titles) via multilingual fine-tuning and light-weight adapters.


Active learning loop that prioritizes uncertain or rare entities for manual labeling, improving coverage on the long-tail product space.


Confidence-aware deployment with abstention: allow the system to defer to humans on low-confidence spans, further reducing false positives in production settings.
