# Train the AegisAgent XGBoost detector

This pipeline trains a binary flow classifier using CIC-IDS2017 flow tables from the Canadian Institute for Cybersecurity.

## 1. Get the dataset

Open the [official CIC-IDS2017 dataset page](https://www.unb.ca/cic/datasets/ids-2017.html), download MachineLearningCSV.zip, and extract it locally. Keep dataset files and generated model artifacts outside Git.

## 2. Install dependencies

From the repository root:

```bash
python -m pip install -r backend/requirements.txt
```

## 3. Train and evaluate

```bash
python backend/train_model.py --data-dir "C:/path/to/extracted/MachineLearningCSV"
```

By default, the script uses a stratified sample of up to 500,000 rows, an 80/20 stratified train/test split, and XGBoost. It writes:

- `backend/models/aegis_xgb.joblib` — model plus ordered feature list
- `backend/models/metrics.json` — accuracy, attack precision/recall/F1, confusion matrix, and per-class metrics

Use `--max-rows 0` to use all loaded rows, or `--estimators 600` to change the number of boosting rounds. Full-data training may require substantial RAM and time.

## Scope and limitations

- The model predicts BENIGN versus ATTACK from CICFlowMeter-style flow features. It does not directly accept the packet dictionaries consumed by `backend/detector.py`; compatible flow feature extraction is required before live packet-level integration.
- A random stratified split is a baseline and may overestimate performance if near-duplicate flows occur across splits. Treat metrics as a first benchmark, not proof of production readiness.
- CIC-IDS2017 was collected in a lab in 2017. Validate on separate, newer traffic before relying on it operationally.
- Cite the dataset paper when publishing results: Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization,” ICISSP 2018.
