# Datasets folder

This project expects a CSV with **exactly these columns** (the ones in your file):

```text
price, area, bedrooms, bathrooms, stories, mainroad, guestroom,
basement, hotwaterheating, airconditioning, parking, prefarea,
furnishingstatus
```

- `price` is the target (the number the model predicts).
- `area`, `bedrooms`, `bathrooms`, `stories`, `parking` are already numbers.
- `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`,
  `prefarea` are yes/no text. The code turns them into 1 and 0.
- `furnishingstatus` is furnished / semi-furnished / unfurnished. The code
  one-hot encodes it.

## Option A — practise with the sample file

File:

`datasets/raw/sample_housing.csv`

It uses the **same column names** as your Kaggle file.

To regenerate it:

```bash
python -m src.generate_sample_data
```

## Option B — add your real CSV

A matching public dataset is:

**Housing Prices Dataset**  
https://www.kaggle.com/datasets/yasserh/housing-prices-dataset

### Steps

1. Download or export your CSV.
2. Copy it into this folder. A good name is:

```text
datasets/raw/Housing.csv
```

Any `.csv` in `datasets/raw/` that has a `price` column will also be picked up.

3. Re-run the notebook or:

```bash
python -m src.pipeline
```

The loader uses your file when it is present. If it is missing, it uses the sample CSV.

## Column meanings

| Column | Meaning | What the code does |
| --- | --- | --- |
| price | House price (target) | Predict this |
| area | Size in square feet | Keep as a number |
| bedrooms | Number of bedrooms | Keep as a number |
| bathrooms | Number of bathrooms | Keep as a number |
| stories | Number of storeys | Keep as a number |
| mainroad | Connected to main road (yes/no) | Convert to 1/0 |
| guestroom | Has a guest room (yes/no) | Convert to 1/0 |
| basement | Has a basement (yes/no) | Convert to 1/0 |
| hotwaterheating | Has hot-water heating (yes/no) | Convert to 1/0 |
| airconditioning | Has AC (yes/no) | Convert to 1/0 |
| parking | Parking spots | Keep as a number |
| prefarea | Preferred neighbourhood (yes/no) | Convert to 1/0 |
| furnishingstatus | furnished / semi-furnished / unfurnished | One-hot encode |

## Processed data

After cleaning, a file is written to:

`datasets/processed/cleaned_housing.csv`

You do not need to create this yourself.
