# House Rent Price Prediction (Linear Regression)

A beginner-friendly machine learning project that predicts **house price** from listing details such as area, bedrooms, bathrooms, air conditioning, and furnishing.

This repository is meant to be:

- a university / portfolio project you can put on GitHub
- a learning path if you are new to Python and ML
- a complete workflow: load → clean → explore → train → evaluate → predict

Your CSV columns are:

```text
price, area, bedrooms, bathrooms, stories, mainroad, guestroom,
basement, hotwaterheating, airconditioning, parking, prefarea,
furnishingstatus
```

You can run it **today** with a bundled sample CSV that uses those same names. Drop your real file into `datasets/raw/` when you are ready.

---

## What you will learn

- How a regression problem is set up (predict a number, not a category)
- How to clean tabular data with **pandas** (yes/no text → 1/0)
- How to explore data with **matplotlib**
- How **Linear Regression** in **scikit-learn** is trained and scored
- What MAE, MSE, RMSE, and R² actually mean
- Why we split data into train and test sets

If you are new, start here: [docs/learning_guide.md](docs/learning_guide.md)

The full workflow in plain language: [docs/machine_learning_workflow.md](docs/machine_learning_workflow.md)

---

## Project structure

```text
.
├── datasets/
│   ├── raw/                  # Put your CSV here (Housing.csv)
│   │   └── sample_housing.csv
│   ├── processed/            # Cleaned CSV is saved here after preprocessing
│   └── README.md             # Dataset download instructions
├── notebooks/
│   └── 01_house_rent_prediction.ipynb   # Main learning notebook
├── src/                      # Reusable Python modules
│   ├── config.py             # Paths and column names in one place
│   ├── data_loader.py        # Load the CSV
│   ├── preprocessing.py      # Clean, encode, select columns
│   ├── features.py           # Feature correlation / selection
│   ├── visualization.py      # Graphs
│   ├── model.py              # Train / save Linear Regression
│   ├── evaluation.py         # MAE, MSE, RMSE, R²
│   ├── predict.py            # Predict price for new houses
│   ├── pipeline.py           # Run the whole flow from the terminal
│   └── generate_sample_data.py
├── models/                   # Saved model + scaler after training
├── outputs/figures/          # Saved charts
├── docs/                     # Extra explanations
├── requirements.txt
└── README.md
```

---

## Machine learning workflow (short version)

1. **Load** the housing CSV.
2. **Clean** missing values and duplicates.
3. **Encode** yes/no columns as 1/0 and one-hot encode `furnishingstatus`.
4. **Explore** with graphs (price distribution, area vs price, furnishing, AC, correlations).
5. **Select features** (this dataset keeps all of them).
6. **Split** into a training set (80%) and a test set (20%).
7. **Scale** numeric features (fit on train only).
8. **Train** `sklearn.linear_model.LinearRegression`.
9. **Evaluate** with MAE, MSE, RMSE, and R² on the test set.
10. **Predict** price for a new house and plot actual vs predicted.

Linear Regression fits a weighted sum:

```text
predicted_price = intercept + w1*area + w2*bedrooms + w3*bathrooms + ...
```

---

## Requirements

- Python **3.10+**
- The libraries in `requirements.txt` (pandas, numpy, matplotlib, scikit-learn, jupyter, joblib)

---

## Installation

Open a terminal in this project folder.

### 1. Create a virtual environment (recommended)

Linux / macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows (Command Prompt):

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install libraries

```bash
pip install -r requirements.txt
```

---

## Dataset

### Run immediately (sample data)

A practice file is included:

`datasets/raw/sample_housing.csv`

It uses the same columns as your CSV, so the code stays the same when you switch.

### Add your real CSV

1. Copy your file into `datasets/raw/`. A good name is `Housing.csv`.
2. Matching public dataset:  
   https://www.kaggle.com/datasets/yasserh/housing-prices-dataset
3. Re-run the notebook or `python -m src.pipeline`.

The loader uses your file when it is present (any CSV with a `price` column), otherwise it falls back to the sample file.

Full notes: [datasets/README.md](datasets/README.md)

---

## How to run

### Option 1 — Jupyter Notebook (best for learning)

```bash
jupyter notebook
```

Then open `notebooks/01_house_rent_prediction.ipynb` and run the cells from top to bottom (Shift+Enter).

In JupyterLab:

```bash
jupyter lab
```

### Option 2 — one command in the terminal

```bash
python -m src.pipeline
```

This loads data, trains the model, prints MAE / MSE / RMSE / R², and writes charts to `outputs/figures/`.

Always run commands from the **project root** (the folder that contains `src/` and `notebooks/`), so Python can find the `src` package.

---

## Example prediction (after training)

After the notebook or pipeline has saved files in `models/`:

```python
from src.predict import predict_from_raw_row

price = predict_from_raw_row({
    "area": 7420,
    "bedrooms": 4,
    "bathrooms": 2,
    "stories": 3,
    "mainroad": "yes",
    "guestroom": "no",
    "basement": "no",
    "hotwaterheating": "no",
    "airconditioning": "yes",
    "parking": 2,
    "prefarea": "yes",
    "furnishingstatus": "furnished",
})
print(price)
```

---

## How to read the scores

Suppose the test set shows `RMSE = 900000` and `R² = 0.68`:

- **RMSE 900000**: a typical prediction is about that many currency units away from the true price.
- **R² 0.68**: the model explains about 68% of the variation in price. The rest is location details, negotiation, missing features, and noise.

R² will almost never be 1.0. That does not mean the project failed.

---

## Libraries used

| Library | Role in this project |
| --- | --- |
| pandas | Load and clean tables |
| numpy | Numeric operations |
| matplotlib | Graphs |
| scikit-learn | Train/test split, scaling, Linear Regression, metrics |

---

## Notes for a GitHub / university portfolio

- Keep your downloaded Kaggle CSV **out of git** if the licence asks you not to republish it. The sample file is fine to keep.
- Trained `.joblib` files are generated, not source code; they are gitignored.
- If you change settings, edit `src/config.py` only (random seed, test size, file names).
