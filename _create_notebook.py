"""One-off helper to write notebooks/01_house_rent_prediction.ipynb."""

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parent / "notebooks" / "01_house_rent_prediction.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip("\n").split("\n")],
    }


def code(text: str) -> dict:
    lines = text.strip("\n").split("\n")
    source = [line + "\n" for line in lines]
    if source:
        source[-1] = source[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source,
    }


cells = [
    md("""# House Rent Price Prediction using Linear Regression

Welcome. This notebook is the **main learning path** for this project.

You will build a complete machine learning pipeline:

1. Load house listing data
2. Clean and preprocess it
3. Explore it with graphs (EDA)
4. Select features
5. Split into train and test sets
6. Train a **Linear Regression** model
7. Evaluate with MAE, MSE, RMSE, and R²
8. Predict price for a new house

Your CSV uses these columns:

`price, area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus`

`price` is the target (the number we predict). Some columns are already numbers. Others are text (`yes`/`no`, or furnishing status) and must be turned into numbers first.

You do **not** need to know machine learning already. Read the markdown cells, then run the code cells from top to bottom with **Shift+Enter**.

If a word is new, open `docs/learning_guide.md` in the project folder.
"""),
    md("""## 0. How this notebook is organised

- **Markdown cells** (like this one) explain *why*.
- **Code cells** do the work. Comments inside the code explain *what each line does*.

Most heavy lifting lives in the `src/` folder so the project stays tidy. The notebook **calls** those functions so you can see the full story in one place.

**Important:** always run cells in order. A later cell needs variables created earlier (`raw_df`, `model`, ...).
"""),
    md("""## 1. Setup: folders, libraries, and the `src` package

Python needs to know where the project root is. This notebook lives in `notebooks/`, while our helper code lives in `src/`. The first cell adds the project root to `sys.path` so `from src...` works.
"""),
    code("""# Tell Jupyter to show matplotlib charts under each cell
%matplotlib inline

import sys
from pathlib import Path

# Path.cwd() is "where Jupyter was launched from".
# We look for the folder that contains src/ so imports work either way.
cwd = Path.cwd().resolve()
if (cwd / "src").exists():
    PROJECT_ROOT = cwd
elif (cwd.parent / "src").exists():
    PROJECT_ROOT = cwd.parent
else:
    raise FileNotFoundError(
        "Could not find the src/ folder. "
        "Open the notebook from this project, or start Jupyter in the project root."
    )

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("Project root:", PROJECT_ROOT)
"""),
    code("""# The four libraries required by this project
import pandas as pd          # tables (like Excel in Python)
import numpy as np           # numeric arrays and math
import matplotlib.pyplot as plt  # graphs

# Our own modules (beginner-friendly wrappers around sklearn + pandas)
from src.data_loader import load_dataset, dataset_overview
from src.preprocessing import preprocess_data, encode_yes_no_columns
from src.features import correlation_with_price, print_feature_advice
from src.visualization import (
    plot_price_distribution,
    plot_price_vs_area,
    plot_price_by_furnishing,
    plot_price_by_airconditioning,
    plot_correlation_heatmap,
    plot_actual_vs_predicted,
    plot_residuals,
)
from src.model import (
    split_features_and_target,
    make_train_test_split,
    scale_features,
    train_linear_regression,
    save_artefacts,
)
from src.evaluation import evaluate_model
from src.predict import predict_from_raw_row
from src.config import TARGET_COLUMN

print("Imports successful.")
"""),
    md("""## 2. What is Linear Regression? (tiny theory)

We want a formula that maps house features to price:

\\[
\\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \\cdots + w_n x_n
\\]

- \\(\\hat{y}\\) is the **predicted price**
- \\(x_1, x_2, \\ldots\\) are features (area, bedrooms, ...)
- \\(w_1, w_2, \\ldots\\) are **weights** the algorithm learns
- \\(w_0\\) is the **intercept** (baseline price)

Training means: choose weights that make predictions close to the real prices in the training set.

This model draws a **straight line** (in many dimensions). It cannot capture strong curves unless we add extra columns ourselves. That limitation is normal for a first project.
"""),
    md("""## 3. Load the data

`load_dataset()` looks for your file in `datasets/raw/` (for example `Housing.csv`).

If that file is not there yet, it uses the bundled sample CSV so you can still practise. Add your real CSV later; you do not need to change this notebook as long as the column names stay the same.
"""),
    code("""# Load the CSV into a pandas DataFrame (a table)
raw_df = load_dataset()

# Print shape, dtypes, missing values, and the first rows
dataset_overview(raw_df)
"""),
    md("""### Peek at the table yourself

`head()` shows the first rows. `describe()` summarises numeric columns (mean, min, max). These two commands are the habit every data project starts with.
"""),
    code("""# First 5 listings
raw_df.head()
"""),
    code("""# Numeric summary: count, mean, std, min, quartiles, max
raw_df.describe()
"""),
    md("""## 4. Exploratory data analysis (EDA) and graphs

We explore **before** turning yes/no text into 1/0, because charts of `furnishingstatus` are easier to read on the original table.

Questions we ask:

- Is price skewed (lots of cheaper homes, a few very expensive ones)?
- Does bigger area go with higher price?
- Do furnished homes cost more?
- Does air conditioning go with higher price?
"""),
    code("""# Histogram: how prices are spread
fig = plot_price_distribution(raw_df)
plt.show()
"""),
    code("""# Scatter plot: area vs price. A rising cloud means area is a useful feature.
fig = plot_price_vs_area(raw_df)
plt.show()
"""),
    code("""# Bar chart: average price by furnishing status
fig = plot_price_by_furnishing(raw_df)
plt.show()
"""),
    code("""# Bar chart: average price with / without air conditioning
fig = plot_price_by_airconditioning(raw_df)
plt.show()
"""),
    md("""### A quick extra chart: bedrooms vs price

Box plots show the **middle** of the data and the spread. If 4-bedroom boxes sit higher than 1-bedroom boxes, bedrooms are related to price.
"""),
    code("""fig, ax = plt.subplots(figsize=(8, 5))
raw_df.boxplot(column="price", by="bedrooms", ax=ax, grid=False)
ax.set_title("Price by number of bedrooms")
ax.set_xlabel("Bedrooms")
ax.set_ylabel("Price")
plt.suptitle("")  # removes pandas' automatic extra title
fig.tight_layout()
plt.show()
"""),
    md("""## 5. Data cleaning and preprocessing

Raw data is not model-ready. Our `preprocess_data()` function does these steps in order:

1. Drop unused columns (none in this dataset — every column is useful)
2. Remove duplicate rows
3. Fill missing numeric values with the **median**, missing categories with the **mode**
4. Remove extreme prices with the IQR rule so outliers do not tilt the line
5. Convert yes/no columns (`mainroad`, `guestroom`, ...) into 1 and 0
6. One-hot encode `furnishingstatus` into dummy columns
7. Keep only columns the model should see

Linear Regression cannot multiply a weight by the word `"yes"`. Encoding solves that.
"""),
    code("""# Demo: how yes/no columns become 1 and 0
yes_no_demo = encode_yes_no_columns(raw_df[["mainroad", "airconditioning", "prefarea"]].head())
yes_no_demo
"""),
    code("""# Run the full cleaning pipeline and save datasets/processed/cleaned_housing.csv
clean_df = preprocess_data(raw_df, save=True)
clean_df.head()
"""),
    md("""## 6. Feature selection

We keep all columns in this dataset. Now we **rank** them by correlation with price.

- Correlation near **+1**: feature goes up, price goes up
- Correlation near **-1**: feature goes up, price goes down
- Correlation near **0**: little straight-line relationship

Yes/no columns can still matter even when correlation looks modest, because they are on/off switches.
"""),
    code("""print_feature_advice(clean_df)

# The Series itself is useful if you want to plot later
corr_ranks = correlation_with_price(clean_df)
corr_ranks
"""),
    code("""# Heatmap of every numeric column vs every other (including price)
fig = plot_correlation_heatmap(clean_df)
plt.show()
"""),
    md("""## 7. Train / test split

We hide 20% of houses as a **test set**. The model is only allowed to learn from the other 80%.

If we trained and scored on the same rows, the score could look great just because the model memorised those houses. The test set is the exam.
"""),
    code("""# X = inputs (features). y = the thing we want to predict (price)
X, y = split_features_and_target(clean_df)

print("Feature columns:")
print(list(X.columns))
print("\\nNumber of houses:", len(X))
print("Target column:", TARGET_COLUMN)
"""),
    code("""X_train, X_test, y_train, y_test = make_train_test_split(X, y)

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
"""),
    md("""## 8. Scale the features

`area` might be 8000 while `bedrooms` is 3. StandardScaler transforms each column to mean 0 and standard deviation 1.

**Critical rule:** `fit` only on the training set, then `transform` the test set. Fitting on the test set would leak exam information into the preparation step.
"""),
    code("""X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

# After scaling, training columns should be near mean 0
X_train_scaled.describe().loc[["mean", "std"]]
"""),
    md("""## 9. Train Linear Regression

`model.fit(X, y)` is the whole training step. Sklearn chooses the weights that minimise squared error on the training houses.
"""),
    code("""model = train_linear_regression(X_train_scaled, y_train)

# Save model + scaler + column order so we can predict later without retraining
save_artefacts(model, scaler, X_train.columns)
"""),
    md("""### Learned weights

Each coefficient is the change in predicted price when that (scaled) feature goes up by 1, holding the others fixed.

Because features are scaled, compare **signs** and **relative size**, not "currency per square foot".
"""),
    code("""weights = pd.Series(model.coef_, index=X_train.columns).sort_values()

print(f"Intercept (baseline after scaling): {model.intercept_:,.2f}\\n")
print("Coefficients (smallest to largest):")
print(weights.to_string())

fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(weights))))
weights.plot(kind="barh", ax=ax, color="slategray")
ax.set_title("Linear Regression coefficients")
ax.set_xlabel("Weight (on scaled features)")
ax.axvline(0, color="black", linewidth=0.8)
fig.tight_layout()
plt.show()
"""),
    md("""## 10. Evaluate on the test set

We predict prices for houses the model has **not** trained on, then compare to the true prices.

| Metric | What it means |
| --- | --- |
| **MAE** | Average absolute error (easy to say in currency) |
| **MSE** | Average squared error (big mistakes hurt more) |
| **RMSE** | Typical error size, same unit as price |
| **R²** | Share of price variation explained (1.0 = perfect) |
"""),
    code("""y_pred = model.predict(X_test_scaled)
metrics = evaluate_model(y_test, y_pred)
metrics
"""),
    md("""## 11. Graphs of model quality

- **Actual vs predicted:** points should hug the dashed diagonal.
- **Residuals:** leftover error. A healthy linear model shows a random cloud around 0, not a curve.
"""),
    code("""fig = plot_actual_vs_predicted(y_test, y_pred)
plt.show()
"""),
    code("""fig = plot_residuals(y_test, y_pred)
plt.show()
"""),
    md("""## 12. Predict price for a new house

Describe a house with the **original** column names (the same ones as in the CSV). The helper function cleans, encodes, scales, and predicts.

Change the numbers below and re-run the cell.
"""),
    code("""example_house = {
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
}

predicted = predict_from_raw_row(example_house)
print("Predicted house price:", round(predicted, 2))
"""),
    md("""## 13. What you just did (checklist)

You completed a standard supervised-learning workflow:

- [x] Loaded a tabular dataset
- [x] Cleaned and encoded it (yes/no → 1/0, furnishing → dummy columns)
- [x] Explored relationships with graphs
- [x] Ranked features by correlation
- [x] Split train/test
- [x] Trained Linear Regression
- [x] Scored MAE, MSE, RMSE, and R²
- [x] Predicted price for a new listing

Saved files:

- cleaned table → `datasets/processed/cleaned_housing.csv`
- model artefacts → `models/`
- charts → `outputs/figures/`

### Next steps when you feel ready

1. Put your real CSV in `datasets/raw/Housing.csv` (see `datasets/README.md`).
2. Re-run this notebook from the top. Compare R² with the sample data.
3. Change `TEST_SIZE` in `src/config.py` and see how scores move.
4. Later courses: try Ridge/Lasso, decision trees, or log-transforming price if the histogram is very skewed.

If something failed, `docs/learning_guide.md` has a short troubleshooting table.
"""),
]


notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    },
    "cells": cells,
}

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH}")
