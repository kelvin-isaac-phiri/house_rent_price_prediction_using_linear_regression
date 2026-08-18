# Beginner learning guide

You do **not** need to memorise this. Use it as a dictionary while you read the notebook.

## The four libraries

### Pandas

Think of pandas as **Excel inside Python**.

- `DataFrame`: the whole table
- `Series`: one column
- `df.head()`: first 5 rows
- `df.shape`: `(rows, columns)`
- `df.isna().sum()`: missing-value counts

### NumPy

NumPy is the **math engine**. Pandas tables sit on top of NumPy arrays. We use it for square roots, random numbers, and converting lists to arrays.

### Matplotlib

Matplotlib **draws graphs**. In the notebook, a cell that calls `plt.show()` (or just ends with a figure) displays the chart under the cell.

### Scikit-learn (sklearn)

Sklearn is a toolkit of **ready-made machine learning models**. We use:

- `train_test_split` to hide some data for testing
- `StandardScaler` to standardise numbers
- `LinearRegression` to fit the line
- metric functions for MAE, MSE, RMSE, and R²

## What is a model?

A model is a formula that turns inputs into a prediction.

Linear Regression’s formula is a **weighted sum**:

```text
price ≈ intercept + (weight_area × area) + (weight_bedrooms × bedrooms) + ...
```

- A **positive** weight means “this going up tends to raise price”.
- A **negative** weight means “this going up tends to lower price”.

Because we scale features, do not read a weight as “rupees per square foot” unless you skip scaling.

## Train vs test (the most important idea)

| Set | Used for | Analogy |
| --- | --- | --- |
| Training set | Fitting the line | Homework you can study |
| Test set | Scoring the line | Unseen exam |

A good model does well on the **exam**, not only on homework.

## Overfitting, in one sentence

Overfitting is when the model memorises training houses and then fails on new ones. Linear Regression is a simple model, so it overfits less than huge neural networks, but a leaky test set can still fool you.

## How to run a Jupyter notebook

1. Open a terminal in the project folder.
2. Activate your virtual environment (see the main README).
3. Run: `jupyter notebook`
4. Open `notebooks/01_house_rent_prediction.ipynb`.
5. Click **Cell → Run All**, or run cells one by one with Shift+Enter.

Run cells **from top to bottom**. Later cells depend on earlier ones.

## If something breaks

| Problem | What to try |
| --- | --- |
| `ModuleNotFoundError: pandas` | The virtual environment is not active, or you skipped `pip install -r requirements.txt` |
| `FileNotFoundError` for the CSV | The sample file should exist in `datasets/raw/`. Run `python -m src.generate_sample_data` |
| `Missing models/...joblib` | Train first: run the notebook through the training cells, or `python -m src.pipeline` |
| Plots do not appear | In Jupyter, run the cell that creates the figure. Do not close the figure before it renders |
| Kaggle file ignored | Put it in `datasets/raw/` (Housing.csv is a good name). It must have a `price` column |

## Suggested study order

1. Read the main README (setup only).
2. Skim `docs/machine_learning_workflow.md`.
3. Work through the notebook slowly. After each section, explain it out loud in one sentence.
4. Change one thing (for example `TEST_SIZE` in `src/config.py`) and re-run. Watch how RMSE and R² move.
5. Only then download the Kaggle file and re-run. Compare sample-data scores with real-data scores.
