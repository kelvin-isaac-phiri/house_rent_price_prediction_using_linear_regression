# Machine learning workflow used in this project

This page is the “why” behind the notebook. Read it once, then follow the notebook with the same steps on screen.

## 1. Define the problem

We want to **predict house price** from listing details such as area, bedrooms, bathrooms, air conditioning, and furnishing.

This is a **regression** problem: the answer is a number (`price`), not a category (like “cheap / expensive”).

## 2. Collect data

We use a table of past listings. Each **row** is one house. Each **column** is one piece of information. Your CSV columns are:

`price, area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus`

## 3. Load and inspect

We open the CSV with pandas and ask:

- How many houses?
- What are the column names?
- Which columns have missing values?
- What does `price` look like (min, max, typical value)?

Never train a model before looking at the table.

## 4. Clean and preprocess

Raw data is messy. We:

- strip extra spaces from column names
- remove duplicate rows
- fill missing numbers with the median
- fill missing text with the most common value
- remove extreme prices (outliers) so a few luxury homes do not dominate
- convert yes/no columns into 1 and 0
- one-hot encode `furnishingstatus`

Linear Regression only understands numbers, so the last two steps are required.

## 5. Exploratory data analysis (EDA)

We draw graphs **before** modelling:

- histogram of price
- scatter of area vs price
- average price by furnishing status
- average price with/without air conditioning
- correlation heatmap

EDA tells us whether a linear model is even reasonable. If area and price form a rising cloud of points, a straight-line model can work.

## 6. Feature selection

We choose inputs (`X`) and the target (`y` = price).

We keep:

- numbers: `area`, `bedrooms`, `bathrooms`, `stories`, `parking`
- yes/no flags as 0/1
- dummy columns for furnishing status

This dataset has no ID or date columns to drop.

## 7. Train / test split

We keep **20%** of houses hidden during training. That test set is the exam.

If we scored the model on the same houses it trained on, it could look excellent just by memorising.

## 8. Scale features

We standardise columns so they have mean 0 and standard deviation 1. Fit the scaler on the **training** set only, then transform the test set.

## 9. Train Linear Regression

Sklearn finds weights `w` such that:

```text
predicted_price = intercept + w1*area + w2*bedrooms + w3*bathrooms + ...
```

Training means: choose weights that make squared errors small on the training houses.

## 10. Evaluate

On the test set we compute:

| Metric | Meaning |
| --- | --- |
| MAE | Average absolute mistake (easy to explain in currency) |
| MSE | Average squared mistake (punishes big errors) |
| RMSE | Typical error size, same unit as price |
| R² | How much of price variation the model explains (1.0 is perfect) |

## 11. Predict

For a new house we:

1. apply the same yes/no encoding and dummy columns
2. put columns in the **same order** as training
3. scale with the **saved** scaler
4. call `model.predict(...)`

## 12. Save the model

We store three files in `models/`:

- the regression itself
- the scaler
- the feature column order

All three are needed later. The model alone is not enough.

## What Linear Regression cannot do well

- Strongly curved relationships
- Weird interactions unless you add extra columns yourself

Those limits are okay for a first university project. They are also why your R² will not be 1.0 — and that is expected.
