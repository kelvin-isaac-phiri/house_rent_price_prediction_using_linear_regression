"""
Create a small practice CSV that uses the SAME column names as your dataset:

price, area, bedrooms, bathrooms, stories, mainroad, guestroom, basement,
hotwaterheating, airconditioning, parking, prefarea, furnishingstatus

You can run the whole project immediately with this file. Later, replace
it with your real Kaggle download (see datasets/README.md).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import SAMPLE_CSV_PATH


def generate_sample_dataset(n_rows: int = 400, seed: int = 42) -> pd.DataFrame:
    """
    Build a table with the same column names as the housing CSV.

    The numbers are invented, but they follow realistic patterns:
    bigger area, more bathrooms, AC, and preferred area cost more.
    That way Linear Regression has a real relationship to learn.

    Args:
        n_rows: how many fake houses to create.
        seed: random seed so the file is the same every time you regenerate it.

    Returns:
        pd.DataFrame: sample housing table.
    """
    rng = np.random.default_rng(seed)

    area = rng.integers(1650, 12000, size=n_rows)
    bedrooms = rng.integers(1, 6, size=n_rows)
    bathrooms = np.clip(rng.integers(1, 5, size=n_rows), 1, 4)
    stories = rng.integers(1, 5, size=n_rows)
    parking = rng.integers(0, 4, size=n_rows)

    def yes_no(probability_yes: float) -> np.ndarray:
        return rng.choice(["yes", "no"], size=n_rows, p=[probability_yes, 1 - probability_yes])

    mainroad = yes_no(0.85)
    guestroom = yes_no(0.18)
    basement = yes_no(0.35)
    hotwaterheating = yes_no(0.05)
    airconditioning = yes_no(0.32)
    prefarea = yes_no(0.23)
    furnishingstatus = rng.choice(
        ["furnished", "semi-furnished", "unfurnished"],
        size=n_rows,
        p=[0.25, 0.42, 0.33],
    )

    furnish_bonus = np.array(
        [
            {"furnished": 800000, "semi-furnished": 350000, "unfurnished": 0}[item]
            for item in furnishingstatus
        ]
    )

    price = (
        500000
        + area * 350
        + bedrooms * 250000
        + bathrooms * 450000
        + stories * 200000
        + parking * 150000
        + (mainroad == "yes") * 400000
        + (guestroom == "yes") * 250000
        + (basement == "yes") * 200000
        + (hotwaterheating == "yes") * 150000
        + (airconditioning == "yes") * 500000
        + (prefarea == "yes") * 450000
        + furnish_bonus
        + rng.normal(0, 400000, size=n_rows)
    )
    price = np.clip(price, 800000, None).round(0).astype(int)

    return pd.DataFrame(
        {
            "price": price,
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "mainroad": mainroad,
            "guestroom": guestroom,
            "basement": basement,
            "hotwaterheating": hotwaterheating,
            "airconditioning": airconditioning,
            "parking": parking,
            "prefarea": prefarea,
            "furnishingstatus": furnishingstatus,
        }
    )


def save_sample_dataset(path: Path | None = None, n_rows: int = 400) -> Path:
    """
    Generate the sample table and write it as CSV.

    Args:
        path: destination file. Defaults to datasets/raw/sample_housing.csv.
        n_rows: number of rows to generate.

    Returns:
        Path: where the CSV was saved.
    """
    destination = Path(path) if path is not None else SAMPLE_CSV_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    df = generate_sample_dataset(n_rows=n_rows)
    df.to_csv(destination, index=False)
    print(f"Wrote {len(df)} sample rows to {destination}")
    return destination


if __name__ == "__main__":
    save_sample_dataset()
