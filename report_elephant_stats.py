"""Utility script to summarize counts by sex and repeated measurements.

Outputs two sections for zoo and wild datasets:
- Counts of individuals by recorded sex.
- Distribution statistics for number of measurements per individual.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("Data")
ZOO_PATH = DATA_DIR / "zoo.csv"
WILD_PATH = DATA_DIR / "wild.csv"


def load_zoo() -> pd.DataFrame:
    df = pd.read_csv(ZOO_PATH)
    df["Sex"] = (
        df["Sex (M/F)"].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})
    )
    df.loc[~df["Sex"].isin(["M", "F"]), "Sex"] = np.nan
    df["Name"] = df["Name"].astype(str).str.strip()
    return df


def load_wild() -> pd.DataFrame:
    df = pd.read_csv(WILD_PATH)
    df["Sex"] = df["Sex"].astype(str).str.strip()
    df.loc[df["Sex"].isin(["", "nan", "NA"]), "Sex"] = np.nan
    df["AID"] = df["AID"].astype(str).str.strip()
    return df


def summarize_counts(df: pd.DataFrame, id_col: str) -> tuple[pd.Series, pd.Series]:
    representative = df[[id_col, "Sex"]].drop_duplicates(subset=[id_col])
    sex_counts = representative["Sex"].value_counts(dropna=False).rename("individuals")
    per_individual = df.groupby(id_col)[id_col].size().rename("n_measurements")
    return sex_counts, per_individual


def format_stats(values: pd.Series) -> str:
    mean = values.mean()
    median = values.median()
    std = values.std(ddof=1)
    return f"mean={mean:.2f}, median={median:.2f}, std={std:.2f}"


def main() -> None:
    print("Zoo dataset summary")
    zoo = load_zoo()
    zoo_counts, zoo_measurements = summarize_counts(zoo, "Name")
    print("Counts by sex (including unknown):")
    print(zoo_counts.to_string())
    print("Measurements per individual:")
    print(format_stats(zoo_measurements))

    print("\nWild dataset summary")
    wild = load_wild()
    wild_counts, wild_measurements = summarize_counts(wild, "AID")
    print("Counts by sex (including unknown):")
    print(wild_counts.to_string())
    print("Measurements per individual:")
    print(format_stats(wild_measurements))


if __name__ == "__main__":
    main()
