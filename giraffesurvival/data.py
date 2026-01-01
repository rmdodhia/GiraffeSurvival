from pathlib import Path
import numpy as np
import pandas as pd


# Default data locations (relative to repo root)
WILD_PATH = Path("Data/wild.csv")
ZOO_PATH = Path("Data/zoo.csv")


def load_prepare_zoo(zoo_path: Path) -> pd.DataFrame:
    """Load zoo calves, standardize age in months, and clean sex labels."""
    zoo = pd.read_csv(zoo_path)

    zoo["Sex"] = zoo["Sex (M/F)"].astype(str).str.strip()
    zoo.loc[~zoo["Sex"].isin(["M", "F"]), "Sex"] = np.nan

    if "Age (days)" in zoo.columns:
        zoo["age_months"] = zoo["Age (days)"] / 30.4
    else:
        zoo["age_months"] = zoo["Age (months)"]

    zoo["height_cm"] = zoo["Height (cm)"]
    zoo = zoo.dropna(subset=["age_months", "height_cm"])
    return zoo


def load_prepare_wild(wild_path: Path) -> pd.DataFrame:
    """Load wild giraffe data, compute total height (TH), clean sex, parse dates."""
    df = pd.read_csv(wild_path)

    df["Sex"] = df["Sex"].astype(str).str.strip()
    df.loc[df["Sex"].isin(["nan", "", "NA"]), "Sex"] = np.nan

    date_col = "SurveyDate(M/D/Y)"
    df["SurveyDate"] = pd.to_datetime(df[date_col])

    df["TH"] = (
        df["avg TOO_TOHcm"]
        + df["avg TOH_NIcm"]
        + df["avg NI_FBHcm"]
    )

    if "VTB_Umb" in df.columns:
        df["VTB_Umb"] = pd.to_numeric(df["VTB_Umb"], errors="coerce").fillna(0)
    else:
        df["VTB_Umb"] = 0

    cols_to_avg = ["avg TOO_TOHcm", "avg TOH_NIcm", "avg NI_FBHcm", "TH", "VTB_Umb"]
    group_cols = ["AID", "Sex", "MinAge", "SurveyDate"]

    wild = (
        df.groupby(group_cols, dropna=False)[cols_to_avg]
          .mean()
          .reset_index()
    )
    return wild


def add_vtb_umb_flag(wild: pd.DataFrame) -> pd.DataFrame:
    wild = wild.copy()
    if "VTB_Umb" not in wild.columns:
        wild["VTB_Umb"] = 0
    # AID-level flag: if an individual (AID) ever has VTB_Umb>0, mark ALL of its
    # measurements as Umb>0.
    if "AID" in wild.columns:
        umb_any_by_aid = (
            wild.groupby("AID", dropna=False)["VTB_Umb"]
            .apply(lambda s: bool(pd.to_numeric(s, errors="coerce").fillna(0).gt(0).any()))
        )
        wild["VTB_Umb_Flag"] = wild["AID"].map(umb_any_by_aid).fillna(False)
        wild["VTB_Umb_Flag"] = np.where(wild["VTB_Umb_Flag"].astype(bool), "Umb>0", "Umb=0")
    else:
        wild["VTB_Umb_Flag"] = np.where(pd.to_numeric(wild["VTB_Umb"], errors="coerce").fillna(0) > 0, "Umb>0", "Umb=0")
    return wild


def add_age_class_midpoints(wild: pd.DataFrame) -> pd.DataFrame:
    age_map_months = {
        "C": 1.0,
        "C/SA": 3.5,
        "SA/C": 6.5,
        "SA": 10.5,
        "A/SA": 18.0,
        "ASA": 30.0,
        "A": 60.0,
    }
    wild["age_class_mid_mo"] = wild["MinAge"].map(age_map_months)
    return wild


def assign_initial_ages_from_classes(wild: pd.DataFrame) -> pd.DataFrame:
    wild = wild.sort_values(["AID", "SurveyDate"]).copy()
    first_date = wild.groupby("AID")["SurveyDate"].transform("min")
    days_since_first = (wild["SurveyDate"] - first_date).dt.days
    wild["age_months_initial"] = wild["age_class_mid_mo"]
    wild["age_months"] = wild["age_months_initial"] + days_since_first / 30.4
    return wild
