"""
Giraffe growth analysis using wild and zoo datasets,
with sex ignored until the final modeling step for wild data.

Files expected in the working directory:
- "wild.csv" from "measuring-giraffes RAHUL.xlsx"  (wild)
- "zoo.csv" from "Zoo Giraffe Heights Data RAHUL.xlsx"  (zoo juveniles)
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, brentq
import matplotlib.pyplot as plt
from pathlib import Path


# ----------------------------------------------------------------------
# 1. File paths
# ----------------------------------------------------------------------

WILD_PATH = Path("Data/wild.csv")
ZOO_PATH = Path("Data/zoo.csv")


# ----------------------------------------------------------------------
# 2. Gompertz growth function
# ----------------------------------------------------------------------

def gompertz(t, A, k, t0):
    """
    Gompertz growth function:
        H(t) = A * exp( -exp( -k * (t - t0) ) )

    t  : age (months)
    A  : asymptotic size
    k  : growth rate
    t0 : inflection-shift parameter (months)
    """
    return A * np.exp(-np.exp(-k * (t - t0)))


# ----------------------------------------------------------------------
# 3. Load and prepare zoo data
# ----------------------------------------------------------------------

def load_prepare_zoo(zoo_path: Path) -> pd.DataFrame:
    """
    Load zoo calves, standardize age in months, and clean sex labels.
    Expected columns:
        - 'Name'
        - 'Sex (M/F)'
        - 'Age (days)' or 'Age (months)'
        - 'Height (cm)'
    """
    zoo = pd.read_csv(zoo_path)

    # Clean sex
    zoo["Sex"] = zoo["Sex (M/F)"].astype(str).str.strip()
    zoo.loc[~zoo["Sex"].isin(["M", "F"]), "Sex"] = np.nan

    # Age in months
    if "Age (days)" in zoo.columns:
        zoo["age_months"] = zoo["Age (days)"] / 30.4
    else:
        zoo["age_months"] = zoo["Age (months)"]

    zoo["height_cm"] = zoo["Height (cm)"]

    # Drop incomplete
    zoo = zoo.dropna(subset=["age_months", "height_cm"])

    return zoo


def fit_gompertz_zoo_overall(zoo: pd.DataFrame):
    """
    Fit a single Gompertz juvenile curve using all zoo calves (ignoring sex).

    Returns:
        (A, k, t0)
    """
    t = zoo["age_months"].values
    y = zoo["height_cm"].values

    A0 = y.max() * 1.1
    k0 = 0.05
    t0_0 = 0.0

    popt, _ = curve_fit(
        gompertz, t, y,
        p0=[A0, k0, t0_0],
        maxfev=20000
    )
    return tuple(popt)


def fit_gompertz_zoo_by_sex(zoo: pd.DataFrame):
    """
    Optional: Fit Gompertz juvenile growth curves to zoo calves for each sex.

    Returns:
        dict: { sex: (A, k, t0) }
    """
    params_by_sex = {}
    for sex in ["M", "F"]:
        z = zoo[zoo["Sex"] == sex].copy()
        z = z.dropna(subset=["age_months", "height_cm"])
        if z.empty:
            continue

        t = z["age_months"].values
        y = z["height_cm"].values

        A0 = y.max() * 1.1
        k0 = 0.05
        t0_0 = 0.0

        try:
            popt, _ = curve_fit(
                gompertz, t, y,
                p0=[A0, k0, t0_0],
                maxfev=20000
            )
            params_by_sex[sex] = tuple(popt)
        except RuntimeError:
            print(f"Could not fit Gompertz for zoo sex={sex}")

    return params_by_sex


# ----------------------------------------------------------------------
# 4. Load and prepare wild data
# ----------------------------------------------------------------------

def load_prepare_wild(wild_path: Path) -> pd.DataFrame:
    """
    Load wild giraffe data, compute total height (TH),
    clean sex labels, parse dates, and collapse multiple photos
    into one record per AID x date.
    """
    df = pd.read_csv(wild_path)

    # Clean sex (kept for later, but not used in early steps)
    df["Sex"] = df["Sex"].astype(str).str.strip()
    df.loc[df["Sex"].isin(["nan", "", "NA"]), "Sex"] = np.nan

    # Parse survey date
    date_col = "SurveyDate(M/D/Y)"
    df["SurveyDate"] = pd.to_datetime(df[date_col])

    # Compute total height: TH = TOO + TOH + NI
    df["TH"] = (
        df["avg TOO_TOHcm"]
        + df["avg TOH_NIcm"]
        + df["avg NI_FBHcm"]
    )

    # Collapse to AID x date
    cols_to_avg = ["avg TOO_TOHcm", "avg TOH_NIcm", "avg NI_FBHcm", "TH"]
    group_cols = ["AID", "Sex", "MinAge", "SurveyDate"]

    wild = (
        df.groupby(group_cols)[cols_to_avg]
          .mean()
          .reset_index()
    )

    return wild


# ----------------------------------------------------------------------
# 5. Age-class mapping and initial ages (sex-agnostic)
# ----------------------------------------------------------------------

def add_age_class_midpoints(wild: pd.DataFrame) -> pd.DataFrame:
    """
    Map age classes in wild data to midpoint ages in months.

    Interpretation:
        C   = 0–2 months      -> 1
        C/SA = 3–4 months     -> 3.5
        SA/C = 5–8 months     -> 6.5
        SA  = 9–12 months     -> 10.5
        A/SA = 1–2 years      -> 18
        ASA = 2–3 years       -> 30
        A   > 3 years         -> 60 (working value)
    """
    age_map_months = {
        "C": 1.0,
        "C/SA": 3.5,
        "SA/C": 6.5,
        "SA": 10.5,
        "A/SA": 18.0,
        "ASA": 30.0,
        "A": 60.0
    }

    wild["age_class_mid_mo"] = wild["MinAge"].map(age_map_months)
    return wild


def assign_initial_ages_from_classes(wild: pd.DataFrame) -> pd.DataFrame:
    """
    Sex-agnostic initial age assignment.

    For each individual, assign an initial age at first sighting based on
    the age-class midpoint, then refine for subsequent measurements using
    days since first sighting.
    """
    wild = wild.sort_values(["AID", "SurveyDate"]).copy()

    first_date = wild.groupby("AID")["SurveyDate"].transform("min")
    days_since_first = (wild["SurveyDate"] - first_date).dt.days

    # Initial age based purely on age-class midpoint
    wild["age_months_initial"] = wild["age_class_mid_mo"]
    wild["age_months"] = wild["age_months_initial"] + days_since_first / 30.4

    return wild


# ----------------------------------------------------------------------
# 6. Refine ages with zoo juvenile model (sex-agnostic)
# ----------------------------------------------------------------------

def estimate_age_from_height(height, params, t_min=2.0, t_max=240.0):
    """
    Invert Gompertz to get age t given height and parameters (A, k, t0).
    Uses brentq over [t_min, t_max] months.
    """
    A, k, t0 = params

    def f(t):
        return gompertz(t, A, k, t0) - height

    try:
        h_min = gompertz(t_min, A, k, t0)
        h_max = gompertz(t_max, A, k, t0)
        h = height

        if not (min(h_min, h_max) <= h <= max(h_min, h_max)):
            return np.nan

        t_root = brentq(f, t_min, t_max)
        return t_root
    except Exception:
        return np.nan


def refine_ages_with_zoo_model_overall(
    wild: pd.DataFrame,
    zoo_params_overall,
    juvenile_classes=("C", "C/SA", "SA/C", "SA"),
) -> pd.DataFrame:
    """
    Use a single (sex-agnostic) zoo Gompertz curve to refine ages
    for wild animals whose first sighting is beyond juvenile classes.

    For older classes, estimate age at first sighting from TH.
    For juvenile classes, keep the class-based midpoint.

    Parameters
    ----------
    wild : DataFrame with columns:
        - AID, MinAge, SurveyDate, TH, age_class_mid_mo
    zoo_params_overall : (A, k, t0) for the pooled zoo data
    juvenile_classes : tuple of age-class strings considered juvenile

    Returns
    -------
    DataFrame with updated 'age_months' column.
    """
    wild = wild.sort_values(["AID", "SurveyDate"]).copy()

    # Identify first sighting per AID
    first_idx = wild.groupby("AID")["SurveyDate"].idxmin()

    wild["age_at_first_months"] = np.nan

    for idx in first_idx:
        row = wild.loc[idx]
        aid = row["AID"]
        age_class = row["MinAge"]
        th = row["TH"]

        if age_class in juvenile_classes or pd.isna(age_class):
            age0 = row["age_class_mid_mo"]
        else:
            # Older class: estimate from pooled zoo curve
            if pd.isna(th):
                age0 = row["age_class_mid_mo"]
            else:
                age_est = estimate_age_from_height(th, zoo_params_overall)
                if np.isnan(age_est):
                    age0 = row["age_class_mid_mo"]
                else:
                    age0 = age_est

        wild.loc[idx, "age_at_first_months"] = age0

        # Propagate to all records of this AID
        mask = wild["AID"] == aid
        first_date = row["SurveyDate"]
        days_since_first = (wild.loc[mask, "SurveyDate"] - first_date).dt.days
        wild.loc[mask, "age_months"] = age0 + days_since_first / 30.4

    return wild


# ----------------------------------------------------------------------
# 7. Fit Gompertz growth curves to wild data (overall, then by sex)
# ----------------------------------------------------------------------

def fit_gompertz_wild(
    wild: pd.DataFrame,
    value_col: str,
    group_col: str | None = None,
    min_points: int = 50,
):
    """
    Fit Gompertz growth curves to wild data.

    If group_col is None, fits one curve to all rows with data.
    If group_col is provided (e.g. 'Sex'), fits one curve per group.

    Returns
    -------
    dict: {group_value: (A, k, t0)}
          When group_col is None, the key is 'overall'.
    """
    results = {}

    if group_col is None:
        # Overall fit
        w = wild.dropna(subset=["age_months", value_col]).copy()
        if len(w) < min_points:
            print(f"Not enough data to fit wild overall {value_col}")
            return results

        t = w["age_months"].values
        y = w[value_col].values

        A0 = y.max() * 1.05
        k0 = 0.02
        t0_0 = 5.0

        try:
            popt, _ = curve_fit(
                gompertz, t, y,
                p0=[A0, k0, t0_0],
                maxfev=50000
            )
            results["overall"] = tuple(popt)
        except RuntimeError:
            print(f"Could not fit Gompertz for wild overall {value_col}")

        return results

    # Grouped fit
    for group_val in wild[group_col].dropna().unique():
        w = wild[wild[group_col] == group_val].copy()
        w = w.dropna(subset=["age_months", value_col])

        if len(w) < min_points:
            print(f"Not enough data to fit {value_col} for {group_col}={group_val}")
            continue

        t = w["age_months"].values
        y = w[value_col].values

        A0 = y.max() * 1.05
        k0 = 0.02
        t0_0 = 5.0

        try:
            popt, _ = curve_fit(
                gompertz, t, y,
                p0=[A0, k0, t0_0],
                maxfev=50000
            )
            results[group_val] = tuple(popt)
        except RuntimeError:
            print(f"Could not fit Gompertz for wild {value_col}, {group_col}={group_val}")

    return results


# ----------------------------------------------------------------------
# 8. Optional plotting helpers
# ----------------------------------------------------------------------


def _sanitize_for_filename(label) -> str:
    label = "" if label is None else str(label)
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")


def _strip_label_prefix(label: str) -> str:
    if label is None:
        return ""
    label_lower = label.lower()
    for prefix in ("params_", "df_", "data_"):
        if label_lower.startswith(prefix):
            return label[len(prefix):]
    return label


def plot_growth_curve_overall(
    wild: pd.DataFrame,
    params_overall,
    value_col: str,
    title: str,
    age_max_months: float = 240.0,
    df_label=None,
):
    """Scatter wild data, overlay a single Gompertz curve, and save to disk."""
    if params_overall is None or not params_overall:
        print("A PLOT CANCELED: No parameters provided.")
        return

    plt.figure(figsize=(7, 5))
    w = wild.dropna(subset=["age_months", value_col])
    if not w.empty:
        plt.scatter(w["age_months"] / 12.0, w[value_col], alpha=0.3, s=10, label="Data")

    A, k, t0 = params_overall.get('overall', (None, None, None))
    if A is None:
        print("B PLOT CANCELED: No parameters provided.")
        return
    
    t_grid = np.linspace(0.1, age_max_months, 300)
    curve = gompertz(t_grid, A, k, t0)
    plt.plot(t_grid / 12.0, curve, linewidth=2, label="Gompertz overall", color="black")

    plt.xlabel("Age (years)")
    plt.ylabel(value_col)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    attrs = getattr(wild, "attrs", {})
    if not isinstance(attrs, dict):
        attrs = {}
    inferred_label = _strip_label_prefix(
        df_label
        or attrs.get("label")
        or attrs.get("name")
        or getattr(wild, "name", None)
        or "data"
    )

    label_part = _sanitize_for_filename(inferred_label)
    if not label_part or label_part == "data":
        label_part = _sanitize_for_filename(value_col) or "value"

    filename = f"Graph/{label_part}_overall.png"
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_growth_curve_by_sex(
    wild: pd.DataFrame,
    params_by_sex: dict,
    value_col: str,
    title: str,
    age_max_months: float = 240.0,
    df_label=None,
):
    """Scatter wild data, overlay Gompertz curves by sex, and save to disk."""
    plt.figure(figsize=(7, 5))
    plotted_any = False

    for sex, color in zip(["M", "F"], ["tab:blue", "tab:orange"]):
        w = wild[(wild["Sex"] == sex)].copy()
        w = w.dropna(subset=["age_months", value_col])

        if not w.empty:
            plt.scatter(
                w["age_months"] / 12.0,
                w[value_col],
                alpha=0.3,
                s=10,
                label=f"Data {sex}",
                color=color,
            )
            plotted_any = True

        if sex in params_by_sex:
            A, k, t0 = params_by_sex[sex]
            t_grid = np.linspace(0.1, age_max_months, 300)
            curve = gompertz(t_grid, A, k, t0)
            plt.plot(
                t_grid / 12.0,
                curve,
                linewidth=2,
                label=f"Gompertz {sex}",
                color=color,
            )
            plotted_any = True

    plt.xlabel("Age (years)")
    plt.ylabel(value_col)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if plotted_any:
        attrs = getattr(wild, "attrs", {})
        if not isinstance(attrs, dict):
            attrs = {}
        inferred_label = (
            df_label
            or attrs.get("label")
            or attrs.get("name")
            or getattr(wild, "name", None)
            or "data"
        )

        inferred_label = _strip_label_prefix(inferred_label)
        label_part = _sanitize_for_filename(inferred_label)
        if not label_part or label_part == "data":
            label_part = _sanitize_for_filename(value_col) or "value"

        filename = f"Graph/{label_part}_by_sex.png"
        plt.savefig(filename, dpi=300)

    plt.close()


# ----------------------------------------------------------------------
# 9. Main driver
# ----------------------------------------------------------------------

def main():
    # --- Zoo juvenile modeling (sex-agnostic primary, sex-specific optional) ---
    print("Loading zoo data...")
    zoo = load_prepare_zoo(ZOO_PATH)

    print("Fitting pooled Gompertz curve to zoo calves (overall)...")
    zoo_params_overall = fit_gompertz_zoo_overall(zoo)
    print("Zoo Gompertz parameters overall (A, k, t0):")
    print(f"  A={zoo_params_overall[0]:.2f}, k={zoo_params_overall[1]:.4f}, t0={zoo_params_overall[2]:.2f}")

    # Optional: sex-specific juvenile curves for zoo
    zoo_params_by_sex = fit_gompertz_zoo_by_sex(zoo)
    if zoo_params_by_sex:
        print("\nZoo Gompertz parameters by sex (A, k, t0):")
        for sex, params in zoo_params_by_sex.items():
            print(f"  Sex {sex}: A={params[0]:.2f}, k={params[1]:.4f}, t0={params[2]:.2f}")

    # --- Wild data preparation ---
    print("\nLoading and preparing wild data...")
    wild = load_prepare_wild(WILD_PATH)
    wild = add_age_class_midpoints(wild)
    wild = assign_initial_ages_from_classes(wild)

    print("Refining ages for wild animals using pooled zoo juvenile model...")
    wild = refine_ages_with_zoo_model_overall(wild, zoo_params_overall)

    # --- Fit wild growth curves overall (ignoring sex) ---
    print("\nFitting wild Gompertz curves (overall, no sex)...")

    params_th_overall = fit_gompertz_wild(wild, "TH", group_col=None, min_points=50)
    print("\nWild Gompertz parameters for TH overall (A, k, t0):")
    for sex, params in params_th_overall.items():
        print(f"  Sex {sex}: A={params[0]:.2f}, k={params[1]:.4f}, t0={params[2]:.2f}")

    params_ossicone_overall = fit_gompertz_wild(wild, "avg TOO_TOHcm", group_col=None, min_points=50)
    print("\nWild Gompertz parameters for ossicone (avg TOO_TOHcm) overall:")
    for sex, params in params_ossicone_overall.items():
        print(f"  Sex {sex}: A={params[0]:.2f}, k={params[1]:.4f}, t0={params[2]:.2f}")

    params_neck_overall = fit_gompertz_wild(wild, "avg TOH_NIcm", group_col=None, min_points=50)
    print("\nWild Gompertz parameters for neck (avg TOH_NIcm) overall:")
    for sex, params in params_neck_overall.items():
        print(f"  Sex {sex}: A={params[0]:.2f}, k={params[1]:.4f}, t0={params[2]:.2f}")

    params_leg_overall = fit_gompertz_wild(wild, "avg NI_FBHcm", group_col=None, min_points=50)
    print("\nWild Gompertz parameters for foreleg (avg NI_FBHcm) overall:")
    for sex, params in params_leg_overall.items():
        print(f"  Sex {sex}: A={params[0]:.2f}, k={params[1]:.4f}, t0={params[2]:.2f}")

    # Optional plots (overall)
    plot_growth_curve_overall(wild, params_th_overall, "TH", "Wild total height (TH) - overall", df_label="wild_th")
    plot_growth_curve_overall(wild, params_ossicone_overall, "avg TOO_TOHcm", "Wild ossicone length - overall", df_label="wild_ossicone")
    plot_growth_curve_overall(wild, params_neck_overall, "avg TOH_NIcm", "Wild neck length - overall", df_label="wild_neck")
    plot_growth_curve_overall(wild, params_leg_overall, "avg NI_FBHcm", "Wild foreleg length - overall", df_label="wild_leg")

    # --- Final step: fit growth curves by sex (only for records with known sex) ---
    print("\nFitting wild Gompertz curves by sex (subset with known sex)...")

    params_th_by_sex = fit_gompertz_wild(wild, "TH", group_col="Sex", min_points=30)
    print("\nWild Gompertz parameters for TH by sex (A, k, t0):")
    for sex, params in params_th_by_sex.items():
        print(f"  Sex {sex}: A={params[0]:.2f}, k={params[1]:.4f}, t0={params[2]:.2f}")

    params_ossicone_by_sex = fit_gompertz_wild(wild, "avg TOO_TOHcm", group_col="Sex", min_points=30)
    print("\nWild Gompertz parameters for ossicone (avg TOO_TOHcm) by sex:")
    for sex, params in params_ossicone_by_sex.items():
        print(f"  Sex {sex}: A={params[0]:.2f}, k={params[1]:.4f}, t0={params[2]:.2f}")

    params_neck_by_sex = fit_gompertz_wild(wild, "avg TOH_NIcm", group_col="Sex", min_points=30)
    print("\nWild Gompertz parameters for neck (avg TOH_NIcm) by sex:")
    for sex, params in params_neck_by_sex.items():
        print(f"  Sex {sex}: A={params[0]:.2f}, k={params[1]:.4f}, t0={params[2]:.2f}")

    params_leg_by_sex = fit_gompertz_wild(wild, "avg NI_FBHcm", group_col="Sex", min_points=30)
    print("\nWild Gompertz parameters for foreleg (avg NI_FBHcm) by sex:")
    for sex, params in params_leg_by_sex.items():
        print(f"  Sex {sex}: A={params[0]:.2f}, k={params[1]:.4f}, t0={params[2]:.2f}")

    # Optional plots by sex
    plot_growth_curve_by_sex(wild, params_th_by_sex, "TH", "Wild total height (TH) - by sex", df_label="wild_th")
    plot_growth_curve_by_sex(wild, params_ossicone_by_sex, "avg TOO_TOHcm", "Wild ossicone length - by sex", df_label="wild_ossicone")
    plot_growth_curve_by_sex(wild, params_neck_by_sex, "avg TOH_NIcm", "Wild neck length - by sex", df_label="wild_neck")
    plot_growth_curve_by_sex(wild, params_leg_by_sex, "avg NI_FBHcm", "Wild foreleg length - by sex", df_label="wild_leg")

    # Save the processed wild dataset with age estimates
    out_path = Path("wild_with_age_estimates_sex_agnostic_then_by_sex.csv")
    wild.to_csv(out_path, index=False)
    print(f"\nSaved wild dataset with age estimates to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
