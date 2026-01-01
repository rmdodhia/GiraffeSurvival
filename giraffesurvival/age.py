import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar

from .models import FitResult


def estimate_age_from_height(
    height: float,
    fit: FitResult | None,
    t_min: float = 2.0,
    t_max: float = 240.0,
) -> float:
    """Invert the fitted juvenile curve to estimate age from height."""
    if not fit or not fit.success:
        return np.nan

    func = fit.model.func
    params = fit.params

    def f(t: float) -> float:
        return func(t, *params) - height

    try:
        h_min = func(t_min, *params)
        h_max = func(t_max, *params)
        if not (min(h_min, h_max) <= height <= max(h_min, h_max)):
            return np.nan
        return brentq(f, t_min, t_max)
    except Exception:
        return np.nan


def refine_ages_with_zoo_models(
    wild: pd.DataFrame,
    zoo_fit_overall: FitResult | None,
    zoo_fits_by_sex: dict[str, FitResult] | None = None,
    juvenile_classes=("C", "C/SA", "SA/C", "SA"),
) -> pd.DataFrame:
    """Refine wild ages using available juvenile zoo models."""
    wild = wild.sort_values(["AID", "SurveyDate"]).copy()

    first_idx = wild.groupby("AID")["SurveyDate"].idxmin()
    wild["age_at_first_months"] = np.nan
    zoo_fits_by_sex = zoo_fits_by_sex or {}

    for idx in first_idx:
        row = wild.loc[idx]
        aid = row["AID"]
        age_class = row["MinAge"]
        th = row["TH"]
        sex = row.get("Sex")

        if age_class in juvenile_classes or pd.isna(age_class):
            age0 = row["age_class_mid_mo"]
        else:
            if pd.isna(th):
                age0 = row["age_class_mid_mo"]
            else:
                fit_for_row: FitResult | None = None
                if pd.notna(sex):
                    fit_for_row = zoo_fits_by_sex.get(str(sex))
                    if fit_for_row and not fit_for_row.success:
                        fit_for_row = None
                if fit_for_row is None and zoo_fit_overall and zoo_fit_overall.success:
                    fit_for_row = zoo_fit_overall

                if fit_for_row is None:
                    age0 = row["age_class_mid_mo"]
                else:
                    age_est = estimate_age_from_height(th, fit_for_row)
                    age0 = age_est if not np.isnan(age_est) else row["age_class_mid_mo"]

        mask = wild["AID"] == aid
        first_date = row["SurveyDate"]
        days_since_first = (wild.loc[mask, "SurveyDate"] - first_date).dt.days
        wild.loc[mask, "age_months"] = age0 + days_since_first / 30.4
        wild.loc[mask, "age_at_first_months"] = age0

    return wild


def refine_ages_with_individual_alignment(
    wild: pd.DataFrame,
    measurement: str,
    fit_overall: FitResult | None,
    fits_by_sex: dict[str, FitResult] | None = None,
    min_points: int = 3,
    max_age_months: float = 240.0,
) -> pd.DataFrame:
    """Refine ages by aligning each animal's measurements to population curves."""
    if fit_overall is None or not fit_overall.success:
        print("Skipping individual alignment: overall fit unavailable or failed.")
        return wild

    fits_by_sex = fits_by_sex or {}
    wild = wild.sort_values(["AID", "SurveyDate"]).copy()

    for aid, group in wild.groupby("AID"):
        subset = group.dropna(subset=[measurement])
        if len(subset) < min_points:
            continue

        sex = subset["Sex"].iloc[0]
        candidate_fit: FitResult | None = None
        if pd.notna(sex):
            candidate_fit = fits_by_sex.get(str(sex))
            if candidate_fit and not candidate_fit.success:
                candidate_fit = None
        if candidate_fit is None:
            candidate_fit = fit_overall

        if candidate_fit is None or not candidate_fit.success:
            continue

        first_date = subset["SurveyDate"].min()
        rel_months = (subset["SurveyDate"] - first_date).dt.days / 30.4
        values = subset[measurement].to_numpy(dtype=float)

        if np.isnan(values).all():
            continue

        rel_months = rel_months.to_numpy(dtype=float)
        mask = ~np.isnan(values)
        rel_months = rel_months[mask]
        values = values[mask]
        if len(values) < min_points:
            continue

        lower_bound, upper_bound = 0.1, max_age_months
        initial_age = float(group["age_at_first_months"].iloc[0])
        if not np.isfinite(initial_age):
            initial_age = float(group["age_months"].iloc[0])
        if not np.isfinite(initial_age):
            continue
        initial_age = float(np.clip(initial_age, lower_bound, upper_bound))

        penalty_weight = 0.25

        def residual_only(age0: float) -> float:
            ages = age0 + rel_months
            preds = candidate_fit.predict(ages)
            return float(np.sum((preds - values) ** 2))

        baseline_sse = residual_only(initial_age)
        if not np.isfinite(baseline_sse):
            continue

        def objective(age0: float) -> float:
            residual = residual_only(age0)
            penalty = penalty_weight * (age0 - initial_age) ** 2
            return residual + penalty

        try:
            result = minimize_scalar(
                objective,
                bounds=(lower_bound, upper_bound),
                method="bounded",
            )
        except Exception:
            continue

        if not result.success:
            continue

        candidate_age0 = float(result.x)
        candidate_sse = residual_only(candidate_age0)
        hits_bound = (
            candidate_age0 <= lower_bound + 1e-3
            or candidate_age0 >= upper_bound - 1e-3
        )
        worse_than_seed = (
            np.isfinite(candidate_sse)
            and candidate_sse > baseline_sse * 1.05
        )

        age0 = initial_age if (hits_bound or worse_than_seed or not np.isfinite(candidate_sse)) else candidate_age0

        full_rel_months = (group["SurveyDate"] - first_date).dt.days / 30.4
        wild.loc[group.index, "age_months"] = age0 + full_rel_months
        wild.loc[group.index, "age_at_first_months"] = age0

    return wild


def refine_ages_with_individual_alignment_multimeasure(
    wild: pd.DataFrame,
    measurements: list[str],
    fits_overall_by_measure: dict[str, FitResult | None],
    fits_by_sex_by_measure: dict[str, dict[str, FitResult]] | None = None,
    min_total_points: int = 6,
    min_points_per_measure: int = 2,
    max_age_months: float = 240.0,
    prior_sd_months: float = 24.0,
) -> pd.DataFrame:
    """Refine ages by aligning each animal's multi-trait time series to population curves.

    Implements a single per-individual shift (age_at_first_months) estimated by minimizing a
    weighted sum of squared residuals across measures, where residuals are normalized by an
    estimated measurement-specific RMSE derived from the population fit (sqrt(SSE/n)).

    Notes:
    - This does not change the growth model families; it only updates age_months/age_at_first_months.
    - Uses sex-specific population fits when available for a given measure and Sex.
    """
    if not measurements:
        return wild

    fits_by_sex_by_measure = fits_by_sex_by_measure or {}
    wild = wild.sort_values(["AID", "SurveyDate"]).copy()

    def _select_fit(measure: str, sex: object) -> FitResult | None:
        # Prefer sex-specific, else overall.
        if pd.notna(sex):
            fit = (fits_by_sex_by_measure.get(measure) or {}).get(str(sex))
            if fit is not None and fit.success:
                return fit
        fit = fits_overall_by_measure.get(measure)
        if fit is not None and fit.success:
            return fit
        return None

    def _rmse_from_fit(fit: FitResult) -> float:
        # Use population-fit RMSE as a scale for residual normalization.
        n = float(getattr(fit, "n_obs", 0) or 0)
        sse = float(getattr(fit, "sse", np.nan))
        if n <= 0 or not np.isfinite(sse) or sse < 0:
            return 1.0
        rmse = float(np.sqrt(max(sse / n, 1e-12)))
        return rmse if np.isfinite(rmse) and rmse > 0 else 1.0

    # Prior penalty weight in objective units (normalized SSE units).
    prior_sd = float(prior_sd_months)
    if not np.isfinite(prior_sd) or prior_sd <= 0:
        prior_sd = 24.0
    penalty_weight = 1.0 / (prior_sd ** 2)

    for aid, group in wild.groupby("AID"):
        sex = group["Sex"].iloc[0] if "Sex" in group.columns else np.nan
        first_date = group["SurveyDate"].min()
        if pd.isna(first_date):
            continue
        full_rel_months = (group["SurveyDate"] - first_date).dt.days / 30.4
        if full_rel_months.isna().all():
            continue

        # Build per-measure observed arrays.
        per_measure: list[tuple[np.ndarray, np.ndarray, FitResult, float]] = []
        total_points = 0
        for meas in measurements:
            if meas not in group.columns:
                continue
            subset = group.dropna(subset=[meas])
            if subset.empty:
                continue
            rel = ((subset["SurveyDate"] - first_date).dt.days / 30.4).to_numpy(dtype=float)
            vals = subset[meas].to_numpy(dtype=float)
            mask = np.isfinite(rel) & np.isfinite(vals)
            rel = rel[mask]
            vals = vals[mask]
            if len(vals) < min_points_per_measure:
                continue
            fit = _select_fit(meas, sex)
            if fit is None:
                continue
            rmse = _rmse_from_fit(fit)
            per_measure.append((rel, vals, fit, rmse))
            total_points += len(vals)

        if total_points < min_total_points or not per_measure:
            continue

        lower_bound, upper_bound = 0.1, float(max_age_months)
        initial_age = float(group.get("age_at_first_months", pd.Series([np.nan])).iloc[0])
        if not np.isfinite(initial_age):
            initial_age = float(group.get("age_months", pd.Series([np.nan])).iloc[0])
        if not np.isfinite(initial_age):
            continue
        initial_age = float(np.clip(initial_age, lower_bound, upper_bound))

        def residual_only(age0: float) -> float:
            sse = 0.0
            for rel, vals, fit, rmse in per_measure:
                ages = age0 + rel
                preds = fit.predict(ages)
                if preds is None:
                    continue
                resid = (preds - vals) / rmse
                sse += float(np.sum(resid ** 2))
            return float(sse)

        baseline_sse = residual_only(initial_age)
        if not np.isfinite(baseline_sse):
            continue

        def objective(age0: float) -> float:
            residual = residual_only(age0)
            penalty = penalty_weight * (age0 - initial_age) ** 2
            return float(residual + penalty)

        try:
            result = minimize_scalar(
                objective,
                bounds=(lower_bound, upper_bound),
                method="bounded",
            )
        except Exception:
            continue

        if not result.success:
            continue

        candidate_age0 = float(result.x)
        candidate_sse = residual_only(candidate_age0)
        hits_bound = (
            candidate_age0 <= lower_bound + 1e-3
            or candidate_age0 >= upper_bound - 1e-3
        )
        worse_than_seed = (
            np.isfinite(candidate_sse)
            and candidate_sse > baseline_sse * 1.05
        )

        age0 = initial_age if (hits_bound or worse_than_seed or not np.isfinite(candidate_sse)) else candidate_age0

        wild.loc[group.index, "age_months"] = age0 + full_rel_months
        wild.loc[group.index, "age_at_first_months"] = age0

    return wild
