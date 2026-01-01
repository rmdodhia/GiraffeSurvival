from __future__ import annotations
from pathlib import Path
import sys
import re
import datetime as dt
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict

try:
    from scipy.stats import linregress
except Exception:
    linregress = None
try:
    from scipy.stats import chi2
except Exception:
    chi2 = None

# Candidate outputs roots (prefer Linux home)
HOME_OUTPUTS = Path("Outputs")
WORKSPACE_OUTPUTS = Path("Outputs")

MEASURE_DESCRIPTORS = {
    "TH": "Total height (TH)",
    "avg TOO_TOHcm": "Ossicone length (avg TOO_TOHcm)",
    "avg TOH_NIcm": "Neck length (avg TOH_NIcm)",
    "avg NI_FBHcm": "Foreleg length (avg NI_FBHcm)",
}

ASYMPTOTIC_MODELS = {"gompertz", "logistic", "von_bertalanffy", "richards"}


_PARAM_CI_RE = re.compile(
    r"(?P<name>[A-Za-z0-9_]+)="
    r"(?P<est>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?:\s*\[(?P<lo>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?),\s*(?P<hi>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\])?"
)


def _parse_params_with_ci(params: str) -> dict[str, dict[str, float]]:
    """Parse params field like `A=448.4 [442.3, 454.4]; k=...; t0=...`.

    Returns mapping param -> {est, lo?, hi?}.
    """
    out: dict[str, dict[str, float]] = {}
    if not isinstance(params, str) or not params.strip():
        return out
    for m in _PARAM_CI_RE.finditer(params):
        name = m.group("name")
        try:
            est = float(m.group("est"))
        except Exception:
            continue
        entry: dict[str, float] = {"est": est}
        lo_s = m.group("lo")
        hi_s = m.group("hi")
        if lo_s is not None and hi_s is not None:
            try:
                entry["lo"] = float(lo_s)
                entry["hi"] = float(hi_s)
            except Exception:
                pass
        out[name] = entry
    return out


def _fmt_est_ci(entry: dict[str, float] | None, digits: int = 3) -> str:
    if not entry or "est" not in entry:
        return "NA"
    est = entry.get("est")
    lo = entry.get("lo")
    hi = entry.get("hi")
    if lo is None or hi is None:
        return f"{est:.{digits}g}"
    return f"{est:.{digits}g} [{lo:.{digits}g}, {hi:.{digits}g}]"


def umbilicus_models_table(diag: pd.DataFrame) -> str:
    """Build a Markdown table for Gompertz-only models by umbilicus flag.

    Uses diagnostics contexts `wild_<measurement>_by_vtb_umb`.
    """
    if diag.empty:
        return "Diagnostics unavailable; cannot summarize umbilicus models."

    rows = diag[(diag.get("model") == "gompertz") & (diag.get("success") == True)].copy()
    if rows.empty:
        return "No successful Gompertz fits found for umbilicus models."

    measures = ["TH", "avg TOO_TOHcm", "avg TOH_NIcm", "avg NI_FBHcm"]
    groups = ["Umb=0", "Umb>0"]

    lines: list[str] = []
    lines.append("| Measure | Group | n | A (95% CI) | k (95% CI) | t0 (95% CI) | AIC |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")

    for meas in measures:
        # Support both historical (wild_th_by_vtb_umb) and current (wild_TH_by_vtb_umb) casing.
        ctx_candidates = {f"wild_{meas}_by_vtb_umb", f"wild_{meas.lower()}_by_vtb_umb"}
        meas_rows = rows[rows["context"].isin(ctx_candidates) & (rows["measurement"] == meas)].copy()
        if meas_rows.empty:
            # Still show a placeholder line for scanability
            desc = MEASURE_DESCRIPTORS.get(meas, meas)
            lines.append(f"| {desc} | (missing) |  |  |  |  |  |")
            continue

        desc = MEASURE_DESCRIPTORS.get(meas, meas)
        for grp in groups:
            r = meas_rows[meas_rows["group"] == grp]
            if r.empty:
                lines.append(f"| {desc} | {grp} |  |  |  |  |  |")
                continue
            # Use selected row if available; otherwise take the best AIC.
            r_sel = r[r.get("is_selected") == True]
            use = r_sel.iloc[0] if not r_sel.empty else r.sort_values(by="aic", ascending=True).iloc[0]
            params = _parse_params_with_ci(str(use.get("params", "")))
            n_obs = use.get("n_obs", "")
            aic = use.get("aic", "")
            A_txt = _fmt_est_ci(params.get("A"), digits=4)
            k_txt = _fmt_est_ci(params.get("k"), digits=4)
            t0_txt = _fmt_est_ci(params.get("t0"), digits=4)
            n_txt = f"{int(n_obs)}" if pd.notna(n_obs) else ""
            aic_txt = f"{float(aic):.2f}" if pd.notna(aic) else ""
            lines.append(f"| {desc} | {grp} | {n_txt} | {A_txt} | {k_txt} | {t0_txt} | {aic_txt} |")

    lines.append("")
    lines.append(
        "Notes: `A` is the adult/asymptotic size (same units as the measurement). `k` controls how fast the curve rises (per month). `t0` is the age (months) where growth rate is maximal (the curve’s inflection point). "
        "CIs are Wald-style 95% intervals from the nonlinear least-squares covariance (approximate; can be unstable when n is small or parameters are weakly identified)."
    )
    return "\n".join(lines)


def find_outputs_dir() -> Path:
    """Pick the most likely Outputs directory.
    Preference order:
      1) Any existing Outputs dir that contains model_fit_diagnostics.csv
      2) Any existing Outputs dir that contains the wild dataset CSV
      3) Workspace Outputs if it exists
      4) Home Outputs if it exists
      5) Create workspace Outputs
    This avoids writing to an unexpected location when both exist.
    """
    candidates: list[Path] = []
    if WORKSPACE_OUTPUTS.exists():
        candidates.append(WORKSPACE_OUTPUTS)
    if HOME_OUTPUTS.exists():
        candidates.append(HOME_OUTPUTS)

    # Prefer a directory that already has diagnostics
    for p in candidates:
        if (p / "model_fit_diagnostics.csv").exists():
            return p

    # Next, prefer one that already has the wild dataset
    for p in candidates:
        if (p / "wild_with_age_estimates_sex_agnostic_then_by_sex.csv").exists():
            return p

    # Otherwise, prefer workspace Outputs if present
    if WORKSPACE_OUTPUTS in candidates:
        return WORKSPACE_OUTPUTS
    if HOME_OUTPUTS in candidates:
        return HOME_OUTPUTS

    # Create workspace Outputs by default
    WORKSPACE_OUTPUTS.mkdir(parents=True, exist_ok=True)
    return WORKSPACE_OUTPUTS


def find_wild_dataset(outputs_dir: Path) -> Optional[Path]:
    base = outputs_dir / "wild_with_age_estimates_sex_agnostic_then_by_sex.csv"
    if base.exists():
        return base
    # fallback: timestamped variants
    candidates = sorted(outputs_dir.glob("wild_with_age_estimates_sex_agnostic_then_by_sex_*.csv"))
    return candidates[-1] if candidates else None


def load_diagnostics(outputs_dir: Path) -> pd.DataFrame:
    path = outputs_dir / "model_fit_diagnostics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Diagnostics CSV not found at {path}")
    return pd.read_csv(path)


def summarize_selected_models(diag: pd.DataFrame) -> str:
    lines = []
    selected = diag[diag["is_selected"] == True].copy()
    if selected.empty:
        return "No selected models found."
    selected.sort_values(["context", "measurement", "group"], inplace=True)
    for _, row in selected.iterrows():
        context = str(row.get("context", ""))
        meas = str(row.get("measurement", ""))
        group = str(row.get("group", ""))
        model = str(row.get("model", ""))
        aic = row.get("aic", np.nan)
        n = row.get("n_obs", np.nan)
        params = str(row.get("params", ""))
        descriptor = MEASURE_DESCRIPTORS.get(meas, meas)
        lines.append(f"- {descriptor} — {context} — group {group}: {model} (AIC={aic:.2f}, n={int(n) if pd.notna(n) else 'NA'}); {params}")
    return "\n".join(lines)


def _param_count_for_model(model_name: str) -> int:
    # Match counts in giraffesurvival/models.py
    return {
        "gompertz": 3,
        "logistic": 3,
        "von_bertalanffy": 3,
        "richards": 4,
        "poly3": 4,
    }.get(model_name, 3)


def collect_sex_deltas(diag: pd.DataFrame) -> pd.DataFrame:
    """Build a per-measure table with AICs and ΔAIC = AIC_overall - (AIC_M + AIC_F).
    Returns a DataFrame with one row per measurement found.
    """
    rows: list[dict] = []
    if diag.empty:
        return pd.DataFrame(rows)

    selected = diag[diag["is_selected"] == True].copy()
    for meas in sorted(selected["measurement"].dropna().unique()):
        # overall (pooled)
        over = selected[(selected["context"] == f"wild_{meas}_overall") & (selected["group"] == "overall")]
        by_sex = selected[(selected["context"] == f"wild_{meas}_by_sex")]
        mrow = by_sex[by_sex["group"] == "M"]
        frow = by_sex[by_sex["group"] == "F"]

        if over.empty or mrow.empty or frow.empty:
            # Incomplete pieces; skip but record partial if helpful
            continue

        aic_over = float(over.iloc[0]["aic"]) if pd.notna(over.iloc[0]["aic"]) else np.nan
        aic_m = float(mrow.iloc[0]["aic"]) if pd.notna(mrow.iloc[0]["aic"]) else np.nan
        aic_f = float(frow.iloc[0]["aic"]) if pd.notna(frow.iloc[0]["aic"]) else np.nan

        n_over = int(over.iloc[0]["n_obs"]) if pd.notna(over.iloc[0]["n_obs"]) else np.nan
        n_m = int(mrow.iloc[0]["n_obs"]) if pd.notna(mrow.iloc[0]["n_obs"]) else np.nan
        n_f = int(frow.iloc[0]["n_obs"]) if pd.notna(frow.iloc[0]["n_obs"]) else np.nan

        model_over = str(over.iloc[0]["model"]).strip()
        model_m = str(mrow.iloc[0]["model"]).strip()
        model_f = str(frow.iloc[0]["model"]).strip()

        delta = aic_over - (aic_m + aic_f)

        # Optional: simple LR statistic using AIC relation (approximate)
        k_over = _param_count_for_model(model_over)
        k_m = _param_count_for_model(model_m)
        k_f = _param_count_for_model(model_f)
        df = (k_m + k_f) - k_over

        rows.append(
            {
                "measurement": meas,
                "descriptor": MEASURE_DESCRIPTORS.get(meas, meas),
                "AIC_overall": aic_over,
                "AIC_M": aic_m,
                "AIC_F": aic_f,
                "n_overall": n_over,
                "n_M": n_m,
                "n_F": n_f,
                "model_overall": model_over,
                "model_M": model_m,
                "model_F": model_f,
                "delta_AIC": delta,
                "df_diff": df,
            }
        )

    return pd.DataFrame(rows)


def lr_test_same_family(diag: pd.DataFrame, measurement: str) -> Optional[dict]:
    """Compute LR test comparing pooled vs by-sex fits using the same model family.
    Uses candidate AICs for a priority family: gompertz > logistic > von_bertalanffy > richards > poly3.
    Returns dict with family, LR, df, p, AIC_over, AIC_M, AIC_F, k; or None if unavailable.
    """
    if diag.empty:
        return None

    fam_order = [
        "gompertz",
        "logistic",
        "von_bertalanffy",
        "richards",
        "poly3",
    ]
    cand = diag[(diag["measurement"] == measurement)].copy()
    if cand.empty:
        return None

    # Look for contexts
    over = cand[cand["context"] == f"wild_{measurement}_overall"]
    by_sex = cand[cand["context"] == f"wild_{measurement}_by_sex"]
    if over.empty or by_sex.empty:
        return None

    for fam in fam_order:
        over_f = over[over["model"] == fam]
        m_f = by_sex[(by_sex["group"] == "M") & (by_sex["model"] == fam)]
        f_f = by_sex[(by_sex["group"] == "F") & (by_sex["model"] == fam)]
        if over_f.empty or m_f.empty or f_f.empty:
            continue

        aic_over = float(over_f.iloc[0]["aic"]) if pd.notna(over_f.iloc[0]["aic"]) else np.nan
        aic_m = float(m_f.iloc[0]["aic"]) if pd.notna(m_f.iloc[0]["aic"]) else np.nan
        aic_f = float(f_f.iloc[0]["aic"]) if pd.notna(f_f.iloc[0]["aic"]) else np.nan
        if not (np.isfinite(aic_over) and np.isfinite(aic_m) and np.isfinite(aic_f)):
            continue

        k = _param_count_for_model(fam)
        delta = aic_over - (aic_m + aic_f)
        LR = delta + 2 * k  # LR = ΔAIC + 2k for same-family comparisons
        df = k  # params: pooled=k vs by-sex=2k
        if chi2 is not None and np.isfinite(LR) and LR >= 0:
            p = float(1.0 - chi2.cdf(LR, df))
        else:
            p = np.nan
        return {
            "family": fam,
            "LR": float(LR),
            "df": int(df),
            "p": p,
            "AIC_overall": aic_over,
            "AIC_M": aic_m,
            "AIC_F": aic_f,
            "k": int(k),
        }

    return None


def sex_difference_summary(diag: pd.DataFrame, measurement: str = "TH") -> Tuple[str, Optional[float]]:
    """Compare pooled vs sex-specific fits via ΔAIC.
    ΔAIC = AIC_pooled_overall - (AIC_M + AIC_F). If large positive, sex-specific fits are better.
    """
    # overall
    overall = diag[(diag["context"] == f"wild_{measurement}_overall") & (diag["is_selected"] == True) & (diag["group"] == "overall")]
    if overall.empty:
        return ("Sex difference: overall pooled fit missing; cannot compare.", None)
    aic_overall = float(overall.iloc[0]["aic"]) if pd.notna(overall.iloc[0]["aic"]) else np.nan

    # by sex
    by_sex = diag[(diag["context"] == f"wild_{measurement}_by_sex") & (diag["is_selected"] == True)]
    aic_M = by_sex[by_sex["group"] == "M"]["aic"].values
    aic_F = by_sex[by_sex["group"] == "F"]["aic"].values
    if len(aic_M) == 0 or len(aic_F) == 0:
        msg = "Sex difference: one or both sex-specific fits missing; treating sex as unknown for those entries."
        return (msg, None)

    aic_sum = float(aic_M[0] + aic_F[0])
    delta = aic_overall - aic_sum
    # Interpret ΔAIC thresholds
    if delta > 10:
        verdict = "Strong evidence that male and female curves differ (sex-specific fits preferred)."
    elif delta > 4:
        verdict = "Moderate evidence for sex-specific differences."
    elif delta > 2:
        verdict = "Weak evidence for sex-specific differences."
    else:
        verdict = "Little to no evidence; pooled curve adequate."

    unknown = diag[(diag["context"] == f"wild_{measurement}_sex_unknown") & (diag["is_selected"] == True)]
    unknown_note = "Unknown-sex subset fitted separately and excluded from sex difference test." if not unknown.empty else "Unknown-sex records not separately fitted."

    msg = (
        f"Sex difference (measurement={measurement}): ΔAIC = AIC_overall - (AIC_M + AIC_F) = {delta:.2f}. {verdict} "
        f"{unknown_note}"
    )
    return (msg, delta)


def determine_indeterminate_growth(wild: pd.DataFrame, measurement: str = "TH") -> str:
    """Assess indeterminate growth using adult slope and model type.
    Adult slope: OLS of value vs age for age_months >= 120; slope_year = slope_month*12.
    """
    df = wild.dropna(subset=[measurement, "age_months"]).copy()
    adults = df[df["age_months"] >= 120]
    if adults.empty or linregress is None:
        return "Indeterminate growth: insufficient adult data or scipy unavailable to assess; refer to model type (asymptotic implies determinate)."

    res = linregress(adults["age_months"].to_numpy(), adults[measurement].to_numpy())
    slope_year = res.slope * 12.0
    p = res.pvalue

    verdict: str
    detail = f"adult slope ~ {slope_year:.2f} cm/year (p={p:.3g}). "
    if p < 0.05 and slope_year > 1.0:
        verdict = "Evidence of continued growth in adults (possible indeterminate growth). "
    elif p < 0.05 and slope_year < -1.0:
        verdict = "Significant negative adult slope; likely measurement variability or senescence effects—no indeterminate growth. "
    else:
        verdict = "No significant adult growth; determinate growth likely. "
    return verdict + detail


def _selected_overall_model(diag: pd.DataFrame, measurement: str) -> Optional[str]:
    if diag.empty:
        return None
    sel = diag[
        (diag["is_selected"] == True)
        & (diag["context"] == f"wild_{measurement}_overall")
        & (diag["group"] == "overall")
    ]
    if sel.empty:
        return None
    return str(sel.iloc[0].get("model", "")).strip() or None


def _gompertz_vs_poly3_support_from_diagnostics(diag: pd.DataFrame, measurement: str) -> Optional[dict]:
    """Summarize model-selection support for Gompertz vs poly3.

    This is not a hypothesis test. Gompertz and poly3 are non-nested, so LR-test p-values
    are not appropriate.

    Returns a dict with AICs, ΔAIC (poly3 − gompertz), and 2-model Akaike weights.
    Returns None if one of the candidates is missing in diagnostics.
    """
    if diag.empty:
        return None

    cand = diag[
        (diag["context"] == f"wild_{measurement}_overall")
        & (diag["group"] == "overall")
        & (diag["success"] == True)
    ].copy()
    if cand.empty:
        return None

    cand = cand.copy()
    cand["aic_num"] = pd.to_numeric(cand["aic"], errors="coerce")
    cand = cand.dropna(subset=["aic_num"])
    if cand.empty:
        return None

    gomp = cand[cand["model"] == "gompertz"]
    poly = cand[cand["model"] == "poly3"]
    if gomp.empty or poly.empty:
        return None

    # In case there are multiple rows per model, keep the best AIC.
    aic_gomp = float(gomp["aic_num"].min())
    aic_poly3 = float(poly["aic_num"].min())

    aic_best = min(aic_gomp, aic_poly3)
    d_gomp = aic_gomp - aic_best
    d_poly3 = aic_poly3 - aic_best
    w_gomp = float(np.exp(-0.5 * d_gomp))
    w_poly3 = float(np.exp(-0.5 * d_poly3))
    denom = w_gomp + w_poly3
    if denom <= 0 or not np.isfinite(denom):
        return None
    w_gomp /= denom
    w_poly3 /= denom

    return {
        "best_is_gompertz": bool(aic_gomp <= aic_poly3),
        "aic_gompertz": aic_gomp,
        "aic_poly3": aic_poly3,
        "delta_aic_poly3_minus_gompertz": aic_poly3 - aic_gomp,
        "w_gompertz": w_gomp,
        "w_poly3": w_poly3,
    }


def determine_indeterminate_growth_all_measures(
    wild: Optional[pd.DataFrame],
    diag: pd.DataFrame,
    adult_age_months: float = 120.0,
) -> str:
    """Summarize indeterminate growth evidence for all measures.

    Two complementary signals:
      1) Adult slope test: OLS slope of value vs age for age_months >= adult_age_months.
      2) Model-type check: asymptotic model selection (Gompertz/Logistic/VB/Richards) supports determinate growth.

    Returns Markdown-formatted lines.
    """
    measures = [
        "TH",
        "avg TOO_TOHcm",
        "avg TOH_NIcm",
        "avg NI_FBHcm",
    ]

    lines: list[str] = []
    for meas in measures:
        desc = MEASURE_DESCRIPTORS.get(meas, meas)
        model_name = _selected_overall_model(diag, meas)
        model_support = _gompertz_vs_poly3_support_from_diagnostics(diag, meas)
        model_note = ""
        if model_name is None:
            model_note = "selected overall model: NA"
        else:
            model_note = f"selected overall model: {model_name}"
            if model_name in ASYMPTOTIC_MODELS:
                model_note += " (asymptotic; supports determinate growth)"
            else:
                model_note += " (non-asymptotic; does not support a determinate-growth inference on its own)"

        if model_support is None:
            model_sel_note = "model comparison (gompertz vs poly3): NA"
        else:
            model_sel_note = (
                "model comparison (overall candidates): "
                f"AIC_gompertz={model_support['aic_gompertz']:.1f} vs "
                f"AIC_poly3={model_support['aic_poly3']:.1f}; "
                f"ΔAIC(poly3−gompertz)={model_support['delta_aic_poly3_minus_gompertz']:.1f}; "
                f"Akaike weights: w_gompertz={model_support['w_gompertz']:.3f}, w_poly3={model_support['w_poly3']:.3f}"
            )

        if wild is None or linregress is None:
            slope_note = "adult slope: NA (wild dataset or scipy unavailable)"
            lines.append(f"- {desc}: {slope_note}; {model_note}; {model_sel_note}.")
            continue

        df = wild.dropna(subset=[meas, "age_months"]).copy()
        adults = df[df["age_months"] >= adult_age_months]
        if adults.empty:
            lines.append(f"- {desc}: adult slope: NA (no records with age_months >= {adult_age_months:g}); {model_note}; {model_sel_note}.")
            continue

        try:
            res = linregress(adults["age_months"].to_numpy(), adults[meas].to_numpy())
            slope_year = res.slope * 12.0
            p = res.pvalue
            df_test = int(max(len(adults) - 2, 0))
            stderr = getattr(res, "stderr", None)
            if stderr is not None and np.isfinite(stderr) and stderr > 0:
                t_stat = float(res.slope / stderr)
                stat_note = f"t={t_stat:.2f}, df={df_test}"
            else:
                stat_note = f"df={df_test}"
            # Keep the same heuristic as the TH-only function for consistency.
            if p < 0.05 and slope_year > 1.0:
                verdict = "adult slope suggests continued growth"
            elif p < 0.05 and slope_year < -1.0:
                verdict = "adult slope significantly negative (likely variability/senescence)"
            else:
                verdict = "no significant adult slope"
            lines.append(
                f"- {desc}: adult slope ≈ {slope_year:.2f} units/year (n={len(adults)}, {stat_note}, p={p:.3g}; {verdict}); {model_note}; {model_sel_note}."
            )
        except Exception as exc:
            lines.append(f"- {desc}: adult slope: NA (failed to compute: {exc}); {model_note}; {model_sel_note}.")

    lines.append(
        "Criteria (adult slope): OLS on records with age_months ≥ 120; a positive slope > 1 unit/year with p<0.05 suggests continued growth. "
        "Criteria (model type): asymptotic model selection supports determinate growth; ΔAIC and Akaike weights summarize Gompertz vs poly3 support. "
        "Note: Gompertz vs poly3 is non-nested, so LR-test p-values are not reported for that comparison."
    )
    return "\n".join(lines)


def main() -> None:
    outputs = find_outputs_dir()
    missing_notes: list[str] = []

    # Try to load diagnostics; if unavailable, note but continue with a stub report
    try:
        diag = load_diagnostics(outputs)
    except FileNotFoundError as e:
        diag = pd.DataFrame()
        missing_notes.append(str(e))

    # Try to locate wild dataset for slope/indeterminate section
    wild_path = find_wild_dataset(outputs)
    if wild_path is None:
        missing_notes.append("Wild dataset CSV not found in outputs.")
        wild = None
    else:
        try:
            wild = pd.read_csv(wild_path, parse_dates=["SurveyDate"]) if wild_path.exists() else None
        except Exception as e:
            missing_notes.append(f"Failed to read wild dataset at {wild_path}: {e}")
            wild = None

    # Build report
    lines: list[str] = []
    lines.append(f"# Giraffe Growth Analysis Report\n")
    lines.append(f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}\n")
    lines.append(f"Outputs directory: {outputs}\n")
    if missing_notes:
        lines.append("\n## Data Availability Notes")
        for note in missing_notes:
            lines.append(f"- {note}")

    lines.append("## Final Selected Models")
    lines.append(summarize_selected_models(diag))

    if not diag.empty:
        lines.append("\n## Umbilicus (VTB_Umb) Models (Gompertz-only)")
        lines.append(umbilicus_models_table(diag))

    if not diag.empty:
        lines.append("\n## Sex Differences (TH)")
        sex_msg, delta = sex_difference_summary(diag, measurement="TH")
        lines.append(sex_msg)
        lines.append("Handling unknown sex: Use the pooled overall curve for unknown-sex individuals; if an unknown-sex subset fit exists, it is summarized above but excluded from the ΔAIC test.")

        # New: summarize across all measures
        all_meas = collect_sex_deltas(diag)
        if not all_meas.empty:
            lines.append("\n## Sex Differences (All Measures)")
            # Sorted by largest improvement
            all_meas_sorted = all_meas.sort_values(by="delta_AIC", ascending=False)
            dmin = float(all_meas_sorted["delta_AIC"].min())
            dmax = float(all_meas_sorted["delta_AIC"].max())
            lines.append(f"ΔAIC range (pooled → sex-specific): {dmin:.1f} to {dmax:.1f}.")
            for _, r in all_meas_sorted.iterrows():
                lines.append(
                    f"- {r['descriptor']}: ΔAIC = {r['delta_AIC']:.1f} (overall {r['model_overall']} AIC={r['AIC_overall']:.1f}, M {r['model_M']} AIC={r['AIC_M']:.1f}, F {r['model_F']} AIC={r['AIC_F']:.1f})"
                )

        # LR tests per measure using same-family candidates
        measures = ["TH", "avg NI_FBHcm", "avg TOH_NIcm", "avg TOO_TOHcm"]
        lr_rows = []
        for meas in measures:
            res = lr_test_same_family(diag, meas)
            if res is None:
                continue
            lr_rows.append((meas, res))
        if lr_rows:
            lines.append("\n## Sex Differences (Likelihood-Ratio Tests, Same Family)")
            for meas, res in lr_rows:
                desc = MEASURE_DESCRIPTORS.get(meas, meas)
                p_txt = f"p={res['p']:.3g}" if np.isfinite(res.get("p", np.nan)) else "p=NA (scipy unavailable)"
                lines.append(
                    f"- {desc}: family={res['family']}, LR={res['LR']:.1f}, df={res['df']}, {p_txt}; AIC_overall={res['AIC_overall']:.1f}, AIC_M={res['AIC_M']:.1f}, AIC_F={res['AIC_F']:.1f}"
                )
    else:
        lines.append("\n## Sex Differences (TH)")
        lines.append("Diagnostics unavailable; cannot compute ΔAIC for sex differences.")

    lines.append("\n## Indeterminate Growth Assessment (All Measures)")
    lines.append(determine_indeterminate_growth_all_measures(wild, diag))

    # Save report
    report_path = outputs / "growth_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
