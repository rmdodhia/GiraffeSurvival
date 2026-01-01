from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np
import pandas as pd

from .models import (
    FitResult,
    select_best_model,
    DEFAULT_MODEL_SEQUENCE,
    JUVENILE_MODEL_SEQUENCE,
    CONSTRAINED_MODEL_SEQUENCE,
    FIXED_Y0_MODEL_SEQUENCE,
)

"""
MODEL SEQUENCES:
  DEFAULT_MODEL_SEQUENCE      - gompertz, logistic, von_bertalanffy, richards, poly3, poly4 (unconstrained)
  JUVENILE_MODEL_SEQUENCE     - same as DEFAULT
  CONSTRAINED_MODEL_SEQUENCE  - *_constrained versions, birth height bounded (180 ± 35 cm)
  FIXED_Y0_MODEL_SEQUENCE     - *_fixed_y0 versions, birth height fixed at exactly 180 cm
"""

# Map CLI option names to model sequences for TH
BIRTH_HEIGHT_MODE_MAP: dict[str, tuple[str, ...]] = {
    "none": DEFAULT_MODEL_SEQUENCE,
    "bounded": CONSTRAINED_MODEL_SEQUENCE,
    "fixed": FIXED_Y0_MODEL_SEQUENCE,
}


@dataclass(frozen=True)
class MeasurementConfig:
    column: str
    descriptor: str
    label: str
    overall_min_points: int = 50
    by_sex_min_points: int = 30
    model_names: tuple[str, ...] | None = None
    constrain_birth_height: bool = False  # If True, use birth-height-constrained models


MEASUREMENTS: list[MeasurementConfig] = [
    MeasurementConfig(
        "TH",
        "total height (TH)",
        "wild_th",
        # model_names will be set dynamically based on birth_height_mode
        model_names=None,  # Placeholder - use get_th_model_sequence()
        constrain_birth_height=True,
    ),
    MeasurementConfig(
        "avg TOO_TOHcm",
        "ossicone length (avg TOO_TOHcm)",
        "wild_ossicone",
    ),
    MeasurementConfig(
        "avg TOH_NIcm",
        "neck length (avg TOH_NIcm)",
        "wild_neck",
    ),
    MeasurementConfig(
        "avg NI_FBHcm",
        "foreleg length (avg NI_FBHcm)",
        "wild_leg",
    ),
]


@dataclass(frozen=True)
class AnalysisConfig:
    fit_zoo_overall: bool = True
    fit_zoo_by_sex: bool = True
    fit_wild_overall: bool = True
    fit_wild_by_sex: bool = True
    age_strategy: str = "mixed_effects"  # "mixed_effects" (multi-measure), "mixed_effects_th_only", or "first_measurement"
    birth_height_mode: str = "fixed"  # "none", "bounded", or "fixed" - controls TH model constraint
    outputs_dir: str | None = None
    graphs_dir: str | None = None


def get_measurements_for_config(config: AnalysisConfig) -> list[MeasurementConfig]:
    """Return MEASUREMENTS with TH model_names set based on birth_height_mode."""
    th_models = BIRTH_HEIGHT_MODE_MAP.get(config.birth_height_mode, FIXED_Y0_MODEL_SEQUENCE)
    
    result = []
    for m in MEASUREMENTS:
        if m.column == "TH":
            # Replace TH config with correct model sequence
            result.append(MeasurementConfig(
                column=m.column,
                descriptor=m.descriptor,
                label=m.label,
                overall_min_points=m.overall_min_points,
                by_sex_min_points=m.by_sex_min_points,
                model_names=th_models,
                constrain_birth_height=config.birth_height_mode != "none",
            ))
        else:
            result.append(m)
    return result


def fit_growth_models(
    data: pd.DataFrame,
    value_col: str,
    group_col: str | None = None,
    min_points: int = 50,
    model_names: Sequence[str] | None = None,
) -> tuple[dict[str, FitResult], dict[str, list[FitResult]]]:
    """Fit the best growth model for each requested group."""
    candidate_models = model_names or ("gompertz", "logistic", "von_bertalanffy", "richards", "poly3")
    results: dict[str, FitResult] = {}
    candidate_results: dict[str, list[FitResult]] = {}

    def _fit_subset(subset: pd.DataFrame, key: str) -> None:
        subset = subset.dropna(subset=["age_months", value_col]).copy()
        if len(subset) < min_points:
            print(f"Not enough data to fit {value_col} for {key} - fitting.py:118")
            return

        t = subset["age_months"].to_numpy(dtype=float)
        y = subset[value_col].to_numpy(dtype=float)
        best_fit, candidates = select_best_model(t, y, candidate_models)
        results[key] = best_fit
        candidate_results[key] = candidates

    if group_col is None:
        _fit_subset(data, "overall")
        return results, candidate_results

    for group_val in sorted(data[group_col].dropna().unique()):
        subset = data[data[group_col] == group_val]
        _fit_subset(subset, str(group_val))

    return results, candidate_results


def report_fit_results(
    header: str,
    fits: dict[str, FitResult],
    label_format: str,
) -> None:
    print(f"\n{header} - fitting.py:143")
    if not fits:
        print("No parameters estimated. - fitting.py:145")
        return

    use_template = "{" in label_format and "}" in label_format
    for key, fit in fits.items():
        label = label_format.format(key=key) if use_template else label_format
        if not fit.success:
            print(f"{label}: fit failed ({fit.message}) - fitting.py:152")
            continue

        params_text = ", ".join(f"{name}={value:.4g}" for name, value in fit.param_pairs)
        print(f"{label}: model={fit.model.name}, {params_text}, AIC={fit.aic:.2f} - fitting.py:156")


def fit_individual_growth_models(
    data: pd.DataFrame,
    value_col: str,
    animal_id_col: str = "AID",
    age_col: str = "age_months",
    min_points: int = 4,
    model_names: Sequence[str] | None = None,
    force_single_model: str | None = None,
) -> pd.DataFrame:
    """
    Fit growth models to individual animals with ≥min_points observations.
    
    Args:
        force_single_model: If provided, fit only this model to all animals (no model selection).
                           Useful for ensuring consistent model type across individuals.
    
    Returns DataFrame with columns:
      animal_id, sex, n_obs, model_y0free, A_free, k_free, t0_free, AIC_free,
      model_y0fixed, A_fixed, k_fixed, t0_fixed, AIC_fixed
    
    For each animal:
      - Fits all candidate models in both y0-free and y0=180cm-fixed variants
      - Selects best model for each variant (or uses forced model)
      - Returns parameters and AIC for both variants
    """
    if force_single_model:
        # Use only the specified model
        candidate_models = (force_single_model,)
    else:
        candidate_models = model_names or ("gompertz", "logistic", "von_bertalanffy", "richards", "poly3")
    
    results = []
    
    for animal_id, group in data.groupby(animal_id_col):
        subset = group.dropna(subset=[age_col, value_col]).copy()
        
        if len(subset) < min_points:
            continue
        
        t = subset[age_col].to_numpy(dtype=float)
        y = subset[value_col].to_numpy(dtype=float)
        sex = subset["Sex"].iloc[0] if "Sex" in subset.columns else np.nan
        
        # Fit y0-free variant
        best_fit_free, _ = select_best_model(t, y, candidate_models)
        
        # Fit y0=180cm-fixed variant
        if force_single_model:
            fixed_models = (f"{force_single_model}_fixed_y0",)
        else:
            fixed_models = tuple(f"{m}_fixed_y0" for m in candidate_models)
        best_fit_fixed, _ = select_best_model(t, y, fixed_models)
        
        # Extract parameters
        record = {
            animal_id_col: animal_id,
            "Sex": sex,
            "n_obs": len(subset),
            "model_y0free": best_fit_free.model.name if best_fit_free.success else None,
            "AIC_free": best_fit_free.aic if best_fit_free.success else np.nan,
        }
        
        # Add free-variant parameters (variable number depending on model)
        if best_fit_free.success:
            for param_name, param_value in best_fit_free.param_pairs:
                record[f"{param_name}_free"] = param_value
        
        # Add fixed-variant results
        record["model_y0fixed"] = best_fit_fixed.model.name if best_fit_fixed.success else None
        record["AIC_fixed"] = best_fit_fixed.aic if best_fit_fixed.success else np.nan
        
        if best_fit_fixed.success:
            for param_name, param_value in best_fit_fixed.param_pairs:
                record[f"{param_name}_fixed"] = param_value
        
        results.append(record)
    
    return pd.DataFrame(results)
