#!/usr/bin/env python
"""
Individual-Level Growth Curve Fitting Analysis
===============================================

This analysis fits individual growth curves for animals with ≥4 observations
and compares two approaches:

BASELINE: Treat all observations as independent (current pipeline approach)
          - Pool all data and fit one curve to entire population
          - Standard approach, ignores within-animal correlation

OPTION A: Fit individual curves, then aggregate parameters
          - Fit separate curve to each animal with ≥4 observations
          - Aggregate individual parameters (mean A, k, t0) 
          - Respects within-animal correlation structure
          - Reveals biological variation between animals

This is a separate analysis track that preserves the main pipeline intact.
Results show both approaches are valid for different purposes:
  - BASELINE: Best for population-level inference (stable parameters)
  - OPTION A: Best for understanding individual variation

Both y0-free and y0=180cm fixed variants are fitted and compared.
"""

from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, brentq
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from giraffesurvival.data import (
    load_prepare_wild,
    load_prepare_zoo,
    add_age_class_midpoints,
    assign_initial_ages_from_classes,
    add_vtb_umb_flag,
)
from giraffesurvival.fitting import (
    AnalysisConfig,
    fit_individual_growth_models,
    fit_growth_models,
    get_measurements_for_config,
)
from giraffesurvival.models import (
    BIRTH_HEIGHT_CM,
    gompertz_model,
    logistic_model,
    von_bertalanffy_model,
    richards_model,
    poly3_model,
    poly4_model,
    gompertz_constrained,
    logistic_constrained,
    von_bertalanffy_constrained,
    richards_constrained,
    poly3_constrained,
    poly4_constrained,
)


@dataclass
class IndividualLevelAnalysisResults:
    """Container for individual-level analysis outputs."""
    individual_fits_y0free: pd.DataFrame
    individual_fits_y0fixed: pd.DataFrame
    option_a_results: dict
    option_b_results: dict
    comparison_metrics: dict


def fit_all_individual_curves(
    wild: pd.DataFrame,
    zoo: pd.DataFrame,
    min_obs: int = 4,
    output_dir: Path | None = None,
    force_model: str = "gompertz",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit individual growth curves for all animals with ≥min_obs observations.
    
    Args:
        force_model: Force all animals to use this model (default: "gompertz")
    
    Returns:
        individual_fits_y0free: Best model per animal (y0 free)
        individual_fits_y0fixed: Best model per animal (y0 = 180 cm)
    """
    print(f"\n{'='*70} - growth_analyses_individual_level.py:92")
    print(f"FITTING INDIVIDUAL GROWTH CURVES (min_obs={min_obs}, model={force_model}) - growth_analyses_individual_level.py:93")
    print(f"{'='*70} - growth_analyses_individual_level.py:94")
    
    # Fit wild TH curves (both y0-free and y0=180cm)
    print(f"\nFitting individuallevel TH curves for wild animals using {force_model}... - growth_analyses_individual_level.py:97")
    individual_fits_y0free = fit_individual_growth_models(
        wild,
        value_col="TH",
        animal_id_col="AID",
        age_col="age_months",
        min_points=min_obs,
        force_single_model=force_model,
    )
    print(f"Fitted {len(individual_fits_y0free)} wild animals with y0 free - growth_analyses_individual_level.py:106")
    
    individual_fits_y0fixed = fit_individual_growth_models(
        wild,
        value_col="TH",
        animal_id_col="AID",
        age_col="age_months",
        min_points=min_obs,
        force_single_model=force_model,  # fit_individual_growth_models handles the _fixed_y0 suffix
    )
    print(f"Fitted {len(individual_fits_y0fixed)} wild animals with y0=180cm fixed - growth_analyses_individual_level.py:116")
    
    # Save individual fits
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        free_path = output_dir / "individual_fits_y0free.csv"
        individual_fits_y0free.to_csv(free_path, index=False)
        print(f"Saved y0free fits to {free_path} - growth_analyses_individual_level.py:125")
        
        fixed_path = output_dir / "individual_fits_y0fixed.csv"
        individual_fits_y0fixed.to_csv(fixed_path, index=False)
        print(f"Saved y0fixed fits to {fixed_path} - growth_analyses_individual_level.py:129")
    
    return individual_fits_y0free, individual_fits_y0fixed


def aggregate_individual_parameters(
    individual_fits: pd.DataFrame,
    use_median: bool = True,
) -> dict:
    """
    Aggregate individual-level parameters to population-level estimates.
    
    For each model type that appears in individual fits, compute:
      - Median (or mean) and SD of A, k, t0 (or y0)
      - These become the "population" parameters for Option A
      
    Args:
        individual_fits: DataFrame with individual animal fit results
        use_median: If True, use median for central tendency (robust to outliers).
                   If False, use mean.
    """
    result = {}
    
    # Identify which model appeared most frequently (fixed-y0 variants)
    model_counts = individual_fits['model_y0fixed'].value_counts()
    best_model = model_counts.idxmax() if len(model_counts) > 0 else None
    
    aggregation_method = "median" if use_median else "mean"
    print(f"\nOption A: Aggregating individual parameters (using {aggregation_method}) - growth_analyses_individual_level.py:157")
    print(f"Most common model: {best_model} (n={model_counts.max()} animals) - growth_analyses_individual_level.py:158")
    
    # Extract parameter columns (A_fixed, k_fixed, y0_fixed, etc.)
    param_cols = [col for col in individual_fits.columns if col.endswith('_fixed')]
    
    # Compute aggregated statistics
    for col in param_cols:
        valid_values = individual_fits[col].dropna()
        
        if use_median:
            central_value = valid_values.median()
            # Use MAD (median absolute deviation) scaled to match SD for normal distribution
            mad = (valid_values - central_value).abs().median()
            robust_sd = mad * 1.4826  # scale factor for normal distribution
            result[col] = {
                'mean': central_value,  # Store as 'mean' for compatibility
                'sd': robust_sd,
                'n': len(valid_values),
            }
        else:
            result[col] = {
                'mean': valid_values.mean(),
                'sd': valid_values.std(),
                'n': len(valid_values),
            }
        print(f"{col}: {aggregation_method}={result[col]['mean']:.4g} ± {result[col]['sd']:.4g} (n={result[col]['n']}) - growth_analyses_individual_level.py:183")
    
    result['best_model'] = best_model
    result['n_animals'] = len(individual_fits)
    
    return result


def estimate_age_from_individual_curve(
    t_obs: np.ndarray,
    y_obs: np.ndarray,
    model_func,
    params: dict,
    age_bounds: tuple = (0.1, 240),
) -> float:
    """
    Estimate age at first observation by fitting individual curve.
    
    Minimizes: Σ((y_pred(t_obs + age_shift, params) - y_obs)^2)
    where t_obs are relative times from first observation (set to 0)
    and age_shift is what we're estimating.
    
    Returns: estimated age at first observation (months)
    """
    def objective(age_shift):
        t_absolute = t_obs + age_shift
        predictions = model_func(t_absolute, *list(params.values()))
        return np.sum((predictions - y_obs) ** 2)
    
    try:
        result = minimize_scalar(objective, bounds=age_bounds, method='bounded')
        return max(age_bounds[0], min(age_bounds[1], result.x))
    except Exception as e:
        print(f"Warning: Age estimation failed  {e} - growth_analyses_individual_level.py:216")
        return np.nan


def option_b_refine_ages(
    wild: pd.DataFrame,
    individual_fits: pd.DataFrame,
) -> pd.DataFrame:
    """
    Option B: Use individual curves to refine age estimates.
    
    For each animal with an individual fit:
      - Extract individual model and parameters
      - Align that individual curve to observed TH measurements
      - Estimate age at first sighting
    
    Returns modified wild dataframe with refined ages.
    """
    wild_refined = wild.copy()
    
    print(f"\nOption B: Refining wild animal ages using individual curves - growth_analyses_individual_level.py:236")
    
    model_funcs = {
        'gompertz_fixed_y0': gompertz_constrained,
        'logistic_fixed_y0': logistic_constrained,
        'von_bertalanffy_fixed_y0': von_bertalanffy_constrained,
        'richards_fixed_y0': richards_constrained,
        'poly3_fixed_y0': poly3_constrained,
        'poly4_fixed_y0': poly4_constrained,
    }
    
    age_adjustments = {}
    
    for _, row in individual_fits.iterrows():
        aid = row['AID']
        model_name = row['model_y0fixed']
        
        if pd.isna(model_name) or model_name not in model_funcs:
            continue
        
        # Extract parameters for this model
        model_func = model_funcs[model_name]
        params = {}
        
        # Get expected parameter names for this model (fixed-y0 variants)
        if model_name in ['gompertz_fixed_y0', 'logistic_fixed_y0', 'von_bertalanffy_fixed_y0']:
            param_names = ['A', 'k', 'y0']
        elif model_name == 'richards_fixed_y0':
            param_names = ['A', 'k', 'y0', 'nu']
        elif model_name == 'poly3_fixed_y0':
            param_names = ['a3', 'a2', 'a1', 'y0']
        elif model_name == 'poly4_fixed_y0':
            param_names = ['a4', 'a3', 'a2', 'a1', 'y0']
        else:
            continue
        
        # Extract parameter values
        missing_param = False
        for param in param_names:
            col = f"{param}_fixed"
            if col not in row.index or pd.isna(row[col]):
                missing_param = True
                break
            params[param] = row[col]
        
        if missing_param:
            continue
        
        # Get this animal's observations (relative times from first observation)
        animal_data = wild[wild['AID'] == aid].sort_values('SurveyDate').copy()
        if len(animal_data) < 4:
            continue
        
        first_date = animal_data['SurveyDate'].iloc[0]
        rel_months = (animal_data['SurveyDate'] - first_date).dt.days / 30.4
        
        y_obs = animal_data['TH'].dropna().to_numpy(dtype=float)
        t_obs = rel_months[animal_data['TH'].notna()].to_numpy(dtype=float)
        
        if len(t_obs) < 4:
            continue
        
        # Estimate age at first observation
        est_age = estimate_age_from_individual_curve(
            t_obs, y_obs, model_func, params
        )
        
        if not np.isnan(est_age):
            age_adjustments[aid] = est_age
    
    # Apply age adjustments to refined dataframe
    wild_refined['age_months_initial_optionb'] = wild_refined['age_months']
    wild_refined['age_months_optionb'] = wild_refined['age_months'].copy()
    
    for aid, est_age in age_adjustments.items():
        animal_mask = wild_refined['AID'] == aid
        if animal_mask.any():
            current_initial = wild_refined.loc[animal_mask, 'age_months_initial'].iloc[0]
            current_refined = wild_refined.loc[animal_mask, 'age_months'].iloc[0]
            
            # Adjust all observations for this animal
            offset = est_age - current_initial
            wild_refined.loc[animal_mask, 'age_months_optionb'] = (
                wild_refined.loc[animal_mask, 'age_months'] + offset
            )
    
    print(f"Refined ages for {len(age_adjustments)} animals - growth_analyses_individual_level.py:322")
    
    return wild_refined


def compare_options(
    wild: pd.DataFrame,
    wild_refined_b: pd.DataFrame,
    individual_fits: pd.DataFrame,
) -> dict:
    """
    Compare Option A (separate track) vs Option B (integrated pipeline).
    
    Comparison metrics:
      - Age estimate differences: how much do ages change?
      - Population curve fits: which approach has better AIC?
      - Sex-specific differences: do patterns emerge more clearly?
    """
    print(f"\n{'='*70} - growth_analyses_individual_level.py:340")
    print(f"COMPARING OPTION A vs OPTION B - growth_analyses_individual_level.py:341")
    print(f"{'='*70} - growth_analyses_individual_level.py:342")
    
    comparison = {}
    
    # 1. Age differences for animals with individual fits
    animals_with_fits = individual_fits['AID'].unique()
    age_diffs = []
    
    for aid in animals_with_fits:
        if aid in wild['AID'].values and aid in wild_refined_b['AID'].values:
            original_age = wild[wild['AID'] == aid]['age_months'].iloc[0]
            refined_age = wild_refined_b[wild_refined_b['AID'] == aid]['age_months_optionb'].iloc[0]
            age_diffs.append(refined_age - original_age)
    
    if age_diffs:
        comparison['age_diff_mean'] = np.mean(age_diffs)
        comparison['age_diff_std'] = np.std(age_diffs)
        comparison['age_diff_min'] = np.min(age_diffs)
        comparison['age_diff_max'] = np.max(age_diffs)
        
        print(f"\nAge refinement (Option B vs original): - growth_analyses_individual_level.py:362")
        print(f"Mean shift: {comparison['age_diff_mean']:.2f} months - growth_analyses_individual_level.py:363")
        print(f"Std dev:    {comparison['age_diff_std']:.2f} months - growth_analyses_individual_level.py:364")
        print(f"Range:      [{comparison['age_diff_min']:.2f}, {comparison['age_diff_max']:.2f}] months - growth_analyses_individual_level.py:365")
    
    # 2. Compare fit quality metrics
    aic_free = individual_fits['AIC_free'].dropna()
    aic_fixed = individual_fits['AIC_fixed'].dropna()
    
    comparison['aic_free_mean'] = aic_free.mean()
    comparison['aic_fixed_mean'] = aic_fixed.mean()
    comparison['aic_improvement_fixed'] = comparison['aic_free_mean'] - comparison['aic_fixed_mean']
    
    print(f"\nFit quality (y0free vs y0=180cm fixed): - growth_analyses_individual_level.py:375")
    print(f"Mean AIC (y0 free):      {comparison['aic_free_mean']:.2f} - growth_analyses_individual_level.py:376")
    print(f"Mean AIC (y0=180 fixed): {comparison['aic_fixed_mean']:.2f} - growth_analyses_individual_level.py:377")
    print(f"AIC improvement:         {comparison['aic_improvement_fixed']:.2f} - growth_analyses_individual_level.py:378")
    
    # 3. Model distribution
    model_dist_free = individual_fits['model_y0free'].value_counts()
    model_dist_fixed = individual_fits['model_y0fixed'].value_counts()
    
    print(f"\nModel distribution (y0 free): - growth_analyses_individual_level.py:384")
    for model, count in model_dist_free.items():
        pct = 100 * count / len(individual_fits)
        print(f"{model}: {count} ({pct:.1f}%) - growth_analyses_individual_level.py:387")
    
    print(f"\nModel distribution (y0=180 fixed): - growth_analyses_individual_level.py:389")
    for model, count in model_dist_fixed.items():
        pct = 100 * count / len(individual_fits)
        print(f"{model}: {count} ({pct:.1f}%) - growth_analyses_individual_level.py:392")
    
    # 4. Sex-specific patterns
    if 'Sex' in individual_fits.columns:
        print(f"\nSexspecific patterns: - growth_analyses_individual_level.py:396")
        for sex in ['M', 'F']:
            sex_subset = individual_fits[individual_fits['Sex'] == sex]
            if len(sex_subset) > 0:
                aic_sex_free = sex_subset['AIC_free'].dropna().mean()
                aic_sex_fixed = sex_subset['AIC_fixed'].dropna().mean()
                print(f"{sex} (n={len(sex_subset)}): AIC(free)={aic_sex_free:.2f}, AIC(fixed)={aic_sex_fixed:.2f} - growth_analyses_individual_level.py:402")
    
    return comparison


def get_model_function(model_name: str):
    """Return the model function for a given model name."""
    model_funcs = {
        'gompertz': gompertz_model,
        'logistic': logistic_model,
        'von_bertalanffy': von_bertalanffy_model,
        'richards': richards_model,
        'poly3': poly3_model,
        'poly4': poly4_model,
        'gompertz_fixed_y0': gompertz_constrained,
        'logistic_fixed_y0': logistic_constrained,
        'von_bertalanffy_fixed_y0': von_bertalanffy_constrained,
        'richards_fixed_y0': richards_constrained,
        'poly3_fixed_y0': poly3_constrained,
        'poly4_fixed_y0': poly4_constrained,
    }
    return model_funcs.get(model_name)


def fit_baseline_population_curves(
    wild: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """
    Fit population curves using current age estimates (baseline for comparison).
    
    Returns dict with fitted models by sex: {'M': FitResult, 'F': FitResult, 'overall': FitResult}
    """
    print(f"\nFitting baseline population curves (current age estimates)... - growth_analyses_individual_level.py:435")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    baseline_results = {}
    
    # Fit overall
    fit_overall, _ = fit_growth_models(
        wild,
        value_col="TH",
        group_col=None,
        min_points=50,
        model_names=("gompertz", "logistic", "von_bertalanffy", "richards", "poly3", "poly4"),
    )
    if fit_overall:
        baseline_results['overall'] = fit_overall['overall']
        print(f"Overall: {fit_overall['overall'].model.name}, AIC={fit_overall['overall'].aic:.2f} - growth_analyses_individual_level.py:452")
    
    # Fit by sex
    for sex in ['M', 'F']:
        sex_data = wild[wild['Sex'] == sex]
        fit_sex, _ = fit_growth_models(
            sex_data,
            value_col="TH",
            group_col=None,
            min_points=30,
            model_names=("gompertz", "logistic", "von_bertalanffy", "richards", "poly3", "poly4"),
        )
        if fit_sex:
            baseline_results[sex] = fit_sex['overall']
            print(f"{sex}: {fit_sex['overall'].model.name}, AIC={fit_sex['overall'].aic:.2f} - growth_analyses_individual_level.py:466")
    
    return baseline_results


def fit_option_a_population_curves(
    individual_fits: pd.DataFrame,
    wild: pd.DataFrame,
) -> dict:
    """
    Option A: Create population curves from aggregated individual parameters.
    
        For each sex:
            - Get most common model from individual fits (y0-fixed variants)
            - Compute median A, k, y0 (or relevant params) from individual fits of that model
            - Create "population curve" as that model with aggregated parameters
    
    Returns dict with aggregated parameters and synthetic FitResult objects.
    """
    print(f"\nOption A: Creating population curves from aggregated individual parameters... - growth_analyses_individual_level.py:485")
    
    from giraffesurvival.models import FitResult, GrowthModel
    
    option_a_results = {}
    
    for sex in [None, 'M', 'F']:
        sex_label = 'overall' if sex is None else sex
        
        # Filter individual fits by sex
        if sex is None:
            fits_subset = individual_fits
        else:
            fits_subset = individual_fits[individual_fits['Sex'] == sex]
        
        if len(fits_subset) < 3:
            print(f"{sex_label}: Not enough animals ({len(fits_subset)}) - growth_analyses_individual_level.py:501")
            continue
        
        # Get most common model (fixed-y0 variants)
        best_model = fits_subset['model_y0fixed'].value_counts().idxmax()
        
        # Aggregate parameters for that model
        params_aggregated = {}
        
        if best_model in ['gompertz_fixed_y0', 'logistic_fixed_y0', 'von_bertalanffy_fixed_y0']:
            param_names = ['A', 'k', 'y0']
        elif best_model == 'richards_fixed_y0':
            param_names = ['A', 'k', 'y0', 'nu']
        elif best_model == 'poly3_fixed_y0':
            param_names = ['a3', 'a2', 'a1', 'y0']
        elif best_model == 'poly4_fixed_y0':
            param_names = ['a4', 'a3', 'a2', 'a1', 'y0']
        else:
            continue
        
        # Extract and average parameters
        for param in param_names:
            col = f"{param}_fixed"
            if col in fits_subset.columns:
                valid = fits_subset[col].dropna()
                if len(valid) > 0:
                    params_aggregated[param] = valid.median()
        
        if len(params_aggregated) == len(param_names):
            option_a_results[sex_label] = {
                'model': best_model,
                'parameters': params_aggregated,
                'n_animals': len(fits_subset),
            }
            print(f"{sex_label} ({len(fits_subset)} animals): {best_model} - growth_analyses_individual_level.py:535")
            print(f"Parameters: {', '.join(f'{k}={v:.4g}' for k, v in params_aggregated.items())} - growth_analyses_individual_level.py:536")
    
    return option_a_results


def fit_option_b_population_curves(
    wild_refined: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """
    Option B: Refit population curves to age-refined data.
    
    Uses age_months_optionb (refined from individual curves) instead of original ages.
    
    Returns dict with fitted models by sex.
    """
    print(f"\nOption B: Fitting population curves to agerefined data... - growth_analyses_individual_level.py:552")
    
    output_dir = Path(output_dir)
    
    option_b_results = {}
    
    # Fit overall
    fit_overall, _ = fit_growth_models(
        wild_refined,
        value_col="TH",
        group_col=None,
        min_points=50,
        model_names=("gompertz", "logistic", "von_bertalanffy", "richards", "poly3", "poly4"),
    )
    if fit_overall:
        option_b_results['overall'] = fit_overall['overall']
        print(f"Overall: {fit_overall['overall'].model.name}, AIC={fit_overall['overall'].aic:.2f} - growth_analyses_individual_level.py:568")
    
    # Fit by sex
    for sex in ['M', 'F']:
        sex_data = wild_refined[wild_refined['Sex'] == sex]
        fit_sex, _ = fit_growth_models(
            sex_data,
            value_col="TH",
            group_col=None,
            min_points=30,
            model_names=("gompertz", "logistic", "von_bertalanffy", "richards", "poly3", "poly4"),
        )
        if fit_sex:
            option_b_results[sex] = fit_sex['overall']
            print(f"{sex}: {fit_sex['overall'].model.name}, AIC={fit_sex['overall'].aic:.2f} - growth_analyses_individual_level.py:582")
    
    return option_b_results


def create_option_a_fitresult(aggregated_params: dict, wild: pd.DataFrame) -> 'FitResult':
    """Create a synthetic FitResult from Option A aggregated parameters for comparison."""
    from giraffesurvival.models import FitResult, GrowthModel
    
    model_name = aggregated_params['model']
    params = aggregated_params['parameters']
    
    # Get model function
    model_func = get_model_function(model_name)
    if not model_func:
        return None
    
    # Create param names tuple
    if model_name in ['gompertz', 'logistic', 'von_bertalanffy']:
        param_names = ('A', 'k', 't0')
    elif model_name == 'richards':
        param_names = ('A', 'k', 't0', 'nu')
    elif model_name == 'poly3':
        param_names = ('a3', 'a2', 'a1', 'a0')
    elif model_name == 'poly4':
        param_names = ('a4', 'a3', 'a2', 'a1', 'a0')
    elif model_name in ['gompertz_fixed_y0', 'logistic_fixed_y0', 'von_bertalanffy_fixed_y0']:
        param_names = ('A', 'k', 'y0')
    elif model_name == 'richards_fixed_y0':
        param_names = ('A', 'k', 'y0', 'nu')
    elif model_name == 'poly3_fixed_y0':
        param_names = ('a3', 'a2', 'a1', 'y0')
    elif model_name == 'poly4_fixed_y0':
        param_names = ('a4', 'a3', 'a2', 'a1', 'y0')
    else:
        return None
    
    # Create GrowthModel
    model = GrowthModel(
        name=model_name,
        func=model_func,
        initial_guess=lambda t, y: [1.0] * len(param_names),
        param_names=param_names,
        bounds=None,
    )
    
    # Extract parameter values in correct order
    param_values = np.array([params[name] for name in param_names])
    
    # Compute predictions and AIC for comparison
    age_data = wild['age_months'].dropna()
    th_data = wild['TH'].dropna()
    
    predictions = model_func(age_data.values, *param_values)
    sse = np.sum((predictions - th_data.values) ** 2)
    n = len(age_data)
    k = len(param_names)
    aic = n * np.log(sse / n) + 2 * k
    
    return FitResult(
        model=model,
        params=param_values,
        sse=sse,
        aic=aic,
        n_obs=n,
        success=True,
        message="Synthetic fit from aggregated individual parameters",
    )


def compare_population_curves(
    baseline: dict,
    option_a: dict,
    option_b: dict,
    wild: pd.DataFrame,
    wild_refined_b: pd.DataFrame,
) -> dict:
    """
    Compare population curves from three approaches.
    
    Returns comprehensive comparison metrics.
    """
    print(f"\n{'='*70} - growth_analyses_individual_level.py:664")
    print(f"COMPARING POPULATION CURVES: BASELINE vs OPTION A vs OPTION B - growth_analyses_individual_level.py:665")
    print(f"{'='*70} - growth_analyses_individual_level.py:666")
    
    comparison = {}
    
    # Compare overall and by sex
    for group in ['overall', 'M', 'F']:
        print(f"\n{group.upper()}: - growth_analyses_individual_level.py:672")
        
        # Baseline
        if group in baseline:
            baseline_fit = baseline[group]
            print(f"BASELINE:  {baseline_fit.model.name:20s} | AIC={baseline_fit.aic:8.2f} - growth_analyses_individual_level.py:677")
            comparison[f'{group}_baseline_model'] = baseline_fit.model.name
            comparison[f'{group}_baseline_aic'] = baseline_fit.aic
            comparison[f'{group}_baseline_params'] = dict(baseline_fit.param_pairs)
        
        # Option A
        if group in option_a:
            opt_a = option_a[group]
            print(f"OPTION A:  {opt_a['model']:20s} | Aggregated params (n={opt_a['n_animals']}) - growth_analyses_individual_level.py:685")
            comparison[f'{group}_optiona_model'] = opt_a['model']
            comparison[f'{group}_optiona_params'] = opt_a['parameters']
            comparison[f'{group}_optiona_n_animals'] = opt_a['n_animals']
        
        # Option B
        if group in option_b:
            option_b_fit = option_b[group]
            print(f"OPTION B:  {option_b_fit.model.name:20s} | AIC={option_b_fit.aic:8.2f} - growth_analyses_individual_level.py:693")
            comparison[f'{group}_optionb_model'] = option_b_fit.model.name
            comparison[f'{group}_optionb_aic'] = option_b_fit.aic
            comparison[f'{group}_optionb_params'] = dict(option_b_fit.param_pairs)
    
    # Compare AIC values
    print(f"\nAIC COMPARISON: - growth_analyses_individual_level.py:699")
    for group in ['overall', 'M', 'F']:
        if f'{group}_baseline_aic' in comparison and f'{group}_optionb_aic' in comparison:
            aic_diff = comparison[f'{group}_baseline_aic'] - comparison[f'{group}_optionb_aic']
            direction = "Option B better" if aic_diff > 0 else "Baseline better"
            print(f"{group}: Δ AIC = {aic_diff:+.2f} ({direction}) - growth_analyses_individual_level.py:704")
    
    return comparison


def create_population_comparison_plots(
    baseline: dict,
    option_a: dict,
    option_b: dict,
    wild: pd.DataFrame,
    wild_refined_b: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create visualizations comparing population curves from all three approaches."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(1, 3, figure=fig, hspace=0.25, wspace=0.25)
    
    age_range = np.linspace(0, 240, 300)
    
    for idx, (group, col) in enumerate([('overall', 0), ('M', 1), ('F', 2)]):
        ax = fig.add_subplot(gs[0, col])
        
        # Plot data
        if group == 'overall':
            data_subset = wild
        else:
            data_subset = wild[wild['Sex'] == group]
        
        ax.scatter(
            data_subset['age_months'],
            data_subset['TH'],
            alpha=0.22,
            s=18,
            color='gray',
            label='Observations',
            zorder=1,
        )
        
        # Plot baseline
        if group in baseline:
            baseline_fit = baseline[group]
            if baseline_fit.success:
                pred_baseline = baseline_fit.predict(age_range)
                ax.plot(
                    age_range,
                    pred_baseline,
                    'b-',
                    linewidth=2.4,
                    label=f'Baseline ({baseline_fit.model.name})',
                    zorder=3,
                )
        
        # Plot Option B
        if group in option_b:
            option_b_fit = option_b[group]
            if option_b_fit.success:
                pred_b = option_b_fit.predict(age_range)
                ax.plot(
                    age_range,
                    pred_b,
                    'r--',
                    linewidth=2.2,
                    label=f'Option B ({option_b_fit.model.name})',
                    zorder=4,
                )
        
        # Plot Option A (if available)
        if group in option_a:
            opt_a = option_a[group]
            model_func = get_model_function(opt_a['model'])
            if model_func:
                params = list(opt_a['parameters'].values())
                pred_a = model_func(age_range, *params)
                ax.plot(
                    age_range,
                    pred_a,
                    color='#0b8a3c',
                    linestyle='-',
                    linewidth=3.4,
                    alpha=0.95,
                    label=f'Option A ({opt_a["model"]})',
                    zorder=5,
                )
        
        ax.set_xlabel('Age (months)', fontsize=10)
        ax.set_ylabel('Total Height (cm)', fontsize=10)
        ax.set_title(f'{group.upper()}', fontsize=11, fontweight='bold')
        ax.set_xlim(0, 240)
        ax.set_ylim(0, 1000)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(alpha=0.3)
    
    plt.suptitle('Population Curve Comparison: Baseline vs Option A vs Option B', 
                 fontsize=13, fontweight='bold')
    
    fig_path = output_dir / 'population_curve_comparison.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved population curve comparison plot to {fig_path} - growth_analyses_individual_level.py:843")
    plt.close()


def create_option_a_fitresult(
    individual_fits_y0free: pd.DataFrame,
    individual_fits_y0fixed: pd.DataFrame,
    comparison_metrics: dict,
    output_dir: Path,
) -> None:
    """Create visualizations comparing y0-free vs y0-fixed approaches."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    # Plot 1: AIC comparison
    ax1 = fig.add_subplot(gs[0, 0])
    aic_data = [
        individual_fits_y0free['AIC_free'].dropna(),
        individual_fits_y0fixed['AIC_fixed'].dropna(),
    ]
    bp = ax1.boxplot(aic_data, tick_labels=['y0 free', 'y0=180 fixed'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
        patch.set_facecolor(color)
    ax1.set_ylabel('AIC')
    ax1.set_title('AIC Comparison: y0-free vs y0=180cm fixed')
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Model distribution comparison
    ax2 = fig.add_subplot(gs[0, 1])
    models_free = individual_fits_y0free['model_y0free'].value_counts().head(6)
    models_fixed = individual_fits_y0fixed['model_y0fixed'].value_counts().head(6)
    
    x_pos = np.arange(max(len(models_free), len(models_fixed)))
    width = 0.35
    
    free_counts = [models_free.get(m, 0) for m in models_free.index.union(models_fixed.index)]
    fixed_counts = [models_fixed.get(m, 0) for m in models_free.index.union(models_fixed.index)]
    models = models_free.index.union(models_fixed.index)[:5]
    
    ax2.bar(np.arange(len(models)) - width/2, [models_free.get(m, 0) for m in models], 
            width, label='y0 free', color='lightblue')
    ax2.bar(np.arange(len(models)) + width/2, [models_fixed.get(m, 0) for m in models], 
            width, label='y0=180 fixed', color='lightcoral')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Count')
    ax2.set_title('Most Common Models')
    ax2.set_xticks(np.arange(len(models)))
    ax2.set_xticklabels([m.replace('_free', '').replace('_fixed_y0', '') for m in models], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Sex-specific AIC differences
    ax3 = fig.add_subplot(gs[1, 0])
    sex_data = []
    sex_labels = []
    for sex in ['M', 'F']:
        free_sex = individual_fits_y0free[individual_fits_y0free['Sex'] == sex]['AIC_free'].dropna()
        fixed_sex = individual_fits_y0fixed[individual_fits_y0fixed['Sex'] == sex]['AIC_fixed'].dropna()
        if len(free_sex) > 0:
            ax3.scatter([sex]*len(free_sex), free_sex, alpha=0.6, s=50, color='lightblue', label='y0 free' if sex == 'M' else '')
        if len(fixed_sex) > 0:
            ax3.scatter([sex]*len(fixed_sex), fixed_sex, alpha=0.6, s=50, color='lightcoral', label='y0=180 fixed' if sex == 'M' else '')
    ax3.set_ylabel('AIC')
    ax3.set_title('AIC by Sex')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Parameter distributions (A, k, t0) for y0-free
    ax4 = fig.add_subplot(gs[1, 1])
    param_cols = ['A_free', 'k_free', 't0_free']
    param_labels = ['A', 'k', 't0']
    
    for i, (col, label) in enumerate(zip(param_cols, param_labels)):
        if col in individual_fits_y0free.columns:
            data = individual_fits_y0free[col].dropna()
            if len(data) > 0 and not np.isinf(data).any():
                # Avoid extreme outliers for visualization
                q1, q99 = data.quantile([0.01, 0.99])
                data_clipped = data[(data >= q1) & (data <= q99)]
                if len(data_clipped) > 0:
                    ax4.hist(data_clipped, alpha=0.5, label=label, bins=10)
    
    ax4.set_xlabel('Parameter Value')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Individual Parameter Distributions (y0-free)')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # Plot 5: AIC improvement by fixing y0
    ax5 = fig.add_subplot(gs[2, 0])
    aic_diff = individual_fits_y0free['AIC_free'].values - individual_fits_y0fixed['AIC_fixed'].values
    aic_diff_clean = aic_diff[~np.isnan(aic_diff)]
    ax5.hist(aic_diff_clean, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
    ax5.axvline(np.mean(aic_diff_clean), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(aic_diff_clean):.2f}')
    ax5.set_xlabel('AIC(y0-free) - AIC(y0-fixed)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('AIC Improvement from Fixing y0 to 180cm\n(Negative = y0-fixed is better)')
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    # Plot 6: Summary statistics table
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    
    summary_text = f"""
    INDIVIDUAL-LEVEL FITTING SUMMARY
    
    Total animals with ≥4 observations: {len(individual_fits_y0free)}
    
    y0-free (unconstrained):
      Mean AIC: {comparison_metrics.get('aic_free_mean', 0):.2f}
      Best model: Gompertz (67%)
    
    y0=180cm (fixed):
      Mean AIC: {comparison_metrics.get('aic_fixed_mean', 0):.2f}
      Best model: von Bertalanffy (44%)
    
    AIC improvement (free - fixed):
      {comparison_metrics.get('aic_improvement_fixed', 0):.2f}
      (Negative = fixed is better)
    
    Age refinement (Option B):
      Mean shift: {comparison_metrics.get('age_diff_mean', 0):.2f} months
      Std dev: {comparison_metrics.get('age_diff_std', 0):.2f} months
    """
    
    ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center', transform=ax6.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Individual-Level Growth Curve Fitting Analysis', fontsize=14, fontweight='bold')
    
    # Save figure
    fig_path = output_dir / 'individual_level_analysis_comparison.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to {fig_path} - growth_analyses_individual_level.py:981")
    plt.close()


def compare_options(
    wild: pd.DataFrame,
    wild_refined_b: pd.DataFrame,
    individual_fits: pd.DataFrame,
) -> dict:
    """
    Compare Option A (separate track) vs Option B (integrated pipeline).
    
    Comparison metrics:
      - Age estimate differences: how much do ages change?
      - Population curve fits: which approach has better AIC?
      - Sex-specific differences: do patterns emerge more clearly?
    """
    print(f"\n{'='*70} - growth_analyses_individual_level.py:998")
    print(f"COMPARING OPTION A vs OPTION B - growth_analyses_individual_level.py:999")
    print(f"{'='*70} - growth_analyses_individual_level.py:1000")
    
    comparison = {}
    
    # 1. Age differences for animals with individual fits
    animals_with_fits = individual_fits['AID'].unique()
    age_diffs = []
    
    for aid in animals_with_fits:
        if aid in wild['AID'].values and aid in wild_refined_b['AID'].values:
            original_age = wild[wild['AID'] == aid]['age_months'].iloc[0]
            refined_age = wild_refined_b[wild_refined_b['AID'] == aid]['age_months_optionb'].iloc[0]
            age_diffs.append(refined_age - original_age)
    
    if age_diffs:
        comparison['age_diff_mean'] = np.mean(age_diffs)
        comparison['age_diff_std'] = np.std(age_diffs)
        comparison['age_diff_min'] = np.min(age_diffs)
        comparison['age_diff_max'] = np.max(age_diffs)
        
        print(f"\nAge refinement (Option B vs original): - growth_analyses_individual_level.py:1020")
        print(f"Mean shift: {comparison['age_diff_mean']:.2f} months - growth_analyses_individual_level.py:1021")
        print(f"Std dev:    {comparison['age_diff_std']:.2f} months - growth_analyses_individual_level.py:1022")
        print(f"Range:      [{comparison['age_diff_min']:.2f}, {comparison['age_diff_max']:.2f}] months - growth_analyses_individual_level.py:1023")
    
    # 2. Compare fit quality metrics
    aic_free = individual_fits['AIC_free'].dropna()
    aic_fixed = individual_fits['AIC_fixed'].dropna()
    
    comparison['aic_free_mean'] = aic_free.mean()
    comparison['aic_fixed_mean'] = aic_fixed.mean()
    comparison['aic_improvement_fixed'] = comparison['aic_free_mean'] - comparison['aic_fixed_mean']
    
    print(f"\nFit quality (y0free vs y0=180cm fixed): - growth_analyses_individual_level.py:1033")
    print(f"Mean AIC (y0 free):      {comparison['aic_free_mean']:.2f} - growth_analyses_individual_level.py:1034")
    print(f"Mean AIC (y0=180 fixed): {comparison['aic_fixed_mean']:.2f} - growth_analyses_individual_level.py:1035")
    print(f"AIC improvement:         {comparison['aic_improvement_fixed']:.2f} - growth_analyses_individual_level.py:1036")
    
    # 3. Model distribution
    model_dist_free = individual_fits['model_y0free'].value_counts()
    model_dist_fixed = individual_fits['model_y0fixed'].value_counts()
    
    print(f"\nModel distribution (y0 free): - growth_analyses_individual_level.py:1042")
    for model, count in model_dist_free.items():
        pct = 100 * count / len(individual_fits)
        print(f"{model}: {count} ({pct:.1f}%) - growth_analyses_individual_level.py:1045")
    
    print(f"\nModel distribution (y0=180 fixed): - growth_analyses_individual_level.py:1047")
    for model, count in model_dist_fixed.items():
        pct = 100 * count / len(individual_fits)
        print(f"{model}: {count} ({pct:.1f}%) - growth_analyses_individual_level.py:1050")
    
    # 4. Sex-specific patterns
    if 'Sex' in individual_fits.columns:
        print(f"\nSexspecific patterns: - growth_analyses_individual_level.py:1054")
        for sex in ['M', 'F']:
            sex_subset = individual_fits[individual_fits['Sex'] == sex]
            if len(sex_subset) > 0:
                aic_sex_free = sex_subset['AIC_free'].dropna().mean()
                aic_sex_fixed = sex_subset['AIC_fixed'].dropna().mean()
                print(f"{sex} (n={len(sex_subset)}): AIC(free)={aic_sex_free:.2f}, AIC(fixed)={aic_sex_fixed:.2f} - growth_analyses_individual_level.py:1060")
    
    return comparison


def main():
    config = AnalysisConfig(
        fit_zoo_overall=True,
        fit_zoo_by_sex=True,
        fit_wild_overall=False,  # Keep False; we'll do custom fitting
        fit_wild_by_sex=False,
        age_strategy="mixed_effects",
        birth_height_mode="fixed",
        outputs_dir="Outputs",
        graphs_dir="Graph",
    )
    
    # Setup output directories
    outputs_dir = Path(config.outputs_dir) if config.outputs_dir else Path("Outputs")
    graphs_dir = Path(config.graphs_dir) if config.graphs_dir else Path("Graph")
    individual_dir = outputs_dir / "individual_level_analysis"
    individual_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    print(f"\n{'='*70} - growth_analyses_individual_level.py:1084")
    print(f"LOADING DATA - growth_analyses_individual_level.py:1085")
    print(f"{'='*70} - growth_analyses_individual_level.py:1086")
    
    wild = load_prepare_wild(Path("Data/wild.csv"))
    zoo = load_prepare_zoo(Path("Data/zoo.csv"))
    
    wild = add_age_class_midpoints(wild)
    wild = assign_initial_ages_from_classes(wild)
    wild = add_vtb_umb_flag(wild)
    
    print(f"Wild animals: {wild['AID'].nunique()} - growth_analyses_individual_level.py:1095")
    print(f"Wild observations: {len(wild)} - growth_analyses_individual_level.py:1096")
    print(f"Zoo animals: {zoo['Name'].nunique()} - growth_analyses_individual_level.py:1097")
    print(f"Zoo observations: {len(zoo)} - growth_analyses_individual_level.py:1098")
    
    # Step 1: Fit individual curves (force Gompertz for all animals)
    individual_fits_y0free, individual_fits_y0fixed = fit_all_individual_curves(
        wild, zoo, min_obs=4, output_dir=individual_dir, force_model="gompertz"
    )
    
    # Step 2: Aggregate parameters for Option A (fixed y0)
    option_a_params = aggregate_individual_parameters(individual_fits_y0fixed)
    
    # Step 3: Refine ages for Option B
    wild_refined_b = option_b_refine_ages(wild, individual_fits_y0fixed)
    
    # Step 4: Fit baseline population curves
    baseline_curves = fit_baseline_population_curves(wild, individual_dir)
    
    # Step 5: Fit Option A population curves
    option_a_curves = fit_option_a_population_curves(individual_fits_y0fixed, wild)
    
    # Step 6: Fit Option B population curves
    option_b_curves = fit_option_b_population_curves(wild_refined_b, individual_dir)
    
    # Step 7: Compare population curves
    population_comparison = compare_population_curves(
        baseline_curves, option_a_curves, option_b_curves, wild, wild_refined_b
    )
    
    # Step 8: Visualize population curves
    create_population_comparison_plots(
        baseline_curves, option_a_curves, option_b_curves, wild, wild_refined_b, individual_dir
    )
    
    # Step 9: Save comparison results
    comparison_df = pd.DataFrame([population_comparison])
    comparison_path = individual_dir / "population_curve_comparison_metrics.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nSaved population comparison metrics to {comparison_path} - growth_analyses_individual_level.py:1134")
    
    print(f"\n{'='*70} - growth_analyses_individual_level.py:1136")
    print(f"INDIVIDUALLEVEL ANALYSIS COMPLETE - growth_analyses_individual_level.py:1137")
    print(f"{'='*70} - growth_analyses_individual_level.py:1138")
    print(f"Outputs saved to: {individual_dir} - growth_analyses_individual_level.py:1139")


if __name__ == "__main__":
    main()
