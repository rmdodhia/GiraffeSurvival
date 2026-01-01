"""
Compare zoo and umbilicus wild giraffe growth curves using equivalence testing.

This script validates that zoo animals can serve as a reference for wild giraffe
age estimation by demonstrating equivalence in growth patterns during the first
24 months of life.

Approach:
- Use umbilicus wild animals as a nearly-known-age validation set
- Compare their TH (total height) growth to zoo animals
- Apply equivalence testing with cluster bootstrap (resampling by individual)
- Report whether curves are equivalent within a pre-specified margin

Usage:
    python compare_zoo_umbilicus.py [--margin 20] [--n-bootstrap 1000]

Outputs:
    - Equivalence test results (console and markdown report)
    - Figure: overlaid growth curves with bootstrap CIs
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from typing import Callable

from giraffesurvival.data import (
    WILD_PATH,
    ZOO_PATH,
    load_prepare_zoo,
    load_prepare_wild,
    add_age_class_midpoints,
    assign_initial_ages_from_classes,
    add_vtb_umb_flag,
)
from giraffesurvival.models import fit_single_model, AVAILABLE_MODELS


# Constants
MAX_AGE_MONTHS = 24  # First 2 years
AGE_GRID = np.linspace(0, MAX_AGE_MONTHS, 49)  # 0.5-month resolution


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare zoo and umbilicus wild data."""
    # Zoo data
    zoo = load_prepare_zoo(ZOO_PATH)
    zoo = zoo[zoo["age_months"] <= MAX_AGE_MONTHS].copy()
    zoo["group"] = "zoo"
    zoo["individual_id"] = zoo.get("Name", zoo.index.astype(str))
    
    # Wild data with umbilicus flag
    wild = load_prepare_wild(WILD_PATH)
    wild = add_age_class_midpoints(wild)
    wild = assign_initial_ages_from_classes(wild)
    wild = add_vtb_umb_flag(wild)
    
    # Filter to umbilicus animals and first 24 months
    umb = wild[wild["VTB_Umb_Flag"] == "Umb>0"].copy()
    umb = umb[umb["age_months"] <= MAX_AGE_MONTHS].copy()
    umb["group"] = "umbilicus_wild"
    umb["individual_id"] = umb["AID"]
    umb["height_cm"] = umb["TH"]  # Use TH for wild
    
    return zoo, umb


def fit_mean_curve(df: pd.DataFrame, age_col: str = "age_months", 
                   height_col: str = "height_cm") -> Callable[[np.ndarray], np.ndarray]:
    """Fit a Gompertz curve to the data and return a prediction function."""
    t = df[age_col].to_numpy(dtype=float)
    y = df[height_col].to_numpy(dtype=float)
    
    # Remove NaN
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    
    if len(t) < 10:
        raise ValueError(f"Not enough data points: {len(t)}")
    
    # Fit Gompertz model
    model = AVAILABLE_MODELS["poly3"]
    result = fit_single_model(model, t, y)
    
    if not result.success:
        raise ValueError(f"Model fit failed: {result.message}")
    
    def predict(ages: np.ndarray) -> np.ndarray:
        return result.predict(ages)
    
    return predict, result


def compute_deviation_curve(zoo_df: pd.DataFrame, umb_df: pd.DataFrame, 
                            age_grid: np.ndarray) -> np.ndarray:
    """Compute the deviation between zoo and umbilicus curves at each age point."""
    zoo_pred, _ = fit_mean_curve(zoo_df)
    umb_pred, _ = fit_mean_curve(umb_df)
    
    zoo_heights = zoo_pred(age_grid)
    umb_heights = umb_pred(age_grid)
    
    deviation = umb_heights - zoo_heights
    return deviation


def cluster_bootstrap_max_deviation(
    zoo_df: pd.DataFrame, 
    umb_df: pd.DataFrame,
    age_grid: np.ndarray,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Cluster bootstrap to estimate distribution of max absolute deviation.
    
    Resamples individuals (not individual measurements) to preserve within-animal
    correlation structure.
    
    Returns:
        observed_max_dev: Max absolute deviation from original data
        bootstrap_max_devs: Array of max deviations from bootstrap samples
        deviation_curves: Array of deviation curves from bootstrap samples
    """
    rng = np.random.default_rng(random_state)
    
    # Get unique individual IDs
    zoo_ids = zoo_df["individual_id"].unique()
    umb_ids = umb_df["individual_id"].unique()
    
    print(f"Zoo animals: {len(zoo_ids)}, Umbilicus wild animals: {len(umb_ids)}")
    
    # Observed deviation
    try:
        observed_dev = compute_deviation_curve(zoo_df, umb_df, age_grid)
        observed_max_dev = np.max(np.abs(observed_dev))
    except Exception as e:
        print(f"Warning: Could not compute observed deviation: {e} - compare_zoo_umbilicus.py:141")
        observed_max_dev = np.nan
        observed_dev = np.full_like(age_grid, np.nan)
    
    # Bootstrap
    bootstrap_max_devs = []
    deviation_curves = []
    
    for i in range(n_bootstrap):
        if (i + 1) % 100 == 0:
            print(f"Bootstrap iteration {i + 1}/{n_bootstrap}")
        
        # Resample individuals with replacement
        zoo_sample_ids = rng.choice(zoo_ids, size=len(zoo_ids), replace=True)
        umb_sample_ids = rng.choice(umb_ids, size=len(umb_ids), replace=True)
        
        # Get all measurements for sampled individuals
        zoo_sample = pd.concat([zoo_df[zoo_df["individual_id"] == id_] for id_ in zoo_sample_ids], 
                               ignore_index=True)
        umb_sample = pd.concat([umb_df[umb_df["individual_id"] == id_] for id_ in umb_sample_ids],
                               ignore_index=True)
        
        try:
            dev_curve = compute_deviation_curve(zoo_sample, umb_sample, age_grid)
            max_dev = np.max(np.abs(dev_curve))
            bootstrap_max_devs.append(max_dev)
            deviation_curves.append(dev_curve)
        except Exception:
            # Skip failed fits
            continue
    
    return observed_max_dev, np.array(bootstrap_max_devs), np.array(deviation_curves)


def equivalence_test(
    observed_max_dev: float,
    bootstrap_max_devs: np.ndarray,
    margin: float,
    alpha: float = 0.05,
) -> dict:
    """
    Perform equivalence test: is the 95% CI of max deviation within the margin?
    
    Returns dict with test results.
    """
    # Bootstrap confidence interval for max deviation
    ci_lower = np.percentile(bootstrap_max_devs, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_max_devs, 100 * (1 - alpha / 2))
    
    # Equivalence conclusion: entire CI must be within margin
    equivalent = ci_upper <= margin
    
    return {
        "observed_max_deviation_cm": observed_max_dev,
        "bootstrap_mean_max_deviation_cm": np.mean(bootstrap_max_devs),
        "bootstrap_median_max_deviation_cm": np.median(bootstrap_max_devs),
        "ci_lower_cm": ci_lower,
        "ci_upper_cm": ci_upper,
        "equivalence_margin_cm": margin,
        "equivalent": equivalent,
        "n_bootstrap": len(bootstrap_max_devs),
        "alpha": alpha,
    }


def plot_comparison(
    zoo_df: pd.DataFrame,
    umb_df: pd.DataFrame,
    age_grid: np.ndarray,
    deviation_curves: np.ndarray,
    results: dict,
    output_path: Path,
) -> None:
    """Generate comparison figure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left panel: Growth curves
    ax1 = axes[0]
    
    # Scatter plots
    ax1.scatter(zoo_df["age_months"], zoo_df["height_cm"], 
                alpha=0.3, s=20, c="blue", label="Zoo (n={})".format(len(zoo_df)))
    ax1.scatter(umb_df["age_months"], umb_df["height_cm"], 
                alpha=0.3, s=20, c="green", label="Umbilicus wild (n={})".format(len(umb_df)))
    
    # Fitted curves
    try:
        zoo_pred, zoo_fit = fit_mean_curve(zoo_df)
        umb_pred, umb_fit = fit_mean_curve(umb_df)
        
        ax1.plot(age_grid, zoo_pred(age_grid), "b-", linewidth=2, label="Zoo Gompertz fit")
        ax1.plot(age_grid, umb_pred(age_grid), "g-", linewidth=2, label="Umbilicus Gompertz fit")
    except Exception as e:
        print(f"Could not plot fitted curves: {e} - compare_zoo_umbilicus.py:234")
    
    ax1.set_xlabel("Age (months)")
    ax1.set_ylabel("Height (cm)")
    ax1.set_title("Growth Curves: Zoo vs Umbilicus Wild Giraffe (0-24 months)")
    ax1.legend(loc="lower right")
    ax1.set_xlim(0, MAX_AGE_MONTHS)
    ax1.grid(True, alpha=0.3)
    
    # Right panel: Deviation curve with CI
    ax2 = axes[1]
    
    if len(deviation_curves) > 0:
        # Bootstrap CI for deviation at each age
        dev_lower = np.percentile(deviation_curves, 2.5, axis=0)
        dev_upper = np.percentile(deviation_curves, 97.5, axis=0)
        dev_median = np.median(deviation_curves, axis=0)
        
        ax2.fill_between(age_grid, dev_lower, dev_upper, alpha=0.3, color="purple",
                         label="95% Bootstrap CI")
        ax2.plot(age_grid, dev_median, "purple", linewidth=2, label="Median deviation")
    
    # Equivalence margin
    margin = results["equivalence_margin_cm"]
    ax2.axhline(margin, color="red", linestyle="--", linewidth=1.5, label=f"±{margin} cm margin")
    ax2.axhline(-margin, color="red", linestyle="--", linewidth=1.5)
    ax2.axhline(0, color="black", linestyle="-", linewidth=0.5)
    
    ax2.set_xlabel("Age (months)")
    ax2.set_ylabel("Deviation: Umbilicus - Zoo (cm)")
    ax2.set_title("Height Deviation with Equivalence Margin")
    ax2.legend(loc="upper right")
    ax2.set_xlim(0, MAX_AGE_MONTHS)
    ax2.grid(True, alpha=0.3)
    
    # Add equivalence conclusion
    equiv_text = "EQUIVALENT ✓" if results["equivalent"] else "NOT EQUIVALENT ✗"
    color = "green" if results["equivalent"] else "red"
    ax2.text(0.5, 0.95, equiv_text, transform=ax2.transAxes, fontsize=14, 
             fontweight="bold", color=color, ha="center", va="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to: {output_path}")


def generate_report(
    zoo_df: pd.DataFrame,
    umb_df: pd.DataFrame,
    results: dict,
    output_path: Path,
) -> str:
    """Generate markdown report."""
    
    # Descriptive stats
    zoo_n_animals = zoo_df["individual_id"].nunique()
    zoo_n_obs = len(zoo_df)
    zoo_age_range = (zoo_df["age_months"].min(), zoo_df["age_months"].max())
    
    umb_n_animals = umb_df["individual_id"].nunique()
    umb_n_obs = len(umb_df)
    umb_age_range = (umb_df["age_months"].min(), umb_df["age_months"].max())
    
    equiv_text = "**EQUIVALENT**" if results["equivalent"] else "**NOT EQUIVALENT**"
    
    report = f"""# Zoo vs Umbilicus Wild Giraffe Growth Equivalence Test

## Summary

This analysis tests whether zoo giraffe growth curves can serve as a valid reference 
for estimating ages of wild giraffes by comparing zoo animals to umbilicus-identified 
wild animals (nearly known-age) during the first 24 months of life.

### Conclusion

{equiv_text} within ±{results['equivalence_margin_cm']} cm margin

## Data Summary

| Group | N Animals | N Observations | Age Range (months) |
|-------|-----------|----------------|-------------------|
| Zoo | {zoo_n_animals} | {zoo_n_obs} | {zoo_age_range[0]:.1f} - {zoo_age_range[1]:.1f} |
| Umbilicus Wild | {umb_n_animals} | {umb_n_obs} | {umb_age_range[0]:.1f} - {umb_age_range[1]:.1f} |

## Equivalence Test Results

| Metric | Value |
|--------|-------|
| Observed max absolute deviation | {results['observed_max_deviation_cm']:.2f} cm |
| Bootstrap mean max deviation | {results['bootstrap_mean_max_deviation_cm']:.2f} cm |
| Bootstrap median max deviation | {results['bootstrap_median_max_deviation_cm']:.2f} cm |
| 95% CI lower bound | {results['ci_lower_cm']:.2f} cm |
| 95% CI upper bound | {results['ci_upper_cm']:.2f} cm |
| Equivalence margin | ±{results['equivalence_margin_cm']} cm |
| Number of bootstrap samples | {results['n_bootstrap']} |

## Interpretation

The equivalence test uses cluster bootstrap (resampling individuals, not measurements) 
to estimate the distribution of the maximum absolute deviation between the zoo and 
umbilicus wild growth curves across the 0-24 month age range.

**Equivalence criterion**: The 95% confidence interval for the maximum deviation must 
fall entirely within the pre-specified equivalence margin of ±{results['equivalence_margin_cm']} cm.

- 95% CI: [{results['ci_lower_cm']:.2f}, {results['ci_upper_cm']:.2f}] cm
- Margin: ±{results['equivalence_margin_cm']} cm
- Result: {equiv_text}

{"The growth curves are statistically equivalent within the specified margin, supporting the use of zoo animals as a reference for wild giraffe age estimation." if results['equivalent'] else "The growth curves are NOT statistically equivalent within the specified margin. The differences between zoo and umbilicus wild animals may be too large to justify using zoo curves as a reference without adjustment."}

## Methods

1. **Data**: Zoo height measurements and wild giraffe TH (total height) for 
   umbilicus-identified animals, restricted to ages 0-24 months.

2. **Curve fitting**: Gompertz growth model fitted to each group.

3. **Deviation**: Difference between umbilicus wild and zoo fitted curves computed 
   at 0.5-month intervals.

4. **Bootstrap**: Cluster bootstrap with {results['n_bootstrap']} iterations, resampling 
   individuals (not measurements) to preserve within-animal correlation.

5. **Equivalence test**: 95% bootstrap CI for max absolute deviation compared to 
   pre-specified ±{results['equivalence_margin_cm']} cm margin.

## Figure

See `zoo_umbilicus_comparison.png` for visual comparison.
"""
    
    output_path.write_text(report)
    print(f"Report saved to: {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Compare zoo and umbilicus wild giraffe growth curves"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=20.0,
        help="Equivalence margin in cm (default: 20)",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap iterations (default: 1000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Outputs",
        help="Output directory for report and figure",
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("ZOO vs UMBILICUS WILD GIRAFFE EQUIVALENCE TEST")
    print("=" * 60)
    print(f"Equivalence margin: ±{args.margin} cm")
    print(f"Bootstrap iterations: {args.n_bootstrap}")
    print()
    
    # Prepare data
    print("Loading and preparing data...")
    zoo_df, umb_df = prepare_data()
    
    print(f"\nZoo data: {len(zoo_df)} observations from {zoo_df['individual_id'].nunique()} animals")
    print(f"Umbilicus wild data: {len(umb_df)} observations from {umb_df['individual_id'].nunique()} animals")
    print(f"Age range (zoo): {zoo_df['age_months'].min():.1f}  {zoo_df['age_months'].max():.1f} months")
    print(f"Age range (umb): {umb_df['age_months'].min():.1f}  {umb_df['age_months'].max():.1f} months")
    
    # Run bootstrap
    print(f"\nRunning cluster bootstrap ({args.n_bootstrap} iterations)...")
    observed_max_dev, bootstrap_max_devs, deviation_curves = cluster_bootstrap_max_deviation(
        zoo_df, umb_df, AGE_GRID, n_bootstrap=args.n_bootstrap
    )
    
    # Equivalence test
    print("\nPerforming equivalence test...")
    results = equivalence_test(observed_max_dev, bootstrap_max_devs, args.margin)
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Observed max absolute deviation: {results['observed_max_deviation_cm']:.2f} cm")
    print(f"Bootstrap 95% CI: [{results['ci_lower_cm']:.2f}, {results['ci_upper_cm']:.2f}] cm")
    print(f"Equivalence margin: ±{results['equivalence_margin_cm']} cm")
    print()
    if results['equivalent']:
        print("✓ EQUIVALENT: 95% CI is within the equivalence margin")
        print("Zoo curves can serve as a valid reference for wild giraffe age estimation.")
    else:
        print("✗ NOT EQUIVALENT: 95% CI exceeds the equivalence margin")
        print("Zoo curves may not adequately represent wild giraffe growth.")
    print("=" * 60)
    
    # Generate outputs
    print("\nGenerating outputs...")
    plot_comparison(
        zoo_df, umb_df, AGE_GRID, deviation_curves, results,
        output_dir / "zoo_umbilicus_comparison.png"
    )
    generate_report(
        zoo_df, umb_df, results,
        output_dir / "zoo_umbilicus_equivalence_report.md"
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
