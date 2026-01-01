"""Check for evidence of indeterminate giraffe growth using adult height trends.

This script filters measurements to adult ages (default >= 6 years), fits a simple
linear model ``height = slope * age_years + intercept`` for each dataset, and tests
whether the slope differs significantly from zero.

It currently assesses two data sources:
- Zoo data (``Data/zoo.csv``)
- Wild data with estimated ages (``wild_with_age_estimates_sex_agnostic_then_by_sex.csv``)

The wild dataset is expected to contain an ``age_months`` column. If you have not
run the broader growth pipeline yet, generate that file first or point the script
to a CSV that includes age estimates.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
import datetime as dt

import numpy as np
import pandas as pd
from scipy.stats import linregress


@dataclass
class SlopeResult:
    dataset: str
    group: str
    n_obs: int
    slope_cm_per_year: float
    intercept_cm: float
    p_value: float
    r_squared: float
    slope_stderr: float

    @property
    def interpretation(self) -> str:
        if self.n_obs < 3:
            return "insufficient data"
        if np.isnan(self.p_value):
            return "fit failed"
        return "significant" if self.p_value < 0.05 else "not significant"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate adult height slopes to check for indeterminate growth.",
    )
    parser.add_argument(
        "--adult-min-years",
        type=float,
        default=8.0,
        help="Minimum age in years for the adult subset (default: 6.0).",
    )
    parser.add_argument(
        "--zoo-path",
        type=Path,
        default=Path("Data/zoo.csv"),
        help="Path to the zoo measurement CSV (default: Data/zoo.csv).",
    )
    parser.add_argument(
        "--wild-path",
        type=Path,
        default=Path("wild_with_age_estimates_sex_agnostic_then_by_sex.csv"),
        help=(
            "Path to the wild dataset CSV. Must include an age_months column "
            "(default: wild_with_age_estimates_sex_agnostic_then_by_sex.csv)."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for interpreting the slope test (default: 0.05).",
    )
    return parser.parse_args()


def _prepare_zoo(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Age (days)" in df.columns:
        df["age_months"] = df["Age (days)"] / 30.4375
    elif "Age (months)" in df.columns:
        df["age_months"] = df["Age (months)"]
    else:
        raise ValueError("Zoo dataset must include either 'Age (days)' or 'Age (months)'.")

    if "Height (cm)" not in df.columns:
        raise ValueError("Zoo dataset must include 'Height (cm)'.")

    df["age_years"] = df["age_months"] / 12.0
    df["height_cm"] = df["Height (cm)"]
    df["Sex"] = df.get("Sex (M/F)", pd.Series(dtype=str)).astype(str).str.strip()
    df.loc[~df["Sex"].isin(["M", "F"]), "Sex"] = np.nan
    return df[["Name", "Sex", "age_years", "height_cm"]]


def _prepare_wild(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "age_months" not in df.columns:
        raise ValueError(
            "Wild dataset must contain an 'age_months' column. Run the growth pipeline "
            "to generate ages or supply a file that already includes them."
        )

    if "TH" in df.columns:
        df["height_cm"] = df["TH"]
    else:
        required = {"avg TOO_TOHcm", "avg TOH_NIcm", "avg NI_FBHcm"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(
                "Wild dataset is missing height columns: " + ", ".join(sorted(missing))
            )
        df["height_cm"] = df["avg TOO_TOHcm"] + df["avg TOH_NIcm"] + df["avg NI_FBHcm"]

    df["age_years"] = df["age_months"] / 12.0
    df["Sex"] = df.get("Sex", pd.Series(dtype=str)).astype(str).str.strip()
    df.loc[df["Sex"].isin(["", "nan", "NA"]) | df["Sex"].isna(), "Sex"] = np.nan
    return df[["AID", "Sex", "age_years", "height_cm"]]


def _run_linear_fit(label: str, group: str, ages: pd.Series, heights: pd.Series) -> SlopeResult | None:
    mask = ages.notna() & heights.notna()
    x = ages[mask]
    y = heights[mask]
    if len(x) < 3 or np.isclose(np.std(x), 0.0):
        return None

    fit = linregress(x, y)
    return SlopeResult(
        dataset=label,
        group=group,
        n_obs=len(x),
        slope_cm_per_year=float(fit.slope),
        intercept_cm=float(fit.intercept),
        p_value=float(fit.pvalue),
        r_squared=float(fit.rvalue ** 2),
        slope_stderr=float(fit.stderr),
    )


def _summarise_results(results: Iterable[SlopeResult], alpha: float) -> None:
    header = (
        f"{'Dataset':<12} {'Group':<12} {'N':>4} {'Slope (cm/yr)':>14} "
        f"{'Std Err':>10} {'p-value':>10} {'R^2':>8} {'Conclusion':>14}"
    )
    print(header)
    print("-" * len(header))
    for res in results:
        conclusion = "significant" if res.p_value < alpha else "not significant"
        print(
            f"{res.dataset:<12} {res.group:<12} {res.n_obs:>4} "
            f"{res.slope_cm_per_year:>14.3f} {res.slope_stderr:>10.3f} "
            f"{res.p_value:>10.3g} {res.r_squared:>8.3f} {conclusion:>14}"
        )


def save_results_to_csv(results: list[SlopeResult], output_path: Path) -> None:
    """Save results to a CSV file."""
    data = [asdict(res) for res in results]
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Saved CSV to: {output_path}")


def save_results_to_markdown(results: list[SlopeResult], alpha: float, output_path: Path) -> None:
    """Save results to a markdown report."""
    lines = [
        "# Indeterminate Growth Assessment (Sex-Specific Adult Slopes)",
        "",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Summary",
        "",
        "This analysis tests for indeterminate growth by examining adult height slopes.",
        "Animals aged ≥ 120 months (10 years) are classified as adults.",
        "A positive slope significantly different from zero (p < 0.05) suggests continued growth (indeterminate).",
        "",
        "## Results Table",
        "",
    ]
    
    # Create markdown table
    header = "| Dataset | Group | N | Slope (cm/yr) | Std Err | p-value | R² | Interpretation |"
    separator = "|---|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(separator)
    
    for res in results:
        conclusion = "**SIGNIFICANT** ✓" if res.p_value < alpha else "not significant"
        lines.append(
            f"| {res.dataset} | {res.group} | {res.n_obs} | {res.slope_cm_per_year:.3f} | "
            f"{res.slope_stderr:.3f} | {res.p_value:.3g} | {res.r_squared:.3f} | {conclusion} |"
        )
    
    lines.append("")
    lines.append("## Detailed Interpretation")
    lines.append("")
    
    # Organize by dataset
    for dataset in ["zoo", "wild"]:
        dataset_results = [r for r in results if r.dataset == dataset]
        if not dataset_results:
            continue
        
        lines.append(f"### {dataset.upper()}")
        lines.append("")
        
        for res in dataset_results:
            lines.append(f"**{res.group.upper()}** (n={res.n_obs})")
            lines.append("")
            lines.append(f"- Slope: {res.slope_cm_per_year:.3f} cm/year (± {res.slope_stderr:.3f})")
            lines.append(f"- p-value: {res.p_value:.3g}")
            lines.append(f"- R²: {res.r_squared:.3f}")
            lines.append("")
            
            if res.n_obs < 3:
                interpretation = "**Insufficient data** for reliable estimation."
            elif np.isnan(res.p_value):
                interpretation = "**Fit failed** due to singular or degenerate data."
            elif res.p_value < alpha:
                if res.slope_cm_per_year > 1.0:
                    interpretation = (
                        f"**SIGNIFICANT POSITIVE SLOPE** suggests **indeterminate growth** "
                        f"(continued height increase in adults at ~{res.slope_cm_per_year:.2f} cm/year)."
                    )
                else:
                    interpretation = (
                        f"**SIGNIFICANT negative or small positive slope** "
                        f"(p < 0.05, but slope magnitude is small)."
                    )
            else:
                interpretation = (
                    "**No significant adult slope** suggests **determinate growth** "
                    "(adults have reached final height)."
                )
            
            lines.append(interpretation)
            lines.append("")
    
    lines.append("## Biological Interpretation")
    lines.append("")
    
    # Check for sex differences in wild
    wild_results = [r for r in results if r.dataset == "wild"]
    if len(wild_results) > 1:
        male_res = next((r for r in wild_results if r.group in ["sex=M", "M"]), None)
        female_res = next((r for r in wild_results if r.group in ["sex=F", "F"]), None)
        
        if male_res and female_res:
            lines.append("### Sex Differences in Wild Giraffes")
            lines.append("")
            
            male_sig = male_res.p_value < alpha
            female_sig = female_res.p_value < alpha
            
            if male_sig and not female_sig:
                lines.append(
                    f"**Important Finding**: Males show significant continued adult growth "
                    f"({male_res.slope_cm_per_year:.2f} cm/yr, p={male_res.p_value:.3g}), "
                    f"while females do not ({female_res.slope_cm_per_year:.2f} cm/yr, p={female_res.p_value:.3g}). "
                    f"This suggests **sex-specific indeterminate growth**, with males continuing to grow throughout adulthood."
                )
            elif female_sig and not male_sig:
                lines.append(
                    f"**Important Finding**: Females show significant continued adult growth "
                    f"({female_res.slope_cm_per_year:.2f} cm/yr, p={female_res.p_value:.3g}), "
                    f"while males do not ({male_res.slope_cm_per_year:.2f} cm/yr, p={male_res.p_value:.3g}). "
                    f"This suggests **sex-specific indeterminate growth**, with females continuing to grow throughout adulthood."
                )
            elif male_sig and female_sig:
                lines.append(
                    f"Both sexes show significant adult slopes: males ({male_res.slope_cm_per_year:.2f} cm/yr, p={male_res.p_value:.3g}) "
                    f"and females ({female_res.slope_cm_per_year:.2f} cm/yr, p={female_res.p_value:.3g}). "
                    f"Both show evidence of indeterminate growth, though rates may differ."
                )
            else:
                lines.append(
                    f"Neither sex shows significant adult growth: males (p={male_res.p_value:.3g}) "
                    f"or females (p={female_res.p_value:.3g}). This suggests determinate growth in both sexes."
                )
            
            lines.append("")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved markdown report to: {output_path}")


def analyse_dataset(
    label: str,
    df: pd.DataFrame,
    min_years: float,
    alpha: float,
) -> list[SlopeResult]:
    adults = df[df["age_years"] >= min_years].copy()
    results: list[SlopeResult] = []

    overall = _run_linear_fit(label, "overall", adults["age_years"], adults["height_cm"])
    if overall:
        results.append(overall)

    if "Sex" in adults.columns:
        for sex_value, group_df in adults.groupby("Sex", dropna=False):
            sex_label = "sex=unknown" if pd.isna(sex_value) else f"sex={sex_value}"
            group_result = _run_linear_fit(label, sex_label, group_df["age_years"], group_df["height_cm"])
            if group_result:
                results.append(group_result)

    return results


def main() -> None:
    args = parse_args()

    zoo_df = _prepare_zoo(args.zoo_path)
    wild_df = _prepare_wild(args.wild_path)

    results: list[SlopeResult] = []
    results.extend(analyse_dataset("zoo", zoo_df, args.adult_min_years, args.alpha))
    results.extend(analyse_dataset("wild", wild_df, args.adult_min_years, args.alpha))

    if not results:
        print("No adult records available for the requested age threshold.")
        return

    # Print to console
    print()
    _summarise_results(results, args.alpha)
    print()
    
    # Save to files
    output_dir = Path("Outputs")
    output_dir.mkdir(exist_ok=True)
    
    csv_path = output_dir / "indeterminate_growth_results.csv"
    md_path = output_dir / "indeterminate_growth_report.md"
    
    save_results_to_csv(results, csv_path)
    save_results_to_markdown(results, args.alpha, md_path)


if __name__ == "__main__":
    main()
