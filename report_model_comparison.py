"""
Generate formal model comparison tables for giraffe growth curves.

This script reads the model_fit_diagnostics.csv and produces:
1. A summary table showing AIC/SSE for all models by measurement and group
2. Statistical comparison (delta AIC, Akaike weights)
3. Markdown report suitable for publication

Usage:
    python report_model_comparison.py [--input PATH] [--output PATH]
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def df_to_markdown(df: pd.DataFrame) -> str:
    """Convert DataFrame to markdown table without tabulate dependency."""
    cols = df.columns.tolist()
    # Header
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"
    # Rows
    rows = []
    for _, row in df.iterrows():
        row_str = "| " + " | ".join(str(row[c]) for c in cols) + " |"
        rows.append(row_str)
    return "\n".join([header, separator] + rows)


def load_diagnostics(path: Path) -> pd.DataFrame:
    """Load and clean the diagnostics CSV."""
    df = pd.read_csv(path)
    # Filter to successful fits only
    df = df[df["success"] == True].copy()
    return df


def compute_akaike_weights(aic_values: pd.Series) -> pd.Series:
    """Compute Akaike weights from AIC values.
    
    w_i = exp(-0.5 * delta_i) / sum(exp(-0.5 * delta_j))
    where delta_i = AIC_i - AIC_min
    """
    aic_min = aic_values.min()
    delta = aic_values - aic_min
    exp_delta = np.exp(-0.5 * delta)
    weights = exp_delta / exp_delta.sum()
    return weights


def generate_comparison_table(df: pd.DataFrame, context_filter: str | None = None) -> pd.DataFrame:
    """Generate a comparison table for all models within each measurement/group."""
    
    if context_filter:
        df = df[df["context"].str.contains(context_filter, case=False, na=False)]
    
    results = []
    
    # Group by context, measurement, group
    for (context, measurement, group), group_df in df.groupby(["context", "measurement", "group"]):
        group_df = group_df.sort_values("aic")
        aic_min = group_df["aic"].min()
        
        # Compute Akaike weights
        weights = compute_akaike_weights(group_df["aic"])
        
        for idx, (_, row) in enumerate(group_df.iterrows()):
            delta_aic = row["aic"] - aic_min
            results.append({
                "Context": context,
                "Measurement": measurement,
                "Group": group,
                "Rank": idx + 1,
                "Model": row["model"],
                "AIC": row["aic"],
                "ΔAIC": delta_aic,
                "Akaike Weight": weights.iloc[idx],
                "SSE": row["sse"],
                "N": row["n_obs"],
                "Selected": "✓" if row["is_selected"] else "",
                "Parameters": row["params"] if pd.notna(row["params"]) else "",
            })
    
    return pd.DataFrame(results)


def generate_summary_table(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Generate a condensed summary showing best model per context/measurement/group."""
    
    # Get rank 1 models only
    best = comparison_df[comparison_df["Rank"] == 1].copy()
    
    summary = best[["Context", "Measurement", "Group", "Model", "AIC", "Akaike Weight", "N"]].copy()
    summary = summary.sort_values(["Context", "Measurement", "Group"])
    
    return summary


def generate_substantially_supported_table(comparison_df: pd.DataFrame, tolerance_pct: float = 1.0) -> pd.DataFrame:
    """
    Generate a summary of all models within tolerance_pct of the best model's AIC.
    
    This implements the "substantially supported" criterion: if a model's AIC is within
    tolerance_pct of the best AIC, it should be reported alongside the best model.
    """
    results = []
    
    # Group by context, measurement, group
    for (context, measurement, group), group_df in comparison_df.groupby(["Context", "Measurement", "Group"]):
        best_aic = group_df["AIC"].min()
        tolerance = (tolerance_pct / 100.0) * best_aic
        
        # Get all models within tolerance
        supported = group_df[group_df["AIC"] <= (best_aic + tolerance)].copy()
        supported = supported.sort_values("AIC")
        
        for idx, (_, row) in enumerate(supported.iterrows()):
            delta_aic = row["AIC"] - best_aic
            results.append({
                "Context": context,
                "Measurement": measurement,
                "Group": group,
                "Model": row["Model"],
                "AIC": row["AIC"],
                "ΔAIC": delta_aic,
                "Akaike Weight": row["Akaike Weight"],
                "Status": "BEST" if idx == 0 else f"Supported ({delta_aic:.2f} ΔAIC)",
            })
    
    return pd.DataFrame(results)


def generate_model_frequency_table(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Count how often each model is selected as best (rank 1)."""
    best = comparison_df[comparison_df["Rank"] == 1]
    freq = best.groupby("Model").size().reset_index(name="Times Best")
    freq = freq.sort_values("Times Best", ascending=False)
    return freq


def format_markdown_report(
    comparison_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    substantially_supported_df: pd.DataFrame,
    freq_df: pd.DataFrame,
) -> str:
    """Format the comparison as a Markdown report, reorganized by model form."""
    
    lines = [
        "# Growth Model Comparison Report",
        "",
        "This report compares multiple growth models fitted to giraffe measurement data.",
        "Model selection is based on AIC (Akaike Information Criterion), with lower values indicating better fit.",
        "",
        "## Summary: Best Models and Substantially Supported Alternatives",
        "",
        "The following table shows the best-fitting model for each measurement and group, plus any alternatives",
        "within 1% of the best model's AIC (considered substantially supported).",
        "",
        substantially_supported_df.to_markdown(index=False),
        "",
        "## Model Selection Summary",
        "",
        "### How Often Each Model Was Selected as Best",
        "",
        freq_df.to_markdown(index=False),
        "",
        "## Model Comparison by Model Form",
        "",
        "The following sections organize model performance by model type (Gompertz, Logistic, Polynomial, etc.),",
        "showing how each model performs across different measurements, sexes, datasets, and contexts.",
        "",
    ]
    
    # Extract model form (e.g., "gompertz" from "gompertz_constrained" or "gompertz_fixed_y0")
    comparison_df["model_form"] = comparison_df["Model"].str.split("_").str[0]
    
    # Group by model form
    for model_form in sorted(comparison_df["model_form"].unique()):
        model_df = comparison_df[comparison_df["model_form"] == model_form]
        
        lines.append(f"### {model_form.upper()}")
        lines.append("")
        
        # Group by context, measurement, group
        for (context, measurement, group), grp_df in model_df.groupby(["Context", "Measurement", "Group"]):
            grp_df_sorted = grp_df.sort_values("Rank")
            
            lines.append(f"**{measurement}** ({group}) — {context} (N={grp_df['N'].iloc[0]})")
            lines.append("")
            
            # Format table
            display_df = grp_df_sorted[["Rank", "Model", "AIC", "ΔAIC", "Akaike Weight"]].copy()
            display_df["AIC"] = display_df["AIC"].apply(lambda x: f"{x:.2f}")
            display_df["ΔAIC"] = display_df["ΔAIC"].apply(lambda x: f"{x:.2f}")
            display_df["Akaike Weight"] = display_df["Akaike Weight"].apply(lambda x: f"{x:.4f}")
            
            lines.append(display_df.to_markdown(index=False))
            lines.append("")
    
    # Add interpretation guide
    lines.extend([
        "## Interpretation Guide",
        "",
        "- **ΔAIC**: Difference from best model (0 = best)",
        "- **Akaike Weight**: Probability this is the best model given the candidates",
        "- Models with ΔAIC < 1% of best AIC: substantially supported alternatives",
        "- Models with ΔAIC < 2: have meaningful support",
        "- Models with ΔAIC > 10: have essentially no support",
        "",
    ])
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate model comparison report")
    parser.add_argument(
        "--input",
        type=str,
        default="Outputs/model_fit_diagnostics.csv",
        help="Path to model_fit_diagnostics.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="Outputs/model_comparison_report.md",
        help="Path for output Markdown report",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default="Outputs/model_comparison_table.csv",
        help="Path for output CSV table",
    )
    parser.add_argument(
        "--context-filter",
        type=str,
        default=None,
        help="Filter to specific context (e.g., 'wild_TH' or 'overall')",
    )
    
    args = parser.parse_args()
    
    print(f"Loading diagnostics from: {args.input}")
    df = load_diagnostics(Path(args.input))
    
    print("Generating comparison tables...")
    comparison_df = generate_comparison_table(df, args.context_filter)
    summary_df = generate_summary_table(comparison_df)
    substantially_supported_df = generate_substantially_supported_table(comparison_df, tolerance_pct=1.0)
    freq_df = generate_model_frequency_table(comparison_df)
    
    # Save CSV
    print(f"Saving comparison table to: {args.csv_output}")
    comparison_df.to_csv(args.csv_output, index=False)
    
    # Generate and save Markdown report
    print(f"Saving Markdown report to: {args.output}")
    report = format_markdown_report(comparison_df, summary_df, substantially_supported_df, freq_df)
    Path(args.output).write_text(report)
    
    # Print summary to console
    print("\n" + "=" * 60)
    print("MODEL SELECTION SUMMARY")
    print("=" * 60)
    print("\nHow often each model was best:")
    print(freq_df.to_string(index=False))
    print("\nBest model by measurement/group:")
    print(summary_df.to_string(index=False))
    print("\nSubstantially supported models (within 1% of best AIC):")
    print(substantially_supported_df.to_string(index=False))
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
