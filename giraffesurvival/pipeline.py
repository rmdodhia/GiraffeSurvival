from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np

from .data import (
    WILD_PATH,
    ZOO_PATH,
    load_prepare_zoo,
    load_prepare_wild,
    add_age_class_midpoints,
    assign_initial_ages_from_classes,
    add_vtb_umb_flag,
)
from .models import JUVENILE_MODEL_SEQUENCE
from .fitting import (
    AnalysisConfig,
    MeasurementConfig,
    MEASUREMENTS,
    fit_growth_models,
    report_fit_results,
    get_measurements_for_config,
)
from .age import (
    refine_ages_with_zoo_models,
    refine_ages_with_individual_alignment,
    refine_ages_with_individual_alignment_multimeasure,
)
from .plotting import (
    plot_growth_curve_overall,
    plot_growth_curve_by_sex,
    plot_growth_curve_by_group,
    plot_individual_trajectories,
)


def main(config: AnalysisConfig | None = None) -> None:
    config = config or AnalysisConfig()
    linux_root = Path(".")
    default_graphs = linux_root / "Graph"
    graphs_dir = Path(config.graphs_dir) if config.graphs_dir else default_graphs
    graphs_dir.mkdir(parents=True, exist_ok=True)

    # Get measurements with correct TH model sequence based on birth_height_mode
    measurements = get_measurements_for_config(config)
    print(f"Birth height mode: {config.birth_height_mode}")

    # Zoo juvenile modeling
    print("Loading zoo data...")
    zoo = load_prepare_zoo(ZOO_PATH)
    fit_diagnostics: list[dict[str, Any]] = []

    def record_diagnostics(
        context: str,
        measurement: str,
        best_fits: dict[str, Any],
        candidates_by_group: dict[str, list[Any]],
        descriptor: str | None = None,
    ) -> None:
        def _format_params(candidate: Any) -> str:
            # Include 95% CIs when scipy curve_fit covariance is available.
            pairs = getattr(candidate, "param_pairs", [])
            cov = getattr(candidate, "covariance", None)
            if not getattr(candidate, "success", False) or cov is None:
                return "; ".join(f"{name}={value:.4g}" for name, value in pairs)

            try:
                cov = np.asarray(cov, dtype=float)
                diag = np.diag(cov)
                if len(diag) != len(pairs) or not np.all(np.isfinite(diag)):
                    raise ValueError("Invalid covariance")
                diag = np.maximum(diag, 0.0)
                se = np.sqrt(diag)
                z = 1.96
                parts: list[str] = []
                for (name, value), s in zip(pairs, se, strict=False):
                    if not np.isfinite(value) or not np.isfinite(s):
                        parts.append(f"{name}={value:.4g}")
                        continue
                    lo = value - z * s
                    hi = value + z * s
                    parts.append(f"{name}={value:.4g} [{lo:.4g}, {hi:.4g}]")
                return "; ".join(parts)
            except Exception:
                return "; ".join(f"{name}={value:.4g}" for name, value in pairs)

        for group_key, candidate_list in candidates_by_group.items():
            best_for_group = best_fits.get(group_key)
            for candidate in candidate_list:
                param_summary = _format_params(candidate)
                fit_diagnostics.append(
                    {
                        "context": context,
                        "measurement": measurement,
                        "descriptor": descriptor,
                        "group": group_key,
                        "model": candidate.model.name,
                        "success": candidate.success,
                        "aic": candidate.aic,
                        "sse": candidate.sse,
                        "n_obs": candidate.n_obs,
                        "message": candidate.message,
                        "is_selected": candidate is best_for_group,
                        "params": param_summary,
                    }
                )

    zoo_overall_fits: dict[str, Any] = {}
    if config.fit_zoo_overall:
        print("Fitting juvenile growth models to zoo calves (overall)...")
        zoo_overall_fits, zoo_overall_candidates = fit_growth_models(
            zoo,
            "height_cm",
            group_col=None,
            min_points=20,
            model_names=JUVENILE_MODEL_SEQUENCE,
        )
        record_diagnostics(
            "zoo_height_cm_overall",
            "height_cm",
            zoo_overall_fits,
            zoo_overall_candidates,
            descriptor="Zoo juvenile height",
        )
        report_fit_results("Zoo juvenile model (overall):", zoo_overall_fits, "Overall")
    else:
        print("Skipping juvenile overall models (disabled).")

    zoo_by_sex_fits: dict[str, Any] = {}
    if config.fit_zoo_by_sex:
        print("\nFitting juvenile growth models by sex (zoo)...")
        zoo_by_sex_fits, zoo_by_sex_candidates = fit_growth_models(
            zoo,
            "height_cm",
            group_col="Sex",
            min_points=15,
            model_names=JUVENILE_MODEL_SEQUENCE,
        )
        record_diagnostics(
            "zoo_height_cm_by_sex",
            "height_cm",
            zoo_by_sex_fits,
            zoo_by_sex_candidates,
            descriptor="Zoo juvenile height",
        )
        report_fit_results("Zoo juvenile models by sex:", zoo_by_sex_fits, "Sex {key}")
    else:
        print("\nSkipping juvenile sex-specific models (disabled).")

    # Wild data preparation
    print("\nLoading and preparing wild data...")
    wild = load_prepare_wild(WILD_PATH)
    wild = add_age_class_midpoints(wild)
    wild = assign_initial_ages_from_classes(wild)
    wild = add_vtb_umb_flag(wild)

    print("Refining ages for wild animals using available juvenile models...")
    juvenile_fit_overall = zoo_overall_fits.get("overall") if zoo_overall_fits else None
    juvenile_fits_by_sex = zoo_by_sex_fits if zoo_by_sex_fits else {}

    has_successful_overall = juvenile_fit_overall is not None and juvenile_fit_overall.success
    has_successful_sex = any(fit.success for fit in juvenile_fits_by_sex.values())

    if has_successful_overall or has_successful_sex:
        wild = refine_ages_with_zoo_models(wild, juvenile_fit_overall, juvenile_fits_by_sex)
    else:
        print("WARNING: juvenile fits unavailable; skipping age refinement.")

    if config.age_strategy == "mixed_effects":
        print("Applying individual alignment using longitudinal multi-measurements (neck+foreleg)...")

        # Seed population curves for alignment using current (zoo-refined / class-seeded) ages.
        fits_overall_by_measure: dict[str, Any] = {}
        fits_by_sex_by_measure: dict[str, dict[str, Any]] = {}

        for measurement in measurements:
            overall_fits, overall_candidates = fit_growth_models(
                wild,
                measurement.column,
                group_col=None,
                min_points=measurement.overall_min_points,
                model_names=measurement.model_names,
            )
            record_diagnostics(
                f"wild_alignment_seed_{measurement.column}_overall",
                measurement.column,
                overall_fits,
                overall_candidates,
                descriptor=measurement.descriptor,
            )
            fits_overall_by_measure[measurement.column] = overall_fits.get("overall")

            by_sex_fits, by_sex_candidates = fit_growth_models(
                wild,
                measurement.column,
                group_col="Sex",
                min_points=measurement.by_sex_min_points,
                model_names=measurement.model_names,
            )
            record_diagnostics(
                f"wild_alignment_seed_{measurement.column}_by_sex",
                measurement.column,
                by_sex_fits,
                by_sex_candidates,
                descriptor=measurement.descriptor,
            )
            fits_by_sex_by_measure[measurement.column] = by_sex_fits

        # Align using neck + foreleg only (exclude ossicone; TH often missing).
        align_columns = ["avg TOH_NIcm", "avg NI_FBHcm"]
        wild = refine_ages_with_individual_alignment_multimeasure(
            wild,
            measurements=align_columns,
            fits_overall_by_measure=fits_overall_by_measure,
            fits_by_sex_by_measure=fits_by_sex_by_measure,
        )
    elif config.age_strategy == "mixed_effects_th_only":
        th_config = next((m for m in MEASUREMENTS if m.column == "TH"), None)
        if th_config is None:
            print("Skipping TH-only mixed-effects refinement: TH measurement configuration missing.")
        else:
            print("Applying individual alignment using longitudinal TH measurements (legacy mode)...")
            th_overall_fits, th_overall_candidates = fit_growth_models(
                wild,
                th_config.column,
                group_col=None,
                min_points=th_config.overall_min_points,
                model_names=th_config.model_names,
            )
            record_diagnostics(
                "wild_th_alignment_overall",
                th_config.column,
                th_overall_fits,
                th_overall_candidates,
                descriptor=th_config.descriptor,
            )

            th_by_sex_fits, th_by_sex_candidates = fit_growth_models(
                wild,
                th_config.column,
                group_col="Sex",
                min_points=th_config.by_sex_min_points,
                model_names=th_config.model_names,
            )
            record_diagnostics(
                "wild_th_alignment_by_sex",
                th_config.column,
                th_by_sex_fits,
                th_by_sex_candidates,
                descriptor=th_config.descriptor,
            )

            wild = refine_ages_with_individual_alignment(
                wild,
                th_config.column,
                th_overall_fits.get("overall"),
                th_by_sex_fits,
            )
    elif config.age_strategy != "first_measurement":
        print(f"Unknown age strategy '{config.age_strategy}', falling back to first-measurement method.")

    # Fit and plot wild growth curves (overall)
    if config.fit_wild_overall:
        print("\nFitting wild growth curves (overall, no sex)...")
        for measurement in measurements:
            overall_fits, overall_candidates = fit_growth_models(
                wild,
                measurement.column,
                group_col=None,
                min_points=measurement.overall_min_points,
                model_names=measurement.model_names,
            )
            record_diagnostics(
                f"wild_{measurement.column}_overall",
                measurement.column,
                overall_fits,
                overall_candidates,
                descriptor=measurement.descriptor,
            )
            report_fit_results(
                f"Wild growth models for {measurement.descriptor} (overall):",
                overall_fits,
                "Overall",
            )
            plot_growth_curve_overall(
                wild,
                overall_fits,
                measurement.column,
                f"Wild {measurement.descriptor} - overall",
                df_label=measurement.label,
                graph_dir=graphs_dir,
            )
    else:
        print("\nSkipping wild overall models (disabled).")

    # Fit and plot wild growth curves by sex
    if config.fit_wild_by_sex:
        print("\nFitting wild growth curves by sex (subset with known sex)...")
        for measurement in measurements:
            fits_by_sex, candidates_by_sex = fit_growth_models(
                wild,
                measurement.column,
                group_col="Sex",
                min_points=measurement.by_sex_min_points,
                model_names=measurement.model_names,
            )
            record_diagnostics(
                f"wild_{measurement.column}_by_sex",
                measurement.column,
                fits_by_sex,
                candidates_by_sex,
                descriptor=measurement.descriptor,
            )
            report_fit_results(
                f"Wild growth models for {measurement.descriptor} by sex:",
                fits_by_sex,
                "Sex {key}",
            )
            if fits_by_sex:
                plot_growth_curve_by_sex(
                    wild,
                    fits_by_sex,
                    measurement.column,
                    f"Wild {measurement.descriptor} - by sex",
                    df_label=measurement.label,
                    graph_dir=graphs_dir,
                )

            unknown_mask = wild["Sex"].isna()
            if unknown_mask.any():
                unknown_subset = wild[unknown_mask]
                unknown_fits, unknown_candidates = fit_growth_models(
                    unknown_subset,
                    measurement.column,
                    group_col=None,
                    min_points=max(1, measurement.overall_min_points // 3),
                    model_names=measurement.model_names,
                )
                if unknown_fits:
                    record_diagnostics(
                        f"wild_{measurement.column}_sex_unknown",
                        measurement.column,
                        unknown_fits,
                        unknown_candidates,
                        descriptor=f"{measurement.descriptor} (sex unknown)",
                    )
                    report_fit_results(
                        f"Wild growth models for {measurement.descriptor} (sex unknown):",
                        unknown_fits,
                        "Overall",
                    )
                    plot_growth_curve_overall(
                        unknown_subset,
                        unknown_fits,
                        measurement.column,
                        f"Wild {measurement.descriptor} - sex unknown",
                        df_label=f"{measurement.label}_unknown",
                        graph_dir=graphs_dir,
                    )

    else:
        print("\nSkipping wild sex-specific models (disabled).")

    # Additional analysis: curves by VTB_Umb presence (AID-level flag) for all measures.
    # Use Gompertz only to keep comparisons consistent.
    print("\nFitting wild growth curves by VTB_Umb presence (Gompertz-only) for all measures...")
    for measurement in measurements:
        print(f"  - {measurement.descriptor}...")
        by_umb_fits, by_umb_candidates = fit_growth_models(
            wild,
            measurement.column,
            group_col="VTB_Umb_Flag",
            min_points=5,
            model_names=("gompertz",),
        )
        context = f"wild_{measurement.column}_by_vtb_umb"
        record_diagnostics(
            context,
            measurement.column,
            by_umb_fits,
            by_umb_candidates,
            descriptor=f"{measurement.descriptor} (by VTB_Umb)",
        )
        report_fit_results(
            f"Wild {measurement.descriptor} models by VTB_Umb (Gompertz-only):",
            by_umb_fits,
            "VTB {key}",
        )
        if by_umb_fits:
            plot_growth_curve_by_group(
                wild,
                by_umb_fits,
                group_col="VTB_Umb_Flag",
                value_col=measurement.column,
                title=f"Wild {measurement.descriptor} - by VTB_Umb",
                df_label=f"{measurement.label}_vtb_umb",
                graph_dir=graphs_dir,
            )

    default_outputs = linux_root / "Outputs"
    outputs_dir = Path(config.outputs_dir) if config.outputs_dir else default_outputs
    outputs_dir.mkdir(parents=True, exist_ok=True)
    out_path = outputs_dir / "wild_with_age_estimates_sex_agnostic_then_by_sex.csv"
    try:
        wild.to_csv(out_path, index=False)
        print(f"\nSaved wild dataset with age estimates to: {out_path.resolve()}")
    except PermissionError:
        from datetime import datetime
        fallback = outputs_dir / f"wild_with_age_estimates_sex_agnostic_then_by_sex_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        wild.to_csv(fallback, index=False)
        print(f"\nWARNING: Permission denied writing {out_path.resolve()}; saved to: {fallback.resolve()}")

    diagnostics_path = outputs_dir / "model_fit_diagnostics.csv"
    if fit_diagnostics:
        diagnostics_df = pd.DataFrame(fit_diagnostics)
        diagnostics_df.sort_values(by=["context", "measurement", "group", "model"], inplace=True)
        diagnostics_df.to_csv(diagnostics_path, index=False)
        print(f"Saved detailed model diagnostics to: {diagnostics_path.resolve()}")
    else:
        print("No model diagnostics recorded.")
