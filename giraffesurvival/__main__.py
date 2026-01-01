import argparse
from .pipeline import main
from .fitting import AnalysisConfig


def cli() -> None:
    parser = argparse.ArgumentParser(description="Run giraffe growth analysis (package entry point)")
    parser.add_argument(
        "--age-strategy",
        choices=["mixed_effects", "mixed_effects_th_only", "first_measurement"],
        default="mixed_effects",
        help="How to estimate ages for wild animals",
    )
    parser.add_argument("--skip-zoo-overall", action="store_true", help="Skip zoo overall juvenile fits")
    parser.add_argument("--skip-zoo-by-sex", action="store_true", help="Skip zoo sex-specific juvenile fits")
    parser.add_argument("--skip-wild-overall", action="store_true", help="Skip wild overall fits")
    parser.add_argument("--skip-wild-by-sex", action="store_true", help="Skip wild sex-specific fits")
    parser.add_argument("--outputs-dir", type=str, default=None, help="Directory for CSV outputs (default: ~/GiraffeSurvival/Outputs)")
    parser.add_argument("--graphs-dir", type=str, default=None, help="Directory for plots (default: ~/GiraffeSurvival/Graph)")

    args = parser.parse_args()

    config = AnalysisConfig(
        fit_zoo_overall=not args.skip_zoo_overall,
        fit_zoo_by_sex=not args.skip_zoo_by_sex,
        fit_wild_overall=not args.skip_wild_overall,
        fit_wild_by_sex=not args.skip_wild_by_sex,
        age_strategy=args.age_strategy,
        outputs_dir=args.outputs_dir,
        graphs_dir=args.graphs_dir,
    )
    main(config)


if __name__ == "__main__":
    cli()