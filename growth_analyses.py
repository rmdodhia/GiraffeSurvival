"""
Giraffe growth analysis — wrapper for the modular pipeline.

How to run:
  - Wrapper script (recommended):
      python growth_analyses.py [options]
  - Package entry point (same options):
      python -m giraffesurvival [options]
  - Programmatic:
      from giraffesurvival.pipeline import main
      from giraffesurvival.fitting import AnalysisConfig
      main(AnalysisConfig(...))

Command-line options:
  --age-strategy {mixed_effects,first_measurement}
      Select wild age estimation method (default: mixed_effects).
      - mixed_effects: align each animal's TH series to the population curve
      - first_measurement: estimate age at first sighting from height only

  --skip-zoo-overall
      Skip juvenile height fits on zoo data (overall pooled).

  --skip-zoo-by-sex
      Skip juvenile height fits on zoo data by sex.

  --skip-wild-overall
      Skip wild growth fits (overall, sex-agnostic).

  --skip-wild-by-sex
      Skip wild growth fits by sex.

  --outputs-dir PATH
      Directory for CSV outputs (default: ~/GiraffeSurvival/Outputs). Use a
      Linux path to avoid OneDrive locks.

  --graphs-dir PATH
      Directory for plot images (default: ~/GiraffeSurvival/Graph). Use a
      Linux path to avoid OneDrive locks when saving many files.

  --birth-height {none,bounded,fixed}
      Control birth height constraint for TH models (default: fixed).
      - none: unconstrained models (original behavior)
      - bounded: birth height can vary within 145-215 cm (±2 SD)
      - fixed: birth height locked at exactly 180 cm

Examples:
  - Run everything with defaults:
      python growth_analyses.py

  - Quick run without sex-specific fits (zoo and wild):
      python growth_analyses.py --skip-wild-by-sex --skip-zoo-by-sex

  - Use historical age estimation method:
      python growth_analyses.py --age-strategy first_measurement
"""

import argparse
from giraffesurvival.pipeline import main
from giraffesurvival.fitting import AnalysisConfig


def cli() -> None:
    parser = argparse.ArgumentParser(description="Run giraffe growth analysis (wrapper for modular pipeline)")
    parser.add_argument(
        "--age-strategy",
        choices=["mixed_effects", "first_measurement"],
        default="mixed_effects",
        help="How to estimate ages for wild animals",
    )
    parser.add_argument("--skip-zoo-overall", action="store_true", help="Skip zoo overall juvenile fits")
    parser.add_argument("--skip-zoo-by-sex", action="store_true", help="Skip zoo sex-specific juvenile fits")
    parser.add_argument("--skip-wild-overall", action="store_true", help="Skip wild overall fits")
    parser.add_argument("--skip-wild-by-sex", action="store_true", help="Skip wild sex-specific fits")
    parser.add_argument("--outputs-dir", type=str, default=None, help="Directory for CSV outputs (default: Outputs)")
    parser.add_argument("--graphs-dir", type=str, default=None, help="Directory for plots (default: Graph)")
    parser.add_argument(
        "--birth-height",
        choices=["none", "bounded", "fixed"],
        default="fixed",
        help="Birth height constraint for TH: none (unconstrained), bounded (145-215cm), fixed (180cm)",
    )

    args = parser.parse_args()

    config = AnalysisConfig(
        fit_zoo_overall=not args.skip_zoo_overall,
        fit_zoo_by_sex=not args.skip_zoo_by_sex,
        fit_wild_overall=not args.skip_wild_overall,
        fit_wild_by_sex=not args.skip_wild_by_sex,
        age_strategy=args.age_strategy,
        birth_height_mode=args.birth_height,
        outputs_dir=args.outputs_dir,
        graphs_dir=args.graphs_dir,
    )
    main(config)


if __name__ == "__main__":
    cli()
