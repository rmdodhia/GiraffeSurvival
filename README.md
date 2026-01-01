# Giraffe Growth Analysis (Modular)

This repo contains a modular package, `giraffesurvival`, for estimating giraffe growth curves from zoo and wild measurements. It includes the original population pipeline plus an individual-level track that fits per-animal Gompertz curves. Individual fits are run in two flavors: **y0-free** (birth height estimated) and **y0-fixed** (birth height constrained to 180 cm); reporting uses the y0-fixed results for biological plausibility.

## Study Questions This Analysis Addresses

- Infer ages of wild animals by leveraging zoo calves with known ages to map height-to-age, then back-estimate wild ages and growth trajectories.
- Quantify sex differences in growth curves across morphological measures (total height, neck length, foreleg length, ossicone length).
- Test for sex differences in indeterminate growth (continued growth after maturity).
- Assess whether calves with a visible umbilicus grow the same as other newborns, and whether wild calves with an umbilicus match zoo calves through ~48 months.
- Compare curve families (Gompertz, logistic, von Bertalanffy, Richards, poly3, poly4) to capture plausible growth shapes and select appropriate forms.

## What the analysis does

- Population pipeline: fits candidate curves (Gompertz, logistic, von Bertalanffy, Richards, poly3, poly4) to zoo data (known ages) and wild data (estimated ages); supports overall and sex-specific fits.
- Age assignment for wild: seeds ages from classes, then refines by aligning repeated TH measurements to population curves (mixed-effects style).
- Individual-level track: fits Gompertz per animal (≥4 obs), compares y0-free vs y0-fixed (180 cm), and aggregates fixed-y0 medians for population summaries (Option A). Option B (age refinement from individual curves) exists in code but is not used for reporting.
- Outputs plots and CSVs for both the population pipeline and the individual-level analysis.

## Data Required

Place the following files under the `Data/` folder:

- `wild.csv` – field data with repeated measurements per animal.
- `zoo.csv` – zoo measurements with known ages and sexes.

The default column names are wired into the loaders. If your files differ, update the loader functions in [giraffesurvival/data.py](giraffesurvival/data.py).

## Installing Prerequisites

The Python environment needs the usual scientific stack. Inside the project folder
(after activating the virtual environment), run:

```bash
pip install -r requirements.txt
```

This project uses `numpy`, `pandas`, `scipy`, and `matplotlib`.

## Running the Analysis

Population pipeline (wrapper):

```bash
python growth_analyses.py
```

Population pipeline (module entry point):

```bash
python -m giraffesurvival
```

Individual-level track (fits per-animal Gompertz, y0 fixed at 180 cm, regenerates comparison plots):

```bash
python growth_analyses_individual_level.py
```

Key outputs:

- `Graph/` – population growth curve PNGs (overall and by sex).
- `wild_with_age_estimates_sex_agnostic_then_by_sex.csv` – wild data with refined ages.
- `Outputs/individual_level_analysis/` – per-animal fits (y0-free and y0-fixed), aggregated Option A/B comparisons, and plots including `population_curve_comparison.png`.

### Example Variations

Run everything but skip the sex-specific fits:

Using flags with the wrapper or the package entry point:

```bash
# Skip sex-specific fits
python growth_analyses.py --skip-wild-by-sex --skip-zoo-by-sex

# Use historical age estimation
python -m giraffesurvival --age-strategy first_measurement

# Only run wild overall curves
python growth_analyses.py --skip-zoo-overall --skip-zoo-by-sex --skip-wild-by-sex
```

Programmatic control (importing the pipeline):

```bash
python -c "from giraffesurvival.pipeline import main; from giraffesurvival.fitting import AnalysisConfig; main(AnalysisConfig(fit_wild_by_sex=False, fit_zoo_by_sex=False))"
```

Use the historical age-estimation method while keeping the rest of the defaults:

```bash
python -c "from giraffesurvival.pipeline import main; from giraffesurvival.fitting import AnalysisConfig; main(AnalysisConfig(age_strategy='first_measurement'))"
```

Fit only the wild overall curves (helpful for a quick look) and leave out all zoo modelling:

```bash
python -c "from giraffesurvival.pipeline import main; from giraffesurvival.fitting import AnalysisConfig; main(AnalysisConfig(fit_zoo_overall=False, fit_zoo_by_sex=False, fit_wild_by_sex=False))"
```

## Tuning the Analysis

You can adjust behaviour through the `AnalysisConfig` data class in [giraffesurvival/fitting.py](giraffesurvival/fitting.py). Key options:

- `fit_zoo_overall`, `fit_zoo_by_sex`, `fit_wild_overall`, `fit_wild_by_sex`
	(True/False) to enable or disable specific model fits.
- `age_strategy`: set to `"mixed_effects"` (default) to align each animal’s
	longitudinal total-height series, or `"first_measurement"` to fall back to the
	original single-measurement approximation.

Pass an `AnalysisConfig` into `main()` to run only a subset of models.

## Interpreting the Outputs

- Console output lists the chosen growth curve for each measurement and the
	estimated parameters (with AIC for model comparison).
- Plots illustrate fitted curves and the observed data points; filenames encode
	the measurement and whether the curve is overall or sex-specific.
- The CSV output can be shared with collaborators who need the refined age
	estimates without rerunning the analysis.

## Where Things Live Now

- [giraffesurvival/models.py](giraffesurvival/models.py): curve definitions, candidate models, and model selection.
- [giraffesurvival/data.py](giraffesurvival/data.py): data loading/cleaning and age-class seeding.
- [giraffesurvival/age.py](giraffesurvival/age.py): age estimation from height and mixed-effects refinement.
- [giraffesurvival/fitting.py](giraffesurvival/fitting.py): measurement configuration and batch fitting utilities.
- [giraffesurvival/plotting.py](giraffesurvival/plotting.py): plotting helpers for overall/sex/group curves.
- [giraffesurvival/pipeline.py](giraffesurvival/pipeline.py): end-to-end `main()` driver.

Top-level scripts (utilities and reports):

- [growth_analyses.py](growth_analyses.py): population pipeline wrapper (overall/sex fits, age refinement, plots, outputs in Graph/ and CSVs).
- [growth_analyses_individual_level.py](growth_analyses_individual_level.py): individual-level Gompertz fits (y0-free and y0-fixed), aggregates the fixed-y0 medians for Option A population summaries, and writes plots/CSVs under Outputs/individual_level_analysis/ (Option B code present but not used in reporting).
- [plot_individual_growth_curves.py](plot_individual_growth_curves.py): helper to visualize per-animal fits and comparison grids.
- [combine_overall_plots.py](combine_overall_plots.py): stitches overall/sex plots into combined figures.
- [compare_zoo_umbilicus.py](compare_zoo_umbilicus.py): tests whether calves with visible umbilicus grow like other newborns and compares wild vs zoo early-life growth.
- [check_indeterminate_growth.py](check_indeterminate_growth.py): examines sex differences in indeterminate growth.
- [report_growth_summary.py](report_growth_summary.py): generates growth summary report artifacts from outputs.
- [report_model_comparison.py](report_model_comparison.py): summarizes model comparison diagnostics and tables.

If you want to tweak data cleaning, model choices, or plotting, edit the respective module. For scientific changes (e.g., parameter bounds), start in [giraffesurvival/models.py](giraffesurvival/models.py).
