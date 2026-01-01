# Giraffe Growth Analysis (Modular)

This repository now provides a modular package, `giraffesurvival`, that estimates giraffe growth curves from zoo and wild measurements. The previous monolithic script has been split into clear modules so you can follow and tweak the analysis more easily.

## What the analysis does

- Imports zoo data (known ages) and tests several candidate growth curves
	(Gompertz, logistic, von Bertalanffy, Richards, and a cubic polynomial).
- Imports wild field data (unknown exact ages) and assigns ages in two stages:
	1. Uses zoo curves to back-calculate the age at first sighting.
	2. Refines each animal’s age trajectory by aligning all of its total-height
		 measurements to the best population curve (mixed-effects style).
- Fits and plots growth curves for each wild measurement, both overall and by sex
	(optional via configuration).
- Saves updated wild data with the new age estimates plus PNG plots for each curve.

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

From the project directory:

Option 1 — quick start (use the wrapper):

```bash
python growth_analyses.py
```

Option 2 — run the package entry point:

```bash
python -m giraffesurvival
```

The pipeline prints progress messages and writes outputs to:

- `Graph/` – PNG plots of growth curves (overall and by sex).
- `wild_with_age_estimates_sex_agnostic_then_by_sex.csv` – the wild dataset with
	refined ages.

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

If you want to tweak data cleaning, model choices, or plotting, edit the respective module. For scientific changes (e.g., parameter bounds), start in [giraffesurvival/models.py](giraffesurvival/models.py).
