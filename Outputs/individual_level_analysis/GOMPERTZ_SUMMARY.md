# Individual-Level Gompertz Growth Curve Fitting

## Summary

Successfully fitted Gompertz growth curves to 55 wild giraffes with ≥4 observations each. By enforcing consistent use of the Gompertz model across all animals (rather than allowing each animal to select its own best-fit model), we obtained biologically realistic aggregated population parameters.

## Key Results

### Aggregated Population Parameters (Median-Based, y0=180cm fixed)
- **A = 425.8 cm** - Asymptotic height
- **k = 0.0791** - Growth rate constant
- **t0 (derived) ≈ 0 months** - Birth-height constraint implies t0 near 0

### Sex-Specific Parameters (Medians, y0 fixed)
- **Males (n=10)**: A=511.3 cm, k=0.0638, t0≈0.05 months
- **Females (n=45)**: A=417.1 cm, k=0.0791, t0≈-1.1 months

### Why Gompertz-Only?

Previous analysis allowed each animal to select its own best-fit model from {Gompertz, poly3, poly4, logistic, von Bertalanffy, Richards}. This produced:
- 67% Gompertz fits
- 18% poly3 fits
- 9% poly4 fits
- 5.5% logistic fits

**Problem**: Averaging parameters across different model families (Gompertz vs polynomials) produced nonsensical results:
- A = 997,529 cm (unrealistic!)
- t0 = 3,488 months (291 years!)

**Solution**: Force consistent Gompertz model across all animals → parameters can be meaningfully aggregated.

### Robustness: Median vs Mean

Even with consistent Gompertz fitting, 2 outlier animals (with only 4 observations each) had extreme parameter estimates due to sparse data:
- Animal 293: A=9.4 million cm
- Animal 876: A=13.4 million cm, t0=32,087 months

**Solution**: Use **median** (not mean) for aggregation → robust to outliers.

## Files Generated

1. `individual_fits_y0free.csv` - Individual Gompertz parameters (55 animals)
2. `individual_fits_y0fixed.csv` - Individual fits with y0=180cm constraint
3. `individual_curves_min{3,4,5}_all.png` - All individual curves on one plot
4. `individual_curves_grid_min{3,4,5}.png` - 3-panel plots (Overall/M/F)
5. `population_curve_comparison.png` - BASELINE vs OPTION A comparison
6. `individual_level_analysis_comparison.png` - Model comparison metrics

## Interpretation

The aggregated Gompertz curve (OPTION A) with median parameters provides a reasonable representation of the population growth trajectory, derived from individual animal trajectories. This differs from BASELINE (pooling all observations independently) by explicitly accounting for individual-level variation and repeated measures structure.

Both fits are kept in the single `individual_fits_y0free.csv` table (columns `_free` and `_fixed`). The y0-fixed variant (180 cm) stabilizes extreme A/k estimates; mean AIC rises slightly vs. y0-free, but we’ll likely report the fixed-y0 parameters for biological plausibility while retaining the free-y0 fits for reference.

**Recommendation**: Use median-aggregated Gompertz parameters for reporting population growth patterns when individual-level fitting is desired. The median approach is robust to the few outlier fits that inevitably arise when fitting sparse individual data (4-6 observations per animal).
