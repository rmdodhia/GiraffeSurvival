# Zoo vs Umbilicus Wild Giraffe Growth Equivalence Test

## Summary

This analysis tests whether zoo giraffe growth curves can serve as a valid reference 
for estimating ages of wild giraffes by comparing zoo animals to umbilicus-identified 
wild animals (nearly known-age) during the first 24 months of life.

### Conclusion

**NOT EQUIVALENT** within ±20.0 cm margin

## Data Summary

| Group | N Animals | N Observations | Age Range (months) |
|-------|-----------|----------------|-------------------|
| Zoo | 12 | 68 | 0.0 - 23.8 |
| Umbilicus Wild | 35 | 81 | 1.0 - 23.8 |

## Equivalence Test Results

| Metric | Value |
|--------|-------|
| Observed max absolute deviation | 46.93 cm |
| Bootstrap mean max deviation | 126.85 cm |
| Bootstrap median max deviation | 69.66 cm |
| 95% CI lower bound | 23.55 cm |
| 95% CI upper bound | 566.24 cm |
| Equivalence margin | ±20.0 cm |
| Number of bootstrap samples | 100 |

## Interpretation

The equivalence test uses cluster bootstrap (resampling individuals, not measurements) 
to estimate the distribution of the maximum absolute deviation between the zoo and 
umbilicus wild growth curves across the 0-24 month age range.

**Equivalence criterion**: The 95% confidence interval for the maximum deviation must 
fall entirely within the pre-specified equivalence margin of ±20.0 cm.

- 95% CI: [23.55, 566.24] cm
- Margin: ±20.0 cm
- Result: **NOT EQUIVALENT**

The growth curves are NOT statistically equivalent within the specified margin. The differences between zoo and umbilicus wild animals may be too large to justify using zoo curves as a reference without adjustment.

## Methods

1. **Data**: Zoo height measurements and wild giraffe TH (total height) for 
   umbilicus-identified animals, restricted to ages 0-24 months.

2. **Curve fitting**: Gompertz growth model fitted to each group.

3. **Deviation**: Difference between umbilicus wild and zoo fitted curves computed 
   at 0.5-month intervals.

4. **Bootstrap**: Cluster bootstrap with 100 iterations, resampling 
   individuals (not measurements) to preserve within-animal correlation.

5. **Equivalence test**: 95% bootstrap CI for max absolute deviation compared to 
   pre-specified ±20.0 cm margin.

## Figure

See `zoo_umbilicus_comparison.png` for visual comparison.
