# Growth Model Comparison Report

This report compares multiple growth models fitted to giraffe measurement data.
Model selection is based on AIC (Akaike Information Criterion), with lower values indicating better fit.

## Summary: Best Models and Substantially Supported Alternatives

The following table shows the best-fitting model for each measurement and group, plus any alternatives
within 1% of the best model's AIC (considered substantially supported).

| Context                                   | Measurement   | Group   | Model           |       AIC |        ΔAIC |   Akaike Weight | Status                 |
|:------------------------------------------|:--------------|:--------|:----------------|----------:|------------:|----------------:|:-----------------------|
| wild_TH_by_sex                            | TH            | F       | von_bertalanffy |  7506.36  |  0          |     0.40136     | BEST                   |
| wild_TH_by_sex                            | TH            | F       | gompertz        |  7506.77  |  0.414443   |     0.326241    | Supported (0.41 ΔAIC)  |
| wild_TH_by_sex                            | TH            | F       | logistic        |  7508.43  |  2.07488    |     0.142226    | Supported (2.07 ΔAIC)  |
| wild_TH_by_sex                            | TH            | F       | richards        |  7508.78  |  2.42809    |     0.119201    | Supported (2.43 ΔAIC)  |
| wild_TH_by_sex                            | TH            | F       | poly3           |  7513.55  |  7.19903    |     0.010972    | Supported (7.20 ΔAIC)  |
| wild_TH_by_sex                            | TH            | M       | poly3           |  4474.39  |  0          |     0.994101    | BEST                   |
| wild_TH_by_sex                            | TH            | M       | von_bertalanffy |  4485.74  | 11.3427     |     0.00342294  | Supported (11.34 ΔAIC) |
| wild_TH_by_sex                            | TH            | M       | gompertz        |  4487.16  | 12.767      |     0.00167924  | Supported (12.77 ΔAIC) |
| wild_TH_by_sex                            | TH            | M       | richards        |  4489.2   | 14.8101     |     0.0006046   | Supported (14.81 ΔAIC) |
| wild_TH_by_sex                            | TH            | M       | logistic        |  4491.49  | 17.0982     |     0.000192573 | Supported (17.10 ΔAIC) |
| wild_TH_by_vtb_umb                        | TH            | Umb=0   | gompertz        | 12480     |  0          |     1           | BEST                   |
| wild_TH_by_vtb_umb                        | TH            | Umb>0   | gompertz        |   255.953 |  0          |     1           | BEST                   |
| wild_TH_overall                           | TH            | overall | poly3           | 12719.8   |  0          |     1           | BEST                   |
| wild_TH_overall                           | TH            | overall | von_bertalanffy | 12759.9   | 40.1001     |     1.96057e-09 | Supported (40.10 ΔAIC) |
| wild_TH_overall                           | TH            | overall | gompertz        | 12763.4   | 43.5866     |     3.42998e-10 | Supported (43.59 ΔAIC) |
| wild_TH_overall                           | TH            | overall | richards        | 12765.5   | 45.6917     |     1.1972e-10  | Supported (45.69 ΔAIC) |
| wild_TH_overall                           | TH            | overall | logistic        | 12773.9   | 54.1479     |     1.74553e-12 | Supported (54.15 ΔAIC) |
| wild_TH_sex_unknown                       | TH            | overall | gompertz        |   497.951 |  0          |     0.339154    | BEST                   |
| wild_TH_sex_unknown                       | TH            | overall | logistic        |   497.968 |  0.0172126  |     0.336248    | Supported (0.02 ΔAIC)  |
| wild_TH_sex_unknown                       | TH            | overall | poly3           |   499.011 |  1.06004    |     0.199624    | Supported (1.06 ΔAIC)  |
| wild_TH_sex_unknown                       | TH            | overall | richards        |   499.951 |  2.00017    |     0.124758    | Supported (2.00 ΔAIC)  |
| wild_alignment_seed_TH_by_sex             | TH            | F       | logistic        |  7592.27  |  0          |     0.287983    | BEST                   |
| wild_alignment_seed_TH_by_sex             | TH            | F       | gompertz        |  7592.76  |  0.49223    |     0.225154    | Supported (0.49 ΔAIC)  |
| wild_alignment_seed_TH_by_sex             | TH            | F       | von_bertalanffy |  7593.07  |  0.795618   |     0.193464    | Supported (0.80 ΔAIC)  |
| wild_alignment_seed_TH_by_sex             | TH            | F       | poly3           |  7593.16  |  0.888925   |     0.184646    | Supported (0.89 ΔAIC)  |
| wild_alignment_seed_TH_by_sex             | TH            | F       | richards        |  7594.22  |  1.94763    |     0.108753    | Supported (1.95 ΔAIC)  |
| wild_alignment_seed_TH_by_sex             | TH            | M       | poly3           |  4547.51  |  0          |     0.969745    | BEST                   |
| wild_alignment_seed_TH_by_sex             | TH            | M       | von_bertalanffy |  4555.65  |  8.13912    |     0.016568    | Supported (8.14 ΔAIC)  |
| wild_alignment_seed_TH_by_sex             | TH            | M       | gompertz        |  4556.88  |  9.3658     |     0.00897222  | Supported (9.37 ΔAIC)  |
| wild_alignment_seed_TH_by_sex             | TH            | M       | richards        |  4558.91  | 11.4025     |     0.00324064  | Supported (11.40 ΔAIC) |
| wild_alignment_seed_TH_by_sex             | TH            | M       | logistic        |  4560.49  | 12.9776     |     0.00147434  | Supported (12.98 ΔAIC) |
| wild_alignment_seed_TH_overall            | TH            | overall | poly3           | 12889.8   |  0          |     1           | BEST                   |
| wild_alignment_seed_TH_overall            | TH            | overall | von_bertalanffy | 12930     | 40.2054     |     1.86001e-09 | Supported (40.21 ΔAIC) |
| wild_alignment_seed_TH_overall            | TH            | overall | gompertz        | 12932.3   | 42.4846     |     5.9509e-10  | Supported (42.48 ΔAIC) |
| wild_alignment_seed_TH_overall            | TH            | overall | richards        | 12934.3   | 44.5533     |     2.11531e-10 | Supported (44.55 ΔAIC) |
| wild_alignment_seed_TH_overall            | TH            | overall | logistic        | 12939.1   | 49.365      |     1.90778e-11 | Supported (49.36 ΔAIC) |
| wild_alignment_seed_avg NI_FBHcm_by_sex   | avg NI_FBHcm  | F       | poly3           |  6595.58  |  0          |     0.457772    | BEST                   |
| wild_alignment_seed_avg NI_FBHcm_by_sex   | avg NI_FBHcm  | F       | richards        |  6597.26  |  1.67911    |     0.197713    | Supported (1.68 ΔAIC)  |
| wild_alignment_seed_avg NI_FBHcm_by_sex   | avg NI_FBHcm  | F       | logistic        |  6597.63  |  2.04515    |     0.164646    | Supported (2.05 ΔAIC)  |
| wild_alignment_seed_avg NI_FBHcm_by_sex   | avg NI_FBHcm  | F       | gompertz        |  6598.65  |  3.06125    |     0.0990621   | Supported (3.06 ΔAIC)  |
| wild_alignment_seed_avg NI_FBHcm_by_sex   | avg NI_FBHcm  | F       | von_bertalanffy |  6599.05  |  3.4686     |     0.0808076   | Supported (3.47 ΔAIC)  |
| wild_alignment_seed_avg NI_FBHcm_by_sex   | avg NI_FBHcm  | M       | poly3           |  3924.15  |  0          |     0.967779    | BEST                   |
| wild_alignment_seed_avg NI_FBHcm_by_sex   | avg NI_FBHcm  | M       | von_bertalanffy |  3932.65  |  8.50173    |     0.0137927   | Supported (8.50 ΔAIC)  |
| wild_alignment_seed_avg NI_FBHcm_by_sex   | avg NI_FBHcm  | M       | gompertz        |  3933.24  |  9.08701    |     0.0102934   | Supported (9.09 ΔAIC)  |
| wild_alignment_seed_avg NI_FBHcm_by_sex   | avg NI_FBHcm  | M       | logistic        |  3934.95  | 10.7953     |     0.00438123  | Supported (10.80 ΔAIC) |
| wild_alignment_seed_avg NI_FBHcm_by_sex   | avg NI_FBHcm  | M       | richards        |  3935.26  | 11.1045     |     0.00375378  | Supported (11.10 ΔAIC) |
| wild_alignment_seed_avg NI_FBHcm_overall  | avg NI_FBHcm  | overall | poly3           | 11097.1   |  0          |     1           | BEST                   |
| wild_alignment_seed_avg NI_FBHcm_overall  | avg NI_FBHcm  | overall | von_bertalanffy | 11130.2   | 33.1623     |     6.29359e-08 | Supported (33.16 ΔAIC) |
| wild_alignment_seed_avg NI_FBHcm_overall  | avg NI_FBHcm  | overall | gompertz        | 11131.2   | 34.0997     |     3.93862e-08 | Supported (34.10 ΔAIC) |
| wild_alignment_seed_avg NI_FBHcm_overall  | avg NI_FBHcm  | overall | richards        | 11133.2   | 36.128      |     1.42856e-08 | Supported (36.13 ΔAIC) |
| wild_alignment_seed_avg NI_FBHcm_overall  | avg NI_FBHcm  | overall | logistic        | 11134     | 36.9658     |     9.39674e-09 | Supported (36.97 ΔAIC) |
| wild_alignment_seed_avg TOH_NIcm_by_sex   | avg TOH_NIcm  | F       | von_bertalanffy | 35037.7   |  0          |     0.788405    | BEST                   |
| wild_alignment_seed_avg TOH_NIcm_by_sex   | avg TOH_NIcm  | F       | gompertz        | 35041     |  3.24348    |     0.155753    | Supported (3.24 ΔAIC)  |
| wild_alignment_seed_avg TOH_NIcm_by_sex   | avg TOH_NIcm  | F       | richards        | 35043.1   |  5.34646    |     0.0544228   | Supported (5.35 ΔAIC)  |
| wild_alignment_seed_avg TOH_NIcm_by_sex   | avg TOH_NIcm  | F       | poly3           | 35051.2   | 13.4725     |     0.000935899 | Supported (13.47 ΔAIC) |
| wild_alignment_seed_avg TOH_NIcm_by_sex   | avg TOH_NIcm  | F       | logistic        | 35052.5   | 14.7961     |     0.000482851 | Supported (14.80 ΔAIC) |
| wild_alignment_seed_avg TOH_NIcm_by_sex   | avg TOH_NIcm  | M       | von_bertalanffy | 22423.1   |  0          |     0.618522    | BEST                   |
| wild_alignment_seed_avg TOH_NIcm_by_sex   | avg TOH_NIcm  | M       | poly3           | 22424.7   |  1.56438    |     0.282914    | Supported (1.56 ΔAIC)  |
| wild_alignment_seed_avg TOH_NIcm_by_sex   | avg TOH_NIcm  | M       | gompertz        | 22427.4   |  4.26559    |     0.0732982   | Supported (4.27 ΔAIC)  |
| wild_alignment_seed_avg TOH_NIcm_by_sex   | avg TOH_NIcm  | M       | richards        | 22429.5   |  6.39968    |     0.0252164   | Supported (6.40 ΔAIC)  |
| wild_alignment_seed_avg TOH_NIcm_by_sex   | avg TOH_NIcm  | M       | logistic        | 22442     | 18.8935     |     4.88298e-05 | Supported (18.89 ΔAIC) |
| wild_alignment_seed_avg TOH_NIcm_overall  | avg TOH_NIcm  | overall | poly3           | 60303     |  0          |     1           | BEST                   |
| wild_alignment_seed_avg TOH_NIcm_overall  | avg TOH_NIcm  | overall | von_bertalanffy | 60338.9   | 35.8559     |     1.63681e-08 | Supported (35.86 ΔAIC) |
| wild_alignment_seed_avg TOH_NIcm_overall  | avg TOH_NIcm  | overall | gompertz        | 60350.7   | 47.6529     |     4.49059e-11 | Supported (47.65 ΔAIC) |
| wild_alignment_seed_avg TOH_NIcm_overall  | avg TOH_NIcm  | overall | richards        | 60353     | 50.0145     |     1.37876e-11 | Supported (50.01 ΔAIC) |
| wild_alignment_seed_avg TOH_NIcm_overall  | avg TOH_NIcm  | overall | logistic        | 60388.2   | 85.1813     |     3.1851e-19  | Supported (85.18 ΔAIC) |
| wild_alignment_seed_avg TOO_TOHcm_by_sex  | avg TOO_TOHcm | F       | poly3           | 29709.8   |  0          |     0.841569    | BEST                   |
| wild_alignment_seed_avg TOO_TOHcm_by_sex  | avg TOO_TOHcm | F       | von_bertalanffy | 29713.8   |  4.0629     |     0.110368    | Supported (4.06 ΔAIC)  |
| wild_alignment_seed_avg TOO_TOHcm_by_sex  | avg TOO_TOHcm | F       | logistic        | 29717.2   |  7.44949    |     0.020298    | Supported (7.45 ΔAIC)  |
| wild_alignment_seed_avg TOO_TOHcm_by_sex  | avg TOO_TOHcm | F       | gompertz        | 29717.2   |  7.44949    |     0.020298    | Supported (7.45 ΔAIC)  |
| wild_alignment_seed_avg TOO_TOHcm_by_sex  | avg TOO_TOHcm | F       | richards        | 29719.2   |  9.44949    |     0.00746721  | Supported (9.45 ΔAIC)  |
| wild_alignment_seed_avg TOO_TOHcm_by_sex  | avg TOO_TOHcm | M       | richards        | 18535.9   |  0          |     0.29587     | BEST                   |
| wild_alignment_seed_avg TOO_TOHcm_by_sex  | avg TOO_TOHcm | M       | logistic        | 18536.1   |  0.22644    |     0.264198    | Supported (0.23 ΔAIC)  |
| wild_alignment_seed_avg TOO_TOHcm_by_sex  | avg TOO_TOHcm | M       | gompertz        | 18536.7   |  0.808377   |     0.197499    | Supported (0.81 ΔAIC)  |
| wild_alignment_seed_avg TOO_TOHcm_by_sex  | avg TOO_TOHcm | M       | von_bertalanffy | 18536.9   |  1.02639    |     0.177102    | Supported (1.03 ΔAIC)  |
| wild_alignment_seed_avg TOO_TOHcm_by_sex  | avg TOO_TOHcm | M       | poly3           | 18538.9   |  3.0209     |     0.0653312   | Supported (3.02 ΔAIC)  |
| wild_alignment_seed_avg TOO_TOHcm_overall | avg TOO_TOHcm | overall | poly3           | 49893.1   |  0          |     0.675863    | BEST                   |
| wild_alignment_seed_avg TOO_TOHcm_overall | avg TOO_TOHcm | overall | richards        | 49896.3   |  3.18811    |     0.137268    | Supported (3.19 ΔAIC)  |
| wild_alignment_seed_avg TOO_TOHcm_overall | avg TOO_TOHcm | overall | logistic        | 49897.7   |  4.62596    |     0.0668872   | Supported (4.63 ΔAIC)  |
| wild_alignment_seed_avg TOO_TOHcm_overall | avg TOO_TOHcm | overall | gompertz        | 49897.9   |  4.81191    |     0.0609489   | Supported (4.81 ΔAIC)  |
| wild_alignment_seed_avg TOO_TOHcm_overall | avg TOO_TOHcm | overall | von_bertalanffy | 49897.9   |  4.8758     |     0.0590327   | Supported (4.88 ΔAIC)  |
| wild_avg NI_FBHcm_by_sex                  | avg NI_FBHcm  | F       | von_bertalanffy |  6545.39  |  0          |     0.284932    | BEST                   |
| wild_avg NI_FBHcm_by_sex                  | avg NI_FBHcm  | F       | gompertz        |  6545.4   |  0.0109488  |     0.283376    | Supported (0.01 ΔAIC)  |
| wild_avg NI_FBHcm_by_sex                  | avg NI_FBHcm  | F       | logistic        |  6545.63  |  0.244916   |     0.252091    | Supported (0.24 ΔAIC)  |
| wild_avg NI_FBHcm_by_sex                  | avg NI_FBHcm  | F       | richards        |  6547.4   |  2.01184    |     0.104202    | Supported (2.01 ΔAIC)  |
| wild_avg NI_FBHcm_by_sex                  | avg NI_FBHcm  | F       | poly3           |  6548.04  |  2.65891    |     0.0753991   | Supported (2.66 ΔAIC)  |
| wild_avg NI_FBHcm_by_sex                  | avg NI_FBHcm  | M       | poly3           |  3869.57  |  0          |     0.996387    | BEST                   |
| wild_avg NI_FBHcm_by_sex                  | avg NI_FBHcm  | M       | von_bertalanffy |  3882.38  | 12.803      |     0.00165305  | Supported (12.80 ΔAIC) |
| wild_avg NI_FBHcm_by_sex                  | avg NI_FBHcm  | M       | gompertz        |  3883.1   | 13.5253     |     0.001152    | Supported (13.53 ΔAIC) |
| wild_avg NI_FBHcm_by_sex                  | avg NI_FBHcm  | M       | richards        |  3885.12  | 15.547      |     0.000419216 | Supported (15.55 ΔAIC) |
| wild_avg NI_FBHcm_by_sex                  | avg NI_FBHcm  | M       | logistic        |  3885.27  | 15.6962     |     0.000389087 | Supported (15.70 ΔAIC) |
| wild_avg NI_FBHcm_by_vtb_umb              | avg NI_FBHcm  | Umb=0   | gompertz        | 10779     |  0          |     1           | BEST                   |
| wild_avg NI_FBHcm_by_vtb_umb              | avg NI_FBHcm  | Umb>0   | gompertz        |   219.881 |  0          |     1           | BEST                   |
| wild_avg NI_FBHcm_overall                 | avg NI_FBHcm  | overall | poly3           | 10981.2   |  0          |     1           | BEST                   |
| wild_avg NI_FBHcm_overall                 | avg NI_FBHcm  | overall | von_bertalanffy | 11014.6   | 33.4056     |     5.57259e-08 | Supported (33.41 ΔAIC) |
| wild_avg NI_FBHcm_overall                 | avg NI_FBHcm  | overall | gompertz        | 11016.2   | 35.0277     |     2.47647e-08 | Supported (35.03 ΔAIC) |
| wild_avg NI_FBHcm_overall                 | avg NI_FBHcm  | overall | richards        | 11018.3   | 37.0767     |     8.89004e-09 | Supported (37.08 ΔAIC) |
| wild_avg NI_FBHcm_overall                 | avg NI_FBHcm  | overall | logistic        | 11021.2   | 39.9753     |     2.08674e-09 | Supported (39.98 ΔAIC) |
| wild_avg NI_FBHcm_sex_unknown             | avg NI_FBHcm  | overall | von_bertalanffy |   410.983 |  0          |     0.265465    | BEST                   |
| wild_avg NI_FBHcm_sex_unknown             | avg NI_FBHcm  | overall | gompertz        |   410.987 |  0.00384684 |     0.264955    | Supported (0.00 ΔAIC)  |
| wild_avg NI_FBHcm_sex_unknown             | avg NI_FBHcm  | overall | logistic        |   410.999 |  0.0158484  |     0.26337     | Supported (0.02 ΔAIC)  |
| wild_avg NI_FBHcm_sex_unknown             | avg NI_FBHcm  | overall | poly3           |   412.768 |  1.78499    |     0.108743    | Supported (1.78 ΔAIC)  |
| wild_avg NI_FBHcm_sex_unknown             | avg NI_FBHcm  | overall | richards        |   412.987 |  2.00396    |     0.0974659   | Supported (2.00 ΔAIC)  |
| wild_avg TOH_NIcm_by_sex                  | avg TOH_NIcm  | F       | von_bertalanffy | 34313.9   |  0          |     0.541654    | BEST                   |
| wild_avg TOH_NIcm_by_sex                  | avg TOH_NIcm  | F       | poly3           | 34314.8   |  0.930936   |     0.340072    | Supported (0.93 ΔAIC)  |
| wild_avg TOH_NIcm_by_sex                  | avg TOH_NIcm  | F       | gompertz        | 34317.5   |  3.64077    |     0.0877282   | Supported (3.64 ΔAIC)  |
| wild_avg TOH_NIcm_by_sex                  | avg TOH_NIcm  | F       | richards        | 34319.7   |  5.75766    |     0.0304411   | Supported (5.76 ΔAIC)  |
| wild_avg TOH_NIcm_by_sex                  | avg TOH_NIcm  | F       | logistic        | 34331     | 17.1049     |     0.000104578 | Supported (17.10 ΔAIC) |
| wild_avg TOH_NIcm_by_sex                  | avg TOH_NIcm  | M       | von_bertalanffy | 22130.1   |  0          |     0.604433    | BEST                   |
| wild_avg TOH_NIcm_by_sex                  | avg TOH_NIcm  | M       | poly3           | 22131.5   |  1.41286    |     0.298229    | Supported (1.41 ΔAIC)  |
| wild_avg TOH_NIcm_by_sex                  | avg TOH_NIcm  | M       | gompertz        | 22134.4   |  4.24404    |     0.0724048   | Supported (4.24 ΔAIC)  |
| wild_avg TOH_NIcm_by_sex                  | avg TOH_NIcm  | M       | richards        | 22136.5   |  6.37914    |     0.0248964   | Supported (6.38 ΔAIC)  |
| wild_avg TOH_NIcm_by_sex                  | avg TOH_NIcm  | M       | logistic        | 22149.5   | 19.3887     |     3.72516e-05 | Supported (19.39 ΔAIC) |
| wild_avg TOH_NIcm_by_vtb_umb              | avg TOH_NIcm  | Umb=0   | gompertz        | 58826.1   |  0          |     1           | BEST                   |
| wild_avg TOH_NIcm_by_vtb_umb              | avg TOH_NIcm  | Umb>0   | gompertz        |   539.97  |  0          |     1           | BEST                   |
| wild_avg TOH_NIcm_overall                 | avg TOH_NIcm  | overall | poly3           | 59357.9   |  0          |     1           | BEST                   |
| wild_avg TOH_NIcm_overall                 | avg TOH_NIcm  | overall | von_bertalanffy | 59398.4   | 40.4245     |     1.66699e-09 | Supported (40.42 ΔAIC) |
| wild_avg TOH_NIcm_overall                 | avg TOH_NIcm  | overall | gompertz        | 59412.2   | 54.3116     |     1.60834e-12 | Supported (54.31 ΔAIC) |
| wild_avg TOH_NIcm_overall                 | avg TOH_NIcm  | overall | richards        | 59414.7   | 56.7374     |     4.78229e-13 | Supported (56.74 ΔAIC) |
| wild_avg TOH_NIcm_overall                 | avg TOH_NIcm  | overall | logistic        | 59456.5   | 98.5842     |     3.91484e-22 | Supported (98.58 ΔAIC) |
| wild_avg TOH_NIcm_sex_unknown             | avg TOH_NIcm  | overall | von_bertalanffy |  1463.83  |  0          |     0.290991    | BEST                   |
| wild_avg TOH_NIcm_sex_unknown             | avg TOH_NIcm  | overall | gompertz        |  1463.97  |  0.136238   |     0.27183     | Supported (0.14 ΔAIC)  |
| wild_avg TOH_NIcm_sex_unknown             | avg TOH_NIcm  | overall | logistic        |  1464.37  |  0.537358   |     0.222431    | Supported (0.54 ΔAIC)  |
| wild_avg TOH_NIcm_sex_unknown             | avg TOH_NIcm  | overall | poly3           |  1465.69  |  1.85757    |     0.114951    | Supported (1.86 ΔAIC)  |
| wild_avg TOH_NIcm_sex_unknown             | avg TOH_NIcm  | overall | richards        |  1465.97  |  2.14031    |     0.099797    | Supported (2.14 ΔAIC)  |
| wild_avg TOO_TOHcm_by_sex                 | avg TOO_TOHcm | F       | von_bertalanffy | 29710.7   |  0          |     0.49131     | BEST                   |
| wild_avg TOO_TOHcm_by_sex                 | avg TOO_TOHcm | F       | poly3           | 29710.9   |  0.169346   |     0.451422    | Supported (0.17 ΔAIC)  |
| wild_avg TOO_TOHcm_by_sex                 | avg TOO_TOHcm | F       | logistic        | 29716.7   |  5.99474    |     0.0245253   | Supported (5.99 ΔAIC)  |
| wild_avg TOO_TOHcm_by_sex                 | avg TOO_TOHcm | F       | gompertz        | 29716.8   |  6.02787    |     0.0241224   | Supported (6.03 ΔAIC)  |
| wild_avg TOO_TOHcm_by_sex                 | avg TOO_TOHcm | F       | richards        | 29718.8   |  8.08578    |     0.00862087  | Supported (8.09 ΔAIC)  |
| wild_avg TOO_TOHcm_by_sex                 | avg TOO_TOHcm | M       | richards        | 18532.2   |  0          |     0.500005    | BEST                   |
| wild_avg TOO_TOHcm_by_sex                 | avg TOO_TOHcm | M       | poly3           | 18534.2   |  2.06273    |     0.178262    | Supported (2.06 ΔAIC)  |
| wild_avg TOO_TOHcm_by_sex                 | avg TOO_TOHcm | M       | logistic        | 18534.6   |  2.43944    |     0.147658    | Supported (2.44 ΔAIC)  |
| wild_avg TOO_TOHcm_by_sex                 | avg TOO_TOHcm | M       | gompertz        | 18535.5   |  3.33837    |     0.0942013   | Supported (3.34 ΔAIC)  |
| wild_avg TOO_TOHcm_by_sex                 | avg TOO_TOHcm | M       | von_bertalanffy | 18535.8   |  3.66833    |     0.0798741   | Supported (3.67 ΔAIC)  |
| wild_avg TOO_TOHcm_by_vtb_umb             | avg TOO_TOHcm | Umb=0   | gompertz        | 49308.5   |  0          |     1           | BEST                   |
| wild_avg TOO_TOHcm_by_vtb_umb             | avg TOO_TOHcm | Umb>0   | gompertz        |   559.127 |  0          |     1           | BEST                   |
| wild_avg TOO_TOHcm_overall                | avg TOO_TOHcm | overall | poly3           | 49887.4   |  0          |     0.400425    | BEST                   |
| wild_avg TOO_TOHcm_overall                | avg TOO_TOHcm | overall | richards        | 49887.6   |  0.261917   |     0.351274    | Supported (0.26 ΔAIC)  |
| wild_avg TOO_TOHcm_overall                | avg TOO_TOHcm | overall | logistic        | 49890.3   |  2.94035    |     0.0920518   | Supported (2.94 ΔAIC)  |
| wild_avg TOO_TOHcm_overall                | avg TOO_TOHcm | overall | gompertz        | 49890.6   |  3.2205     |     0.0800198   | Supported (3.22 ΔAIC)  |
| wild_avg TOO_TOHcm_overall                | avg TOO_TOHcm | overall | von_bertalanffy | 49890.7   |  3.31755    |     0.0762295   | Supported (3.32 ΔAIC)  |
| wild_avg TOO_TOHcm_sex_unknown            | avg TOO_TOHcm | overall | logistic        |  1409.55  |  0          |     0.258843    | BEST                   |
| wild_avg TOO_TOHcm_sex_unknown            | avg TOO_TOHcm | overall | gompertz        |  1409.57  |  0.0194927  |     0.256333    | Supported (0.02 ΔAIC)  |
| wild_avg TOO_TOHcm_sex_unknown            | avg TOO_TOHcm | overall | von_bertalanffy |  1409.57  |  0.0259979  |     0.2555      | Supported (0.03 ΔAIC)  |
| wild_avg TOO_TOHcm_sex_unknown            | avg TOO_TOHcm | overall | poly3           |  1411.16  |  1.6073     |     0.115882    | Supported (1.61 ΔAIC)  |
| wild_avg TOO_TOHcm_sex_unknown            | avg TOO_TOHcm | overall | richards        |  1411.2   |  1.64987    |     0.113442    | Supported (1.65 ΔAIC)  |
| zoo_height_cm_by_sex                      | height_cm     | F       | von_bertalanffy |   126.205 |  0          |     0.326585    | BEST                   |
| zoo_height_cm_by_sex                      | height_cm     | F       | gompertz        |   126.596 |  0.391354   |     0.268544    | Supported (0.39 ΔAIC)  |
| zoo_height_cm_by_sex                      | height_cm     | F       | poly3           |   127.391 |  1.18663    |     0.180436    | Supported (1.19 ΔAIC)  |
| zoo_height_cm_by_sex                      | height_cm     | M       | poly3           |   308.65  |  0          |     0.999693    | BEST                   |
| zoo_height_cm_overall                     | height_cm     | overall | von_bertalanffy |   472.104 |  0          |     0.633334    | BEST                   |
| zoo_height_cm_overall                     | height_cm     | overall | gompertz        |   474.553 |  2.4494     |     0.186103    | Supported (2.45 ΔAIC)  |
| zoo_height_cm_overall                     | height_cm     | overall | poly3           |   475.648 |  3.54412    |     0.107656    | Supported (3.54 ΔAIC)  |
| zoo_height_cm_overall                     | height_cm     | overall | richards        |   476.625 |  4.52102    |     0.0660551   | Supported (4.52 ΔAIC)  |

## Model Selection Summary

### How Often Each Model Was Selected as Best

| Model           |   Times Best |
|:----------------|-------------:|
| poly3           |           15 |
| von_bertalanffy |           11 |
| gompertz        |            9 |
| logistic        |            2 |
| richards        |            2 |

## Model Comparison by Model Form

The following sections organize model performance by model type (Gompertz, Logistic, Polynomial, etc.),
showing how each model performs across different measurements, sexes, datasets, and contexts.

### GOMPERTZ

**TH** (F) — wild_TH_by_sex (N=993)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      2 | gompertz | 7506.77 |   0.41 |          0.3262 |

**TH** (M) — wild_TH_by_sex (N=563)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      3 | gompertz | 4487.16 |  12.77 |          0.0017 |

**TH** (Umb=0) — wild_TH_by_vtb_umb (N=1590)

|   Rank | Model    |   AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|------:|-------:|----------------:|
|      1 | gompertz | 12480 |      0 |               1 |

**TH** (Umb>0) — wild_TH_by_vtb_umb (N=38)

|   Rank | Model    |    AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|-------:|-------:|----------------:|
|      1 | gompertz | 255.95 |      0 |               1 |

**TH** (overall) — wild_TH_overall (N=1628)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      3 | gompertz | 12763.4 |  43.59 |               0 |

**TH** (overall) — wild_TH_sex_unknown (N=72)

|   Rank | Model    |    AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|-------:|-------:|----------------:|
|      1 | gompertz | 497.95 |      0 |          0.3392 |

**TH** (F) — wild_alignment_seed_TH_by_sex (N=993)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      2 | gompertz | 7592.76 |   0.49 |          0.2252 |

**TH** (M) — wild_alignment_seed_TH_by_sex (N=563)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      3 | gompertz | 4556.88 |   9.37 |           0.009 |

**TH** (overall) — wild_alignment_seed_TH_overall (N=1628)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      3 | gompertz | 12932.3 |  42.48 |               0 |

**avg NI_FBHcm** (F) — wild_alignment_seed_avg NI_FBHcm_by_sex (N=997)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      4 | gompertz | 6598.65 |   3.06 |          0.0991 |

**avg NI_FBHcm** (M) — wild_alignment_seed_avg NI_FBHcm_by_sex (N=565)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      3 | gompertz | 3933.24 |   9.09 |          0.0103 |

**avg NI_FBHcm** (overall) — wild_alignment_seed_avg NI_FBHcm_overall (N=1634)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      3 | gompertz | 11131.1 |   34.1 |               0 |

**avg TOH_NIcm** (F) — wild_alignment_seed_avg TOH_NIcm_by_sex (N=5708)

|   Rank | Model    |   AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|------:|-------:|----------------:|
|      2 | gompertz | 35041 |   3.24 |          0.1558 |

**avg TOH_NIcm** (M) — wild_alignment_seed_avg TOH_NIcm_by_sex (N=3471)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      3 | gompertz | 22427.4 |   4.27 |          0.0733 |

**avg TOH_NIcm** (overall) — wild_alignment_seed_avg TOH_NIcm_overall (N=9428)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      3 | gompertz | 60350.7 |  47.65 |               0 |

**avg TOO_TOHcm** (F) — wild_alignment_seed_avg TOO_TOHcm_by_sex (N=5681)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      4 | gompertz | 29717.2 |   7.45 |          0.0203 |

**avg TOO_TOHcm** (M) — wild_alignment_seed_avg TOO_TOHcm_by_sex (N=3447)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      3 | gompertz | 18536.7 |   0.81 |          0.1975 |

**avg TOO_TOHcm** (overall) — wild_alignment_seed_avg TOO_TOHcm_overall (N=9376)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      4 | gompertz | 49897.9 |   4.81 |          0.0609 |

**avg NI_FBHcm** (F) — wild_avg NI_FBHcm_by_sex (N=997)

|   Rank | Model    |    AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|-------:|-------:|----------------:|
|      2 | gompertz | 6545.4 |   0.01 |          0.2834 |

**avg NI_FBHcm** (M) — wild_avg NI_FBHcm_by_sex (N=565)

|   Rank | Model    |    AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|-------:|-------:|----------------:|
|      3 | gompertz | 3883.1 |  13.53 |          0.0012 |

**avg NI_FBHcm** (Umb=0) — wild_avg NI_FBHcm_by_vtb_umb (N=1596)

|   Rank | Model    |   AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|------:|-------:|----------------:|
|      1 | gompertz | 10779 |      0 |               1 |

**avg NI_FBHcm** (Umb>0) — wild_avg NI_FBHcm_by_vtb_umb (N=38)

|   Rank | Model    |    AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|-------:|-------:|----------------:|
|      1 | gompertz | 219.88 |      0 |               1 |

**avg NI_FBHcm** (overall) — wild_avg NI_FBHcm_overall (N=1634)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      3 | gompertz | 11016.2 |  35.03 |               0 |

**avg NI_FBHcm** (overall) — wild_avg NI_FBHcm_sex_unknown (N=72)

|   Rank | Model    |    AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|-------:|-------:|----------------:|
|      2 | gompertz | 410.99 |      0 |           0.265 |

**avg TOH_NIcm** (F) — wild_avg TOH_NIcm_by_sex (N=5708)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      3 | gompertz | 34317.5 |   3.64 |          0.0877 |

**avg TOH_NIcm** (M) — wild_avg TOH_NIcm_by_sex (N=3471)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      3 | gompertz | 22134.3 |   4.24 |          0.0724 |

**avg TOH_NIcm** (Umb=0) — wild_avg TOH_NIcm_by_vtb_umb (N=9335)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      1 | gompertz | 58826.1 |      0 |               1 |

**avg TOH_NIcm** (Umb>0) — wild_avg TOH_NIcm_by_vtb_umb (N=93)

|   Rank | Model    |    AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|-------:|-------:|----------------:|
|      1 | gompertz | 539.97 |      0 |               1 |

**avg TOH_NIcm** (overall) — wild_avg TOH_NIcm_overall (N=9428)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      3 | gompertz | 59412.2 |  54.31 |               0 |

**avg TOH_NIcm** (overall) — wild_avg TOH_NIcm_sex_unknown (N=249)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      2 | gompertz | 1463.97 |   0.14 |          0.2718 |

**avg TOO_TOHcm** (F) — wild_avg TOO_TOHcm_by_sex (N=5681)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      4 | gompertz | 29716.8 |   6.03 |          0.0241 |

**avg TOO_TOHcm** (M) — wild_avg TOO_TOHcm_by_sex (N=3447)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      4 | gompertz | 18535.5 |   3.34 |          0.0942 |

**avg TOO_TOHcm** (Umb=0) — wild_avg TOO_TOHcm_by_vtb_umb (N=9283)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      1 | gompertz | 49308.5 |      0 |               1 |

**avg TOO_TOHcm** (Umb>0) — wild_avg TOO_TOHcm_by_vtb_umb (N=93)

|   Rank | Model    |    AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|-------:|-------:|----------------:|
|      1 | gompertz | 559.13 |      0 |               1 |

**avg TOO_TOHcm** (overall) — wild_avg TOO_TOHcm_overall (N=9376)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      4 | gompertz | 49890.6 |   3.22 |            0.08 |

**avg TOO_TOHcm** (overall) — wild_avg TOO_TOHcm_sex_unknown (N=248)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      2 | gompertz | 1409.57 |   0.02 |          0.2563 |

**height_cm** (F) — zoo_height_cm_by_sex (N=21)

|   Rank | Model    |   AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|------:|-------:|----------------:|
|      2 | gompertz | 126.6 |   0.39 |          0.2685 |

**height_cm** (M) — zoo_height_cm_by_sex (N=56)

|   Rank | Model    |    AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|-------:|-------:|----------------:|
|      3 | gompertz | 328.86 |  20.21 |               0 |

**height_cm** (overall) — zoo_height_cm_overall (N=77)

|   Rank | Model    |    AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|-------:|-------:|----------------:|
|      2 | gompertz | 474.55 |   2.45 |          0.1861 |

### LOGISTIC

**TH** (F) — wild_TH_by_sex (N=993)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      3 | logistic | 7508.43 |   2.07 |          0.1422 |

**TH** (M) — wild_TH_by_sex (N=563)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      5 | logistic | 4491.49 |   17.1 |          0.0002 |

**TH** (overall) — wild_TH_overall (N=1628)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      5 | logistic | 12773.9 |  54.15 |               0 |

**TH** (overall) — wild_TH_sex_unknown (N=72)

|   Rank | Model    |    AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|-------:|-------:|----------------:|
|      2 | logistic | 497.97 |   0.02 |          0.3362 |

**TH** (F) — wild_alignment_seed_TH_by_sex (N=993)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      1 | logistic | 7592.27 |      0 |           0.288 |

**TH** (M) — wild_alignment_seed_TH_by_sex (N=563)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      5 | logistic | 4560.49 |  12.98 |          0.0015 |

**TH** (overall) — wild_alignment_seed_TH_overall (N=1628)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      5 | logistic | 12939.1 |  49.36 |               0 |

**avg NI_FBHcm** (F) — wild_alignment_seed_avg NI_FBHcm_by_sex (N=997)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      3 | logistic | 6597.63 |   2.05 |          0.1646 |

**avg NI_FBHcm** (M) — wild_alignment_seed_avg NI_FBHcm_by_sex (N=565)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      4 | logistic | 3934.95 |   10.8 |          0.0044 |

**avg NI_FBHcm** (overall) — wild_alignment_seed_avg NI_FBHcm_overall (N=1634)

|   Rank | Model    |   AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|------:|-------:|----------------:|
|      5 | logistic | 11134 |  36.97 |               0 |

**avg TOH_NIcm** (F) — wild_alignment_seed_avg TOH_NIcm_by_sex (N=5708)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      5 | logistic | 35052.5 |   14.8 |          0.0005 |

**avg TOH_NIcm** (M) — wild_alignment_seed_avg TOH_NIcm_by_sex (N=3471)

|   Rank | Model    |   AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|------:|-------:|----------------:|
|      5 | logistic | 22442 |  18.89 |               0 |

**avg TOH_NIcm** (overall) — wild_alignment_seed_avg TOH_NIcm_overall (N=9428)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      5 | logistic | 60388.2 |  85.18 |               0 |

**avg TOO_TOHcm** (F) — wild_alignment_seed_avg TOO_TOHcm_by_sex (N=5681)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      3 | logistic | 29717.2 |   7.45 |          0.0203 |

**avg TOO_TOHcm** (M) — wild_alignment_seed_avg TOO_TOHcm_by_sex (N=3447)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      2 | logistic | 18536.1 |   0.23 |          0.2642 |

**avg TOO_TOHcm** (overall) — wild_alignment_seed_avg TOO_TOHcm_overall (N=9376)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      3 | logistic | 49897.7 |   4.63 |          0.0669 |

**avg NI_FBHcm** (F) — wild_avg NI_FBHcm_by_sex (N=997)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      3 | logistic | 6545.63 |   0.24 |          0.2521 |

**avg NI_FBHcm** (M) — wild_avg NI_FBHcm_by_sex (N=565)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      5 | logistic | 3885.27 |   15.7 |          0.0004 |

**avg NI_FBHcm** (overall) — wild_avg NI_FBHcm_overall (N=1634)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      5 | logistic | 11021.2 |  39.98 |               0 |

**avg NI_FBHcm** (overall) — wild_avg NI_FBHcm_sex_unknown (N=72)

|   Rank | Model    |   AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|------:|-------:|----------------:|
|      3 | logistic |   411 |   0.02 |          0.2634 |

**avg TOH_NIcm** (F) — wild_avg TOH_NIcm_by_sex (N=5708)

|   Rank | Model    |   AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|------:|-------:|----------------:|
|      5 | logistic | 34331 |   17.1 |          0.0001 |

**avg TOH_NIcm** (M) — wild_avg TOH_NIcm_by_sex (N=3471)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      5 | logistic | 22149.5 |  19.39 |               0 |

**avg TOH_NIcm** (overall) — wild_avg TOH_NIcm_overall (N=9428)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      5 | logistic | 59456.5 |  98.58 |               0 |

**avg TOH_NIcm** (overall) — wild_avg TOH_NIcm_sex_unknown (N=249)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      3 | logistic | 1464.37 |   0.54 |          0.2224 |

**avg TOO_TOHcm** (F) — wild_avg TOO_TOHcm_by_sex (N=5681)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      3 | logistic | 29716.7 |   5.99 |          0.0245 |

**avg TOO_TOHcm** (M) — wild_avg TOO_TOHcm_by_sex (N=3447)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      3 | logistic | 18534.6 |   2.44 |          0.1477 |

**avg TOO_TOHcm** (overall) — wild_avg TOO_TOHcm_overall (N=9376)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      3 | logistic | 49890.3 |   2.94 |          0.0921 |

**avg TOO_TOHcm** (overall) — wild_avg TOO_TOHcm_sex_unknown (N=248)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      1 | logistic | 1409.55 |      0 |          0.2588 |

**height_cm** (F) — zoo_height_cm_by_sex (N=21)

|   Rank | Model    |    AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|-------:|-------:|----------------:|
|      4 | logistic | 128.11 |    1.9 |          0.1263 |

**height_cm** (M) — zoo_height_cm_by_sex (N=56)

|   Rank | Model    |    AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|-------:|-------:|----------------:|
|      5 | logistic | 337.73 |  29.08 |               0 |

**height_cm** (overall) — zoo_height_cm_overall (N=77)

|   Rank | Model    |    AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|-------:|-------:|----------------:|
|      5 | logistic | 481.16 |   9.05 |          0.0069 |

### POLY3

**TH** (F) — wild_TH_by_sex (N=993)

|   Rank | Model   |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:--------|--------:|-------:|----------------:|
|      5 | poly3   | 7513.55 |    7.2 |           0.011 |

**TH** (M) — wild_TH_by_sex (N=563)

|   Rank | Model   |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:--------|--------:|-------:|----------------:|
|      1 | poly3   | 4474.39 |      0 |          0.9941 |

**TH** (overall) — wild_TH_overall (N=1628)

|   Rank | Model   |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:--------|--------:|-------:|----------------:|
|      1 | poly3   | 12719.8 |      0 |               1 |

**TH** (overall) — wild_TH_sex_unknown (N=72)

|   Rank | Model   |    AIC |   ΔAIC |   Akaike Weight |
|-------:|:--------|-------:|-------:|----------------:|
|      3 | poly3   | 499.01 |   1.06 |          0.1996 |

**TH** (F) — wild_alignment_seed_TH_by_sex (N=993)

|   Rank | Model   |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:--------|--------:|-------:|----------------:|
|      4 | poly3   | 7593.16 |   0.89 |          0.1846 |

**TH** (M) — wild_alignment_seed_TH_by_sex (N=563)

|   Rank | Model   |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:--------|--------:|-------:|----------------:|
|      1 | poly3   | 4547.51 |      0 |          0.9697 |

**TH** (overall) — wild_alignment_seed_TH_overall (N=1628)

|   Rank | Model   |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:--------|--------:|-------:|----------------:|
|      1 | poly3   | 12889.8 |      0 |               1 |

**avg NI_FBHcm** (F) — wild_alignment_seed_avg NI_FBHcm_by_sex (N=997)

|   Rank | Model   |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:--------|--------:|-------:|----------------:|
|      1 | poly3   | 6595.58 |      0 |          0.4578 |

**avg NI_FBHcm** (M) — wild_alignment_seed_avg NI_FBHcm_by_sex (N=565)

|   Rank | Model   |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:--------|--------:|-------:|----------------:|
|      1 | poly3   | 3924.15 |      0 |          0.9678 |

**avg NI_FBHcm** (overall) — wild_alignment_seed_avg NI_FBHcm_overall (N=1634)

|   Rank | Model   |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:--------|--------:|-------:|----------------:|
|      1 | poly3   | 11097.1 |      0 |               1 |

**avg TOH_NIcm** (F) — wild_alignment_seed_avg TOH_NIcm_by_sex (N=5708)

|   Rank | Model   |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:--------|--------:|-------:|----------------:|
|      4 | poly3   | 35051.2 |  13.47 |          0.0009 |

**avg TOH_NIcm** (M) — wild_alignment_seed_avg TOH_NIcm_by_sex (N=3471)

|   Rank | Model   |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:--------|--------:|-------:|----------------:|
|      2 | poly3   | 22424.7 |   1.56 |          0.2829 |

**avg TOH_NIcm** (overall) — wild_alignment_seed_avg TOH_NIcm_overall (N=9428)

|   Rank | Model   |   AIC |   ΔAIC |   Akaike Weight |
|-------:|:--------|------:|-------:|----------------:|
|      1 | poly3   | 60303 |      0 |               1 |

**avg TOO_TOHcm** (F) — wild_alignment_seed_avg TOO_TOHcm_by_sex (N=5681)

|   Rank | Model   |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:--------|--------:|-------:|----------------:|
|      1 | poly3   | 29709.8 |      0 |          0.8416 |

**avg TOO_TOHcm** (M) — wild_alignment_seed_avg TOO_TOHcm_by_sex (N=3447)

|   Rank | Model   |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:--------|--------:|-------:|----------------:|
|      5 | poly3   | 18538.9 |   3.02 |          0.0653 |

**avg TOO_TOHcm** (overall) — wild_alignment_seed_avg TOO_TOHcm_overall (N=9376)

|   Rank | Model   |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:--------|--------:|-------:|----------------:|
|      1 | poly3   | 49893.1 |      0 |          0.6759 |

**avg NI_FBHcm** (F) — wild_avg NI_FBHcm_by_sex (N=997)

|   Rank | Model   |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:--------|--------:|-------:|----------------:|
|      5 | poly3   | 6548.04 |   2.66 |          0.0754 |

**avg NI_FBHcm** (M) — wild_avg NI_FBHcm_by_sex (N=565)

|   Rank | Model   |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:--------|--------:|-------:|----------------:|
|      1 | poly3   | 3869.57 |      0 |          0.9964 |

**avg NI_FBHcm** (overall) — wild_avg NI_FBHcm_overall (N=1634)

|   Rank | Model   |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:--------|--------:|-------:|----------------:|
|      1 | poly3   | 10981.2 |      0 |               1 |

**avg NI_FBHcm** (overall) — wild_avg NI_FBHcm_sex_unknown (N=72)

|   Rank | Model   |    AIC |   ΔAIC |   Akaike Weight |
|-------:|:--------|-------:|-------:|----------------:|
|      4 | poly3   | 412.77 |   1.78 |          0.1087 |

**avg TOH_NIcm** (F) — wild_avg TOH_NIcm_by_sex (N=5708)

|   Rank | Model   |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:--------|--------:|-------:|----------------:|
|      2 | poly3   | 34314.8 |   0.93 |          0.3401 |

**avg TOH_NIcm** (M) — wild_avg TOH_NIcm_by_sex (N=3471)

|   Rank | Model   |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:--------|--------:|-------:|----------------:|
|      2 | poly3   | 22131.5 |   1.41 |          0.2982 |

**avg TOH_NIcm** (overall) — wild_avg TOH_NIcm_overall (N=9428)

|   Rank | Model   |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:--------|--------:|-------:|----------------:|
|      1 | poly3   | 59357.9 |      0 |               1 |

**avg TOH_NIcm** (overall) — wild_avg TOH_NIcm_sex_unknown (N=249)

|   Rank | Model   |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:--------|--------:|-------:|----------------:|
|      4 | poly3   | 1465.69 |   1.86 |           0.115 |

**avg TOO_TOHcm** (F) — wild_avg TOO_TOHcm_by_sex (N=5681)

|   Rank | Model   |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:--------|--------:|-------:|----------------:|
|      2 | poly3   | 29710.9 |   0.17 |          0.4514 |

**avg TOO_TOHcm** (M) — wild_avg TOO_TOHcm_by_sex (N=3447)

|   Rank | Model   |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:--------|--------:|-------:|----------------:|
|      2 | poly3   | 18534.2 |   2.06 |          0.1783 |

**avg TOO_TOHcm** (overall) — wild_avg TOO_TOHcm_overall (N=9376)

|   Rank | Model   |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:--------|--------:|-------:|----------------:|
|      1 | poly3   | 49887.4 |      0 |          0.4004 |

**avg TOO_TOHcm** (overall) — wild_avg TOO_TOHcm_sex_unknown (N=248)

|   Rank | Model   |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:--------|--------:|-------:|----------------:|
|      4 | poly3   | 1411.16 |   1.61 |          0.1159 |

**height_cm** (F) — zoo_height_cm_by_sex (N=21)

|   Rank | Model   |    AIC |   ΔAIC |   Akaike Weight |
|-------:|:--------|-------:|-------:|----------------:|
|      3 | poly3   | 127.39 |   1.19 |          0.1804 |

**height_cm** (M) — zoo_height_cm_by_sex (N=56)

|   Rank | Model   |    AIC |   ΔAIC |   Akaike Weight |
|-------:|:--------|-------:|-------:|----------------:|
|      1 | poly3   | 308.65 |      0 |          0.9997 |

**height_cm** (overall) — zoo_height_cm_overall (N=77)

|   Rank | Model   |    AIC |   ΔAIC |   Akaike Weight |
|-------:|:--------|-------:|-------:|----------------:|
|      3 | poly3   | 475.65 |   3.54 |          0.1077 |

### RICHARDS

**TH** (F) — wild_TH_by_sex (N=993)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      4 | richards | 7508.78 |   2.43 |          0.1192 |

**TH** (M) — wild_TH_by_sex (N=563)

|   Rank | Model    |    AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|-------:|-------:|----------------:|
|      4 | richards | 4489.2 |  14.81 |          0.0006 |

**TH** (overall) — wild_TH_overall (N=1628)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      4 | richards | 12765.5 |  45.69 |               0 |

**TH** (overall) — wild_TH_sex_unknown (N=72)

|   Rank | Model    |    AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|-------:|-------:|----------------:|
|      4 | richards | 499.95 |      2 |          0.1248 |

**TH** (F) — wild_alignment_seed_TH_by_sex (N=993)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      5 | richards | 7594.22 |   1.95 |          0.1088 |

**TH** (M) — wild_alignment_seed_TH_by_sex (N=563)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      4 | richards | 4558.91 |   11.4 |          0.0032 |

**TH** (overall) — wild_alignment_seed_TH_overall (N=1628)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      4 | richards | 12934.3 |  44.55 |               0 |

**avg NI_FBHcm** (F) — wild_alignment_seed_avg NI_FBHcm_by_sex (N=997)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      2 | richards | 6597.26 |   1.68 |          0.1977 |

**avg NI_FBHcm** (M) — wild_alignment_seed_avg NI_FBHcm_by_sex (N=565)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      5 | richards | 3935.26 |   11.1 |          0.0038 |

**avg NI_FBHcm** (overall) — wild_alignment_seed_avg NI_FBHcm_overall (N=1634)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      4 | richards | 11133.2 |  36.13 |               0 |

**avg TOH_NIcm** (F) — wild_alignment_seed_avg TOH_NIcm_by_sex (N=5708)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      3 | richards | 35043.1 |   5.35 |          0.0544 |

**avg TOH_NIcm** (M) — wild_alignment_seed_avg TOH_NIcm_by_sex (N=3471)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      4 | richards | 22429.5 |    6.4 |          0.0252 |

**avg TOH_NIcm** (overall) — wild_alignment_seed_avg TOH_NIcm_overall (N=9428)

|   Rank | Model    |   AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|------:|-------:|----------------:|
|      4 | richards | 60353 |  50.01 |               0 |

**avg TOO_TOHcm** (F) — wild_alignment_seed_avg TOO_TOHcm_by_sex (N=5681)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      5 | richards | 29719.2 |   9.45 |          0.0075 |

**avg TOO_TOHcm** (M) — wild_alignment_seed_avg TOO_TOHcm_by_sex (N=3447)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      1 | richards | 18535.9 |      0 |          0.2959 |

**avg TOO_TOHcm** (overall) — wild_alignment_seed_avg TOO_TOHcm_overall (N=9376)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      2 | richards | 49896.2 |   3.19 |          0.1373 |

**avg NI_FBHcm** (F) — wild_avg NI_FBHcm_by_sex (N=997)

|   Rank | Model    |    AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|-------:|-------:|----------------:|
|      4 | richards | 6547.4 |   2.01 |          0.1042 |

**avg NI_FBHcm** (M) — wild_avg NI_FBHcm_by_sex (N=565)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      4 | richards | 3885.12 |  15.55 |          0.0004 |

**avg NI_FBHcm** (overall) — wild_avg NI_FBHcm_overall (N=1634)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      4 | richards | 11018.3 |  37.08 |               0 |

**avg NI_FBHcm** (overall) — wild_avg NI_FBHcm_sex_unknown (N=72)

|   Rank | Model    |    AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|-------:|-------:|----------------:|
|      5 | richards | 412.99 |      2 |          0.0975 |

**avg TOH_NIcm** (F) — wild_avg TOH_NIcm_by_sex (N=5708)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      4 | richards | 34319.7 |   5.76 |          0.0304 |

**avg TOH_NIcm** (M) — wild_avg TOH_NIcm_by_sex (N=3471)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      4 | richards | 22136.5 |   6.38 |          0.0249 |

**avg TOH_NIcm** (overall) — wild_avg TOH_NIcm_overall (N=9428)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      4 | richards | 59414.7 |  56.74 |               0 |

**avg TOH_NIcm** (overall) — wild_avg TOH_NIcm_sex_unknown (N=249)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      5 | richards | 1465.97 |   2.14 |          0.0998 |

**avg TOO_TOHcm** (F) — wild_avg TOO_TOHcm_by_sex (N=5681)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      5 | richards | 29718.8 |   8.09 |          0.0086 |

**avg TOO_TOHcm** (M) — wild_avg TOO_TOHcm_by_sex (N=3447)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      1 | richards | 18532.2 |      0 |             0.5 |

**avg TOO_TOHcm** (overall) — wild_avg TOO_TOHcm_overall (N=9376)

|   Rank | Model    |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|--------:|-------:|----------------:|
|      2 | richards | 49887.6 |   0.26 |          0.3513 |

**avg TOO_TOHcm** (overall) — wild_avg TOO_TOHcm_sex_unknown (N=248)

|   Rank | Model    |    AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|-------:|-------:|----------------:|
|      5 | richards | 1411.2 |   1.65 |          0.1134 |

**height_cm** (F) — zoo_height_cm_by_sex (N=21)

|   Rank | Model    |    AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|-------:|-------:|----------------:|
|      5 | richards | 128.61 |    2.4 |          0.0982 |

**height_cm** (M) — zoo_height_cm_by_sex (N=56)

|   Rank | Model    |    AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|-------:|-------:|----------------:|
|      4 | richards | 330.96 |  22.31 |               0 |

**height_cm** (overall) — zoo_height_cm_overall (N=77)

|   Rank | Model    |    AIC |   ΔAIC |   Akaike Weight |
|-------:|:---------|-------:|-------:|----------------:|
|      4 | richards | 476.62 |   4.52 |          0.0661 |

### VON

**TH** (F) — wild_TH_by_sex (N=993)

|   Rank | Model           |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:----------------|--------:|-------:|----------------:|
|      1 | von_bertalanffy | 7506.36 |      0 |          0.4014 |

**TH** (M) — wild_TH_by_sex (N=563)

|   Rank | Model           |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:----------------|--------:|-------:|----------------:|
|      2 | von_bertalanffy | 4485.74 |  11.34 |          0.0034 |

**TH** (overall) — wild_TH_overall (N=1628)

|   Rank | Model           |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:----------------|--------:|-------:|----------------:|
|      2 | von_bertalanffy | 12759.9 |   40.1 |               0 |

**TH** (overall) — wild_TH_sex_unknown (N=72)

|   Rank | Model           |    AIC |   ΔAIC |   Akaike Weight |
|-------:|:----------------|-------:|-------:|----------------:|
|      5 | von_bertalanffy | 512.66 |  14.71 |          0.0002 |

**TH** (F) — wild_alignment_seed_TH_by_sex (N=993)

|   Rank | Model           |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:----------------|--------:|-------:|----------------:|
|      3 | von_bertalanffy | 7593.07 |    0.8 |          0.1935 |

**TH** (M) — wild_alignment_seed_TH_by_sex (N=563)

|   Rank | Model           |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:----------------|--------:|-------:|----------------:|
|      2 | von_bertalanffy | 4555.65 |   8.14 |          0.0166 |

**TH** (overall) — wild_alignment_seed_TH_overall (N=1628)

|   Rank | Model           |   AIC |   ΔAIC |   Akaike Weight |
|-------:|:----------------|------:|-------:|----------------:|
|      2 | von_bertalanffy | 12930 |  40.21 |               0 |

**avg NI_FBHcm** (F) — wild_alignment_seed_avg NI_FBHcm_by_sex (N=997)

|   Rank | Model           |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:----------------|--------:|-------:|----------------:|
|      5 | von_bertalanffy | 6599.05 |   3.47 |          0.0808 |

**avg NI_FBHcm** (M) — wild_alignment_seed_avg NI_FBHcm_by_sex (N=565)

|   Rank | Model           |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:----------------|--------:|-------:|----------------:|
|      2 | von_bertalanffy | 3932.65 |    8.5 |          0.0138 |

**avg NI_FBHcm** (overall) — wild_alignment_seed_avg NI_FBHcm_overall (N=1634)

|   Rank | Model           |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:----------------|--------:|-------:|----------------:|
|      2 | von_bertalanffy | 11130.2 |  33.16 |               0 |

**avg TOH_NIcm** (F) — wild_alignment_seed_avg TOH_NIcm_by_sex (N=5708)

|   Rank | Model           |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:----------------|--------:|-------:|----------------:|
|      1 | von_bertalanffy | 35037.7 |      0 |          0.7884 |

**avg TOH_NIcm** (M) — wild_alignment_seed_avg TOH_NIcm_by_sex (N=3471)

|   Rank | Model           |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:----------------|--------:|-------:|----------------:|
|      1 | von_bertalanffy | 22423.1 |      0 |          0.6185 |

**avg TOH_NIcm** (overall) — wild_alignment_seed_avg TOH_NIcm_overall (N=9428)

|   Rank | Model           |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:----------------|--------:|-------:|----------------:|
|      2 | von_bertalanffy | 60338.8 |  35.86 |               0 |

**avg TOO_TOHcm** (F) — wild_alignment_seed_avg TOO_TOHcm_by_sex (N=5681)

|   Rank | Model           |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:----------------|--------:|-------:|----------------:|
|      2 | von_bertalanffy | 29713.8 |   4.06 |          0.1104 |

**avg TOO_TOHcm** (M) — wild_alignment_seed_avg TOO_TOHcm_by_sex (N=3447)

|   Rank | Model           |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:----------------|--------:|-------:|----------------:|
|      4 | von_bertalanffy | 18536.9 |   1.03 |          0.1771 |

**avg TOO_TOHcm** (overall) — wild_alignment_seed_avg TOO_TOHcm_overall (N=9376)

|   Rank | Model           |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:----------------|--------:|-------:|----------------:|
|      5 | von_bertalanffy | 49897.9 |   4.88 |           0.059 |

**avg NI_FBHcm** (F) — wild_avg NI_FBHcm_by_sex (N=997)

|   Rank | Model           |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:----------------|--------:|-------:|----------------:|
|      1 | von_bertalanffy | 6545.39 |      0 |          0.2849 |

**avg NI_FBHcm** (M) — wild_avg NI_FBHcm_by_sex (N=565)

|   Rank | Model           |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:----------------|--------:|-------:|----------------:|
|      2 | von_bertalanffy | 3882.38 |   12.8 |          0.0017 |

**avg NI_FBHcm** (overall) — wild_avg NI_FBHcm_overall (N=1634)

|   Rank | Model           |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:----------------|--------:|-------:|----------------:|
|      2 | von_bertalanffy | 11014.6 |  33.41 |               0 |

**avg NI_FBHcm** (overall) — wild_avg NI_FBHcm_sex_unknown (N=72)

|   Rank | Model           |    AIC |   ΔAIC |   Akaike Weight |
|-------:|:----------------|-------:|-------:|----------------:|
|      1 | von_bertalanffy | 410.98 |      0 |          0.2655 |

**avg TOH_NIcm** (F) — wild_avg TOH_NIcm_by_sex (N=5708)

|   Rank | Model           |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:----------------|--------:|-------:|----------------:|
|      1 | von_bertalanffy | 34313.9 |      0 |          0.5417 |

**avg TOH_NIcm** (M) — wild_avg TOH_NIcm_by_sex (N=3471)

|   Rank | Model           |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:----------------|--------:|-------:|----------------:|
|      1 | von_bertalanffy | 22130.1 |      0 |          0.6044 |

**avg TOH_NIcm** (overall) — wild_avg TOH_NIcm_overall (N=9428)

|   Rank | Model           |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:----------------|--------:|-------:|----------------:|
|      2 | von_bertalanffy | 59398.4 |  40.42 |               0 |

**avg TOH_NIcm** (overall) — wild_avg TOH_NIcm_sex_unknown (N=249)

|   Rank | Model           |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:----------------|--------:|-------:|----------------:|
|      1 | von_bertalanffy | 1463.83 |      0 |           0.291 |

**avg TOO_TOHcm** (F) — wild_avg TOO_TOHcm_by_sex (N=5681)

|   Rank | Model           |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:----------------|--------:|-------:|----------------:|
|      1 | von_bertalanffy | 29710.7 |      0 |          0.4913 |

**avg TOO_TOHcm** (M) — wild_avg TOO_TOHcm_by_sex (N=3447)

|   Rank | Model           |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:----------------|--------:|-------:|----------------:|
|      5 | von_bertalanffy | 18535.8 |   3.67 |          0.0799 |

**avg TOO_TOHcm** (overall) — wild_avg TOO_TOHcm_overall (N=9376)

|   Rank | Model           |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:----------------|--------:|-------:|----------------:|
|      5 | von_bertalanffy | 49890.7 |   3.32 |          0.0762 |

**avg TOO_TOHcm** (overall) — wild_avg TOO_TOHcm_sex_unknown (N=248)

|   Rank | Model           |     AIC |   ΔAIC |   Akaike Weight |
|-------:|:----------------|--------:|-------:|----------------:|
|      3 | von_bertalanffy | 1409.57 |   0.03 |          0.2555 |

**height_cm** (F) — zoo_height_cm_by_sex (N=21)

|   Rank | Model           |   AIC |   ΔAIC |   Akaike Weight |
|-------:|:----------------|------:|-------:|----------------:|
|      1 | von_bertalanffy | 126.2 |      0 |          0.3266 |

**height_cm** (M) — zoo_height_cm_by_sex (N=56)

|   Rank | Model           |    AIC |   ΔAIC |   Akaike Weight |
|-------:|:----------------|-------:|-------:|----------------:|
|      2 | von_bertalanffy | 325.23 |  16.58 |          0.0003 |

**height_cm** (overall) — zoo_height_cm_overall (N=77)

|   Rank | Model           |   AIC |   ΔAIC |   Akaike Weight |
|-------:|:----------------|------:|-------:|----------------:|
|      1 | von_bertalanffy | 472.1 |      0 |          0.6333 |

## Interpretation Guide

- **ΔAIC**: Difference from best model (0 = best)
- **Akaike Weight**: Probability this is the best model given the candidates
- Models with ΔAIC < 1% of best AIC: substantially supported alternatives
- Models with ΔAIC < 2: have meaningful support
- Models with ΔAIC > 10: have essentially no support
