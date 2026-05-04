# ymat

| id | variable          | dimensions       | units       | long_name                                     |
|----|-------------------|------------------|-------------|-----------------------------------------------|
|  1 | enh               | xc, yc, zeta     | -           | Enhancement factor                            |
|  2 | enh_bnd           | xc, yc, zeta     | -           | Imposed enhancement factor                    |
|  3 | enh_bar           | xc, yc           | -           | Depth-averaged enhancement                    |
|  4 | ATT               | xc, yc, zeta     | -           | Rate factor                                   |
|  5 | ATT_bar           | xc, yc           | -           | Depth-averaged rate factor                    |
|  6 | visc              | xc, yc, zeta     | Pa yr       | Ice viscosity                                 |
|  7 | visc_bar          | xc, yc           | Pa yr       | Depth-averaged ice viscosity                  |
|  8 | visc_int          | xc, yc           | Pa yr m     | Ice viscosity interpolated at interfaces      |
|  9 | dep_time          | xc, yc, zeta     | yr          | Ice deposition time (for online age tracing)  |
| 10 | depth_iso         | xc, yc, age_iso  | m           | Depth of specific isochronal layers           |
| 11 | strs2D_txx        | xc, yc           | Pa          | 2D stress tensor component txx                |
| 12 | strs2D_tyy        | xc, yc           | Pa          | 2D stress tensor component tyy                |
| 13 | strs2D_txy        | xc, yc           | Pa          | 2D stress tensor component txy                |
| 14 | strs2D_txz        | xc, yc           | Pa          | 2D stress tensor component txz                |
| 15 | strs2D_tyz        | xc, yc           | Pa          | 2D stress tensor component tyz                |
| 16 | strs2D_te         | xc, yc           | Pa          | 2D effective stress                           |
| 17 | strs2D_tau_eig_1  | xc, yc           | Pa          | 2D stress first principal eigenvalue          |
| 18 | strs2D_tau_eig_2  | xc, yc           | Pa          | 2D stress second principal eigenvalue         |
| 19 | strs_txx          | xc, yc, zeta     | Pa          | Stress tensor component txx                   |
| 20 | strs_tyy          | xc, yc, zeta     | Pa          | Stress tensor component tyy                   |
| 21 | strs_txy          | xc, yc, zeta     | Pa          | Stress tensor component txy                   |
| 22 | strs_txz          | xc, yc, zeta     | Pa          | Stress tensor component txz                   |
| 23 | strs_tyz          | xc, yc, zeta     | Pa          | Stress tensor component tyz                   |
| 24 | strs_te           | xc, yc, zeta     | Pa          | Effective stress                              |
