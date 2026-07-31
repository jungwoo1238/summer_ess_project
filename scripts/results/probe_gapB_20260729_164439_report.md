# probe_gapB

## 표 1

| point_id | j_net_round0 | j_net_round1 | j_net_round2 | delta_round1_minus_round0 | delta_round1_minus_round0_pct_abs_j0 | delta_round2_minus_round1 | delta_b_energy_1_minus_0 | delta_b_defer_1_minus_0 | delta_b_energy_2_minus_1 | delta_b_defer_2_minus_1 | delta10_sign | delta21_sign |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1_old_full_opt | 3917912.154 | 3905284.299 | 3905288.32 | -12627.8551 | -0.3223108279 | 4.02130807 | -9734.944947 | -2892.910154 | 0.005287693348 | 4.016020376 | negative | positive |
| P2_dev_run1 | 3422585.379 | 3418169.077 | 3418177.838 | -4416.30159 | -0.129034081 | 8.761238545 | 1054.223488 | -5470.525078 | -1.951846445 | 10.71308499 | negative | positive |
| P3_dev_run0 | 2021363.585 | 2029044.526 | 2029027.807 | 7680.941108 | 0.3799881014 | -16.71899592 | 6.95149141 | 7673.989617 | 0.3876833268 | -17.10667925 | positive | negative |
| P4_normal | 3933988.601 | 3920730.436 | 3920735.13 | -13258.16509 | -0.3370158493 | 4.694321465 | -10228.54162 | -3029.623478 | -0.001430277713 | 4.695751742 | negative | positive |

## 표 2

| point_id | coefficient | relative_change_median | relative_change_p95 | relative_change_max |
|---|---|---|---|---|
| P1_old_full_opt | a_P | 0.000957747841 | 0.00690786442 | 0.007299300716 |
| P1_old_full_opt | a_Q | 0.003077059817 | 0.02388282793 | 0.02896829937 |
| P1_old_full_opt | b_PP | 1.369570611 | 52.17938173 | 608.4318393 |
| P1_old_full_opt | b_QQ | 0.03671910714 | 0.1905239297 | 0.2212168012 |
| P1_old_full_opt | b_PQ | 6.215453375 | 16.4385751 | 28.58427052 |
| P2_dev_run1 | a_P | 0.001296132408 | 0.007215881427 | 0.007683937996 |
| P2_dev_run1 | a_Q | 0.002068966322 | 0.01499604963 | 0.02288611652 |
| P2_dev_run1 | b_PP | 0.08936491998 | 1.06389476 | 1.340759436 |
| P2_dev_run1 | b_QQ | 0.01309688578 | 0.08940223953 | 0.1057998433 |
| P2_dev_run1 | b_PQ | 3.590446501 | 29.23937277 | 148.2203771 |
| P3_dev_run0 | a_P | 0.0001274869114 | 0.007322341179 | 0.008581709887 |
| P3_dev_run0 | a_Q | 6.213898058e-05 | 0.003120883989 | 0.004893605199 |
| P3_dev_run0 | b_PP | 0.001657118349 | 0.08859439707 | 0.2976087375 |
| P3_dev_run0 | b_QQ | 0.0002248265725 | 0.01349192416 | 0.02227766335 |
| P3_dev_run0 | b_PQ | 0.01943741112 | 2.154377898 | 9.296517623 |
| P4_normal | a_P | 0.001016122352 | 0.006979867282 | 0.007312035309 |
| P4_normal | a_Q | 0.003059152714 | 0.02384808455 | 0.02896758847 |
| P4_normal | b_PP | 1.307033592 | 51.5624949 | 608.7347877 |
| P4_normal | b_QQ | 0.03647233298 | 0.1905175946 | 0.2212152426 |
| P4_normal | b_PQ | 6.18738305 | 16.65626246 | 28.60130396 |

## 표 3

| b | S | E_values | coefficient | n_groups | relative_span_median | relative_span_max | rank_lt_5_count_across_E | psd_violation_count_across_E |
|---|---|---|---|---|---|---|---|---|
| 15 | 0.176 | (0.2, 0.4, 0.8) | a_P | 120 | 0 | 0 | 0 | 60 |
| 15 | 0.176 | (0.2, 0.4, 0.8) | a_Q | 120 | 0 | 0 | 0 | 60 |
| 15 | 0.176 | (0.2, 0.4, 0.8) | b_PP | 120 | 0 | 0 | 0 | 60 |
| 15 | 0.176 | (0.2, 0.4, 0.8) | b_QQ | 120 | 0 | 0 | 0 | 60 |
| 15 | 0.176 | (0.2, 0.4, 0.8) | b_PQ | 120 | 0 | 0 | 0 | 60 |
| 17 | 0.303 | (0.2, 0.4, 0.8) | a_P | 120 | 0 | 0 | 0 | 0 |
| 17 | 0.303 | (0.2, 0.4, 0.8) | a_Q | 120 | 0 | 0 | 0 | 0 |
| 17 | 0.303 | (0.2, 0.4, 0.8) | b_PP | 120 | 0 | 0 | 0 | 0 |
| 17 | 0.303 | (0.2, 0.4, 0.8) | b_QQ | 120 | 0 | 0 | 0 | 0 |
| 17 | 0.303 | (0.2, 0.4, 0.8) | b_PQ | 120 | 0 | 0 | 0 | 0 |
| 31 | 1.045 | (0.2, 0.4, 0.8) | a_P | 120 | 9.726809977e-05 | 0.0001125222089 | 0 | 0 |
| 31 | 1.045 | (0.2, 0.4, 0.8) | a_Q | 120 | 0.004641541387 | 0.004672023438 | 0 | 0 |
| 31 | 1.045 | (0.2, 0.4, 0.8) | b_PP | 120 | 0.3800134418 | 0.4956253313 | 0 | 0 |
| 31 | 1.045 | (0.2, 0.4, 0.8) | b_QQ | 120 | 0.009282088222 | 0.01157706812 | 0 | 0 |
| 31 | 1.045 | (0.2, 0.4, 0.8) | b_PQ | 120 | 0.1438399709 | 0.5885953968 | 0 | 0 |

## 표 4

| point_id | alpha_actual_over_model_ls | coefficient_scale_1_over_alpha | j_net_round0 | j_net_round1 | j_net_grossup | grossup_minus_round1 | absolute_gap_fraction_closed |
|---|---|---|---|---|---|---|---|
| P1_old_full_opt | 0.9910924996 | 1.008987557 | 3917912.154 | 3905284.299 | 3915974.592 | 10690.2927 | 0.1534355904 |
| P2_dev_run1 | 0.9962416869 | 1.003772491 | 3422585.379 | 3418169.077 | 3421558.248 | 3389.171103 | 0.2325770705 |
| P3_dev_run0 | 0.9974003805 | 1.002606395 | 2021363.585 | 2029044.526 | 2020576.859 | -8467.666993 | -0.102425715 |
| P4_normal | 0.9911196713 | 1.008959896 | 3933988.601 | 3920730.436 | 3931695.436 | 10965.00037 | 0.1729624504 |

## 표 5

| point_id | solution_label | scenario | solver | status | rank_lt_5_count | psd_violation_count | peak_sign_bad_count | hessian_projection_count |
|---|---|---|---|---|---|---|---|---|
| P1_old_full_opt | round0 | summer | CLARABEL | optimal | 0 | 20 | 0 | 0 |
| P1_old_full_opt | round0 | winter | CLARABEL | optimal | 0 | 20 | 0 | 0 |
| P1_old_full_opt | round0 | shoulder | CLARABEL | optimal | 0 | 20 | 0 | 0 |
| P1_old_full_opt | round0 | summer_peak | CLARABEL | optimal | 0 | 20 | 0 | 0 |
| P1_old_full_opt | round0 | winter_peak | CLARABEL | optimal | 0 | 20 | 0 | 0 |
| P1_old_full_opt | round1 | summer | CLARABEL | optimal | 0 | 2 | 0 | 0 |
| P1_old_full_opt | round1 | winter | CLARABEL | optimal | 0 | 2 | 0 | 0 |
| P1_old_full_opt | round1 | shoulder | CLARABEL | optimal | 0 | 2 | 0 | 0 |
| P1_old_full_opt | round1 | summer_peak | CLARABEL | optimal | 0 | 2 | 0 | 0 |
| P1_old_full_opt | round1 | winter_peak | CLARABEL | optimal | 0 | 2 | 0 | 0 |
| P1_old_full_opt | round2 | summer | CLARABEL | optimal | 0 | 1 | 0 | 0 |
| P1_old_full_opt | round2 | winter | CLARABEL | optimal | 0 | 1 | 0 | 0 |
| P1_old_full_opt | round2 | shoulder | CLARABEL | optimal | 0 | 1 | 0 | 0 |
| P1_old_full_opt | round2 | summer_peak | CLARABEL | optimal | 0 | 1 | 0 | 0 |
| P1_old_full_opt | round2 | winter_peak | CLARABEL | optimal | 0 | 1 | 0 | 0 |
| P1_old_full_opt | grossup | summer | CLARABEL | optimal | 0 | 20 | 0 | 0 |
| P1_old_full_opt | grossup | winter | CLARABEL | optimal | 0 | 20 | 0 | 0 |
| P1_old_full_opt | grossup | shoulder | CLARABEL | optimal | 0 | 20 | 0 | 0 |
| P1_old_full_opt | grossup | summer_peak | CLARABEL | optimal | 0 | 20 | 0 | 0 |
| P1_old_full_opt | grossup | winter_peak | CLARABEL | optimal | 0 | 20 | 0 | 0 |
| P2_dev_run1 | round0 | summer | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P2_dev_run1 | round0 | winter | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P2_dev_run1 | round0 | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P2_dev_run1 | round0 | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P2_dev_run1 | round0 | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P2_dev_run1 | round1 | summer | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P2_dev_run1 | round1 | winter | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P2_dev_run1 | round1 | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P2_dev_run1 | round1 | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P2_dev_run1 | round1 | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P2_dev_run1 | round2 | summer | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P2_dev_run1 | round2 | winter | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P2_dev_run1 | round2 | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P2_dev_run1 | round2 | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P2_dev_run1 | round2 | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P2_dev_run1 | grossup | summer | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P2_dev_run1 | grossup | winter | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P2_dev_run1 | grossup | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P2_dev_run1 | grossup | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P2_dev_run1 | grossup | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P3_dev_run0 | round0 | summer | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P3_dev_run0 | round0 | winter | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P3_dev_run0 | round0 | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P3_dev_run0 | round0 | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P3_dev_run0 | round0 | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P3_dev_run0 | round1 | summer | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P3_dev_run0 | round1 | winter | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P3_dev_run0 | round1 | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P3_dev_run0 | round1 | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P3_dev_run0 | round1 | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P3_dev_run0 | round2 | summer | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P3_dev_run0 | round2 | winter | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P3_dev_run0 | round2 | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P3_dev_run0 | round2 | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P3_dev_run0 | round2 | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P3_dev_run0 | grossup | summer | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P3_dev_run0 | grossup | winter | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P3_dev_run0 | grossup | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P3_dev_run0 | grossup | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P3_dev_run0 | grossup | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 |
| P4_normal | round0 | summer | CLARABEL | optimal | 0 | 20 | 0 | 0 |
| P4_normal | round0 | winter | CLARABEL | optimal | 0 | 20 | 0 | 0 |
| P4_normal | round0 | shoulder | CLARABEL | optimal | 0 | 20 | 0 | 0 |
| P4_normal | round0 | summer_peak | CLARABEL | optimal | 0 | 20 | 0 | 0 |
| P4_normal | round0 | winter_peak | CLARABEL | optimal | 0 | 20 | 0 | 0 |
| P4_normal | round1 | summer | CLARABEL | optimal | 0 | 1 | 0 | 0 |
| P4_normal | round1 | winter | CLARABEL | optimal | 0 | 1 | 0 | 0 |
| P4_normal | round1 | shoulder | CLARABEL | optimal | 0 | 1 | 0 | 0 |
| P4_normal | round1 | summer_peak | CLARABEL | optimal | 0 | 1 | 0 | 0 |
| P4_normal | round1 | winter_peak | CLARABEL | optimal | 0 | 1 | 0 | 0 |
| P4_normal | round2 | summer | CLARABEL | optimal | 0 | 1 | 0 | 0 |
| P4_normal | round2 | winter | CLARABEL | optimal | 0 | 1 | 0 | 0 |
| P4_normal | round2 | shoulder | CLARABEL | optimal | 0 | 1 | 0 | 0 |
| P4_normal | round2 | summer_peak | CLARABEL | optimal | 0 | 1 | 0 | 0 |
| P4_normal | round2 | winter_peak | CLARABEL | optimal | 0 | 1 | 0 | 0 |
| P4_normal | grossup | summer | CLARABEL | optimal | 0 | 20 | 0 | 0 |
| P4_normal | grossup | winter | CLARABEL | optimal | 0 | 20 | 0 | 0 |
| P4_normal | grossup | shoulder | CLARABEL | optimal | 0 | 20 | 0 | 0 |
| P4_normal | grossup | summer_peak | CLARABEL | optimal | 0 | 20 | 0 | 0 |
| P4_normal | grossup | winter_peak | CLARABEL | optimal | 0 | 20 | 0 | 0 |

## 환경

| item | value |
|---|---|
| timestamp | 2026-07-29T16:49:37 |
| hostname | PSL |
| platform | Windows-10-10.0.26200-SP0 |
| python | 3.11.15 |
| numpy | 2.2.6 |
| pandapower | 3.2.1 |
| installed_solvers | CLARABEL,SCS,SCIPY,HIGHS,OSQP |
| POLY_N | 128 |
| runpp_calls_total | 45543 |
| pf_retry_events | 0 |
| pf_diverged_calls | 0 |
| ac_elapsed_s | 297.353273400 |
| runpp_calls_per_ac_s | 153.161253210 |
| total_execution_time_s | 297.682336800 |
| alpha_definition | sum(actual_Lcost*model_Lcost)/sum(model_Lcost^2) |
| grossup_scale_definition | 1/alpha |
| hessian_projection_count | 0 |
| coefficient_measurement_groups | 1440 |
| round0_feasible_grid_rows_per_time_all_points | 100 |
| round0_surface_runpp_exact_no_retry | 11520 |
| E_refit_groups | 1080 |
| baseline_runpp_no_retry | 120 |
| core_surface_runpp_upper_no_retry | 35520 |
| E_surface_runpp_no_retry | 8640 |
| schedule_runpp_no_retry | 1920 |
| total_runpp_upper_no_retry | 46200 |
| expected_seconds_at_230_calls_per_s | 200.8695652 |
| expected_seconds_at_80_calls_per_s | 577.5 |
| solver_CLARABEL_version | 0.11.1 |
| solver_SCS_version | 3.2.11 |
| solver_OSQP_version | 1.1.3 |
| solver_SCIPY_version | 1.13.1 |
| solver_HIGHS_version | 1.15.1 |
