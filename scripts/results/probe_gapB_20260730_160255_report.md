# probe_gapB

## 표 1

| point_id | j_net_round0 | j_net_round1 | j_net_round2 | b_energy_round0 | b_energy_round1 | b_energy_round2 | b_defer_round0 | b_defer_round1 | b_defer_round2 | delta_round1_minus_round0 | delta_round1_minus_round0_pct_abs_j0 | delta_round2_minus_round1 | delta_b_energy_1_minus_0 | delta_b_defer_1_minus_0 | delta_b_energy_2_minus_1 | delta_b_defer_2_minus_1 | delta10_sign | delta21_sign |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1_old_full_opt | 2832817.668 | 2830148.759 | 2830166.107 | 1467921.783 | 1465989.434 | 1466009.3 | 11872568.32 | 11871831.76 | 11871829.24 | -2668.909465 | -0.09421395153 | 17.34771635 | -1932.349135 | -736.5603303 | 19.86631146 | -2.518595111 | negative | positive |
| P2_dev_run1 | 922164.1332 | 920907.0486 | 920922.5068 | -70901.34785 | -68856.0205 | -68834.81486 | 11526130.65 | 11522828.24 | 11522822.49 | -1257.084587 | -0.1363189634 | 15.45815919 | 2045.327348 | -3302.411935 | 21.20564138 | -5.747482184 | negative | positive |
| P3_dev_run0 | -8430058.943 | -8428863.596 | -8428868.57 | -6870867.6 | -6874913.452 | -6874912.713 | 11232927.95 | 11238169.15 | 11238163.44 | 1195.3469 | 0.01417957938 | -4.973839194 | -4045.851987 | 5241.198887 | 0.7393451137 | -5.713184308 | positive | negative |
| P4_normal | 2801687.98 | 2798822.735 | 2798848.821 | 1435508.742 | 1433538.905 | 1433565.679 | 11707163.46 | 11706268.05 | 11706267.36 | -2865.244904 | -0.1022685226 | 26.08653683 | -1969.837855 | -895.4070495 | 26.7741955 | -0.6876586713 | negative | positive |

## 표 2

| point_id | coefficient | relative_change_median | relative_change_p95 | relative_change_max |
|---|---|---|---|---|
| P1_old_full_opt | a_P | 0.0005710053266 | 0.005783788818 | 0.006093445098 |
| P1_old_full_opt | a_Q | 0.004698783018 | 0.04233265378 | 0.04499234437 |
| P1_old_full_opt | b_PP | 0.8549508307 | 21.79291494 | 68.18374928 |
| P1_old_full_opt | b_QQ | 0.03367869082 | 0.1845694967 | 0.2107461271 |
| P1_old_full_opt | b_PQ | 5.828955017 | 14.50208418 | 24.93490138 |
| P2_dev_run1 | a_P | 0.001137578001 | 0.005887888181 | 0.006303652968 |
| P2_dev_run1 | a_Q | 0.002598613664 | 0.02616466527 | 0.04045473486 |
| P2_dev_run1 | b_PP | 0.07558926907 | 0.9183254969 | 1.116659984 |
| P2_dev_run1 | b_QQ | 0.01174494614 | 0.08360766715 | 0.1067369573 |
| P2_dev_run1 | b_PQ | 3.029328126 | 19.46034032 | 1785.517247 |
| P3_dev_run0 | a_P | 4.050568426e-05 | 0.006257834055 | 0.0070017233 |
| P3_dev_run0 | a_Q | 9.21093462e-06 | 0.00484197426 | 0.007734927446 |
| P3_dev_run0 | b_PP | 0.000411856435 | 0.07813550643 | 0.2660808133 |
| P3_dev_run0 | b_QQ | 1.183330349e-05 | 0.01272686872 | 0.02289935236 |
| P3_dev_run0 | b_PQ | 0.007537435587 | 3.042789981 | 33.24043355 |
| P4_normal | a_P | 0.0005772269705 | 0.00575196596 | 0.006072560965 |
| P4_normal | a_Q | 0.004918435919 | 0.04233284047 | 0.04499297648 |
| P4_normal | b_PP | 0.8766292356 | 21.79253112 | 68.21032701 |
| P4_normal | b_QQ | 0.03351085749 | 0.1845694732 | 0.2107469635 |
| P4_normal | b_PQ | 4.748051731 | 14.54057473 | 21.53522156 |

## 표 3

| b | S | E_values | coefficient | n_groups | relative_span_median | relative_span_max | rank_lt_5_count_across_E | psd_violation_count_across_E |
|---|---|---|---|---|---|---|---|---|
| 15 | 0.176 | (0.061599999999999995, 0.176, 0.528) | a_P | 120 | 0 | 0 | 0 | 0 |
| 15 | 0.176 | (0.061599999999999995, 0.176, 0.528) | a_Q | 120 | 0 | 0 | 0 | 0 |
| 15 | 0.176 | (0.061599999999999995, 0.176, 0.528) | b_PP | 120 | 0 | 0 | 0 | 0 |
| 15 | 0.176 | (0.061599999999999995, 0.176, 0.528) | b_QQ | 120 | 0 | 0 | 0 | 0 |
| 15 | 0.176 | (0.061599999999999995, 0.176, 0.528) | b_PQ | 120 | 0 | 0 | 0 | 0 |
| 17 | 0.303 | (0.10604999999999999, 0.303, 0.909) | a_P | 120 | 0 | 0 | 0 | 0 |
| 17 | 0.303 | (0.10604999999999999, 0.303, 0.909) | a_Q | 120 | 0 | 0 | 0 | 0 |
| 17 | 0.303 | (0.10604999999999999, 0.303, 0.909) | b_PP | 120 | 0 | 0 | 0 | 0 |
| 17 | 0.303 | (0.10604999999999999, 0.303, 0.909) | b_QQ | 120 | 0 | 0 | 0 | 0 |
| 17 | 0.303 | (0.10604999999999999, 0.303, 0.909) | b_PQ | 120 | 0 | 0 | 0 | 0 |
| 31 | 1.045 | (0.36574999999999996, 1.045, 3.135) | a_P | 120 | 0 | 0 | 0 | 0 |
| 31 | 1.045 | (0.36574999999999996, 1.045, 3.135) | a_Q | 120 | 0 | 0 | 0 | 0 |
| 31 | 1.045 | (0.36574999999999996, 1.045, 3.135) | b_PP | 120 | 0 | 0 | 0 | 0 |
| 31 | 1.045 | (0.36574999999999996, 1.045, 3.135) | b_QQ | 120 | 0 | 0 | 0 | 0 |
| 31 | 1.045 | (0.36574999999999996, 1.045, 3.135) | b_PQ | 120 | 0 | 0 | 0 | 0 |

## 표 4

| point_id | alpha_actual_over_model_ls | coefficient_scale_1_over_alpha | j_net_round0 | j_net_round1 | j_net_grossup | grossup_minus_round1 | absolute_gap_fraction_closed |
|---|---|---|---|---|---|---|---|
| P1_old_full_opt | 0.9893025751 | 1.010813097 | 2832817.668 | 2830148.759 | 2831661.199 | 1512.440167 | 0.4333115502 |
| P2_dev_run1 | 0.9959331073 | 1.0040835 | 922164.1332 | 920907.0486 | 921554.4496 | 647.4010094 | 0.4849980534 |
| P3_dev_run0 | 0.9957520404 | 1.004266082 | -8430058.943 | -8428863.596 | -8430676.468 | -1812.872551 | -0.5166078998 |
| P4_normal | 0.9893039857 | 1.010811656 | 2801687.98 | 2798822.735 | 2800575.208 | 1752.473559 | 0.3883686672 |

## 표 5

| point_id | solution_label | scenario | solver | status | rank_lt_5_count | psd_violation_count | peak_sign_bad_count | peak_discharge_count | peak_loss_reduction_bad_count | peak_loss_reduction_min | hessian_projection_count |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P1_old_full_opt | round0 | summer | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.0005911646375 | 0 |
| P1_old_full_opt | round0 | winter | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.0005911646375 | 0 |
| P1_old_full_opt | round0 | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.0005911646375 | 0 |
| P1_old_full_opt | round0 | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.0005911646375 | 0 |
| P1_old_full_opt | round0 | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.0005911646375 | 0 |
| P1_old_full_opt | round1 | summer | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.0005879257647 | 0 |
| P1_old_full_opt | round1 | winter | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.0005879257647 | 0 |
| P1_old_full_opt | round1 | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.0005879257647 | 0 |
| P1_old_full_opt | round1 | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.0005879257647 | 0 |
| P1_old_full_opt | round1 | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.0005879257647 | 0 |
| P1_old_full_opt | round2 | summer | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.0005879331938 | 0 |
| P1_old_full_opt | round2 | winter | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.0005879331938 | 0 |
| P1_old_full_opt | round2 | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.0005879331938 | 0 |
| P1_old_full_opt | round2 | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.0005879331938 | 0 |
| P1_old_full_opt | round2 | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.0005879331938 | 0 |
| P1_old_full_opt | grossup | summer | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.0006096768398 | 0 |
| P1_old_full_opt | grossup | winter | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.0006096768398 | 0 |
| P1_old_full_opt | grossup | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.0006096768398 | 0 |
| P1_old_full_opt | grossup | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.0006096768398 | 0 |
| P1_old_full_opt | grossup | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.0006096768398 | 0 |
| P2_dev_run1 | round0 | summer | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.0007211770069 | 0 |
| P2_dev_run1 | round0 | winter | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.0007211770069 | 0 |
| P2_dev_run1 | round0 | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.0007211770069 | 0 |
| P2_dev_run1 | round0 | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.0007211770069 | 0 |
| P2_dev_run1 | round0 | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.0007211770069 | 0 |
| P2_dev_run1 | round1 | summer | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.0007330718934 | 0 |
| P2_dev_run1 | round1 | winter | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.0007330718934 | 0 |
| P2_dev_run1 | round1 | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.0007330718934 | 0 |
| P2_dev_run1 | round1 | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.0007330718934 | 0 |
| P2_dev_run1 | round1 | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.0007330718934 | 0 |
| P2_dev_run1 | round2 | summer | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.0007331245303 | 0 |
| P2_dev_run1 | round2 | winter | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.0007331245303 | 0 |
| P2_dev_run1 | round2 | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.0007331245303 | 0 |
| P2_dev_run1 | round2 | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.0007331245303 | 0 |
| P2_dev_run1 | round2 | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.0007331245303 | 0 |
| P2_dev_run1 | grossup | summer | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.0007360264351 | 0 |
| P2_dev_run1 | grossup | winter | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.0007360264351 | 0 |
| P2_dev_run1 | grossup | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.0007360264351 | 0 |
| P2_dev_run1 | grossup | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.0007360264351 | 0 |
| P2_dev_run1 | grossup | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.0007360264351 | 0 |
| P3_dev_run0 | round0 | summer | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.005249033513 | 0 |
| P3_dev_run0 | round0 | winter | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.005249033513 | 0 |
| P3_dev_run0 | round0 | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.005249033513 | 0 |
| P3_dev_run0 | round0 | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.005249033513 | 0 |
| P3_dev_run0 | round0 | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.005249033513 | 0 |
| P3_dev_run0 | round1 | summer | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.005243696034 | 0 |
| P3_dev_run0 | round1 | winter | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.005243696034 | 0 |
| P3_dev_run0 | round1 | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.005243696034 | 0 |
| P3_dev_run0 | round1 | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.005243696034 | 0 |
| P3_dev_run0 | round1 | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.005243696034 | 0 |
| P3_dev_run0 | round2 | summer | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.005243697396 | 0 |
| P3_dev_run0 | round2 | winter | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.005243697396 | 0 |
| P3_dev_run0 | round2 | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.005243697396 | 0 |
| P3_dev_run0 | round2 | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.005243697396 | 0 |
| P3_dev_run0 | round2 | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.005243697396 | 0 |
| P3_dev_run0 | grossup | summer | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.005289476431 | 0 |
| P3_dev_run0 | grossup | winter | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.005289476431 | 0 |
| P3_dev_run0 | grossup | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.005289476431 | 0 |
| P3_dev_run0 | grossup | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.005289476431 | 0 |
| P3_dev_run0 | grossup | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 8 | 0 | 0.005289476431 | 0 |
| P4_normal | round0 | summer | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.0005868211461 | 0 |
| P4_normal | round0 | winter | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.0005868211461 | 0 |
| P4_normal | round0 | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.0005868211461 | 0 |
| P4_normal | round0 | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.0005868211461 | 0 |
| P4_normal | round0 | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.0005868211461 | 0 |
| P4_normal | round1 | summer | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.0005839427557 | 0 |
| P4_normal | round1 | winter | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.0005839427557 | 0 |
| P4_normal | round1 | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.0005839427557 | 0 |
| P4_normal | round1 | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.0005839427557 | 0 |
| P4_normal | round1 | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.0005839427557 | 0 |
| P4_normal | round2 | summer | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.000583950017 | 0 |
| P4_normal | round2 | winter | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.000583950017 | 0 |
| P4_normal | round2 | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.000583950017 | 0 |
| P4_normal | round2 | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.000583950017 | 0 |
| P4_normal | round2 | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.000583950017 | 0 |
| P4_normal | grossup | summer | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.0006053486418 | 0 |
| P4_normal | grossup | winter | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.0006053486418 | 0 |
| P4_normal | grossup | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.0006053486418 | 0 |
| P4_normal | grossup | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.0006053486418 | 0 |
| P4_normal | grossup | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 7 | 0 | 0.0006053486418 | 0 |

## 환경

| item | value |
|---|---|
| timestamp | 2026-07-30T16:06:15 |
| hostname | PSL |
| platform | Windows-10-10.0.26200-SP0 |
| python | 3.11.15 |
| numpy | 2.2.6 |
| pandapower | 3.2.1 |
| installed_solvers | CLARABEL,SCS,SCIPY,HIGHS,OSQP |
| POLY_N | 128 |
| E_test_multipliers | (0.35, 1.0, 3.0) |
| runpp_calls_total | 45361 |
| pf_retry_events | 0 |
| pf_diverged_calls | 0 |
| ac_elapsed_s | 199.903221700 |
| runpp_calls_per_ac_s | 226.914802144 |
| total_execution_time_s | 200.229636700 |
| alpha_definition | sum(actual_Lcost*model_Lcost)/sum(model_Lcost^2) |
| grossup_scale_definition | 1/alpha |
| hessian_projection_count | 0 |
| load_p_sum_mw | 9.5 |
| load_q_sum_mvar | 3.122498999 |
| load_s_sum_mva | 10 |
| load_total_pf | 0.95 |
| K_P | 2.557200538 |
| K_Q | 1.357608261 |
| ETA_PCS | 0.975 |
| SLACK_VM_PU | 1.02 |
| coefficient_measurement_groups | 1440 |
| round0_feasible_grid_rows_per_time_all_points | 100 |
| round0_surface_runpp_exact_no_retry | 11520 |
| E_refit_groups | 1080 |
| baseline_runpp_no_retry | 120 |
| core_surface_runpp_upper_no_retry | 35520 |
| E_surface_runpp_no_retry | 8640 |
| schedule_runpp_no_retry | 1920 |
| total_runpp_upper_no_retry | 46200 |
| expected_seconds_at_200_calls_per_s | 231 |
| expected_seconds_at_230_calls_per_s | 200.8695652 |
| expected_seconds_at_80_calls_per_s | 577.5 |
| solver_CLARABEL_version | 0.11.1 |
| solver_SCS_version | 3.2.11 |
| solver_OSQP_version | 1.1.3 |
| solver_SCIPY_version | 1.13.1 |
| solver_HIGHS_version | 1.15.1 |
