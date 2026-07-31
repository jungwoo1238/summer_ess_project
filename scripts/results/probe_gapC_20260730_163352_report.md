# probe_gapC

## 표 1

| point_id | b_defer_round0 | b_defer_round1 | b_defer_round2 | delta_b_defer_1_minus_0 | delta_b_defer_2_minus_1 | j_net_round0 | j_net_round1 | j_net_round2 | delta_j_net_1_minus_0 | delta_j_net_2_minus_1 |
|---|---|---|---|---|---|---|---|---|---|---|
| P1_old_full_opt | 11872568.32 | 11871831.76 | 11871829.24 | -736.56035 | -2.518583758 | 2832817.668 | 2832081.108 | 2832078.59 | -736.5603512 | -2.518585313 |
| P2_dev_run1 | 11526130.65 | 11522828.24 | 11522822.49 | -3302.411943 | -5.747493673 | 922164.1332 | 918861.6547 | 918855.9072 | -3302.478519 | -5.747492163 |
| P3_dev_run0 | 11232927.95 | 11238169.15 | 11238163.44 | 5241.198871 | -5.713179577 | -8430058.943 | -8424817.74 | -8424823.453 | 5241.203122 | -5.713181851 |
| P4_normal | 11707163.46 | 11706268.05 | 11706267.36 | -895.4073184 | -0.6875754204 | 2801687.98 | 2800792.573 | 2800791.885 | -895.4073211 | -0.6875745915 |

## 표 2

| point_id | scenario | lp_pk_round0_mw | ac_slack_peak_round0_mw | gap_round0_mw | lp_pk_round1_mw | ac_slack_peak_round1_mw | gap_round1_mw | lp_pk_round2_mw | ac_slack_peak_round2_mw | gap_round2_mw | delta_gap_1_minus_0_mw | delta_gap_2_minus_1_mw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1_old_full_opt | summer_peak | 9.342518295 | 9.62519723 | 0.2826789358 | 9.342409518 | 9.625206911 | 0.2827973932 | 9.342409581 | 9.625206945 | 0.2827973637 | 0.0001184573928 | -2.9504859e-08 |
| P1_old_full_opt | winter_peak | 9.046216487 | 9.31221153 | 0.2659950433 | 9.046210318 | 9.312211797 | 0.2660014791 | 9.046210317 | 9.312211797 | 0.2660014803 | 6.435851791e-06 | 1.197980382e-09 |
| P2_dev_run1 | summer_peak | 9.342159847 | 9.629750696 | 0.2875908492 | 9.342049393 | 9.629794102 | 0.2877447089 | 9.342049523 | 9.629794177 | 0.2877446538 | 0.0001538596542 | -5.503965816e-08 |
| P2_dev_run1 | winter_peak | 9.022093403 | 9.292379911 | 0.2702865088 | 9.021914504 | 9.292395016 | 0.2704805115 | 9.021914505 | 9.292394999 | 0.2704804939 | 0.0001940026571 | -1.761449475e-08 |
| P3_dev_run0 | summer_peak | 9.314720396 | 9.633604458 | 0.3188840623 | 9.31478601 | 9.633535569 | 0.3187495596 | 9.314785853 | 9.633535644 | 0.3187497915 | -0.0001345026897 | 2.319249823e-07 |
| P3_dev_run0 | winter_peak | 8.995559486 | 9.297019458 | 0.3014599726 | 8.995595456 | 9.297062682 | 0.3014672252 | 8.995595431 | 9.297062578 | 0.3014671479 | 7.252642067e-06 | -7.737085106e-08 |
| P4_normal | summer_peak | 9.344641783 | 9.627371259 | 0.2827294759 | 9.344533022 | 9.627383028 | 0.2828500058 | 9.344533063 | 9.627383037 | 0.2828499738 | 0.0001205299033 | -3.194031173e-08 |
| P4_normal | winter_peak | 9.046216038 | 9.312211167 | 0.2659951289 | 9.046209569 | 9.312211171 | 0.2660016024 | 9.046209569 | 9.312211171 | 0.2660016024 | 6.473412867e-06 | 1.737809896e-11 |

## 표 3

| point_id | ac_annual_peak_round0_mw | ac_annual_peak_round1_mw | ac_annual_peak_round2_mw | delta_peak_1_minus_0_mw | delta_peak_2_minus_1_mw |
|---|---|---|---|---|---|
| P1_old_full_opt | 9.62519723 | 9.625206911 | 9.625206945 | 9.681112665e-06 | 3.310345598e-08 |
| P2_dev_run1 | 9.629750696 | 9.629794102 | 9.629794177 | 4.340584188e-05 | 7.55432108e-08 |
| P3_dev_run0 | 9.633604458 | 9.633535569 | 9.633535644 | -6.888863454e-05 | 7.509219735e-08 |
| P4_normal | 9.627371259 | 9.627383028 | 9.627383037 | 1.17689462e-05 | 9.037270488e-09 |

## 표 4

| point_id | coefficient_round | scenario | solver | status | avg_rank_lt_5_count | peak_rank_lt_5_count | avg_psd_violation_count | peak_psd_violation_count | peak_sign_bad_count | peak_discharge_count | peak_loss_reduction_bad_count | peak_loss_reduction_min | hessian_projection_count |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1_old_full_opt | round0 | summer | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0.0005911646375 | 0 |
| P1_old_full_opt | round0 | winter | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0.0005911646375 | 0 |
| P1_old_full_opt | round0 | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0.0005911646375 | 0 |
| P1_old_full_opt | round0 | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0.0005911646375 | 0 |
| P1_old_full_opt | round0 | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0.0005911646375 | 0 |
| P1_old_full_opt | round1 | summer | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0.0005879257646 | 0 |
| P1_old_full_opt | round1 | winter | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0.0005879257646 | 0 |
| P1_old_full_opt | round1 | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0.0005879257646 | 0 |
| P1_old_full_opt | round1 | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0.0005879257646 | 0 |
| P1_old_full_opt | round1 | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0.0005879257646 | 0 |
| P1_old_full_opt | round2 | summer | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0.0005879331939 | 0 |
| P1_old_full_opt | round2 | winter | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0.0005879331939 | 0 |
| P1_old_full_opt | round2 | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0.0005879331939 | 0 |
| P1_old_full_opt | round2 | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0.0005879331939 | 0 |
| P1_old_full_opt | round2 | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0.0005879331939 | 0 |
| P2_dev_run1 | round0 | summer | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0.0007211770011 | 0 |
| P2_dev_run1 | round0 | winter | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0.0007211770011 | 0 |
| P2_dev_run1 | round0 | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0.0007211770011 | 0 |
| P2_dev_run1 | round0 | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0.0007211770011 | 0 |
| P2_dev_run1 | round0 | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0.0007211770011 | 0 |
| P2_dev_run1 | round1 | summer | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0.0007330718911 | 0 |
| P2_dev_run1 | round1 | winter | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0.0007330718911 | 0 |
| P2_dev_run1 | round1 | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0.0007330718911 | 0 |
| P2_dev_run1 | round1 | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0.0007330718911 | 0 |
| P2_dev_run1 | round1 | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0.0007330718911 | 0 |
| P2_dev_run1 | round2 | summer | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0.0007331245302 | 0 |
| P2_dev_run1 | round2 | winter | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0.0007331245302 | 0 |
| P2_dev_run1 | round2 | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0.0007331245302 | 0 |
| P2_dev_run1 | round2 | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0.0007331245302 | 0 |
| P2_dev_run1 | round2 | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0.0007331245302 | 0 |
| P3_dev_run0 | round0 | summer | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0.005249033513 | 0 |
| P3_dev_run0 | round0 | winter | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0.005249033513 | 0 |
| P3_dev_run0 | round0 | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0.005249033513 | 0 |
| P3_dev_run0 | round0 | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0.005249033513 | 0 |
| P3_dev_run0 | round0 | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0.005249033513 | 0 |
| P3_dev_run0 | round1 | summer | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0.005243696034 | 0 |
| P3_dev_run0 | round1 | winter | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0.005243696034 | 0 |
| P3_dev_run0 | round1 | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0.005243696034 | 0 |
| P3_dev_run0 | round1 | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0.005243696034 | 0 |
| P3_dev_run0 | round1 | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0.005243696034 | 0 |
| P3_dev_run0 | round2 | summer | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0.005243697396 | 0 |
| P3_dev_run0 | round2 | winter | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0.005243697396 | 0 |
| P3_dev_run0 | round2 | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0.005243697396 | 0 |
| P3_dev_run0 | round2 | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0.005243697396 | 0 |
| P3_dev_run0 | round2 | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0.005243697396 | 0 |
| P4_normal | round0 | summer | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0.0005868211463 | 0 |
| P4_normal | round0 | winter | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0.0005868211463 | 0 |
| P4_normal | round0 | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0.0005868211463 | 0 |
| P4_normal | round0 | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0.0005868211463 | 0 |
| P4_normal | round0 | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0.0005868211463 | 0 |
| P4_normal | round1 | summer | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0.000583942756 | 0 |
| P4_normal | round1 | winter | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0.000583942756 | 0 |
| P4_normal | round1 | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0.000583942756 | 0 |
| P4_normal | round1 | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0.000583942756 | 0 |
| P4_normal | round1 | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0.000583942756 | 0 |
| P4_normal | round2 | summer | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0.0005839500178 | 0 |
| P4_normal | round2 | winter | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0.0005839500178 | 0 |
| P4_normal | round2 | shoulder | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0.0005839500178 | 0 |
| P4_normal | round2 | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0.0005839500178 | 0 |
| P4_normal | round2 | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0.0005839500178 | 0 |

## 표 5

| point_id | b_energy_round0 | b_energy_round1 | b_energy_round2 | delta_b_energy_1_minus_0 | delta_b_energy_2_minus_1 | max_abs_delta_b_energy | max_abs_delta_over_abs_round0 | rounds_isclose_rtol_1e_9 |
|---|---|---|---|---|---|---|---|---|
| P1_old_full_opt | 1467921.783 | 1467921.783 | 1467921.783 | -1.174630597e-06 | -1.554843038e-06 | 1.554843038e-06 | 1.05921382e-12 | True |
| P2_dev_run1 | -70901.34784 | -70901.41442 | -70901.41442 | -0.06657526165 | 1.511943992e-06 | 0.06657526165 | 9.389844294e-07 | False |
| P3_dev_run0 | -6870867.6 | -6870867.596 | -6870867.596 | 0.004251507111 | -2.274289727e-06 | 0.004251507111 | 6.187729641e-10 | True |
| P4_normal | 1435508.742 | 1435508.742 | 1435508.742 | -2.666842192e-06 | 8.281785995e-07 | 2.666842192e-06 | 1.857767991e-12 | True |

## 환경

| item | value |
|---|---|
| timestamp | 2026-07-30T16:35:42 |
| hostname | PSL |
| platform | Windows-10-10.0.26200-SP0 |
| python | 3.11.15 |
| numpy | 2.2.6 |
| pandapower | 3.2.1 |
| installed_solvers | CLARABEL,SCS,SCIPY,HIGHS,OSQP |
| POLY_N | 128 |
| AVG_coefficient_round | P=0 fixed |
| PEAK_coefficient_rounds | P=0,P_round0,P_round1 |
| runpp_calls_total | 22506 |
| pf_retry_events | 0 |
| pf_diverged_calls | 0 |
| measured_elapsed_s | 109.368886 |
| runpp_calls_per_s | 205.7806459 |
| total_execution_time_s | 109.7089827 |
| hessian_projection_count | 0 |
| load_p_sum_mw | 9.5 |
| load_q_sum_mvar | 3.122498999 |
| load_s_sum_mva | 10 |
| load_total_pf | 0.95 |
| K_P | 2.557200538 |
| K_Q | 1.357608261 |
| ETA_PCS | 0.975 |
| SLACK_VM_PU | 1.02 |
| baseline_runpp_no_retry | 120 |
| p0_coefficient_groups | 480 |
| p0_feasible_grid_rows_per_time_all_points | 100 |
| p0_surface_runpp_exact_no_retry | 11520 |
| peak_remeasurement_groups | 384 |
| peak_remeasurement_runpp_upper_no_retry | 9600 |
| schedule_ac_runpp_no_retry | 1440 |
| total_runpp_upper_no_retry | 22680 |
| expected_seconds_at_200_calls_per_s | 113.4 |
| solver_CLARABEL_version | 0.11.1 |
| solver_SCS_version | 3.2.11 |
