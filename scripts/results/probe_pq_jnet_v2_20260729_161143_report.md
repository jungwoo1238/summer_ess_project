# probe_pq_jnet_v2

## 표 1

| point_id | j_net_A0 | j_net_A | j_net_B | delta_A_minus_A0 | delta_A_minus_A0_sign | delta_b_energy_A_minus_A0 | delta_b_defer_A_minus_A0 | delta_B_minus_A | delta_B_minus_A_sign | delta_b_energy_B_minus_A | delta_b_defer_B_minus_A | cost | capex | opex | delta_definition |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1_old_full_opt | 3673675.474 | 3833370.988 | 3899001.095 | 159695.514 | positive | -0.0003397362307 | 159695.5144 | 65630.10707 | positive | 20838.67429 | 44791.43278 | 10507672.44 | 98413550.91 | 2942032.297 | A-A0;B-A |
| P2_dev_run1 | 3039905.686 | 3408083.253 | 3424679.967 | 368177.5667 | positive | 3.550667316e-06 | 368177.5667 | 16596.71386 | positive | 2946.01901 | 13650.69485 | 10533065.17 | 99761358.22 | 2863810.997 | A-A0;B-A |
| P3_dev_run0 | 582430.3967 | 2066340.289 | 2021573.89 | 1483909.893 | positive | 6.367452443e-06 | 1483909.893 | -44766.39923 | negative | 2613.656026 | -47380.05526 | 12792119.29 | 127094442 | 3021606.975 | A-A0;B-A |
| P4_normal | 3708125.264 | 3897412.062 | 3925695.962 | 189286.7976 | positive | 4.528090358e-06 | 189286.7975 | 28283.9004 | positive | 24273.29534 | 4010.605059 | 10340984.22 | 96876853.72 | 2893479.214 | A-A0;B-A |

## 표 2

| point_id | A0_summer_peak_mw | A_summer_peak_mw | B_summer_peak_mw | A0_winter_peak_mw | A_winter_peak_mw | B_winter_peak_mw | A0_annual_peak_mw | A_annual_peak_mw | B_annual_peak_mw | delta_A_minus_A0_peak_mw | delta_B_minus_A_peak_mw |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P1_old_full_opt | 8.648775031 | 8.646676044 | 8.64608732 | 8.358411295 | 8.358243319 | 8.358197118 | 8.648775031 | 8.646676044 | 8.64608732 | -0.002098986548 | -0.0005887242059 |
| P2_dev_run1 | 8.652473876 | 8.647634668 | 8.647455248 | 8.342403105 | 8.336511025 | 8.336924361 | 8.652473876 | 8.647634668 | 8.647455248 | -0.004839207682 | -0.0001794203486 |
| P3_dev_run0 | 8.65427757 | 8.634773535 | 8.635396284 | 8.342624688 | 8.324884372 | 8.326152661 | 8.65427757 | 8.634773535 | 8.635396284 | -0.01950403501 | 0.0006227482284 |
| P4_normal | 8.650242422 | 8.647754497 | 8.647701783 | 8.358411252 | 8.358243335 | 8.358197456 | 8.650242422 | 8.647754497 | 8.647701783 | -0.00248792487 | -5.271410473e-05 |

## 표 3

| point_id | pair | day_group | max_abs_delta_P_mw | max_abs_delta_Q_mvar | sum_abs_delta_P_mwh |
|---|---|---|---|---|---|
| P1_old_full_opt | A_minus_A0 | AVG | 0 | 0 | 0 |
| P1_old_full_opt | A_minus_A0 | PEAK | 0.003828290074 | 0.1562535225 | 0.01289328361 |
| P1_old_full_opt | B_minus_A | AVG | 0.01722502262 | 0.00766022723 | 0.1424718002 |
| P1_old_full_opt | B_minus_A | PEAK | 0.01343604094 | 0.05807772005 | 0.09871295241 |
| P2_dev_run1 | A_minus_A0 | AVG | 0 | 0 | 0 |
| P2_dev_run1 | A_minus_A0 | PEAK | 0.01005128052 | 0.2910313789 | 0.08426066972 |
| P2_dev_run1 | B_minus_A | AVG | 0.02229006209 | 0.01910817257 | 0.1799572147 |
| P2_dev_run1 | B_minus_A | PEAK | 0.02546505496 | 0.1692081695 | 0.1381648739 |
| P3_dev_run0 | A_minus_A0 | AVG | 0 | 0 | 0 |
| P3_dev_run0 | A_minus_A0 | PEAK | 0.06873735174 | 1.055218665 | 0.4103390602 |
| P3_dev_run0 | B_minus_A | AVG | 0.08603952351 | 0.02962932709 | 0.2094638565 |
| P3_dev_run0 | B_minus_A | PEAK | 0.01448362873 | 0.1584968123 | 0.1045936366 |
| P4_normal | A_minus_A0 | AVG | 0 | 0 | 0 |
| P4_normal | A_minus_A0 | PEAK | 0.003777992973 | 0.1572151055 | 0.01505958851 |
| P4_normal | B_minus_A | AVG | 0.01714192923 | 0.008126156928 | 0.143602803 |
| P4_normal | B_minus_A | PEAK | 0.01356230702 | 0.08899534142 | 0.0955683947 |

## 표 4

| point_id | method | scenario | solver | status | hessian_projection_count | n_a_P_nonnegative | n_a_Q_nonnegative | peak_sign_check_passed |
|---|---|---|---|---|---|---|---|---|
| P1_old_full_opt | A0_no_peak_loss | summer | CLARABEL | optimal | 0 |  |  |  |
| P1_old_full_opt | A0_no_peak_loss | winter | CLARABEL | optimal | 0 |  |  |  |
| P1_old_full_opt | A0_no_peak_loss | shoulder | CLARABEL | optimal | 0 |  |  |  |
| P1_old_full_opt | A0_no_peak_loss | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | True |
| P1_old_full_opt | A0_no_peak_loss | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | True |
| P1_old_full_opt | A_peak_Q | summer | CLARABEL | optimal | 0 |  |  |  |
| P1_old_full_opt | A_peak_Q | winter | CLARABEL | optimal | 0 |  |  |  |
| P1_old_full_opt | A_peak_Q | shoulder | CLARABEL | optimal | 0 |  |  |  |
| P1_old_full_opt | A_peak_Q | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | True |
| P1_old_full_opt | A_peak_Q | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | True |
| P1_old_full_opt | B_peak_PQ | summer | CLARABEL | optimal | 0 |  |  |  |
| P1_old_full_opt | B_peak_PQ | winter | CLARABEL | optimal | 0 |  |  |  |
| P1_old_full_opt | B_peak_PQ | shoulder | CLARABEL | optimal | 0 |  |  |  |
| P1_old_full_opt | B_peak_PQ | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | True |
| P1_old_full_opt | B_peak_PQ | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | True |
| P2_dev_run1 | A0_no_peak_loss | summer | CLARABEL | optimal | 0 |  |  |  |
| P2_dev_run1 | A0_no_peak_loss | winter | CLARABEL | optimal | 0 |  |  |  |
| P2_dev_run1 | A0_no_peak_loss | shoulder | CLARABEL | optimal | 0 |  |  |  |
| P2_dev_run1 | A0_no_peak_loss | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | True |
| P2_dev_run1 | A0_no_peak_loss | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | True |
| P2_dev_run1 | A_peak_Q | summer | CLARABEL | optimal | 0 |  |  |  |
| P2_dev_run1 | A_peak_Q | winter | CLARABEL | optimal | 0 |  |  |  |
| P2_dev_run1 | A_peak_Q | shoulder | CLARABEL | optimal | 0 |  |  |  |
| P2_dev_run1 | A_peak_Q | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | True |
| P2_dev_run1 | A_peak_Q | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | True |
| P2_dev_run1 | B_peak_PQ | summer | CLARABEL | optimal | 0 |  |  |  |
| P2_dev_run1 | B_peak_PQ | winter | CLARABEL | optimal | 0 |  |  |  |
| P2_dev_run1 | B_peak_PQ | shoulder | CLARABEL | optimal | 0 |  |  |  |
| P2_dev_run1 | B_peak_PQ | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | True |
| P2_dev_run1 | B_peak_PQ | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | True |
| P3_dev_run0 | A0_no_peak_loss | summer | CLARABEL | optimal | 0 |  |  |  |
| P3_dev_run0 | A0_no_peak_loss | winter | CLARABEL | optimal | 0 |  |  |  |
| P3_dev_run0 | A0_no_peak_loss | shoulder | CLARABEL | optimal | 0 |  |  |  |
| P3_dev_run0 | A0_no_peak_loss | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | True |
| P3_dev_run0 | A0_no_peak_loss | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | True |
| P3_dev_run0 | A_peak_Q | summer | CLARABEL | optimal | 0 |  |  |  |
| P3_dev_run0 | A_peak_Q | winter | CLARABEL | optimal | 0 |  |  |  |
| P3_dev_run0 | A_peak_Q | shoulder | CLARABEL | optimal | 0 |  |  |  |
| P3_dev_run0 | A_peak_Q | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | True |
| P3_dev_run0 | A_peak_Q | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | True |
| P3_dev_run0 | B_peak_PQ | summer | CLARABEL | optimal | 0 |  |  |  |
| P3_dev_run0 | B_peak_PQ | winter | CLARABEL | optimal | 0 |  |  |  |
| P3_dev_run0 | B_peak_PQ | shoulder | CLARABEL | optimal | 0 |  |  |  |
| P3_dev_run0 | B_peak_PQ | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | True |
| P3_dev_run0 | B_peak_PQ | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | True |
| P4_normal | A0_no_peak_loss | summer | CLARABEL | optimal | 0 |  |  |  |
| P4_normal | A0_no_peak_loss | winter | CLARABEL | optimal | 0 |  |  |  |
| P4_normal | A0_no_peak_loss | shoulder | CLARABEL | optimal | 0 |  |  |  |
| P4_normal | A0_no_peak_loss | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | True |
| P4_normal | A0_no_peak_loss | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | True |
| P4_normal | A_peak_Q | summer | CLARABEL | optimal | 0 |  |  |  |
| P4_normal | A_peak_Q | winter | CLARABEL | optimal | 0 |  |  |  |
| P4_normal | A_peak_Q | shoulder | CLARABEL | optimal | 0 |  |  |  |
| P4_normal | A_peak_Q | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | True |
| P4_normal | A_peak_Q | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | True |
| P4_normal | B_peak_PQ | summer | CLARABEL | optimal | 0 |  |  |  |
| P4_normal | B_peak_PQ | winter | CLARABEL | optimal | 0 |  |  |  |
| P4_normal | B_peak_PQ | shoulder | CLARABEL | optimal | 0 |  |  |  |
| P4_normal | B_peak_PQ | summer_peak | CLARABEL | optimal | 0 | 0 | 0 | True |
| P4_normal | B_peak_PQ | winter_peak | CLARABEL | optimal | 0 | 0 | 0 | True |

## 환경

| item | value |
|---|---|
| timestamp | 2026-07-29T16:12:28 |
| hostname | PSL |
| platform | Windows-10-10.0.26200-SP0 |
| python | 3.11.15 |
| numpy | 2.2.6 |
| pandapower | 3.2.1 |
| cvxpy | 1.9.2 |
| installed_solvers | CLARABEL,SCS,SCIPY,HIGHS,OSQP |
| POLY_N | 128 |
| avg_coef_csv | C:\Users\PowerSL\summer_ess_project\scripts\results\probe_pq_loss_v2_20260729_145353_coef.csv |
| peak_coef_csv | C:\Users\PowerSL\summer_ess_project\scripts\results\probe_pq_jnet_20260729_152807_coef.csv |
| expected_schedule_runpp_calls | 1440 |
| schedule_runpp_calls | 1440 |
| pf_retry_events | 0 |
| pf_diverged_calls | 0 |
| ac_elapsed_s | 18.482014900 |
| runpp_calls_per_ac_s | 77.913582896 |
| total_execution_time_s | 45.292163000 |
| peak_constraint_A0 | pk>=load-P |
| peak_constraint_A | pk>=load-P-(-a_Q*Q)=load-P+a_Q*Q |
| peak_constraint_B | pk>=load-P-(-a_Q*Q-a_P*P)=load-P+a_Q*Q+a_P*P |
| delta_B_minus_A_signs_mixed | True |
| hessian_projection_count | 0 |
| baseline_source | evaluate._get_base_flow |
| benefits_signature_b_energy | (p_slack_base, p_slack_ess, smp_mwh, n_weekdays) |
| benefits_signature_b_defer | (p_slack_base, p_slack_ess) |
| benefits_signature_capex | (s_mva, e_mwh) |
| benefits_signature_opex | (s_mva, e_mwh) |
| benefits_signature_total_cost | (s_mva, e_mwh) |
| benefits_signature_j_net | (b_energy_val, b_defer_val, s_mva, e_mwh) |
| solver_CLARABEL_version | 0.11.1 |
| solver_SCS_version | 3.2.11 |
| solver_OSQP_version | 1.1.3 |
| solver_SCIPY_version | 1.13.1 |
| solver_HIGHS_version | 1.15.1 |
