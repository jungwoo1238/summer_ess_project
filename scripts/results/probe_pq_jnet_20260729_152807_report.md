# probe_pq_jnet

## 표 1

| point_id | j_net_A_won_per_year | j_net_B_won_per_year | delta_j_net_B_minus_A_won_per_year | delta_j_net_sign | delta_b_energy_B_minus_A_won_per_year | delta_b_defer_B_minus_A_won_per_year | b_energy_A | b_energy_B | b_defer_A | b_defer_B | capex | opex | cost | delta_definition |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1_old_full_opt | 2688387.284 | 2345195.006 | -343192.2781 | negative | 20838.64241 | -364030.9205 | 2731785.69 | 2752624.333 | 10464274.03 | 10100243.11 | 98413550.91 | 2942032.297 | 10507672.44 | B-A; positive=B_benefit |
| P2_dev_run1 | 1076635.957 | 722621.0166 | -354014.9407 | negative | 2946.019012 | -356960.9597 | 2404824.905 | 2407770.924 | 9204876.224 | 8847915.264 | 99761358.22 | 2863810.997 | 10533065.17 | B-A; positive=B_benefit |
| P3_dev_run0 | -7844440.255 | -8128827.933 | -284387.6781 | negative | 2613.656022 | -287001.3341 | 2343632.763 | 2346246.419 | 2604046.276 | 2317044.942 | 127094442 | 3021606.975 | 12792119.29 | B-A; positive=B_benefit |
| P4_normal | 2666374.146 | 2327281.46 | -339092.6859 | negative | 24273.29534 | -363365.9813 | 2711189.572 | 2735462.867 | 10296168.79 | 9932802.813 | 96876853.72 | 2893479.214 | 10340984.22 | B-A; positive=B_benefit |

## 표 2

| point_id | day_group | max_abs_delta_P_net_mw | max_abs_delta_Q_mvar | sum_abs_delta_P_net_mwh |
|---|---|---|---|---|
| P1_old_full_opt | AVG | 0.01722502262 | 0.00766022723 | 0.1424718002 |
| P1_old_full_opt | PEAK | 0.01914223068 | 0.09941531767 | 0.1093147669 |
| P2_dev_run1 | AVG | 0.02229006209 | 0.01910817257 | 0.1799572147 |
| P2_dev_run1 | PEAK | 0.01767243335 | 0.1225388911 | 0.1454535856 |
| P3_dev_run0 | AVG | 0.08603952351 | 0.02962932709 | 0.2094638565 |
| P3_dev_run0 | PEAK | 0.01038601419 | 0.1910266638 | 0.1171960804 |
| P4_normal | AVG | 0.01714192923 | 0.008126156928 | 0.143602803 |
| P4_normal | PEAK | 0.0197875171 | 0.06224030184 | 0.1138435118 |

## 표 3

| point_id | E_over_S | delta_j_net_B_minus_A_won_per_year | delta_b_energy_B_minus_A_won_per_year | delta_b_defer_B_minus_A_won_per_year |
|---|---|---|---|---|
| P1_old_full_opt | 2.380681818 | -343192.2781 | 20838.64241 | -364030.9205 |
| P2_dev_run1 | 1.333333333 | -354014.9407 | 2946.019012 | -356960.9597 |
| P3_dev_run0 | 0.3875598086 | -284387.6781 | 2613.656022 | -287001.3341 |
| P4_normal | 2.340909091 | -339092.6859 | 24273.29534 | -363365.9813 |

## 표 4

| point_id | scenario | n_cases | n_rank_lt_5 | n_psd_violations | minimum_rank | minimum_lambda_min_cost | relative_max_median | relative_max_max | rmse_mw_median | rmse_mw_max |
|---|---|---|---|---|---|---|---|---|---|---|
| P1_old_full_opt | summer_peak | 24 | 0 | 0 | 5 | 0.01858266167 | 0.006504570286 | 0.006606819034 | 3.550006364e-05 | 3.982344679e-05 |
| P1_old_full_opt | winter_peak | 24 | 0 | 0 | 5 | 0.01878380562 | 0.006512243412 | 0.006576512085 | 3.441538023e-05 | 3.855030306e-05 |
| P2_dev_run1 | summer_peak | 24 | 0 | 0 | 5 | 0.02344647302 | 0.006526787974 | 0.006564381503 | 6.462568366e-05 | 7.236584412e-05 |
| P2_dev_run1 | winter_peak | 24 | 0 | 0 | 5 | 0.02375520195 | 0.006515801624 | 0.006554152943 | 6.268546719e-05 | 7.008548757e-05 |
| P3_dev_run0 | summer_peak | 24 | 0 | 0 | 5 | 0.01080310775 | 0.004601372867 | 0.005239867599 | 0.0001210267802 | 0.0001352420798 |
| P3_dev_run0 | winter_peak | 24 | 0 | 0 | 5 | 0.01082052871 | 0.004646340207 | 0.005040992243 | 0.0001174609577 | 0.0001310556706 |
| P4_normal | summer | 24 | 0 | 0 | 5 | 0.01844676683 | 0.00653703814 | 0.006627305345 | 3.129451825e-05 | 3.45450068e-05 |
| P4_normal | winter | 24 | 0 | 0 | 5 | 0.01862177414 | 0.006538107172 | 0.006600427708 | 3.117490169e-05 | 3.463884001e-05 |
| P4_normal | shoulder | 24 | 0 | 0 | 5 | 0.01833177023 | 0.006587584384 | 0.006627158616 | 2.651864493e-05 | 2.942254504e-05 |
| P4_normal | summer_peak | 24 | 0 | 0 | 5 | 0.01858266167 | 0.006504570284 | 0.006606819026 | 3.550006363e-05 | 3.982344678e-05 |
| P4_normal | winter_peak | 24 | 0 | 0 | 5 | 0.01878380562 | 0.006512243411 | 0.006576512088 | 3.441538023e-05 | 3.855030308e-05 |

## 표 5

| point_id | method | scenario | day_group | solver | status | objective | lp_pk_proxy_mw | hessian_projection_count |
|---|---|---|---|---|---|---|---|---|
| P1_old_full_opt | A_Q_only | summer | AVG | CLARABEL | optimal | -25854.64291 | nan | 0 |
| P1_old_full_opt | A_Q_only | winter | AVG | CLARABEL | optimal | -29731.11084 | nan | 0 |
| P1_old_full_opt | A_Q_only | shoulder | AVG | CLARABEL | optimal | -24611.16347 | nan | 0 |
| P1_old_full_opt | A_Q_only | summer_peak | PEAK | CLARABEL | optimal | 8.356207898 | 8.356207898 | 0 |
| P1_old_full_opt | A_Q_only | winter_peak | PEAK | CLARABEL | optimal | 8.092432899 | 8.092432899 | 0 |
| P1_old_full_opt | B_PQ_full | summer | AVG | CLARABEL | optimal | -26832.67558 | nan | 0 |
| P1_old_full_opt | B_PQ_full | winter | AVG | CLARABEL | optimal | -30791.1013 | nan | 0 |
| P1_old_full_opt | B_PQ_full | shoulder | AVG | CLARABEL | optimal | -25275.43661 | nan | 0 |
| P1_old_full_opt | B_PQ_full | summer_peak | PEAK | CLARABEL | optimal | 8.366688012 | 8.366688012 | 0 |
| P1_old_full_opt | B_PQ_full | winter_peak | PEAK | CLARABEL | optimal | 8.10730161 | 8.10730161 | 0 |
| P2_dev_run1 | A_Q_only | summer | AVG | CLARABEL | optimal | -38039.52525 | nan | 0 |
| P2_dev_run1 | A_Q_only | winter | AVG | CLARABEL | optimal | -44927.74574 | nan | 0 |
| P2_dev_run1 | A_Q_only | shoulder | AVG | CLARABEL | optimal | -35650.08196 | nan | 0 |
| P2_dev_run1 | A_Q_only | summer_peak | PEAK | CLARABEL | optimal | 8.352307271 | 8.352307271 | 0 |
| P2_dev_run1 | A_Q_only | winter_peak | PEAK | CLARABEL | optimal | 8.065407609 | 8.065407609 | 0 |
| P2_dev_run1 | B_PQ_full | summer | AVG | CLARABEL | optimal | -38922.41064 | nan | 0 |
| P2_dev_run1 | B_PQ_full | winter | AVG | CLARABEL | optimal | -45943.99427 | nan | 0 |
| P2_dev_run1 | B_PQ_full | shoulder | AVG | CLARABEL | optimal | -36214.27506 | nan | 0 |
| P2_dev_run1 | B_PQ_full | summer_peak | PEAK | CLARABEL | optimal | 8.362699402 | 8.362699402 | 0 |
| P2_dev_run1 | B_PQ_full | winter_peak | PEAK | CLARABEL | optimal | 8.080471142 | 8.080471142 | 0 |
| P3_dev_run0 | A_Q_only | summer | AVG | CLARABEL | optimal | -117165.1423 | nan | 0 |
| P3_dev_run0 | A_Q_only | winter | AVG | CLARABEL | optimal | -143018.6071 | nan | 0 |
| P3_dev_run0 | A_Q_only | shoulder | AVG | CLARABEL | optimal | -106905.7896 | nan | 0 |
| P3_dev_run0 | A_Q_only | summer_peak | PEAK | CLARABEL | optimal | 8.303688073 | 8.303688073 | 0 |
| P3_dev_run0 | A_Q_only | winter_peak | PEAK | CLARABEL | optimal | 8.017119144 | 8.017119144 | 0 |
| P3_dev_run0 | B_PQ_full | summer | AVG | CLARABEL | optimal | -118007.4857 | nan | 0 |
| P3_dev_run0 | B_PQ_full | winter | AVG | CLARABEL | optimal | -143870.579 | nan | 0 |
| P3_dev_run0 | B_PQ_full | shoulder | AVG | CLARABEL | optimal | -107356.4216 | nan | 0 |
| P3_dev_run0 | B_PQ_full | summer_peak | PEAK | CLARABEL | optimal | 8.312625544 | 8.312625544 | 0 |
| P3_dev_run0 | B_PQ_full | winter_peak | PEAK | CLARABEL | optimal | 8.030084508 | 8.030084508 | 0 |
| P4_normal | A_Q_only | summer | AVG | CLARABEL | optimal | -24956.22363 | nan | 0 |
| P4_normal | A_Q_only | winter | AVG | CLARABEL | optimal | -28670.33167 | nan | 0 |
| P4_normal | A_Q_only | shoulder | AVG | CLARABEL | optimal | -23760.39692 | nan | 0 |
| P4_normal | A_Q_only | summer_peak | PEAK | CLARABEL | optimal | 8.358097285 | 8.358097285 | 0 |
| P4_normal | A_Q_only | winter_peak | PEAK | CLARABEL | optimal | 8.092432921 | 8.092432921 | 0 |
| P4_normal | B_PQ_full | summer | AVG | CLARABEL | optimal | -25932.5054 | nan | 0 |
| P4_normal | B_PQ_full | winter | AVG | CLARABEL | optimal | -29749.19795 | nan | 0 |
| P4_normal | B_PQ_full | shoulder | AVG | CLARABEL | optimal | -24427.51254 | nan | 0 |
| P4_normal | B_PQ_full | summer_peak | PEAK | CLARABEL | optimal | 8.368398611 | 8.368398611 | 0 |
| P4_normal | B_PQ_full | winter_peak | PEAK | CLARABEL | optimal | 8.107301607 | 8.107301607 | 0 |

## 환경

| item | value |
|---|---|
| timestamp | 2026-07-29T15:28:56 |
| hostname | PSL |
| platform | Windows-10-10.0.26200-SP0 |
| python | 3.11.15 |
| numpy | 2.2.6 |
| pandapower | 3.2.1 |
| cvxpy | 1.9.2 |
| installed_solvers | CLARABEL,SCS,SCIPY,HIGHS,OSQP |
| POLY_N | 128 |
| input_avg_coef_csv | C:\Users\PowerSL\summer_ess_project\scripts\results\probe_pq_loss_v2_20260729_145353_coef.csv |
| new_coef_csv | C:\Users\PowerSL\summer_ess_project\scripts\results\probe_pq_jnet_20260729_152807_coef.csv |
| reused_v2_coef_cases | 216 |
| new_surface_runpp_calls | 8448 |
| expected_runpp_calls_no_retry | 9528 |
| runpp_calls_total | 9528 |
| pf_retry_events | 0 |
| pf_diverged_calls | 0 |
| ac_phase_elapsed_s | 41.361980000 |
| runpp_calls_per_ac_phase_s | 230.356477132 |
| total_execution_time_s | 49.051052100 |
| delta_j_net_definition | j_net_B-j_net_A;positive=B_benefit |
| peak_linear_loss_definition | pk>=load_total-P-a_Q*Q-[B:a_P*P] |
| delta_j_net_signs_mixed | False |
| hessian_projection_count | 0 |
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
