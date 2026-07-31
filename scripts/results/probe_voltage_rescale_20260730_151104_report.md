# probe_voltage_rescale

## Table 0

| built_net_p_sum_before_inverse_mw | built_net_q_sum_before_inverse_mvar | original_p_sum_after_inverse_mw | original_q_sum_after_inverse_mvar | K_P_applied | K_Q_applied | K_P_formula_unrounded | K_Q_formula_unrounded | applied_p_sum_mw | target_p_sum_mw | p_sum_error_mw | applied_q_sum_mvar | target_q_sum_mvar | q_sum_error_mvar | applied_s_sum_mva | target_s_sum_mva | s_sum_error_mva | applied_total_pf | target_total_pf | pf_error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8.50240905 | 5.263941 | 3.715 | 2.3 | 2.557201 | 1.357609 | 2.55720053836 | 1.35760826052 | 9.500001715 | 9.5 | 1.71499999979e-06 | 3.1225007 | 3.1224989992 | 1.70080080109e-06 | 10.0000021603 | 10 | 2.16032493761e-06 | 0.949999966269 | 0.95 | -3.37308617704e-08 |

## Table 1

| load_case | vm_pu | vmin_pu | vmin_bus | vmin_scenario | vmin_t | lower_violation_l1_pu | vmax_pu | vmax_bus | vmax_scenario | vmax_t | upper_violation_l1_pu | total_line_loss_kw_sum | mean_line_loss_kw | max_line_loss_kw | max_line_utilization_pu | max_util_line | max_util_scenario | max_util_t | converged_cases | diverged_cases |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| legacy_reference | 1.02 | 0.96199 |  |  |  | 0 |  |  |  |  |  |  |  |  |  |  |  |  | 120 | 0 |
| new_pf_0p95 | 1.02 | 0.964080638619 | 17 | summer_peak | 18 | 0 | 1.01884562547 | 1 | shoulder | 4 | 0 | 21388.4694546 | 178.237245455 | 281.246587941 | 0.62252033085 | 0 | summer_peak | 18 | 120 | 0 |
| new_pf_0p95 | 1.03 | 0.974678530049 | 17 | summer_peak | 18 | 0 | 1.02885727397 | 1 | shoulder | 4 | 0 | 20946.1628405 | 174.551357005 | 275.338144043 | 0.61606696076 | 0 | summer_peak | 18 | 120 | 0 |
| new_pf_0p95 | 1.04 | 0.985263148879 | 17 | summer_peak | 18 | 0 | 1.03886868511 | 1 | shoulder | 4 | 0 | 20517.754078 | 170.981283984 | 269.619303343 | 0.609750730566 | 0 | summer_peak | 18 | 120 | 0 |
| new_pf_0p95 | 1.05 | 0.99583495222 | 17 | summer_peak | 18 | 0 | 1.04887986621 | 1 | shoulder | 4 | 0 | 20102.6536651 | 167.522113876 | 264.081811261 | 0.603567158718 | 0 | summer_peak | 18 | 120 | 0 |
| new_pf_0p95 | 1.06 | 1.00639437559 | 17 | summer_peak | 18 | 0 | 1.05889082429 | 1 | shoulder | 4 | 4.43958794617 | 19700.3037195 | 164.169197662 | 258.717869691 | 0.5975119637 | 0 | summer_peak | 18 | 120 | 0 |

## Table 2

| minimum_vm_pu_with_zero_lower_violation | vmax_pu_at_minimum_vm | upper_margin_1p05_minus_vmax_pu | upper_violation_l1_pu | zero_lower_violation_found |
|---|---|---|---|---|
| 1.02 | 1.01884562547 | 0.0311543745325 | 0 | True |

## Environment

| item | value |
|---|---|
| python_version | 3.11.15 |
| pandapower_version | 3.2.1 |
| numpy_version | 2.2.6 |
| hostname | PSL |
| all_days_count | 5 |
| time_steps | 24 |
| non_slack_bus_count | 32 |
| power_flow_case_calls | 600 |
| diverged_case_count | 0 |
| total_execution_time_s | 6.8175681 |
| violation_definition | sum(max(0,0.95-Vbus)); non-slack buses; ALL_DAYS x 24 |
| legacy_reference_p_sum_mw | 8.502 |
| legacy_reference_q_sum_mvar | 5.264 |
| legacy_reference_total_pf | 0.850241 |
