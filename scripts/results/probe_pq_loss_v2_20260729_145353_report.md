# probe_pq_loss_v2

## 표 1

| point_id | scenario | n_cases | n_psd_violations | psd_violation_pct | minimum_lambda_min_cost | minimum_lambda_min_over_lambda_max_pct | psd_text |
|---|---|---|---|---|---|---|---|
| P1_old_full_opt | summer | 24 | 0 | 0 | 0.01829434337 | nan | 전 케이스 PSD |
| P1_old_full_opt | winter | 24 | 0 | 0 | 0.01845318184 | nan | 전 케이스 PSD |
| P1_old_full_opt | shoulder | 24 | 0 | 0 | 0.01818973098 | nan | 전 케이스 PSD |
| P2_dev_run1 | summer | 24 | 0 | 0 | 0.02248539462 | nan | 전 케이스 PSD |
| P2_dev_run1 | winter | 24 | 0 | 0 | 0.02269821991 | nan | 전 케이스 PSD |
| P2_dev_run1 | shoulder | 24 | 0 | 0 | 0.02234526761 | nan | 전 케이스 PSD |
| P3_dev_run0 | summer | 24 | 0 | 0 | 0.0125385459 | nan | 전 케이스 PSD |
| P3_dev_run0 | winter | 24 | 0 | 0 | 0.01264068896 | nan | 전 케이스 PSD |
| P3_dev_run0 | shoulder | 24 | 0 | 0 | 0.01247099549 | nan | 전 케이스 PSD |

## 표 1b

| point_id | n_cases | n_rank_lt_5 | rank_lt_5_pct | minimum_rank | minimum_unique_p | minimum_unique_q |
|---|---|---|---|---|---|---|
| P1_old_full_opt | 72 | 0 | 0 | 5 | 9 | 17 |
| P2_dev_run1 | 72 | 0 | 0 | 5 | 9 | 17 |
| P3_dev_run0 | 72 | 0 | 0 | 5 | 3 | 9 |

## 표 2

| point_id | n_cases | relative_max_min | relative_max_median | relative_max_p95 | relative_max_max | rmse_mw_min | rmse_mw_median | rmse_mw_p95 | rmse_mw_max |
|---|---|---|---|---|---|---|---|---|---|
| P1_old_full_opt | 72 | 0.0001424288651 | 0.0001622256798 | 0.0001918303601 | 0.0001970701547 | 5.667149205e-07 | 6.114374382e-07 | 6.524538746e-07 | 6.549018825e-07 |
| P2_dev_run1 | 72 | 0.0006210651249 | 0.0006919854876 | 0.0007927964929 | 0.0008105418332 | 4.388317972e-06 | 4.735523762e-06 | 5.057410001e-06 | 5.07593685e-06 |
| P3_dev_run0 | 72 | 0.000398995175 | 0.0004695370237 | 0.0005857331588 | 0.00060845895 | 1.06493222e-05 | 1.1480421e-05 | 1.225163501e-05 | 1.229604496e-05 |

## 표 3

| form | expression_is_dcp | expression_is_dpp | problem_is_dcp | problem_is_dcp_dpp_true |
|---|---|---|---|---|
| a_quad_form_H_parameter_PSD | True | False | True | False |
| b_separate_parameter_terms | False | False | False | False |
| c_quad_form_H_constant | True | True | True | True |

## 표 4

| point_id | force_q_zero | max_abs_delta_p_net_mw | sum_abs_delta_p_net_mwh | abs_objective_difference | objective_no_loss | objective_with_loss | status_no_loss | status_with_loss | solver_no_loss | solver_with_loss | hessian_projected_count | coefficient_unavailable_count | non_psd_hessian_count | delta_j_net_won_per_year |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1_old_full_opt | False | 0.1044017149 | 1.245622712 | 59150.40398 | -23748.80951 | -82899.21349 | optimal | optimal | CLARABEL | CLARABEL | 0 | 0 | 0 | 620625.9826 |
| P2_dev_run1 | False | 0.1267111285 | 1.190379031 | 96738.68583 | -24341.99414 | -121080.68 | optimal | optimal | CLARABEL | CLARABEL | 0 | 0 | 0 | 255104.2467 |
| P3_dev_run0 | False | 0.188632678 | 0.6564985063 | 344340.4111 | -24894.07529 | -369234.4864 | optimal | optimal | CLARABEL | CLARABEL | 0 | 0 | 0 | -176888.7521 |

## 표 5

| point_id | force_q_zero | max_abs_delta_p_net_mw | sum_abs_delta_p_net_mwh | abs_objective_difference | objective_no_loss | objective_with_loss | status_no_loss | status_with_loss | solver_no_loss | solver_with_loss | hessian_projected_count | coefficient_unavailable_count | non_psd_hessian_count |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1_old_full_opt | True | 0.05397605675 | 0.2209049395 | 2584.65047 | -23751.28096 | -26335.93143 | optimal | optimal | CLARABEL | CLARABEL | 0 | 0 | 0 |
| P2_dev_run1 | True | 0.109287689 | 0.4403561915 | 2100.196162 | -24342.76981 | -26442.96597 | optimal | optimal | CLARABEL | CLARABEL | 0 | 0 | 0 |
| P3_dev_run0 | True | 0.1885875806 | 0.4966504521 | 2175.524451 | -24894.04272 | -27069.56717 | optimal | optimal | CLARABEL | CLARABEL | 0 | 0 | 0 |

## 환경

| item | value |
|---|---|
| timestamp | 2026-07-29T14:54:28 |
| hostname | PSL |
| platform | Windows-10-10.0.26200-SP0 |
| python | 3.11.15 |
| numpy | 2.2.6 |
| pandapower | 3.2.1 |
| cvxpy | 1.9.2 |
| installed_solvers | CLARABEL,SCS,SCIPY,HIGHS,OSQP |
| POLY_N | 128 |
| N_Q_ADAPTIVE | 5 |
| baseline_definition | loss_p0_q0_same_point_scenario_time |
| grid_definition | p=S*[-1,-.75,-.5,-.25,0,.25,.5,.75,1];q=unique(linspace(0,sqrt(max(S^2-p^2,0)),5)) |
| feasibility_definition | pcs_circle_and_1h_using_(soc_max-soc_min)*E |
| reusable_source |  |
| grid_rows | 7992 |
| feasible_grid_rows | 6408 |
| rows_reused | 0 |
| rows_measured | 6408 |
| rows_structural_infeasible | 1584 |
| rows_pf_diverged | 0 |
| runpp_calls_total | 6840 |
| pf_retry_events | 0 |
| pf_diverged_calls | 0 |
| measurement_elapsed_s | 29.817403500 |
| measurement_runpp_calls | 6408 |
| runpp_calls_per_measurement_s | 214.908048583 |
| total_execution_time_s | 34.970484100 |
| hessian_projection_count | 0 |
| solver_CLARABEL_version | 0.11.1 |
| solver_SCS_version | 3.2.11 |
| solver_OSQP_version | 1.1.3 |
| solver_SCIPY_version | 1.13.1 |
| solver_HIGHS_version | 1.15.1 |
