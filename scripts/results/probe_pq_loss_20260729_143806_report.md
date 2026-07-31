# probe_pq_loss

## 표 1

| point_id | scenario | n_cases | n_psd_violations | psd_violation_pct | minimum_lambda_min | minimum_lambda_min_over_lambda_max_pct | psd_text |
|---|---|---|---|---|---|---|---|
| P1_old_full_opt | summer | 0 | 0 | nan | nan | nan |  |
| P1_old_full_opt | winter | 0 | 0 | nan | nan | nan |  |
| P1_old_full_opt | shoulder | 0 | 0 | nan | nan | nan |  |
| P2_dev_run1 | summer | 24 | 24 | 100 | -0.02634003386 | 107.8755971 |  |
| P2_dev_run1 | winter | 24 | 24 | 100 | -0.02635814589 | 108.5359599 |  |
| P2_dev_run1 | shoulder | 24 | 24 | 100 | -0.02536254568 | 107.4405617 |  |
| P3_dev_run0 | summer | 24 | 24 | 100 | -0.01492168797 | 108.8191475 |  |
| P3_dev_run0 | winter | 24 | 24 | 100 | -0.01493203706 | 109.6531694 |  |
| P3_dev_run0 | shoulder | 24 | 24 | 100 | -0.0143633139 | 108.2705365 |  |

## 표 2

| point_id | n_cases | relative_max_min | relative_max_median | relative_max_p95 | relative_max_max | rmse_mw_min | rmse_mw_median | rmse_mw_p95 | rmse_mw_max |
|---|---|---|---|---|---|---|---|---|---|
| P1_old_full_opt | 0 | nan | nan | nan | nan | nan | nan | nan | nan |
| P2_dev_run1 | 72 | 0.0005124087197 | 0.0005709615547 | 0.0006541876264 | 0.0006688371936 | 4.866362081e-06 | 5.250480317e-06 | 5.606535054e-06 | 5.627027103e-06 |
| P3_dev_run0 | 72 | 0.000150438034 | 0.0001709094225 | 0.0002023782391 | 0.0002082173407 | 2.027148725e-06 | 2.214351183e-06 | 2.389575475e-06 | 2.399707511e-06 |

## 표 3

| form | expression_is_dcp | expression_is_dpp | problem_is_dcp | problem_is_dcp_dpp_true |
|---|---|---|---|---|
| a_quad_form_psd_wrap | True | False | True | False |
| b_separate_terms | False | False | False | False |

## 표 4

| point_id | force_q_zero | max_abs_delta_p_net_mw | sum_abs_delta_p_net_mwh | abs_objective_difference | objective_no_loss | objective_with_loss | status_no_loss | status_with_loss | solver_no_loss | solver_with_loss | objective_hessian_projected_count | objective_hessian_projection_fro_sum | coefficient_unavailable_count | delta_j_net_won_per_year |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1_old_full_opt | False | nan | nan | nan | nan | nan | coefficient_unavailable | coefficient_unavailable |  |  | nan | nan | 72 | nan |
| P2_dev_run1 | False | 0.1569260209 | 1.396582494 | 96346.60616 | -24341.99415 | -120688.6003 | optimal | optimal,user_limit | CLARABEL | OSQP | 32 | 5.971475303e-11 |  | 187842.7958 |
| P3_dev_run0 | False | 0.2065553877 | 0.7126651484 | 343062.7512 | -24894.08617 | -367956.8374 | optimal | optimal | CLARABEL | OSQP | 4 | 3.297168041e-11 |  | -177328.0228 |

## 표 5

| point_id | force_q_zero | max_abs_delta_p_net_mw | sum_abs_delta_p_net_mwh | abs_objective_difference | objective_no_loss | objective_with_loss | status_no_loss | status_with_loss | solver_no_loss | solver_with_loss | objective_hessian_projected_count | objective_hessian_projection_fro_sum | coefficient_unavailable_count |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1_old_full_opt | True | nan | nan | nan | nan | nan | coefficient_unavailable | coefficient_unavailable |  |  | nan | nan | 72 |
| P2_dev_run1 | True | 0.109288473 | 0.4405283186 | 2099.672553 | -24342.77225 | -26442.4448 | optimal | optimal | CLARABEL | OSQP | 32 | 5.971475303e-11 |  |
| P3_dev_run0 | True | 0.1886088004 | 0.4995018576 | 2171.340545 | -24894.05185 | -27065.3924 | optimal,optimal_inaccurate | optimal,user_limit | CLARABEL | OSQP | 4 | 3.297168041e-11 |  |

## 환경

| item | value |
|---|---|
| timestamp | 2026-07-29T14:38:48 |
| hostname | PSL |
| platform | Windows-10-10.0.26200-SP0 |
| python | 3.11.15 |
| numpy | 2.2.6 |
| pandapower | 3.2.1 |
| cvxpy | 1.9.2 |
| installed_solvers | CLARABEL,SCS,SCIPY,HIGHS,OSQP |
| POLY_N | 128 |
| baseline_definition | loss_p0_q0_same_point_scenario_time |
| grid_definition | p=S*[-1,-.75,-.5,-.25,0,.25,.5,.75,1];q=[0,.1,.2,.3,.4] |
| feasibility_definition | pcs_circle_and_1h_using_(soc_max-soc_min)*E |
| reusable_source |  |
| rows_reused | 0 |
| rows_measured | 3960 |
| rows_structural_infeasible | 5760 |
| rows_pf_diverged | 0 |
| runpp_calls | 4248 |
| pf_retry_events | 0 |
| pf_diverged_calls | 0 |
| total_execution_time_s | 42.017398700 |
| solver_CLARABEL_version | 0.11.1 |
| solver_OSQP_version | 1.1.3 |
| solver_SCIPY_version | 1.13.1 |
| solver_SCS_version | 3.2.11 |
| solver_HIGHS_version | 1.15.1 |
