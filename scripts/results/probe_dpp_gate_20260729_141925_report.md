# probe_dpp_gate

## 표 1

| method | expression | is_dcp | is_dcp_dpp_true | linear_term_is_dpp | square_q_is_dpp | weighted_quadratic_is_dpp | loss_term_is_dpp | objective_expr_is_dpp |
|---|---|---|---|---|---|---|---|---|
| M0 | none | True | True | None | True | None | None | True |
| M1 | none | True | True | True | True | None | True | True |
| M2 | square | True | True | True | True | True | True | True |
| M2 | power | True | True | True | True | True | True | True |
| M3 | square | True | True | True | True | True | True | True |
| M3 | power | True | True | True | True | True | True | True |

## 환경

| item | value |
|---|---|
| timestamp | 2026-07-29T14:19:25 |
| hostname | PSL |
| platform | Windows-10-10.0.26200-SP0 |
| python | 3.11.15 |
| cvxpy | 1.9.2 |
| installed_solvers | CLARABEL,SCS,SCIPY,HIGHS,OSQP |
| params_poly_n | 32 |
| total_time_s | 0.470575300 |
| solver_CLARABEL_version | 0.11.1 |
| solver_OSQP_version | 1.1.3 |
| solver_SCIPY_version | 1.13.1 |
| solver_SCS_version | 3.2.11 |
| solver_HIGHS_version | 1.15.1 |
