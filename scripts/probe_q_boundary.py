# Sweep the Q supply cap (s_util_cap) to test whether boundary operation is
# genuinely optimal or a flat-objective artifact.  Only the capacity polygon
# is reduced; the PCS-loss model keeps the original apparent-power geometry.
# One-off diagnostic script; production modules are not modified.

import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cvxpy as cp
import numpy as np

import benefits
import evaluate
import loss_coeffs
import lower_lp
import params as PM


# Best row (maximum j_net) in results/runs_n1_20260803_212003.csv.
X_OPT = np.array(
    [31.0, 0.21069418956369043, 0.44078383768391327], dtype=float
)
S_UTIL_CAPS = (1.0, 0.95, 0.90, 0.85, 0.80)
POLY_BINDING_REL_THRESHOLD = 0.99
REGRESSION_ATOL = 1e-5


def _build_avg_problem_with_cap(
    n,
    eta_c,
    eta_d,
    self_discharge,
    soc_init,
    soc_min,
    soc_max,
    mu_volt,
    coeffs,
    s_util_cap,
):
    """Copy of lower_lp._build_problem's avg/force_q_zero=False QP.

    The sole model change is the C4 capacity RHS named s_cap_capacity below.
    C5 (pcs_u >= lhs_k) and its objective term are unchanged.
    """
    T = PM.TIME_STEPS
    dt = PM.DT_HOURS
    H_full, h_full_min_eig = lower_lp._build_h_full(coeffs, n, T)
    topo = lower_lp._get_topology()
    D, r_pu, x_pu = topo["D"], topo["r_pu"], topo["x_pu"]
    n_bus = topo["n_bus"]
    R_MAT = np.tile(r_pu[:, None], (1, T))
    X_MAT = np.tile(x_pu[:, None], (1, T))

    P_ch = cp.Variable((n, T), nonneg=True)
    P_dis = cp.Variable((n, T), nonneg=True)
    Q = cp.Variable((n, T))
    soc = cp.Variable((n, T + 1))

    S_param = cp.Parameter(n, nonneg=True)
    E_param = cp.Parameter(n, nonneg=True)
    bus_onehot = cp.Parameter((n, n_bus))
    load_p_bus = cp.Parameter((n_bus, T))
    load_q_bus = cp.Parameter((n_bus, T))
    mu_volt_const = float(mu_volt)

    S_col = cp.reshape(S_param, (n, 1), order="C")
    E_col = cp.reshape(E_param, (n, 1), order="C")

    constraints = []

    # ---- Individual PCS active-power bounds ----
    constraints += [P_ch <= S_col, P_dis <= S_col]

    # ---- SOC recursion (absolute MWh convention) ----
    constraints += [
        soc[:, 0] == soc_init * E_param,
        soc[:, T] == soc_init * E_param,
        soc >= soc_min * E_col,
        soc <= soc_max * E_col,
    ]
    for t in range(T):
        constraints.append(
            soc[:, t + 1]
            == soc[:, t] * (1 - self_discharge)
            + eta_c * P_ch[:, t] * dt
            - P_dis[:, t] / eta_d * dt
        )

    P_net = P_dis - P_ch

    # ---- C4 capacity polygon and unchanged C5 PCS-loss epigraph ----
    pcs_u = cp.Variable((n, T))
    s_cap_capacity = (
        float(s_util_cap) * S_col * float(np.cos(np.pi / PM.POLY_N))
    )
    for k in range(PM.POLY_N):
        theta = 2.0 * np.pi * k / PM.POLY_N
        lhs_k = (
            P_net * float(np.cos(theta)) + Q * float(np.sin(theta))
        )
        constraints.append(lhs_k <= s_cap_capacity)  # C4: reduced only here
        constraints.append(pcs_u >= lhs_k)  # C5: unchanged, no shrunken cap
    pcs_loss_mw = pcs_u - P_ch - P_dis

    # ---- LinDistFlow ----
    netinj_p = (
        load_p_bus - bus_onehot.T @ P_net
    ) / PM.S_BASE_MVA
    netinj_q = (load_q_bus - bus_onehot.T @ Q) / PM.S_BASE_MVA
    P_e = D @ netinj_p
    Q_e = D @ netinj_q
    v = PM.V_SLACK_SQ - 2.0 * (
        D.T
        @ (
            cp.multiply(R_MAT, P_e)
            + cp.multiply(X_MAT, Q_e)
        )
    )
    v_nonslack = v[1:, :]
    volt_penalty = mu_volt_const * cp.sum(
        cp.pos(v_nonslack - PM.V_SQ_MAX)
        + cp.pos(PM.V_SQ_MIN - v_nonslack)
    )

    params = dict(
        S=S_param,
        E=E_param,
        bus_onehot=bus_onehot,
        load_p_bus=load_p_bus,
        load_q_bus=load_q_bus,
    )
    varset = dict(
        P_ch=P_ch,
        P_dis=P_dis,
        Q=Q,
        soc=soc,
        P_net=P_net,
        v_lindistflow_sq=v,
        pcs_u=pcs_u,
    )

    # ---- Original avg objective ----
    smp_param = cp.Parameter(T, nonneg=True)
    loss_cost = 0
    for t in range(T):
        z_t = cp.hstack(
            [
                component
                for i in range(n)
                for component in (P_net[i, t], Q[i, t])
            ]
        )
        linear_t = cp.sum(
            cp.multiply(coeffs["a_P"][:, t], P_net[:, t])
            + cp.multiply(coeffs["a_Q"][:, t], Q[:, t])
        )
        loss_cost += smp_param[t] * (
            linear_t + cp.quad_form(z_t, H_full[t])
        ) * dt
    objective_expr = (
        cp.sum(
            cp.multiply(
                cp.reshape(smp_param, (1, T), order="C"),
                P_ch - P_dis,
            )
        )
        * dt
        + loss_cost
        + 1e-6 * cp.sum(P_ch + P_dis)
        + volt_penalty
    )
    pcs_cost = cp.sum(
        cp.multiply(
            cp.reshape(smp_param, (1, T), order="C"),
            pcs_loss_mw,
        )
    ) * (1.0 - PM.ETA_PCS) * dt
    objective_expr += pcs_cost
    problem = cp.Problem(cp.Minimize(objective_expr), constraints)
    params["smp"] = smp_param

    assert problem.is_dcp(dpp=True), (
        "avg capped problem is not DPP "
        f"(n={n}, s_util_cap={s_util_cap:.12g}, "
        f"H_full_lambda_min={h_full_min_eig:.12g})"
    )
    return dict(
        problem=problem,
        params=params,
        vars=varset,
        coeffs=coeffs,
        h_full=H_full,
        h_full_min_eig=h_full_min_eig,
        is_dcp_dpp=problem.is_dcp(dpp=True),
    )


def _solve_avg_with_cap(
    S_mva,
    E_mwh,
    bus_idx,
    smp,
    profile,
    coeffs,
    s_util_cap,
    *,
    eta_c=PM.ETA_C,
    eta_d=PM.ETA_D,
    self_discharge=PM.SELF_DISCHARGE_HOURLY,
    soc_init=PM.SOC_INIT_FRAC,
    soc_min=PM.SOC_MIN_FRAC,
    soc_max=PM.SOC_MAX_FRAC,
    mu_volt=PM.MU_VOLT,
):
    T = PM.TIME_STEPS
    smp = np.asarray(smp, dtype=float)
    if smp.shape != (T,):
        raise ValueError(f"smp shape {smp.shape} != ({T},)")
    if not np.all(np.isfinite(smp)) or np.any(smp < 0.0):
        raise ValueError("smp must be finite and nonnegative")
    if not 0.0 < float(s_util_cap) <= 1.0:
        raise ValueError(f"invalid s_util_cap={s_util_cap}")

    n, S, E, onehot, load_p_val, load_q_val = lower_lp._prepare_common(
        S_mva,
        E_mwh,
        bus_idx,
        profile,
        self_discharge,
        soc_init,
    )
    coeffs = lower_lp._normalize_coeffs(coeffs, n, T)
    entry = _build_avg_problem_with_cap(
        n,
        eta_c,
        eta_d,
        self_discharge,
        soc_init,
        soc_min,
        soc_max,
        mu_volt,
        coeffs,
        s_util_cap,
    )
    p = entry["params"]
    p["S"].value = S
    p["E"].value = E
    p["bus_onehot"].value = onehot
    p["load_p_bus"].value = load_p_val
    p["load_q_bus"].value = load_q_val
    p["smp"].value = smp

    lower_lp._solve_qp(entry["problem"], "probe_q_boundary.solve_avg")

    v = entry["vars"]
    p_ch = np.asarray(v["P_ch"].value, dtype=float)
    p_dis = np.asarray(v["P_dis"].value, dtype=float)
    q = np.asarray(v["Q"].value, dtype=float)
    soc = np.asarray(v["soc"].value, dtype=float)
    return p_dis - p_ch, q, soc


def _assert_cap1_regression(custom, original, scenario):
    for name, custom_value, original_value in zip(
        ("P_net", "Q", "soc"), custom, original[:3]
    ):
        if not np.allclose(
            custom_value,
            original_value,
            rtol=0.0,
            atol=REGRESSION_ATOL,
        ):
            max_abs = float(
                np.max(np.abs(custom_value - original_value))
            )
            raise AssertionError(
                f"cap=1 regression failed: scenario={scenario}, "
                f"field={name}, max_abs={max_abs:.12g}"
            )


def _measure_coefficients(bus, S):
    grid = loss_coeffs.measure_coeffs_grid(
        int(bus[0]),
        float(S[0]),
        scenarios=PM.ALL_DAYS,
        times=range(PM.TIME_STEPS),
        p_center=0.0,
        strict=True,
    )
    return {
        scenario: evaluate._assemble_coeffs_1unit(grid, scenario)
        for scenario in PM.ALL_DAYS
    }


def _build_schedules(bus, S, E, coeffs_by_scenario, s_util_cap):
    unit_p = {}
    unit_q = {}
    unit_soc = {}
    base_p_sum = float(evaluate._BASE_P.sum())

    for scenario in PM.AVG_DAYS:
        smp = np.asarray(PM.SMP_PER_MWH[scenario], dtype=float)
        profile = np.asarray(PM.LOAD[scenario], dtype=float)
        solved = _solve_avg_with_cap(
            S,
            E,
            bus,
            smp,
            profile,
            coeffs_by_scenario[scenario],
            s_util_cap,
        )
        if float(s_util_cap) == 1.0:
            original = lower_lp.solve_avg(
                S,
                E,
                bus,
                smp,
                profile,
                coeffs_by_scenario[scenario],
                force_q_zero=False,
                assert_physics=False,
            )
            _assert_cap1_regression(solved, original, scenario)
        unit_p[scenario], unit_q[scenario], unit_soc[scenario] = solved

    for scenario in PM.PEAK_DAYS:
        profile = np.asarray(PM.LOAD[scenario], dtype=float)
        solved = lower_lp.solve_peak(
            S,
            E,
            bus,
            base_p_sum * profile,
            np.asarray(PM.SMP_PER_MWH[scenario], dtype=float),
            profile,
            coeffs_by_scenario[scenario],
            force_q_zero=False,
            assert_physics=False,
        )
        unit_p[scenario], unit_q[scenario], unit_soc[scenario] = solved[:3]

    return unit_p, unit_q, unit_soc


def _avg_benefit_components(base_flow, ac_result, unit_p, accounting):
    adjusted_season = accounting["adjusted_season"]
    n_weekdays_adjusted = {
        scenario: float(PM.N_WEEKDAYS[scenario])
        - (1.0 if scenario == adjusted_season else 0.0)
        for scenario in PM.AVG_DAYS
    }
    p_ch_agg = {
        scenario: np.maximum(-unit_p[scenario], 0.0).sum(axis=0)
        for scenario in PM.AVG_DAYS
    }
    p_dis_agg = {
        scenario: np.maximum(unit_p[scenario], 0.0).sum(axis=0)
        for scenario in PM.AVG_DAYS
    }
    b_arb = benefits.b_arb(
        p_ch_agg,
        p_dis_agg,
        PM.SMP_PER_MWH,
        n_weekdays_adjusted,
    )
    b_loss = benefits.b_loss(
        {
            scenario: base_flow["loss"][scenario]
            for scenario in PM.AVG_DAYS
        },
        {
            scenario: ac_result["loss_ess"][scenario]
            for scenario in PM.AVG_DAYS
        },
        PM.SMP_PER_MWH,
        n_weekdays_adjusted,
    )
    return float(b_arb), float(b_loss)


def _avg_q_metrics(unit_p, unit_q, S, s_util_cap):
    p_values = np.concatenate(
        [np.asarray(unit_p[s], dtype=float).ravel() for s in PM.AVG_DAYS]
    )
    q_values = np.concatenate(
        [np.asarray(unit_q[s], dtype=float).ravel() for s in PM.AVG_DAYS]
    )
    s_util = np.hypot(p_values, q_values) / float(S[0])
    binding_threshold = (
        float(s_util_cap) * POLY_BINDING_REL_THRESHOLD
    )
    return dict(
        avg_s_util_median=float(np.median(s_util)),
        avg_s_util_max=float(np.max(s_util)),
        poly_binding_hours=int(
            np.count_nonzero(s_util >= binding_threshold)
        ),
        q_sum_abs_mvar=float(np.sum(np.abs(q_values))),
    )


def _evaluate_cap(bus, S, E, coeffs_by_scenario, s_util_cap):
    unit_p, unit_q, _unit_soc = _build_schedules(
        bus, S, E, coeffs_by_scenario, s_util_cap
    )
    ac_result = evaluate._run_schedule_ac(unit_p, unit_q, bus, S, E)
    if ac_result["diverged"]:
        raise RuntimeError(
            f"AC power flow diverged for s_util_cap={s_util_cap}: "
            f"{ac_result['diverge_info']}"
        )
    base_flow = evaluate._get_base_flow()
    accounting = evaluate._accounting_values(
        base_flow, ac_result["p_slack_ess"], S, E
    )
    b_arb, b_loss = _avg_benefit_components(
        base_flow, ac_result, unit_p, accounting
    )
    result = dict(
        s_util_cap=float(s_util_cap),
        j_net_won=float(accounting["j_net"]),
        b_defer_won=float(accounting["b_defer"]),
        b_arb_won=b_arb,
        b_loss_won=b_loss,
    )
    result.update(_avg_q_metrics(unit_p, unit_q, S, s_util_cap))
    return result


def _money_cell(value):
    return f"{float(value):.6g}({float(value):.0f})"


def _print_results(results):
    baseline = results[0]["j_net_won"]
    print("[Q_BOUNDARY_SWEEP]")
    print(
        "s_util_cap,j_net_won,b_defer_won,b_arb_won,b_loss_won,"
        "avg_s_util_median,avg_s_util_max,poly_binding_hours,"
        "q_sum_abs_mvar,dj_net_vs_cap1_won,dj_net_vs_cap1_pct"
    )
    for row in results:
        delta = row["j_net_won"] - baseline
        delta_pct = (
            100.0 * delta / baseline
            if abs(baseline) > 0.0
            else float("nan")
        )
        print(
            f"{row['s_util_cap']:.6g},"
            f"{_money_cell(row['j_net_won'])},"
            f"{_money_cell(row['b_defer_won'])},"
            f"{_money_cell(row['b_arb_won'])},"
            f"{_money_cell(row['b_loss_won'])},"
            f"{row['avg_s_util_median']:.6g},"
            f"{row['avg_s_util_max']:.6g},"
            f"{row['poly_binding_hours']},"
            f"{row['q_sum_abs_mvar']:.6g},"
            f"{_money_cell(delta)},"
            f"{delta_pct:.6g}"
        )


def main():
    evaluate.init_worker()
    bus = np.array([int(round(X_OPT[0]))], dtype=int)
    S = np.array([float(X_OPT[1])], dtype=float)
    E = np.array([float(X_OPT[2])], dtype=float)
    coeffs_by_scenario = _measure_coefficients(bus, S)
    results = [
        _evaluate_cap(bus, S, E, coeffs_by_scenario, cap)
        for cap in S_UTIL_CAPS
    ]
    _print_results(results)


if __name__ == "__main__":
    main()
