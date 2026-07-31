"""PCS 다각형 정밀도(POLY_N) 스윕: 속도·정확도·실행가능집합·편익 원장 계측.

본코드(lower_lp/evaluate/params)는 수정하지 않는다. probe_lp_loss_proto의 문제 빌더와
손실 테이블 계측식을 읽기 전용으로 재사용하며, 이 파일 안에서만 PM.POLY_N을 순차 변경한다.
"""

import csv
import datetime
import importlib.metadata
import os
import platform
import socket
import sys
import time

import cvxpy as cp
import numpy as np
import pandapower as pp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import benefits
import evaluate
import params as PM
import probe_lp_loss_proto as PROTO


POLY_N_VALUES = [12, 16, 32, 64, 128]
SWEEP_METHODS = [('baseline_a', None), ('qp', None), ('pwl', 9)]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
AT_S_CAP_TOL = 1e-9
NONZERO_Q_TOL = 1e-6
NORM_OVER_S_TOL = 1e-9

TS_EXTRA_FIELDS = ['poly_n', 's_cap', 'norm_true', 's_app_deficit', 'at_s_cap']
TS_FIELDS = list(PROTO.TS_CSV_FIELDS) + TS_EXTRA_FIELDS

SUMMARY_FIELDS = [
    'poly_n', 'method', 'M', 'point_id', 'bus', 'S', 'E',
    'solve_time_s_per_scenario', 'solve_time_s_total',
    'n_variables', 'n_constraints', 'dpp_ok', 'solver_status', 'build_time_s',
    'theory_radius_err', 'max_s_app_deficit_mva', 'pcs_gap_won_per_yr',
    'pcs_undercharge_frac_weighted',
    'pcs_ratio_min', 'pcs_ratio_p25', 'pcs_ratio_median', 'pcs_ratio_p75', 'pcs_ratio_max',
    's_cap', 'n_hours_at_s_cap', 'max_norm_over_S', 'q_sum_mvar',
    'n_nonzero_q_hours',
    'arb_proxy', 'q_loss_measured', 'pcs_true_cost', 'pcs_charged_cost',
    'actual_line_loss_reduction', 'b_energy', 'ledger_residual',
    'j_net', 'b_minus_a',
]


def _result_paths():
    host = socket.gethostname()
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    stem = os.path.join(RESULTS_DIR, f'probe_polyn_sweep_{host}_{stamp}')
    return stem + '.csv', stem + '_summary.csv', stem + '_report.md'


def _problem_size(entry):
    n_variables = int(sum(int(v.size) for v in entry['problem'].variables()))
    n_constraints = int(sum(int(c.size) for c in entry['problem'].constraints))
    return n_variables, n_constraints


def _entry_key(poly_n, method, point):
    return int(poly_n), str(method), str(point['point_id'])


def _build_entry(poly_n, method, M, point, cache, build_times):
    key = _entry_key(poly_n, method, point)
    if key in cache:
        return cache[key]
    PM.POLY_N = int(poly_n)
    t0 = time.perf_counter()
    if method == 'qp':
        entry = PROTO._build_problem_proto(
            'qp', n=1, T=PM.TIME_STEPS, bus_idx=int(point['b'])
        )
    elif method == 'pwl':
        entry = PROTO._build_problem_proto('pwl', n=1, T=PM.TIME_STEPS, M=M)
    else:
        entry = PROTO._build_problem_proto('none', n=1, T=PM.TIME_STEPS)
    build_times[key] = time.perf_counter() - t0
    cache[key] = entry
    return entry


def _validate_polyn_rebuild(cache, build_times):
    point = PROTO.POINTS[0]
    records = []
    for poly_n in (12, 128):
        entry = _build_entry(poly_n, 'qp', None, point, cache, build_times)
        n_variables, n_constraints = _problem_size(entry)
        records.append(dict(
            poly_n=poly_n, n_variables=n_variables, n_constraints=n_constraints,
            s_cap_factor=float(np.cos(np.pi / poly_n)),
            dpp_ok=bool(entry['problem'].is_dcp(dpp=True)),
        ))
        print(
            f"cache verification: POLY_N={poly_n} n_variables={n_variables} "
            f"n_constraints={n_constraints} s_cap_factor={np.cos(np.pi / poly_n):.12f} "
            f"dpp_ok={entry['problem'].is_dcp(dpp=True)}",
            flush=True,
        )
    if records[0]['n_constraints'] == records[1]['n_constraints']:
        raise RuntimeError(
            f"POLY_N constraint rows identical: N=12 {records[0]['n_constraints']}, "
            f"N=128 {records[1]['n_constraints']}"
        )
    return records


def _solve_combo(entry, poly_n, method, M, point, loss_table,
                 v_sq_line_table, ac_flow_table):
    PM.POLY_N = int(poly_n)
    bus, S, E = int(point['b']), float(point['S']), float(point['E'])
    unit_p, unit_q = {}, {}
    aux = {name: {} for name in ('p_ch', 'p_dis', 's_app', 'q_penalty')}
    solve_times, statuses = {}, {}

    for scenario in PM.AVG_DAYS:
        lhs_rows = None
        if method == 'pwl':
            lhs_rows = PROTO._lhs_rows_for(loss_table, bus, scenario, M)
        v_sq = v_sq_line_table[scenario] if method == 'qp' else None
        q_flow = ac_flow_table[scenario]['q_from'] if method == 'qp' else None
        PROTO._set_params(
            entry, S, E, [bus], PM.LOAD[scenario], PM.SMP_PER_MWH[scenario],
            lhs_row_values=lhs_rows, v_sq_line=v_sq, qe_base_ac_mvar=q_flow,
        )
        elapsed, _solver, _inaccurate = PROTO._solve_timed(entry)
        stats = entry['problem'].solver_stats
        solver_time = getattr(stats, 'solve_time', None) if stats is not None else None
        solve_times[scenario] = float(solver_time if solver_time is not None else elapsed)
        statuses[scenario] = str(entry['problem'].status)

        variables = entry['vars']
        unit_p[scenario] = np.array(variables['P_net'].value, copy=True)
        unit_q[scenario] = np.array(variables['Q'].value, copy=True)
        aux['p_ch'][scenario] = np.array(variables['P_ch'].value, copy=True)
        aux['p_dis'][scenario] = np.array(variables['P_dis'].value, copy=True)
        aux['s_app'][scenario] = (
            np.array(variables['s_app'].value, copy=True)
            if variables['s_app'] is not None
            else np.full((1, PM.TIME_STEPS), np.nan)
        )
        aux['q_penalty'][scenario] = (
            np.array(variables['q_penalty'].value, copy=True)
            if variables['q_penalty'] is not None
            else np.zeros((1, PM.TIME_STEPS))
        )

    PROTO._assert_pcs_circle(
        unit_p, unit_q, np.array([S]),
        f"POLY_N={poly_n}/{method}/point={point['point_id']}",
    )
    return unit_p, unit_q, aux, solve_times, statuses


def _evaluate_avg(point, unit_p, unit_q, baseline):
    net = evaluate._NET
    base_p, base_q = evaluate._BASE_P, evaluate._BASE_Q
    base_flow = evaluate._get_base_flow()
    bus, S, E = int(point['b']), float(point['S']), float(point['E'])

    evaluate._ensure_sgens(net, 1)
    net.sgen.at[0, 'bus'] = bus
    unit_loss_pcs = benefits.loss_pcs(unit_p, unit_q)
    p_slack, line_loss = {}, {}
    for scenario in PM.AVG_DAYS:
        p_slack[scenario] = np.zeros(PM.TIME_STEPS)
        line_loss[scenario] = np.zeros(PM.TIME_STEPS)
        for t in range(PM.TIME_STEPS):
            net.load['p_mw'] = base_p * PM.LOAD[scenario][t]
            net.load['q_mvar'] = base_q * PM.LOAD[scenario][t]
            net.sgen.at[0, 'p_mw'] = float(
                unit_p[scenario][0, t] - unit_loss_pcs[scenario][0, t]
            )
            net.sgen.at[0, 'q_mvar'] = float(unit_q[scenario][0, t])
            ok = evaluate._run_pf_with_retry(
                net,
                log_context=dict(
                    b=[bus], S=[S], E=[E], scenario=scenario, t=t
                ),
            )
            if not ok:
                raise RuntimeError(
                    f"AC divergence: point={point['point_id']} scenario={scenario} t={t}"
                )
            p_slack[scenario][t] = float(net.res_ext_grid.p_mw.sum())
            line_loss[scenario][t] = float(net.res_line.pl_mw.sum())

    b_energy = benefits.b_energy(
        {s: base_flow['p_slack'][s] for s in PM.AVG_DAYS},
        p_slack, PM.SMP_PER_MWH, PM.N_WEEKDAYS,
    )
    j_net = benefits.j_net(b_energy, baseline['b_defer'], S, E)
    return dict(
        line_loss=line_loss, b_energy=float(b_energy), j_net=float(j_net),
    )


def _annual_metrics(point, unit_p, unit_q, aux, loss_table, ac_result):
    bus = int(point['b'])
    arb = p0_q_loss = pcs_true_cost = pcs_charged_cost = actual_line_loss = 0.0
    ratio_values = []
    ts_metrics = {}
    base_flow = evaluate._get_base_flow()

    for scenario in PM.AVG_DAYS:
        smp = np.asarray(PM.SMP_PER_MWH[scenario], dtype=float)
        weight = float(PM.N_WEEKDAYS[scenario]) * PM.DT_HOURS
        P = unit_p[scenario][0]
        Q = unit_q[scenario][0]
        s_app = aux['s_app'][scenario][0]
        q_penalty = aux['q_penalty'][scenario][0]
        norm_true = np.hypot(P, Q)
        pcs_true = PROTO.C_PCS * (norm_true - np.abs(P))
        pcs_charged = PROTO.C_PCS * q_penalty
        loss_arr = loss_table[bus][scenario]
        q_loss = np.array([
            np.interp(Q[t], PROTO.Q_BOUNDARY_POINTS, loss_arr[t, 0] - loss_arr[t, :])
            for t in range(PM.TIME_STEPS)
        ])

        arb += float(np.sum(smp * P)) * weight
        p0_q_loss += float(np.sum(smp * q_loss)) * weight
        pcs_true_cost += float(np.sum(smp * pcs_true)) * weight
        pcs_charged_cost += float(np.sum(smp * pcs_charged)) * weight
        actual_line_loss += float(np.sum(
            smp * (
                np.asarray(base_flow['loss'][scenario])
                - np.asarray(ac_result['line_loss'][scenario])
            )
        )) * weight

        mask = (Q > NONZERO_Q_TOL) & (pcs_true > 0.0)
        if np.any(mask):
            ratio_values.extend((pcs_charged[mask] / pcs_true[mask]).tolist())
        for t in range(PM.TIME_STEPS):
            ts_metrics[(scenario, t)] = dict(
                norm_true=float(norm_true[t]), pcs_true=float(pcs_true[t]),
                pcs_charged=float(pcs_charged[t]), s_app=float(s_app[t]),
                q_penalty=float(q_penalty[t]),
            )

    ledger_rhs = arb - pcs_true_cost + actual_line_loss
    ratios = (
        np.percentile(np.asarray(ratio_values), [0, 25, 50, 75, 100])
        if ratio_values else np.full(5, np.nan)
    )
    return dict(
        arb_proxy=arb, q_loss_measured=p0_q_loss,
        pcs_true_cost=pcs_true_cost, pcs_charged_cost=pcs_charged_cost,
        actual_line_loss_reduction=actual_line_loss,
        b_energy=ac_result['b_energy'],
        ledger_residual=ac_result['b_energy'] - ledger_rhs,
        pcs_gap_won_per_yr=pcs_true_cost - pcs_charged_cost,
        pcs_undercharge_frac_weighted=(
            (pcs_true_cost - pcs_charged_cost) / pcs_true_cost
            if pcs_true_cost != 0.0 else np.nan
        ),
        pcs_ratio_min=float(ratios[0]), pcs_ratio_p25=float(ratios[1]),
        pcs_ratio_median=float(ratios[2]), pcs_ratio_p75=float(ratios[3]),
        pcs_ratio_max=float(ratios[4]), ts_metrics=ts_metrics,
    )


def _build_ts_rows(poly_n, method, M, point, unit_p, unit_q, aux, annual):
    S = float(point['S'])
    s_cap = S * float(np.cos(np.pi / poly_n))
    q_nonzero = sum(
        int(np.sum(unit_q[s][0] > NONZERO_Q_TOL)) for s in PM.AVG_DAYS
    )
    q_sum = float(sum(np.sum(unit_q[s][0]) for s in PM.AVG_DAYS))
    free_zone = S * float(np.sin(np.pi / poly_n))
    rows = []
    for scenario in PM.AVG_DAYS:
        for t in range(PM.TIME_STEPS):
            base = {field: '' for field in PROTO.TS_CSV_FIELDS}
            P = float(unit_p[scenario][0, t])
            Q = float(unit_q[scenario][0, t])
            p_ch = float(aux['p_ch'][scenario][0, t])
            p_dis = float(aux['p_dis'][scenario][0, t])
            s_app = float(aux['s_app'][scenario][0, t])
            q_penalty = float(aux['q_penalty'][scenario][0, t])
            metric = annual['ts_metrics'][(scenario, t)]
            norm_true = metric['norm_true']
            base.update(
                method=method, M=(M if M is not None else ''),
                point_id=point['point_id'], scenario=scenario, t=t,
                q_lp=Q, inaccurate=False,
                free_zone_width_mvar=free_zone,
                max_uncounted_loss_mva=PROTO.C_PCS * free_zone,
                n_nonzero_q_hours=q_nonzero,
                p_ch=p_ch, p_dis=p_dis, p_net=P, s_app=s_app,
                q_penalty=q_penalty, pcs_true=metric['pcs_true'],
                pcs_charged=metric['pcs_charged'],
                qp_v2_correction=PROTO.QP_V2_CORRECTION,
            )
            base.update(
                poly_n=poly_n, s_cap=s_cap, norm_true=norm_true,
                s_app_deficit=norm_true - s_app,
                at_s_cap=bool(np.isfinite(s_app) and s_app >= s_cap - AT_S_CAP_TOL),
            )
            rows.append(base)
    return rows, q_nonzero, q_sum, s_cap


def _summary_row(poly_n, method, M, point, entry, build_time, solve_times,
                 statuses, ts_rows, q_nonzero, q_sum, s_cap, annual, ac_result):
    n_variables, n_constraints = _problem_size(entry)
    deficits = np.asarray([row['s_app_deficit'] for row in ts_rows], dtype=float)
    norms = np.asarray([row['norm_true'] for row in ts_rows], dtype=float)
    at_cap = [bool(row['at_s_cap']) for row in ts_rows]
    S = float(point['S'])
    max_norm_over_s = float(np.max(norms / S))
    if max_norm_over_s > 1.0 + NORM_OVER_S_TOL:
        raise RuntimeError(
            f"max_norm_over_S={max_norm_over_s:.12f}, POLY_N={poly_n}, "
            f"method={method}, point={point['point_id']}"
        )
    return dict(
        poly_n=poly_n, method=method, M=(M if M is not None else ''),
        point_id=point['point_id'], bus=int(point['b']), S=S, E=float(point['E']),
        solve_time_s_per_scenario=';'.join(
            f'{s}:{solve_times[s]:.9f}' for s in PM.AVG_DAYS
        ),
        solve_time_s_total=float(sum(solve_times.values())),
        n_variables=n_variables, n_constraints=n_constraints,
        dpp_ok=bool(entry['problem'].is_dcp(dpp=True)),
        solver_status=';'.join(f'{s}:{statuses[s]}' for s in PM.AVG_DAYS),
        build_time_s=build_time,
        theory_radius_err=1.0 - float(np.cos(np.pi / poly_n)),
        max_s_app_deficit_mva=(
            float(np.nanmax(deficits)) if np.any(np.isfinite(deficits)) else np.nan
        ),
        pcs_gap_won_per_yr=annual['pcs_gap_won_per_yr'],
        pcs_undercharge_frac_weighted=annual['pcs_undercharge_frac_weighted'],
        pcs_ratio_min=annual['pcs_ratio_min'], pcs_ratio_p25=annual['pcs_ratio_p25'],
        pcs_ratio_median=annual['pcs_ratio_median'], pcs_ratio_p75=annual['pcs_ratio_p75'],
        pcs_ratio_max=annual['pcs_ratio_max'],
        s_cap=s_cap, n_hours_at_s_cap=int(sum(at_cap)),
        max_norm_over_S=max_norm_over_s, q_sum_mvar=q_sum,
        n_nonzero_q_hours=q_nonzero,
        arb_proxy=annual['arb_proxy'], q_loss_measured=annual['q_loss_measured'],
        pcs_true_cost=annual['pcs_true_cost'],
        pcs_charged_cost=annual['pcs_charged_cost'],
        actual_line_loss_reduction=annual['actual_line_loss_reduction'],
        b_energy=annual['b_energy'], ledger_residual=annual['ledger_residual'],
        j_net=ac_result['j_net'], b_minus_a=np.nan,
    )


def _write_csv(path, fields, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value, digits=6):
    if value is None:
        return 'N/A'
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f'{value:.{digits}f}' if np.isfinite(value) else str(value)


def _write_report(path, cache_records, loss_build_count, summaries,
                  total_elapsed, total_runpp_calls):
    by_key = {
        (int(r['poly_n']), r['method'], r['point_id']): r for r in summaries
    }
    lines = ['# POLY_N sweep numeric report', '']

    lines += [
        '## 1. Cache-key verification', '',
        '| POLY_N | n_variables | n_constraints | s_cap_factor | dpp_ok |',
        '|---:|---:|---:|---:|---:|',
    ]
    for r in cache_records:
        lines.append(
            f"| {r['poly_n']} | {r['n_variables']} | {r['n_constraints']} | "
            f"{_fmt(r['s_cap_factor'], 12)} | {r['dpp_ok']} |"
        )

    lines += [
        '', '## 2. Loss-table reuse', '',
        '| loss_table_build_count |', '|---:|', f'| {loss_build_count} |',
        '', '## 3. Speed and problem size', '',
    ]
    columns = [
        (method, point['point_id'])
        for method, _M in SWEEP_METHODS for point in PROTO.POINTS
    ]
    lines.append('| POLY_N | ' + ' | '.join(f'{m}/{p}' for m, p in columns) + ' |')
    lines.append('|---:|' + '|'.join('---:' for _ in columns) + '|')
    for n in POLY_N_VALUES:
        cells = []
        for method, pid in columns:
            r = by_key[(n, method, pid)]
            cells.append(
                f"{_fmt(r['solve_time_s_total'], 9)} / {r['n_constraints']}"
            )
        lines.append(f"| {n} | " + ' | '.join(cells) + ' |')

    lines += [
        '', '## 4. PCS accuracy — QP', '',
        '| POLY_N | point | theory_radius_err | pcs_undercharge_frac_weighted | '
        'pcs_gap_won_per_yr | pcs_ratio_median | max_s_app_deficit_mva |',
        '|---:|---|---:|---:|---:|---:|---:|',
    ]
    qp_rows = [r for r in summaries if r['method'] == 'qp']
    for r in qp_rows:
        lines.append(
            f"| {r['poly_n']} | {r['point_id']} | {_fmt(r['theory_radius_err'], 9)} | "
            f"{_fmt(r['pcs_undercharge_frac_weighted'], 9)} | "
            f"{_fmt(r['pcs_gap_won_per_yr'], 2)} | {_fmt(r['pcs_ratio_median'], 9)} | "
            f"{_fmt(r['max_s_app_deficit_mva'], 9)} |"
        )

    theory12 = 1.0 - np.cos(np.pi / 12)
    lines += [
        '', '## 5. 1/N^2 scale', '',
        '| POLY_N | point | theory_radius_err | theory_vs_N12 | '
        'pcs_undercharge_frac_weighted | undercharge_vs_N12 |',
        '|---:|---|---:|---:|---:|---:|',
    ]
    for point in PROTO.POINTS:
        pid = point['point_id']
        base_under = by_key[(12, 'qp', pid)]['pcs_undercharge_frac_weighted']
        for n in POLY_N_VALUES:
            r = by_key[(n, 'qp', pid)]
            lines.append(
                f"| {n} | {pid} | {_fmt(r['theory_radius_err'], 9)} | "
                f"{_fmt(r['theory_radius_err'] / theory12, 9)} | "
                f"{_fmt(r['pcs_undercharge_frac_weighted'], 9)} | "
                f"{_fmt(r['pcs_undercharge_frac_weighted'] / base_under, 9)} |"
            )

    lines += [
        '', '## 6. Feasible-set metrics — QP', '',
        '| POLY_N | point | s_cap | n_hours_at_s_cap | q_sum_mvar | max_norm_over_S |',
        '|---:|---|---:|---:|---:|---:|',
    ]
    for r in qp_rows:
        lines.append(
            f"| {r['poly_n']} | {r['point_id']} | {_fmt(r['s_cap'], 9)} | "
            f"{r['n_hours_at_s_cap']} | {_fmt(r['q_sum_mvar'], 9)} | "
            f"{_fmt(r['max_norm_over_S'], 12)} |"
        )

    lines += [
        '', '## 7. Annual ledger', '',
        '| POLY_N | method | point | (1) | (2) | (3) | (4) | (5) | (6) | '
        'residual | j_net | (b-a) |',
        '|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in summaries:
        lines.append(
            f"| {r['poly_n']} | {r['method']} | {r['point_id']} | "
            f"{_fmt(r['arb_proxy'], 2)} | {_fmt(r['q_loss_measured'], 2)} | "
            f"{_fmt(r['pcs_true_cost'], 2)} | {_fmt(r['pcs_charged_cost'], 2)} | "
            f"{_fmt(r['actual_line_loss_reduction'], 2)} | {_fmt(r['b_energy'], 2)} | "
            f"{_fmt(r['ledger_residual'], 6)} | {_fmt(r['j_net'], 2)} | "
            f"{_fmt(r['b_minus_a'], 2)} |"
        )
    residual_rows = [r for r in summaries if r['ledger_residual'] != 0.0]
    if residual_rows:
        lines += [
            '', '### Residual rows', '',
            '| POLY_N | method | point | residual |',
            '|---:|---|---|---:|',
        ]
        for r in residual_rows:
            lines.append(
                f"| {r['poly_n']} | {r['method']} | {r['point_id']} | "
                f"{_fmt(r['ledger_residual'], 9)} |"
            )

    solver_versions = []
    for package in ('clarabel', 'osqp', 'scs'):
        try:
            solver_versions.append(f'{package}={importlib.metadata.version(package)}')
        except importlib.metadata.PackageNotFoundError:
            pass
    lines += [
        '', '## 8. Environment', '',
        '| key | value |', '|---|---:|',
        f'| cvxpy | {cp.__version__} |',
        f"| solvers | {', '.join(solver_versions)} |",
        f'| installed_solvers | {cp.installed_solvers()} |',
        f'| python | {platform.python_version()} |',
        f'| platform | {platform.platform()} |',
        f'| total_elapsed_s | {_fmt(total_elapsed, 6)} |',
        f'| total_runpp_calls | {total_runpp_calls} |',
    ]
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    started = time.perf_counter()
    original_poly_n = PM.POLY_N
    ts_path, summary_path, report_path = _result_paths()
    cache, build_times = {}, {}
    ts_rows, summary_rows = [], []
    loss_table_build_count = 0

    try:
        PROTO._check_env()
        cache_records = _validate_polyn_rebuild(cache, build_times)

        net, _q_scale, _p_total, _q_before = PROTO._build_net_with_pf(PROTO.TARGET_PF)
        base_p, base_q = PROTO._prepare_condition(net)

        with PROTO._count_runpp_calls() as runpp_counter:
            baselines = {}
            for point in PROTO.POINTS:
                x = np.array([point['b'], point['S'], point['E']], dtype=float)
                detail = PROTO._evaluate_with_force_q(x, True)
                if detail.get('diverged'):
                    raise RuntimeError(
                        f"baseline divergence: point={point['point_id']} "
                        f"{detail.get('diverge_info')}"
                    )
                baselines[point['point_id']] = dict(
                    b_defer=float(detail['b_defer']),
                )

            loss_table_build_count += 1
            loss_table, v_sq_line_table, ac_flow_table, _vbus, _acfull = (
                PROTO._measure_loss_table(net, base_p, base_q)
            )
            print(f'loss_table build count = {loss_table_build_count}', flush=True)

            for poly_n in POLY_N_VALUES:
                PM.POLY_N = poly_n
                for method, M in SWEEP_METHODS:
                    for point in PROTO.POINTS:
                        key = _entry_key(poly_n, method, point)
                        entry = _build_entry(
                            poly_n, method, M, point, cache, build_times
                        )
                        unit_p, unit_q, aux, solve_times, statuses = _solve_combo(
                            entry, poly_n, method, M, point, loss_table,
                            v_sq_line_table, ac_flow_table,
                        )
                        ac_result = _evaluate_avg(
                            point, unit_p, unit_q,
                            baselines[point['point_id']],
                        )
                        annual = _annual_metrics(
                            point, unit_p, unit_q, aux, loss_table, ac_result
                        )
                        rows, q_nonzero, q_sum, s_cap = _build_ts_rows(
                            poly_n, method, M, point, unit_p, unit_q, aux, annual
                        )
                        ts_rows.extend(rows)
                        summary_rows.append(_summary_row(
                            poly_n, method, M, point, entry, build_times[key],
                            solve_times, statuses, rows, q_nonzero, q_sum,
                            s_cap, annual, ac_result,
                        ))
                        print(
                            f"N={poly_n} method={method} point={point['point_id']} "
                            f"solve_total={sum(solve_times.values()):.9f}s "
                            f"j_net={ac_result['j_net']:.2f}",
                            flush=True,
                        )

            for poly_n in POLY_N_VALUES:
                for point in PROTO.POINTS:
                    baseline_j = next(
                        r['j_net'] for r in summary_rows
                        if r['poly_n'] == poly_n
                        and r['method'] == 'baseline_a'
                        and r['point_id'] == point['point_id']
                    )
                    for row in summary_rows:
                        if (
                            row['poly_n'] == poly_n
                            and row['point_id'] == point['point_id']
                        ):
                            row['b_minus_a'] = row['j_net'] - baseline_j

            total_runpp_calls = int(runpp_counter['n'])

        _write_csv(ts_path, TS_FIELDS, ts_rows)
        _write_csv(summary_path, SUMMARY_FIELDS, summary_rows)
        total_elapsed = time.perf_counter() - started
        _write_report(
            report_path, cache_records, loss_table_build_count,
            summary_rows, total_elapsed, total_runpp_calls,
        )
        print(f'time-series CSV: {ts_path}', flush=True)
        print(f'summary CSV: {summary_path}', flush=True)
        print(f'report MD: {report_path}', flush=True)
        print(f'total elapsed: {total_elapsed:.6f}s', flush=True)
        print(f'total runpp calls: {total_runpp_calls}', flush=True)
    finally:
        PM.POLY_N = original_poly_n
        PROTO._restore_evaluate_state()


if __name__ == '__main__':
    main()
