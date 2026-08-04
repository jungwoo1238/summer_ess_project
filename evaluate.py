"""평가 지휘자 (CLAUDE.md 부록A, 7절). lower_lp + 조류계산 + benefits + 페널티를 엮는다.

워커 초기화 패턴(전역 NET/BASE_P/BASE_Q, CLAUDE.md 7절)을 따른다. init_worker()가
build_net 1회 + 기저(ESS 없음) 조류계산 ALL_DAYS x 24h(120회, 입자와 무관하므로 워커당 1회만)를
캐싱한다. 이후 evaluate_particle 1회당 발생하는 조류계산은 ESS 주입 상태의 120회뿐이다
(7절 실측 기준 1.178초/회).

n기 일반형: 입자 x는 3n 차원 (b_i,S_i,E_i) x n (★ C.6-3 LinDistFlow 편입 이후 - q_ratio는
더 이상 PSO 변수가 아니다. Q는 lower_lp의 시변 LP 변수로 전 기 조인트로 결정된다 -
부록C.4-(3)). n기는 하나의 조인트 LP로 함께 풀리고(LinDistFlow 전압유도항이 전 기 주입을
하나의 조류식으로 묶으므로 - lower_lp.py 모듈 docstring 참조), 조류계산(AC PF)에서는
여전히 sgen 개별 주입으로 만난다(동일 버스 중복배치는 sgen 2개로 별도 생성, 병합 안 함 -
(a)방식).
"""

import numpy as np
import pandapower as pp

import params as PM
from build_net import build_net
from lower_lp import solve_avg, solve_peak
import loss_coeffs
import benefits


# ============================================================
# 워커 전역 상태 (CLAUDE.md 7절 속도최적화: 워커당 1회 build_net + 캐싱)
# ============================================================
_NET = None
_BASE_P = None
_BASE_Q = None
_BASE_FLOW = None

# 발산 로깅 (CLAUDE.md 지시 (d)): 재시도 이벤트마다 1건씩 누적.
# 항목: b,S,E(입자 전체), scenario, t, recovered(2차 flat 재시도로 살아났는지).
DIVERGENCE_LOG = []


def init_worker():
    """워커 시작 시 1회 호출. build_net + 기저부하 캐싱 + 기저 조류계산(120회) 선계산."""
    global _NET, _BASE_P, _BASE_Q, _BASE_FLOW
    _NET = build_net()
    _BASE_P = _NET.load['p_mw'].to_numpy().copy()
    _BASE_Q = _NET.load['q_mvar'].to_numpy().copy()
    _BASE_FLOW = _compute_base_flow(_NET, _BASE_P, _BASE_Q)


def _ensure_worker_state():
    """단일 프로세스(테스트 등)에서 init_worker를 직접 호출하지 않았을 때 지연 초기화."""
    if _NET is None:
        init_worker()


def _get_base_flow():
    _ensure_worker_state()
    return _BASE_FLOW


def reset_divergence_log():
    DIVERGENCE_LOG.clear()


def get_divergence_stats():
    """발산 로그 요약: 총 재시도 횟수, flat 재시도로 살아난 횟수·비율 (지시 (d) 로깅 항목 3)."""
    total = len(DIVERGENCE_LOG)
    recovered = sum(1 for d in DIVERGENCE_LOG if d['recovered'])
    return dict(
        total_retries=total,
        recovered=recovered,
        recovered_ratio=(recovered / total if total else float('nan')),
    )


# ============================================================
# 조류계산 (발산 재시도 포함, CLAUDE.md 지시 (d))
# ============================================================

def _run_pf_with_retry(net, log_context=None):
    """1차: init='results'(warm start). 실패 시 2차: init='flat'(오염된 res_bus를 명시적으로
    끊음 - init='auto'는 방금 발산해 오염된 결과를 다시 집어 재시도가 무의미해지므로 금지).
    둘 다 실패하면 False(발산 확정). log_context가 주어지면 재시도 발생 시(1차 실패 시)만 기록한다."""
    try:
        pp.runpp(net, numba=True, init='results')
        return True
    except Exception:
        pass

    try:
        pp.runpp(net, numba=True, init='flat')
        recovered = True
    except Exception:
        recovered = False

    if log_context is not None:
        DIVERGENCE_LOG.append(dict(log_context, recovered=recovered))
    return recovered


def _compute_base_flow(net, base_p, base_q):
    """ESS 없는 기저 조류계산, ALL_DAYS x 24h(120회). 입자(x)와 무관하므로 워커당 1회만
    계산해 캐싱한다(CLAUDE.md 7절 "기저부하 프로파일도 고정 -> 한 번 계산해 저장").

    v_violation/i_violation도 같이 누적해 둔다 - S=0/E=0 입자의 위반량이 기저 계통 그대로인지
    (evaluate_particle의 v_violation/i_violation과 정확히 일치하는지) 재계산 없이 대조하기 위함.
    """
    p_slack = {s: np.zeros(PM.TIME_STEPS) for s in PM.ALL_DAYS}
    loss = {s: np.zeros(PM.TIME_STEPS) for s in PM.ALL_DAYS}

    line_in_service = net.line['in_service'].to_numpy()
    line_rating = net.line['max_i_ka'].to_numpy()
    n_line = len(net.line)

    v_violation = 0.0
    i_violation = 0.0

    for s in PM.ALL_DAYS:
        profile = PM.LOAD[s]
        for t in range(PM.TIME_STEPS):
            scale = profile[t]
            net.load['p_mw'] = base_p * scale
            net.load['q_mvar'] = base_q * scale

            ok = _run_pf_with_retry(net)
            if not ok:
                raise RuntimeError(
                    f'기저(ESS 없음) 조류계산 발산: 시나리오={s}, t={t}. '
                    '정상 부하범위에서 기저 발산은 비정상이므로 그대로 보고한다.'
                )

            p_slack[s][t] = net.res_ext_grid.p_mw.sum()
            loss[s][t] = net.res_line.pl_mw.sum()

            v = net.res_bus.vm_pu.to_numpy()
            v_violation += float(np.sum(np.maximum(0.0, v - PM.V_MAX) + np.maximum(0.0, PM.V_MIN - v)))

            i_ratio = np.zeros(n_line)
            i_ratio[line_in_service] = (
                net.res_line.i_ka.to_numpy()[line_in_service] / line_rating[line_in_service]
            )
            i_violation += float(np.sum(np.maximum(0.0, i_ratio[line_in_service] - 1.0)))

    return dict(p_slack=p_slack, loss=loss, v_violation=v_violation, i_violation=i_violation)


# ============================================================
# 입자 파싱 (n기 일반형, CLAUDE.md 지시 (f))
# ============================================================

def _parse_particle(x):
    """x(3n차원, C.6-3 이후) -> b(int,버스),S(MVA),E(MWh). 경계 clamp는 방어적으로 여기서도
    수행한다(pso_core가 이미 clamp하지만 evaluate_particle을 테스트 등에서 직접 호출할 수 있어서)."""
    x = np.asarray(x, dtype=float)
    assert x.size > 0 and x.size % 3 == 0, f'입자 차원 {x.size}는 3의 배수(3n, n>=1)여야 함'
    n = x.size // 3
    x3 = x.reshape(n, 3)

    b = np.clip(np.round(x3[:, 0]), PM.B_BOUNDS[0], PM.B_BOUNDS[1]).astype(int)
    S = np.clip(x3[:, 1], PM.S_BOUNDS[0], PM.S_BOUNDS[1])
    E = np.clip(x3[:, 2], PM.E_BOUNDS[0], PM.E_BOUNDS[1])
    return b, S, E


def _ensure_sgens(net, n):
    """sgen 개수를 n에 맞춘다. 기존 것들의 숫자값만 바꿔쓰는 것이 기본 원칙(지시 (c))이고,
    이 함수는 개수 자체가 달라질 때(입자 차원 n이 바뀔 때)만 호출된다 - 같은 n으로 반복
    평가하는 동안에는 드롭/생성이 전혀 일어나지 않는다."""
    current = len(net.sgen)
    if current == n:
        return
    if current > n:
        net.sgen.drop(net.sgen.index[n:], inplace=True)
    else:
        for _ in range(current, n):
            pp.create_sgen(net, bus=PM.B_BOUNDS[0], p_mw=0.0, q_mvar=0.0, name='ESS')
    # loop에서 net.sgen.at[i, ...]로 위치 i를 직접 주소하므로 인덱스를 0..n-1로 고정해 둔다.
    net.sgen.index = np.arange(len(net.sgen))


# ============================================================
# 하위 LP: 전 기 조인트 스케줄 (C.6-3 LinDistFlow 편입 - "독립최적화"에서 "조인트"로 전환)
# ============================================================
# ★ 2026-07 C.6-3: 예전에는 기별로 solve_avg/solve_peak를 따로 불렀다(CLAUDE.md 10절
# "독립최적화" - LinDistFlow 편입 전에는 기 사이에 결합 제약이 없었으므로 정확했다). 이제는
# LinDistFlow 전압유도항이 "그 버스 하류 전 기 주입 합"에 의존하므로 기별로 따로 풀면 다른
# 기의 존재를 볼 수 없다 - lower_lp.solve_avg/solve_peak가 n기를 한 번에 받는 조인트
# 시그니처로 바뀌었다(lower_lp.py 모듈 docstring 참조). 여기서는 시나리오당 1회씩만
# 호출하면 되므로(예전 "시나리오 x 기"만큼 호출하던 것에서 "시나리오"만큼으로 줄었다) 오히려
# 더 단순해졌다.

_LOSS_COEFF_NAMES = ("a_P", "a_Q", "b_PP", "b_QQ", "b_PQ")
_COEF_REL_EPS = 1e-12
S_ACTIVE_MIN = 1e-6  # MVA (1 kVA); 이보다 작으면 조류·편익 관점에서 비활성
E_ACTIVE_MIN = 1e-6  # MWh (1 Wh); S 또는 E 하나라도 미만이면 비활성


def _assemble_coeffs_1unit(grid_result, scenario):
    """measure_coeffs_grid의 한 시나리오를 lower_lp용 (1,T) dict로 변환한다."""
    scenarios = tuple(grid_result["scenarios"])
    try:
        scenario_index = scenarios.index(str(scenario))
    except ValueError as exc:
        raise KeyError(f"scenario={scenario!r} not in coefficient grid") from exc
    return {
        name: np.asarray(
            grid_result[name][scenario_index, :], dtype=float
        ).reshape(1, PM.TIME_STEPS)
        for name in _LOSS_COEFF_NAMES
    }


def _measure_recomputed_coeffs_1unit(
    bus, S, scenario, p_center, *, net
):
    """시각마다 다른 운영점 중심으로 2차 계수를 측정해 (1,T)로 조립한다."""
    p_center = np.asarray(p_center, dtype=float)
    if p_center.shape == (1, PM.TIME_STEPS):
        p_center = p_center[0]
    if p_center.shape != (PM.TIME_STEPS,):
        raise ValueError(
            f"p_center shape={p_center.shape}, expected ({PM.TIME_STEPS},)"
        )
    excess = float(np.max(np.maximum(np.abs(p_center) - float(S), 0.0)))
    if excess > 1e-4:
        raise ValueError(f"solve1 p_center exceeds S by {excess:.12g} MW")
    # 솔버 허용오차 수준의 PCS 경계 초과만 측정격자 범위로 되돌린다.
    p_center = np.clip(p_center, -float(S), float(S))

    assembled = {
        name: np.empty((1, PM.TIME_STEPS), dtype=float)
        for name in _LOSS_COEFF_NAMES
    }
    for t in range(PM.TIME_STEPS):
        measured = loss_coeffs.get_or_measure(
            bus,
            S,
            scenario,
            t,
            p_center=float(p_center[t]),
            net=net,
            strict=True,
        )
        for name in _LOSS_COEFF_NAMES:
            assembled[name][0, t] = measured[name]
    return assembled


def _solution_dict(kind, solved, *, has_voltage=False):
    """lower_lp 반환 튜플을 진단용 이름 기반 dict로 바꾼다."""
    if kind == "avg":
        if has_voltage:
            P_net, Q, soc, voltage_sq = solved
        else:
            P_net, Q, soc = solved
            voltage_sq = None
        result = {"P_net": P_net, "Q": Q, "soc": soc}
    else:
        if has_voltage:
            P_net, Q, soc, pk, voltage_sq = solved
        else:
            P_net, Q, soc, pk = solved
            voltage_sq = None
        result = {"P_net": P_net, "Q": Q, "soc": soc, "pk": pk}
    if has_voltage:
        result["v_lindistflow_sq"] = voltage_sq
    return result


def _loss_cost_model_by_time(coeffs, P_net, Q):
    """Return the five-coefficient L_cost contribution in MW for each hour."""
    P_net = np.asarray(P_net, dtype=float).reshape(1, PM.TIME_STEPS)
    Q = np.asarray(Q, dtype=float).reshape(1, PM.TIME_STEPS)
    return (
        np.asarray(coeffs["a_P"], dtype=float) * P_net
        + np.asarray(coeffs["a_Q"], dtype=float) * Q
        + np.asarray(coeffs["b_PP"], dtype=float) * P_net**2
        + np.asarray(coeffs["b_QQ"], dtype=float) * Q**2
        + np.asarray(coeffs["b_PQ"], dtype=float) * P_net * Q
    )[0]


def _recompute_time_diagnostics(S, scenario, X0, X1, C0, C1):
    """Build trigger-design diagnostics from already available X0/X1/C0/C1.

    No solve or power flow is performed here.  ``jnet_delta_won`` is the
    scenario-day loss-cost proxy ``cost(C0,X0)-cost(C1,X1)``; positive means
    the recomputed path improves the monetary loss contribution.  PEAK rows
    use their scenario SMP in the same one-day proxy and are not annualized.
    """
    p0 = np.asarray(X0["P_net"], dtype=float).reshape(1, PM.TIME_STEPS)[0]
    # The first shared coefficient surface is measured at p_center=0 for all t.
    p_shift_mw = np.abs(p0)
    p_shift_frac_s = (
        p_shift_mw / float(S)
        if float(S) > 0.0
        else np.full(PM.TIME_STEPS, np.nan, dtype=float)
    )

    relative_changes = []
    for name in _LOSS_COEFF_NAMES:
        c0 = np.asarray(C0[name], dtype=float).reshape(1, PM.TIME_STEPS)[0]
        c1 = np.asarray(C1[name], dtype=float).reshape(1, PM.TIME_STEPS)[0]
        relative_changes.append(np.abs(c1 - c0) / np.maximum(np.abs(c0), _COEF_REL_EPS))
    coef_delta_max = np.max(np.asarray(relative_changes, dtype=float), axis=0)

    loss_cost_before_mw = _loss_cost_model_by_time(
        C0, X0["P_net"], X0["Q"]
    )
    loss_cost_after_mw = _loss_cost_model_by_time(
        C1, X1["P_net"], X1["Q"]
    )
    smp_won_per_mwh = np.asarray(PM.SMP_PER_MWH[scenario], dtype=float)
    jnet_delta_won = (
        loss_cost_before_mw - loss_cost_after_mw
    ) * smp_won_per_mwh * PM.DT_HOURS
    return {
        "p_shift_mw": p_shift_mw,
        "p_shift_frac_s": p_shift_frac_s,
        "coef_delta_max": coef_delta_max,
        "jnet_delta_won": jnet_delta_won,
    }


def _solve_with_recompute(
    kind,
    bus,
    S,
    E,
    scenario,
    *,
    smp=None,
    load_total=None,
    profile,
    net=None,
    force_q_zero=False,
    return_intermediates=False,
    _c0_grid=None,
):
    """n=1의 측정1→solve1→운영점 측정2→solve2 파이프라인.

    기본 반환은 RECOMPUTE_ENABLED=False에서 X0, True에서 종전 X1의 lower_lp
    튜플이다. ``return_intermediates=True``이면 스위치와 무관하게 X0/X1/X2와
    C0/C1/C2를 담은 dict를 반환한다. ``net``은 진단/복원 경로의 2차 측정 전용이며
    시각 간 재사용한다. ``_c0_grid``는 _solve_unit_schedules가 ALL_DAYS 1차 측정을
    공유하기 위한 내부 인자다.
    """
    if kind not in {"avg", "peak"}:
        raise ValueError(f"kind must be 'avg' or 'peak', got {kind!r}")
    bus = int(bus)
    S = float(S)
    E = float(E)
    scenario = str(scenario)
    profile = np.asarray(profile, dtype=float)
    if profile.shape != (PM.TIME_STEPS,):
        raise ValueError(
            f"profile shape={profile.shape}, expected ({PM.TIME_STEPS},)"
        )

    if _c0_grid is None:
        _c0_grid = loss_coeffs.measure_coeffs_grid(
            bus,
            S,
            scenarios=[scenario],
            times=range(PM.TIME_STEPS),
            p_center=0.0,
            strict=True,
        )
    C0 = _assemble_coeffs_1unit(_c0_grid, scenario)

    if kind == "avg":
        if smp is None:
            raise ValueError("smp is required for kind='avg'")
        smp = np.asarray(smp, dtype=float)
        solved0 = solve_avg(
            S, E, bus, smp, profile, C0,
            force_q_zero=force_q_zero, assert_physics=False,
            return_voltage=return_intermediates,
        )
    else:
        if load_total is None:
            raise ValueError("load_total is required for kind='peak'")
        if smp is None:
            raise ValueError("smp is required for kind='peak'")
        load_total = np.asarray(load_total, dtype=float)
        smp = np.asarray(smp, dtype=float)
        solved0 = solve_peak(
            S, E, bus, load_total, smp, profile, C0,
            force_q_zero=force_q_zero, assert_physics=False,
            return_voltage=return_intermediates,
        )

    # 최적화 기본 경로는 C0로 얻은 X0를 최종 해로 쓴다. 진단 경로는
    # RECOMPUTE_ENABLED와 무관하게 C1/X1/C2/X2를 계속 계산한다.
    if not return_intermediates and not PM.RECOMPUTE_ENABLED:
        return solved0

    X0 = _solution_dict(kind, solved0, has_voltage=return_intermediates)
    measurement_net = build_net() if net is None else net
    C1 = _measure_recomputed_coeffs_1unit(
        bus, S, scenario, X0["P_net"], net=measurement_net
    )

    if kind == "avg":
        solved1 = solve_avg(
            S, E, bus, smp, profile, C1,
            force_q_zero=force_q_zero, assert_physics=False,
            return_voltage=return_intermediates,
        )
    else:
        solved1 = solve_peak(
            S, E, bus, load_total, smp, profile, C1,
            force_q_zero=force_q_zero, assert_physics=False,
            return_voltage=return_intermediates,
        )

    if not return_intermediates:
        return solved1
    X1 = _solution_dict(kind, solved1, has_voltage=True)
    recompute_time_diag = _recompute_time_diagnostics(
        S, scenario, X0, X1, C0, C1
    )

    # X2는 진단 경로에서만 계산한다. 최종 편익 해는 _solve_unit_schedules가
    # RECOMPUTE_ENABLED에 따라 X0/X1 중 선택한다.
    C2 = _measure_recomputed_coeffs_1unit(
        bus, S, scenario, X1["P_net"], net=measurement_net
    )
    if kind == "avg":
        solved2 = solve_avg(
            S, E, bus, smp, profile, C2,
            force_q_zero=force_q_zero, assert_physics=False,
            return_voltage=True,
        )
    else:
        solved2 = solve_peak(
            S, E, bus, load_total, smp, profile, C2,
            force_q_zero=force_q_zero, assert_physics=False,
            return_voltage=True,
        )
    return {
        "kind": kind,
        "scenario": scenario,
        "X0": X0,
        "X1": X1,
        "X2": _solution_dict(kind, solved2, has_voltage=True),
        "C0": C0,
        "C1": C1,
        "C2": C2,
        "recompute_time_diag": recompute_time_diag,
    }


def _solve_unit_schedules(
    b, S, E, base_p_sum, *, return_soc=False, return_detail=False,
    force_q_zero=False,
):
    """ALL_DAYS n=1 스케줄을 구성한다.

    최적화 기본값(RECOMPUTE_ENABLED=False)은 공유 C0의 X0를 최종 해로 쓴다.
    return_detail 진단은 X1/X2까지 계산하되 최종 편익 스케줄은 같은 스위치에
    따라 X0 또는 X1을 선택해 최적화 경로와 일치시킨다.

    향후 다수기에서는 각 기별 active mask를 적용할 수 있게 임계값 판정을 배열로 만든다.
    현재 명령 범위는 n=1이며 비활성 기는 측정·QP를 전부 건너뛴다.
    """
    b = np.asarray(b)
    S = np.asarray(S, dtype=float)
    E = np.asarray(E, dtype=float)
    if not (b.shape == S.shape == E.shape == (1,)):
        raise NotImplementedError(
            "QP coefficient recompute is currently defined for n=1 only"
        )
    bus, s_mva, e_mwh = int(b[0]), float(S[0]), float(E[0])
    active = (S >= S_ACTIVE_MIN) & (E >= E_ACTIVE_MIN)
    if not bool(active[0]):
        unit_p = {
            scenario: np.zeros((1, PM.TIME_STEPS), dtype=float)
            for scenario in PM.ALL_DAYS
        }
        unit_q = {
            scenario: np.zeros((1, PM.TIME_STEPS), dtype=float)
            for scenario in PM.ALL_DAYS
        }
        if not return_soc:
            if return_detail:
                return unit_p, unit_q, {}
            return unit_p, unit_q
        unit_soc = {
            scenario: np.zeros((1, PM.TIME_STEPS + 1), dtype=float)
            for scenario in PM.ALL_DAYS
        }
        if return_detail:
            return unit_p, unit_q, unit_soc, {}
        return unit_p, unit_q, unit_soc

    # 가장 비싼 p_center=0 측정은 AVG/PEAK 전 시나리오가 공유한다.
    c0_grid = loss_coeffs.measure_coeffs_grid(
        bus,
        s_mva,
        scenarios=PM.ALL_DAYS,
        times=range(PM.TIME_STEPS),
        p_center=0.0,
        strict=True,
    )
    recompute_net = (
        build_net() if (return_detail or PM.RECOMPUTE_ENABLED) else None
    )
    unit_p = {}
    unit_q = {}
    unit_soc = {}
    intermediates = {}
    final_solution_key = "X1" if PM.RECOMPUTE_ENABLED else "X0"

    for scenario in PM.AVG_DAYS:
        profile = np.asarray(PM.LOAD[scenario], dtype=float)
        solved = _solve_with_recompute(
            "avg",
            bus,
            s_mva,
            e_mwh,
            scenario,
            smp=np.asarray(PM.SMP_PER_MWH[scenario], dtype=float),
            profile=profile,
            net=recompute_net,
            _c0_grid=c0_grid,
            return_intermediates=return_detail,
            force_q_zero=force_q_zero,
        )
        if return_detail:
            intermediates[scenario] = solved
            P_net, Q, soc = (
                solved[final_solution_key]["P_net"],
                solved[final_solution_key]["Q"],
                solved[final_solution_key]["soc"],
            )
        else:
            P_net, Q, soc = solved
        unit_p[scenario] = P_net
        unit_q[scenario] = Q
        unit_soc[scenario] = soc

    for scenario in PM.PEAK_DAYS:
        profile = np.asarray(PM.LOAD[scenario], dtype=float)
        solved = _solve_with_recompute(
            "peak",
            bus,
            s_mva,
            e_mwh,
            scenario,
            load_total=float(base_p_sum) * profile,
            smp=np.asarray(PM.SMP_PER_MWH[scenario], dtype=float),
            profile=profile,
            net=recompute_net,
            _c0_grid=c0_grid,
            return_intermediates=return_detail,
            force_q_zero=force_q_zero,
        )
        if return_detail:
            intermediates[scenario] = solved
            P_net, Q, soc, _pk = (
                solved[final_solution_key]["P_net"],
                solved[final_solution_key]["Q"],
                solved[final_solution_key]["soc"],
                solved[final_solution_key]["pk"],
            )
        else:
            P_net, Q, soc, _pk = solved
        unit_p[scenario] = P_net
        unit_q[scenario] = Q
        unit_soc[scenario] = soc

    if return_soc:
        if return_detail:
            return unit_p, unit_q, unit_soc, intermediates
        return unit_p, unit_q, unit_soc
    if return_detail:
        return unit_p, unit_q, intermediates
    return unit_p, unit_q


# ============================================================
# 평가 본체
# ============================================================

def _run_schedule_ac(unit_p, unit_q, b, S, E, *, collect_voltage=False):
    """한 스케줄 집합을 기존 AC 평가 규약으로 사후 평가한다."""
    _ensure_worker_state()
    net = _NET
    base_p, base_q = _BASE_P, _BASE_Q
    n = len(b)
    unit_loss_pcs = benefits.loss_pcs(unit_p, unit_q)

    _ensure_sgens(net, n)
    for i in range(n):
        net.sgen.at[i, 'bus'] = int(b[i])

    line_in_service = net.line['in_service'].to_numpy()
    line_rating = net.line['max_i_ka'].to_numpy()
    n_line = len(net.line)
    p_slack_ess = {s: np.zeros(PM.TIME_STEPS) for s in PM.ALL_DAYS}
    loss_ess = {s: np.zeros(PM.TIME_STEPS) for s in PM.ALL_DAYS}
    v_ac = (
        {s: np.zeros((PM.N_BUS, PM.TIME_STEPS)) for s in PM.ALL_DAYS}
        if collect_voltage else None
    )
    v_viol = 0.0
    i_viol = 0.0

    for scenario in PM.ALL_DAYS:
        profile = PM.LOAD[scenario]
        for t in range(PM.TIME_STEPS):
            scale = profile[t]
            net.load['p_mw'] = base_p * scale
            net.load['q_mvar'] = base_q * scale
            for i in range(n):
                net.sgen.at[i, 'p_mw'] = float(
                    unit_p[scenario][i, t] - unit_loss_pcs[scenario][i, t]
                )
                net.sgen.at[i, 'q_mvar'] = float(unit_q[scenario][i, t])

            log_ctx = dict(
                b=np.asarray(b).tolist(),
                S=np.asarray(S).tolist(),
                E=np.asarray(E).tolist(),
                scenario=scenario,
                t=t,
            )
            if not _run_pf_with_retry(net, log_context=log_ctx):
                return dict(diverged=True, diverge_info=log_ctx)

            p_slack_ess[scenario][t] = net.res_ext_grid.p_mw.sum()
            loss_ess[scenario][t] = (
                net.res_line.pl_mw.sum()
                + float(unit_loss_pcs[scenario][:, t].sum())
            )
            voltage = net.res_bus.vm_pu.to_numpy()
            if collect_voltage:
                v_ac[scenario][:, t] = voltage
            v_viol += float(
                np.sum(
                    np.maximum(0.0, voltage - PM.V_MAX)
                    + np.maximum(0.0, PM.V_MIN - voltage)
                )
            )
            i_ratio = np.zeros(n_line)
            i_ratio[line_in_service] = (
                net.res_line.i_ka.to_numpy()[line_in_service]
                / line_rating[line_in_service]
            )
            i_viol += float(
                np.sum(np.maximum(0.0, i_ratio[line_in_service] - 1.0))
            )

    load_sum = {
        s: base_p.sum() * np.asarray(PM.LOAD[s]) for s in PM.ALL_DAYS
    }
    p_ess_total = {s: unit_p[s].sum(axis=0) for s in PM.ALL_DAYS}
    benefits.assert_slack_balance(
        p_slack_ess,
        load_sum,
        loss_ess,
        p_ess_total,
        scenarios=PM.ALL_DAYS,
    )
    return dict(
        diverged=False,
        p_slack_ess=p_slack_ess,
        loss_ess=loss_ess,
        p_ess_total=p_ess_total,
        unit_loss_pcs=unit_loss_pcs,
        v_violation=v_viol,
        i_violation=i_viol,
        v_ac=v_ac,
    )


def _accounting_values(base_flow, p_slack_ess, S, E):
    """기존 benefits 함수만 조립해 새 피크일 회계의 편익 값을 반환한다."""
    smp_mwh = PM.SMP_PER_MWH
    peak_day = benefits.select_peak_day(
        {s: p_slack_ess[s] for s in PM.PEAK_DAYS}
    )
    adjusted_season = benefits.PEAK_TO_SEASON[peak_day]
    b_energy_avg = benefits.b_energy_adjusted(
        {s: base_flow['p_slack'][s] for s in PM.AVG_DAYS},
        {s: p_slack_ess[s] for s in PM.AVG_DAYS},
        smp_mwh,
        PM.N_WEEKDAYS,
        adjusted_season,
    )
    b_energy_peak_day_val = benefits.b_energy_peak_day(
        {s: base_flow['p_slack'][s] for s in PM.PEAK_DAYS},
        {s: p_slack_ess[s] for s in PM.PEAK_DAYS},
        smp_mwh,
        peak_day,
    )
    b_energy_val = b_energy_avg + b_energy_peak_day_val
    b_defer_val = benefits.b_defer(
        {s: base_flow['p_slack'][s] for s in PM.PEAK_DAYS},
        {s: p_slack_ess[s] for s in PM.PEAK_DAYS},
    )
    s_total = float(np.sum(S))
    e_total = float(np.sum(E))
    return dict(
        peak_day=peak_day,
        adjusted_season=adjusted_season,
        b_energy_avg=b_energy_avg,
        b_energy_peak_day=b_energy_peak_day_val,
        b_energy=b_energy_val,
        b_defer=b_defer_val,
        j_net=benefits.j_net(
            b_energy_val, b_defer_val, s_total, e_total
        ),
    )


def evaluate_particle(
    x, return_detail=False, collect_diagnostics=False, force_q_zero=False
):
    """입자 x(3n차원, C.6-3 이후) 평가. 기본 반환: fitness(float, PSO 최소화용).
    return_detail=True면 편익 분해·위반량·발산정보·스케줄을 담은 dict를 반환한다(8절 후처리,
    디버깅용). collect_diagnostics=True를 함께 주면 X0/X1/X2·전압·추가 AC 진단을 수행한다.
    PSO의 return_detail=True 경로는 collect_diagnostics=False라 이 추가 비용을 치르지 않는다.
    """
    _ensure_worker_state()
    net = _NET
    base_p, base_q = _BASE_P, _BASE_Q
    base_flow = _get_base_flow()

    b, S, E = _parse_particle(x)
    n = len(b)

    assert not collect_diagnostics or return_detail, (
        'collect_diagnostics=True는 return_detail=True와 함께 써야 함'
    )
    diagnostic_error = ''
    if collect_diagnostics:
        try:
            unit_p, unit_q, intermediates = _solve_unit_schedules(
                b, S, E, base_p.sum(), return_detail=True,
                force_q_zero=force_q_zero,
            )
        except Exception as exc:
            # 계측(X2/전압) 실패는 최적해 평가를 죽이지 않는다. 기존 X1 경로로 재시도한다.
            diagnostic_error = f'{type(exc).__name__}: {exc}'
            unit_p, unit_q = _solve_unit_schedules(
                b, S, E, base_p.sum(), force_q_zero=force_q_zero
            )
            intermediates = None
    else:
        unit_p, unit_q = _solve_unit_schedules(
            b, S, E, base_p.sum(), force_q_zero=force_q_zero
        )
        intermediates = None
    ac_result = _run_schedule_ac(
        unit_p, unit_q, b, S, E, collect_voltage=collect_diagnostics
    )
    if ac_result['diverged']:
        # CLAUDE.md 지시 (d): 예외를 위로 던지지 않고 스칼라 페널티로 확정 처리 (PSO가 죽지 않게).
        if return_detail:
            return dict(
                fitness=PM.PENALTY_DIVERGE,
                diverged=True,
                diverge_info=ac_result['diverge_info'],
            )
        return float(PM.PENALTY_DIVERGE)
    p_slack_ess = ac_result['p_slack_ess']
    loss_ess = ac_result['loss_ess']
    p_ess_total = ac_result['p_ess_total']
    unit_loss_pcs = ac_result['unit_loss_pcs']
    v_viol = ac_result['v_violation']
    i_viol = ac_result['i_violation']

    smp_mwh = PM.SMP_PER_MWH
    accounting = _accounting_values(base_flow, p_slack_ess, S, E)
    peak_day = accounting['peak_day']
    adjusted_season = accounting['adjusted_season']
    b_energy_avg = accounting['b_energy_avg']
    b_energy_peak_day_val = accounting['b_energy_peak_day']
    b_energy_val = accounting['b_energy']
    b_defer_val = accounting['b_defer']
    s_total = float(S.sum())
    e_total = float(E.sum())
    j_net_val = accounting['j_net']

    fitness = -j_net_val + PM.LAMBDA_V * v_viol + PM.LAMBDA_LINE * i_viol
    fitness = float(fitness)

    if not return_detail:
        return fitness

    p_ch_agg = {s: np.maximum(-unit_p[s], 0.0).sum(axis=0) for s in PM.AVG_DAYS}
    p_dis_agg = {s: np.maximum(unit_p[s], 0.0).sum(axis=0) for s in PM.AVG_DAYS}
    # b_arb/b_loss는 AVG_DAYS 분해이므로 확정 피크일의 별도 에너지 항을 포함하지 않는다.
    # b_energy_avg와 같은 조정 가중을 써야 분해 항등식이 유지된다.
    n_weekdays_adjusted = {
        s: float(PM.N_WEEKDAYS[s]) - (1.0 if s == adjusted_season else 0.0)
        for s in PM.AVG_DAYS
    }
    b_arb_val = benefits.b_arb(
        p_ch_agg, p_dis_agg, smp_mwh, n_weekdays_adjusted
    )
    b_loss_val = benefits.b_loss(
        {s: base_flow['loss'][s] for s in PM.AVG_DAYS},
        {s: loss_ess[s] for s in PM.AVG_DAYS},
        smp_mwh,
        n_weekdays_adjusted,
    )
    p_ch_peak = {
        peak_day: np.maximum(-unit_p[peak_day], 0.0).sum(axis=0)
    }
    p_dis_peak = {
        peak_day: np.maximum(unit_p[peak_day], 0.0).sum(axis=0)
    }
    b_arb_total_val = benefits.b_arb_total(
        p_ch_agg,
        p_dis_agg,
        smp_mwh,
        PM.N_WEEKDAYS,
        adjusted_season,
        peak_day,
        p_ch_peak,
        p_dis_peak,
    )
    b_loss_total_val = benefits.b_loss_total(
        {s: base_flow['loss'][s] for s in PM.AVG_DAYS},
        {s: loss_ess[s] for s in PM.AVG_DAYS},
        smp_mwh,
        PM.N_WEEKDAYS,
        adjusted_season,
        peak_day,
        {peak_day: base_flow['loss'][peak_day]},
        {peak_day: loss_ess[peak_day]},
    )

    recompute_jnet_delta = float('nan')
    recompute_convergence_ratio = float('nan')
    recompute_max_p_shift = float('nan')
    recompute_time_diag = None
    v_lindistflow_sq = None
    if intermediates:
        try:
            recompute_time_diag = {
                s: {
                    name: np.asarray(values, dtype=float)
                    for name, values in intermediates[s][
                        'recompute_time_diag'
                    ].items()
                }
                for s in PM.ALL_DAYS
            }
            unit_p_x0 = {
                s: np.asarray(intermediates[s]['X0']['P_net'])
                for s in PM.ALL_DAYS
            }
            unit_q_x0 = {
                s: np.asarray(intermediates[s]['X0']['Q'])
                for s in PM.ALL_DAYS
            }
            unit_p_x1 = {
                s: np.asarray(intermediates[s]['X1']['P_net'])
                for s in PM.ALL_DAYS
            }
            unit_q_x1 = {
                s: np.asarray(intermediates[s]['X1']['Q'])
                for s in PM.ALL_DAYS
            }
            unit_p_x2 = {
                s: np.asarray(intermediates[s]['X2']['P_net'])
                for s in PM.ALL_DAYS
            }
            unit_q_x2 = {
                s: np.asarray(intermediates[s]['X2']['Q'])
                for s in PM.ALL_DAYS
            }
            final_solution_key = 'X1' if PM.RECOMPUTE_ENABLED else 'X0'
            v_lindistflow_sq = {
                s: np.asarray(
                    intermediates[s][final_solution_key]['v_lindistflow_sq'],
                    dtype=float,
                )
                for s in PM.ALL_DAYS
            }
            recompute_max_p_shift = max(
                float(
                    np.max(
                        np.abs(
                            np.asarray(intermediates[s]['X1']['P_net'])
                            - unit_p_x0[s]
                        )
                    )
                )
                for s in PM.ALL_DAYS
            )

            # ac_result는 스위치에 따른 실제 최종 경로의 AC 결과다. 이를 X0 또는
            # X1에 재사용해 진단 AC 호출 수를 종전과 같은 총 3회로 유지한다.
            if PM.RECOMPUTE_ENABLED:
                ac_x1 = ac_result
                ac_x0 = _run_schedule_ac(unit_p_x0, unit_q_x0, b, S, E)
            else:
                ac_x0 = ac_result
                ac_x1 = _run_schedule_ac(unit_p_x1, unit_q_x1, b, S, E)
            ac_x2 = _run_schedule_ac(unit_p_x2, unit_q_x2, b, S, E)
            if ac_x0['diverged'] or ac_x1['diverged'] or ac_x2['diverged']:
                diagnostic_error = (
                    diagnostic_error
                    or 'X0/X1/X2 diagnostic AC power flow diverged'
                )
            else:
                j0 = _accounting_values(
                    base_flow, ac_x0['p_slack_ess'], S, E
                )['j_net']
                j1 = _accounting_values(
                    base_flow, ac_x1['p_slack_ess'], S, E
                )['j_net']
                j2 = _accounting_values(
                    base_flow, ac_x2['p_slack_ess'], S, E
                )['j_net']
                delta_10 = float(j1 - j0)
                delta_21 = float(j2 - j1)
                recompute_jnet_delta = delta_10
                if abs(delta_10) > 0.0:
                    recompute_convergence_ratio = abs(delta_21) / abs(delta_10)
        except Exception as exc:
            diagnostic_error = (
                diagnostic_error
                or f'{type(exc).__name__}: {exc}'
            )

    return dict(
        fitness=fitness,
        diverged=False,
        j_net=j_net_val,
        b_energy=b_energy_val,
        b_energy_avg=b_energy_avg,
        b_energy_peak_day=b_energy_peak_day_val,
        peak_day=peak_day,
        adjusted_season=adjusted_season,
        b_defer=b_defer_val,
        b_arb=b_arb_val,
        b_loss=b_loss_val,
        b_arb_total=b_arb_total_val,
        b_loss_total=b_loss_total_val,
        decomposition_ok=benefits.check_b_energy_decomposition(
            b_arb_total_val, b_loss_total_val, b_energy_val
        ),
        cost=benefits.total_cost(s_total, e_total),
        v_violation=v_viol,
        i_violation=i_viol,
        p_slack_ess=p_slack_ess,
        loss_ess=loss_ess,
        p_ess_total=p_ess_total,
        unit_p=unit_p,
        unit_q=unit_q,
        v_lindistflow_sq=v_lindistflow_sq,
        v_ac=ac_result['v_ac'],
        recompute_jnet_delta=recompute_jnet_delta,
        recompute_convergence_ratio=recompute_convergence_ratio,
        recompute_max_p_shift=recompute_max_p_shift,
        recompute_time_diag=recompute_time_diag,
        diagnostic_error=diagnostic_error,
        b=b, S=S, E=E,
    )


def evaluate_batch(X):
    """pso_core 벡터화 인터페이스: (n_particles, n_dims) -> (n_particles,).
    단일 프로세스 순차 평가(병렬화는 main.py의 Pool 몫 - CLAUDE.md 7절 병렬화 절 참조)."""
    X = np.asarray(X, dtype=float)
    return np.array([evaluate_particle(x) for x in X], dtype=float)
