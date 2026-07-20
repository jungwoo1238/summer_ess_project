"""평가 지휘자 (CLAUDE.md 부록A, 7절). lower_lp + 조류계산 + benefits + 페널티를 엮는다.

워커 초기화 패턴(전역 NET/BASE_P/BASE_Q, CLAUDE.md 7절)을 따른다. init_worker()가
build_net 1회 + 기저(ESS 없음) 조류계산 ALL_DAYS x 24h(120회, 입자와 무관하므로 워커당 1회만)를
캐싱한다. 이후 evaluate_particle 1회당 발생하는 조류계산은 ESS 주입 상태의 120회뿐이다
(7절 실측 기준 1.178초/회).

n기 일반형: 입자 x는 4n 차원 (b_i,S_i,E_i,q_ratio_i) x n. 각 기는 독립적으로 자기 LP를 풀고
(CLAUDE.md 10절 "독립최적화"), 조류계산에서만 함께 만난다(동일 버스 중복배치는 sgen 2개로
별도 생성, 병합 안 함 - (a)방식).
"""

import numpy as np
import pandapower as pp

import params as PM
from build_net import build_net
from lower_lp import solve_avg, solve_peak
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
    """x(4n차원) -> b(int,버스),S(MVA),E(MWh),q_ratio. 경계 clamp는 방어적으로 여기서도
    수행한다(pso_core가 이미 clamp하지만 evaluate_particle을 테스트 등에서 직접 호출할 수 있어서)."""
    x = np.asarray(x, dtype=float)
    assert x.size > 0 and x.size % 4 == 0, f'입자 차원 {x.size}는 4의 배수(4n, n>=1)여야 함'
    n = x.size // 4
    x4 = x.reshape(n, 4)

    b = np.clip(np.round(x4[:, 0]), PM.B_BOUNDS[0], PM.B_BOUNDS[1]).astype(int)
    S = np.clip(x4[:, 1], PM.S_BOUNDS[0], PM.S_BOUNDS[1])
    E = np.clip(x4[:, 2], PM.E_BOUNDS[0], PM.E_BOUNDS[1])
    q = np.clip(x4[:, 3], PM.Q_RATIO_BOUNDS[0], PM.Q_RATIO_BOUNDS[1])
    return b, S, E, q


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
# 하위 LP: 기(unit)별 독립 스케줄 (CLAUDE.md 10절 "독립최적화")
# ============================================================

def _solve_unit_schedules(base_p_sum, S, E):
    """각 기 i에 대해 ALL_DAYS 전 시나리오 스케줄 P_net[s][i,:]을 독립적으로 푼다.
    AVG_DAYS는 solve_avg(조달비 대리), PEAK_DAYS는 solve_peak(자기 피크 대리) - CLAUDE.md 2절.

    반환: dict[scenario] -> (n, T) ndarray. S_i<=0 또는 E_i<=0인 기는 0 스케줄(LP 미호출).
    본실험 속도를 위해 assert_physics=False로 호출한다(물리보존 검증은 lower_lp 자체 테스트에서
    이미 끝남 - CLAUDE.md 7절 LP검증 항목은 test_lp.py가 담당).
    """
    n = len(S)
    unit_p = {s: np.zeros((n, PM.TIME_STEPS)) for s in PM.ALL_DAYS}

    for i in range(n):
        if S[i] <= 0 or E[i] <= 0:
            continue
        for s in PM.AVG_DAYS:
            smp = np.asarray(PM.SMP[s])
            P_net, _ = solve_avg(S[i], E[i], smp, assert_physics=False)
            unit_p[s][i] = P_net
        for s in PM.PEAK_DAYS:
            load_mw = base_p_sum * np.asarray(PM.LOAD[s])
            P_net, _, _pk = solve_peak(S[i], E[i], load_mw, assert_physics=False)
            unit_p[s][i] = P_net

    return unit_p


def _unit_reactive(S, unit_p):
    """Q[t] = q_ratio * sqrt(S^2 - P[t]^2), 기별로. sqrt 인자는 수치오차로 음수가 될 수 있어
    0으로 클립한다(|P|<=S는 LP 제약이라 이론상 항상 >=0 - CLAUDE.md 2절 대수적 항등).
    q_ratio는 evaluate_particle에서 곱해진다(이 함수는 sqrt(S^2-P^2) 항만 계산)."""
    return {
        s: np.sqrt(np.maximum(S[:, None] ** 2 - unit_p[s] ** 2, 0.0))
        for s in PM.ALL_DAYS
    }


# ============================================================
# 평가 본체
# ============================================================

def evaluate_particle(x, return_detail=False):
    """입자 x(4n차원) 평가. 기본 반환: fitness(float, PSO 최소화용).
    return_detail=True면 편익 분해·위반량·발산정보·스케줄을 담은 dict를 반환한다(8절 후처리,
    디버깅용 - PSO 루프 자체는 스칼라 경로만 쓴다).
    """
    _ensure_worker_state()
    net = _NET
    base_p, base_q = _BASE_P, _BASE_Q
    base_flow = _get_base_flow()

    b, S, E, q = _parse_particle(x)
    n = len(b)

    unit_p = _solve_unit_schedules(base_p.sum(), S, E)
    unit_q_base = _unit_reactive(S, unit_p)
    unit_q = {s: q[:, None] * unit_q_base[s] for s in PM.ALL_DAYS}

    _ensure_sgens(net, n)
    for i in range(n):
        net.sgen.at[i, 'bus'] = int(b[i])

    line_in_service = net.line['in_service'].to_numpy()
    line_rating = net.line['max_i_ka'].to_numpy()
    n_line = len(net.line)

    p_slack_ess = {s: np.zeros(PM.TIME_STEPS) for s in PM.ALL_DAYS}
    loss_ess = {s: np.zeros(PM.TIME_STEPS) for s in PM.ALL_DAYS}
    v_viol = 0.0
    i_viol = 0.0
    diverged = False
    diverge_info = None

    for s in PM.ALL_DAYS:
        profile = PM.LOAD[s]
        for t in range(PM.TIME_STEPS):
            scale = profile[t]
            net.load['p_mw'] = base_p * scale
            net.load['q_mvar'] = base_q * scale
            for i in range(n):
                net.sgen.at[i, 'p_mw'] = float(unit_p[s][i, t])
                net.sgen.at[i, 'q_mvar'] = float(unit_q[s][i, t])

            log_ctx = dict(b=b.tolist(), S=S.tolist(), E=E.tolist(), scenario=s, t=t)
            ok = _run_pf_with_retry(net, log_context=log_ctx)
            if not ok:
                diverged = True
                diverge_info = log_ctx
                break

            p_slack_ess[s][t] = net.res_ext_grid.p_mw.sum()
            loss_ess[s][t] = net.res_line.pl_mw.sum()

            v = net.res_bus.vm_pu.to_numpy()
            v_viol += float(np.sum(np.maximum(0.0, v - PM.V_MAX) + np.maximum(0.0, PM.V_MIN - v)))

            i_ratio = np.zeros(n_line)
            i_ratio[line_in_service] = (
                net.res_line.i_ka.to_numpy()[line_in_service] / line_rating[line_in_service]
            )
            i_viol += float(np.sum(np.maximum(0.0, i_ratio[line_in_service] - 1.0)))
        if diverged:
            break

    if diverged:
        # CLAUDE.md 지시 (d): 예외를 위로 던지지 않고 스칼라 페널티로 확정 처리 (PSO가 죽지 않게).
        if return_detail:
            return dict(fitness=PM.PENALTY_DIVERGE, diverged=True, diverge_info=diverge_info)
        return float(PM.PENALTY_DIVERGE)

    load_sum = {s: base_p.sum() * np.asarray(PM.LOAD[s]) for s in PM.ALL_DAYS}
    p_ess_total = {s: unit_p[s].sum(axis=0) for s in PM.ALL_DAYS}

    # 실데이터 수지 검증 (CLAUDE.md 지시 (i)): P_slack = SigmaLoad + Loss - P_ESS.
    # sgen에 실제로 주입한 값(unit_p, 위 루프에서 net.sgen.at[i,'p_mw']로 그대로 넣은 값)을
    # 그대로 넘긴다 - LP 원본과 다른 배열을 넘기면 이 검증의 취지가 무너진다.
    benefits.assert_slack_balance(p_slack_ess, load_sum, loss_ess, p_ess_total, scenarios=PM.ALL_DAYS)

    smp_mwh = PM.SMP_PER_MWH
    b_energy_val = benefits.b_energy(
        {s: base_flow['p_slack'][s] for s in PM.AVG_DAYS},
        {s: p_slack_ess[s] for s in PM.AVG_DAYS},
        smp_mwh, PM.N_WEEKDAYS,
    )
    b_defer_val = benefits.b_defer(
        {s: base_flow['p_slack'][s] for s in PM.PEAK_DAYS},
        {s: p_slack_ess[s] for s in PM.PEAK_DAYS},
    )

    s_total = float(S.sum())
    e_total = float(E.sum())
    j_net_val = benefits.j_net(b_energy_val, b_defer_val, s_total, e_total)

    fitness = -j_net_val + PM.LAMBDA_V * v_viol + PM.LAMBDA_LINE * i_viol
    fitness = float(fitness)

    if not return_detail:
        return fitness

    p_ch_agg = {s: np.maximum(-unit_p[s], 0.0).sum(axis=0) for s in PM.AVG_DAYS}
    p_dis_agg = {s: np.maximum(unit_p[s], 0.0).sum(axis=0) for s in PM.AVG_DAYS}
    b_arb_val = benefits.b_arb(p_ch_agg, p_dis_agg, smp_mwh, PM.N_WEEKDAYS)
    b_loss_val = benefits.b_loss(
        {s: base_flow['loss'][s] for s in PM.AVG_DAYS},
        {s: loss_ess[s] for s in PM.AVG_DAYS},
        smp_mwh, PM.N_WEEKDAYS,
    )

    return dict(
        fitness=fitness,
        diverged=False,
        j_net=j_net_val,
        b_energy=b_energy_val,
        b_defer=b_defer_val,
        b_arb=b_arb_val,
        b_loss=b_loss_val,
        decomposition_ok=benefits.check_b_energy_decomposition(b_arb_val, b_loss_val, b_energy_val),
        cost=benefits.total_cost(s_total, e_total),
        v_violation=v_viol,
        i_violation=i_viol,
        p_slack_ess=p_slack_ess,
        loss_ess=loss_ess,
        p_ess_total=p_ess_total,
        unit_p=unit_p,
        unit_q=unit_q,
        b=b, S=S, E=E, q=q,
    )


def evaluate_batch(X):
    """pso_core 벡터화 인터페이스: (n_particles, n_dims) -> (n_particles,).
    단일 프로세스 순차 평가(병렬화는 main.py의 Pool 몫 - CLAUDE.md 7절 병렬화 절 참조)."""
    X = np.asarray(X, dtype=float)
    return np.array([evaluate_particle(x) for x in X], dtype=float)
