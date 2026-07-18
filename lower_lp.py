"""하위 분리 LP (CLAUDE.md 2절).
AVG_DAYS: solve_avg - 조달비(슬랙 유입 대리) 최소화.
PEAK_DAYS: solve_peak - 자기 피크 pk_s 최소화. 최종 이연은 사후 max(pk_summer, pk_winter).
시나리오 간 독립(분리 LP = 통합 LP, 수학적으로 동일 - CLAUDE.md 2절 결정 근거 참조).

Q는 상위(PSO) 변수라 이 모듈은 다루지 않는다: PCS 원(circle) 제약 sqrt(P^2+Q^2)<=S_rated는
Q = q_ratio * sqrt(S_rated^2 - P^2) 구성상 q_ratio 값에 무관하게 항상 |P| <= S_rated로 귀결된다
(P^2 + Q^2 = P^2*(1-q_ratio^2) + q_ratio^2*S_rated^2 <= S_rated^2 <=> P^2 <= S_rated^2, q_ratio<1일 때;
 q_ratio=1이면 애초에 Q가 실수이려면 |P|<=S_rated가 정의역 조건). 따라서 LP는 |P|<=S_rated만 강제하면
원 제약을 대수적으로 이미 만족한다 (축약이 아니라 항등식).
"""

import numpy as np
import cvxpy as cp

import params as PM


def _empty_result(T, soc_init_frac, E_mwh):
    return np.zeros(T), np.full(T + 1, soc_init_frac * E_mwh)


def _soc_constraints(P_ch, P_dis, soc, T, E_mwh, eta_c, eta_d, dt, self_discharge,
                      soc_init_frac, soc_min_frac, soc_max_frac):
    """soc(cvxpy 변수)는 MWh 절대량 규약 (CLAUDE.md 2절 SOC 제약, 부록A #1).
    soc_init_frac/soc_min_frac/soc_max_frac은 비율(0~1)로 받아 여기서 E_mwh를 곱해 변환."""
    cons = [
        soc[0] == soc_init_frac * E_mwh,
        soc[T] == soc_init_frac * E_mwh,
        soc >= soc_min_frac * E_mwh,
        soc <= soc_max_frac * E_mwh,
    ]
    for t in range(T):
        cons.append(
            soc[t + 1] == soc[t] * (1 - self_discharge)
            + eta_c * P_ch[t] * dt - P_dis[t] / eta_d * dt
        )
    return cons


def _check_solved(prob, name):
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f'{name}: LP 미해결 (status={prob.status})')


def _assert_physics(p_ch, p_dis, soc, S_mw, E_mwh, eta_c, eta_d, dt, self_discharge,
                     soc_init_frac, soc_min_frac, soc_max_frac, tol=1e-4):
    """물리 보존 assert (CLAUDE.md 7절 LP검증#2). 개발 중 기본 켜짐(assert_physics=True),
    본실험 시 호출부에서 assert_physics=False로 꺼서 속도 확보."""
    T = len(p_ch)
    soc_init_mwh = soc_init_frac * E_mwh
    soc_min_mwh = soc_min_frac * E_mwh
    soc_max_mwh = soc_max_frac * E_mwh
    assert np.isclose(soc[0], soc_init_mwh, atol=tol)
    assert np.isclose(soc[T], soc_init_mwh, atol=tol)
    assert np.all(soc >= soc_min_mwh - tol) and np.all(soc <= soc_max_mwh + tol)
    assert np.all(p_ch <= S_mw + tol) and np.all(p_dis <= S_mw + tol)
    assert np.all(p_ch >= -tol) and np.all(p_dis >= -tol)

    # 에너지수지: SOC 재귀식을 직접 재적분해 LP 결과와 일치하는지 확인 (sqrt(P^2+Q^2)<=S는
    # 모듈 docstring의 대수적 귀결에 의해 |P|<=S 체크로 충분, 위에서 이미 확인)
    soc_recheck = np.empty(T + 1)
    soc_recheck[0] = soc[0]
    for t in range(T):
        soc_recheck[t + 1] = soc_recheck[t] * (1 - self_discharge) + (
            eta_c * p_ch[t] * dt - p_dis[t] / eta_d * dt
        )
    assert np.allclose(soc_recheck, soc, atol=tol), '에너지수지(SOC 재귀식) 불일치'


def solve_avg(
    S_mw, E_mwh, smp,
    *, eta_c=PM.ETA_C, eta_d=PM.ETA_D, dt=PM.DT_HOURS, T=PM.TIME_STEPS,
    self_discharge=PM.SELF_DISCHARGE_HOURLY,
    soc_init=PM.SOC_INIT_FRAC, soc_min=PM.SOC_MIN_FRAC, soc_max=PM.SOC_MAX_FRAC,
    assert_physics=True,
):
    """AVG_DAYS: 조달비(슬랙 유입 대리) 최소화.
    min Σ_t SMP[t]*(P_ch[t]-P_dis[t])*dt + 1e-6*Σ_t(P_ch[t]+P_dis[t])  (CLAUDE.md 2절)

    smp: 길이 T 배열 (원/kWh, 절대값, 정규화 안 함).
    soc_init/soc_min/soc_max: 비율(0~1). 함수 내부에서 E_mwh를 곱해 절대량(MWh)으로 사용.
    반환: (P_net, soc)
      P_net: 길이 T (MW, +방전/-충전)
      soc:   길이 T+1 (SOC[0]=SOC[T]=soc_init*E_mwh, 단위 MWh - 부록A #1 규약)
    """
    smp = np.asarray(smp, dtype=float)
    assert smp.shape == (T,), f'smp shape {smp.shape} != ({T},)'

    if S_mw <= 0 or E_mwh <= 0:
        return _empty_result(T, soc_init, E_mwh)

    P_ch = cp.Variable(T, nonneg=True)
    P_dis = cp.Variable(T, nonneg=True)
    soc = cp.Variable(T + 1)

    constraints = [P_ch <= S_mw, P_dis <= S_mw]
    constraints += _soc_constraints(
        P_ch, P_dis, soc, T, E_mwh, eta_c, eta_d, dt, self_discharge, soc_init, soc_min, soc_max
    )

    # 1e-6*(P_ch+P_dis) 정규화항: SMP 평탄·eta=1처럼 주 목적함수가 degenerate해질 때
    # (동시충방전을 포함한 여러 해가 같은 목적값을 갖는 상황) 동시충방전 해를 배제.
    # 주 목적함수 대비 6자리 작아 정상적인(스프레드 있는) 차익거래 최적해는 왜곡하지 않음.
    objective = cp.Minimize(
        cp.sum(cp.multiply(smp, P_ch - P_dis)) * dt + 1e-6 * cp.sum(P_ch + P_dis)
    )
    prob = cp.Problem(objective, constraints)
    prob.solve()
    _check_solved(prob, 'solve_avg')

    p_ch_val, p_dis_val, soc_val = P_ch.value, P_dis.value, soc.value
    if assert_physics:
        _assert_physics(p_ch_val, p_dis_val, soc_val, S_mw, E_mwh, eta_c, eta_d, dt,
                         self_discharge, soc_init, soc_min, soc_max)

    return p_dis_val - p_ch_val, soc_val


def solve_peak(
    S_mw, E_mwh, load_mw,
    *, eta_c=PM.ETA_C, eta_d=PM.ETA_D, dt=PM.DT_HOURS, T=PM.TIME_STEPS,
    self_discharge=PM.SELF_DISCHARGE_HOURLY,
    soc_init=PM.SOC_INIT_FRAC, soc_min=PM.SOC_MIN_FRAC, soc_max=PM.SOC_MAX_FRAC,
    assert_physics=True,
):
    """PEAK_DAYS: 자기 피크 pk_s = max_t(load[t]-P_net[t]) 최소화. (CLAUDE.md 2절)
    최종 이연 산정은 사후에 max(pk_summer, pk_winter) (호출부 책임, 여기선 단일 시나리오만).

    ★ 주의: 반환하는 pk_s는 손실을 무시한 "부하단" 대리값 max_t(load[t]-P_net[t])이며,
    B_defer 산정에 직접 쓰지 말 것. 3절 B_defer는 "슬랙 유입 피크" max_t(P_slack)이고
    P_slack = Σ부하 + Loss - P_ESS라 pk_s와는 손실만큼 다르다. B_defer는 반드시 이 함수가
    반환한 스케줄(P_net)을 사후 조류계산에 넣어 얻은 슬랙 피크로 계산해야 한다
    (CLAUDE.md 2절 "★ 주의", 3절 B_defer). pk_s는 스케줄을 정하기 위한 내부 대리값일 뿐이다.

    load_mw: 길이 T 배열, 해당 시나리오 시간별 피더 총부하(MW, 유효전력).
    soc_init/soc_min/soc_max: 비율(0~1). 함수 내부에서 E_mwh를 곱해 절대량(MWh)으로 사용.
    반환: (P_net, soc, pk_s)
      P_net: 길이 T (MW, +방전/-충전)
      soc:   길이 T+1 (단위 MWh - 부록A #1 규약)
      pk_s:  스칼라 (MW), 이 시나리오의 "부하단" 대리 피크 (위 경고 참조).
    """
    load_mw = np.asarray(load_mw, dtype=float)
    assert load_mw.shape == (T,), f'load_mw shape {load_mw.shape} != ({T},)'

    if S_mw <= 0 or E_mwh <= 0:
        P_net, soc_val = _empty_result(T, soc_init, E_mwh)
        return P_net, soc_val, float(load_mw.max())

    P_ch = cp.Variable(T, nonneg=True)
    P_dis = cp.Variable(T, nonneg=True)
    soc = cp.Variable(T + 1)
    pk = cp.Variable()

    constraints = [P_ch <= S_mw, P_dis <= S_mw, pk >= load_mw - P_dis + P_ch]
    constraints += _soc_constraints(
        P_ch, P_dis, soc, T, E_mwh, eta_c, eta_d, dt, self_discharge, soc_init, soc_min, soc_max
    )

    objective = cp.Minimize(pk)
    prob = cp.Problem(objective, constraints)
    prob.solve()
    _check_solved(prob, 'solve_peak')

    p_ch_val, p_dis_val, soc_val = P_ch.value, P_dis.value, soc.value
    if assert_physics:
        _assert_physics(p_ch_val, p_dis_val, soc_val, S_mw, E_mwh, eta_c, eta_d, dt,
                         self_discharge, soc_init, soc_min, soc_max)

    return p_dis_val - p_ch_val, soc_val, float(pk.value)
