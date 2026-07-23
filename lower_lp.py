"""하위 LP (CLAUDE.md 2절, 부록C - LinDistFlow 본체 편입, C.6-3/C.6-4).

★ 다수기 n대를 하나의 조인트 LP로 푼다(단일기 루프가 아니다). 이유: LinDistFlow 전압
유도항(부록C.4-(1))은 버스별 전압이 "그 버스 하류의 전 기 주입 합"에 의존하므로, 기별로
LP를 따로 풀면 다른 기의 주입을 볼 수 없다 - n기를 한 조류식으로 묶어야 한다(부록C.4 서두).
solve_avg: AVG_DAYS 조달비 대리 최소화. solve_peak: PEAK_DAYS **시스템 전체** 피크 pk
최소화(2절/7-A절 "기수 축 d* 중복적용" 결함 해소 - 통합 solve_peak, C.6-3 3절).

정식화 자산(scripts/test_lindistflow.py 게이트 실측 검증 완료, 2026-07-23 - 재유도 금지):
  V^2 원형 변수, Baran-Wu load-positive 부호규약(P_ij = 하류부하 - 하류발전, 통상 양수),
  v_j = v_i - 2*(r_ij*P_ij + x_ij*Q_ij), Z_BASE_OHM=VN_KV^2/S_BASE_MVA, v_slack=SLACK_VM_PU^2.
  게이트 실측: 방문영역(1092케이스) 글로벌 max err_v = 0.001421 pu (기대범위 0.001~0.003 이내).

전압은 하드 제약이 아니라 목적함수 페널티다(부록C.4-(1)). 하드 제약으로 넣으면 무거운
시나리오에서 infeasible이 될 수 있고(PSO 평가가 죽는다), 근사오차가 가부 판정을 뒤집을
위험이 있다. 페널티는 "얼마나 세게 밀지"에만 영향을 주고 가부를 뒤집지 않는다 - 최종
제약 판정은 여전히 사후 AC 조류계산 + evaluate.py의 LAMBDA_V가 담당한다(변경 없음).
★ v는 V^2 공간이므로 페널티 임계도 V_SQ_MIN/V_SQ_MAX(제곱값)를 쓴다.

PCS 원 제약: force_q_zero=False(기본, 시변 Q 자유)일 때는 다각형 내접 근사(부록C.4-(2),
POLY_N=12, QCP/SOCP 회피 - LP 유지)를 P_ch<=S/P_dis<=S(개별 PCS 출력한계, Q 무관하게
항상 성립)와 **함께** 건다. force_q_zero=True(C.6-4 회귀검증·8절-6 대체용)일 때는
Q==0을 강제하고 다각형은 걸지 않는다 - Q=0이면 sqrt(P^2+Q^2)<=S가 이미 |P|<=S와 대수적으로
동치이므로(2절 원 구단, 편입 전 lower_lp.py 모듈 docstring과 동일 논거) 다각형 근사가
필요 없고, 오히려 다각형을 걸면 cos(pi/12)만큼 불필요하게 좁아져 **편입 전(P_ch<=S,
P_dis<=S만 있던 시절) 결과와의 정확한 수치 일치가 깨진다**(C.6-4 회귀검증 요구사항).

성능: cvxpy Parameter 기반 DPP로 (kind, n, force_q_zero, eta_c, eta_d, self_discharge,
soc_init, soc_min, soc_max)별 Problem을 최초 호출 시 1회만 만들고 캐싱한다(부록C.4-(4), C.5).
S/E/버스배치(bus_onehot)/SMP/부하(profile)/mu_volt는 Parameter라 평가마다 재컴파일 없이
.value만 갱신한다 - eta_c 등 물리상수는 나눗셈에 쓰여 Parameter화하면 DPP가 깨지므로
(Parameter로 나누는 것은 affine이 아님) 캐시 키에 포함하는 쪽을 택했다(실전에서는 항상
PM 기본값 하나만 쓰이므로 캐시가 늘어나지 않는다 - 물리상수를 바꿔 부르는 것은 test_lp.py뿐).
"""

import numpy as np
import cvxpy as cp

import params as PM
from build_net import build_net


# ============================================================
# 0. 계통 토폴로지 + 버스별 기저부하 (모듈 전역, 지연 1회 계산)
# ============================================================
# CLAUDE.md 7절 캐싱 원칙("고정된 건 한 번 만들어 재활용")과 동일 패턴 - evaluate.py의
# 워커 전역 _NET/_BASE_P 캐싱과 대응. 여기서는 조류계산이 아니라 LP 토폴로지용이라
# 워커별이 아니라 프로세스(모듈)당 1회면 충분하다.

_TOPOLOGY = None


def _build_topology():
    """방사형 트리(슬랙=bus0 루트, in_service 32선로만 - tie 5개는 build_net이 이미 제외).
    D[e,bus]=1이면 bus가 branch e의 하류(자손) - scripts/test_lindistflow.py와 동일 구성,
    단 branch 인덱스를 0..n_branch-1로 압축한다(원 pandapower line index는 tie 선로가
    섞여 있어 32개가 반드시 연속이라는 보장을 코드에서 가정하지 않기 위함)."""
    net = build_net()
    n_bus = PM.N_BUS
    lines = net.line[net.line['in_service']]

    children = {int(b): [] for b in net.bus.index}
    r_pu_by_idx = {}
    x_pu_by_idx = {}
    for idx, row in lines.iterrows():
        i, j = int(row['from_bus']), int(row['to_bus'])
        r_pu_by_idx[idx] = row['r_ohm_per_km'] * row['length_km'] / PM.Z_BASE_OHM
        x_pu_by_idx[idx] = row['x_ohm_per_km'] * row['length_km'] / PM.Z_BASE_OHM
        children[i].append((j, idx))

    order = []
    parent_of = {PM.SLACK_BUS: None}
    line_of = {}
    stack = [PM.SLACK_BUS]
    while stack:
        u = stack.pop()
        order.append(u)
        for v, lidx in children[u]:
            parent_of[v] = u
            line_of[v] = lidx
            stack.append(v)

    line_idxs = sorted(lines.index)
    n_branch = len(line_idxs)
    branch_pos = {lidx: k for k, lidx in enumerate(line_idxs)}
    r_pu = np.zeros(n_branch)
    x_pu = np.zeros(n_branch)
    for lidx, k in branch_pos.items():
        r_pu[k] = r_pu_by_idx[lidx]
        x_pu[k] = x_pu_by_idx[lidx]

    D = np.zeros((n_branch, n_bus))
    for bus in order[1:]:
        u = bus
        while parent_of[u] is not None:
            k = branch_pos[line_of[u]]
            D[k, bus] = 1.0
            u = parent_of[u]

    base_load_p_bus = np.zeros(n_bus)
    base_load_q_bus = np.zeros(n_bus)
    load_bus = net.load['bus'].to_numpy()
    np.add.at(base_load_p_bus, load_bus, net.load['p_mw'].to_numpy())
    np.add.at(base_load_q_bus, load_bus, net.load['q_mvar'].to_numpy())

    return dict(D=D, r_pu=r_pu, x_pu=x_pu, n_branch=n_branch, n_bus=n_bus,
                base_load_p_bus=base_load_p_bus, base_load_q_bus=base_load_q_bus)


def _get_topology():
    global _TOPOLOGY
    if _TOPOLOGY is None:
        _TOPOLOGY = _build_topology()
    return _TOPOLOGY


def base_load_bus_arrays():
    """(base_load_p_bus, base_load_q_bus) - 33길이, MW/Mvar. profile(시나리오 배율)을
    곱하는 건 호출부(solve_avg/solve_peak) 몫."""
    topo = _get_topology()
    return topo['base_load_p_bus'], topo['base_load_q_bus']


# ============================================================
# 1. 조인트 LP 문제 구성 (n기 공통 골격) + Problem 캐시 (DPP, 부록C.4-(4))
# ============================================================

_PROBLEM_CACHE = {}


def _cache_key(kind, n, force_q_zero, eta_c, eta_d, self_discharge, soc_init, soc_min, soc_max,
                mu_volt):
    return (kind, int(n), bool(force_q_zero), float(eta_c), float(eta_d),
            float(self_discharge), float(soc_init), float(soc_min), float(soc_max), float(mu_volt))


def _build_problem(kind, n, force_q_zero, eta_c, eta_d, self_discharge,
                    soc_init, soc_min, soc_max, mu_volt):
    # ★ mu_volt는 Parameter가 아니라 빌드시 상수로 굽는다(캐시 키에 포함). 이유(실측):
    # v_nonslack이 이미 bus_onehot/load_p_bus/load_q_bus(다른 Parameter)를 내부에 물고 있는
    # convex 표현식이라, 그 위에 또 다른 Parameter(mu_volt)를 곱하면 cvxpy DPP가 위반된다
    # (Parameter*Parameter 중첩으로 간주됨 - 각 조각은 개별적으로 is_dpp()=True인데도 곱하면
    # False가 되는 것을 직접 확인함). mu_volt를 float로 구우면 이 문제가 사라진다. 실전에서는
    # 항상 PM.MU_VOLT 하나만 쓰이므로 캐시가 늘지 않고, 값을 바꿔 부르는 것은 test_lp.py의
    # mu_volt=0.0 회귀검증뿐이다(그 경우 한 번 더 빌드되는 비용은 무시할 만함).
    T = PM.TIME_STEPS
    dt = PM.DT_HOURS
    topo = _get_topology()
    D, r_pu, x_pu = topo['D'], topo['r_pu'], topo['x_pu']
    n_bus, n_branch = topo['n_bus'], topo['n_branch']
    R_MAT = np.tile(r_pu[:, None], (1, T))
    X_MAT = np.tile(x_pu[:, None], (1, T))

    P_ch = cp.Variable((n, T), nonneg=True)
    P_dis = cp.Variable((n, T), nonneg=True)
    Q = cp.Variable((n, T))
    soc = cp.Variable((n, T + 1))

    S_param = cp.Parameter(n, nonneg=True)
    E_param = cp.Parameter(n, nonneg=True)
    bus_onehot = cp.Parameter((n, n_bus))          # 행별로 정확히 1개 1 (기i의 설치버스)
    load_p_bus = cp.Parameter((n_bus, T))           # 시나리오별 버스 유효부하 (MW, 소비=양수)
    load_q_bus = cp.Parameter((n_bus, T))
    mu_volt_const = float(mu_volt)

    S_col = cp.reshape(S_param, (n, 1), order='C')
    E_col = cp.reshape(E_param, (n, 1), order='C')

    constraints = []

    # ---- 개별 PCS 출력한계 (force_q_zero 무관하게 항상 성립 - 동시충방전 폭주 방지) ----
    constraints += [P_ch <= S_col, P_dis <= S_col]

    # ---- SOC (기별 재귀식, MWh 절대량 규약 - CLAUDE.md 2절/부록A #1) ----
    constraints += [
        soc[:, 0] == soc_init * E_param,
        soc[:, T] == soc_init * E_param,
        soc >= soc_min * E_col,
        soc <= soc_max * E_col,
    ]
    for t in range(T):
        constraints.append(
            soc[:, t + 1] == soc[:, t] * (1 - self_discharge)
            + eta_c * P_ch[:, t] * dt - P_dis[:, t] / eta_d * dt
        )

    P_net = P_dis - P_ch  # (n,T) affine, +방전/-충전

    # ---- PCS 원 제약: force_q_zero면 정확히 Q=0(다각형 불필요), 아니면 다각형 내접 근사 ----
    if force_q_zero:
        constraints.append(Q == 0)
    else:
        s_cap = S_col * float(np.cos(np.pi / PM.POLY_N))
        for k in range(PM.POLY_N):
            theta = 2.0 * np.pi * k / PM.POLY_N
            constraints.append(P_net * float(np.cos(theta)) + Q * float(np.sin(theta)) <= s_cap)

    # ---- LinDistFlow: 버스별 순부하 -> 선로조류 -> V^2 (부록C.4, Baran-Wu load-positive) ----
    # ★ MW/Mvar -> pu 변환(÷S_BASE_MVA) 필수 - r_pu/x_pu가 무차원 pu 임피던스이므로 P/Q도
    # pu여야 재귀식이 성립한다(scripts/test_lindistflow.py에서 실측된 것과 동일 함정 -
    # MW 그대로 넣으면 v가 음수로 튄다. 여기서는 P_net이 cvxpy 변수라 값 대신 식 자체를
    # S_BASE_MVA로 나눈다 - load/injection 둘 다 MW 단위라 한 번에 나누면 충분).
    netinj_p = (load_p_bus - bus_onehot.T @ P_net) / PM.S_BASE_MVA   # (n_bus,T), pu
    netinj_q = (load_q_bus - bus_onehot.T @ Q) / PM.S_BASE_MVA        # (n_bus,T), pu
    P_e = D @ netinj_p                               # (n_branch,T)
    Q_e = D @ netinj_q
    v = PM.V_SLACK_SQ - 2.0 * (D.T @ (cp.multiply(R_MAT, P_e) + cp.multiply(X_MAT, Q_e)))
    v_nonslack = v[1:, :]  # bus0(슬랙)은 항상 V_SLACK_SQ로 자명 - 페널티에서 제외(계산량 절약)

    volt_penalty = mu_volt_const * cp.sum(
        cp.pos(v_nonslack - PM.V_SQ_MAX) + cp.pos(PM.V_SQ_MIN - v_nonslack)
    )

    params = dict(S=S_param, E=E_param, bus_onehot=bus_onehot,
                  load_p_bus=load_p_bus, load_q_bus=load_q_bus)
    varset = dict(P_ch=P_ch, P_dis=P_dis, Q=Q, soc=soc, P_net=P_net)

    if kind == 'avg':
        smp_param = cp.Parameter(T)
        # EPS_REG 정규화항(1e-6): SMP 평탄 등으로 목적함수가 degenerate해질 때 동시충방전
        # 해를 배제한다(주 목적함수 대비 6자리 작아 정상 차익거래 최적해는 왜곡하지 않음 -
        # 편입 전 solve_avg와 동일 논거).
        objective_expr = (
            cp.sum(cp.multiply(cp.reshape(smp_param, (1, T), order='C'), P_ch - P_dis)) * dt
            + 1e-6 * cp.sum(P_ch + P_dis)
            + volt_penalty
        )
        problem = cp.Problem(cp.Minimize(objective_expr), constraints)
        params['smp'] = smp_param
    elif kind == 'peak':
        load_total_param = cp.Parameter(T)
        pk = cp.Variable()
        constraints.append(pk >= load_total_param - cp.sum(P_net, axis=0))
        objective_expr = pk + volt_penalty
        problem = cp.Problem(cp.Minimize(objective_expr), constraints)
        params['load_total'] = load_total_param
        varset['pk'] = pk
    else:
        raise ValueError(f'알 수 없는 kind: {kind}')

    assert problem.is_dcp(dpp=True), f'{kind} 문제가 DPP가 아님 (n={n}, force_q_zero={force_q_zero})'
    return dict(problem=problem, params=params, vars=varset)


def _get_problem(kind, n, force_q_zero, eta_c, eta_d, self_discharge, soc_init, soc_min, soc_max,
                  mu_volt):
    key = _cache_key(kind, n, force_q_zero, eta_c, eta_d, self_discharge, soc_init, soc_min, soc_max,
                      mu_volt)
    entry = _PROBLEM_CACHE.get(key)
    if entry is None:
        entry = _build_problem(kind, n, force_q_zero, eta_c, eta_d, self_discharge,
                                soc_init, soc_min, soc_max, mu_volt)
        _PROBLEM_CACHE[key] = entry
    return entry


def _check_solved(prob, name):
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f'{name}: LP 미해결 (status={prob.status})')


# ============================================================
# 2. 물리 보존 assert (CLAUDE.md 7절 LP검증#2 - 다수기로 일반화)
# ============================================================

def _assert_physics(p_ch, p_dis, q, soc, S, E, eta_c, eta_d, dt, self_discharge,
                     soc_init_frac, soc_min_frac, soc_max_frac, force_q_zero, tol=1e-4):
    n, T = p_ch.shape
    for i in range(n):
        soc_init_mwh = soc_init_frac * E[i]
        soc_min_mwh = soc_min_frac * E[i]
        soc_max_mwh = soc_max_frac * E[i]
        assert np.isclose(soc[i, 0], soc_init_mwh, atol=tol)
        assert np.isclose(soc[i, T], soc_init_mwh, atol=tol)
        assert np.all(soc[i, :] >= soc_min_mwh - tol) and np.all(soc[i, :] <= soc_max_mwh + tol)
        assert np.all(p_ch[i, :] <= S[i] + tol) and np.all(p_dis[i, :] <= S[i] + tol)
        assert np.all(p_ch[i, :] >= -tol) and np.all(p_dis[i, :] >= -tol)

        soc_recheck = np.empty(T + 1)
        soc_recheck[0] = soc[i, 0]
        for t in range(T):
            soc_recheck[t + 1] = soc_recheck[t] * (1 - self_discharge) + (
                eta_c * p_ch[i, t] * dt - p_dis[i, t] / eta_d * dt
            )
        assert np.allclose(soc_recheck, soc[i, :], atol=tol), f'unit {i}: 에너지수지(SOC 재귀식) 불일치'

        if force_q_zero:
            assert np.allclose(q[i, :], 0.0, atol=tol), f'unit {i}: force_q_zero인데 Q!=0'
        else:
            # 원 제약(다각형이 근사하는 참 제약) 상한 - 다각형은 원 안쪽이라 이 부등식은
            # 항상 여유를 두고 성립해야 함(어기면 정식화 오류).
            apparent = np.sqrt((p_dis[i, :] - p_ch[i, :]) ** 2 + q[i, :] ** 2)
            assert np.all(apparent <= S[i] + tol), f'unit {i}: sqrt(P^2+Q^2) > S (다각형 근사 오류 의심)'


# ============================================================
# 3. 공개 API: solve_avg / solve_peak (다수기 조인트)
# ============================================================

_S_FLOOR_ABS_MVA = 1e-9   # self_discharge=0이면 이 절대바닥만 적용(수치적 여유, 활동 없음)
_S_FLOOR_MARGIN = 1.1      # 자기방전 상쇄 소요량 대비 안전마진(10%) - 너무 크면 그 바닥
                           # 자체가 실경제성(SMP 차익)을 갖게 되어 "S=0=비활성"의 의미가
                           # 흐려진다(실측: margin=2.0에서 E=5 케이스 b_energy가 5만원대까지
                           # 샘 - 최적해 j_net(~3e6원) 대비는 작지만 원칙4 atol=10원 관례에는
                           # 한참 못 미친다). 1.1로 줄여 feasibility에 필요한 최소치에 근접시킨다.


def _s_floor_for_self_discharge(E, self_discharge, soc_init_frac, T):
    """S=0(정확히)이면서 E>0인 유닛은 SOC[0]=SOC[T](사이클 등식) 제약이 자기방전
    (self_discharge>0)과 함께 진짜로 infeasible해진다 - 방전할 능력이 0인데 자기방전으로
    빠진 만큼을 채워 넣을 수단(충전)도 0이라 등식을 못 맞춘다(수학적으로 실재하는 결함이지
    버그가 아니다 - 편입 전에는 S<=0/E<=0을 LP 호출 자체를 건너뛰는 방식으로 우회했으나,
    지금은 조인트 LP라 한 기라도 infeasible하면 시스템 전체가 infeasible해진다).

    self_discharge=0이면(무손실 테스트 등) 감쇠 자체가 없어 floor도 필요 없다(그대로 0).
    self_discharge>0이면 T시간 누적 감쇠분(soc_init_frac*E*(1-(1-self_discharge)^T))을 단
    1시간 안에 되채울 수 있는 만큼만(margin 2배) 바닥을 깐다 - E에 비례하므로 큰 E를 쓰는
    테스트(예: E=100)에서도 자동으로 충분히 커지고, 작은 E에서는 무시할 만큼 작게 유지된다."""
    if self_discharge <= 0.0:
        return _S_FLOOR_ABS_MVA
    decay_frac = 1.0 - (1.0 - self_discharge) ** T
    needed_mwh = soc_init_frac * E * decay_frac
    return max(_S_FLOOR_ABS_MVA, needed_mwh * _S_FLOOR_MARGIN)


def _prepare_common(S_mva, E_mwh, bus_idx, profile, self_discharge, soc_init):
    S = np.atleast_1d(np.asarray(S_mva, dtype=float))
    E = np.atleast_1d(np.asarray(E_mwh, dtype=float))
    bus_idx = np.atleast_1d(np.asarray(bus_idx, dtype=int))
    n = S.shape[0]
    assert E.shape == (n,) and bus_idx.shape == (n,), \
        f'S_mva/E_mwh/bus_idx 길이가 일치해야 함: {S.shape},{E.shape},{bus_idx.shape}'

    s_floor = np.array([
        _s_floor_for_self_discharge(E[i], self_discharge, soc_init, PM.TIME_STEPS) for i in range(n)
    ])
    S = np.maximum(S, s_floor)

    profile = np.asarray(profile, dtype=float)
    assert profile.shape == (PM.TIME_STEPS,), f'profile shape {profile.shape} != ({PM.TIME_STEPS},)'

    base_load_p_bus, base_load_q_bus = base_load_bus_arrays()
    load_p_bus_val = base_load_p_bus[:, None] * profile[None, :]   # (n_bus,T)
    load_q_bus_val = base_load_q_bus[:, None] * profile[None, :]

    onehot = np.zeros((n, PM.N_BUS))
    onehot[np.arange(n), bus_idx] = 1.0

    return n, S, E, onehot, load_p_bus_val, load_q_bus_val


def solve_avg(
    S_mva, E_mwh, bus_idx, smp, profile,
    *, eta_c=PM.ETA_C, eta_d=PM.ETA_D,
    self_discharge=PM.SELF_DISCHARGE_HOURLY,
    soc_init=PM.SOC_INIT_FRAC, soc_min=PM.SOC_MIN_FRAC, soc_max=PM.SOC_MAX_FRAC,
    mu_volt=PM.MU_VOLT, force_q_zero=False, assert_physics=True,
):
    """AVG_DAYS: 조달비(슬랙 유입 대리) 최소화 + mu_volt*전압유도항. 다수기(n) 조인트.

    S_mva/E_mwh/bus_idx: 길이 n 배열형(list/np.ndarray, 스칼라도 허용 - n=1로 승격).
    smp: 길이 T (원/kWh, 절대값). profile: 길이 T, 해당 시나리오의 LOAD 배율
      (LinDistFlow 버스별 부하 스케일 전용 - 조달비 목적함수와 무관. mu_volt=0이면
      이 인자의 실제 값이 최적해에 영향을 주지 않는다 - 유도항이 사라지므로).
    반환: (P_net (n,T), Q (n,T), soc (n,T+1))
      P_net: +방전/-충전 (MW). Q: 자유부호 (Mvar, force_q_zero면 전부 0).
      soc: MWh 절대량 (부록A #1 규약).
    """
    T = PM.TIME_STEPS
    smp = np.asarray(smp, dtype=float)
    assert smp.shape == (T,), f'smp shape {smp.shape} != ({T},)'

    n, S, E, onehot, load_p_val, load_q_val = _prepare_common(
        S_mva, E_mwh, bus_idx, profile, self_discharge, soc_init
    )

    entry = _get_problem('avg', n, force_q_zero, eta_c, eta_d, self_discharge,
                          soc_init, soc_min, soc_max, mu_volt)
    p = entry['params']
    p['S'].value = S
    p['E'].value = E
    p['bus_onehot'].value = onehot
    p['load_p_bus'].value = load_p_val
    p['load_q_bus'].value = load_q_val
    p['smp'].value = smp

    entry['problem'].solve()
    _check_solved(entry['problem'], 'solve_avg')

    v = entry['vars']
    p_ch_val, p_dis_val, q_val, soc_val = v['P_ch'].value, v['P_dis'].value, v['Q'].value, v['soc'].value

    if assert_physics:
        _assert_physics(p_ch_val, p_dis_val, q_val, soc_val, S, E, eta_c, eta_d, PM.DT_HOURS,
                         self_discharge, soc_init, soc_min, soc_max, force_q_zero)

    return p_dis_val - p_ch_val, q_val, soc_val


def solve_peak(
    S_mva, E_mwh, bus_idx, load_total, profile,
    *, eta_c=PM.ETA_C, eta_d=PM.ETA_D,
    self_discharge=PM.SELF_DISCHARGE_HOURLY,
    soc_init=PM.SOC_INIT_FRAC, soc_min=PM.SOC_MIN_FRAC, soc_max=PM.SOC_MAX_FRAC,
    mu_volt=PM.MU_VOLT, force_q_zero=False, assert_physics=True,
):
    """PEAK_DAYS: **시스템 전체** 피크 pk = max_t(load_total[t] - Sigma_i P_net[i,t]) 최소화
    + mu_volt*전압유도항. ★ 통합 solve_peak(C.6-3 3절) - pk가 기별로 따로가 아니라 전 기
    합산 기준 단일 스칼라라 2절/7-A절의 "기수 축 d* 중복적용" 결함이 구조적으로 사라진다
    (기존 분리LP는 각 기가 자기 d*까지 독립적으로 방전해 시스템 기준으로는 되갚음이
    중복 적용됐다 - probe_split.py 실측: b_defer가 0.53배로 붕괴).

    ★ 주의: 반환하는 pk는 손실을 무시한 "부하단" 대리값이며, B_defer 산정에 직접 쓰지
    말 것(2절 주의, 3절 B_defer 시그니처가 애초에 pk를 받지 않음 - benefits.b_defer 참조).
    B_defer는 반드시 이 함수가 반환한 스케줄(P_net)을 사후 조류계산에 넣어 얻은 슬랙
    피크로 계산해야 한다.

    S_mva/E_mwh/bus_idx: 길이 n 배열형. load_total: 길이 T, 시스템 전체 피더 부하(MW) -
      pk 제약의 기준값(기존 solve_peak의 load_mw와 동일 의미, 이제 "시스템 전체" 기준임을
      이름으로 명시). profile: 길이 T, LinDistFlow 버스별 부하 스케일 전용(load_total과
      독립 - 실전에서는 둘 다 base_p.sum()*LOAD[s]/LOAD[s]로 서로 정합하지만, 함수
      시그니처상으로는 별개 인자다).
    반환: (P_net (n,T), Q (n,T), soc (n,T+1), pk (스칼라, MW))
    """
    T = PM.TIME_STEPS
    load_total = np.asarray(load_total, dtype=float)
    assert load_total.shape == (T,), f'load_total shape {load_total.shape} != ({T},)'

    n, S, E, onehot, load_p_val, load_q_val = _prepare_common(
        S_mva, E_mwh, bus_idx, profile, self_discharge, soc_init
    )

    entry = _get_problem('peak', n, force_q_zero, eta_c, eta_d, self_discharge,
                          soc_init, soc_min, soc_max, mu_volt)
    p = entry['params']
    p['S'].value = S
    p['E'].value = E
    p['bus_onehot'].value = onehot
    p['load_p_bus'].value = load_p_val
    p['load_q_bus'].value = load_q_val
    p['load_total'].value = load_total

    entry['problem'].solve()
    _check_solved(entry['problem'], 'solve_peak')

    v = entry['vars']
    p_ch_val, p_dis_val, q_val, soc_val = v['P_ch'].value, v['P_dis'].value, v['Q'].value, v['soc'].value
    pk_val = float(v['pk'].value)

    if assert_physics:
        _assert_physics(p_ch_val, p_dis_val, q_val, soc_val, S, E, eta_c, eta_d, PM.DT_HOURS,
                         self_discharge, soc_init, soc_min, soc_max, force_q_zero)

    return p_dis_val - p_ch_val, q_val, soc_val, pk_val
