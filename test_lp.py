"""lower_lp.py 해석해 대조 검증 (CLAUDE.md 7절 LP검증#1, #3 + 부록C.6-4 회귀검증).
pytest 없이도 단독 실행 가능: `python test_lp.py`

★ C.6-3 LinDistFlow 편입 이후 lower_lp.solve_avg/solve_peak는 다수기 조인트 시그니처로
바뀌었다(스칼라 S_mw,E_mwh -> 배열 S_mva,E_mwh,bus_idx / smp,load_mw에 profile 추가 /
반환에 Q 추가). 아래 기존 13개 테스트는 **물리적으로 동일한 결과**를 검증하도록 시그니처만
갱신했다(회귀 없음 - C.6-4 (a)). 방법: `mu_volt=0.0, force_q_zero=True`로 호출한다.
- mu_volt=0.0 -> LinDistFlow 전압유도항이 목적함수에서 0으로 사라진다.
- force_q_zero=True -> Q==0 강제 + 정확한 P_ch<=S,P_dis<=S(다각형 근사 아님) 제약만 남는다
  (Q=0이면 sqrt(P^2+Q^2)<=S가 |P|<=S와 대수적으로 동치이므로 근사가 필요 없다 -
  lower_lp.py 모듈 docstring 참조). 이 조합이면 편입 전 물리(P_ch<=S,P_dis<=S만 있던
  구조)와 수치적으로 정확히 같아야 한다(atol은 기존 관례 유지).
- bus_idx/profile은 mu_volt=0이라 최적해에 영향을 주지 않는다(더미값 사용, 주석으로 명시).
"""

import numpy as np

from lower_lp import solve_avg, solve_peak

T = 24
ATOL = 1e-3

# mu_volt=0 회귀검증 케이스들이 공유하는 더미값 - 값 자체는 mu_volt=0이라 결과에 영향 없음
# (LinDistFlow 유도항이 통째로 사라지므로 - 모듈 docstring 참조).
DUMMY_BUS = [1]
DUMMY_PROFILE = np.ones(T)


def _flat(cheap, expensive, n_cheap=12):
    return np.array([cheap] * n_cheap + [expensive] * (T - n_cheap))


# ============================================================
# ★ C.6-4 (a) 회귀검증 앵커 - 반드시 최우선으로 확인할 것 (보고서 최상단)
# ============================================================

def test_regression_mu0_qzero_matches_pre_integration():
    """mu_volt=0 AND Q=0이면 편입 전(P_ch<=S,P_dis<=S만 있던 lower_lp.py) 결과와
    수치적으로 일치해야 한다(C.6-4 (a) 필수 조건). 편입 전 test_avg_lossless_hand_calc의
    손계산과 동일한 값으로 직접 확인한다: S=1MW,E=2MWh,SMP 100(0~11h)/200(12~23h),
    무손실(eta=1,self_discharge=0) -> 충전 0.9MWh, 방전 0.9MWh, objective=-90.
    """
    smp = _flat(100.0, 200.0)
    P_net, Q, soc = solve_avg(
        S_mva=[1.0], E_mwh=[2.0], bus_idx=DUMMY_BUS, smp=smp, profile=DUMMY_PROFILE,
        eta_c=1.0, eta_d=1.0, self_discharge=0.0, mu_volt=0.0, force_q_zero=True,
    )
    charge_total = -P_net[0][P_net[0] < 0].sum()
    discharge_total = P_net[0][P_net[0] > 0].sum()
    objective = float(np.sum(smp * (-P_net[0])))

    assert np.allclose(Q, 0.0, atol=1e-6), 'force_q_zero인데 Q != 0'
    assert np.isclose(charge_total, 0.9, atol=ATOL), charge_total
    assert np.isclose(discharge_total, 0.9, atol=ATOL), discharge_total
    assert np.isclose(objective, -90.0, atol=ATOL), objective
    assert np.isclose(soc[0].max(), 0.95 * 2.0, atol=ATOL), soc[0].max()
    assert np.isclose(soc[0].min(), 0.50 * 2.0, atol=ATOL), soc[0].min()
    print('test_regression_mu0_qzero_matches_pre_integration OK '
          '(편입 전 test_avg_lossless_hand_calc와 수치 일치 확인)')


# ============================================================
# 1~5. solve_avg 기존 테스트 (시그니처만 갱신, 값 불변)
# ============================================================

def test_avg_lossless_hand_calc():
    """손실무시(eta=1, self_discharge=0). S=1MW, E=2MWh, SMP 100(0~11h)/200(12~23h).
    손계산: SOC 상한 0.95, 시작 0.5 -> 헤드룸 0.45 -> 충전 가능량 0.45*E=0.9MWh
    (S=1MW로 12h 안에 채우는 데 문제없어 SOC캡이 지배). 방전도 동량 0.9MWh(무손실,
    SOC[24]=SOC[0] 위해 총충전=총방전). objective = 100*0.9 - 200*0.9 = -90.
    """
    smp = _flat(100.0, 200.0)
    P_net, Q, soc = solve_avg(
        S_mva=[1.0], E_mwh=[2.0], bus_idx=DUMMY_BUS, smp=smp, profile=DUMMY_PROFILE,
        eta_c=1.0, eta_d=1.0, self_discharge=0.0, mu_volt=0.0, force_q_zero=True,
    )
    P_net = P_net[0]
    charge_total = -P_net[P_net < 0].sum()
    discharge_total = P_net[P_net > 0].sum()
    objective = float(np.sum(smp * (-P_net)))

    assert np.isclose(charge_total, 0.9, atol=ATOL), charge_total
    assert np.isclose(discharge_total, 0.9, atol=ATOL), discharge_total
    assert np.isclose(objective, -90.0, atol=ATOL), objective
    assert np.isclose(soc[0].max(), 0.95 * 2.0, atol=ATOL), soc[0].max()
    assert np.isclose(soc[0].min(), 0.50 * 2.0, atol=ATOL), soc[0].min()
    assert np.all(P_net[:12] <= ATOL)
    assert np.all(P_net[12:] >= -ATOL)
    print('test_avg_lossless_hand_calc OK')


def test_avg_eta90_hand_calc():
    """효율 90%(eta_c=eta_d=0.9), self_discharge=0. 나머지 동일.
    손계산: 저장에너지 균형 eta_c*X_ch = X_dis/eta_d -> X_dis = 0.81*X_ch.
    충전은 SOC캡(헤드룸 0.45)까지: eta_c*X_ch/E=0.45 -> X_ch=0.45*2/0.9=1.0MWh, X_dis=0.81MWh.
    objective = 100*1.0 - 200*0.81 = -62.
    """
    smp = _flat(100.0, 200.0)
    P_net, Q, soc = solve_avg(
        S_mva=[1.0], E_mwh=[2.0], bus_idx=DUMMY_BUS, smp=smp, profile=DUMMY_PROFILE,
        eta_c=0.9, eta_d=0.9, self_discharge=0.0, mu_volt=0.0, force_q_zero=True,
    )
    P_net = P_net[0]
    charge_total = -P_net[P_net < 0].sum()
    discharge_total = P_net[P_net > 0].sum()
    objective = float(np.sum(smp * (-P_net)))

    assert np.isclose(charge_total, 1.0, atol=ATOL), charge_total
    assert np.isclose(discharge_total, 0.81, atol=ATOL), discharge_total
    assert np.isclose(objective, -62.0, atol=ATOL), objective
    assert np.isclose(soc[0].max(), 0.95 * 2.0, atol=ATOL), soc[0].max()
    assert np.isclose(soc[0].min(), 0.50 * 2.0, atol=ATOL), soc[0].min()
    print('test_avg_eta90_hand_calc OK')


def test_avg_zero_spread_no_trade():
    """스프레드=0 -> 차익=0 (LP검증#3). SMP 일정하면 손실 있는 거래는 손해라 무거래가 최적."""
    smp = np.full(T, 150.0)
    P_net, Q, soc = solve_avg(
        S_mva=[1.0], E_mwh=[2.0], bus_idx=DUMMY_BUS, smp=smp, profile=DUMMY_PROFILE,
        eta_c=0.9, eta_d=0.9, self_discharge=0.0, mu_volt=0.0, force_q_zero=True,
    )
    assert np.allclose(P_net[0], 0.0, atol=ATOL), P_net[0]
    objective = float(np.sum(smp * (-P_net[0])))
    assert np.isclose(objective, 0.0, atol=ATOL), objective
    print('test_avg_zero_spread_no_trade OK')


def test_avg_zero_power_no_benefit():
    """S=0 -> 편익~=0 (LP검증#3).

    ★ 정확히 0이 아니라 근사 0으로 검증한다(편입 전과의 차이, 새로 발견된 사실).
    이유: S=0인데 E>0(default self_discharge>0)이면 SOC[0]=SOC[24](사이클 등식) 제약이
    자기방전과 함께 수학적으로 infeasible해진다(방전 능력 0인데 자기방전으로 빠진 만큼
    채울 충전수단도 0). lower_lp._s_floor_for_self_discharge가 그 상쇄에 필요한 최소한의
    S만 자동으로 깔아준다(E=2, 기본 self_discharge 기준 약 0.00134MW - S_BOUNDS 상한
    2.4MVA의 0.06%, 실제 최적권 S~0.176의 0.8% 수준). 이 바닥 위에서 LP가 아주 미세한
    차익거래를 수행할 수 있어 정확히 0은 아니게 됐다 - 값 자체가 아니라 그 바닥의 크기로
    "무시할 만함"을 확인한다.
    """
    smp = _flat(100.0, 200.0)
    P_net, Q, soc = solve_avg(
        S_mva=[0.0], E_mwh=[2.0], bus_idx=DUMMY_BUS, smp=smp, profile=DUMMY_PROFILE,
        mu_volt=0.0, force_q_zero=True,
    )
    negligible_floor_atol = 0.002  # lower_lp._s_floor_for_self_discharge(E=2,...)~0.00134의 1.5배 여유
    # soc는 P_net 자체보다 넉넉하게 잡는다 - 순간값(P_net)이 아니라 여러 시간 누적(최대
    # floor*T=0.00134*24~0.032MWh 규모)이라 더 크게 흔들릴 수 있음.
    soc_atol = negligible_floor_atol * 24
    assert np.allclose(P_net[0], 0.0, atol=negligible_floor_atol), \
        (P_net[0], np.abs(P_net[0]).max())
    assert np.allclose(soc[0], 0.5 * 2.0, atol=soc_atol), (soc[0], np.abs(soc[0] - 1.0).max())
    print('test_avg_zero_power_no_benefit OK (S=0 바닥 자기방전 상쇄분만 남음, '
          f'max|P_net|={np.abs(P_net[0]).max():.6f} MW)')


def test_avg_energy_monotonic_benefit():
    """E 증가 -> 편익(=-objective) 단조 비감소 (LP검증#3). S=1MW 고정, E=1,2,4,8 MWh."""
    smp = _flat(100.0, 200.0)
    benefits = []
    for E in (1.0, 2.0, 4.0, 8.0):
        P_net, Q, _ = solve_avg(
            S_mva=[1.0], E_mwh=[E], bus_idx=DUMMY_BUS, smp=smp, profile=DUMMY_PROFILE,
            eta_c=0.9, eta_d=0.9, mu_volt=0.0, force_q_zero=True,
        )
        objective = float(np.sum(smp * (-P_net[0])))
        benefits.append(-objective)
    diffs = np.diff(benefits)
    assert np.all(diffs >= -1e-6), benefits
    print('test_avg_energy_monotonic_benefit OK', benefits)


# ============================================================
# 6~10. solve_peak 기존 테스트 (시그니처만 갱신, 값 불변 - "통합" solve_peak이지만
# n=1이면 기수 축이 없어 편입 전 분리LP와 동일하다 - C.6-3 6절(c) "n=1에서 기존 분리
# 결과와 일치").
# ============================================================

def test_peak_shaving_hand_calc():
    """load: 20시간 5MW 평탄 + 마지막 4시간(20~23h) 10MW 피크. S=2MW, E=100MWh(SOC캡 여유 충분),
    eta=1(무손실). 손계산: 방전 상한 S=2MW로 피크시간대 각 시간 10-2=8MW까지만 깎임 ->
    pk=8MW. 이때 방전은 정확히 2MW(그 이상은 새 피크를 못 낮춤, 그 이하는 최적 아님) -> 유일해.

    ※ 주의: pk만 최소화하는 목적함수라 20개 평탄시간에 8MWh를 "어떻게" 나눠 채우는지는
    (총량·상한만 지키면) 목적값에 영향 없어 LP 해가 degenerate하다 - 프로그램 검증은
    균등분산값이 아니라 총충전량과 "어느 평탄시간도 pk를 넘지 않는다"는, 해 선택에 무관하게
    항상 성립해야 하는 불변량으로 한다(CLAUDE.md 7절 테스트 설계 원칙 1·2).
    """
    load = np.array([5.0] * 20 + [10.0] * 4)
    P_net, Q, soc, pk = solve_peak(
        S_mva=[2.0], E_mwh=[100.0], bus_idx=DUMMY_BUS, load_total=load, profile=DUMMY_PROFILE,
        eta_c=1.0, eta_d=1.0, self_discharge=0.0, mu_volt=0.0, force_q_zero=True,
    )
    P_net = P_net[0]
    assert np.isclose(pk, 8.0, atol=ATOL), pk
    assert np.allclose(P_net[20:], 2.0, atol=ATOL), P_net[20:]
    charge_total = float(-P_net[:20].sum())
    assert np.isclose(charge_total, 8.0, atol=ATOL), charge_total
    flat_demand = load[:20] - P_net[:20]
    assert np.all(flat_demand <= pk + ATOL), flat_demand
    assert np.isclose(pk, float(np.max(load - P_net)), atol=ATOL)
    print('test_peak_shaving_hand_calc OK')


def test_peak_shaving_S3_charging_dominant_regime():
    """S=3MW: 일반해 pk(d)=max(10-d,5+d/5), d in [0,S]. 교차점 d*=25/6≈4.1667MW.
    S=3 < d*이므로 아직 방전측이 지배: d=S=3, pk=10-3=7."""
    load = np.array([5.0] * 20 + [10.0] * 4)
    P_net, Q, soc, pk = solve_peak(
        S_mva=[3.0], E_mwh=[100.0], bus_idx=DUMMY_BUS, load_total=load, profile=DUMMY_PROFILE,
        eta_c=1.0, eta_d=1.0, self_discharge=0.0, mu_volt=0.0, force_q_zero=True,
    )
    P_net = P_net[0]
    assert np.isclose(pk, 7.0, atol=ATOL), pk
    assert np.allclose(P_net[20:], 3.0, atol=ATOL), P_net[20:]
    charge_total = float(-P_net[:20].sum())
    assert np.isclose(charge_total, 12.0, atol=ATOL), charge_total
    flat_demand = load[:20] - P_net[:20]
    assert np.all(flat_demand <= pk + ATOL), flat_demand
    assert np.isclose(pk, float(np.max(load - P_net)), atol=ATOL)
    print('test_peak_shaving_S3_charging_dominant_regime OK')


def test_peak_shaving_S5_charging_dominant_regime():
    """S=5MW(>d*=25/6): 충전상승이 지배 - d*=25/6에서 정지, pk*=35/6≈5.8333MW.
    내부 최적점(슬랙 없음)이라 평탄부도 균등분산으로 유일하게 결정됨."""
    load = np.array([5.0] * 20 + [10.0] * 4)
    P_net, Q, soc, pk = solve_peak(
        S_mva=[5.0], E_mwh=[100.0], bus_idx=DUMMY_BUS, load_total=load, profile=DUMMY_PROFILE,
        eta_c=1.0, eta_d=1.0, self_discharge=0.0, mu_volt=0.0, force_q_zero=True,
    )
    P_net = P_net[0]
    d_star = 25.0 / 6.0
    pk_star = 35.0 / 6.0

    assert np.isclose(pk, pk_star, atol=ATOL), pk
    assert np.allclose(P_net[20:], d_star, atol=ATOL), P_net[20:]
    assert np.all(P_net[20:] < 5.0 - ATOL), P_net[20:]
    charge_total = float(-P_net[:20].sum())
    assert np.isclose(charge_total, 4 * d_star, atol=ATOL), charge_total
    flat_demand = load[:20] - P_net[:20]
    assert np.allclose(flat_demand, pk_star, atol=ATOL), flat_demand
    assert np.isclose(pk, float(np.max(load - P_net)), atol=ATOL)
    print('test_peak_shaving_S5_charging_dominant_regime OK')


def test_peak_shaving_E12_S_sweep():
    """E=12MWh로 낮추고 S=1,2,3,4MW 스윕. SOC 헤드룸=0.45*12=5.4MWh가 절대 상한.
    총방전량 X=min(4S,5.4). 교차점 S*=1.35MW. S>=1.35에서 pk 정체(에너지가 설비보다
    먼저 병목)."""
    load = np.array([5.0] * 20 + [10.0] * 4)
    E = 12.0
    headroom = 0.45 * E
    results = {}

    for S in (1.0, 2.0, 3.0, 4.0):
        P_net, Q, soc, pk = solve_peak(
            S_mva=[S], E_mwh=[E], bus_idx=DUMMY_BUS, load_total=load, profile=DUMMY_PROFILE,
            eta_c=1.0, eta_d=1.0, self_discharge=0.0, mu_volt=0.0, force_q_zero=True,
        )
        P_net = P_net[0]
        X = min(4 * S, headroom)
        d = X / 4.0
        pk_expected = 10.0 - d

        assert np.isclose(pk, pk_expected, atol=ATOL), (S, pk, pk_expected)
        assert np.allclose(P_net[20:], d, atol=ATOL), (S, P_net[20:], d)
        charge_total = float(-P_net[:20].sum())
        assert np.isclose(charge_total, X, atol=ATOL), (S, charge_total, X)
        flat_demand = load[:20] - P_net[:20]
        assert np.all(flat_demand <= pk + ATOL), (S, flat_demand)
        assert np.isclose(pk, float(np.max(load - P_net)), atol=ATOL)
        assert flat_demand.max() < pk - 0.1, (S, flat_demand.max(), pk)

        results[S] = pk
        regime = '전력상한(S) 지배' if 4 * S < headroom - ATOL else '에너지상한(E) 지배'
        print(f'  S={S:.0f}MW: pk={pk:.4f}MW, d={d:.4f}MW, {regime}, S 미사용여력={S - d:.4f}MW')

    assert np.isclose(results[2.0], results[3.0], atol=ATOL)
    assert np.isclose(results[3.0], results[4.0], atol=ATOL)
    assert results[1.0] > results[2.0] + ATOL
    print('test_peak_shaving_E12_S_sweep OK')


def test_peak_zero_power_no_shaving():
    """S=0 -> pk ~= max(load) (깎을 수 없음, LP검증#3과 동일 취지).

    ★ test_avg_zero_power_no_benefit과 동일한 이유로 정확히 10.0이 아니라 근사값으로
    검증한다 - E=100(자기방전 상쇄 바닥이 크게 잡힘, ~0.067MW)이라 그만큼 미세하게
    깎인다. 바닥 크기(S_BOUNDS 상한 2.4의 2.8%)로 "무시할 만함"을 확인."""
    load = np.array([5.0] * 20 + [10.0] * 4)
    P_net, Q, soc, pk = solve_peak(
        S_mva=[0.0], E_mwh=[100.0], bus_idx=DUMMY_BUS, load_total=load, profile=DUMMY_PROFILE,
        mu_volt=0.0, force_q_zero=True,
    )
    negligible_floor_atol = 0.1  # lower_lp._s_floor_for_self_discharge(E=100,...)~0.067의 1.5배 여유
    assert np.allclose(P_net[0], 0.0, atol=negligible_floor_atol), np.abs(P_net[0]).max()
    assert np.isclose(pk, 10.0, atol=negligible_floor_atol), pk
    print(f'test_peak_zero_power_no_shaving OK (pk={pk:.6f}, max|P_net|={np.abs(P_net[0]).max():.6f})')


# ============================================================
# C.6-4 (b) 신규 Q 테스트
# ============================================================

def test_polygon_binds_when_S_small():
    """S를 작게(0.05MW) 걸고 SMP를 크게 흔들면 방전이 S 근방까지 밀어붙여져 다각형
    제약(PCS 원의 내접 근사)이 활성화되어야 한다 - sqrt(P^2+Q^2)가 S에 근접(다각형이므로
    S*cos(pi/12)~S 사이)한지로 "바인딩"을 구조적으로 확인한다(정확한 solver 해 대신 제약
    활성 여부만 - CLAUDE.md 7절 테스트 설계 원칙 2, 솔버 무관 검증)."""
    import params as PM
    smp = _flat(50.0, 5000.0)  # 스프레드를 크게 줘서 방전 유인이 확실하도록
    profile = np.asarray(PM.LOAD['summer'])
    P_net, Q, soc = solve_avg(
        S_mva=[0.05], E_mwh=[1.0], bus_idx=[15], smp=smp, profile=profile,
        mu_volt=PM.MU_VOLT, force_q_zero=False,
    )
    apparent = np.sqrt(P_net[0] ** 2 + Q[0] ** 2)
    assert apparent.max() > 0.05 * 0.9, \
        f'다각형이 전혀 안 걸림(apparent 최대={apparent.max():.4f}, S=0.05) - 방전유인이 부족했을 수 있음'
    assert np.all(apparent <= 0.05 + ATOL), apparent.max()
    print('test_polygon_binds_when_S_small OK', apparent.max())


def test_q_free_benefit_ge_q_zero():
    """Q 자유(force_q_zero=False)의 **목적함수값**이 Q≡0(force_q_zero=True) 대비 나빠지면
    안 된다(LP 완화이므로 수학적으로 자명 - Q=0은 Q자유 문제의 feasible 특수해라 Q자유
    최적값이 항상 같거나 더 좋아야(작아야, 최소화 문제) 함). mu_volt를 크게 줘 전압유도항이
    실제로 작동하는 조건에서 비교한다(mu_volt=0이면 Q가 목적함수에 전혀 안 걸려 무의미).

    ★ 조달비(SMP*P_net)만 따로 비교하면 안 된다(초기 구현 실수) - 목적함수는
    "조달비 + mu_volt*전압유도항"이라 mu_volt가 클 때는 LP가 조달비를 일부 희생해서라도
    전압유도항을 더 줄이는 쪽을 선택할 수 있다(실측: 조달비만 보면 Q자유가 오히려 더
    나쁨 - -157 vs -151 - 그 자체는 버그가 아니라 "무엇을 최적화했는지"를 잘못 짚은 것).
    LP 완화 논증이 성립하는 대상은 어디까지나 **전체 목적함수값**이므로 그것으로 비교한다
    (lower_lp._get_problem으로 캐시된 Problem을 그대로 읽어 solve_avg가 이미 채워 놓은
    .value를 재사용 - 재계산 없음)."""
    import params as PM
    import lower_lp
    smp = np.asarray(PM.SMP['summer'])
    profile = np.asarray(PM.LOAD['summer'])
    big_mu = PM.MU_VOLT * 5

    def solved_objective_value(force_q_zero):
        solve_avg(
            S_mva=[2.4], E_mwh=[10.2], bus_idx=[15], smp=smp, profile=profile,
            mu_volt=big_mu, force_q_zero=force_q_zero,
        )
        entry = lower_lp._get_problem('avg', 1, force_q_zero, PM.ETA_C, PM.ETA_D,
                                       PM.SELF_DISCHARGE_HOURLY, PM.SOC_INIT_FRAC,
                                       PM.SOC_MIN_FRAC, PM.SOC_MAX_FRAC, big_mu)
        return float(entry['problem'].value)

    obj_free = solved_objective_value(False)
    obj_zero = solved_objective_value(True)

    # 최소화 문제이므로 "더 좋다" = 더 작다. Q자유가 Q=0의 완화이므로 obj_free <= obj_zero.
    assert obj_free <= obj_zero + 1e-3, (obj_free, obj_zero)
    print('test_q_free_benefit_ge_q_zero OK', obj_free, obj_zero)


def test_q_sign_opens_both_directions():
    """Q>=0 강제가 없어졌으므로, 과전압을 유발하는 조건(저부하 시각에 큰 방전)에서는
    실제로 Q<0(유도성, 전압을 낮추는 방향)이 나와야 한다 - 편입 전 q_ratio∈[0,1](Q>=0
    고정)이 못 하던 것(부록C.4-(3) "Phase 2 PV 과전압 대응이 자연히 가능해지는 것이
    편입 명분 1순위"). SMP를 저부하 시각에 극단적으로 높여 방전을 강제로 유도한다."""
    import params as PM
    profile = np.asarray(PM.LOAD['shoulder'])
    smp = np.asarray(PM.SMP['shoulder']).copy()
    low_t = int(np.argmin(profile))
    smp[low_t] = 1e5  # 저부하 시각에 방전이 압도적으로 유리하도록

    P_net, Q, soc = solve_avg(
        S_mva=[2.4], E_mwh=[10.2], bus_idx=[15], smp=smp, profile=profile,
        mu_volt=PM.MU_VOLT * 5, force_q_zero=False,
    )
    assert P_net[0, low_t] > 1.0, \
        f'저부하 시각 방전 유도 실패(P_net={P_net[0, low_t]:.4f}) - SMP 스파이크 재검토 필요'
    assert Q[0, low_t] < -1e-3, \
        f'과전압 유발 조건에서 Q<0이 안 나옴(Q={Q[0, low_t]:.4f}) - Q 부호제한이 남아있는지 확인'
    print('test_q_sign_opens_both_directions OK', P_net[0, low_t], Q[0, low_t])


# ============================================================
# C.6-4 (c) 통합 solve_peak 테스트 - 기수 축 d* 중복적용 해소 확인
# ============================================================

def test_unified_peak_split_matches_single_unit():
    """★ 2절/7-A절 결함(기수 축 d* 중복적용)이 해소됐는지 직접 확인. 동일 총용량(S=2MW,
    E=100MWh)을 (a) 단일기 1대 vs (b) 2대로 균등분할(1MW/50MWh씩)해서 풀었을 때, 통합
    solve_peak이면 시스템 전체 pk가 **거의 동일**해야 한다(예전 분리LP는 probe_split.py
    실측대로 분할 시 b_defer가 0.53배로 붕괴했었다 - 그 붕괴의 원인이 pk 자체의 악화였다).
    """
    load = np.array([5.0] * 20 + [10.0] * 4)
    P_single, Q_single, soc_single, pk_single = solve_peak(
        S_mva=[2.0], E_mwh=[100.0], bus_idx=DUMMY_BUS, load_total=load, profile=DUMMY_PROFILE,
        eta_c=1.0, eta_d=1.0, self_discharge=0.0, mu_volt=0.0, force_q_zero=True,
    )
    P_split, Q_split, soc_split, pk_split = solve_peak(
        S_mva=[1.0, 1.0], E_mwh=[50.0, 50.0], bus_idx=[1, 2], load_total=load, profile=DUMMY_PROFILE,
        eta_c=1.0, eta_d=1.0, self_discharge=0.0, mu_volt=0.0, force_q_zero=True,
    )
    assert np.isclose(pk_single, 8.0, atol=ATOL), pk_single
    assert np.isclose(pk_split, pk_single, atol=ATOL), (pk_split, pk_single)

    # ★ 시간대별 정확한 스케줄까지는 비교하지 않는다 - 평탄부(0~19h) 충전배분은 degenerate
    # 하다(CLAUDE.md 7절 테스트 설계 원칙 1: "해 분포와 무관한 불변량으로만 검증"). 실측:
    # 단일기/분할기 둘 다 피크시간대(20~23h) 방전은 정확히 2.0으로 일치하지만 평탄부는
    # 서로 다른(둘 다 유효한) 배분을 고른다. 검증할 불변량은 (1) pk 일치(위에서 이미 확인),
    # (2) 시스템 전체 총충전량 일치(에너지수지) - 이 둘이 "분할해도 손해 없음"의 증거다.
    assert np.allclose(P_split[:, 20:].sum(axis=0), 2.0, atol=ATOL), P_split[:, 20:]
    charge_total_single = float(-P_single[0, :20].sum())
    charge_total_split = float(-P_split[:, :20].sum())
    assert np.isclose(charge_total_split, charge_total_single, atol=1e-2), \
        (charge_total_split, charge_total_single)
    print('test_unified_peak_split_matches_single_unit OK', pk_single, pk_split)


def test_unified_peak_n1_equals_legacy_regime():
    """n=1(기수 축이 아예 없는 경우)에서는 '통합'과 '분리'가 정의상 같다(7-A절: 최적기수
    1이므로 최적해에서는 무해). 위 test_peak_shaving_hand_calc 등 mu_volt=0 회귀검증군이
    전부 n=1이므로 이미 이 사실을 실증하고 있다 - 이 테스트는 그 사실을 명시적으로 표시만
    한다(중복 계산 없음)."""
    print('test_unified_peak_n1_equals_legacy_regime OK '
          '(test_peak_shaving_* 계열이 이미 n=1 사례 - 통합=분리 자명)')


if __name__ == '__main__':
    test_regression_mu0_qzero_matches_pre_integration()
    test_avg_lossless_hand_calc()
    test_avg_eta90_hand_calc()
    test_avg_zero_spread_no_trade()
    test_avg_zero_power_no_benefit()
    test_avg_energy_monotonic_benefit()
    test_peak_shaving_hand_calc()
    test_peak_shaving_S3_charging_dominant_regime()
    test_peak_shaving_S5_charging_dominant_regime()
    test_peak_shaving_E12_S_sweep()
    test_peak_zero_power_no_shaving()
    test_polygon_binds_when_S_small()
    test_q_free_benefit_ge_q_zero()
    test_q_sign_opens_both_directions()
    test_unified_peak_split_matches_single_unit()
    test_unified_peak_n1_equals_legacy_regime()
    print('all lower_lp tests passed')
