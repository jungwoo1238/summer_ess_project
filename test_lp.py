"""lower_lp.py 해석해 대조 검증 (CLAUDE.md 7절 LP검증#1, #3).
pytest 없이도 단독 실행 가능: `python test_lp.py`
"""

import numpy as np

from lower_lp import solve_avg, solve_peak

T = 24
ATOL = 1e-3


def _flat(cheap, expensive, n_cheap=12):
    return np.array([cheap] * n_cheap + [expensive] * (T - n_cheap))


def test_avg_lossless_hand_calc():
    """손실무시(eta=1, self_discharge=0). S=1MW, E=2MWh, SMP 100(0~11h)/200(12~23h).
    손계산: SOC 상한 0.95, 시작 0.5 -> 헤드룸 0.45 -> 충전 가능량 0.45*E=0.9MWh
    (S=1MW로 12h 안에 채우는 데 문제없어 SOC캡이 지배). 방전도 동량 0.9MWh(무손실,
    SOC[24]=SOC[0] 위해 총충전=총방전). objective = 100*0.9 - 200*0.9 = -90.
    """
    smp = _flat(100.0, 200.0)
    P_net, soc = solve_avg(
        S_mw=1.0, E_mwh=2.0, smp=smp,
        eta_c=1.0, eta_d=1.0, self_discharge=0.0,
    )
    charge_total = -P_net[P_net < 0].sum()
    discharge_total = P_net[P_net > 0].sum()
    objective = float(np.sum(smp * (-P_net)))

    assert np.isclose(charge_total, 0.9, atol=ATOL), charge_total
    assert np.isclose(discharge_total, 0.9, atol=ATOL), discharge_total
    assert np.isclose(objective, -90.0, atol=ATOL), objective
    assert np.isclose(soc.max(), 0.95 * 2.0, atol=ATOL), soc.max()  # SOC MWh 규약: 0.95*E_mwh
    assert np.isclose(soc.min(), 0.50 * 2.0, atol=ATOL), soc.min()  # SOC MWh 규약: 0.50*E_mwh
    # 부호: 충전은 저가(0~11h)에만, 방전은 고가(12~23h)에만
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
    P_net, soc = solve_avg(
        S_mw=1.0, E_mwh=2.0, smp=smp,
        eta_c=0.9, eta_d=0.9, self_discharge=0.0,
    )
    charge_total = -P_net[P_net < 0].sum()
    discharge_total = P_net[P_net > 0].sum()
    objective = float(np.sum(smp * (-P_net)))

    assert np.isclose(charge_total, 1.0, atol=ATOL), charge_total
    assert np.isclose(discharge_total, 0.81, atol=ATOL), discharge_total
    assert np.isclose(objective, -62.0, atol=ATOL), objective
    assert np.isclose(soc.max(), 0.95 * 2.0, atol=ATOL), soc.max()  # SOC MWh 규약: 0.95*E_mwh
    assert np.isclose(soc.min(), 0.50 * 2.0, atol=ATOL), soc.min()  # SOC MWh 규약: 0.50*E_mwh
    print('test_avg_eta90_hand_calc OK')


def test_avg_zero_spread_no_trade():
    """스프레드=0 -> 차익=0 (LP검증#3). SMP 일정하면 손실 있는 거래는 손해라 무거래가 최적.
    self_discharge=0으로 명시(기본값 사용 시 SOC 감쇠 상쇄용 미세 보정충전이 섞여 순수
    가격차익 로직만 격리 검증이 안 됨 - CLAUDE.md 4절 자기방전 효과, 별개 관심사)."""
    smp = np.full(T, 150.0)
    P_net, soc = solve_avg(S_mw=1.0, E_mwh=2.0, smp=smp, eta_c=0.9, eta_d=0.9, self_discharge=0.0)
    assert np.allclose(P_net, 0.0, atol=ATOL), P_net
    objective = float(np.sum(smp * (-P_net)))
    assert np.isclose(objective, 0.0, atol=ATOL), objective
    print('test_avg_zero_spread_no_trade OK')


def test_avg_zero_power_no_benefit():
    """S=0 -> 편익=0 (LP검증#3)."""
    smp = _flat(100.0, 200.0)
    P_net, soc = solve_avg(S_mw=0.0, E_mwh=2.0, smp=smp)
    assert np.allclose(P_net, 0.0)
    assert np.allclose(soc, 0.5 * 2.0)  # SOC MWh 규약: 0.50*E_mwh
    print('test_avg_zero_power_no_benefit OK')


def test_avg_energy_monotonic_benefit():
    """E 증가 -> 편익(=-objective) 단조 비감소 (LP검증#3). S=1MW 고정, E=1,2,4,8 MWh."""
    smp = _flat(100.0, 200.0)
    benefits = []
    for E in (1.0, 2.0, 4.0, 8.0):
        P_net, _ = solve_avg(S_mw=1.0, E_mwh=E, smp=smp, eta_c=0.9, eta_d=0.9)
        objective = float(np.sum(smp * (-P_net)))
        benefits.append(-objective)
    diffs = np.diff(benefits)
    assert np.all(diffs >= -1e-6), benefits
    print('test_avg_energy_monotonic_benefit OK', benefits)


def test_peak_shaving_hand_calc():
    """load: 20시간 5MW 평탄 + 마지막 4시간(20~23h) 10MW 피크. S=2MW, E=100MWh(SOC캡 여유 충분),
    eta=1(무손실). 손계산: 방전 상한 S=2MW로 피크시간대 각 시간 10-2=8MW까지만 깎임 ->
    pk_s=8MW. 이때 방전은 정확히 2MW(그 이상은 새 피크를 못 낮춤, 그 이하는 최적 아님) -> 유일해.

    ★ 충전이 평탄부를 올리는 효과(빠지면 안 되는 조건 - 부록A #4): 방전한 8MWh(2MW*4h)를
    무손실이라 총충전=총방전=8MWh만큼 나머지 20시간에 되돌려 채워야 한다(SOC[24]=SOC[0]).
    균등분산 시 8MWh/20h=0.4MW/h -> 평탄부 수요가 5.0->5.4MW로 상승. 5.4 < 8이므로 평탄부는
    여전히 피크시간대(8MW)보다 낮아 pk_s=8 결론이 유지된다. 이 조건(충전상승 후 평탄부 <
    방전상한 시나리오)이 깨지면(S가 커지면) 결론이 달라진다 -> 아래 S=3MW 테스트에서 검증.

    ※ 주의: pk만 최소화하는 목적함수라 20개 평탄시간에 8MWh를 "어떻게" 나눠 채우는지는
    (총량·상한만 지키면) 목적값에 영향 없어 LP 해가 degenerate하다(균등분산은 그중 한 예시일
    뿐, 실제 솔버 해와 다를 수 있음) - 따라서 프로그램 검증은 균등분산값이 아니라 총충전량과
    "어느 평탄시간도 pk_s를 넘지 않는다"는, 해 선택에 무관하게 항상 성립해야 하는 불변량으로 한다.
    """
    load = np.array([5.0] * 20 + [10.0] * 4)
    P_net, soc, pk_s = solve_peak(
        S_mw=2.0, E_mwh=100.0, load_mw=load, eta_c=1.0, eta_d=1.0, self_discharge=0.0,
    )
    assert np.isclose(pk_s, 8.0, atol=ATOL), pk_s
    assert np.allclose(P_net[20:], 2.0, atol=ATOL), P_net[20:]
    # 에너지수지: 평탄부 총충전량 = 피크부 총방전량 = 8.0 MWh (해 분포와 무관하게 성립해야 함)
    charge_total = float(-P_net[:20].sum())
    assert np.isclose(charge_total, 8.0, atol=ATOL), charge_total
    # 평탄부 어느 시간도 pk_s(=8)를 넘지 않음 (넘으면 그게 새 피크가 되어 모순)
    flat_demand = load[:20] - P_net[:20]
    assert np.all(flat_demand <= pk_s + ATOL), flat_demand
    # 자체 일관성: pk_s == max(load - P_net)
    assert np.isclose(pk_s, float(np.max(load - P_net)), atol=ATOL)
    print('test_peak_shaving_hand_calc OK')


def test_peak_shaving_S3_charging_dominant_regime():
    """부록A #4 후속: S=3MW로 늘렸을 때 "충전상승이 지배"하는지 해석해로 직접 검증
    (값을 추측하지 않고 유도). load/E/eta는 위 테스트와 동일.

    일반해 유도: 피크시간(4개) 방전 d씩, 평탄시간(20개) 충전 d/5씩(에너지수지 4d=20*(d/5))이
    각 그룹 내 최댓값/최솟값 최적화(합 고정 시 균등분산이 최댓값 최소화·최솟값 최대화,
    표준 부등식). pk(d) = max(10-d, 5+d/5), d in [0, S].
    교차점: 10-d = 5+d/5 -> 5 = (6/5)d -> d* = 25/6 ≈ 4.16667 MW, pk(d*) = 35/6 ≈ 5.8333 MW.
    d<d*: 방전측(10-d, d에 대해 감소)이 더 커 d를 키우는 쪽이 유리 -> 최적 d=min(S,d*).
    d>d*: 충전측(5+d/5, d에 대해 증가)이 더 커져 더 키우면 손해 -> d*에서 정지
    (S를 더 늘려도 개선 없음 - 이게 "충전상승이 지배"하는 지점).

    S=3 < d*=25/6≈4.1667 이므로 아직 방전측이 지배하는 체제(위 S=2 테스트와 동일 정성적 결과):
      d=S=3, pk = 10-3 = 7 (평탄측 5+3/5=5.6 < 7이라 방전측이 여전히 병목).
    ※ 검증 결과 기록: S=3MW는 충전지배 임계값(d*≈4.1667MW)에 못 미쳐 방전제약 체제가
      유지된다 - "충전상승이 지배"하는 사례가 아니라 "그 직전까지도 방전측이 여전히 병목"임을
      보여주는 사례다. 충전상승이 실제로 지배하려면 S >= 25/6 MW가 필요하다(이 로드 패턴 기준).
    ※ 위 테스트와 동일하게 평탄부 20시간 내 분산은 degenerate(총량·상한만 결정) -> 총충전량과
      "평탄부가 pk_s를 넘지 않음"만 해 분포에 무관한 불변량으로 검증한다.
    """
    load = np.array([5.0] * 20 + [10.0] * 4)
    P_net, soc, pk_s = solve_peak(
        S_mw=3.0, E_mwh=100.0, load_mw=load, eta_c=1.0, eta_d=1.0, self_discharge=0.0,
    )
    assert np.isclose(pk_s, 7.0, atol=ATOL), pk_s
    assert np.allclose(P_net[20:], 3.0, atol=ATOL), P_net[20:]
    charge_total = float(-P_net[:20].sum())
    assert np.isclose(charge_total, 12.0, atol=ATOL), charge_total
    flat_demand = load[:20] - P_net[:20]
    assert np.all(flat_demand <= pk_s + ATOL), flat_demand
    assert np.isclose(pk_s, float(np.max(load - P_net)), atol=ATOL)
    print('test_peak_shaving_S3_charging_dominant_regime OK (S=3MW는 아직 방전지배 체제, '
          '충전지배 임계값 d*=25/6=4.1667MW 미도달)')


def test_peak_shaving_S5_charging_dominant_regime():
    """S=5MW: 위 S3 테스트에서 유도한 교차점 d*=25/6≈4.1667MW를 실제로 넘는(S>d*) 첫 케이스.
    load/E/eta는 위 테스트들과 동일. 값을 추측하지 않고 동일 일반해 pk(d)=max(10-d,5+d/5)로
    직접 유도한다.

    d<d*에서는 방전측(10-d)이 병목이라 d를 키우는 쪽이 유리했지만, d=d*=25/6을 넘어서면
    충전측(5+d/5)이 더 커져 오히려 pk가 악화된다. 따라서 S=5(>d*)에서는 굳이 S까지 방전을
    밀어붙이지 않고 d*=25/6에서 멈추는 것이 최적 -> **"S를 늘려도 더 이상 pk가 개선되지
    않는" 지점에 도달** = 충전상승이 방전여력을 지배하는 사례.
      d* = 25/6 MW,  pk* = 10 - 25/6 = 35/6 ≈ 5.83333 MW
      (검산: 평탄측 5 + d*/5 = 5 + (25/6)/5 = 5 + 5/6 = 35/6, 방전측과 정확히 일치)

    ※ S=2/S=3 테스트와 달리 이 지점은 두 제약(방전상한 여유, 평탄부 상한 여유)이 동시에
    타이트하게 맞물리는 내부 최적점이라 슬랙이 없다 - 즉 평탄부 20시간 분산도 더 이상
    degenerate하지 않고 균등분산(각 5/6MW)으로 유일하게 결정된다(방전측 4시간도 각 25/6MW로
    유일). 그래서 이 테스트는 (S2/S3와 달리) flat_demand 균등값 자체를 직접 단언한다.
    """
    load = np.array([5.0] * 20 + [10.0] * 4)
    P_net, soc, pk_s = solve_peak(
        S_mw=5.0, E_mwh=100.0, load_mw=load, eta_c=1.0, eta_d=1.0, self_discharge=0.0,
    )
    d_star = 25.0 / 6.0
    pk_star = 35.0 / 6.0

    assert np.isclose(pk_s, pk_star, atol=ATOL), pk_s
    # 방전은 S=5까지 안 밀어붙이고 d*=25/6에서 멈춤 (S를 다 안 씀 - 충전상승이 지배하는 증거)
    assert np.allclose(P_net[20:], d_star, atol=ATOL), P_net[20:]
    assert np.all(P_net[20:] < 5.0 - ATOL), P_net[20:]  # S=5 미사용 확인
    charge_total = float(-P_net[:20].sum())
    assert np.isclose(charge_total, 4 * d_star, atol=ATOL), charge_total
    # 내부 최적점(슬랙 없음) -> 평탄부도 균등분산으로 유일하게 결정됨
    flat_demand = load[:20] - P_net[:20]
    assert np.allclose(flat_demand, pk_star, atol=ATOL), flat_demand
    assert np.isclose(pk_s, float(np.max(load - P_net)), atol=ATOL)
    print('test_peak_shaving_S5_charging_dominant_regime OK '
          '(S=5MW: 충전상승이 지배, pk*=35/6=5.8333MW에서 정지, 방전 25/6MW < S=5 미사용 확인)')


def test_peak_shaving_E12_S_sweep():
    """E=12MWh(위 테스트들의 E=100과 달리 SOC 용량 자체가 병목이 될 수 있는 값)로 낮추고
    S=1,2,3,4MW를 스윕. load/eta는 위 테스트들과 동일. 값을 추측하지 않고 직접 유도한다.

    SOC 헤드룸(시작 50%에서 상한 95%까지) = 0.45*E = 0.45*12 = 5.4 MWh. 이 헤드룸이
    "하루 안에 미리 충전해 방전으로 쓸 수 있는 총에너지"의 절대 상한이다(SOC[0]=SOC[24]
    고정 + 인과성: 피크가 뒤쪽 4시간이라 그 앞 20시간에 미리 채워둔 만큼만 방전 가능,
    E=100 테스트들의 "무손실 에너지수지" 논거와 동일 구조).

    총방전량 X = min(4S, 0.45E) - 전력상한(4S)과 에너지상한(0.45E) 중 더 타이트한 쪽.
    교차점: 4S = 0.45E -> S* = 0.45*12/4 = 1.35 MW.
    - S < 1.35: 전력상한이 지배 (S=1..4 테스트들의 기존 체제와 동일). d=S, pk=10-S.
    - S >= 1.35: 에너지상한이 지배. X는 5.4로 고정 -> S를 2,3,4로 늘려도 방전은
      d=5.4/4=1.35MW에서 멈추고 pk=10-1.35=8.65MW로 전혀 개선되지 않는다.
    이는 위 S=5(E=100) 테스트의 "전력 vs 전력" 크로스오버(d*=25/6)와는 다른 종류의 정체다:
    거기선 방전전력 자체가 스스로의 충전부담과 부딪혔지만, 여기선 **저장용량(E)이 설비(S)보다
    먼저 바닥나** S를 계속 늘려도 못 쓰는 잉여전력이 생긴다("설비 vs 저장" 병목 전환).

    ★ S=5(E=100) 테스트에서 평탄부까지 균등값으로 유일하게 결정됐던 건 "제약이 타이트해서"라는
    일반론이 아니라, 그 특정 케이스에서 평탄부 수요가 pk에 정확히 붙어(등호로) 있었기 때문이다
    (평탄측 5+d*/5 = 방전측 10-d* = pk*, 두 부등식이 동시에 등호). 여기 E=12 케이스는 병목이
    SOC(에너지)로 옮겨갔을 뿐이고, 평탄부 수요는 pk=8.65에 전혀 안 붙는다(균등분산 가정으로도
    5+X/20 = 5.27~5.62 수준 << 8.65, 느슨함) - 그래서 평탄부 20시간 분배는 여전히 degenerate
    하다. 방전측(P_net[20:])은 반대로 "각 시간 d 미만이면 그 시간이 새 pk가 되고, 총합은 X로
    고정"이라는 별개의(느슨함과 무관한) 논거로 언제나 유일하게 결정되므로 등식으로 검증해도 된다.

    ★ degenerate 여부를 std(표준편차)가 아니라 "제약 활성/비활성"으로 단언하는 이유: std는
    물리가 아니라 그 순간 cvxpy가 고른 특정 해(솔버 구현 디테일)를 검증하는 것이다. degenerate
    문제에서 어느 해가 나오는지는 솔버·버전에 달렸으므로, std 임계값 검증은 cvxpy/솔버가 바뀌면
    물리와 무관한 이유로 실패할 수 있다(실측: S=1,E=12에서 std=0.0164로 임계값 0.01에 아슬아슬
    했음). 게다가 솔버가 우연히 균등해를 냈다면 그것도 엄연히 최적해인데 "LP가 틀렸다"는 잘못된
    신호를 준다. 대신 여기서 검증할 물리적 사실은 "평탄부 제약이 느슨하다(비활성)"는 것 자체이고,
    이는 어떤 해가 나오든 상관없이 max(flat_demand) < pk_s (여유 있음, 0.1 마진)로 구조적으로
    단언 가능하다 - 이게 유일해 케이스(S=5, 등호로 붙음 -> allclose)와 이 케이스(비유일, 여유
    있음 -> max<pk_s)를 가르는 진짜 기준이다.
    """
    load = np.array([5.0] * 20 + [10.0] * 4)
    E = 12.0
    headroom = 0.45 * E  # = 5.4 MWh
    results = {}

    for S in (1.0, 2.0, 3.0, 4.0):
        P_net, soc, pk_s = solve_peak(
            S_mw=S, E_mwh=E, load_mw=load, eta_c=1.0, eta_d=1.0, self_discharge=0.0,
        )
        X = min(4 * S, headroom)
        d = X / 4.0
        pk_expected = 10.0 - d

        assert np.isclose(pk_s, pk_expected, atol=ATOL), (S, pk_s, pk_expected)
        # 방전은 항상 균등분산으로 유일하게 결정됨 (부록A #4 S=2/S=3/S=5 테스트와 동일 논거:
        # 각 시간 d 미만이면 그 시간이 pk를 넘어서고, 총합은 X로 고정)
        assert np.allclose(P_net[20:], d, atol=ATOL), (S, P_net[20:], d)
        charge_total = float(-P_net[:20].sum())
        assert np.isclose(charge_total, X, atol=ATOL), (S, charge_total, X)
        flat_demand = load[:20] - P_net[:20]
        assert np.all(flat_demand <= pk_s + ATOL), (S, flat_demand)
        assert np.isclose(pk_s, float(np.max(load - P_net)), atol=ATOL)
        # 평탄부 제약이 비활성(느슨)임을 구조적으로 단언 (solver가 어떤 degenerate 해를
        # 고르든 무관 - std 같은 해-의존적 지표 대신 "제약에서 얼마나 떨어져 있는가"로 확인)
        assert flat_demand.max() < pk_s - 0.1, (S, flat_demand.max(), pk_s)

        results[S] = pk_s
        regime = '전력상한(S) 지배' if 4 * S < headroom - ATOL else '에너지상한(E) 지배'
        print(f'  S={S:.0f}MW: pk_s={pk_s:.4f}MW, d={d:.4f}MW, {regime}, S 미사용여력={S-d:.4f}MW')

    # 정체 확인: 에너지상한 지배 구간(S=2,3,4)은 전부 동일한 pk (S를 늘려도 무의미)
    assert np.isclose(results[2.0], results[3.0], atol=ATOL)
    assert np.isclose(results[3.0], results[4.0], atol=ATOL)
    assert results[1.0] > results[2.0] + ATOL  # S=1(전력지배)만 아직 개선 여지가 있었음
    print('test_peak_shaving_E12_S_sweep OK (S=2,3,4에서 pk 정체 확인 - 에너지(E)가 설비(S)보다 먼저 병목)')


def test_peak_zero_power_no_shaving():
    """S=0 -> pk_s = max(load) (깎을 수 없음, LP검증#3과 동일 취지)."""
    load = np.array([5.0] * 20 + [10.0] * 4)
    P_net, soc, pk_s = solve_peak(S_mw=0.0, E_mwh=100.0, load_mw=load)
    assert np.allclose(P_net, 0.0)
    assert np.isclose(pk_s, 10.0, atol=ATOL), pk_s
    print('test_peak_zero_power_no_shaving OK')


if __name__ == '__main__':
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
    print('all lower_lp tests passed')
