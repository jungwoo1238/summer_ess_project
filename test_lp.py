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
    assert np.isclose(soc.max(), 0.95, atol=ATOL), soc.max()
    assert np.isclose(soc.min(), 0.50, atol=ATOL), soc.min()
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
    assert np.isclose(soc.max(), 0.95, atol=ATOL), soc.max()
    assert np.isclose(soc.min(), 0.50, atol=ATOL), soc.min()
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
    assert np.allclose(soc, 0.5)
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
    """
    load = np.array([5.0] * 20 + [10.0] * 4)
    P_net, soc, pk_s = solve_peak(
        S_mw=2.0, E_mwh=100.0, load_mw=load, eta_c=1.0, eta_d=1.0, self_discharge=0.0,
    )
    assert np.isclose(pk_s, 8.0, atol=ATOL), pk_s
    assert np.allclose(P_net[20:], 2.0, atol=ATOL), P_net[20:]
    # 자체 일관성: pk_s == max(load - P_net)
    assert np.isclose(pk_s, float(np.max(load - P_net)), atol=ATOL)
    print('test_peak_shaving_hand_calc OK')


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
    test_peak_zero_power_no_shaving()
    print('all lower_lp tests passed')
