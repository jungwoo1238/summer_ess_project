"""benefits.py 순수함수 검증 (CLAUDE.md 7절 검증철학과 동일 패턴).
benefits.py는 조류계산/LP를 호출하지 않으므로 전부 손으로 계산 가능한 장난감 케이스로 검증한다.
pytest 없이도 단독 실행 가능: `python test_benefits.py`
"""

import numpy as np

import params as PM
import benefits


# ============================================================
# b_energy
# ============================================================

def test_b_energy_hand_calc():
    """AVG_DAYS 3개 키 전부 채우되 'shoulder'는 delta=0으로 둬 실질적으로 summer/winter
    2개 시나리오만 기여하도록 구성(손계산 단순화). T=3(24가 아님 - 하드코딩 길이 없음 확인).

    summer: delta=2(MW) 고정, smp=100(원/MWh), 3시각 -> day_won = 2*100*3*dt(1.0) = 600
    winter: delta=3(MW) 고정, smp=200(원/MWh), 3시각 -> day_won = 3*200*3*1.0 = 1800
    shoulder: delta=0 -> day_won = 0 (smp를 999로 둬도 기여 없어야 함 - 곱셈 누락 검증)

    total = N_WD[summer]*600 + N_WD[winter]*1800 + N_WD[shoulder]*0
          = 10*600 + 5*1800 + 1*0 = 6000 + 9000 = 15000
    """
    p_slack_base = {
        'summer': np.array([10.0, 10.0, 10.0]),
        'winter': np.array([20.0, 20.0, 20.0]),
        'shoulder': np.array([5.0, 5.0, 5.0]),
    }
    p_slack_ess = {
        'summer': np.array([8.0, 8.0, 8.0]),      # delta=2
        'winter': np.array([17.0, 17.0, 17.0]),   # delta=3
        'shoulder': np.array([5.0, 5.0, 5.0]),    # delta=0
    }
    smp_mwh = {
        'summer': np.array([100.0, 100.0, 100.0]),
        'winter': np.array([200.0, 200.0, 200.0]),
        'shoulder': np.array([999.0, 999.0, 999.0]),
    }
    n_weekdays = {'summer': 10.0, 'winter': 5.0, 'shoulder': 1.0}

    val = benefits.b_energy(p_slack_base, p_slack_ess, smp_mwh, n_weekdays)
    assert np.isclose(val, 15000.0), val
    print('test_b_energy_hand_calc OK', val)


def test_b_energy_missing_scenario_key_raises():
    """AVG_DAYS 중 'shoulder'를 뺀 dict를 넣으면 즉시 AssertionError (인덱스 방식이면 조용히
    틀린 값을 냈을 상황 - dict+키검증이 이걸 막는다는 것 자체를 검증)."""
    partial = {
        'summer': np.array([1.0, 1.0]),
        'winter': np.array([1.0, 1.0]),
    }
    full = {'summer': np.array([1.0, 1.0]), 'winter': np.array([1.0, 1.0]), 'shoulder': np.array([1.0, 1.0])}
    n_wd = {'summer': 1.0, 'winter': 1.0, 'shoulder': 1.0}
    try:
        benefits.b_energy(partial, full, full, n_wd)
        raised = False
    except AssertionError:
        raised = True
    assert raised, 'shoulder 키 누락인데도 통과함'
    print('test_b_energy_missing_scenario_key_raises OK')


# ============================================================
# b_defer
# ============================================================

def test_b_defer_positive_when_peak_reduced():
    """ESS가 PEAK_DAYS의 슬랙피크를 낮추면 B_defer > 0.
    base: summer_peak max=12, winter_peak max=8 -> combined max=12
    ess:  summer_peak max=9,  winter_peak max=8 -> combined max=9
    reduction = 3 MW -> B_defer = C_CAP_PER_MW_YR * 3
    """
    base = {'summer_peak': np.array([10.0, 12.0, 9.0]), 'winter_peak': np.array([8.0, 7.0, 6.0])}
    ess = {'summer_peak': np.array([9.0, 9.0, 9.0]), 'winter_peak': np.array([8.0, 7.0, 6.0])}
    val = benefits.b_defer(base, ess)
    expected = PM.C_CAP_PER_MW_YR * 3.0
    assert np.isclose(val, expected), (val, expected)
    print('test_b_defer_positive_when_peak_reduced OK', val)


def test_b_defer_negative_when_peak_increased():
    """★ 피크 증가 시 음수가 나와야 한다(클리핑 금지를 못박는 테스트).
    base: 양쪽 다 max=10. ess: summer_peak이 13까지 올라감(피크시각 충전으로 슬랙피크 악화).
    B_defer = C_CAP_PER_MW_YR * (10-13) = -3*C_CAP_PER_MW_YR < 0
    """
    base = {'summer_peak': np.array([10.0, 10.0, 10.0]), 'winter_peak': np.array([8.0, 8.0, 8.0])}
    ess = {'summer_peak': np.array([10.0, 10.0, 13.0]), 'winter_peak': np.array([8.0, 8.0, 8.0])}
    val = benefits.b_defer(base, ess)
    expected = PM.C_CAP_PER_MW_YR * (10.0 - 13.0)
    assert val < 0, val
    assert np.isclose(val, expected), (val, expected)
    print('test_b_defer_negative_when_peak_increased OK', val)


def test_b_defer_selects_larger_of_two_peak_scenarios():
    """PEAK_DAYS 두 시나리오 중 큰 쪽이 단일 max로 선택되는지.
    base: summer_peak max=5, winter_peak max=20 -> winter_peak(20)이 지배해야 함.
    ess:  winter_peak max을 15로 낮춤 -> combined base=20, ess=15, 감소분=5.
    (summer_peak는 base/ess 동일해 기여 없음 - winter_peak만 선택됐는지 확인하는 구성)
    """
    base = {'summer_peak': np.array([5.0, 5.0]), 'winter_peak': np.array([20.0, 3.0])}
    ess = {'summer_peak': np.array([5.0, 5.0]), 'winter_peak': np.array([15.0, 3.0])}
    val = benefits.b_defer(base, ess)
    expected = PM.C_CAP_PER_MW_YR * (20.0 - 15.0)
    assert np.isclose(val, expected), (val, expected)
    print('test_b_defer_selects_larger_of_two_peak_scenarios OK', val)


# ============================================================
# capex / opex / total_cost
# ============================================================

def test_capex_opex_cost_hand_calc():
    """S=1MVA, E=4MWh. benefits 함수 내부 공식과 별개로 params 상수를 직접 곱해
    기대값을 재계산(순환검증 방지 - 오타·단위누락을 잡기 위한 독립 재계산)."""
    S, E = 1.0, 4.0
    expected_capex = PM.C_KW_CAPEX_PER_MVA * S + PM.C_KWH_CAPEX_PER_MWH * E
    expected_opex = PM.C_KWH_OPEX_PER_MWH_YR * E + PM.WARRANTY_RATE * expected_capex
    expected_cost = PM.CRF_20 * expected_capex + expected_opex

    assert np.isclose(benefits.capex(S, E), expected_capex)
    assert np.isclose(benefits.opex(S, E), expected_opex)
    assert np.isclose(benefits.total_cost(S, E), expected_cost)
    print('test_capex_opex_cost_hand_calc OK', expected_capex, expected_opex, expected_cost)


def test_capex_opex_cost_zero_at_zero_size():
    """S=0, E=0 -> CAPEX=OPEX=Cost=0."""
    assert benefits.capex(0.0, 0.0) == 0.0
    assert benefits.opex(0.0, 0.0) == 0.0
    assert benefits.total_cost(0.0, 0.0) == 0.0
    print('test_capex_opex_cost_zero_at_zero_size OK')


def test_total_cost_uses_crf20_not_crf30():
    """CRF_30을 잘못 썼다면 이 값과 달라야 한다 - CRF 구분(CLAUDE.md 4절) 실수를 잡는 테스트."""
    S, E = 2.0, 5.0
    cost_with_crf20 = PM.CRF_20 * benefits.capex(S, E) + benefits.opex(S, E)
    cost_with_wrong_crf30 = PM.CRF_30 * benefits.capex(S, E) + benefits.opex(S, E)
    assert np.isclose(benefits.total_cost(S, E), cost_with_crf20)
    assert not np.isclose(benefits.total_cost(S, E), cost_with_wrong_crf30)
    print('test_total_cost_uses_crf20_not_crf30 OK')


# ============================================================
# assert_slack_balance
# ============================================================

def test_assert_slack_balance_passes_on_consistent_data():
    scenarios = ['summer']
    p_slack = {'summer': np.array([10.0, 11.0])}
    load_sum = {'summer': np.array([9.5, 10.5])}
    loss = {'summer': np.array([0.5, 0.5])}
    p_ess = {'summer': np.array([0.0, 0.0])}
    # p_slack - (load_sum+loss-p_ess) = [10-10, 11-11] = [0,0] -> 통과해야 함
    benefits.assert_slack_balance(p_slack, load_sum, loss, p_ess, scenarios=scenarios)
    print('test_assert_slack_balance_passes_on_consistent_data OK')


def test_assert_slack_balance_fails_on_broken_data():
    """일부러 어긋난 데이터 -> 실제로 AssertionError가 나는지 확인 (assert 자체가 작동하는지)."""
    scenarios = ['summer']
    p_slack = {'summer': np.array([10.0, 11.0])}
    load_sum = {'summer': np.array([9.5, 10.5])}
    loss = {'summer': np.array([0.5, 0.5])}
    p_ess = {'summer': np.array([0.0, 1.0])}  # 두번째 시각에 1MW 어긋남
    try:
        benefits.assert_slack_balance(p_slack, load_sum, loss, p_ess, scenarios=scenarios)
        raised = False
    except AssertionError:
        raised = True
    assert raised, '어긋난 데이터인데도 통과함'
    print('test_assert_slack_balance_fails_on_broken_data OK')


def test_assert_slack_balance_enabled_false_skips_check():
    """enabled=False면 어긋난 데이터라도 그냥 통과(lower_lp._assert_physics와 동일 패턴)."""
    scenarios = ['summer']
    p_slack = {'summer': np.array([10.0])}
    load_sum = {'summer': np.array([0.0])}
    loss = {'summer': np.array([0.0])}
    p_ess = {'summer': np.array([0.0])}  # 완전히 어긋남 (10 vs 0)
    benefits.assert_slack_balance(p_slack, load_sum, loss, p_ess, scenarios=scenarios, enabled=False)
    print('test_assert_slack_balance_enabled_false_skips_check OK')


def test_assert_slack_balance_missing_key_raises():
    scenarios = ['summer', 'winter']
    p_slack = {'summer': np.array([1.0])}  # winter 누락
    load_sum = {'summer': np.array([1.0]), 'winter': np.array([1.0])}
    loss = {'summer': np.array([0.0]), 'winter': np.array([0.0])}
    p_ess = {'summer': np.array([0.0]), 'winter': np.array([0.0])}
    try:
        benefits.assert_slack_balance(p_slack, load_sum, loss, p_ess, scenarios=scenarios)
        raised = False
    except AssertionError:
        raised = True
    assert raised, 'winter 키 누락인데도 통과함'
    print('test_assert_slack_balance_missing_key_raises OK')


# ============================================================
# b_arb / b_loss / 검산(b_arb+b_loss ~= b_energy)
# ============================================================

def test_arb_loss_energy_identity_on_consistent_data():
    """정합 데이터(P_slack = SigmaLoad+Loss-P_ess 항등식을 만족하도록 직접 구성)에서
    b_arb + b_loss == b_energy 확인 (CLAUDE.md 8절 검산).

    P_ess는 "실제 주입값" 기준: p_ess_net[s,t] = p_dis[s,t]-p_ch[s,t].
    p_slack_base = load + loss_base (ESS 없음)
    p_slack_ess  = load + loss_ess - p_ess_net
    """
    load = {
        'summer': np.array([50.0, 50.0, 50.0]),
        'winter': np.array([60.0, 60.0, 60.0]),
        'shoulder': np.array([40.0, 40.0, 40.0]),
    }
    loss_base = {
        'summer': np.array([1.0, 1.0, 1.0]),
        'winter': np.array([2.0, 2.0, 2.0]),
        'shoulder': np.array([0.5, 0.5, 0.5]),
    }
    loss_ess = {
        'summer': np.array([0.8, 0.8, 0.8]),
        'winter': np.array([1.5, 1.5, 1.5]),
        'shoulder': np.array([0.5, 0.5, 0.5]),
    }
    p_ch = {
        'summer': np.array([0.0, 0.0, 0.0]),
        'winter': np.array([0.0, 0.0, 1.0]),
        'shoulder': np.array([0.0, 0.0, 0.0]),
    }
    p_dis = {
        'summer': np.array([2.0, 2.0, 2.0]),
        'winter': np.array([3.0, 3.0, 2.0]),
        'shoulder': np.array([0.0, 0.0, 0.0]),
    }
    smp_mwh = {
        'summer': np.array([100.0, 110.0, 120.0]),
        'winter': np.array([200.0, 210.0, 220.0]),
        'shoulder': np.array([150.0, 150.0, 150.0]),
    }
    n_weekdays = {'summer': 63.6, 'winter': 60.8, 'shoulder': 122.6}

    p_slack_base = {s: load[s] + loss_base[s] for s in PM.AVG_DAYS}
    p_ess_net = {s: p_dis[s] - p_ch[s] for s in PM.AVG_DAYS}
    p_slack_ess = {s: load[s] + loss_ess[s] - p_ess_net[s] for s in PM.AVG_DAYS}

    b_energy_val = benefits.b_energy(p_slack_base, p_slack_ess, smp_mwh, n_weekdays)
    b_arb_val = benefits.b_arb(p_ch, p_dis, smp_mwh, n_weekdays)
    b_loss_val = benefits.b_loss(loss_base, loss_ess, smp_mwh, n_weekdays)

    assert benefits.check_b_energy_decomposition(b_arb_val, b_loss_val, b_energy_val)
    assert np.isclose(b_arb_val + b_loss_val, b_energy_val)
    print('test_arb_loss_energy_identity_on_consistent_data OK',
          b_arb_val, b_loss_val, b_energy_val)


def test_b_loss_can_be_negative():
    """★ 손실 편익이 음수여도 정상(버그 아님) - 심야 대량충전으로 그 시각 손실이 늘어난 경우."""
    loss_base = {'summer': np.array([0.5, 0.5]), 'winter': np.array([0.5, 0.5]), 'shoulder': np.array([0.5, 0.5])}
    loss_ess = {'summer': np.array([1.5, 1.5]), 'winter': np.array([0.5, 0.5]), 'shoulder': np.array([0.5, 0.5])}
    smp_mwh = {'summer': np.array([100.0, 100.0]), 'winter': np.array([100.0, 100.0]), 'shoulder': np.array([100.0, 100.0])}
    n_weekdays = {'summer': 1.0, 'winter': 1.0, 'shoulder': 1.0}
    val = benefits.b_loss(loss_base, loss_ess, smp_mwh, n_weekdays)
    assert val < 0, val
    print('test_b_loss_can_be_negative OK', val)


def test_loss_pcs_hand_calc_reference_table():
    """CMD_pcs_loss.md 2절 참고 수치 표 재현 (eta_pcs=0.97=params.ETA_PCS 기본값).
    Loss_pcs = (1-eta)*[sqrt(P^2+Q^2) - |P|]."""
    p_net = {'x': np.array([0.00, 0.14, 0.00])}
    q_mvar = {'x': np.array([1.20, 1.20, 1.68])}
    result = benefits.loss_pcs(p_net, q_mvar)
    expected = np.array([0.0360, 0.0320, 0.0504])
    assert np.allclose(result['x'], expected, atol=1e-3), result['x']
    print('test_loss_pcs_hand_calc_reference_table OK', result['x'])


def test_loss_pcs_zero_when_q_zero():
    """Q=0이면 P가 무엇이든 정확히 0 (삼각부등식 - 회귀 안전성의 근거)."""
    p_net = {'a': np.array([-2.4, -0.5, 0.0, 0.5, 2.4])}
    q_mvar = {'a': np.zeros(5)}
    result = benefits.loss_pcs(p_net, q_mvar)
    assert np.allclose(result['a'], 0.0, atol=0.0), result['a']
    print('test_loss_pcs_zero_when_q_zero OK')


def test_loss_pcs_custom_eta_and_key_mismatch_raises():
    """eta_pcs 오버라이드가 실제로 반영되는지 + p_net/q_mvar 시나리오 키가 다르면 즉시 실패."""
    p_net = {'a': np.array([0.0])}
    q_mvar = {'a': np.array([1.0])}
    result = benefits.loss_pcs(p_net, q_mvar, eta_pcs=0.9)
    assert np.isclose(result['a'][0], 0.1), result['a']  # (1-0.9)*(1.0-0.0)=0.1

    try:
        benefits.loss_pcs({'a': np.array([0.0])}, {'b': np.array([0.0])})
        raised = False
    except AssertionError:
        raised = True
    assert raised, '시나리오 키 불일치인데 에러가 안 남'
    print('test_loss_pcs_custom_eta_and_key_mismatch_raises OK')


def test_check_b_energy_decomposition_detects_mismatch():
    assert benefits.check_b_energy_decomposition(100.0, 50.0, 150.0) is True
    assert benefits.check_b_energy_decomposition(100.0, 50.0, 200.0) is False
    print('test_check_b_energy_decomposition_detects_mismatch OK')


if __name__ == '__main__':
    test_b_energy_hand_calc()
    test_b_energy_missing_scenario_key_raises()
    test_b_defer_positive_when_peak_reduced()
    test_b_defer_negative_when_peak_increased()
    test_b_defer_selects_larger_of_two_peak_scenarios()
    test_capex_opex_cost_hand_calc()
    test_capex_opex_cost_zero_at_zero_size()
    test_total_cost_uses_crf20_not_crf30()
    test_assert_slack_balance_passes_on_consistent_data()
    test_assert_slack_balance_fails_on_broken_data()
    test_assert_slack_balance_enabled_false_skips_check()
    test_assert_slack_balance_missing_key_raises()
    test_arb_loss_energy_identity_on_consistent_data()
    test_b_loss_can_be_negative()
    test_loss_pcs_hand_calc_reference_table()
    test_loss_pcs_zero_when_q_zero()
    test_loss_pcs_custom_eta_and_key_mismatch_raises()
    test_check_b_energy_decomposition_detects_mismatch()
    print('all benefits tests passed')
