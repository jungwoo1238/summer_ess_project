"""evaluate.py 통합 검증 (CLAUDE.md - evaluate.py 작업 지시 "test_evaluate.py").

조류계산이 붙어 test_lp.py/test_benefits.py(순수함수, 초 단위)와 달리 무겁다
(평가 1회 = 조류계산 120회). 전체를 slow로 표시해 `pytest -m "not slow"`로 기본 실행에서
제외할 수 있게 한다.

test_lp/test_benefits처럼 손계산 대조가 불가능하므로(조류계산은 손으로 못 품) 대신
불변량·극한을 검증한다(CLAUDE.md 7절 원칙 3): S=0/E=0 극한, 기저 재현, 실데이터 수지,
편익 분해 검산, 발산 처리, fitness 유한성, 다수기 구조.

pytest 없이도 단독 실행 가능: `python test_evaluate.py`
"""

import time

import numpy as np
import pandapower as pp
import pytest

import params as PM
import evaluate
from build_net import build_net

pytestmark = pytest.mark.slow


# ============================================================
# 1. 기저 재현 (CLAUDE.md 1절 0번 검증값 + evaluate._compute_base_flow 교차확인)
# ============================================================

def test_base_reproduces_validation():
    """summer_peak, t=18의 배율이 정확히 1.0 -> case33bw 기본부하 상태와 동일해야 한다
    (5절 "부하 정규화 1.0 = 여름최대일 t=18 = case33bw 기본부하"). build_net을 독립적으로
    돌려 0번 검증값을 재현하는지 먼저 확인하고, evaluate.init_worker()가 캐싱한
    _BASE_FLOW의 같은 지점(p_slack, loss)이 정확히 일치하는지 대조한다."""
    scale = PM.LOAD['summer_peak'][18]
    assert np.isclose(scale, 1.0), scale

    net = build_net()
    base_p = net.load['p_mw'].to_numpy().copy()
    base_q = net.load['q_mvar'].to_numpy().copy()
    net.load['p_mw'] = base_p * scale
    net.load['q_mvar'] = base_q * scale
    pp.runpp(net, numba=True, init='results')

    loss_kw = net.res_line.pl_mw.sum() * 1000
    vmin = net.res_bus.vm_pu.min()
    vmin_bus = net.res_bus.vm_pu.idxmin()
    line0_a = net.res_line.at[0, 'i_ka'] * 1000
    slack_mw = net.res_ext_grid.p_mw.sum()

    assert np.isclose(loss_kw, PM.VALIDATION['loss_kw_scaled'], atol=0.05), loss_kw
    assert np.isclose(vmin, PM.VALIDATION['vmin_pu_scaled'], atol=1e-3), vmin
    assert vmin_bus == PM.VALIDATION['vmin_bus'], vmin_bus
    assert np.isclose(line0_a, PM.VALIDATION['line0_current_a_scaled'], atol=0.5), line0_a
    assert np.isclose(slack_mw, PM.VALIDATION['slack_import_mw_scaled'], atol=1e-3), slack_mw

    # evaluate 쪽 캐시와 교차 확인
    evaluate.init_worker()
    base_flow = evaluate._get_base_flow()
    assert np.isclose(base_flow['loss']['summer_peak'][18] * 1000, PM.VALIDATION['loss_kw_scaled'], atol=0.05)
    assert np.isclose(base_flow['p_slack']['summer_peak'][18], PM.VALIDATION['slack_import_mw_scaled'], atol=1e-3)
    print('test_base_reproduces_validation OK', loss_kw, vmin, line0_a, slack_mw)


# ============================================================
# 2. S=0 / E=0 극한
# ============================================================

def test_zero_size_particle_matches_base_exactly():
    """S=0,E=0 (둘 다) -> 편익 0, Cost 0, 위반은 기저 계통 그대로(0이 아님 - 기저 Vmin=0.9407
    < 0.95라 이미 위반). p_slack_ess가 base_flow의 p_slack과 시나리오·시각별로 정확히 일치하는지,
    위반량(v_violation/i_violation)도 base_flow 캐시값과 정확히 일치하는지까지 확인한다."""
    evaluate.init_worker()
    base_flow = evaluate._get_base_flow()

    x = np.array([10.0, 0.0, 0.0, 0.0])
    detail = evaluate.evaluate_particle(x, return_detail=True)

    # ★ 기대값이 정확히 0인 비교(상대오차가 의미 없는 유일한 경우)라 여기서만 절대오차를 쓴다.
    # S=0,E=0이면 물리적으로 ESS 기여가 정확히 0이어야 하지만, base_flow와 이 평가는 같은 net을
    # 재사용하면서도 직전 시각 상태(warm start 이력)가 서로 달라 뉴턴법이 tolerance_mva(기본
    # 1e-8) 안의 다른 점에서 멈춘다 - scripts/probe_noise.py로 원인을 분리 확인함:
    #   실험1(sgen 유무, 이력 통제)  diff = 0 정확히           -> sgen 존재 자체는 무해
    #   실험2(이력만 다르게, sgen 없음) diff = 8.7e-10 MW      -> 이력 차이가 실제 원인
    #   실험3(S=1,E=4 실사용)        diff 최대 1.2e-8 MW/시각  -> 모든 평가에 공통된 잡음
    # (c) 극단 상한 추정(실험3 최대 diff x 최고 SMP x 최대 N_WD x 24h) = 6.91원 -> 10원이면
    # 여유 있게 넉넉하다(실측 0.028원과도 정합).
    won_atol = 10.0
    assert abs(detail['b_energy']) < won_atol, detail['b_energy']
    assert abs(detail['b_defer']) < won_atol, detail['b_defer']
    assert detail['cost'] == 0.0, detail['cost']  # cost는 조류계산과 무관한 순수 공식이라 정확히 0
    assert abs(detail['j_net']) < won_atol, detail['j_net']

    # MW 단위 비교(물리량): 금액과 달리 절대 스케일이 고정(8.8MW대)이므로 atol=1e-7,rtol=0 고정
    # (benefits.assert_slack_balance와 동일 기준 - CLAUDE.md 3절 확정값).
    for s in PM.ALL_DAYS:
        assert np.allclose(detail['p_slack_ess'][s], base_flow['p_slack'][s], atol=1e-7, rtol=0.0), s

    # 기저 계통 그대로이므로 0이 아닐 수 있음(Vmin=0.9407<0.95) - "0" 대신 base_flow와 정확히 대조.
    # pu 단위 물리량(위반량)도 MW와 같은 이유로 절대오차 고정.
    assert np.isclose(detail['v_violation'], base_flow['v_violation'], atol=1e-7, rtol=0.0)
    assert np.isclose(detail['i_violation'], base_flow['i_violation'], atol=1e-7, rtol=0.0)
    assert detail['v_violation'] > 0.0, '기저 Vmin=0.9407<0.95인데 위반량이 0으로 나옴 - 이상함'

    # fitness는 위반량 지배(LAMBDA_V~1e10)라 사실상 원 단위 금액이지만 기대값이 0이 아니므로
    # (수억원대 페널티) 상대오차로 비교한다.
    expected_fitness = (PM.LAMBDA_V * base_flow['v_violation']
                         + PM.LAMBDA_LINE * base_flow['i_violation'])
    assert np.isclose(detail['fitness'], expected_fitness, rtol=1e-9, atol=0.0), (detail['fitness'], expected_fitness)
    print('test_zero_size_particle_matches_base_exactly OK', detail['v_violation'], detail['i_violation'])


def test_zero_S_or_E_alone_gives_zero_operational_benefit_but_nonzero_cost():
    """S=0(E>0) 또는 E=0(S>0) 단독이어도 스케줄은 0(lower_lp의 S<=0 or E<=0 조기 반환)이라
    운영 편익(B_energy/B_defer)은 0이지만, Cost는 CAPEX(S,E) 공식대로 계산되므로 0이 아닐 수
    있다 - "편익 0"과 "Cost 0"을 같은 조건으로 뭉뚱그리면 안 된다는 것을 명시적으로 검증."""
    # 기대값이 정확히 0인 경우의 예외적 절대오차 - 근거는 test_zero_size_particle_matches_base_exactly
    # 상단 주석 및 scripts/probe_noise.py 참조(워밍스타트 이력 차이에 의한 조류계산 수렴오차).
    won_atol = 10.0
    for x in (np.array([10.0, 0.0, 5.0, 0.0]), np.array([10.0, 1.0, 0.0, 0.0])):
        detail = evaluate.evaluate_particle(x, return_detail=True)
        assert abs(detail['b_energy']) < won_atol, (x, detail['b_energy'])
        assert abs(detail['b_defer']) < won_atol, (x, detail['b_defer'])
        assert detail['cost'] > 0.0, (x, detail['cost'])
    print('test_zero_S_or_E_alone_gives_zero_operational_benefit_but_nonzero_cost OK')


# ============================================================
# 3 & 4. 실데이터 수지 검증 + b_arb+b_loss ~= b_energy 검산
# ============================================================

def test_realistic_particle_balance_and_decomposition():
    """중간 크기 입자(S=1.0MVA, E=4.0MWh, bus=17)로 실제 조류계산을 돌린다.
    evaluate_particle 내부에서 assert_slack_balance(atol=1e-7, rtol=0, ALL_DAYS 전부)가
    이미 통과해야 예외 없이 반환되므로, 예외 없이 detail이 나오는 것 자체가 3번 항목의 증거다.
    b_arb+b_loss ~= b_energy는 detail['decomposition_ok']로 별도 확인."""
    x = np.array([17.0, 1.0, 4.0, 0.0])
    detail = evaluate.evaluate_particle(x, return_detail=True)

    assert detail['diverged'] is False
    assert detail['decomposition_ok'] is True, (detail['b_arb'], detail['b_loss'], detail['b_energy'])
    # 금액(원 단위) 비교는 상대오차로: 값이 수백만~수억원 규모라 절대오차는 ESS 크기/기간에 따라
    # 스케일이 달라지는 임계값이 되어버린다(atol=0 명시 - np.isclose 기본 atol=1e-8이 남으면
    # 0 근처 비교에서 의도치 않게 통과할 수 있음).
    assert np.isclose(detail['b_arb'] + detail['b_loss'], detail['b_energy'], rtol=1e-9, atol=0.0)

    print('test_realistic_particle_balance_and_decomposition OK')
    print(f"  b_energy={detail['b_energy']:.2f} b_arb={detail['b_arb']:.2f} b_loss={detail['b_loss']:.2f} "
          f"b_defer={detail['b_defer']:.2f} cost={detail['cost']:.2f} j_net={detail['j_net']:.2f}")
    print(f"  v_violation={detail['v_violation']:.6f} i_violation={detail['i_violation']:.6f}")


# ============================================================
# 5. 발산 처리
# ============================================================

def test_divergence_returns_penalty_not_exception():
    """pp.runpp를 강제로 항상 실패하게 monkeypatch해 발산 경로를 결정론적으로 유도한다
    (물리적으로 PSO 탐색경계 안에서 pandapower를 실제로 발산시키기는 어려움 - 33버스 방사형
    계통은 상당히 견고함). init_worker()는 patch 전에 먼저 호출해 기저 조류계산까지 끝내
    둔다(패치된 runpp로 기저 계산까지 걸리면 그건 이 테스트가 확인하려는 것과 다른 경로)."""
    evaluate.init_worker()
    evaluate.reset_divergence_log()

    original_runpp = evaluate.pp.runpp

    def _always_raise(*args, **kwargs):
        raise RuntimeError('강제 발산 (테스트용 monkeypatch)')

    evaluate.pp.runpp = _always_raise
    try:
        x = np.array([10.0, 1.0, 4.0, 0.0])
        fitness = evaluate.evaluate_particle(x)
        detail = evaluate.evaluate_particle(x, return_detail=True)
    finally:
        evaluate.pp.runpp = original_runpp

    assert fitness == PM.PENALTY_DIVERGE, fitness
    assert np.isfinite(fitness)
    assert detail['diverged'] is True
    assert detail['fitness'] == PM.PENALTY_DIVERGE
    assert detail['diverge_info']['scenario'] in PM.ALL_DAYS

    stats = evaluate.get_divergence_stats()
    assert stats['total_retries'] >= 2, stats  # 두 evaluate_particle 호출 각각 1건씩
    assert stats['recovered'] == 0, stats

    # 패치 복구 후 정상 평가가 다시 되는지도 확인 (전역 상태 오염 없음)
    fitness_after = evaluate.evaluate_particle(x)
    assert np.isfinite(fitness_after) and fitness_after != PM.PENALTY_DIVERGE
    print('test_divergence_returns_penalty_not_exception OK', stats)


# ============================================================
# 6. fitness 유한성
# ============================================================

def test_fitness_always_finite_scalar():
    particles = [
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([32.0, 2.4, 10.2, 0.0]),
        np.array([17.0, 1.2, 5.0, 0.0]),
    ]
    for x in particles:
        fitness = evaluate.evaluate_particle(x)
        assert isinstance(fitness, float), (x, type(fitness))
        assert np.isfinite(fitness), (x, fitness)
    print('test_fitness_always_finite_scalar OK')


# ============================================================
# 7. 다수기 구조 (n=1, n=2, 중복배치)
# ============================================================

def test_multi_unit_n1_and_n2_structural():
    x1 = np.array([17.0, 1.0, 4.0, 0.0])
    detail1 = evaluate.evaluate_particle(x1, return_detail=True)
    assert detail1['b'].shape == (1,)
    assert detail1['unit_p']['summer'].shape == (1, PM.TIME_STEPS)

    # n=2, 서로 다른 버스
    x2 = np.array([17.0, 1.0, 4.0, 0.0, 25.0, 0.5, 2.0, 0.0])
    detail2 = evaluate.evaluate_particle(x2, return_detail=True)
    assert detail2['diverged'] is False
    assert detail2['b'].shape == (2,)
    assert detail2['unit_p']['summer'].shape == (2, PM.TIME_STEPS)
    assert list(detail2['b']) == [17, 25]

    # n=2, 동일 버스 중복배치 ((a)방식: sgen 2개 별도 생성, 병합 안 함)
    x_dup = np.array([17.0, 1.0, 4.0, 0.0, 17.0, 0.8, 3.0, 0.0])
    detail_dup = evaluate.evaluate_particle(x_dup, return_detail=True)
    assert detail_dup['diverged'] is False
    assert list(detail_dup['b']) == [17, 17]
    assert len(evaluate._NET.sgen) == 2

    print('test_multi_unit_n1_and_n2_structural OK')


# ============================================================
# 성능 실측 (평가 1회 = 조류계산 120회, 목표 ~1.178초/회 - CLAUDE.md 7절)
# ============================================================

def test_and_report_single_evaluation_timing():
    evaluate.init_worker()  # 워밍업(기저 조류계산은 시간 측정에서 제외)
    x = np.array([17.0, 1.0, 4.0, 0.0])
    evaluate.evaluate_particle(x)  # JIT/캐시 워밍업 1회

    t0 = time.perf_counter()
    evaluate.evaluate_particle(x)
    elapsed = time.perf_counter() - t0

    print(f'test_and_report_single_evaluation_timing: {elapsed:.3f}초/평가 (목표 참고치 1.178초)')
    assert elapsed < 10.0, f'평가 1회가 비정상적으로 느림: {elapsed:.3f}초'


if __name__ == '__main__':
    test_base_reproduces_validation()
    test_zero_size_particle_matches_base_exactly()
    test_zero_S_or_E_alone_gives_zero_operational_benefit_but_nonzero_cost()
    test_realistic_particle_balance_and_decomposition()
    test_divergence_returns_penalty_not_exception()
    test_fitness_always_finite_scalar()
    test_multi_unit_n1_and_n2_structural()
    test_and_report_single_evaluation_timing()
    print('all evaluate tests passed')
