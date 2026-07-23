"""evaluate.py 통합 검증 (CLAUDE.md - evaluate.py 작업 지시 "test_evaluate.py").

조류계산이 붙어 test_lp.py/test_benefits.py(순수함수, 초 단위)와 달리 무겁다
(평가 1회 = 조류계산 120회). 전체를 slow로 표시해 `pytest -m "not slow"`로 기본 실행에서
제외할 수 있게 한다.

test_lp/test_benefits처럼 손계산 대조가 불가능하므로(조류계산은 손으로 못 품) 대신
불변량·극한을 검증한다(CLAUDE.md 7절 원칙 3): S=0/E=0 극한, 기저 재현, 전압위반 유무
(현행 슬랙 1.02 / 구 슬랙 1.0 회귀), 실데이터 수지, 편익 분해 검산, 발산 처리,
fitness 유한성, 다수기 구조.

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
    돌려(슬랙 전압은 인자 생략 -> PM.SLACK_VM_PU 현행 기본값 1.02) 검증값을 재현하는지
    먼저 확인하고, evaluate.init_worker()가 캐싱한 _BASE_FLOW의 같은 지점(p_slack, loss)이
    정확히 일치하는지 대조한다.

    ★ slack_import_mw_scaled는 PM.VALIDATION에 없다(현행 기준 - CLAUDE.md 1절: 슬랙 유입은
    부하 프로파일·슬랙 전압에 둘 다 의존해 단일 검증값으로 부적절하다고 명시적으로 제외됨).
    구 슬랙(1.0) 기준값에는 남아 있다 - PM.VALIDATION_LEGACY_SLACK_1P0 참조."""
    scale = PM.LOAD['summer_peak'][18]
    assert np.isclose(scale, 1.0), scale

    net = build_net()  # slack_vm_pu 생략 -> PM.SLACK_VM_PU(현행 1.02)
    base_p = net.load['p_mw'].to_numpy().copy()
    base_q = net.load['q_mvar'].to_numpy().copy()
    net.load['p_mw'] = base_p * scale
    net.load['q_mvar'] = base_q * scale
    pp.runpp(net, numba=True, init='results')

    loss_kw = net.res_line.pl_mw.sum() * 1000
    vmin = net.res_bus.vm_pu.min()
    vmin_bus = net.res_bus.vm_pu.idxmin()
    line0_a = net.res_line.at[0, 'i_ka'] * 1000
    slack_mw = net.res_ext_grid.p_mw.sum()  # 참고용 출력만, 검증 비교 대상 아님(위 설명 참조)

    assert np.isclose(loss_kw, PM.VALIDATION['loss_kw_scaled'], atol=0.05), loss_kw
    assert np.isclose(vmin, PM.VALIDATION['vmin_pu_scaled'], atol=1e-3), vmin
    assert vmin_bus == PM.VALIDATION['vmin_bus'], vmin_bus
    assert np.isclose(line0_a, PM.VALIDATION['line0_current_a_scaled'], atol=0.5), line0_a

    # evaluate 쪽 캐시와 교차 확인
    evaluate.init_worker()
    base_flow = evaluate._get_base_flow()
    assert np.isclose(base_flow['loss']['summer_peak'][18] * 1000, PM.VALIDATION['loss_kw_scaled'], atol=0.05)
    print('test_base_reproduces_validation OK', loss_kw, vmin, line0_a, slack_mw)


# ============================================================
# 2. S=0 / E=0 극한
# ============================================================

def test_zero_size_particle_matches_base_exactly():
    """S=0,E=0(둘 다) -> ESS가 계통에 아무 영향도 못 주므로 실제 조류계산 결과가 base_flow와
    시나리오·시각별로 정확히 일치해야 한다. 편익도 0, Cost도 0이다.

    ★ 현행 기본값(슬랙 1.02)에서는 기저 위반량 자체가 0이다(test_base_has_no_voltage_violation
    참조) - 그래서 "이 케이스의 v_violation이 base_flow의 v_violation과 같은가"만 비교하면
    위반 계산 코드가 통째로 죽어 항상 0을 반환해도 0==0으로 자명하게 통과해버린다. 그래서 이
    테스트의 핵심 비교는 절대 0이 될 수 없는 실질 물리량(슬랙 유입 MW, 손실 MW - loss_ess)으로
    한다. 위반 계산 코드 자체가 실제로 살아 있는지는 test_base_violation_at_legacy_slack(구
    슬랙 1.0에서 0이 아닌 값이 나오는지)이 별도로 담당한다.

    ★ 검증 층위(서로 대체 불가 - 하나가 다른 하나를 커버하지 않는다):
      (1) 물리량   p_slack_ess / loss_ess / p_ess_total -> 조류계산 경로
      (2) 금액     b_energy / b_defer / cost / j_net    -> benefits 경로(조류계산이 안 봄)
      (3) 분해검산 decomposition_ok                      -> 편익 분해 항등식
      (4) 위반량   v_violation / i_violation             -> 페널티 입력(현재는 자명, 미래 대비)
      (5) 조립     fitness 재구성                        -> 부호 규약(3-A절 ★)
    """
    evaluate.init_worker()
    base_flow = evaluate._get_base_flow()

    x = np.array([10.0, 0.0, 0.0])
    detail = evaluate.evaluate_particle(x, return_detail=True)

    # ---- (1) 물리량: 조류계산 경로 -------------------------------------------------
    # MW 단위 비교: 금액과 달리 절대 스케일이 고정(8.8MW대)이므로 atol=1e-7, rtol=0 고정
    # (benefits.assert_slack_balance와 동일 기준 - CLAUDE.md 3절 확정값). p_slack_ess/loss_ess
    # 둘 다 절대 0이 될 수 없는 실질 물리량이라(부하가 있는 한 손실도 항상 양수) 위반량과
    # 달리 "우연히 0==0으로 통과"할 위험이 없다.
    for s in PM.ALL_DAYS:
        assert np.allclose(detail['p_slack_ess'][s], base_flow['p_slack'][s], atol=1e-7, rtol=0.0), s
        assert np.allclose(detail['loss_ess'][s], base_flow['loss'][s], atol=1e-7, rtol=0.0), s

    # 위 두 assert는 "결과가 기저와 같다"를 보는데, 이것만으로는 주입이 있었지만 상쇄되어
    # 결과가 우연히 같아진 경우(예: 같은 시각에 충전·방전이 부호만 뒤집혀 들어감)를 배제하지
    # 못한다. 이 assert는 "원인이 아예 없다"를 직접 본다.
    for s in PM.ALL_DAYS:
        assert np.allclose(detail['p_ess_total'][s], 0.0, atol=1e-7, rtol=0.0), \
            (s, np.max(np.abs(np.asarray(detail['p_ess_total'][s]))))

    # ---- (2) 금액: benefits 경로 ---------------------------------------------------
    # ★ 기대값이 정확히 0인 비교(상대오차가 의미 없는 유일한 경우)라 여기서만 절대오차를 쓴다.
    # S=0,E=0이면 물리적으로 ESS 기여가 정확히 0이어야 하지만, base_flow와 이 평가는 같은 net을
    # 재사용하면서도 직전 시각 상태(warm start 이력)가 서로 달라 뉴턴법이 tolerance_mva(기본
    # 1e-8) 안의 다른 점에서 멈춘다 - scripts/probe_noise.py로 원인을 분리 확인함:
    #   실험1(sgen 유무, 이력 통제)  diff = 0 정확히           -> sgen 존재 자체는 무해
    #   실험2(이력만 다르게, sgen 없음) diff = 8.7e-10 MW      -> 이력 차이가 실제 원인
    #   실험3(S=1,E=4 실사용)        diff 최대 1.2e-8 MW/시각  -> 모든 평가에 공통된 잡음
    # (c) 극단 상한 추정(실험3 최대 diff x 최고 SMP x 최대 N_WD x 24h) = 6.91원 -> 10원이면
    # 여유 있게 넉넉하다(실측 0.028원과도 정합).
    # ※ 이 층은 물리량 비교로 대체되지 않는다 - 조류계산이 멀쩡해도 benefits가 그 값을 받아
    #   편익을 조립하는 경로에서 버그가 나면 여기서만 걸린다. 특히 cost는 조류계산과 아무
    #   관계가 없다(S,E만의 순수 공식).
    won_atol = 10.0
    assert abs(detail['b_energy']) < won_atol, detail['b_energy']
    assert abs(detail['b_defer']) < won_atol, detail['b_defer']
    assert detail['cost'] == 0.0, detail['cost']  # cost는 조류계산과 무관한 순수 공식이라 정확히 0
    assert abs(detail['j_net']) < won_atol, detail['j_net']

    # ---- (3) 편익 분해 검산 --------------------------------------------------------
    # B_arb + B_loss ≈ B_energy (CLAUDE.md 3절 "슬랙 수지 검증"). 제로 입자는 모든 항이 0이라
    # 자명하게 통과할 것 같지만, 0-나누기나 NaN이 끼면 여기서 걸린다. detail에 이미 들어 있는
    # 필드를 아무도 보지 않으면 그 필드가 무의미해지므로 확인한다.
    assert detail['decomposition_ok'], \
        (detail['b_arb'], detail['b_loss'], detail['b_energy'])

    # ---- (4) 위반량 ----------------------------------------------------------------
    # 위반량도 base_flow와 일치해야 한다(부호·스케일 실수가 있으면 여기서 어긋남) - 단
    # "0이 아니어야 한다"는 방어는 여기서 하지 않는다. 현행 슬랙 1.02에서는 둘 다 자명하게
    # 0이라(test_base_has_no_voltage_violation) 그 자체로는 위반 계산 코드가 살아있다는
    # 증거가 못 된다 - 그 방어는 test_base_violation_at_legacy_slack이 담당한다.
    # (지금은 자명하지만 지우지 말 것: 3-B절 λ 재조정이나 슬랙 재검토로 기저 위반이 다시
    #  0이 아니게 되면 이 비교가 즉시 실질 검증으로 되살아난다.)
    assert np.isclose(detail['v_violation'], base_flow['v_violation'], atol=1e-7, rtol=0.0)
    assert np.isclose(detail['i_violation'], base_flow['i_violation'], atol=1e-7, rtol=0.0)

    # ---- (5) fitness 조립 ----------------------------------------------------------
    # fitness = -j_net + LAMBDA_V*v_violation + LAMBDA_LINE*i_violation (evaluate.py 소스와
    # 동일한 항 순서). base_flow가 아니라 detail 자신이 반환한 j_net/v_violation/i_violation로
    # 재구성한다 - base_flow는 별도의 조류계산(다른 warm-start 이력)이라 그걸 기준으로 삼으면
    # 그 자체의 미세한 부동소수 차이가 섞인다. 여기서는 같은 evaluate_particle 호출 안에서
    # 나온 값들끼리만 비교하므로 원칙적으로 정확히 같아야 한다(수식을 소스와 동일한 순서로
    # 재현하면 IEEE754 연산은 결정적이라 비트까지 같다). atol=1e-9는 "==" 대신 쓰는 안전장치 -
    # evaluate.py가 나중에 항 순서를 바꾸는 무해한 리팩터로도 "=="는 깨질 수 있는데(덧셈은
    # 결합법칙이 성립 안 함), 실제 버그의 오차 규모(수 원~수십억 원)와는 비교가 안 되게
    # 작은 값이라 검출력은 그대로다(CLAUDE.md 7절 원칙 2 - 구현 디테일이 아니라 물리로 검증).
    # ※ 자기참조라 검증력은 낮지만, 3-A절 ★이 경고한 부호 실수(-J_net을 +J_net으로)는
    #   이 층에서만 잡힌다.
    expected_fitness_internal = (-detail['j_net']
                                  + PM.LAMBDA_V * detail['v_violation']
                                  + PM.LAMBDA_LINE * detail['i_violation'])
    assert np.isclose(detail['fitness'], expected_fitness_internal, rtol=0.0, atol=1e-9), \
        (detail['fitness'], expected_fitness_internal)

    print('test_zero_size_particle_matches_base_exactly OK',
          detail['v_violation'], detail['i_violation'])


def test_base_has_no_voltage_violation():
    """CLAUDE.md 3-B절 1차 조치("슬랙 1.02로 기저 위반 제거")의 핵심 주장을 독립 테스트로
    명시한다: 현행 기본값(슬랙 1.02, params.SLACK_VM_PU)에서 ESS 없는 기저 조류계산의
    v_violation/i_violation이 정확히 0이어야 한다(scripts/probe_voltage.py 실측 - measurement1,
    PM.VALIDATION['v_violation_total_scaled']=0.0과 일치)."""
    evaluate.init_worker()
    base_flow = evaluate._get_base_flow()

    assert base_flow['v_violation'] < 1e-9, base_flow['v_violation']
    assert base_flow['i_violation'] < 1e-9, base_flow['i_violation']
    assert np.isclose(base_flow['v_violation'], PM.VALIDATION['v_violation_total_scaled'], atol=1e-9)
    print('test_base_has_no_voltage_violation OK', base_flow['v_violation'], base_flow['i_violation'])


def test_base_violation_at_legacy_slack():
    """구 슬랙 전압(1.0)에서는 v_violation이 0이 아니어야 한다(CLAUDE.md 1절 표,
    PM.VALIDATION_LEGACY_SLACK_1P0 참조).

    ★ 왜 필요한가: 현행 기본값(슬랙 1.02)에서는 정상 경로의 v_violation이 test_base_has_
    no_voltage_violation을 포함해 이 파일 어디서도 항상 0이다. 위반 계산 코드가 버그로
    (예: 부호가 뒤집혀 항상 0을 반환하거나, V_MIN/V_MAX를 잘못 참조해도) 항상 0을 내놓으면
    그 버그가 있어도 다른 모든 테스트가 그대로 통과한다. 구 슬랙(1.0)에서 실측값(probe_voltage.py,
    0.9169pu)과 가까운 0이 아닌 값이 실제로 나오는 것을 확인해야 그 계산 경로가 살아
    있음이 보장된다 - CLAUDE.md 7절 테스트 설계 원칙 5(회귀 방지)와 같은 취지.

    허용오차: PM.VALIDATION_LEGACY_SLACK_1P0의 0.9169는 probe_voltage.py 실측값(0.916866)의
    유효숫자 4자리 반올림이라, 반올림 오차(~3.4e-5)보다 넉넉한 여유를 두어 atol=1e-3으로 둔다
    (CLAUDE.md 7절 원칙 4 - 물리적 근거 있는 값, 임의로 조이지 않음).
    """
    net = build_net(slack_vm_pu=1.0)
    base_p = net.load['p_mw'].to_numpy().copy()
    base_q = net.load['q_mvar'].to_numpy().copy()
    base_flow = evaluate._compute_base_flow(net, base_p, base_q)

    expected = PM.VALIDATION_LEGACY_SLACK_1P0['v_violation_total_scaled']
    assert base_flow['v_violation'] > 0.0, \
        f"구 슬랙(1.0)에서도 v_violation=0 - 위반 계산 경로가 죽어있을 위험, 재확인 필요 ({base_flow['v_violation']})"
    assert np.isclose(base_flow['v_violation'], expected, atol=1e-3, rtol=0.0), \
        (base_flow['v_violation'], expected)
    print('test_base_violation_at_legacy_slack OK', base_flow['v_violation'])


def test_zero_S_alone_gives_negligible_operational_benefit_but_nonzero_cost():
    """S=0(E>0)이면 스케줄은 근사 0(정확히 0은 아님 - lower_lp의 자기방전 상쇄 바닥 때문,
    test_lp.py test_avg_zero_power_no_benefit과 동일 근거)이라 운영 편익(B_energy/B_defer)도
    근사 0이지만, Cost는 CAPEX(S,E) 공식대로 계산되므로 0이 아닐 수 있다 - "편익 0"과
    "Cost 0"을 같은 조건으로 뭉뚱그리면 안 된다는 것을 명시적으로 검증.

    ★ 편입 전에는 S<=0 OR E<=0 어느 쪽이든 lower_lp가 LP를 아예 안 부르고 0을 반환했다
    (조기 반환). 지금은 조인트 LP라 그 조기반환이 없다 - E=0 단독(S>0)은 **더 이상 이
    가정에 포함되지 않는다**(아래 test_e_zero_alone_can_have_real_q_only_benefit 참조,
    별도 테스트로 분리 - S=0과는 성격이 다른 발견이라 병합하면 오해를 부른다)."""
    won_atol = 200000.0  # ★ 아래 참조: 자기방전 상쇄 바닥(lower_lp._s_floor_for_self_discharge)이
    # 실제 SMP로 차익거래·이연에 쓰이면서 leak - E=5, margin=1.1 기준 실측 b_energy~-1.6천원,
    # b_defer~14.5만원(주로 b_defer, c_cap이 원/MW-yr 단위로 커서(76,082,200) 아주 작은 S바닥도
    # 증폭됨). 최적해 j_net(~3e6원) 대비 5% 미만이라 "무시할 만함"의 기준으로 20만원을 쓴다
    # (test_lp.py의 0.002MW atol과 같은 계보 - 여기서는 금액 단위로 환산한 것).
    x = np.array([10.0, 0.0, 5.0])
    detail = evaluate.evaluate_particle(x, return_detail=True)
    assert abs(detail['b_energy']) < won_atol, (x, detail['b_energy'])
    assert abs(detail['b_defer']) < won_atol, (x, detail['b_defer'])
    assert detail['cost'] > 0.0, (x, detail['cost'])
    print('test_zero_S_alone_gives_negligible_operational_benefit_but_nonzero_cost OK',
          detail['b_energy'], detail['b_defer'])


def test_e_zero_alone_can_have_real_q_only_benefit():
    """★ C.6-3 신규 발견(버그 아님, 검증 필요 표시): E=0(S>0)은 더 이상 "비활성 기"가
    아니다. E=0이면 SOC가 0에 고정돼 실효 P(유효전력)는 자기방전 바닥 수준(~1e-9 MW)으로
    묶이지만, **Q(무효전력)는 E와 무관하게 다각형 제약(S 기준)만 따른다** - PCS가 배터리
    없이도 순수 STATCOM처럼 무효전력만 공급할 수 있다는 뜻이고, 물리적으로 타당하다(무효
    전력은 배터리 셀이 아니라 인버터가 만든다). 실측(S=1MVA,E=0,bus=10,summer 등 실데이터):
    P_net~2e-9MW(자명), Q는 최대 ~0.6MVAr까지 사용 - 그 결과 b_energy/b_defer가 수백만원대로
    **0이 아니다**(이 스크립트 작성 시점 실측: b_energy~983만원, b_defer~164만원, cost~301만원,
    j_net~846만원 - 부록C.7의 "R/X 高 배전망이라 Q 효과가 제한적"이라는 정성적 기대와
    자릿수가 안 맞아 보이므로, PSO가 실제로 이 방향(대형 S·E=0 순수 무효보상)으로 수렴하는지
    dev 실행에서 반드시 확인할 것 - 확인 전까지는 부호·크기 모두 참고용."""
    x = np.array([10.0, 1.0, 0.0])
    detail = evaluate.evaluate_particle(x, return_detail=True)
    assert detail['diverged'] is False
    assert detail['cost'] > 0.0, detail['cost']
    # ★ P(유효전력)가 정확히 0에 가까울 것이라 기대했으나(E=0->SOC=0 고정) 실측은
    # 그렇지 않았다(최대 ~0.05MW, S=1MW의 5%) - 다각형 제약의 "Q 최대치" 변은 P가 어느
    # 한 점이 아니라 구간([-0.26S,+0.26S] 부근)에서 폭넓게 degenerate하기 때문(그 구간
    # 안에서는 Q=S*cos(15°)로 동일하게 최대라 목적함수가 P값 자체엔 무차별 - solve_peak
    # 관련 기존 degenerate 테스트들과 같은 성격의 현상). 그래서 강한 상한(1e-3) 대신
    # 물리적으로 항상 성립해야 하는 느슨한 상한(|P|<=S)만 확인한다.
    p_ess_max = max(float(np.max(np.abs(detail['p_ess_total'][s]))) for s in PM.ALL_DAYS)
    assert p_ess_max <= 1.0 + 1e-2, p_ess_max
    print('test_e_zero_alone_can_have_real_q_only_benefit OK (참고용 수치) '
          f"b_energy={detail['b_energy']:.0f} b_defer={detail['b_defer']:.0f} "
          f"cost={detail['cost']:.0f} j_net={detail['j_net']:.0f} p_ess_max={p_ess_max:.2e}")


# ============================================================
# 3 & 4. 실데이터 수지 검증 + b_arb+b_loss ~= b_energy 검산
# ============================================================

def test_realistic_particle_balance_and_decomposition():
    """중간 크기 입자(S=1.0MVA, E=4.0MWh, bus=17)로 실제 조류계산을 돌린다.
    evaluate_particle 내부에서 assert_slack_balance(atol=1e-7, rtol=0, ALL_DAYS 전부)가
    이미 통과해야 예외 없이 반환되므로, 예외 없이 detail이 나오는 것 자체가 3번 항목의 증거다.
    b_arb+b_loss ~= b_energy는 detail['decomposition_ok']로 별도 확인."""
    x = np.array([17.0, 1.0, 4.0])
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
        x = np.array([10.0, 1.0, 4.0])
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
        np.array([1.0, 0.0, 0.0]),
        np.array([32.0, 2.4, 10.2]),
        np.array([17.0, 1.2, 5.0]),
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
    x1 = np.array([17.0, 1.0, 4.0])
    detail1 = evaluate.evaluate_particle(x1, return_detail=True)
    assert detail1['b'].shape == (1,)
    assert detail1['unit_p']['summer'].shape == (1, PM.TIME_STEPS)

    # n=2, 서로 다른 버스
    x2 = np.array([17.0, 1.0, 4.0, 25.0, 0.5, 2.0])
    detail2 = evaluate.evaluate_particle(x2, return_detail=True)
    assert detail2['diverged'] is False
    assert detail2['b'].shape == (2,)
    assert detail2['unit_p']['summer'].shape == (2, PM.TIME_STEPS)
    assert list(detail2['b']) == [17, 25]

    # n=2, 동일 버스 중복배치 ((a)방식: sgen 2개 별도 생성, 병합 안 함)
    x_dup = np.array([17.0, 1.0, 4.0, 17.0, 0.8, 3.0])
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
    x = np.array([17.0, 1.0, 4.0])
    evaluate.evaluate_particle(x)  # JIT/캐시 워밍업 1회

    t0 = time.perf_counter()
    evaluate.evaluate_particle(x)
    elapsed = time.perf_counter() - t0

    print(f'test_and_report_single_evaluation_timing: {elapsed:.3f}초/평가 (목표 참고치 1.178초)')
    assert elapsed < 10.0, f'평가 1회가 비정상적으로 느림: {elapsed:.3f}초'


if __name__ == '__main__':
    test_base_reproduces_validation()
    test_zero_size_particle_matches_base_exactly()
    test_base_has_no_voltage_violation()
    test_base_violation_at_legacy_slack()
    test_zero_S_alone_gives_negligible_operational_benefit_but_nonzero_cost()
    test_e_zero_alone_can_have_real_q_only_benefit()
    test_realistic_particle_balance_and_decomposition()
    test_divergence_returns_penalty_not_exception()
    test_fitness_always_finite_scalar()
    test_multi_unit_n1_and_n2_structural()
    test_and_report_single_evaluation_timing()
    print('all evaluate tests passed')
