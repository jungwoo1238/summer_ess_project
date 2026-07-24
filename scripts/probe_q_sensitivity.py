"""LP가 고른 Q 수준의 임의성이 역률 스윕 결론을 바꾸는지 진단한다 (확인 전용, 본 파이프라인
미포함 - scripts/probe_q_value.py와 같은 성격, 그 결과를 이어받는 후속 스크립트).

배경: probe_q_value.py 실측(2026-07-23) - 동일 (b,S,E)에서 Q on/off 차이(A.b_loss-B.b_loss)가
PF=0.850241(원본)에서는 전부 양수(+17만~+416만원), PF=0.95(보상)에서는 전부 음수
(-16만~-137만원)로 부호 자체가 뒤집혔다. 6차 세션이 보고한 "Q 순 기여 -4.46e6"은 서로 다른
조건을 섞은 오계산이며, 그 값에 기대 짰던 (A)/(B)/(C) 선택지는 이미 폐기됐다.

새 문제: lower_lp.py의 LP 목적함수에는 손실 항이 없다(부록C.4-(1) - LinDistFlow는 손실을
무시한 근사식이고, 손실은 조류계산 사후측정 대상이다). Q는 오직 mu_volt 전압유도항을
통해서만 목적함수에 등장한다(volt_penalty = mu_volt * sum(pos(v-V_SQ_MAX)+pos(V_SQ_MIN-v)),
lower_lp._build_problem 참조). 즉 LP가 고르는 Q의 **방향**은 물리적으로 타당하다(전압을
끌어올리는 쪽 Q = 손실을 줄이는 쪽 Q, x*Q 항의 부호가 그렇게 만든다 - 부록C.2 LinDistFlow
채택 근거). 그러나 그 **수준**(크기)은 mu_volt라는 잠정 상수 하나가 정한다(params.py:
"MU_VOLT = LAMBDA_V / TOTAL_WEEKDAYS_PER_YEAR ... 잠정값, 결과 보고 조정 대상"). volt_penalty가
비활성(LinDistFlow 추정 v가 이미 0.95~1.05 안)인 시간대에는 Q가 목적함수에 전혀 나타나지
않아 LP 입장에서 그 시각 Q는 사실상 degenerate하다(CLAUDE.md 7절 "테스트 설계 원칙 1" -
solve_peak의 평탄부 충전 분배와 같은 성격의 문제가 Q 축에서도 생길 수 있다).

임계 역률(b_loss 부호가 뒤집히는 지점)의 위치가 이 임의성에 얼마나 민감한지 모르는 상태로
역률 스윕(예: 0.85~0.95를 촘촘히)을 돌리면, 그 결과가 "역률의 물리적 효과"인지 "mu_volt가
그날 우연히 고른 Q 수준의 잡음"인지 구분할 수 없다. 이 스크립트는 스윕에 앞서 그 민감도
자체를 먼저 잰다.

방법: probe_q_value.py와 동일한 3개 통제점(P1/P2/P3) x 2개 역률(0.850241 원본 / 0.95 보상,
P 고정 Q만 스케일 - probe_q_value.py의 정정본 규약을 그대로 따른다) 조건마다:
  1. LP를 정상 호출(force_q_zero=False, 시변 Q 자유)해 기준 스케줄 unit_p/unit_q를 얻는다
     - 이것이 "LP가 실제로 고른" Q이고, k=1.0 배율에 해당한다.
  2. unit_q에 배율 k(∈{0,0.25,0.5,0.75,1,1.25,1.5,2})를 곱해 **교체**한다. unit_p는 절대
     건드리지 않는다 - 이 실험이 재려는 것은 "LP가 이미 확정한 P 스케줄에 대해 Q **수준만**
     사후에 흔들었을 때 j_net이 얼마나 민감한가"이지, "각 k에서 LP가 P까지 다시 최적화했을
     때"가 아니다.
  3. k*Q가 PCS 원 제약 sqrt(P^2+(k*Q)^2)<=S를 넘으면(k>1에서 발생 가능 - lower_lp의 다각형
     근사가 이미 원 제약보다 좁게 걸려 있으므로 k=1 스케줄 자체도 원 제약에는 여유가 있지만,
     k*Q로 키우면 넘을 수 있다) 그 시각의 Q만 원 제약 경계로 축소한다(P는 그대로 둔다 -
     원래 solve_avg/solve_peak가 P_ch<=S,P_dis<=S를 개별 제약으로 항상 걸어 두므로 |P|<=S는
     항상 성립하고, 따라서 sqrt(S^2-P^2)>=0이 항상 존재해 클리핑이 항상 가능하다).
  4. loss_pcs(PCS 무효전력 변환손실, benefits.loss_pcs)를 교체된 Q로 재계산하고, evaluate.py의
     주입 로직과 동일하게 sgen에 p_mw = unit_p - loss_pcs, q_mvar = (교체된)unit_q를 넣어
     24h x 5시나리오 조류계산부터 다시 수행한다(evaluate.evaluate_particle을 그대로 쓸 수
     없다 - Q 교체 지점이 LP와 조류계산 사이라 함수 내부에 있다. 이 스크립트가 evaluate.py의
     후반부(주입~조류계산~편익, evaluate_particle 라인 248 이후)를 그대로 재현한 이유가 이것.
     evaluate.py 파일 자체는 건드리지 않는다 - 전역 상태(_NET/_BASE_P/_BASE_Q/_BASE_FLOW)와
     private 헬퍼(_run_pf_with_retry/_ensure_sgens)만 그대로 재사용한다, probe_q_value.py의
     _prepare_condition과 동일한 패턴).

★ 이 스윕이 재지 않는 것 (한계, 상단에 명시): k!=1인 각 스케줄은 "LP가 Q=k*Q_LP일 것을
미리 알고 P까지 다시 최적화한 해"가 아니라 "LP가 Q 자유로 고른 P를 고정한 채 Q 수준만 사후에
흔든" 반사실적(counterfactual) 스케줄이다. 이 스윕의 목적은 그 반사실이 실제 최적해와
얼마나 다른지 재는 것 자체이므로 이는 의도한 설계다 - 다만 곡선이 가파르다고 "k=1이 최적이
아니다"라고 바로 결론 내리면 안 된다(P가 고정된 반사실 위에서의 국소 민감도이지, P까지
자유로운 전역 최적 곡선이 아니다).

실행: `python scripts/probe_q_sensitivity.py`  (★ 이 스크립트는 작성만 하고 실행하지 않는다 -
실행은 사용자가 터미널에서 직접 한다)

# ------------------------------------------------------------------
# 무엇을 확정했는가 (실행 후 채울 것 - scripts/ 규약, CLAUDE.md 부록A 참조)
#   실행 일시:
#   머신 사양:
#   결론:
# ------------------------------------------------------------------
"""

import os
import sys
import csv
import socket
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import params as PM
import evaluate
import benefits

# probe_q_value.py의 통제점·역률 규약·헬퍼를 그대로 재사용한다 - 두 스크립트의 통제 조건이
# 갈리면 이 스크립트의 검산 3/4(아래) 자체가 무의미해지므로, 복사해서 살짝 바꾸는 대신
# import로 원본을 그대로 물려받는다.
from probe_q_value import (
    POINTS,
    TARGET_PF,
    BASE_PF_EXPECTED,
    PF_TOL,
    LOAD_SUM_ATOL_MW,
    _build_net_with_pf,
    _prepare_condition,
    _restore_evaluate_state,
    _evaluate_with_force_q,
    _avg_weighted_mwh,
    section,
    _check_env,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'results')

K_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

# PCS 원 제약 클리핑 판정 허용오차(MVA) - cvxpy 솔버 잔차(통상 1e-8~1e-6 수준) 때문에 k=1.0
# 스케줄(교체 없음, 정의상 항상 원 제약 안쪽)이 부동소수점 잡음만으로 "클리핑 발생"으로
# 오판되지 않도록 두는 여유. 너무 크게 잡으면 k>1에서의 진짜 위반을 놓치므로 물리적 스케일
# (S는 최소 0.05 MVA 이상)보다 한참 작은 값으로 둔다.
PCS_CLIP_TOL_MVA = 1e-6

DECOMP_ATOL_WON = 1e-6           # 검산1 (b_arb+b_loss~=b_energy) - benefits 기본값과 동일
LOSS_PCS_ZERO_ATOL_MWH = 1e-9    # 검산2 (k=0.0 -> loss_pcs 정확히 0)
CROSS_CHECK_RTOL = 1e-9          # 검산4(k=1.0, bit 수준 재사용이라 항상 성립) 기준.
                                  # CLAUDE.md 7절 "테스트 설계 원칙4": 기대값!=0인 금액 비교
                                  # 규약(rtol=1e-9, atol=0)을 그대로 따른다.

# 검산3(k=0.0 vs force_q_zero=True) 전용: ★ 실측(2026-07-24, P1/PF=0.850241)으로 이 둘은
# rtol=1e-9로 일치하지 않는다는 것이 확인됐다 - delta=-58,171.77원. 원인은 잡음이 아니라
# lower_lp.py의 다각형 근사(POLY_N=12)가 force_q_zero=False일 때는 Q값과 무관하게
# |P_net|<=S*cos(pi/12)(~0.966S)로 개별한계 S보다 좁게 묶는 반면(θ=0 꼭짓점 제약),
# force_q_zero=True 경로는 다각형을 아예 안 걸고 개별한계 S만 쓰는 구조적 차이다
# (lower_lp.py "PCS 원 제약" 절). 즉 "k=0.0(자유Q LP의 P를 그대로 두고 Q만 사후 0)"과
# "force_q_zero=True(애초에 다른 P 제약의 LP)"는 처음부터 같은 문제가 아니다 - 아래에서
# 이 비교를 hard assert가 아니라 진단 리포트로 다룬다(전체 스윕을 중단시키지 않기 위함).
NOISE_BOUND_WON = 7.0             # CLAUDE.md 7절 "수치 잡음의 성격" 실측 상한(<6.91원/년)

CSV_FIELDS = [
    'point_id', 'b', 'S', 'E', 'power_factor', 'k',
    'j_net', 'b_energy', 'b_defer', 'b_arb', 'b_loss', 'cost',
    'loss_line_total_mwh', 'loss_pcs_total_mwh',
    'q_total_mvarh', 'n_clipped_hours',
    'v_violation', 'i_violation',
]


# ============================================================
# Q 배율 적용 + PCS 원 제약 클리핑
# ============================================================

def _apply_q_multiplier(unit_p, unit_q_base, k, S):
    """unit_q_base에 배율 k를 곱하고, sqrt(P^2+(k*Q)^2)<=S를 넘는 시각의 Q만 원 제약
    경계로 축소한다(P는 절대 바꾸지 않는다 - 모듈 docstring 3번 참조).

    unit_p/unit_q_base: dict[PM.ALL_DAYS] -> (n,T) ndarray. S: (n,) ndarray, MVA.
    반환: (unit_q_k: dict[PM.ALL_DAYS]->(n,T), n_clipped_hours: int)
      n_clipped_hours는 (시나리오,기,시각) 조합 중 실제로 축소가 일어난 건수의 총합이다.

    ★ P가 이미 원 제약 안쪽이라는 전제(항상 성립): solve_avg/solve_peak는 force_q_zero
    여부와 무관하게 P_ch<=S, P_dis<=S를 개별 제약으로 항상 건다(lower_lp._build_problem
    "개별 PCS 출력한계" 참조) - 따라서 |P_net|<=S가 항상 보장되고, sqrt(max(S^2-P^2,0))이
    허수가 될 일이 없다. k=1.0(교체 없음)은 애초에 다각형 근사(원보다 좁음, cos(pi/12))
    안쪽에서 나온 해라 이 함수를 거쳐도 클리핑이 발생하지 않는다(k=1.0은 실제로는 이
    함수를 호출하지 않는다 - _run_condition의 k==1.0 분기 참조, 그래도 만에 하나
    호출되더라도 안전하다).
    """
    S_col = np.asarray(S, dtype=float)[:, None]  # (n,1)
    unit_q_k = {}
    n_clipped = 0
    for s in PM.ALL_DAYS:
        P = unit_p[s]
        Q = unit_q_base[s] * k
        apparent = np.sqrt(P ** 2 + Q ** 2)
        over = apparent > (S_col + PCS_CLIP_TOL_MVA)
        n_clipped += int(np.sum(over))
        if np.any(over):
            q_max = np.sqrt(np.maximum(S_col ** 2 - P ** 2, 0.0))
            sign = np.sign(Q)
            sign = np.where(sign == 0.0, 1.0, sign)  # Q=0인데 클리핑되는 경로는 정상적으론
            # 도달 불가(위 docstring 전제) - 방어적 처리일 뿐 실질적으로 실행되지 않는다.
            Q = np.where(over, sign * q_max, Q)
        unit_q_k[s] = Q
    return unit_q_k, n_clipped


# ============================================================
# evaluate.evaluate_particle 후반부(주입~조류계산~편익) 재현 - LP는 다시 풀지 않는다
# ============================================================

def _finalize_result(j_net_val, b_energy_val, b_defer_val, b_arb_val, b_loss_val, cost_val,
                      loss_line_total_mwh, loss_pcs_total_mwh, q_total_mvarh,
                      v_violation, i_violation):
    """검산1(b_arb+b_loss~=b_energy)을 한 곳에서만 검증하고 결과 dict를 만든다 - 아래
    _reinject_and_evaluate(k!=1.0 경로)와 _result_from_free_detail(k=1.0 경로) 둘 다
    이 함수를 거치게 해 검산1이 두 경로에서 갈라지지 않게 한다."""
    decomp_ok = benefits.check_b_energy_decomposition(
        b_arb_val, b_loss_val, b_energy_val, atol=DECOMP_ATOL_WON
    )
    assert decomp_ok, (
        f'b_arb({b_arb_val:.6f}) + b_loss({b_loss_val:.6f}) != b_energy({b_energy_val:.6f}) '
        f'(atol={DECOMP_ATOL_WON})'
    )
    return dict(
        diverged=False,
        j_net=j_net_val, b_energy=b_energy_val, b_defer=b_defer_val,
        b_arb=b_arb_val, b_loss=b_loss_val, cost=cost_val,
        loss_line_total_mwh=loss_line_total_mwh, loss_pcs_total_mwh=loss_pcs_total_mwh,
        q_total_mvarh=q_total_mvarh,
        v_violation=v_violation, i_violation=i_violation,
    )


def _reinject_and_evaluate(b, S, E, unit_p, unit_q):
    """evaluate.evaluate_particle(evaluate.py 라인 229-365)의 주입 이후 절반을 그대로
    재현한다. LP(_solve_unit_schedules)는 호출하지 않는다 - unit_p/unit_q가 이미 확정된
    스케줄로 주어진다(이 스크립트의 조작 지점). evaluate.py 파일은 수정하지 않고, 모듈
    전역(_NET/_BASE_P/_BASE_Q/_BASE_FLOW - probe_q_value._prepare_condition이 pf 조건에
    맞게 이미 세팅해 둔 것)과 private 헬퍼(_run_pf_with_retry/_ensure_sgens)만 재사용한다.
    """
    net = evaluate._NET
    base_p, base_q = evaluate._BASE_P, evaluate._BASE_Q
    base_flow = evaluate._get_base_flow()
    n = len(b)

    unit_loss_pcs = benefits.loss_pcs(unit_p, unit_q)

    evaluate._ensure_sgens(net, n)
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
                net.sgen.at[i, 'p_mw'] = float(unit_p[s][i, t] - unit_loss_pcs[s][i, t])
                net.sgen.at[i, 'q_mvar'] = float(unit_q[s][i, t])

            log_ctx = dict(b=[int(v) for v in b], S=[float(v) for v in S],
                            E=[float(v) for v in E], scenario=s, t=t)
            ok = evaluate._run_pf_with_retry(net, log_context=log_ctx)
            if not ok:
                diverged = True
                diverge_info = log_ctx
                break

            p_slack_ess[s][t] = net.res_ext_grid.p_mw.sum()
            loss_ess[s][t] = net.res_line.pl_mw.sum() + float(unit_loss_pcs[s][:, t].sum())

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
        return dict(diverged=True, diverge_info=diverge_info)

    load_sum = {s: base_p.sum() * np.asarray(PM.LOAD[s]) for s in PM.ALL_DAYS}
    p_ess_total = {s: unit_p[s].sum(axis=0) for s in PM.ALL_DAYS}
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
    s_total = float(np.sum(S))
    e_total = float(np.sum(E))
    j_net_val = benefits.j_net(b_energy_val, b_defer_val, s_total, e_total)
    cost_val = benefits.total_cost(s_total, e_total)

    p_ch_agg = {s: np.maximum(-unit_p[s], 0.0).sum(axis=0) for s in PM.AVG_DAYS}
    p_dis_agg = {s: np.maximum(unit_p[s], 0.0).sum(axis=0) for s in PM.AVG_DAYS}
    b_arb_val = benefits.b_arb(p_ch_agg, p_dis_agg, smp_mwh, PM.N_WEEKDAYS)
    b_loss_val = benefits.b_loss(
        {s: base_flow['loss'][s] for s in PM.AVG_DAYS},
        {s: loss_ess[s] for s in PM.AVG_DAYS},
        smp_mwh, PM.N_WEEKDAYS,
    )

    loss_line_by_scen = {s: loss_ess[s] - unit_loss_pcs[s].sum(axis=0) for s in PM.AVG_DAYS}
    loss_pcs_by_scen = {s: unit_loss_pcs[s].sum(axis=0) for s in PM.AVG_DAYS}
    loss_line_total_mwh = _avg_weighted_mwh(loss_line_by_scen)
    loss_pcs_total_mwh = _avg_weighted_mwh(loss_pcs_by_scen)

    # q_total_mvarh: 실제 공급된(k 반영, 클리핑 후) |Q|의 절대량 합. 편익이 아니라 물리적
    # 운영강도 지표라 N_WD 가중을 하지 않는다 - ALL_DAYS 5개 대표일 각각을 1회로 세어
    # 그대로 합산한다(Mvar*h, dt=1h이므로 시간축 합이 곧 Mvarh).
    q_total_mvarh = float(sum(np.sum(np.abs(unit_q[s])) for s in PM.ALL_DAYS) * PM.DT_HOURS)

    return _finalize_result(
        j_net_val, b_energy_val, b_defer_val, b_arb_val, b_loss_val, cost_val,
        loss_line_total_mwh, loss_pcs_total_mwh, q_total_mvarh, v_viol, i_viol,
    )


def _result_from_free_detail(detail_free):
    """k=1.0 전용: 조류계산을 다시 하지 않고 detail_free(force_q_zero=False로 이미 계산된
    기준 스케줄의 평가 결과)를 그대로 재사용한다.

    ★ 재조류계산을 하지 않는 이유(의도적, 우회가 아니다): k=1.0은 Q를 전혀 바꾸지 않으므로
    재조류계산을 해도 수학적으로 같은 지점을 다시 푸는 것뿐인데, CLAUDE.md 7절 "수치 잡음의
    성격"이 규명했듯 이 net은 warm start(init='results')로 이어지므로 *어떤 계산이 먼저
    실행됐는가*에 따라 1e-8 MW 수준의 잡음이 남는다(원인은 sgen 유무가 아니라 웜스타트
    이력 - 무해하지만(<10원/년) 완전히 0은 아니다). k=0.0 스윕이나 case_zero(force_q_zero
    참조 계산)를 먼저 실행한 뒤 k=1.0을 "다시" 계산하면 이 잡음이 끼어들어, 정작 검산4가
    검증하려는 "k=1.0 == probe_q_value 케이스 A/C"를 잡음 이하 정밀도로만 통과시킬 수 있다.
    detail_free를 그대로 재사용하면 그 잡음이 원천적으로 낄 자리가 없다 - detail_free 자체가
    이미 probe_q_value._run_pf_condition의 case A/C(먼저 실행되는 free 케이스)와 정확히
    동일한 순서(build+prepare 직후 첫 evaluate 호출)로 계산된 것이므로, 두 스크립트를 각각
    새 프로세스에서 실행해도(MKL_THREADING_LAYER=SEQUENTIAL이라 실행순서가 결정론적)
    bit 수준으로 같은 값이 나와야 한다."""
    unit_p = detail_free['unit_p']
    unit_q = detail_free['unit_q']
    unit_loss_pcs = benefits.loss_pcs(unit_p, unit_q)

    loss_line_by_scen = {
        s: detail_free['loss_ess'][s] - unit_loss_pcs[s].sum(axis=0) for s in PM.AVG_DAYS
    }
    loss_pcs_by_scen = {s: unit_loss_pcs[s].sum(axis=0) for s in PM.AVG_DAYS}
    q_total_mvarh = float(sum(np.sum(np.abs(unit_q[s])) for s in PM.ALL_DAYS) * PM.DT_HOURS)

    return _finalize_result(
        detail_free['j_net'], detail_free['b_energy'], detail_free['b_defer'],
        detail_free['b_arb'], detail_free['b_loss'], detail_free['cost'],
        _avg_weighted_mwh(loss_line_by_scen), _avg_weighted_mwh(loss_pcs_by_scen),
        q_total_mvarh, detail_free['v_violation'], detail_free['i_violation'],
    )


def _polygon_bind_diagnostics(unit_p_base, S_arr):
    """검산3 진단용: 자유Q LP(force_q_zero=False)가 다각형 θ=0 꼭짓점 한계
    S*cos(pi/12)에 실제로 얼마나 붙어(bind) 있었는지 계산한다. 이 값이 실질적으로
    0이 아니면(=한계 근접 시각이 존재하면) 검산3의 불일치가 잡음이 아니라 다각형
    근사의 구조적 효과라는 직접 증거가 된다."""
    poly_cap = S_arr * float(np.cos(np.pi / PM.POLY_N))  # (n,) MVA, force_q_zero=False의
    # theta=0/pi 꼭짓점이 Q값과 무관하게 항상 부과하는 |P_net| 상한.
    margin_mva = S_arr - poly_cap                          # (n,) 개별한계 대비 다각형이 좁힌 폭
    near_cap_tol = 1e-4  # MVA - 솔버 수치오차 여유
    near_cap_hours = 0
    max_abs_p = 0.0
    for s in PM.ALL_DAYS:
        P = np.abs(unit_p_base[s])  # (n,T)
        near_cap_hours += int(np.sum(P >= (poly_cap[:, None] - near_cap_tol)))
        max_abs_p = max(max_abs_p, float(np.max(P)))
    return dict(poly_cap=poly_cap, margin_mva=margin_mva,
                near_cap_hours=near_cap_hours, max_abs_p=max_abs_p)


# ============================================================
# 행 구성
# ============================================================

def _blank_row(point, pf_label, k):
    return dict(
        point_id=point['point_id'], b=point['b'], S=point['S'], E=point['E'],
        power_factor=pf_label, k=k,
        j_net='', b_energy='', b_defer='', b_arb='', b_loss='', cost='',
        loss_line_total_mwh='', loss_pcs_total_mwh='',
        q_total_mvarh='', n_clipped_hours='',
        v_violation='', i_violation='',
    )


def _row_from_result(point, pf_label, k, result, n_clipped_hours):
    return dict(
        point_id=point['point_id'], b=point['b'], S=point['S'], E=point['E'],
        power_factor=pf_label, k=k,
        j_net=result['j_net'], b_energy=result['b_energy'], b_defer=result['b_defer'],
        b_arb=result['b_arb'], b_loss=result['b_loss'], cost=result['cost'],
        loss_line_total_mwh=result['loss_line_total_mwh'],
        loss_pcs_total_mwh=result['loss_pcs_total_mwh'],
        q_total_mvarh=result['q_total_mvarh'], n_clipped_hours=n_clipped_hours,
        v_violation=result['v_violation'], i_violation=result['i_violation'],
    )


# ============================================================
# 통제점 x 역률 조건 1개 처리
# ============================================================

def _run_condition(point, pf_target, pf_label, reference_base_p):
    """probe_q_value._run_pf_condition과 동일한 순서로 net을 준비한다(build+prepare ->
    free 케이스 -> zero 케이스). free 케이스(detail_free)가 k=1.0 기준이자 이 함수가 도는
    전체 k 스윕의 P 원본이고, zero 케이스(detail_zero_ref)는 검산3의 대조군이다 - 두
    스크립트가 정확히 같은 순서를 밟아야 검산3/4가 "통제 조건이 같다"는 전제 위에서 성립한다
    (모듈 docstring 및 _result_from_free_detail 참조).

    반환: (rows: list[dict] (K_VALUES 순서, 8행), base_p: 이 조건에서 쓰인 부하 유효전력
    배열 - 다음 pf 조건 호출 시 reference_base_p로 넘겨 P 고정 통제(검산4 전제)를 확인하는 데 쓴다).
    """
    net, q_scale, p_total, q_total_before = _build_net_with_pf(pf_target)
    base_p, base_q = _prepare_condition(net)

    s_total_mva = float(np.hypot(base_p.sum(), base_q.sum()))
    pf_actual = float(base_p.sum() / s_total_mva)
    assert abs(pf_actual - pf_label) < PF_TOL, (
        f"{point['point_id']}/{pf_label}: power_factor_actual={pf_actual:.6f}가 "
        f"기대값 {pf_label}±{PF_TOL}를 벗어남"
    )
    if reference_base_p is not None:
        assert np.allclose(base_p, reference_base_p, atol=LOAD_SUM_ATOL_MW, rtol=0.0), (
            f"{point['point_id']}: 역률 조정 후 P(base_p)가 원본과 달라짐 - P 고정 통제가 깨짐"
        )

    x = np.array([point['b'], point['S'], point['E']], dtype=float)

    # ---- 기준(free, k=1.0) ----
    detail_free = _evaluate_with_force_q(x, False)
    if detail_free.get('diverged'):
        print(f"  {point['point_id']}/{pf_label}: 기준(Q자유) 평가 발산 -> 이 조건 전체 건너뜀 "
              f"({detail_free.get('diverge_info')})", flush=True)
        rows = [_blank_row(point, pf_label, k) for k in K_VALUES]
        return rows, base_p

    # ---- 대조군(force_q_zero, 검산3용) - free 바로 다음에 실행해 probe_q_value의 B/D와
    #      동일한 웜스타트 순서를 재현한다 ----
    detail_zero_ref = _evaluate_with_force_q(x, True)

    b_arr, S_arr, E_arr = detail_free['b'], detail_free['S'], detail_free['E']
    unit_p_base = detail_free['unit_p']
    unit_q_base = detail_free['unit_q']

    rows = []
    for k in K_VALUES:
        if k == 1.0:
            # 재조류계산 없이 detail_free를 그대로 재사용 (검산4가 bit 수준으로 항상
            # 성립하도록 하는 설계 - _result_from_free_detail docstring 참조)
            result = _result_from_free_detail(detail_free)
            n_clipped = 0
        else:
            unit_q_k, n_clipped = _apply_q_multiplier(unit_p_base, unit_q_base, k, S_arr)
            result = _reinject_and_evaluate(b_arr, S_arr, E_arr, unit_p_base, unit_q_k)

        if result.get('diverged'):
            print(f"  {point['point_id']}/{pf_label}/k={k}: 발산 -> 건너뜀 "
                  f"({result.get('diverge_info')})", flush=True)
            rows.append(_blank_row(point, pf_label, k))
            continue

        if k == 0.0:
            # ---- 검산2: k=0.0이면 Q=0이므로 loss_pcs는 정확히 0이어야 함 ----
            assert abs(result['loss_pcs_total_mwh']) < LOSS_PCS_ZERO_ATOL_MWH, (
                f"{point['point_id']}/{pf_label}: k=0.0인데 "
                f"loss_pcs_total_mwh={result['loss_pcs_total_mwh']:.3e} "
                f"(Q=0이면 삼각부등식상 정확히 0이어야 함)"
            )
            # ---- 검산3(진단 리포트, hard assert 아님): k=0.0(P는 Q자유 LP에서, Q만
            #      사후 0) vs force_q_zero=True LP ----
            # ★ 2026-07-24 실측(P1/PF=0.850241)으로 이 둘이 rtol=1e-9로 일치하지 않음이
            # 이미 확인됐다(delta=-58,171.77원, 웜스타트 잡음상한 ~7원/년의 8천 배).
            # 원인은 잡음이 아니라 lower_lp.py의 다각형 근사가 force_q_zero=False일 때
            # Q값과 무관하게 |P_net|<=S*cos(pi/12)(~0.966S)로 개별한계 S보다 좁게 묶는
            # 반면(θ=0 꼭짓점), force_q_zero=True 경로는 다각형을 아예 안 걸고 개별한계
            # S만 쓰는 구조적 차이다(lower_lp.py "PCS 원 제약" 절) - 즉 두 LP가 애초에
            # 같은 문제가 아니므로 hard assert로 스윕 전체를 중단시키지 않는다. 대신
            # delta 크기와 다각형 근접 시각 수(near_cap_hours)를 함께 보고해, "잡음"과
            # "다각형 병목"을 구분할 근거를 남긴다.
            if detail_zero_ref.get('diverged'):
                print(f"  {point['point_id']}/{pf_label}: 대조군(Q=0 강제) 발산 -> "
                      f"검산3 건너뜀", flush=True)
            else:
                ref_j_net = detail_zero_ref['j_net']
                delta = result['j_net'] - ref_j_net
                if np.isclose(result['j_net'], ref_j_net, rtol=CROSS_CHECK_RTOL, atol=0.0):
                    print(f"  검산3 일치: {point['point_id']}/{pf_label} k=0.0 j_net == "
                          f"force_q_zero=True 기준 (delta={delta:+.6f}원)", flush=True)
                else:
                    diag = _polygon_bind_diagnostics(unit_p_base, S_arr)
                    cause = ('다각형 병목(구조적) - 아래 near_cap_hours>0이 직접 증거'
                              if abs(delta) > 100 * NOISE_BOUND_WON
                              else '판정 애매 - 잡음 상한의 100배 미만이라 다각형 효과인지 '
                                   '불확실')
                    print(
                        f"  ⚠ 검산3 불일치 (버그 아님, 진단 참조): {point['point_id']}/{pf_label} "
                        f"k=0.0 j_net({result['j_net']:.2f}) != force_q_zero=True 기준"
                        f"({ref_j_net:.2f}), delta={delta:+.2f}원 "
                        f"(|delta|/잡음상한={abs(delta) / NOISE_BOUND_WON:.1f}배)\n"
                        f"    다각형한계 S*cos(pi/12)={diag['poly_cap'][0]:.6f} MVA vs "
                        f"개별한계 S={S_arr[0]:.6f} MVA (차 {diag['margin_mva'][0]:.6f} MVA), "
                        f"자유Q LP 최대|P_net|={diag['max_abs_p']:.6f} MVA, "
                        f"다각형한계 근접 시각 수={diag['near_cap_hours']} -> {cause}",
                        flush=True,
                    )

        rows.append(_row_from_result(point, pf_label, k, result, n_clipped))

    return rows, base_p


# ============================================================
# stdout 요약
# ============================================================

def _print_curves(all_rows):
    section('k-곡선 (통제점 x 역률별 j_net vs k, 판정 없이 수치만 제시)')
    by_cond = {}
    for r in all_rows:
        key = (r['point_id'], r['power_factor'])
        by_cond.setdefault(key, {})[r['k']] = r

    header = f"{'k':>6s}  {'j_net':>14s}  {'b_loss':>12s}  {'n_clipped':>10s}  {'q_total_mvarh':>13s}"
    for (point_id, pf), by_k in by_cond.items():
        print(f"\n[{point_id} / PF={pf:.6f}]", flush=True)
        print(header, flush=True)
        j_nets = {}
        for k in K_VALUES:
            row = by_k.get(k)
            if row is None or row['j_net'] == '':
                print(f"{k:>6.2f}  {'발산/스킵':>14s}", flush=True)
                continue
            j_nets[k] = row['j_net']
            print(f"{k:>6.2f}  {row['j_net']:>14.4e}  {row['b_loss']:>12.4e}  "
                  f"{row['n_clipped_hours']:>10d}  {row['q_total_mvarh']:>13.6f}", flush=True)

        if j_nets:
            spread = max(j_nets.values()) - min(j_nets.values())
            print(f"  최대-최소 폭(spread) = {spread:+.4e}원", flush=True)
        if 0.75 in j_nets and 1.25 in j_nets:
            slope = (j_nets[1.25] - j_nets[0.75]) / (1.25 - 0.75)
            print(f"  k=1.0 국소 기울기 (k=0.75~1.25 차분) = {slope:+.4e}원/(k단위)", flush=True)
        if 1.0 in j_nets and j_nets:
            k_best = max(j_nets, key=j_nets.get)
            print(f"  곡선 내 최댓값 지점: k={k_best:.2f} (j_net={j_nets[k_best]:.4e}원), "
                  f"k=1.0의 j_net={j_nets[1.0]:.4e}원, 차이={j_nets[k_best]-j_nets[1.0]:+.4e}원",
                  flush=True)


def _print_interpretation():
    section('해석 지침 (자동판정 없음 - 수치를 보고 사람이 판단할 것)')
    print(
        "- 곡선이 평탄하면(spread가 j_net 규모의 몇 % 이내) LP가 고른 Q 수준의 임의성이\n"
        "  결론(j_net)을 바꾸지 않는다는 뜻이므로, 역률 스윕을 그대로 진행해도 된다.\n"
        "- 가파르면(spread가 j_net 규모 대비 무시 못 할 수준) 임계 역률의 위치가 mu_volt\n"
        "  잠정값에 의존할 수 있다는 뜻이므로, 역률 스윕보다 LP 목적함수에 PCS 손실 비용을\n"
        "  직접 넣는 수정(부록C 확장)이 먼저다.\n"
        "- k=1.0이 곡선의 최댓값 근처인지도 볼 것. 크게 벗어나 있으면 LP가 고른 Q 수준\n"
        "  자체가 (이 반사실적 P-고정 기준에서도) 국소최적과 멀다는 직접 증거다 - 단, 모듈\n"
        "  docstring의 한계(P가 고정된 반사실 스윕이라 전역 최적 곡선이 아님)를 함께 볼 것.\n"
        "- n_clipped_hours>0인 행(주로 k>1)은 PCS 원 제약이 실제로 걸렸다는 뜻이니, 그 지점의\n"
        "  j_net 변화를 '순수 Q 효과'로 해석할 때는 클리핑이 섞여 있음을 감안할 것.\n"
        "- 임계값을 코드가 자동으로 판정해 출력하지 않는다(probe_q_value.py의 '절반 미만'\n"
        "  자동판정이 부호 전환을 놓쳐 잘못된 결론을 낸 전례가 있다 - 위 수치를 직접 보고\n"
        "  판단할 것).",
        flush=True,
    )


# ============================================================
# 메인
# ============================================================

def _make_path():
    hostname = socket.gethostname()
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(RESULTS_DIR, f'probe_q_sensitivity_{hostname}_{ts}.csv')


def _write_csv(path, records):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        writer.writeheader()
        for r in records:
            writer.writerow(r)
        f.flush()
        os.fsync(f.fileno())
    print(f'CSV 저장: {path}', flush=True)


if __name__ == '__main__':
    _check_env()

    all_rows = []
    for point in POINTS:
        section(f"통제점 {point['point_id']}: b={point['b']}, S={point['S']}, E={point['E']}")

        print('  -- 역률 0.850(원본) --', flush=True)
        rows_base, base_p_ref = _run_condition(point, None, BASE_PF_EXPECTED, None)
        all_rows += rows_base

        print('  -- 역률 0.95(보상, P 고정) --', flush=True)
        rows_target, _ = _run_condition(point, TARGET_PF, TARGET_PF, base_p_ref)
        all_rows += rows_target

    _restore_evaluate_state()

    _write_csv(_make_path(), all_rows)
    _print_curves(all_rows)
    _print_interpretation()
