"""선택적(시각별로 독립적으로 크기가 다른) Q 프로파일이 실제로 내는 이득을 근사·외삽 없이
조류계산으로 직접 측정한다 (확인 전용, 본 파이프라인 미포함 - probe_q_optimal.py의 직접
후속이자 그 결과를 대체하는 실험).

## 배경 - probe_q_optimal.py 결과를 신뢰할 수 없는 이유
probe_q_optimal.py가 산출한 연간 이득(P=1.0MW/S=2.4MVA에서 약 400만원, j_net의 114~160%)은
probe_q_value.py가 실측한 총 선로손실 저감(PF=0.95, S=1.045MVA에서 21.8 MWh/년 ≈ 327만원,
그나마 대부분이 유효전력 P의 기여)을 초과한다. 무효전력 단독의 손실저감 효과가 유효전력을
포함한 총 손실저감보다 클 수는 없으므로 물리적으로 불가능하고, 따라서 과대추정이다.

원인 셋(probe_q_optimal.py의 근사 구조에 내재):
  (a) LHS 기울기(선로손실 저감의 Q에 대한 한계효과 체감률)를 bus 30~32(슬랙까지의 경로가
      가장 긴 말단, path_len 11~13)에서만 실측하고 그 값을 전 32개 버스에 대표값 0.025로
      적용했다. 기울기는 경로상 저항의 합에 비례하므로(probe_q_marginal.py의 LHS 정의 -
      Sum_{e in path} r_e*...) 슬랙에 가까운 짧은 경로 버스에서는 실제 기울기가 훨씬 커야
      하는데 대표값을 그대로 썼다 - 체감이 과소평가되어 q*가 부풀려졌다.
  (b) probe_q_residual.py의 실측 범위는 Q_inj<=0.5 Mvar인데, probe_q_optimal.py는 그
      1차 근사를 최대 2.18 Mvar까지 외삽했다. 이 계통의 총 무효부하가 2.79 Mvar
      (역률 0.850241 기준, 1절 K_SCALE 유도값)이므로 이 영역에서는 애초에 Q_flow의 부호
      자체가 뒤집힐 수 있는 범위라 선형 가정이 구조적으로 붕괴한다.
  (c) probe_q_optimal.py는 버스별로 "독립적으로" 이득을 계산해 상위 5개를 나열했는데,
      상위권 버스들이 슬랙에서 갈라지는 경로를 공유하면 같은 상류 구간의 손실 저감을
      버스마다 중복 계상하게 된다(모듈 docstring의 한계 1번이 이미 경고한 바로 그 문제).

## 목적
LP(lower_lp.py)를 손실 인식형으로 고치기 전에, "각 시각에서 독립적으로 최적화한 Q"가
실제로 얼마의 이득을 내는지 근사·외삽 전혀 없이 매 후보마다 실제 조류계산으로 측정한다.
probe_q_optimal.py의 q_star는 입력으로 쓰지 않는다(그 값이 신뢰 불가라 이 실험을 하는
것이므로 - 지시 사항).

## 왜 "시각별 독립"이 근사가 아니라 정확한 최적인가
lower_lp.py의 SOC 재귀식(시각 간 유일한 결합)은 P_ch/P_dis에만 걸리고 Q에는 전혀
등장하지 않는다(SOC 갱신식: eta_c*P_ch*dt - P_dis/eta_d*dt만 있고 Q항이 없다). 따라서 P를
고정해 두면, 각 시각의 Q_t는 다른 시각의 어떤 선택과도 제약으로 얽히지 않는 완전히 독립인
변수다 - "각 시각에서 손실채널 순이득을 최대화하는 Q_t를 고른다"는 정의상 그 시각의 전역
최적이며(격자 탐색이 충분히 촘촘하다는 전제 하에) 근사가 아니다. probe_q_optimal.py와의
차이는 "독립"이라는 전제가 아니라, **한 시각 안에서 이득을 재는 방법**이다 -
probe_q_optimal.py는 1차 선형근사식으로 재고 이 스크립트는 그 시각의 실제 조류계산으로 잰다.

## 설계
통제점: probe_q_value.py/probe_q_sensitivity.py와 동일한 3점(P1/P2/P3, POINTS 그대로 재사용
- 통제 조건을 새로 정의하지 않는다). 역률: PF=0.95만(원본 0.850241은 이 지시서 범위 밖).

각 통제점에서:
1. force_q_zero=True로 LP를 풀어 P 스케줄을 얻고 **이 P를 고정한다**(probe_q_value.py의
   force_q_zero 우회 메커니즘 - _evaluate_with_force_q - 을 그대로 재사용). 이렇게 얻은
   평가 결과 자체가 이미 "j_net(Q=0)" 기준값이다(재계산할 필요 없음 - 아래 참조).
2. ALL_DAYS x 24h의 각 (시나리오,시각)에서 독립적으로 Q_t를 격자탐색한다:
     Q_t in {0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5} 교집합 {q : sqrt(P_t^2+q^2)<=S}
   각 후보에서 **그 시각만** 조류계산을 다시 돌려
     net_gain(t,q) = [Loss_line(t,0) - Loss_line(t,q)] - Loss_pcs(t,q)
   를 실측하고, net_gain이 최대인 q를 그 시각의 q*_t로 택한다. q=0 기준값도 이 함수
   내부에서 직접 조류계산으로 구한다(다른 곳에서 계산된 값을 가져다 쓰지 않는다 - 그래야
   한 시각 안의 모든 후보가 동일한 웜스타트 이력에서 비교되어 대소 비교가 오염되지 않는다).
3. 이렇게 얻은 q*_t 24x5 프로파일 전체를 실제로 주입해 **전체 평가를 다시 수행**하고
   j_net(Q=q*)을 구한다 - probe_q_sensitivity.py가 이미 갖춘 "LP가 낸 스케줄에서 Q만
   교체해 evaluate_particle 후반부(주입~조류계산~편익)를 재현하는" 함수
   (_reinject_and_evaluate)를 그대로 재사용한다. 그 스크립트의 k-배율 래퍼와 구조가
   동일하고 여기서는 배율 대신 이 스크립트가 새로 찾은 프로파일로 **치환**할 뿐이다.

## 이 실험이 여전히 갖는 한계 (probe_q_optimal.py보다는 훨씬 적지만 완전히 없지는 않다)
- 버스는 각 통제점이 이미 고른 단일 버스로 고정한다(다중 버스 동시 탐색이 아니다) - 따라서
  probe_q_optimal.py의 한계(c)(경로 공유 중복계상)는 이 실험에서 애초에 발생하지 않는다
  (버스가 하나뿐이므로 "경로 공유"라는 개념 자체가 없다).
- 격자 간격(0.02~0.5 Mvar, 9점)보다 촘촘하거나 더 큰 진짜 최적점이 존재할 수 있다 - 다만
  손실은 Q에 대해 매끄러운(연속·오목에 가까운) 함수라 9점 격자로도 큰 폭의 오차는 나지
  않을 것으로 기대된다. "격자 상한(그 시각에서 실현 가능한 최대 q)에서 최적이 잡힌 시각"은
  더 큰 Q를 시도했다면 이득이 더 늘었을 수 있다는 신호이므로 - 전 후보를 다 찍는 대신
  이 경우만 카운트해 stdout에 남긴다(아래 _grid_search_timestep의 boundary_hit 참조).
- P를 force_q_zero=True LP가 낸 값으로 고정한다 - 손실을 아는 LP라면 P 자체도 달라질
  것이므로, 이 스크립트의 결과도 여전히 "지금 이 P 근방에서 Q만 재최적화하면"의 상한이지
  손실인식 LP의 전역최적은 아니다(probe_q_optimal.py 한계 2번과 동일한 성격, 다만 이
  스크립트는 최소한 그 P 근방에서는 근사 없이 정확하다).

실행: `python scripts/probe_q_selective.py`  (★ 이 스크립트는 작성만 하고 실행하지 않는다 -
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
import glob
import socket
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import params as PM
import evaluate

from probe_q_value import (
    POINTS,
    TARGET_PF,
    _build_net_with_pf,
    _prepare_condition,
    _restore_evaluate_state,
    _evaluate_with_force_q,
    section,
    _check_env,
)
from probe_q_sensitivity import (
    _reinject_and_evaluate,
    NOISE_BOUND_WON,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ★ scripts/results (프로젝트 루트 results/가 아니다) - probe_q_marginal.py 등과 동일 관례.
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
ROOT_RESULTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'results')

Q_GRID = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]  # Mvar
FEASIBILITY_TOL = 1e-9   # MVA^2, sqrt(P^2+q^2)<=S 판정 여유(부동소수점)

# 검산2(j_net(Q=q*)>=j_net(Q=0))의 허용오차 - probe_q_sensitivity.py와 동일한 웜스타트
# 이력 잡음 상한을 그대로 쓴다(재정의하지 않고 import).
CHECK2_TOL_WON = NOISE_BOUND_WON

CSV_FIELDS = ['point_id', 'scenario', 't', 'p_t', 'q_star_t', 'gain_t_won', 's_utilization']


# ============================================================
# 시각 1개 격자탐색 (조류계산 직접 실행, 근사 없음)
# ============================================================

def _grid_search_timestep(net, bus, sgen_idx, P_t, S_val, scale, base_p, base_q):
    """단일 (scenario,t)에서 net_gain(q)=[Loss_line(0)-Loss_line(q)]-Loss_pcs(q)를
    실제 조류계산으로 후보마다 재계산해 최댓값을 내는 q*를 찾는다. P_t는 고정.

    q=0 기준도 이 함수 안에서 직접 조류계산으로 구한다(다른 곳에서 계산된 값 재사용 안 함) -
    그래야 이 시각의 모든 후보 비교가 동일한 웜스타트 이력 위에서 이뤄진다.

    반환: (q_star, gain_star, boundary_hit) - boundary_hit=True면 이 시각에서 실현 가능한
    (PCS 원 제약을 만족하는) 격자 후보 중 **가장 큰 q**가 그대로 q*로 선택됐다는 뜻이다
    (feasible 후보가 하나도 없으면(q_max_feasible이 없음) False). 이는 "격자를 더 키웠다면
    더 나은 점을 찾았을 수 있다"는 신호이지 그 자체가 오류는 아니다 - 모듈 docstring의
    한계 절 참조.
    """
    net.load['p_mw'] = base_p * scale
    net.load['q_mvar'] = base_q * scale

    net.sgen.at[sgen_idx, 'bus'] = bus
    net.sgen.at[sgen_idx, 'p_mw'] = P_t
    net.sgen.at[sgen_idx, 'q_mvar'] = 0.0
    ok = evaluate._run_pf_with_retry(net)
    if not ok:
        raise RuntimeError(f'조류계산 발산(q=0 기준): bus={bus} P_t={P_t}')
    loss_line_baseline = float(net.res_line.pl_mw.sum())

    best_q, best_gain = 0.0, 0.0
    max_feasible_q = None

    for q in Q_GRID:
        if q == 0.0:
            continue
        if (P_t ** 2 + q ** 2) > (S_val ** 2 + FEASIBILITY_TOL):
            continue
        max_feasible_q = q   # Q_GRID가 오름차순이므로 마지막에 남는 값이 최대 feasible

        loss_pcs_q = (1.0 - PM.ETA_PCS) * (np.sqrt(P_t ** 2 + q ** 2) - abs(P_t))
        net.sgen.at[sgen_idx, 'p_mw'] = P_t - loss_pcs_q
        net.sgen.at[sgen_idx, 'q_mvar'] = q
        ok = evaluate._run_pf_with_retry(net)
        if not ok:
            raise RuntimeError(f'조류계산 발산: bus={bus} q={q} P_t={P_t}')
        loss_line_q = float(net.res_line.pl_mw.sum())

        gain = (loss_line_baseline - loss_line_q) - loss_pcs_q
        if gain > best_gain:
            best_gain, best_q = gain, q

    boundary_hit = (max_feasible_q is not None and best_q == max_feasible_q)
    return best_q, best_gain, boundary_hit


# ============================================================
# 한 통제점 처리
# ============================================================

def _process_point(point):
    net, q_scale, p_total, q_total_before = _build_net_with_pf(TARGET_PF)
    base_p, base_q = _prepare_condition(net)

    x = np.array([point['b'], point['S'], point['E']], dtype=float)

    # 1) force_q_zero=True로 P 스케줄 확정 - 이 평가 자체가 j_net(Q=0) 기준이다.
    detail_zero = _evaluate_with_force_q(x, True)
    if detail_zero.get('diverged'):
        print(f"  {point['point_id']}: 기준(Q=0 강제) 평가 발산 -> 이 통제점 건너뜀 "
              f"({detail_zero.get('diverge_info')})", flush=True)
        return None

    b_arr, S_arr, E_arr = detail_zero['b'], detail_zero['S'], detail_zero['E']
    bus = int(b_arr[0])
    S_val = float(S_arr[0])
    unit_p = detail_zero['unit_p']   # dict[ALL_DAYS] -> (1,T), 고정

    # ★ 여기서 sgen을 새로 만들지 않는다. _evaluate_with_force_q(x, True) 안의
    # evaluate.evaluate_particle이 이미 evaluate._ensure_sgens(net, n=1)로 sgen을 1개
    # 만들어 뒀다(점들이 전부 n=1 단일기 - CLAUDE.md POINTS 정의 참조). 여기서 pp.create_sgen을
    # 또 부르면 중복 sgen이 생겨(나중에 _reinject_and_evaluate의 _ensure_sgens가 초과분을
    # drop하긴 하지만) 불필요한 혼선이다 - 이미 있는 sgen을 그대로 재사용한다.
    assert len(net.sgen) == 1, (
        f"{point['point_id']}: 예상과 다른 sgen 개수 {len(net.sgen)} (n=1 통제점 가정 위반 - "
        "POINTS가 다중기로 바뀌었는지 확인할 것)"
    )
    sgen_idx = net.sgen.index[0]

    unit_q_star = {s: np.zeros_like(unit_p[s]) for s in PM.ALL_DAYS}
    timestep_rows = []
    n_boundary_hit = 0

    for s in PM.ALL_DAYS:
        profile = PM.LOAD[s]
        for t in range(PM.TIME_STEPS):
            P_t = float(unit_p[s][0, t])
            q_star, gain_star, boundary_hit = _grid_search_timestep(
                net, bus, sgen_idx, P_t, S_val, profile[t], base_p, base_q
            )

            # ---- 검산1: q*=0이면 gain=0 ----
            if q_star == 0.0:
                assert gain_star == 0.0, (
                    f"{point['point_id']}/{s}/t={t}: q*=0인데 gain={gain_star} (구조적으로 "
                    "0이어야 함 - _grid_search_timestep의 초기값 로직 확인)"
                )

            if boundary_hit:
                n_boundary_hit += 1

            unit_q_star[s][0, t] = q_star

            s_util = float(np.sqrt(P_t ** 2 + q_star ** 2) / S_val) if S_val > 0.0 else 0.0
            smp = float(PM.SMP_PER_MWH[s][t])
            gain_t_won = gain_star * smp * PM.DT_HOURS

            timestep_rows.append(dict(
                point_id=point['point_id'], scenario=s, t=t, p_t=P_t,
                q_star_t=q_star, gain_t_won=gain_t_won, s_utilization=s_util,
            ))

    # 3) q*_t 프로파일 전체를 실제로 주입해 재평가 (probe_q_sensitivity.py 재사용)
    result_qstar = _reinject_and_evaluate(b_arr, S_arr, E_arr, unit_p, unit_q_star)

    return dict(
        point=point, bus=bus, detail_zero=detail_zero, result_qstar=result_qstar,
        timestep_rows=timestep_rows, unit_q_star=unit_q_star, n_boundary_hit=n_boundary_hit,
    )


# ============================================================
# probe_q_sensitivity.py 결과와 대조 (검산3, 비-치명적 보고)
# ============================================================

def _find_latest_csv(prefix):
    candidates = []
    for d in (RESULTS_DIR, ROOT_RESULTS_DIR):
        candidates += glob.glob(os.path.join(d, f'{prefix}_*.csv'))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _load_sensitivity_max_jnet(path, point_id, pf_label, tol=1e-6):
    """probe_q_sensitivity.py CSV에서 해당 point_id/PF의 k-곡선 중 j_net 최댓값을 구한다.
    파일이 없거나 매칭되는 행이 없으면 None(검산3은 그때 건너뛴다)."""
    if path is None:
        return None
    best = None
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get('point_id') != point_id or r.get('j_net', '') == '':
                continue
            try:
                pf = float(r['power_factor'])
            except (KeyError, ValueError):
                continue
            if abs(pf - pf_label) > tol:
                continue
            j = float(r['j_net'])
            if best is None or j > best:
                best = j
    return best


# ============================================================
# stdout 요약 (자동판정 없음 - 수치와 대조값만 제시)
# ============================================================

def _print_point_summary(outcome, sensitivity_path):
    point = outcome['point']
    detail_zero = outcome['detail_zero']
    result_qstar = outcome['result_qstar']
    timestep_rows = outcome['timestep_rows']

    section(f"통제점 {point['point_id']} (bus={outcome['bus']})")

    j_net_zero = detail_zero['j_net']
    print(f"j_net(Q=0, force_q_zero=True 기준) = {j_net_zero:,.2f}원/년", flush=True)

    if result_qstar.get('diverged'):
        print(f"j_net(Q=q*) 평가 발산 -> 이 통제점의 이후 비교 생략 "
              f"({result_qstar.get('diverge_info')})", flush=True)
        return

    j_net_qstar = result_qstar['j_net']
    delta = j_net_qstar - j_net_zero
    print(f"j_net(Q=q*, 시각별 격자탐색 프로파일) = {j_net_qstar:,.2f}원/년", flush=True)
    print(f"개선폭 = {delta:+,.2f}원/년  ({delta / j_net_zero * 100:+.4f}% of j_net(Q=0))",
          flush=True)

    # ---- 검산2: j_net(Q=q*) >= j_net(Q=0) (허용오차 CHECK2_TOL_WON) ----
    assert j_net_qstar >= j_net_zero - CHECK2_TOL_WON, (
        f"{point['point_id']}: j_net(Q=q*)={j_net_qstar:.6f} < j_net(Q=0)={j_net_zero:.6f} - "
        "시각별 최적을 골랐다면 정의상 성립해야 하는데 위반됨. P 고정 가정이나 "
        "_reinject_and_evaluate 치환 경로에 오류가 있을 수 있다. 즉시 보고할 것."
    )

    print(f"b_loss(Q=0) = {detail_zero['b_loss']:,.2f}원/년, "
          f"b_loss(Q=q*) = {result_qstar['b_loss']:,.2f}원/년, "
          f"변화 = {result_qstar['b_loss'] - detail_zero['b_loss']:+,.2f}원/년", flush=True)
    print(f"loss_pcs_total_mwh(Q=q*) = {result_qstar['loss_pcs_total_mwh']:.6f} MWh/년",
          flush=True)

    nonzero = [r for r in timestep_rows if r['q_star_t'] != 0.0]
    print(f"q*!=0인 시각 수 = {len(nonzero)} / {len(timestep_rows)}", flush=True)
    if nonzero:
        utils = sorted(r['s_utilization'] for r in nonzero)
        n = len(utils)
        print(f"그 시각들의 s_utilization 분포: min={utils[0]:.4f}, "
              f"median={utils[n // 2]:.4f}, max={utils[-1]:.4f}", flush=True)
    print(f"격자 상한(그 시각에서 실현 가능한 최대 q)에서 q*가 잡힌 시각 수 = "
          f"{outcome['n_boundary_hit']} - 0이 아니면 그 시각들은 더 큰 Q를 시도했을 때 "
          f"이득이 더 늘었을 수 있다(격자 범위 밖 미탐색 신호, 오류 아님)", flush=True)

    # ---- 검산3(비-치명적 보고): probe_q_sensitivity.py k-곡선 최댓값과 대조 ----
    sens_max = _load_sensitivity_max_jnet(sensitivity_path, point['point_id'], TARGET_PF)
    if sens_max is None:
        print("  [검산3 생략] probe_q_sensitivity.py 결과에서 이 point_id/PF=0.95의 j_net을 "
              "찾지 못함(파일 없음 또는 매칭 행 없음)", flush=True)
    else:
        print(f"  [검산3] probe_q_sensitivity.py k-곡선 j_net 최댓값 = {sens_max:,.2f}원/년  "
              f"vs  이 실험 j_net(Q=q*) = {j_net_qstar:,.2f}원/년  "
              f"(차이 = {j_net_qstar - sens_max:+,.2f}원)", flush=True)
        print("  ★ 주의: 두 실험의 P 기준선이 다르다(이 스크립트는 force_q_zero=True LP의 P, "
              "probe_q_sensitivity.py는 자유Q LP의 P를 그대로 스케일) - 따라서 이 비교는 "
              "엄밀한 동일조건 비교가 아니며, 부호가 어느 쪽이든 그 이유를 P 기준선 차이와 "
              "함께 해석할 것.", flush=True)


# ============================================================
# 메인
# ============================================================

def _make_path():
    hostname = socket.gethostname()
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(RESULTS_DIR, f'probe_q_selective_{hostname}_{ts}.csv')


def _write_csv(path, rows):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        f.flush()
        os.fsync(f.fileno())
    print(f'CSV 저장: {path}', flush=True)


if __name__ == '__main__':
    _check_env()

    sensitivity_path = _find_latest_csv('probe_q_sensitivity')

    all_timestep_rows = []
    outcomes = []
    for point in POINTS:
        section(f"통제점 {point['point_id']}: b={point['b']}, S={point['S']}, E={point['E']}")
        outcome = _process_point(point)
        if outcome is None:
            continue
        outcomes.append(outcome)
        all_timestep_rows += outcome['timestep_rows']

    _restore_evaluate_state()

    _write_csv(_make_path(), all_timestep_rows)

    for outcome in outcomes:
        _print_point_summary(outcome, sensitivity_path)
