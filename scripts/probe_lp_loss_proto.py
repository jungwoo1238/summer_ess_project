"""solve_avg 목적함수에 Q의 손실저감 편익을 넣는 두 방식(PWL/QP)을 프로토타입으로 구현해
정확도·속도를 비교한다 (확인 전용, 본 파이프라인 미포함 - probe_q_selective.py의 직접 후속.
본 구현(lower_lp.py 수정) 전 방식 선택을 위한 실험).

## ★★ 이전 실험(같은 파일의 1차 버전)이 무효였던 이유 - 2차 개정의 핵심 동기
기준해 q_star의 Q는 압도적으로 PEAK_DAYS에 몰려 있다(1차 실행 결과 실측):
  P1: AVG 4시각/0.080 Mvar  vs  PEAK 40시각/1.750 Mvar (22배)
  P2: AVG 10시각/0.800      vs  PEAK 45시각/2.230
  P3: AVG 10시각/1.800      vs  PEAK 48시각/4.940
그런데 1차 버전은 solve_avg만 손실 인식형으로 고쳤다 - LP는 AVG_DAYS에서만 Q를 냈고
기준해는 PEAK_DAYS에서 Q를 냈다. 두 시나리오군이 겹치는 시각은 4~10개뿐이라 j_net 격차가
"근사 정확도"가 아니라 "서로 다른 전략을 비교한 결과"였다 - M을 늘릴수록(더 정밀한 근사일수록)
j_net이 오히려 나빠진 것(P1: -2.64% -> -3.04%)이 이 진단의 방증이다(더 정밀해질수록 AVG_DAYS
안에서는 더 정확해지지만, 애초에 PEAK_DAYS의 Q를 아예 못 내므로 전체 그림은 개선되지 않는다).

## 2차 개정 변경 사항
1. **solve_peak 변형에도 동일한 손실 항·PCS 항을 넣고, 속도뿐 아니라 정확도 비교
   대상에도 포함한다.** 손실 테이블 실측 범위를 AVG_DAYS(3개)에서 ALL_DAYS(5개)로
   넓혔다(PEAK_DAYS 두 시나리오의 PWL 계수가 새로 필요해졌으므로).
2. **solve_peak 목적함수를 원 단위로 환산한다** (★ 이것은 표시상의 정리가 아니라 실질적
   버그 수정이다 - 아래 "C_CAP 스케일 검산" 절 참조):
     objective = -C_CAP_PER_MW_YR * (peak_base - pk) + Sum_t SMP*DT*(손실 + PCS)
   1차 버전은 `pk`(MW, 무차원 스케일)를 `loss_term`/`pcs_cost`(원, SMP로 환산된 스케일)와
   **단위를 맞추지 않은 채** 같은 목적함수에서 더했다 - 이는 "피크저감 1MW"와 "손실저감
   1원"을 동등하게 취급한 것과 같아 두 항의 상대적 중요도가 원천적으로 왜곡돼 있었다.
   C_CAP_PER_MW_YR(약 7,608만원/MW-yr)을 곱해야 비로소 "피크 1MW 저감이 실제로 얼마의
   경제적 가치를 갖는가"가 손실 절감액과 같은 저울(원)에 올라간다.
3. **AVG+PEAK 양쪽 Q를 모두 적용한 뒤 j_net을 세 가지로 평가**하고 대조한다:
     (a) Q=0                (force_q_zero=True LP의 P, Q 전부 0)
     (b) Q=q_lp             (AVG+PEAK 모두 손실 인식형 LP가 P,Q를 함께 결정)
     (c) Q=q_star           (force_q_zero=True LP의 P + probe_q_selective.py 기준해 Q)
   (a),(c)는 같은 P(force_q_zero=True)를 공유하므로 "Q만의 순효과"를 정확히 비교하고,
   (b)는 LP가 P까지 다시 정한 결과라 "실제로 이 LP를 배포하면 어떻게 되는가"를 본다.

### C_CAP 스케일 검산 (직접 계산, 코드로 재확인)
`params.C_CAP_PER_MW_YR = 76,082.2 * 1000 = 76,082,200원/MW-yr`. `pk`는 1개 시나리오
(예: summer_peak)의 부하단 피크(MW, 대략 8~9 MW 스케일 - 1절 K_SCALE 기준 10MVA 계통).
`-C_CAP*(peak_base-pk)`가 `pk`를 1kW(0.001MW) 줄일 때 objective를 `76,082.2원`만큼
낮춘다(그 방향이 이득이므로 음의 기여, 즉 목적함수 감소 = 최소화에 유리). 반면 손실 저감
1kWh는 SMP(~100~180원/kWh)어치다 - 두 항의 상대 크기가 비로소 물리적으로 말이 되는
스케일에서 경쟁한다(전자가 자릿수가 훨씬 크다 - CLAUDE.md 8절 실측: b_defer가 총편익의
84.4%를 차지하는 것과 정합적인 크기 관계).

## 방식 A: PWL (조각선형)
Q를 M개 구간으로 나누고 구간별 "실측" 기울기(시컨트, 근사·대표값 아님)를 선형 계수로 준다.
**전 32버스 x ALL_DAYS 5개 x 24시간**에서 Q_BOUNDARY_POINTS 각 점의 손실을 직접
조류계산으로 실측해(probe_q_residual.py의 단일버스 실험을 전면 확장한 것) 세그먼트
경계마다 실제 시컨트 기울기를 쓴다 - 근사는 "구간 안에서 선형"이라는 가정 하나뿐이다.

## 방식 B: QP (2차 손실)
LinDistFlow가 이미 계산하는 분기조류 P_e,Q_e를 그대로 재사용해
Sum_e r_e*(P_e^2+Q_e^2)/V_ref^2 * SMP * DT를 목적함수에 비용으로 더한다. V_ref=1.0 고정 -
방사형 배전망 전압이 1.0 pu 근방에 머무는 것을 이용한 근사(1절: 기저 전압범위
0.962~1.02) - 전압이 1.0에서 먼 버스일수록 손실을 과대평가하는 방향으로 편향된다.
DPP가 깨진다는 것을 1차 실행에서 실측 확인했고, solve_peak 변형 풀이시간이 PWL의
3~5배였으며 OSQP 내부 에러도 겪었다 - **QP는 우선순위를 낮추되, PWL이 기준해를 재현하지
못할 경우를 대비한 비교군으로는 계속 돌린다**(지시 사항).

## PCS 손실 항 (양쪽 공통) - s_app을 다각형의 고정 상수 대신 변수로 승격
기존 lower_lp.py는 force_q_zero=False일 때 다각형 상한을 고정 상수 S*cos(pi/12)로 건다.
이 프로토타입은 그 상수를 변수 s_app(다각형이 실제로 근사하는 피상전력 상한)으로 바꾸고,
q_penalty = max(0, s_app-(P_ch+P_dis))에 비용을 매긴다 - Q=0이면 최소화 목적상 s_app이
|P_net|까지 줄어들어 q_penalty가 정확히 0이 되므로(회귀 앵커 보존), Q>0일 때만 "PCS가
그만큼 더 여유를 확보하는 데 드는 비용"이 원화로 매겨진다. C_PCS=1-ETA_PCS를 계수로 써서
benefits.loss_pcs와 동일한 물리 상수를 공유한다(새 상수 발명 안 함).

## solver 정확도 문제 (1차 실행에서 PWL 전 M·QP 모두 "Solution may be inaccurate" 반복)
`_solve_timed`가 CLARABEL(내점법)을 우선 시도하고, 실패·부정확 시 OSQP(반복한도 5만·
eps_abs=eps_rel=1e-6으로 완화)를 재시도, 그래도 안 되면 cvxpy 기본 자동선택으로 마지막
시도한다. OPTIMAL이 아니라 OPTIMAL_INACCURATE로 끝난 호출은 **값은 그대로 쓰되** 집계에
남겨 실행 후 stdout(솔버 진단 절)에 어떤 (kind,method,M) 조합에서 몇 번 발생했는지
보고한다 - 자동으로 재시도 배정을 더 늘리거나 판정을 내리지 않는다(사람이 보고 판단).
`problem.solver_stats.solver_name`으로 실제 선택된 솔버를 매 호출 기록한다.

**★ "기존 42개 테스트" 재확인 불필요 - 이 스크립트는 lower_lp.py를 한 줄도 수정하지 않는다.**
여기서 시도하는 solver 설정(CLARABEL/OSQP 우선순위, eps 완화)은 이 프로토타입 파일 안의
독자적인 `cp.Problem` 인스턴스에만 적용되고, lower_lp.py의 `_PROBLEM_CACHE`/`solve_avg`/
`solve_peak`와는 아무 상태도 공유하지 않는다 - test_lp.py 등 기존 테스트 스위트는 이
스크립트의 존재나 실행 여부와 무관하게 항상 원본 lower_lp.py만 검증한다. 따라서 "42개
테스트"는 이 스크립트가 아니라 **향후 lower_lp.py에 실제로 통합할 때** 재확인할 대상이다.

★★ lower_lp.py 원본은 이 스크립트 전체에서 한 줄도 수정하지 않는다. 아래
_build_problem_proto()가 lower_lp._build_problem()의 구조를 복사해 확장한 것이고,
lower_lp._get_topology()/_prepare_common()만 읽기 전용으로 재사용한다(원본 solve_peak은
2차 개정에서 더 이상 호출하지 않는다 - AVG/PEAK 둘 다 이 스크립트의 프로토타입으로 푼다).

실행: `python scripts/probe_lp_loss_proto.py`  (★ 이 스크립트는 작성만 하고 실행하지 않는다 -
실행은 사용자가 터미널에서 직접 한다. 입력으로 scripts/results/의 가장 최근
probe_q_selective_*.csv가 필요하다 - 없으면 즉시 에러로 알린다. ALL_DAYS 손실 테이블
실측 때문에 1차 버전보다 오래 걸린다 - 32버스x5시나리오x24h x 5경계점 = 19,200회 조류계산.)

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
import time
import socket
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cvxpy as cp
import pandapower as pp

import params as PM
import evaluate
import lower_lp

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
from probe_q_sensitivity import _reinject_and_evaluate

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')          # ★ scripts/results 관례
ROOT_RESULTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'results')

# ============================================================
# 상수
# ============================================================

# 세그먼트 경계 후보의 합집합 - 이 점들에서만 실측(Loss_line, P_inj=0 고정 -
# probe_q_residual.py와 동일 규약)하고, 각 M의 세그먼트는 이 집합의 부분집합만 쓴다.
Q_BOUNDARY_POINTS = [0.0, 0.05, 0.15, 0.3, 0.5]   # Mvar

# ★ 경계 선택 근거: 0.05/0.15/0.5는 지시서 권장값을 그대로 쓴다. M=1,2는 이 권장값의
# 부분집합으로 충분히 구성되지만(M=1: 양끝만, M=2: 중간점 0.15로 2등분), M=4는 권장값
# 3개(0.05,0.15,0.5)+시작점(0)뿐이라 세그먼트가 3개까지만 나온다 - 4개를 만들려면 점을
# 하나 더 찍어야 하므로 0.15~0.5 구간(폭 0.35로 나머지 두 구간보다 훨씬 넓다)을 0.3에서
# 한 번 더 쪼갠다(그 구간이 가장 넓어 선형근사 오차가 가장 클 구간이므로 거기를 세분하는
# 것이 합리적).
SEGMENT_BOUNDARIES = {
    1: [0.0, 0.5],
    2: [0.0, 0.15, 0.5],
    4: [0.0, 0.05, 0.15, 0.3, 0.5],
}

ALL_BUSES = list(range(PM.B_BOUNDS[0], PM.B_BOUNDS[1] + 1))   # 1..32

C_PCS = 1.0 - PM.ETA_PCS   # benefits.loss_pcs와 동일 물리상수 재사용(새로 만들지 않음)

# 사용자 제공 참조값(이 세션에서 독립 검증하지 않음).
REFERENCE_SOLVE_TIME_SEC = 0.549

METHODS = [('pwl', 1), ('pwl', 2), ('pwl', 4), ('qp', None)]

TS_CSV_FIELDS = ['method', 'M', 'point_id', 'scenario', 't', 'q_lp', 'q_star', 'abs_err', 'rel_err']


# ============================================================
# 1) 손실 테이블 실측 (probe_q_residual.py 방식의 전 버스·ALL_DAYS 확장)
# ============================================================

def _measure_loss_table(net, base_p, base_q):
    """ALL_DAYS x 24h x 전 32버스 x Q_BOUNDARY_POINTS에서 Loss_line(P_inj=0,Q_inj=q)를
    실측한다. 반환: loss_table[bus][scenario] = ndarray shape (24, len(Q_BOUNDARY_POINTS)).
    probe_q_residual.py와 동일하게 P_inj=0 고정 - "Q 단독의 손실저감 효과"를 재는 것이
    이 실험의 정의이기 때문이다. 2차 개정: solve_peak도 정확도 비교 대상이 되어 PEAK_DAYS
    (summer_peak, winter_peak)의 PWL 계수도 진짜 실측이 필요해졌으므로 AVG_DAYS에서
    ALL_DAYS로 범위를 넓혔다(1차 버전은 AVG_DAYS만 실측하고 PEAK_DAYS 타이밍 테스트는
    대표상수로 때웠다 - 그 타이밍 전용 경로는 이번 개정에서 폐기됐다)."""
    if len(net.sgen) == 0:
        pp.create_sgen(net, bus=ALL_BUSES[0], p_mw=0.0, q_mvar=0.0, name='probe_lp_loss_proto')
    sgen_idx = net.sgen.index[0]

    loss_table = {}
    n_total = len(ALL_BUSES) * len(PM.ALL_DAYS)
    done = 0
    for bus in ALL_BUSES:
        loss_table[bus] = {}
        for s in PM.ALL_DAYS:
            profile = PM.LOAD[s]
            arr = np.zeros((PM.TIME_STEPS, len(Q_BOUNDARY_POINTS)))
            for t in range(PM.TIME_STEPS):
                scale = profile[t]
                net.load['p_mw'] = base_p * scale
                net.load['q_mvar'] = base_q * scale
                for qi, q in enumerate(Q_BOUNDARY_POINTS):
                    net.sgen.at[sgen_idx, 'bus'] = bus
                    net.sgen.at[sgen_idx, 'p_mw'] = 0.0
                    net.sgen.at[sgen_idx, 'q_mvar'] = q
                    ok = evaluate._run_pf_with_retry(net)
                    if not ok:
                        raise RuntimeError(
                            f'조류계산 발산: bus={bus} s={s} t={t} q={q} - 정상범위 비정상.'
                        )
                    arr[t, qi] = float(net.res_line.pl_mw.sum())
            loss_table[bus][s] = arr
            done += 1
            if done % 20 == 0 or done == n_total:
                print(f'  손실 테이블 실측 진행: {done}/{n_total} (bus={bus}, scenario={s})',
                      flush=True)
    return loss_table


def _segment_slopes(arr_row, boundaries):
    """arr_row: len(Q_BOUNDARY_POINTS) 배열(한 t의 Loss_line 실측값, Q_BOUNDARY_POINTS와 같은
    순서). boundaries는 Q_BOUNDARY_POINTS의 부분집합(오름차순)이어야 한다. 반환:
    len(boundaries)-1개의 실측 시컨트 기울기(LHS_m, m=0..M-1, MW/Mvar, 양수=손실감소)."""
    idx = {q: i for i, q in enumerate(Q_BOUNDARY_POINTS)}
    slopes = []
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        loss_lo = arr_row[idx[lo]]
        loss_hi = arr_row[idx[hi]]
        slopes.append(float((loss_lo - loss_hi) / (hi - lo)))
    return slopes


def _lhs_rows_for(loss_table, bus, scenario, M):
    """(bus,scenario)의 24시간 각각에 대해 _segment_slopes를 적용해 (M,T) 배열을 만들고,
    lhs_params[m].value에 넣을 (1,T) 조각들의 리스트로 반환한다(n=1 고정)."""
    boundaries = SEGMENT_BOUNDARIES[M]
    arr = loss_table[bus][scenario]   # (T, len(Q_BOUNDARY_POINTS))
    T = arr.shape[0]
    slopes_by_t = np.array([_segment_slopes(arr[t], boundaries) for t in range(T)])  # (T,M)
    return [slopes_by_t[:, m][None, :] for m in range(M)]   # M개의 (1,T) 배열


# ============================================================
# 2) 프로토타입 LP 빌더 (avg/peak x pwl/qp 공통 골격)
# ============================================================

def _build_problem_proto(kind, method, n, T, M=None):
    """lower_lp._build_problem(kind,...)의 구조(SOC/PCS개별한계/LinDistFlow/전압유도항)를
    그대로 복사하되, 다각형 상한을 변수 s_app으로 승격하고 손실편익(method)·PCS손실비용
    (공통)을 목적함수에 추가한다. kind='peak'는 추가로 pk를 C_CAP_PER_MW_YR로 원화환산한다
    (모듈 docstring "C_CAP 스케일 검산" 절 참조 - 표시 정리가 아니라 실질적 스케일 정합).
    force_q_zero 경로는 다루지 않는다(이 실험은 Q 자유 최적화만 대상 - force_q_zero=True
    대조군은 probe_q_selective.py가 이미 만들었다). lower_lp.py 원본은 이 함수가 건드리지
    않는다(읽기 전용 재사용: _get_topology())."""
    dt = PM.DT_HOURS
    topo = lower_lp._get_topology()
    D, r_pu, x_pu = topo['D'], topo['r_pu'], topo['x_pu']
    n_bus = topo['n_bus']
    R_MAT = np.tile(r_pu[:, None], (1, T))
    X_MAT = np.tile(x_pu[:, None], (1, T))

    P_ch = cp.Variable((n, T), nonneg=True)
    P_dis = cp.Variable((n, T), nonneg=True)
    soc = cp.Variable((n, T + 1))
    s_app = cp.Variable((n, T), nonneg=True)
    q_penalty = cp.Variable((n, T), nonneg=True)

    lhs_params = None
    if method == 'pwl':
        Q_seg = cp.Variable((n, T, M), nonneg=True)
        Q = cp.sum(Q_seg, axis=2)
        # ★ nonneg=True를 주지 않는다 - 이 Parameter의 값은 "실측 시컨트 기울기 * SMP"다
        # (DPP 유지를 위한 사전곱 - _set_params 참조). 실측 기울기는 항상 양수라는 보장이
        # 없다(무효조류 총량을 넘는 영역에서는 Q_flow 부호가 뒤집혀 손실이 오히려 늘 수
        # 있다). nonneg=True로 선언했다가 실측 데이터의 음수 기울기로 "Parameter value must
        # be nonnegative" 에러가 실제로 났었다 - 물리적으로 정상이므로 제약을 없앤다.
        lhs_params = [cp.Parameter((n, T)) for _ in range(M)]
    elif method == 'qp':
        Q_seg = None
        Q = cp.Variable((n, T))
    else:
        raise ValueError(method)

    S_param = cp.Parameter(n, nonneg=True)
    E_param = cp.Parameter(n, nonneg=True)
    bus_onehot = cp.Parameter((n, n_bus))
    load_p_bus = cp.Parameter((n_bus, T))
    load_q_bus = cp.Parameter((n_bus, T))
    # ★ nonneg=True 필수(lower_lp.py 원본과의 차이) - 원본 solve_avg의 smp_param은 부호를
    # 몰라도 됐다(곱해지는 대상이 P_ch-P_dis, 즉 affine이라 부호 무관하게 여전히 affine이라
    # DCP 자동 성립). 이 프로토타입의 QP 손실항(method='qp')은 smp_row를 **convex**
    # 표현식(r_e*(P_e^2+Q_e^2))에 곱하므로 부호를 알아야 DCP가 성립한다 - 안 붙였다가
    # 실제로 "Problem does not follow DCP rules"로 걸렸다. SMP는 물리적으로 항상 양수다.
    smp_param = cp.Parameter(T, nonneg=True)

    S_col = cp.reshape(S_param, (n, 1), order='C')
    E_col = cp.reshape(E_param, (n, 1), order='C')
    smp_row = cp.reshape(smp_param, (1, T), order='C')

    constraints = [P_ch <= S_col, P_dis <= S_col]
    constraints += [
        soc[:, 0] == PM.SOC_INIT_FRAC * E_param,
        soc[:, T] == PM.SOC_INIT_FRAC * E_param,
        soc >= PM.SOC_MIN_FRAC * E_col,
        soc <= PM.SOC_MAX_FRAC * E_col,
    ]
    for t in range(T):
        constraints.append(
            soc[:, t + 1] == soc[:, t] * (1 - PM.SELF_DISCHARGE_HOURLY)
            + PM.ETA_C * P_ch[:, t] * dt - P_dis[:, t] / PM.ETA_D * dt
        )

    P_net = P_dis - P_ch

    # ---- 다각형: 고정 s_cap 대신 변수 s_app (모듈 docstring "PCS 손실 항" 참조) ----
    for k in range(PM.POLY_N):
        theta = 2.0 * np.pi * k / PM.POLY_N
        constraints.append(P_net * float(np.cos(theta)) + Q * float(np.sin(theta)) <= s_app)
    constraints.append(s_app <= S_col)
    constraints.append(q_penalty >= s_app - P_ch - P_dis)

    if method == 'pwl':
        boundaries = SEGMENT_BOUNDARIES[M]
        for m in range(M):
            delta_m = float(boundaries[m + 1] - boundaries[m])
            constraints.append(Q_seg[:, :, m] <= delta_m)

    netinj_p = (load_p_bus - bus_onehot.T @ P_net) / PM.S_BASE_MVA
    netinj_q = (load_q_bus - bus_onehot.T @ Q) / PM.S_BASE_MVA
    P_e = D @ netinj_p
    Q_e = D @ netinj_q
    v = PM.V_SLACK_SQ - 2.0 * (D.T @ (cp.multiply(R_MAT, P_e) + cp.multiply(X_MAT, Q_e)))
    v_nonslack = v[1:, :]
    # mu_volt: lower_lp.py와 동일 이유로 Parameter가 아니라 float 상수로 굽는다.
    volt_penalty = float(PM.MU_VOLT) * cp.sum(
        cp.pos(v_nonslack - PM.V_SQ_MAX) + cp.pos(PM.V_SQ_MIN - v_nonslack)
    )

    pcs_cost = float(C_PCS) * cp.sum(cp.multiply(smp_row, q_penalty)) * dt

    if method == 'pwl':
        # ★ DPP 함정(1차 작성 중 실제로 걸림): cp.multiply(smp_row, cp.multiply(lhs_params[m],
        # Q_seg))처럼 두 개의 서로 다른 Parameter(smp_row, lhs_params[m])를 곱하면 DPP가
        # 요구하는 "곱셈의 한쪽은 반드시 parameter-free"를 어긴다. 해결: SMP를
        # lhs_params[m]의 **값**에 미리 곱해 둔다(순수 numpy, cvxpy 밖 - _set_params 참조).
        loss_benefit = 0
        for m in range(M):
            loss_benefit = loss_benefit + cp.sum(cp.multiply(lhs_params[m], Q_seg[:, :, m]))
        loss_term = -loss_benefit * dt   # 비용에서 차감(이득)
    else:
        # V_ref=1.0 고정 - 모듈 docstring 참조. QP: P_e/Q_e 제곱이 DPP를 깬다(1차 실행에서
        # is_dcp(dpp=True)=False 실측 확인).
        loss_qp_mw = cp.multiply(R_MAT, cp.square(P_e) + cp.square(Q_e))
        loss_term = cp.sum(cp.multiply(smp_row, loss_qp_mw)) * dt   # 비용으로 가산

    params = dict(S=S_param, E=E_param, bus_onehot=bus_onehot,
                  load_p_bus=load_p_bus, load_q_bus=load_q_bus, smp=smp_param)
    if method == 'pwl':
        params['lhs'] = lhs_params

    varset = dict(P_ch=P_ch, P_dis=P_dis, Q=Q, soc=soc, P_net=P_net,
                  s_app=s_app, q_penalty=q_penalty)

    if kind == 'avg':
        objective_expr = (
            cp.sum(cp.multiply(smp_row, P_ch - P_dis)) * dt
            + 1e-6 * cp.sum(P_ch + P_dis)   # lower_lp.py의 EPS_REG와 동일 값(그 상수를
                                              # export하지 않으므로 문헌값 그대로 복제 -
                                              # 값이 갈리면 회귀검증 무의미해지니 원본과
                                              # 반드시 대조할 것).
            + volt_penalty + pcs_cost + loss_term
        )
    elif kind == 'peak':
        load_total_param = cp.Parameter(T)
        # ★ 2차 개정 핵심: peak_base(그 시나리오의 ESS 개입 전 원래 피크, 상수)를 Parameter로
        # 받아 pk를 C_CAP_PER_MW_YR로 원화환산한다 - 모듈 docstring "C_CAP 스케일 검산" 참조.
        # -(peak_base-pk)를 전개하면 -peak_base+pk이므로 peak_base 항은 최적화에 영향을
        # 주지 않는 상수 오프셋이지만(argmin 불변), C_CAP_PER_MW_YR로 pk 자체를 스케일하는
        # 것은 실질적 변경이다(다른 항들과 처음으로 같은 저울인 "원"에 놓인다).
        peak_base_param = cp.Parameter(nonneg=True)
        pk = cp.Variable()
        constraints.append(pk >= load_total_param - cp.sum(P_net, axis=0))
        objective_expr = (
            -float(PM.C_CAP_PER_MW_YR) * (peak_base_param - pk)
            + volt_penalty + pcs_cost + loss_term
        )
        params['load_total'] = load_total_param
        params['peak_base'] = peak_base_param
        varset['pk'] = pk
    else:
        raise ValueError(kind)

    problem = cp.Problem(cp.Minimize(objective_expr), constraints)
    dpp_preserved = bool(problem.is_dcp(dpp=True))
    return dict(problem=problem, params=params, vars=varset,
                kind=kind, method=method, M=M, dpp_preserved=dpp_preserved)


def _set_params(entry, S, E, bus_idx, profile, smp, lhs_row_values=None, load_total=None):
    """entry(=_build_problem_proto 반환)의 Parameter들에 실제 값을 채운다.
    lower_lp._prepare_common을 그대로 재사용해 onehot/load_p_bus/load_q_bus를 만든다
    (읽기 전용 재사용 - lower_lp.py 원본 미수정)."""
    n, S_val, E_val, onehot, load_p_val, load_q_val = lower_lp._prepare_common(
        S, E, bus_idx, profile, PM.SELF_DISCHARGE_HOURLY, PM.SOC_INIT_FRAC
    )
    p = entry['params']
    p['S'].value = S_val
    p['E'].value = E_val
    p['bus_onehot'].value = onehot
    p['load_p_bus'].value = load_p_val
    p['load_q_bus'].value = load_q_val
    smp_arr = np.asarray(smp, dtype=float)
    p['smp'].value = smp_arr
    if entry['method'] == 'pwl':
        # ★ SMP를 여기서 미리 곱한다(DPP 유지 목적 - _build_problem_proto의 loss_benefit
        # 주석 참조). lhs_row_values[m]은 순수 MW/Mvar 기울기(_lhs_rows_for)이고, 이 곱셈
        # 이후 Parameter에 담기는 값은 원/(Mvar*h) 단위다.
        for m, param in enumerate(p['lhs']):
            param.value = lhs_row_values[m] * smp_arr[None, :]   # (1,T)*(T,) -> (1,T)
    if entry['kind'] == 'peak':
        load_total_arr = np.asarray(load_total, dtype=float)
        p['load_total'].value = load_total_arr
        p['peak_base'].value = float(np.max(load_total_arr))
    return n


def _solve_timed(entry):
    """★ 솔버 우선순위와 정확도 진단(지시서 "solver 정확도 문제" 절). CLARABEL(내점법 -
    이 문제 규모에서 ADMM 기반 OSQP보다 고정밀 수렴이 기대됨)을 먼저 시도하고, 실패하거나
    OPTIMAL/OPTIMAL_INACCURATE가 아니면 OSQP(반복한도 5만, eps_abs=eps_rel=1e-6으로 완화)로
    재시도, 그래도 안 되면 cvxpy 기본 자동선택으로 마지막 시도한다. OPTIMAL_INACCURATE도
    받아들이되(값을 버리지 않는다) inaccurate=True로 표시해 상위에서 (kind,method,M,
    point_id,scenario) 단위로 집계·보고한다 - 자동으로 판정하지 않는다.
    반환: (elapsed, solver_name, inaccurate)."""
    t0 = time.perf_counter()
    for solver_kwargs in (
        dict(solver=cp.CLARABEL),
        dict(solver=cp.OSQP, max_iter=50000, eps_abs=1e-6, eps_rel=1e-6),
        dict(),
    ):
        try:
            entry['problem'].solve(**solver_kwargs)
        except Exception:
            continue
        if entry['problem'].status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            break
    elapsed = time.perf_counter() - t0
    if entry['problem'].status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(
            f"LP 미해결(status={entry['problem'].status}): kind={entry['kind']} "
            f"method={entry['method']} M={entry['M']} (CLARABEL/OSQP(완화)/기본 모두 실패)"
        )
    stats = entry['problem'].solver_stats
    solver_name = stats.solver_name if stats is not None else 'unknown'
    inaccurate = (entry['problem'].status == cp.OPTIMAL_INACCURATE)
    return elapsed, solver_name, inaccurate


def _count_recompiles(solve_times_by_scenario):
    """★ 휴리스틱 지표(참고용) - cvxpy 내부 컴파일 횟수를 직접 계측하는 공개 API가 없어
    벽시계 시간 패턴으로 간접 추정한다. 같은 Problem 객체를 Parameter 값만 바꿔 재호출할 때,
    DPP가 실제로 유지되면 최초 1회만 느리고 이후 호출은 뚜렷이 빨라지는 것이 기대된다 -
    그 낙차가 없으면 매번 다시 컴파일됐다고 볼 근거가 된다. 최소 시간의 2배를 넘는 호출을
    '재컴파일'로 센다(거친 임계값 - 정확한 값이 필요하면 cProfile 등으로 별도 확인할 것)."""
    times = list(solve_times_by_scenario.values())
    t_min = min(times)
    return sum(1 for t in times if t > 2.0 * t_min)


# ============================================================
# 3) 시나리오군 스케줄 계산 (avg/peak 공통 - kind로 분기)
# ============================================================

def _compute_schedule(entry, S, E, bus, loss_table, base_p_sum, scenarios):
    """scenarios(PM.AVG_DAYS 또는 PM.PEAK_DAYS)를 프로토타입 LP로 풀어 unit_p/unit_q,
    시나리오별 풀이시간·솔버명·부정확 여부를 반환한다. 같은 entry(Problem 객체)를
    재사용하며 Parameter 값만 갱신한다(lower_lp.py의 캐싱 철학과 동일)."""
    unit_p, unit_q = {}, {}
    solve_times, solver_names, inaccurate_flags = {}, {}, {}
    for s in scenarios:
        smp = PM.SMP_PER_MWH[s]
        profile = PM.LOAD[s]
        lhs_rows = None
        if entry['method'] == 'pwl':
            lhs_rows = _lhs_rows_for(loss_table, bus, s, entry['M'])
        load_total = None
        if entry['kind'] == 'peak':
            load_total = base_p_sum * np.asarray(profile, dtype=float)
        _set_params(entry, S, E, [bus], profile, smp, lhs_row_values=lhs_rows,
                    load_total=load_total)
        elapsed, solver_name, inaccurate = _solve_timed(entry)
        solve_times[s] = elapsed
        solver_names[s] = solver_name
        inaccurate_flags[s] = inaccurate
        v = entry['vars']
        unit_p[s] = v['P_net'].value
        unit_q[s] = v['Q'].value
    return unit_p, unit_q, solve_times, solver_names, inaccurate_flags


def _group_stats(unit_q, scenarios, tol=1e-6):
    """scenarios 그룹 안에서 0이 아닌 시각 수와 Q 합(Mvar)을 센다(지시서가 제시한
    "AVG N시각/M Mvar vs PEAK N시각/M Mvar" 형식의 진단 재현)."""
    n_nonzero = 0
    total = 0.0
    for s in scenarios:
        arr = unit_q[s][0]
        n_nonzero += int(np.sum(np.abs(arr) > tol))
        total += float(np.sum(arr))
    return n_nonzero, total


def _build_qstar_unit_q(point_id, qstar_full):
    """qstar_full(dict[(point_id,scenario,t)]->q_star_t)에서 이 point_id의 ALL_DAYS
    (1,T) 배열 딕셔너리를 재구성한다."""
    unit_q_star = {s: np.zeros((1, PM.TIME_STEPS)) for s in PM.ALL_DAYS}
    for s in PM.ALL_DAYS:
        for t in range(PM.TIME_STEPS):
            unit_q_star[s][0, t] = qstar_full.get((point_id, s, t), 0.0)
    return unit_q_star


# ============================================================
# 4) 기준해(probe_q_selective.py) 로딩 + 통제점별 (a)/(c) 기준값
# ============================================================

def _find_latest_csv(prefix):
    candidates = []
    for d in (RESULTS_DIR, ROOT_RESULTS_DIR):
        candidates += glob.glob(os.path.join(d, f'{prefix}_*.csv'))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _load_selective_qstar_full(path):
    """point_id,scenario,t -> q_star_t. ALL_DAYS 전부 싣는다."""
    table = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            table[(r['point_id'], r['scenario'], int(r['t']))] = float(r['q_star_t'])
    return table


def _compute_point_baselines(point, qstar_full):
    """(a) Q=0과 (c) Q=q_star의 j_net을 한 번에 계산한다 - 둘 다 force_q_zero=True LP의
    P를 공유하므로 evaluate_particle을 한 번만 호출하면 된다(detail_zero['j_net'] 자체가
    이미 (a)다 - probe_q_selective.py와 완전히 동일한 순서로 계산하므로 그 스크립트의
    j_net(Q=0)/j_net(Q=q*)와 사실상 같은 값이 나온다, 7절 웜스타트 잡음 <10원 무시 가능)."""
    x = np.array([point['b'], point['S'], point['E']], dtype=float)
    detail_zero = _evaluate_with_force_q(x, True)
    if detail_zero.get('diverged'):
        return dict(j_net_a=None, j_net_c=None, bus=None)

    j_net_a = detail_zero['j_net']
    unit_p = detail_zero['unit_p']
    b_arr, S_arr, E_arr = detail_zero['b'], detail_zero['S'], detail_zero['E']
    bus = int(b_arr[0])

    unit_q_star = _build_qstar_unit_q(point['point_id'], qstar_full)
    result_c = _reinject_and_evaluate(b_arr, S_arr, E_arr, unit_p, unit_q_star)
    j_net_c = result_c['j_net'] if not result_c.get('diverged') else None

    return dict(j_net_a=j_net_a, j_net_c=j_net_c, bus=bus)


# ============================================================
# 5) 통제점 x 방식 1회 처리
# ============================================================

def _process(point, method, M, avg_entry, peak_entry, loss_table, base_p_sum,
             qstar_full, baselines):
    S, E, bus = point['S'], point['E'], point['b']
    b_arr = np.array([bus], dtype=float)
    S_arr = np.array([S], dtype=float)
    E_arr = np.array([E], dtype=float)

    unit_p_avg, unit_q_avg, times_avg, solvers_avg, inacc_avg = _compute_schedule(
        avg_entry, S, E, bus, loss_table, base_p_sum, PM.AVG_DAYS
    )
    unit_p_peak, unit_q_peak, times_peak, solvers_peak, inacc_peak = _compute_schedule(
        peak_entry, S, E, bus, loss_table, base_p_sum, PM.PEAK_DAYS
    )

    unit_p_lp = dict(unit_p_avg, **unit_p_peak)
    unit_q_lp = dict(unit_q_avg, **unit_q_peak)

    result_b = _reinject_and_evaluate(b_arr, S_arr, E_arr, unit_p_lp, unit_q_lp)
    j_net_b = result_b['j_net'] if not result_b.get('diverged') else None

    j_net_a = baselines['j_net_a']
    j_net_c = baselines['j_net_c']

    def _delta(x_, y_):
        return (x_ - y_) if (x_ is not None and y_ is not None) else None

    ts_rows = []
    abs_errs = []
    for s in PM.ALL_DAYS:
        for t in range(PM.TIME_STEPS):
            q_lp = float(unit_q_lp[s][0, t])
            key = (point['point_id'], s, t)
            q_star = qstar_full.get(key)
            if q_star is None:
                continue
            abs_err = abs(q_lp - q_star)
            rel_err = (abs_err / q_star) if q_star != 0.0 else ''
            abs_errs.append(abs_err)
            ts_rows.append(dict(
                method=method, M=(M if M is not None else ''),
                point_id=point['point_id'], scenario=s, t=t,
                q_lp=q_lp, q_star=q_star, abs_err=abs_err, rel_err=rel_err,
            ))

    unit_q_star = _build_qstar_unit_q(point['point_id'], qstar_full)
    n_avg_lp, sum_avg_lp = _group_stats(unit_q_lp, PM.AVG_DAYS)
    n_peak_lp, sum_peak_lp = _group_stats(unit_q_lp, PM.PEAK_DAYS)
    n_avg_star, sum_avg_star = _group_stats(unit_q_star, PM.AVG_DAYS)
    n_peak_star, sum_peak_star = _group_stats(unit_q_star, PM.PEAK_DAYS)

    return dict(
        ts_rows=ts_rows, abs_errs=abs_errs, result_b=result_b,
        j_net_a=j_net_a, j_net_b=j_net_b, j_net_c=j_net_c,
        b_minus_a=_delta(j_net_b, j_net_a), c_minus_a=_delta(j_net_c, j_net_a),
        c_minus_b=_delta(j_net_c, j_net_b),
        solve_time_avg=sum(times_avg.values()), solve_time_peak=sum(times_peak.values()),
        n_recompile_avg=_count_recompiles(times_avg), n_recompile_peak=_count_recompiles(times_peak),
        dpp_preserved_avg=avg_entry['dpp_preserved'], dpp_preserved_peak=peak_entry['dpp_preserved'],
        solvers_avg=solvers_avg, solvers_peak=solvers_peak,
        inaccurate_avg=inacc_avg, inaccurate_peak=inacc_peak,
        q_avg_lp=(n_avg_lp, sum_avg_lp), q_peak_lp=(n_peak_lp, sum_peak_lp),
        q_avg_star=(n_avg_star, sum_avg_star), q_peak_star=(n_peak_star, sum_peak_star),
    )


# ============================================================
# stdout 요약 (자동판정 없음 - 수치만 제시)
# ============================================================

def _fmt_won(v):
    return 'N/A' if v is None else f'{v:,.2f}원'


def _print_method_summary(method, M, per_point_outcomes):
    label = f"{method.upper()}" + (f" M={M}" if M is not None else "")
    section(f'방식 {label}')

    dpp_avg = per_point_outcomes[0][1]['dpp_preserved_avg']
    dpp_peak = per_point_outcomes[0][1]['dpp_preserved_peak']
    print(f"DPP 유지: solve_avg 변형={dpp_avg}, solve_peak 변형={dpp_peak}", flush=True)

    all_abs_errs = []
    signed_errs = []
    for point, outcome in per_point_outcomes:
        all_abs_errs += outcome['abs_errs']
        for row in outcome['ts_rows']:
            signed_errs.append(row['q_lp'] - row['q_star'])

        print(f"\n  {point['point_id']}:", flush=True)
        print(f"    j_net(a:Q=0)={_fmt_won(outcome['j_net_a'])}  "
              f"j_net(b:Q=q_lp)={_fmt_won(outcome['j_net_b'])}  "
              f"j_net(c:Q=q_star)={_fmt_won(outcome['j_net_c'])}", flush=True)
        print(f"    (b-a)={_fmt_won(outcome['b_minus_a'])}  "
              f"(c-a)={_fmt_won(outcome['c_minus_a'])}  "
              f"(c-b)={_fmt_won(outcome['c_minus_b'])}", flush=True)

        n_avg, sum_avg = outcome['q_avg_lp']
        n_peak, sum_peak = outcome['q_peak_lp']
        print(f"    q_lp 분포: AVG {n_avg}시각/{sum_avg:.3f} Mvar  vs  "
              f"PEAK {n_peak}시각/{sum_peak:.3f} Mvar", flush=True)

        print(f"    solve_time: avg={outcome['solve_time_avg']:.4f}초"
              f"({outcome['solve_time_avg'] / REFERENCE_SOLVE_TIME_SEC:.2f}x 기준값, "
              f"n_recompile={outcome['n_recompile_avg']}/3), "
              f"peak={outcome['solve_time_peak']:.4f}초"
              f"(n_recompile={outcome['n_recompile_peak']}/2)", flush=True)

    if all_abs_errs:
        arr = np.array(all_abs_errs)
        print(f"\n  |q_lp-q_star| 중앙값={np.median(arr):.4f} Mvar, 최댓값={np.max(arr):.4f} Mvar",
              flush=True)
    if signed_errs:
        print(f"  부호 편향(평균 q_lp-q_star) = {np.mean(signed_errs):+.4f} Mvar "
              f"(양수면 LP가 과다공급, 음수면 과소공급 경향)", flush=True)


def _print_solver_diagnostics(solver_usage, inaccurate_events, total_solves):
    section('솔버 진단 (지시서 "solver 정확도 문제" 절)')
    print(f"총 LP 호출 수 = {total_solves}", flush=True)
    for name, count in sorted(solver_usage.items(), key=lambda kv: -kv[1]):
        print(f"  선택된 솔버 {name}: {count}회 ({count / total_solves * 100:.1f}%)", flush=True)

    n_inacc = len(inaccurate_events)
    print(f"OPTIMAL_INACCURATE로 마무리된 호출 수 = {n_inacc} "
          f"({n_inacc / total_solves * 100:.1f}%)", flush=True)
    if inaccurate_events:
        by_combo = {}
        for ev in inaccurate_events:
            key = (ev['kind'], ev['method'], ev['M'])
            by_combo[key] = by_combo.get(key, 0) + 1
        print("  조합별 발생 빈도(kind, method, M) -> 횟수:", flush=True)
        for key, count in sorted(by_combo.items(), key=lambda kv: -kv[1]):
            print(f"    {key} -> {count}회", flush=True)
        print("  ⚠ 위 조합의 q_lp/j_net(b)는 근사해 기반이다 - CLARABEL/OSQP(완화된 eps) "
              "재시도까지 거친 뒤에도 남은 것이므로 신뢰도를 낮춰 해석할 것.", flush=True)


def _print_interpretation():
    section('해석 지침 (자동판정 없음 - 수치를 보고 사람이 판단할 것)')
    print(
        "- (c-a)는 'Q만의 순효과 상한'(같은 P, q_star는 근사·외삽 없는 격자탐색 기준해),\n"
        "  (b-a)는 '이 LP를 실제로 배포하면 얻는 순효과'(P도 함께 재최적화), (c-b)는 이\n"
        "  프로토타입이 기준해 대비 얼마나 못 미치는지의 격차다 - 세 값을 함께 봐야\n"
        "  '근사 오차'와 '전략 차이'(1차 실험이 섞었던 문제)를 구분할 수 있다.\n"
        "- q_lp의 AVG/PEAK 분포를 q_star의 그것과 비교할 것(실행 초반 stdout에 q_star 분포가\n"
        "  먼저 출력된다) - 여전히 AVG 쪽에 쏠려 있다면 solve_avg/solve_peak 손실항의 상대\n"
        "  스케일(특히 C_CAP_PER_MW_YR 반영이 제대로 됐는지)을 재점검할 것.\n"
        "- 오차 크기(중앙값·최댓값)뿐 아니라 부호 편향도 볼 것. PWL은 오목함수(손실 체감)를\n"
        "  구간별 시컨트(직선)로 근사하므로 실제 곡선보다 아래에 있어 Q를 과소평가하는 편향이,\n"
        "  QP는 V^2~=1 근사가 전압이 낮은 지점에서 손실을 과대평가해 Q를 과소공급하는 편향이\n"
        "  예상된다 - 실측 부호가 다르면 그 자체가 보고할 발견이다.\n"
        "- DPP가 깨지는 방식(QP)은 PSO 평가마다(수천 회) 재컴파일 비용이 들 수 있다 -\n"
        "  solve_time 배율이 크면서 DPP도 깨졌다면 이중으로 불리한 신호다(지시서: QP는\n"
        "  우선순위를 낮추되 PWL이 기준해를 못 따라갈 경우의 비교군으로 남긴다).\n"
        "- OPTIMAL_INACCURATE 비율이 높은 조합은 위 수치 자체의 신뢰도가 낮다는 뜻이니\n"
        "  '솔버 진단' 절과 함께 읽을 것.",
        flush=True,
    )


# ============================================================
# 메인
# ============================================================

def _make_path():
    hostname = socket.gethostname()
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(RESULTS_DIR, f'probe_lp_loss_proto_{hostname}_{ts}.csv')


def _write_csv(path, rows):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=TS_CSV_FIELDS, extrasaction='ignore')
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        f.flush()
        os.fsync(f.fileno())
    print(f'CSV 저장: {path}', flush=True)


if __name__ == '__main__':
    _check_env()

    selective_path = _find_latest_csv('probe_q_selective')
    if selective_path is None:
        raise FileNotFoundError(
            f"{RESULTS_DIR}(또는 {ROOT_RESULTS_DIR})에 probe_q_selective_*.csv가 없다 - "
            "probe_q_selective.py를 먼저 실행할 것(이 스크립트의 기준해 입력)."
        )
    section(f'기준해 입력: {os.path.basename(selective_path)}')
    qstar_full = _load_selective_qstar_full(selective_path)
    print(f'ALL_DAYS q_star 매칭 항목 {len(qstar_full)}개 로드', flush=True)

    net, q_scale, p_total, q_total_before = _build_net_with_pf(TARGET_PF)
    base_p, base_q = _prepare_condition(net)
    base_p_sum = float(base_p.sum())

    section('기준해 q_star의 시나리오군별 분포 (1차 실험이 놓친 것 - AVG vs PEAK)')
    for point in POINTS:
        unit_q_star = _build_qstar_unit_q(point['point_id'], qstar_full)
        n_avg, sum_avg = _group_stats(unit_q_star, PM.AVG_DAYS)
        n_peak, sum_peak = _group_stats(unit_q_star, PM.PEAK_DAYS)
        print(f"  {point['point_id']}: AVG {n_avg}시각/{sum_avg:.3f} Mvar  vs  "
              f"PEAK {n_peak}시각/{sum_peak:.3f} Mvar", flush=True)

    section('기준값 j_net(a:Q=0) / j_net(c:Q=q_star) 재구성 (probe_q_selective.py 방법 재현)')
    baselines_by_point = {}
    for point in POINTS:
        b = _compute_point_baselines(point, qstar_full)
        baselines_by_point[point['point_id']] = b
        print(f"  {point['point_id']} (bus={b['bus']}): j_net(a)={_fmt_won(b['j_net_a'])}, "
              f"j_net(c)={_fmt_won(b['j_net_c'])}", flush=True)

    section('손실 테이블 실측 (전 32버스 x ALL_DAYS 5개 x 24시간 x 경계점 5개)')
    loss_table = _measure_loss_table(net, base_p, base_q)

    all_ts_rows = []
    solver_usage = {}
    inaccurate_events = []
    total_solves = 0

    for method, M in METHODS:
        avg_entry = _build_problem_proto('avg', method, n=1, T=PM.TIME_STEPS, M=M)
        peak_entry = _build_problem_proto('peak', method, n=1, T=PM.TIME_STEPS, M=M)

        per_point_outcomes = []
        for point in POINTS:
            baselines = baselines_by_point[point['point_id']]
            outcome = _process(point, method, M, avg_entry, peak_entry, loss_table,
                                base_p_sum, qstar_full, baselines)
            per_point_outcomes.append((point, outcome))
            all_ts_rows += outcome['ts_rows']

            for kind, solvers_dict, inacc_dict in (
                ('avg', outcome['solvers_avg'], outcome['inaccurate_avg']),
                ('peak', outcome['solvers_peak'], outcome['inaccurate_peak']),
            ):
                for s, name in solvers_dict.items():
                    solver_usage[name] = solver_usage.get(name, 0) + 1
                    total_solves += 1
                    if inacc_dict[s]:
                        inaccurate_events.append(dict(
                            kind=kind, method=method, M=M,
                            point_id=point['point_id'], scenario=s,
                        ))

        _print_method_summary(method, M, per_point_outcomes)

    _restore_evaluate_state()

    _write_csv(_make_path(), all_ts_rows)
    _print_solver_diagnostics(solver_usage, inaccurate_events, total_solves)
    _print_interpretation()
