"""각 (버스,시나리오,시각)에서 독립적으로 최적 Q를 구해, 선택적(시각별로 다른 크기의) Q
프로파일이 낼 수 있는 이득의 상한을 계산한다 (확인 전용, 본 파이프라인 미포함 -
probe_q_marginal.py/probe_q_residual.py의 후속. 조류계산을 하지 않는다 - 이미 계산된 LHS
데이터와 해석식만으로 작동하는 순수 후처리 스크립트다).

## 배경 - 앞선 분석(probe_q_sensitivity.py)의 한계 정정
probe_q_sensitivity.py의 k-스윕은 LP가 이미 낸 Q 형상(24시간 프로파일 전체)을 단일 배율
k로 통째로 스케일한 **1차원 절단면**이다. 그 절단면 위에서 k=0(=Q 전부 끔)이 최적이었다는
결과는 "이 LP가 낸 그 특정 Q 형상이 나쁘다"는 뜻이지 "Q 자체가 무가치하다"는 뜻이 아니다 -
LP는 mu_volt 전압유도항을 통해서만 Q를 고르므로(probe_q_marginal.py 배경 절 참조) 손실
관점에서 최적화된 형상이 아니었을 수 있다.

핵심 관찰: 하위 LP의 시각 간 결합은 SOC 재귀식(P_ch/P_dis에만 걸림)뿐이고 Q_t는 그 결합에
전혀 등장하지 않는다(lower_lp.py의 SOC 갱신식 참조 - eta_c*P_ch, P_dis/eta_d만 있고 Q는
없다). 즉 Q_t는 시각마다 완전히 독립인 변수라, "손실 관점의 진짜 최적 Q"는 각 시각에서
개별적으로 결정된다 - 어느 한 시각의 Q가 다른 시각의 Q 선택에 아무 제약을 주지 않는다.
이 스크립트는 그 독립최적을 각 (bus,scenario,t)에서 직접 풀어, k-스윕이 놓친 "선택적 공급"
전략의 이득 상한을 계산한다.

## 계산 (probe_q_marginal.py의 LHS를 그대로 재사용, 조류계산 없음)
probe_q_marginal.py가 계산한 LHS(Q_inj=0 극한에서의 -dLoss_line/dQ_inj)를 그 극한 주변의
**1차 근사**로 확장한다(probe_q_residual.py 실측 - LHS는 Q_inj에 대해 선형에 가깝게
감소한다, 기울기 약 -0.024~-0.026 /Mvar):

    LHS(q) ~= LHS_0 - a*q                              (a>0, 기울기 크기)
    RHS(q) = (1-ETA_PCS) * q/sqrt(P^2+q^2)              (probe_q_marginal.py와 동일 정의)

최적 q*는 LHS(q*)=RHS(q*)인 지점(브렌트법 수치해 - "해석해를 억지로 유도하지 말 것"의
대상은 이 근 자체다. P=0인 경우만 예외로 직접 대입한다 - RHS(q)가 P=0에서 q>0 전체에서
상수(1-ETA_PCS)이므로 그 구간이 **실제로 완전히 선형**이라 브렌트법을 적용할 연속함수
자체가 없다(q=0에서 불연속 - 아래 _solve_q_star_and_gain 참조). 이는 "근사를 해석해로
치환"이 아니라 이미 선형인 구간을 선형으로 푸는 것이다). q*가 PCS 제약
sqrt(P^2+q*^2)<=S를 넘으면 그 경계로 자른다(경계해 - LP라면 여기서 제약이 바인딩된다는 뜻).

순 이득의 정의:
    gain(bus,s,t) = Integral_0^{q*} [LHS(q)-RHS(q)] dq * SMP[s][t] * DT_HOURS

적분 자체는 닫힌 형태로 계산한다(수치적분이 아님 - LHS가 선형이라 정확히 적분되고, RHS의
부정적분 Integral q/sqrt(P^2+q^2) dq = sqrt(P^2+q^2)도 표준 결과다. q*를 구하는 "근"과
그 근까지의 "정적분"은 다른 문제이며, 후자를 닫힌식으로 계산하는 것은 지시서의 "해석해를
억지로 유도하지 말 것"과 상충하지 않는다 - 그 문구는 q*(비선형 방정식의 근) 자체를 억지로
대수적으로 풀지 말라는 뜻이다):
    Integral_0^{q*}(LHS_0-a*q) dq = LHS_0*q* - a*q*^2/2
    Integral_0^{q*} RHS(q) dq     = (1-ETA_PCS) * (sqrt(P^2+q*^2) - P)
  (이 두 번째 식은 P=0에서도 그대로 성립한다 - sqrt(q*^2)-0=q*이므로
   (1-ETA_PCS)*q* = RHS_MAX*q*, P=0 구간의 상수-RHS 직접적분과 정확히 일치.
   불연속점은 측도 0인 한 점(q=0)뿐이라 적분값에 영향이 없다.)

## a(기울기)의 출처
probe_q_residual.py는 bus∈{30,31,32} x t∈{16..20}(summer_peak)에서만 Q_inj를 실제로
스윕해 LHS-vs-Q_inj를 측정했다 - 그 결과 CSV가 있으면 해당 (bus,t) 조합에서는 그 실측
데이터를 최소자승 1차 회귀(np.polyfit)해 a를 직접 구한다. 그 밖의 (bus,scenario,t)
조합(대다수)은 실측이 없으므로 **대표값 A_SLOPE_REPRESENTATIVE(=0.025 /Mvar, 실측범위
0.024~0.026의 중앙값)를 쓴다** - 이는 근사이며, CSV의 a_slope 컬럼과 stdout에 각 값이
실측인지 대표값인지 구분해 남긴다(정확히는 a_slope 컬럼 값 자체로 구분 가능 - 0.025가
아니면 실측).

## 집계
AVG_DAYS(summer/winter/shoulder)만 N_WEEKDAYS 가중 합산해 "연간 이득 상한"을 만든다.
PEAK_DAYS(summer_peak/winter_peak)는 CLAUDE.md 3절 표대로 b_energy/b_loss 등 손실 편익
집계에 애초에 들어가지 않는 시나리오군이다(최대일 데이터는 평균일 통계에 이미 녹아 있어
최대일로 손실을 또 세면 이중계상 - "에너지=AVG_DAYS" 규약). 따라서 PEAK_DAYS 행의
gain_won은 CSV에는 남기되(정보 손실 방지) 연간 합산에서는 제외하고 참고용으로만 별도
출력한다.

## ★ 중요한 한계 (반드시 이 넷을 함께 읽을 것 - stdout 말미에도 반복)
1. LHS의 선형 감소 가정은 **단일 버스**에 주입한 probe_q_residual.py 실험에서 얻었다.
   여러 버스에 동시에 Q를 주입하면 각 버스의 path가 겹치는 구간(슬랙에 더 가까운 공통
   상류 선로)에서 상호작용이 생겨(그 구간의 Q_flow가 두 주입 모두의 영향을 받음), 이
   스크립트가 버스별로 "독립적으로" 계산한 이득의 단순 합은 실제보다 **과대**할 수 있다
   - 이 스크립트가 계산하는 것은 상한이지 달성 가능한 값이 아니다.
2. P를 외생 파라미터(고정 그리드)로 스윕했다. 실제로는 LP가 P와 Q를 동시에 결정하고,
   손실을 아는 LP라면 P 자체도 지금과 달라질 것이다 - 이 값은 "현재 LP가 낸 P 근방에서
   Q만 재최적화하면 얼마나 남는가"의 근사다.
3. 전압 페널티(mu_volt, LAMBDA_V)를 고려하지 않는다 - Phase 1 기저 조건에서 위반이
   0이므로(1절 VALIDATION.v_violation_total_scaled=0.0) 이 채널 자체가 지금은 없다.
4. 이 계산 전체가 "이득의 상한"이다(1차 선형근사 + 버스독립가정 + P고정 + 전압무시가
   전부 낙관적 방향으로 작용한다). 실제 달성 가능한 값은 LP에 손실 항을 넣어 재최적화하고
   조류계산으로 검증해야 한다 - 이 CSV의 q_star는 "LP가 손실을 알았다면 대략 이 근방의
   Q를 냈을 것"이라는 **기준해**로서, 향후 LP 수정 시 회귀 검증에 쓴다(추가 지시 참조).

실행: `python scripts/probe_q_optimal.py`  (★ 이 스크립트는 작성만 하고 실행하지 않는다 -
실행은 사용자가 터미널에서 직접 한다. 입력으로 scripts/results/의 가장 최근
probe_q_marginal_*.csv(필수)와 probe_q_residual_*.csv(있으면 사용)를 자동으로 찾는다.)

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
from scipy.optimize import brentq

import params as PM

from probe_q_value import section, _check_env
from probe_q_marginal import RHS_MAX  # = 1 - ETA_PCS, probe_q_marginal.py와 동일 상수 재사용

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ★ scripts/results (프로젝트 루트 results/가 아니다) - probe_q_marginal.py/probe_q_residual.py와
# 동일 관례(루트 results/는 main.py 본실험 전용).
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

PF_TARGET_LABEL = 0.95
PF_MATCH_TOL = 1e-6   # CSV 왕복(float->str->float) 오차 흡수용

# probe_q_residual.py 실측 기울기 범위(지시서 인용, -0.024~-0.026 /Mvar)의 중앙값을
# 대표값으로 쓴다 - 실측이 없는 (bus,scenario,t) 대다수에 적용되는 근사.
A_SLOPE_REPRESENTATIVE = 0.025          # /Mvar (양수 = LHS가 q에 대해 감소하는 기울기 크기)
RESIDUAL_SCENARIO = 'summer_peak'       # probe_q_residual.py가 실측한 시나리오(그 스크립트의
                                          # TARGET_SCENARIO와 동일해야 함 - 값을 다시 정의하지
                                          # 않고 여기 상수 하나로 명시)

P_VALUES = [0.0, 0.1, 0.3, 0.5, 1.0]    # MW, ESS 유효출력 가정 그리드
S_VALUES = [0.3, 1.0, 2.4]              # MVA, PCS 정격용량 가정 그리드 (2.4 = params.S_BOUNDS 상한)

J_NET_REF_LOW = 2.5e6                   # 원/년, full 실측 j_net 규모 참조 하한(CLAUDE.md 8절)
J_NET_REF_HIGH = 3.5e6                  # 원/년, 참조 상한

CSV_FIELDS = ['bus', 'scenario', 't', 'P_assumed', 'S_assumed',
              'lhs_0', 'a_slope', 'q_star', 'gain_won']


# ============================================================
# 입력 CSV 로딩 (probe_q_marginal.py/probe_q_residual.py 결과 - scripts/results 최신 파일)
# ============================================================

def _find_latest_csv(prefix):
    paths = glob.glob(os.path.join(RESULTS_DIR, f'{prefix}_*.csv'))
    if not paths:
        return None
    return max(paths, key=os.path.getmtime)


def _load_marginal_rows(path):
    """probe_q_marginal.py CSV(power_factor,bus,scenario,t,lhs,q_e_sum,path_len)에서
    PF=0.95 행만 골라 (bus,scenario,t,lhs_0) 리스트로 반환한다."""
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            pf = float(r['power_factor'])
            if abs(pf - PF_TARGET_LABEL) > PF_MATCH_TOL:
                continue
            rows.append(dict(bus=int(r['bus']), scenario=r['scenario'], t=int(r['t']),
                              lhs_0=float(r['lhs'])))
    return rows


def _load_residual_slopes(path):
    """probe_q_residual.py CSV(bus,t,q_inj,lhs_after,q_flow_sum_after,vm_pu_at_bus)에서
    (bus,t)별 LHS-vs-Q_inj 최소자승 1차회귀 기울기를 구해 a(양수)로 정규화한다.
    path가 None(파일 없음)이면 빈 dict를 반환하고 전 구간 대표값으로 대체된다."""
    if path is None:
        return {}
    pts_by_bus_t = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = (int(r['bus']), int(r['t']))
            pts_by_bus_t.setdefault(key, []).append(
                (float(r['q_inj']), float(r['lhs_after']))
            )
    slopes = {}
    for key, pts in pts_by_bus_t.items():
        if len(pts) < 2:
            continue
        pts.sort()
        q = np.array([p[0] for p in pts])
        lhs = np.array([p[1] for p in pts])
        coef = np.polyfit(q, lhs, 1)   # coef[0] = d(lhs)/d(q), 음수 기대
        slopes[key] = -float(coef[0])  # a = -slope (양수로 정규화, 위 근사식과 부호 정합)
    return slopes


# ============================================================
# 시각당 독립 최적화: q* 및 그 지점까지의 (LHS-RHS) 정적분
# ============================================================

def _solve_q_star_and_gain(lhs_0, a, P, S):
    """반환: (q_star [Mvar], gain_mw [MW] - 원 환산 전, Integral_0^{q*}(LHS-RHS)dq).

    P=0 특수 처리 이유: RHS(q)=(1-ETA_PCS)*q/sqrt(P^2+q^2)는 P=0일 때 q>0 전 구간에서
    상수 RHS_MAX(=1-ETA_PCS)이고 q=0에서만 0이다(불연속). f(q)=LHS(q)-RHS(q)가 이 점에서
    끊어지므로 브렌트법(연속함수 전제)을 그대로 적용하면 안 된다 - q>0 구간은 완전히
    선형(LHS_0-a*q-RHS_MAX)이므로 그 구간만 직접 대입해 푼다.
    """
    q_max = float(np.sqrt(max(S ** 2 - P ** 2, 0.0)))
    if q_max <= 0.0:
        return 0.0, 0.0

    def f(q):
        if q <= 0.0:
            return lhs_0  # RHS(0)=0 (무주입 상태의 PCS손실은 항상 0, P 무관)
        rhs = RHS_MAX if P == 0.0 else (1.0 - PM.ETA_PCS) * q / np.sqrt(P ** 2 + q ** 2)
        return (lhs_0 - a * q) - rhs

    f0 = lhs_0
    if f0 <= 0.0:
        q_star = 0.0
    else:
        fmax = f(q_max)
        if fmax >= 0.0:
            q_star = q_max   # 경계해 - PCS 제약이 바인딩(이 P,S에서는 더 주입하고 싶어함)
        elif P == 0.0:
            # q>0 구간이 정확히 선형이므로 대수적으로 직접 푼다(근사가 아니라 그 구간의
            # 정확한 해 - 위 모듈 docstring/함수 docstring 참조).
            q_lin = (lhs_0 - RHS_MAX) / a
            q_star = float(min(max(q_lin, 0.0), q_max))
        else:
            # P>0: f는 [0,q_max]에서 연속(q->0+ 극한이 f(0)=lhs_0와 일치)이고
            # f(0)=f0>0>fmax=f(q_max)이므로 브렌트법으로 유일해를 구한다.
            q_star = float(brentq(f, 0.0, q_max))

    gain_mw = (
        (lhs_0 * q_star - 0.5 * a * q_star ** 2)
        - (1.0 - PM.ETA_PCS) * (np.sqrt(P ** 2 + q_star ** 2) - P)
    )
    return q_star, float(gain_mw)


# ============================================================
# 전체 계산: 모든 (bus,scenario,t) x (P,S) 조합
# ============================================================

def _compute_all(marginal_rows, residual_slopes):
    csv_rows = []
    annual_by_ps_bus = {}   # (P,S) -> {bus: 연간 이득(원/년), AVG_DAYS만 N_WD 가중}
    peak_ref_by_ps_bus = {}  # (P,S) -> {bus: PEAK_DAYS 비가중 합(원), 참고용 - 합산 제외}

    for r in marginal_rows:
        bus, scenario, t, lhs_0 = r['bus'], r['scenario'], r['t'], r['lhs_0']
        slope_key = (bus, t)
        if scenario == RESIDUAL_SCENARIO and slope_key in residual_slopes:
            a = residual_slopes[slope_key]
        else:
            a = A_SLOPE_REPRESENTATIVE

        smp = float(PM.SMP_PER_MWH[scenario][t])

        for P in P_VALUES:
            for S in S_VALUES:
                if P > S:
                    continue

                q_star, gain_mw = _solve_q_star_and_gain(lhs_0, a, P, S)
                gain_won = gain_mw * smp * PM.DT_HOURS

                csv_rows.append(dict(
                    bus=bus, scenario=scenario, t=t, P_assumed=P, S_assumed=S,
                    lhs_0=lhs_0, a_slope=a, q_star=q_star, gain_won=gain_won,
                ))

                ps_key = (P, S)
                if scenario in PM.AVG_DAYS:
                    bucket = annual_by_ps_bus.setdefault(ps_key, {})
                    bucket[bus] = bucket.get(bus, 0.0) + gain_won * PM.N_WEEKDAYS[scenario]
                elif scenario in PM.PEAK_DAYS:
                    bucket = peak_ref_by_ps_bus.setdefault(ps_key, {})
                    bucket[bus] = bucket.get(bus, 0.0) + gain_won  # 비가중, 참고용

    return csv_rows, annual_by_ps_bus, peak_ref_by_ps_bus


# ============================================================
# stdout 요약 (자동판정 없음 - 수치만 제시)
# ============================================================

def _print_summary(annual_by_ps_bus, peak_ref_by_ps_bus):
    section('(P,S) 조합별 연간 이득 상한 (AVG_DAYS만 N_WEEKDAYS 가중 합산, PEAK_DAYS는 참고용)')

    for ps_key in sorted(annual_by_ps_bus.keys()):
        P, S = ps_key
        by_bus = annual_by_ps_bus[ps_key]
        ranked = sorted(by_bus.items(), key=lambda kv: kv[1], reverse=True)
        top5 = ranked[:5]
        best_bus, best_gain = top5[0]

        print(f"\n[P={P} MW, S={S} MVA]", flush=True)
        print(f"  최댓값(단일 최적버스 기준 연간 이득 상한) = {best_gain:+.2f}원/년 "
              f"(bus={best_bus})", flush=True)
        print(f"  참조 j_net({J_NET_REF_LOW:.2e}~{J_NET_REF_HIGH:.2e}원/년) 대비 비율 = "
              f"{best_gain / J_NET_REF_HIGH * 100:.4f}%  ~  "
              f"{best_gain / J_NET_REF_LOW * 100:.4f}%", flush=True)
        print("  상위 5개 버스:", flush=True)
        for bus, gain in top5:
            print(f"    bus={bus:2d}: {gain:+.2f}원/년", flush=True)

        peak_by_bus = peak_ref_by_ps_bus.get(ps_key, {})
        if peak_by_bus:
            peak_bus, peak_val = max(peak_by_bus.items(), key=lambda kv: kv[1])
            print(f"  [참고, 연간 합산에서 제외] PEAK_DAYS(summer_peak+winter_peak) 비가중 "
                  f"합 최댓값 = {peak_val:+.2f}원 (bus={peak_bus}) - CLAUDE.md 3절: 손실 편익은 "
                  f"AVG_DAYS만 집계하는 규약(최대일을 또 세면 이중계상)이라 이 값은 연간 "
                  f"이득에 포함하지 않는다", flush=True)


def _print_limitations():
    section('★ 중요한 한계 (모듈 docstring과 동일 - 반드시 함께 읽을 것)')
    print(
        "1. LHS 선형감소 가정은 단일 버스 주입 실험(probe_q_residual.py)에서 얻었다 - 여러\n"
        "   버스 동시 주입 시 경로가 겹치는 상류 구간에서 상호작용이 생겨 버스별 이득의\n"
        "   단순 합은 과대추정(상한)이다.\n"
        "2. P를 외생 파라미터 그리드로 고정했다 - 실제로는 LP가 P,Q를 동시에 정한다.\n"
        "3. 전압 페널티(mu_volt/LAMBDA_V)는 고려하지 않았다(Phase 1 기저 위반이 0이라\n"
        "   현재는 무관하나, Phase 2에서는 재평가 필요).\n"
        "4. 이 값 전체가 '이득의 상한'이다 - 실제 달성 가능한 값은 LP에 손실 항을 넣어\n"
        "   재최적화하고 조류계산으로 검증해야 확정된다. 이 CSV의 q_star는 그 재최적화의\n"
        "   기준해(회귀 검증용)로서 남긴다.\n"
        "판정을 코드가 자동으로 내리지 않는다 - 위 수치와 한계를 함께 보고 사람이 판단할 것.",
        flush=True,
    )


# ============================================================
# 메인
# ============================================================

def _make_path():
    hostname = socket.gethostname()
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(RESULTS_DIR, f'probe_q_optimal_{hostname}_{ts}.csv')


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

    marginal_path = _find_latest_csv('probe_q_marginal')
    if marginal_path is None:
        raise FileNotFoundError(
            f"{RESULTS_DIR}에 probe_q_marginal_*.csv가 없다 - probe_q_marginal.py를 먼저 "
            "실행할 것(이 스크립트는 그 결과를 입력으로 쓴다)."
        )
    residual_path = _find_latest_csv('probe_q_residual')

    section(f'입력: {os.path.basename(marginal_path)}'
            + (f' + {os.path.basename(residual_path)}' if residual_path else ' (residual 없음 -'
               ' 전 구간 대표기울기 사용)'))

    marginal_rows = _load_marginal_rows(marginal_path)
    residual_slopes = _load_residual_slopes(residual_path)
    print(f'PF=0.95 행 {len(marginal_rows)}개, 실측 기울기 확보된 (bus,t) 조합 '
          f'{len(residual_slopes)}개', flush=True)

    csv_rows, annual_by_ps_bus, peak_ref_by_ps_bus = _compute_all(marginal_rows, residual_slopes)

    _write_csv(_make_path(), csv_rows)
    _print_summary(annual_by_ps_bus, peak_ref_by_ps_bus)
    _print_limitations()
