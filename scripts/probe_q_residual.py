"""probe_q_marginal.py가 남긴 잔여 위반 13개(PF=0.95, LHS>=0.03 - RHS 상한)의 실질 크기를
정량화한다 (확인 전용, 본 파이프라인 미포함 - probe_q_marginal.py의 직접 후속).

배경: probe_q_marginal.py 실측 - PF=0.95에서 LHS(Q_inj=0 극한의 순손실저감 한계효과 상한)가
RHS 상한(1-ETA_PCS=0.03)을 넘는 조합이 전부 summer_peak / bus 30~32 / t=16~20에서만 13개
남았고, 최대 초과폭은 2.35%였다. LHS는 **Q_inj=0에서 평가한 극값**이라 "Q를 조금이라도
주입하면 유리하다"는 뜻이지 "얼마나 주입해도 유리하다"는 뜻이 아니다(probe_q_marginal.py
모듈 docstring: "Q_flow가 줄어 체감하므로 이 값이 상한"). 이 스크립트는 그 이득 구간이
Q_inj 크기로 얼마나 좁은지(=실무적으로 의미 있는 크기인지) 직접 조류계산으로 측정한다.

방법: probe_q_marginal.py의 토폴로지/LHS 계산 헬퍼(_build_parent_tracking, _path_lines,
_line_r_pu, RHS_MAX)를 그대로 재사용한다. 대상 13개 조합을 포함하는 격자
(bus∈{30,31,32}) x (t∈{16..20}) = 15개 조합 각각에서, 실제로 그 버스에 sgen을 하나 만들어
Q_inj∈{0,0.01,0.02,0.05,0.1,0.2,0.5} Mvar를 주입하며 조류계산을 다시 돌리고, **주입 후
상태**의 res_line/res_bus 값으로 LHS를 재계산한다(공식은 probe_q_marginal.py와 동일 -
2*Sum_{e in path(bus)} r_e*Q_flow(e)/V_e^2 - 다만 이제 Q_flow(e)/V_e가 Q_inj=0이 아닌
실제 주입 후 조류계산 결과다). P_inj=0으로 고정한다(무효전력 단독 효과를 보려는 것이지
유효전력과 결합한 ESS 전체 운전점을 보려는 게 아니다 - RHS 쪽도 P_inj=0이면
Q_inj/sqrt(P_inj^2+Q_inj^2)=1(0으로 나누는 것 아님 - Q_inj>0이라 분모>0)이 되어 RHS가
전 Q_inj에서 상한(0.03)으로 고정되므로, "LHS_after vs 상수 0.03" 비교가 그대로 유효하다).

sgen 부호 규약: probe_q_marginal.py가 확인한 것과 동일하게(lower_lp.py의 Baran-Wu
load-positive 정식화, `netinj_q = load_q_bus - bus_onehot.T @ Q`), sgen의 q_mvar 양수가
"발전측 관례의 무효전력 공급"이며 그 버스 하류로 흘러야 할 무효조류를 그만큼 대신 공급해
path(bus) 위 상류 Q_flow를 줄인다 - 즉 Q_inj>0을 그대로 net.sgen.q_mvar에 넣으면 된다.

실행: `python scripts/probe_q_residual.py`  (★ 이 스크립트는 작성만 하고 실행하지 않는다 -
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

import pandapower as pp

import params as PM
import evaluate

from probe_q_value import (
    TARGET_PF,
    _build_net_with_pf,
    _prepare_condition,
    section,
    _check_env,
)
from probe_q_marginal import (
    RHS_MAX,
    _build_parent_tracking,
    _path_lines,
    _line_r_pu,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ★ scripts/results (프로젝트 루트 results/가 아니다) - probe_q_marginal.py와 동일 관례를
# 따른다(루트 results/는 main.py 본실험 전용, 이 스크립트는 그 결과의 후속 확인용).
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

TARGET_BUSES = [30, 31, 32]
TARGET_SCENARIO = 'summer_peak'
TARGET_HOURS = [16, 17, 18, 19, 20]
Q_INJ_VALUES = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]  # Mvar, 오름차순(보간 전제)

CSV_FIELDS = ['bus', 't', 'q_inj', 'lhs_after', 'q_flow_sum_after', 'vm_pu_at_bus']


# ============================================================
# 주입 후 LHS 재계산 (probe_q_marginal.py와 동일 공식, 입력만 주입 후 상태)
# ============================================================

def _lhs_after_injection(net, bus, path_lines, r_pu, from_bus_of):
    """path(bus) 위 분기들의 **주입 후** q_from_mvar/from_bus vm_pu로 LHS를 재계산한다.
    공식은 probe_q_marginal.py의 LHS와 완전히 동일하다 - 그 스크립트는 Q_inj=0 상태에서만
    평가했고, 이 함수는 동일 공식을 임의의(주입 후) 상태에 그대로 적용할 뿐이다."""
    lhs_sum = 0.0
    q_flow_sum = 0.0
    for lidx in path_lines:
        q_from_mvar = float(net.res_line.at[lidx, 'q_from_mvar'])
        v_from_pu = float(net.res_bus.at[from_bus_of[lidx], 'vm_pu'])
        lhs_sum += r_pu[lidx] * (q_from_mvar / PM.S_BASE_MVA) / (v_from_pu ** 2)
        q_flow_sum += q_from_mvar
    return 2.0 * lhs_sum, q_flow_sum


# ============================================================
# 선형보간으로 LHS가 RHS_MAX 아래로 내려가는 Q_inj 임계값 추정
# ============================================================

def _interp_threshold(curve, target):
    """curve: [(q_inj, lhs_after), ...] q_inj 오름차순(Q_INJ_VALUES 순서 그대로 전달할 것).
    반환: (threshold_mvar 또는 None, status)
      status='already_below'   - Q_inj=0에서 이미 target 미만(애초에 13개 후보가 아니었어야
                                  할 조합이 섞였다는 뜻 - 격자가 13개를 포함하는 15개라 이런
                                  경우가 2건 있을 수 있음, 버그 아님).
      status='crossed'         - 샘플링 구간 안에서 target을 아래로 가로지름(threshold 유효값).
      status='never_crossed'   - 샘플링 최대 Q_inj(0.5)까지도 target 이상 유지(임계값이 0.5
                                  Mvar보다 큼 - 이 스크립트의 격자 밖).
    """
    lhs0 = curve[0][1]
    if lhs0 < target:
        return None, 'already_below'
    for i in range(len(curve) - 1):
        q0, l0 = curve[i]
        q1, l1 = curve[i + 1]
        if l0 >= target > l1:
            frac = (l0 - target) / (l0 - l1)
            return q0 + frac * (q1 - q0), 'crossed'
    return None, 'never_crossed'


def _trapz_excess(curve, target, upper_bound_q):
    """Sum_{0<=q<=upper_bound_q} (lhs_after(q)-target) dq 를 사다리꼴로 근사한다(참고용 상한
    - "그 구간에서의 최대 누적 이득을 대략 추정" 지시 항목). curve는 Q_inj 오름차순.
    upper_bound_q(threshold 또는 없으면 샘플링 최댓값)까지만 적분한다 - 그 너머는 이미
    lhs<target이라 적분 기여가 음수이므로 포함하면 "상한"이 아니게 된다.

    단위: lhs_after-target은 MW/Mvar(probe_q_marginal.py 모듈독스트링의 단위 규약 참조),
    q_inj는 Mvar이므로 적분값은 MW(그 시각에 Q_inj를 0에서 upper_bound_q까지 올릴 때 순
    손실저감이 PCS손실증분을 초과하는 부분의 상한, 그 시각 한 시간 동안의 순간전력 관점).
    """
    total = 0.0
    for i in range(len(curve) - 1):
        q0, l0 = curve[i]
        q1, l1 = curve[i + 1]
        if q0 >= upper_bound_q:
            break
        q1_clamped = min(q1, upper_bound_q)
        if q1_clamped <= q0:
            continue
        # 구간 끝점이 잘렸으면 그 지점의 lhs를 선형보간
        if q1_clamped < q1:
            l1_clamped = l0 + (l1 - l0) * (q1_clamped - q0) / (q1 - q0)
        else:
            l1_clamped = l1
        avg_excess = ((l0 - target) + (l1_clamped - target)) / 2.0
        total += avg_excess * (q1_clamped - q0)
    return total


# ============================================================
# 메인 스캔
# ============================================================

def _run_scan():
    net, q_scale, p_total, q_total_before = _build_net_with_pf(TARGET_PF)
    base_p, base_q = _prepare_condition(net)

    parent_of, line_of = _build_parent_tracking(net)
    r_pu = _line_r_pu(net)
    from_bus_of = {int(idx): int(net.line.at[idx, 'from_bus']) for idx in r_pu}

    path_cache = {b: _path_lines(b, parent_of, line_of) for b in TARGET_BUSES}

    # sgen 1개만 만들고 이후 bus/q_mvar만 갱신한다(evaluate._ensure_sgens와 동일 패턴 -
    # CLAUDE.md 7절 "속도최적화: sgen drop/create 금지, 값만 갱신").
    pp.create_sgen(net, bus=TARGET_BUSES[0], p_mw=0.0, q_mvar=0.0, name='probe_q_residual')
    sgen_idx = net.sgen.index[0]

    profile = PM.LOAD[TARGET_SCENARIO]

    rows = []
    curves = {}   # (bus,t) -> [(q_inj, lhs_after), ...]

    for bus in TARGET_BUSES:
        for t in TARGET_HOURS:
            scale = profile[t]
            curve = []
            for q_inj in Q_INJ_VALUES:
                net.load['p_mw'] = base_p * scale
                net.load['q_mvar'] = base_q * scale
                net.sgen.at[sgen_idx, 'bus'] = bus
                net.sgen.at[sgen_idx, 'p_mw'] = 0.0
                net.sgen.at[sgen_idx, 'q_mvar'] = float(q_inj)

                ok = evaluate._run_pf_with_retry(net)
                if not ok:
                    raise RuntimeError(
                        f'조류계산 발산: bus={bus} t={t} q_inj={q_inj} - 정상 범위에서 '
                        '비정상이므로 그대로 보고한다.'
                    )

                lhs_after, q_flow_sum = _lhs_after_injection(
                    net, bus, path_cache[bus], r_pu, from_bus_of
                )
                vm_at_bus = float(net.res_bus.at[bus, 'vm_pu'])

                rows.append(dict(bus=bus, t=t, q_inj=q_inj, lhs_after=lhs_after,
                                  q_flow_sum_after=q_flow_sum, vm_pu_at_bus=vm_at_bus))
                curve.append((float(q_inj), lhs_after))

            curves[(bus, t)] = curve

    return rows, curves


# ============================================================
# stdout 요약 (자동판정 없음 - 수치만 제시)
# ============================================================

def _print_summary(curves):
    section('(bus,t)별 LHS-vs-Q_inj 곡선 및 RHS_MAX(0.03) 하향교차 임계값')

    total_upper_bound_won = 0.0
    header = f"{'q_inj':>8s}  {'lhs_after':>12s}"

    for (bus, t), curve in curves.items():
        print(f"\n[bus={bus}, t={t}]", flush=True)
        print(header, flush=True)
        for q_inj, lhs in curve:
            flag = ' <RHS_MAX' if lhs < RHS_MAX else ''
            print(f"{q_inj:>8.3f}  {lhs:>12.6f}{flag}", flush=True)

        threshold, status = _interp_threshold(curve, RHS_MAX)
        if status == 'already_below':
            print(f"  이미 Q_inj=0에서 LHS<RHS_MAX (이 (bus,t)는 애초에 13개 위반 목록에 "
                  f"없었을 가능성 - 격자가 13개를 포함하는 15개 조합이라 발생 가능)", flush=True)
            continue
        if status == 'never_crossed':
            print(f"  Q_inj={Q_INJ_VALUES[-1]} Mvar까지도 LHS>=RHS_MAX 유지 - 임계값이 이 "
                  f"스크립트의 샘플링 범위(<=0.5 Mvar) 밖에 있다", flush=True)
            upper_bound_q = Q_INJ_VALUES[-1]
        else:
            print(f"  선형보간 임계값: Q_inj={threshold:.6f} Mvar에서 LHS가 RHS_MAX(0.03) "
                  f"아래로 교차", flush=True)
            upper_bound_q = threshold

        # 참고용 상한: (LHS-RHS_MAX)를 Q_inj=0~upper_bound_q 구간에서 적분(사다리꼴).
        excess_mw = _trapz_excess(curve, RHS_MAX, upper_bound_q)
        smp_won_per_mwh = float(PM.SMP_PER_MWH[TARGET_SCENARIO][t])
        excess_won_this_hour = excess_mw * smp_won_per_mwh * PM.DT_HOURS
        total_upper_bound_won += excess_won_this_hour
        print(f"  참고용 상한: Sum(LHS-RHS_MAX)dQ_inj = {excess_mw:.6f} MW, "
              f"x SMP({smp_won_per_mwh:.2f}원/MWh) x Δt(1h) = {excess_won_this_hour:+.2f}원 "
              f"(이 시각 1회 기준 - 정확한 값 아님, 조류계산 기반 재평가 필요)", flush=True)

    print(f"\n15개 (bus,t) 조합의 참고용 상한 합계 = {total_upper_bound_won:+.2f}원 "
          f"('시각 수'만큼 곱해 합산한 값 - summer_peak 대표일 1회분 상한이며 연간 가중치는 "
          f"적용하지 않았다)", flush=True)


def _print_interpretation():
    section('해석 지침 (자동판정 없음 - 수치를 보고 사람이 판단할 것)')
    print(
        "- 임계 Q_inj가 작을수록(예: 0.05 Mvar 미만) 이득 구간이 ESS 정격(S~0.1~2.4 MVA) "
        "대비 작은 크기라는 뜻이다 - 실무적 의미는 그 상대적 크기를 보고 판단할 것.\n"
        "- 위 '참고용 상한'(원)은 (LHS-RHS_MAX)를 Q_inj에 대해 적분한 값에 SMP와 Δt를 곱한 "
        "것으로, 실제 j_net 기여치가 아니라 그 위쪽 경계다 - 정확한 값은 해당 (b,S,E)를 "
        "조류계산으로 직접 평가해야 한다(probe_q_sensitivity.py 스타일).\n"
        "- never_crossed로 표시된 (bus,t)는 이 스크립트의 샘플링 범위(Q_inj<=0.5 Mvar) 안에서 "
        "결론이 나지 않았다는 뜻이지 '위반이 크다'는 뜻이 아니다 - 격자를 넓혀 재실행해야 판단 "
        "가능하다.",
        flush=True,
    )


# ============================================================
# 메인
# ============================================================

def _make_path():
    hostname = socket.gethostname()
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(RESULTS_DIR, f'probe_q_residual_{hostname}_{ts}.csv')


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

    section(f'PF={TARGET_PF} / {TARGET_SCENARIO} / bus{TARGET_BUSES} x t{TARGET_HOURS}')
    rows, curves = _run_scan()

    _write_csv(_make_path(), rows)
    _print_summary(curves)
    _print_interpretation()
