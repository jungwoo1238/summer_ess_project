"""Q(무효전력)가 만드는 손실편익(b_loss)이 실제 물리인지, case33bw 원본계통의 비정상적으로
낮은 역률(0.850, 무보상)이 만든 인공물인지 분해한다 (확인 전용, 본 파이프라인 미포함 -
scripts/probe_split.py, probe_bus_sweep_full.py와 같은 성격).

배경: dev 3 run에서 LinDistFlow 편입(부록C.6-3) 후 b_loss가 편입 전(1.9e5원, Q=0 시절)
대비 17배(3.3e6원)로 뛰었고, 총편익 대비 비중이 1.42% -> 19.0%가 됐다. CLAUDE.md 9절/C.7은
R/X가 높은 배전망에서는 무효전력의 손실저감 효과가 작을 것으로 예상하는데 정면 충돌한다.

가설: case33bw 원본 부하의 역률이 0.850(무보상)으로 비정상적으로 낮아, x*Q 항(전압강하)과
Q가 흐르며 만드는 I^2R 손실이 실제 배전망(보통 역률 0.9~0.95대로 보상됨) 대비 과장된
채로 ESS의 무효전력 지원 효과를 부풀리고 있을 가능성이 있다.

방법: 동일 (b,S,E) 3점 각각에서 4개 조건(Q 자유/강제0 x 역률 0.850(원본)/0.95(보상))을
lower_lp.solve_avg/solve_peak의 force_q_zero 경로(이미 존재 - C.6-3에서 구현됨, 부록C.4-(3)
회귀검증·8절-6 대체 용도로 만들어진 그 경로를 그대로 재사용)와 evaluate.evaluate_particle을
조합해 평가한다. evaluate.py는 force_q_zero를 인자로 받지 않으므로(3n 시그니처 고정), 이
스크립트 안에서만 evaluate.solve_avg/evaluate.solve_peak 이름을 일시적으로
functools.partial(force_q_zero=True)로 바꿔치기했다가 되돌리는 방식으로 우회한다(모듈
파일은 건드리지 않음 - evaluate._solve_unit_schedules가 참조하는 것은 evaluate 모듈
네임스페이스의 이름이므로 이렇게 해도 정확히 lower_lp의 네이티브 force_q_zero 경로가 쓰인다.
"unit_q를 사후에 0으로 덮어쓰는" 방식이 아니다 - 그 방식은 LP가 애초에 Q=0을 알고 최적화한
해가 아니라서 P조차 달라질 수 있는 부정확한 비교가 된다).

"역률 0.95 보상"은 P는 그대로 두고 Q만 스케일한다(원안처럼 Q에 상수배만 걸면 피상전력이
함께 줄어 "역률 효과"와 "부하 경감 효과"가 섞이므로, 반드시 P 고정 하에 Q만 조정 - 지시
정정본 참조). 이때 lower_lp의 LinDistFlow 전압유도항이 쓰는 버스별 무효부하
(lower_lp._TOPOLOGY['base_load_q_bus'])도 함께 갱신해야 한다 - 안 하면 LP가 여전히 원본
역률(0.850) 기준 무효부하를 가정한 채 Q를 최적화하게 되어, 실제 주입되는 AC net(역률 0.95)과
LP의 내부 가정이 어긋나는 자기모순적 비교가 된다. 이 스크립트가 lower_lp.py/evaluate.py
파일 자체를 수정하지 않고 모듈 전역(_TOPOLOGY, _NET/_BASE_P/_BASE_Q/_BASE_FLOW)만 스크립트
실행 중에 재대입하는 이유가 이것이다(scripts/ 규약상 파이프라인 파일은 순수 소비자로 남긴다).

실행: `python scripts/probe_q_value.py`  (★ 이 스크립트는 작성만 하고 실행하지 않는다 -
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
import functools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import params as PM
import evaluate
import benefits
import lower_lp
from build_net import build_net

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'results')

# 3개 통제점 (지시서 그대로 - 서로 다른 세션/시절의 대표해).
POINTS = [
    dict(point_id='P1_old_full_opt', b=15, S=0.176, E=0.419),   # 구 full 최적해 (Q=0 시절)
    dict(point_id='P2_dev_run1', b=17, S=0.303, E=0.404),       # 신 dev run1
    dict(point_id='P3_dev_run0', b=31, S=1.045, E=0.405),       # 신 dev run0
]

TARGET_PF = 0.95
BASE_PF_EXPECTED = 0.850241   # CLAUDE.md 1절 "역률 0.850241 보존" - 검증5 기준값
PF_TOL = 1e-4                 # 검증5 허용오차

LOAD_SUM_ATOL_MW = 1e-9       # 검증4 (P 고정 통제) 허용오차
LOSS_PCS_ZERO_ATOL_MWH = 1e-9   # 검증2 (force_q_zero -> loss_pcs 정확히 0) 허용오차
DECOMP_ATOL_WON = 1e-6        # 검증1 (b_arb+b_loss~=b_energy) - benefits.check_b_energy_decomposition 기본값과 동일

# 6차 세션이 보고한 값(대조용 - 서로 다른 조건을 섞었을 가능성이 있어 이 스크립트로 재검산한다).
SIXTH_SESSION_REPORTED_DELTA_WON = -4.46e6

CSV_FIELDS = [
    'point_id', 'b', 'S', 'E', 'case', 'power_factor', 'q_forced_zero',
    'j_net', 'b_energy', 'b_defer', 'b_arb', 'b_loss', 'cost',
    'loss_line_total_mwh', 'loss_pcs_total_mwh',
    'b_loss_share_of_gross',
    'v_violation', 'i_violation',
    's_total_mva', 'power_factor_actual', 'loss_base_total_mwh', 'q_scale',
]

_ORIG_SOLVE_AVG = evaluate.solve_avg
_ORIG_SOLVE_PEAK = evaluate.solve_peak


def section(title):
    print('\n' + '=' * 78, flush=True)
    print(title, flush=True)
    print('=' * 78, flush=True)


def _check_env():
    val = os.environ.get('MKL_THREADING_LAYER')
    print(f'MKL_THREADING_LAYER = {val!r}', flush=True)
    if val != 'SEQUENTIAL':
        print("경고: MKL_THREADING_LAYER가 'SEQUENTIAL'이 아닙니다 - 워커 스레드 풀 생성 "
              "시점에 무증상 종료 위험이 있습니다(CLAUDE.md 7절). "
              "`conda env config vars set MKL_THREADING_LAYER=SEQUENTIAL -n ess`로 설정하세요.",
              flush=True)


# ============================================================
# 역률 조정 net 준비 (P 고정, Q만 스케일 - 지시서 정정본)
# ============================================================

def _build_net_with_pf(target_pf):
    """target_pf=None이면 원본(역률 0.850241, 조정 없음). target_pf=float이면 P를 그대로
    두고 Q만 스케일해 그 역률을 재현한다. build_net()의 fresh 복사본 위에서만 작업하므로
    원본 net을 in-place 수정할 위험이 없다(build_net()은 매 호출마다 nw.case33bw()로
    새로 만든다 - build_net.py 확인 완료).

    반환: (net, q_scale, p_total_mw, q_total_mvar_before_scale)
    """
    net = build_net()
    p_total = float(net.load['p_mw'].sum())
    q_total = float(net.load['q_mvar'].sum())
    if target_pf is None:
        return net, 1.0, p_total, q_total
    q_scale = (p_total * np.tan(np.arccos(target_pf))) / q_total
    net.load['q_mvar'] = net.load['q_mvar'] * q_scale
    return net, float(q_scale), p_total, q_total


def _prepare_condition(net):
    """evaluate 모듈 전역(_NET/_BASE_P/_BASE_Q/_BASE_FLOW)과 lower_lp 모듈 전역
    (_TOPOLOGY의 버스별 무효부하)을 주어진 net 기준으로 갱신하고 기저(ESS 없음) 흐름을
    재계산한다. evaluate.py/lower_lp.py 파일은 건드리지 않는다 - 두 모듈 다 "고정된 건
    한 번 계산해 캐싱"(CLAUDE.md 7절) 패턴으로 지연 전역변수를 두고 있어서, 스크립트가
    그 전역변수를 직접 재대입하는 것만으로 다른 net/역률 조건을 반영시킬 수 있다.

    ★ base_load_q_bus를 갱신하지 않으면 LinDistFlow 전압유도항이 원본 역률(0.850) 기준
    무효부하를 계속 가정한 채 Q를 최적화하게 되어, 실제 주입되는 net(조정된 역률)과
    LP 내부 가정이 어긋난다 - 그러면 이 실험의 핵심 통제(Q 자유 최적화가 실제 계통의
    무효부하 수준에 맞게 반응하는가)가 깨진다.
    """
    base_p = net.load['p_mw'].to_numpy().copy()
    base_q = net.load['q_mvar'].to_numpy().copy()
    load_bus = net.load['bus'].to_numpy()

    topo = lower_lp._get_topology()   # 최초 1회 호출 시 원본 net 기준으로 캐시 생성
    p_bus = np.zeros(PM.N_BUS)
    q_bus = np.zeros(PM.N_BUS)
    np.add.at(p_bus, load_bus, base_p)
    np.add.at(q_bus, load_bus, base_q)
    topo['base_load_p_bus'] = p_bus
    topo['base_load_q_bus'] = q_bus

    evaluate._NET = net
    evaluate._BASE_P = base_p
    evaluate._BASE_Q = base_q
    evaluate._BASE_FLOW = evaluate._compute_base_flow(net, base_p, base_q)
    return base_p, base_q


def _restore_evaluate_state():
    """정리(선택) - 스크립트 종료 시 evaluate/lower_lp 모듈 전역을 원본 조건으로 되돌린다.
    이 스크립트가 유일한 소비자라면 필요 없지만, 같은 인터프리터에서 다른 코드가 이어서
    evaluate를 쓸 가능성을 방어한다."""
    evaluate._NET = None
    evaluate._BASE_P = None
    evaluate._BASE_Q = None
    evaluate._BASE_FLOW = None
    lower_lp._TOPOLOGY = None


# ============================================================
# force_q_zero 우회 (evaluate.py 미수정 - 모듈 네임스페이스의 이름만 일시 교체)
# ============================================================

def _evaluate_with_force_q(x, force_q_zero):
    """force_q_zero=True면 evaluate._solve_unit_schedules가 참조하는 solve_avg/solve_peak
    이름을 lower_lp의 네이티브 force_q_zero=True 경로로 바꿔치기한 채 평가하고, 끝나면
    반드시 원본으로 되돌린다(다음 호출이 영향받지 않도록 try/finally로 보장)."""
    if force_q_zero:
        evaluate.solve_avg = functools.partial(_ORIG_SOLVE_AVG, force_q_zero=True)
        evaluate.solve_peak = functools.partial(_ORIG_SOLVE_PEAK, force_q_zero=True)
    try:
        return evaluate.evaluate_particle(x, return_detail=True)
    finally:
        evaluate.solve_avg = _ORIG_SOLVE_AVG
        evaluate.solve_peak = _ORIG_SOLVE_PEAK


# ============================================================
# 손실 분해 (선로손실 vs PCS손실, AVG_DAYS N_WD 가중 대표일 합)
# ============================================================

def _avg_weighted_mwh(per_scenario_mw):
    """dict[scenario(AVG_DAYS 부분집합) -> ndarray(T,) MW] -> N_WD 가중 대표일 합(MWh).
    b_energy/b_loss 등이 쓰는 것과 동일한 집계 방식(CLAUDE.md 3절 표: 에너지=AVG_DAYS,
    일수가중 N_WD) - 다만 SMP를 곱하지 않은 순수 물리량(MWh) 버전이다."""
    total = 0.0
    for s in PM.AVG_DAYS:
        day_mwh = float(np.sum(per_scenario_mw[s])) * PM.DT_HOURS
        total += PM.N_WEEKDAYS[s] * day_mwh
    return total


def _blank_record(point, case_label, pf_label, force_q_zero, s_total_mva, pf_actual,
                   loss_base_total_mwh, q_scale):
    """발산 등으로 detail이 불완전할 때 채우는 자리표시 레코드(probe_split._build_record의
    diverged 분기와 같은 패턴)."""
    return dict(
        point_id=point['point_id'], b=point['b'], S=point['S'], E=point['E'],
        case=case_label, power_factor=pf_label, q_forced_zero=force_q_zero,
        j_net='', b_energy='', b_defer='', b_arb='', b_loss='', cost='',
        loss_line_total_mwh='', loss_pcs_total_mwh='',
        b_loss_share_of_gross='',
        v_violation='', i_violation='',
        s_total_mva=s_total_mva, power_factor_actual=pf_actual,
        loss_base_total_mwh=loss_base_total_mwh, q_scale=q_scale,
    )


def _run_case(point, case_label, pf_label, force_q_zero, s_total_mva, pf_actual,
              loss_base_total_mwh, q_scale):
    x = np.array([point['b'], point['S'], point['E']], dtype=float)
    detail = _evaluate_with_force_q(x, force_q_zero)

    if detail.get('diverged'):
        print(f"  {point['point_id']}/{case_label}: 발산 -> 건너뜀 ({detail.get('diverge_info')})",
              flush=True)
        return _blank_record(point, case_label, pf_label, force_q_zero, s_total_mva, pf_actual,
                              loss_base_total_mwh, q_scale), None

    unit_p_avg = {s: detail['unit_p'][s] for s in PM.AVG_DAYS}
    unit_q_avg = {s: detail['unit_q'][s] for s in PM.AVG_DAYS}
    unit_loss_pcs = benefits.loss_pcs(unit_p_avg, unit_q_avg)

    loss_pcs_by_scen = {s: unit_loss_pcs[s].sum(axis=0) for s in PM.AVG_DAYS}
    loss_line_by_scen = {s: detail['loss_ess'][s] - loss_pcs_by_scen[s] for s in PM.AVG_DAYS}

    loss_line_total_mwh = _avg_weighted_mwh(loss_line_by_scen)
    loss_pcs_total_mwh = _avg_weighted_mwh(loss_pcs_by_scen)

    b_gross = detail['b_defer'] + detail['b_energy']
    b_loss_share = detail['b_loss'] / b_gross if b_gross != 0 else float('nan')

    # ---- 검산 1: b_arb + b_loss ~= b_energy ----
    decomp_ok = benefits.check_b_energy_decomposition(
        detail['b_arb'], detail['b_loss'], detail['b_energy'], atol=DECOMP_ATOL_WON
    )
    assert decomp_ok, (
        f"{point['point_id']}/{case_label}: b_arb({detail['b_arb']:.6f})+b_loss("
        f"{detail['b_loss']:.6f}) != b_energy({detail['b_energy']:.6f})"
    )

    # ---- 검산 2: force_q_zero 케이스는 loss_pcs가 (수치적으로) 정확히 0 ----
    if force_q_zero:
        assert abs(loss_pcs_total_mwh) < LOSS_PCS_ZERO_ATOL_MWH, (
            f"{point['point_id']}/{case_label}: force_q_zero인데 "
            f"loss_pcs_total_mwh={loss_pcs_total_mwh:.3e} (삼각부등식상 Q=0이면 정확히 0이어야 함)"
        )

    record = dict(
        point_id=point['point_id'], b=point['b'], S=point['S'], E=point['E'],
        case=case_label, power_factor=pf_label, q_forced_zero=force_q_zero,
        j_net=detail['j_net'], b_energy=detail['b_energy'], b_defer=detail['b_defer'],
        b_arb=detail['b_arb'], b_loss=detail['b_loss'], cost=detail['cost'],
        loss_line_total_mwh=loss_line_total_mwh, loss_pcs_total_mwh=loss_pcs_total_mwh,
        b_loss_share_of_gross=b_loss_share,
        v_violation=detail['v_violation'], i_violation=detail['i_violation'],
        s_total_mva=s_total_mva, power_factor_actual=pf_actual,
        loss_base_total_mwh=loss_base_total_mwh, q_scale=q_scale,
    )
    return record, detail


# ============================================================
# 조건(역률) 단위 실행 - base_flow는 역률 조건당 1회만 계산해 A/B(또는 C/D)가 공유한다
# ============================================================

def _run_pf_condition(point, pf_target, case_free_label, case_zero_label,
                       reference_base_p=None):
    net, q_scale, p_total, q_total_before = _build_net_with_pf(pf_target)
    base_p, base_q = _prepare_condition(net)

    s_total_mva = float(np.hypot(base_p.sum(), base_q.sum()))
    pf_actual = float(base_p.sum() / s_total_mva)
    loss_base_total_mwh = _avg_weighted_mwh({s: evaluate._BASE_FLOW['loss'][s] for s in PM.AVG_DAYS})

    pf_label = BASE_PF_EXPECTED if pf_target is None else pf_target
    expected_pf = BASE_PF_EXPECTED if pf_target is None else pf_target
    assert abs(pf_actual - expected_pf) < PF_TOL, (
        f"{point['point_id']}/{pf_label}: power_factor_actual={pf_actual:.6f}가 기대값 "
        f"{expected_pf}±{PF_TOL}를 벗어남 (검증5)"
    )

    # ---- 검산 4: P(load_sum)는 역률 조정과 무관하게 시각별로 정확히 동일해야 함 ----
    if reference_base_p is not None:
        assert np.allclose(base_p, reference_base_p, atol=LOAD_SUM_ATOL_MW, rtol=0.0), (
            f"{point['point_id']}: 역률 조정 후 P(base_p)가 원본과 달라짐 - P 고정 통제가 깨짐"
        )

    rec_free, _ = _run_case(point, case_free_label, pf_label, False,
                             s_total_mva, pf_actual, loss_base_total_mwh, q_scale)
    rec_zero, _ = _run_case(point, case_zero_label, pf_label, True,
                             s_total_mva, pf_actual, loss_base_total_mwh, q_scale)

    return [rec_free, rec_zero], base_p


# ============================================================
# 메인
# ============================================================

def _make_path():
    hostname = socket.gethostname()
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(RESULTS_DIR, f'probe_q_value_{hostname}_{ts}.csv')


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


def _print_summary_table(records):
    section('12행 요약표 (3점 x 4케이스)')
    header = (f"{'point_id':18s} {'case':4s} {'PF':6s} {'q0':5s} "
              f"{'j_net':>13s} {'b_energy':>12s} {'b_defer':>12s} {'b_loss':>12s} "
              f"{'loss_line_MWh':>13s} {'loss_pcs_MWh':>12s} {'share%':>7s}")
    print(header, flush=True)
    print('-' * len(header), flush=True)
    for r in records:
        if r['j_net'] == '':
            print(f"{r['point_id']:18s} {r['case']:4s} {r['power_factor']:<6.4f} "
                  f"{str(r['q_forced_zero']):5s}  발산/스킵", flush=True)
            continue
        print(f"{r['point_id']:18s} {r['case']:4s} {r['power_factor']:<6.4f} "
              f"{str(r['q_forced_zero']):5s} "
              f"{r['j_net']:>13.4e} {r['b_energy']:>12.4e} {r['b_defer']:>12.4e} "
              f"{r['b_loss']:>12.4e} {r['loss_line_total_mwh']:>13.4f} "
              f"{r['loss_pcs_total_mwh']:>12.6f} {r['b_loss_share_of_gross'] * 100:>6.2f}%",
              flush=True)


def _print_interpretation(records):
    section('해석 지침 (검증3·6차 세션 값 대조)')
    by_point = {}
    for r in records:
        by_point.setdefault(r['point_id'], {})[r['case']] = r

    for point_id, cases in by_point.items():
        print(f"\n[{point_id}]", flush=True)
        if 'A' in cases and 'B' in cases and cases['A']['j_net'] != '' and cases['B']['j_net'] != '':
            delta_base = cases['A']['b_loss'] - cases['B']['b_loss']
            print(f"  (A.b_loss - B.b_loss) @ PF=0.850(원본) = {delta_base:+.2f}원  "
                  f"<- 'Q의 순 손실 기여'의 유일하게 올바른 정의(동일 b,S,E에서 Q만 켜고 끈 차이)",
                  flush=True)
            print(f"    6차 세션 보고값({SIXTH_SESSION_REPORTED_DELTA_WON:+.2e}원)과 대조: "
                  f"{'부호/자릿수 대략 일치' if delta_base * SIXTH_SESSION_REPORTED_DELTA_WON > 0 else '★부호 불일치 - 6차 값은 다른 조건을 섞었을 가능성'}",
                  flush=True)
            print(f"    양수 여부: {'양수(Q가 손실을 줄임 - 기대와 일치)' if delta_base > 0 else '음수(Q가 손실을 오히려 늘림 - 재검토 필요)'}",
                  flush=True)
        if 'C' in cases and 'D' in cases and cases['C']['j_net'] != '' and cases['D']['j_net'] != '':
            delta_target = cases['C']['b_loss'] - cases['D']['b_loss']
            print(f"  (C.b_loss - D.b_loss) @ PF=0.95(보상) = {delta_target:+.2f}원", flush=True)
            if 'A' in cases and cases['A']['j_net'] != '':
                delta_base = cases['A']['b_loss'] - cases['B']['b_loss']
                if abs(delta_base) > 1e-9 and abs(delta_target) < abs(delta_base) * 0.5:
                    print("    -> delta_target이 delta_base의 절반 미만: "
                          "'Q의 가치는 낮은 역률이 만든 인공물'이라는 가설이 지지됨.", flush=True)
                else:
                    print("    -> delta_target이 delta_base 대비 뚜렷이 작지 않음: "
                          "가설(저역률 인공물)이 지지되지 않음 - Q의 손실저감 효과는 "
                          "역률과 무관하게 실재할 가능성.", flush=True)
        if 'A' in cases and 'C' in cases and cases['A']['j_net'] != '' and cases['C']['j_net'] != '':
            print(f"  loss_base_total_mwh: A(PF=0.850)={cases['A']['loss_base_total_mwh']:.4f} MWh  "
                  f"vs  C(PF=0.95)={cases['C']['loss_base_total_mwh']:.4f} MWh  "
                  f"(감소분 {cases['A']['loss_base_total_mwh'] - cases['C']['loss_base_total_mwh']:+.4f} MWh "
                  "- 역률 보상만으로 줄어든 기저 손실. 이것이 위 delta 축소의 주 경로인지 참고)",
                  flush=True)


if __name__ == '__main__':
    _check_env()

    all_records = []
    for point in POINTS:
        section(f"통제점 {point['point_id']}: b={point['b']}, S={point['S']}, E={point['E']}")

        print('  -- 역률 0.850(원본): 케이스 A(Q자유)/B(Q=0) --', flush=True)
        recs_base, base_p_ref = _run_pf_condition(point, None, 'A', 'B')
        all_records += recs_base

        print('  -- 역률 0.95(보상, P 고정): 케이스 C(Q자유)/D(Q=0) --', flush=True)
        recs_target, _ = _run_pf_condition(point, TARGET_PF, 'C', 'D',
                                            reference_base_p=base_p_ref)
        all_records += recs_target

    _restore_evaluate_state()

    _write_csv(_make_path(), all_records)
    _print_summary_table(all_records)
    _print_interpretation(all_records)
