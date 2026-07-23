"""부록 C.6-2 게이트: LinDistFlow 정식화 정확도 검증 (확인 전용, 본 파이프라인 미포함 -
scripts/check_balance.py, probe_noise.py, probe_voltage.py와 같은 성격).

이 스크립트는 LP 통합이 아니다. lower_lp.py/main.py는 import하지 않는다(독립성 유지) -
LinDistFlow 정식화 자체의 정확도(전압 오차)만 순수 numpy로 재구현해 pandapower 완전 AC
조류계산과 대조한다. 통과해야 C.6-3(본체 lower_lp.py 편입)로 넘어간다. 매몰비용 없음 -
불통과여도 잃는 건 이 파일 하나, 검증된 뼈대(현 lower_lp.py)는 그대로 남는다.

이 게이트가 보증하는 것: "LinDistFlow가 나쁜 스케줄을 유도해 결과를 오염시키지 않는다"
(전압을 충분히 맞혀서). 보증하지 않는 것: "LinDistFlow 해가 최적이다" - 최적성은 편입 후에도
여전히 사후 AC 조류계산이 담당한다(2절 "정직한 단서" 구조). 즉 이 게이트는 정확도 보증이
아니라 회귀 안전판이다.

★ 부호 컨벤션 참고 (구현 중 발견, CMD 문서와의 정합화):
CMD 문서 item1은 "P_ij = Σ_{k∈하류(j)}(−load_k+P_ess,k)"(순주입, 발전 양수)라 적었지만,
헤더의 "v_j = v_i − 2(r·P_ij+x·Q_ij)"(마이너스 부호)와 그대로 결합하면 순수 부하 케이스에서
하류로 갈수록 전압이 오히려 오르는(물리적으로 틀린) 결과가 나온다. 표준 Baran-Wu LinDistFlow는
P_ij를 "하류 부하 - 하류 발전"(부하 양수, 상류에서 공급해야 하는 방향)으로 정의해야
마이너스 부호 공식이 전압강하를 올바르게 재현한다. 이 스크립트는 후자(부하 양수 컨벤션)를
쓰고, 기저(ESS 없음) 케이스에서 전압이 슬랙에서 멀어질수록 단조감소하는지로 결과 자체가
이를 실증한다(아래 실행기록의 measurement A 결과 참조).

실행: `python scripts/test_lindistflow.py`

# ------------------------------------------------------------------
# 실행 기록 (scripts/probe_noise.py, bench_workers.py와 같은 규약)
#   실행 일시: 2026-07-23
#   머신: PSL (조건: 슬랙 1.02, 데스크탑)
#   결과 요약: 1002케이스 전부 완주(발산 0, LinDistFlow v^2<0 0). 방문영역(A_base 120 +
#     B_visited 180 + C_reactive 360 = 660케이스) 글로벌 max err_v = 0.001202 pu
#     (bus17, B_visited, summer_peak t=18, bus32에 0MW 주입 - 사실상 기저 케이스와 동일).
#     1차(<=0.005) PASS, 2차(기대범위 0.001~0.003) PASS - "정식화 건강함" 판정.
#     버스별 median err_v: bus17=0.000859, bus32=0.000859 (전체 median 0.000773 대비
#     1.11배 - 균일편향, 말단 특이적 열화 없음). |P_ess| vs 오차 상관 -0.146
#     (방문영역 0~0.25MW가 총부하 8.5MW 대비 작아, 이 구간에서는 주입크기보다 기저
#     계통 위상이 오차를 지배 - 상관이 약한 것 자체가 문제 신호는 아님).
#     대칭성(B_boundary 충전 vs D_reverse): 54쌍 매칭, |오차차이| 평균 0.00118/
#     최대 0.00306 pu(개별 오차와 같은 자릿수 - 방향 비대칭 문제 없음).
#     → 판정: 게이트 통과. C.6-3(본체 lower_lp.py 편입) 착수 가능.
#
#   실행 일시(2차, 보완명령 - 방문영역 충전(E/F) + 진짜 역조류(G) 케이스 추가): 2026-07-23
#   머신: PSL
#   결과 요약: 1482케이스(A~G) 전부 완주(발산 0, v^2<0 0), 8.4초. 판정 대상이 A_base+
#     B_visited+C_reactive+E_visited_charge+F_charge_reactive(1092케이스)로 확대.
#     방문영역(통합) 글로벌 max err_v = 0.001421 pu(E_visited_charge, bus16, P=-0.25MW,
#     bus17 최대오차 - 1차 아니라 충전 케이스가 최댓값을 갱신). 1차(<=0.005)/2차(0.001~0.003)
#     둘 다 PASS 유지 - "충전을 포함해도 결론 동일". 기존(방전만, 660건) max 0.001202 대비
#     신규 충전(E+F, 432건) max 0.001421 - 1.18배(경계영역 격자(0.5~2.4MW)에서 관찰된
#     1.5~4배보다는 완만 - 방문영역 자체가 작아(≤0.25MW) 절대 영향이 제한적).
#     버스별 오차는 "균일 편향"이 아니라 슬랙->bus17 간선을 따라 단조 누적됨을 재확인
#     (bus1=0.000051 -> bus17=0.000912, 17.9배 - 손실항 누적의 정상 패턴, 국소 폭발 아님).
#     |P_ess| 상관: 방전(B_visited) -0.146(약함), 충전(E_visited_charge) +0.241(양의 상관,
#     예상대로 방전보다 뚜렷 - 충전이 부하에 더해져 |선로조류|를 키우는 효과가 이 구간에서도
#     감지됨). G_true_reverse(다중버스 동시주입) 48건 전부 vm_pu_max>1.0 달성(최대 1.1047pu,
#     feeder_2_17군·shoulder·t=4·버스당0.8MW) - 진짜 역조류 조건 확보, Phase 2 대비 완료.
#     → 판정: 게이트 통과 재확인(구멍 2개 모두 메움). C.6-3 착수 가능 결론 불변.
# ------------------------------------------------------------------
"""
import os
import sys
import time
import socket
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pandapower as pp

import params as PM
from build_net import build_net
import evaluate  # _run_pf_with_retry 재사용(발산 재시도 로직 재구현 안 함 - probe_voltage.py와 동일 패턴)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 0. pu 기준 (params에서 파생 - 숫자 하드코딩 금지, CMD 문서 "계통 pu 기준" 절)
# ============================================================
S_BASE_MVA = PM.TARGET_MVA                  # 10.0 MVA (1절 "상시운전용량 10,000kVA" 스케일 기준).
                                             # params.py에 별도 상수가 없어 TARGET_MVA에서 파생.
Z_BASE_OHM = PM.VN_KV ** 2 / S_BASE_MVA      # = 22.9^2/10 = 52.440999... Ohm
V_SLACK_SQ = PM.SLACK_VM_PU ** 2             # = 1.02^2 = 1.0404

# 판정 문턱 (C.8 근거. 1차=이상탐지 상한선, 2차=LinDistFlow 기대성능 범위)
GATE_THRESHOLD_PU = 0.005
EXPECTED_RANGE_PU = (0.001, 0.003)

# 테스트 케이스 격자 (CMD 문서 3절)
BUSES_B = [15, 16, 31, 32]                              # 최적권 4버스 (8절-5)
P_VISITED = [0.0, 0.05, 0.10, 0.176, 0.25]               # 방문영역(판정 대상)
P_BOUNDARY = [-2.4, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 2.4]  # 경계영역(기록만)
S_FIXED_C = 0.25                                         # 무효전력 케이스 S 고정값
Q_RATIOS_C = [0.5, 1.0]
BUSES_D = [15, 31]                                       # 과전압(역주입) 참고 케이스
D_VALUES = [0.5, 1.0, 2.0]

# --- 보완명령(게이트 구멍 메우기) 추가 격자 ---
P_VISITED_CHARGE = [-0.05, -0.10, -0.176, -0.25]         # (E) 방문영역 충전(부호 대칭, 0은 B_visited가 이미 커버)

G_BRANCH_BUSES = list(range(26, 33))                     # (G-1) 가지 전체: 26~32
G_FEEDER_BUSES = list(range(2, 18))                      # (G-2) 피더 광역: 2~17
G_VALUES = [0.2, 0.4, 0.8]                                # 버스당 동시주입(MW)
G_SCENARIOS = ['shoulder', 'summer']                      # 경부하 조건
SGEN_POOL_SIZE = max(len(G_BRANCH_BUSES), len(G_FEEDER_BUSES))  # 다중주입 케이스가 요구하는 최대 동시 sgen 수


def _g_hours(s):
    """경부하(최소부하) 시각 + 정오대(t=12~14) 고정 포함 - PV 덕커브 역조류는 피크가 아니라
    경부하 시간대(정오 부근)에 발생하므로 최소부하 시각만으로는 못 잡을 수 있어 명시적으로 더한다."""
    return sorted({int(np.argmin(PM.LOAD[s])), 12, 13, 14})


def section(title):
    print('\n' + '=' * 78, flush=True)
    print(title, flush=True)
    print('=' * 78, flush=True)


def _check_env():
    val = os.environ.get('MKL_THREADING_LAYER')
    print(f'MKL_THREADING_LAYER = {val!r}', flush=True)
    if val != 'SEQUENTIAL':
        print("경고: MKL_THREADING_LAYER가 'SEQUENTIAL'이 아닙니다 (CLAUDE.md 7절 참조).", flush=True)


# ============================================================
# 1. LinDistFlow 정식화 (독립 numpy 구현, cvxpy/lower_lp 불필요)
# ============================================================

def _build_topology(net):
    """방사형 트리(슬랙=bus0 루트). in_service 선로 32개만 사용(tie 5개는 build_net이 이미 제외).
    children[bus] = [(child_bus, line_idx), ...]. r_pu/x_pu는 Z_BASE_OHM으로 정규화한 pu 임피던스."""
    children = {int(b): [] for b in net.bus.index}
    r_pu = {}
    x_pu = {}
    lines = net.line[net.line['in_service']]
    for idx, row in lines.iterrows():
        i, j = int(row['from_bus']), int(row['to_bus'])
        r_pu[idx] = row['r_ohm_per_km'] * row['length_km'] / Z_BASE_OHM
        x_pu[idx] = row['x_ohm_per_km'] * row['length_km'] / Z_BASE_OHM
        children[i].append((j, idx))
    return children, r_pu, x_pu


def _topo_order(children, root):
    """전위(preorder) DFS. 부모가 항상 자손보다 먼저 나온다(역순으로 훑으면 자손이 부모보다
    먼저 처리됨 - 하류 합산에 필요)."""
    order = []
    parent_of = {root: None}
    line_of = {}
    stack = [root]
    while stack:
        u = stack.pop()
        order.append(u)
        for v, lidx in children[u]:
            parent_of[v] = u
            line_of[v] = lidx
            stack.append(v)
    return order, parent_of, line_of


def _lindistflow_v_sq(order, parent_of, line_of, r_pu, x_pu, own_p, own_q, n_bus):
    """v_j = v_i - 2*(r_ij*P_ij + x_ij*Q_ij).

    own_p/own_q: bus별 '상류에서 공급해야 하는 순부하'(부하 양수, ESS 방전은 그만큼을 뺀다 -
    own_p[bus] = load_p[bus] - ess_p[bus]). P_ij/Q_ij는 그 하류(j 포함) 전체 합.
    (스크립트 상단 docstring "★ 부호 컨벤션 참고" 참조.)
    """
    subtree_p = own_p.copy()
    subtree_q = own_q.copy()
    for u in reversed(order):
        p = parent_of[u]
        if p is not None:
            subtree_p[p] += subtree_p[u]
            subtree_q[p] += subtree_q[u]

    v_sq = np.empty(n_bus)
    v_sq[order[0]] = V_SLACK_SQ
    for u in order[1:]:
        p = parent_of[u]
        lidx = line_of[u]
        v_sq[u] = v_sq[p] - 2.0 * (r_pu[lidx] * subtree_p[u] + x_pu[lidx] * subtree_q[u])

    bad = np.where(v_sq < -1e-9)[0]
    assert bad.size == 0, (
        f'LinDistFlow v^2 < 0 발생 (버스 {bad.tolist()}) - 정식화 오류 신호. '
        '조용히 넘기지 않음(CMD 문서 item3 "음수 clip 금지").'
    )
    return np.maximum(v_sq, 0.0)  # -1e-9~0 사이의 부동소수 잡음만 0으로 (실질 음수는 위 assert가 이미 막음)


def _case_injections(case):
    """case dict -> [(bus,p_mw,q_mvar), ...] 균일 표현. 단일버스 케이스(A~D,E,F)는 기존
    ess_bus/p_mw/q_mvar 필드에서 자동 도출하고, 다중버스 케이스(G)는 'injections' 필드를
    직접 쓴다(보완명령 1절 - 정식화·판정 로직은 불변, 여기는 실행 배관만 일반화한 것)."""
    if case.get('injections') is not None:
        return case['injections']
    if case['ess_bus'] is None:
        return []
    return [(int(case['ess_bus']), float(case['p_mw']), float(case['q_mvar']))]


def _own_flow(base_load_p, base_load_q, load_bus, n_bus, scale, injections):
    """MW/Mvar 절대값을 pu로 변환해 반환한다(÷S_BASE_MVA) - r_pu/x_pu가 무차원 pu 임피던스이므로
    P/Q도 반드시 pu여야 v_j=v_i-2(r_pu*P_pu+x_pu*Q_pu) 재귀식이 성립한다(MW 그대로 넣으면
    S_BASE_MVA배만큼 과도하게 커져 v^2가 음수로 튄다 - 초기 구현에서 실측됨).

    injections: [(bus,p_mw,q_mvar), ...] - 0개(기저)·1개(단일기)·다수개(G, 다중버스 동시주입) 전부
    같은 코드경로로 처리한다(리스트 길이 0/1일 때는 이전 단일버스 버전과 수학적으로 동일)."""
    own_p = np.zeros(n_bus)
    own_q = np.zeros(n_bus)
    np.add.at(own_p, load_bus, base_load_p * scale)
    np.add.at(own_q, load_bus, base_load_q * scale)
    for bus, p, q in injections:
        own_p[bus] -= p
        own_q[bus] -= q
    return own_p / S_BASE_MVA, own_q / S_BASE_MVA


def _run_ac(net, base_load_p, base_load_q, scale, injections, sgen_pool):
    """ground truth: pandapower 완전 AC 조류계산. 부호 규약(item7): sgen 발전(주입) 양수 =
    ESS 방전 = 주입 = sgen 양수. LinDistFlow own_p도 동일 부호(ESS 주입만큼 순부하에서 뺌).

    sgen_pool: 미리 만들어 둔 sgen 인덱스 풀(길이 >= 케이스가 요구하는 최대 동시주입 수).
    이번 케이스가 쓰지 않는 나머지 sgen은 bus=1,p=q=0으로 비활성화한다(전기적으로 무해 -
    이미 기저(ESS없음) 케이스가 검증된 방식과 동일)."""
    net.load['p_mw'] = base_load_p * scale
    net.load['q_mvar'] = base_load_q * scale
    n = len(injections)
    assert n <= len(sgen_pool), f'sgen_pool 부족: 필요 {n}, 보유 {len(sgen_pool)}'
    for k, idx in enumerate(sgen_pool):
        if k < n:
            bus, p, q = injections[k]
            net.sgen.at[idx, 'bus'] = int(bus)
            net.sgen.at[idx, 'p_mw'] = float(p)
            net.sgen.at[idx, 'q_mvar'] = float(q)
        else:
            net.sgen.at[idx, 'bus'] = 1
            net.sgen.at[idx, 'p_mw'] = 0.0
            net.sgen.at[idx, 'q_mvar'] = 0.0
    ok = evaluate._run_pf_with_retry(net)
    if not ok:
        return None
    return net.res_bus.vm_pu.to_numpy().copy()


# ============================================================
# 2. 테스트 케이스 집합 (CMD 문서 3절)
# ============================================================

def _scenario_hour_pairs():
    """각 시나리오의 피크시각 + t=18(1절 bus17 최악 강하점) 고정 포함. 하드코딩 대신
    PM.LOAD에서 argmax로 도출."""
    pairs = []
    seen = set()
    for s in PM.ALL_DAYS:
        peak_t = int(np.argmax(PM.LOAD[s]))
        for t in sorted({peak_t, 18}):
            if (s, t) not in seen:
                seen.add((s, t))
                pairs.append((s, t))
    return pairs


def build_cases():
    cases = []

    # (A) 기저(ESS 없음) - 필수, 판정 대상. 5개 시나리오 x 24시각 전부.
    for s in PM.ALL_DAYS:
        for t in range(PM.TIME_STEPS):
            cases.append(dict(section='A_base', judged=True, scenario=s, hour=t,
                               ess_bus=None, p_mw=0.0, q_mvar=0.0, direction='base'))

    sh_pairs = _scenario_hour_pairs()

    # (B) 유효전력 주입 - 방문영역(판정) + 경계영역(기록만)
    for s, t in sh_pairs:
        for bus in BUSES_B:
            for p in P_VISITED:
                direction = 'discharge' if p > 0 else 'zero'
                cases.append(dict(section='B_visited', judged=True, scenario=s, hour=t,
                                   ess_bus=bus, p_mw=p, q_mvar=0.0, direction=direction))
            for p in P_BOUNDARY:
                direction = 'discharge' if p > 0 else 'charge'
                cases.append(dict(section='B_boundary', judged=False, scenario=s, hour=t,
                                   ess_bus=bus, p_mw=p, q_mvar=0.0, direction=direction))

    # (C) 무효전력 주입 - 필수, 판정 대상. B_visited 격자에 Q=q*sqrt(S_FIXED_C^2-P^2) 얹음.
    for s, t in sh_pairs:
        for bus in BUSES_B:
            for p in P_VISITED:
                q_mag = float(np.sqrt(max(S_FIXED_C ** 2 - p ** 2, 0.0)))
                for qr in Q_RATIOS_C:
                    cases.append(dict(section='C_reactive', judged=True, scenario=s, hour=t,
                                       ess_bus=bus, p_mw=p, q_mvar=qr * q_mag,
                                       direction='discharge+Q'))

    # (D) 과전압(역주입) 참고 - 판정 제외, 기록만. B_boundary 충전(음수)과의 대칭성 점검용.
    for s, t in sh_pairs:
        for bus in BUSES_D:
            for p in D_VALUES:
                cases.append(dict(section='D_reverse', judged=False, scenario=s, hour=t,
                                   ess_bus=bus, p_mw=p, q_mvar=0.0, direction='reverse_gen'))

    # (E) 방문영역 충전 - 필수, 판정 대상. B_visited와 완전히 동일한 (시나리오,시각) 조합·버스,
    # P만 부호 대칭(0은 B_visited가 이미 커버하므로 제외). 보완명령 "구멍1" 해소.
    for s, t in sh_pairs:
        for bus in BUSES_B:
            for p in P_VISITED_CHARGE:
                cases.append(dict(section='E_visited_charge', judged=True, scenario=s, hour=t,
                                   ess_bus=bus, p_mw=p, q_mvar=0.0, direction='charge'))

    # (F) 방문영역 충전 + 무효전력 - 필수, 판정 대상. C_reactive와 동일한 Q 생성규칙을 (E)에 적용.
    for s, t in sh_pairs:
        for bus in BUSES_B:
            for p in P_VISITED_CHARGE:
                q_mag = float(np.sqrt(max(S_FIXED_C ** 2 - p ** 2, 0.0)))
                for qr in Q_RATIOS_C:
                    cases.append(dict(section='F_charge_reactive', judged=True, scenario=s, hour=t,
                                       ess_bus=bus, p_mw=p, q_mvar=qr * q_mag,
                                       direction='charge+Q'))

    # (G) 진짜 역조류/과전압 - 판정 제외, 기록만(Phase 2 대비). 다중버스 동시주입으로 간선조류를
    # 실제 역전시킨다(단일버스로는 D_reverse가 이미 실패 - 보완명령 "구멍2" 해소).
    for s in G_SCENARIOS:
        for t in _g_hours(s):
            for group_name, buses in [('branch_26_32', G_BRANCH_BUSES), ('feeder_2_17', G_FEEDER_BUSES)]:
                for p in G_VALUES:
                    injections = [(b, p, 0.0) for b in buses]
                    cases.append(dict(section='G_true_reverse', judged=False, scenario=s, hour=t,
                                       ess_bus=None, p_mw=p, q_mvar=0.0, direction='reverse_multi',
                                       injections=injections, injection_group=group_name,
                                       n_injections=len(buses),
                                       injection_bus_list=','.join(map(str, buses))))

    return cases


# ============================================================
# 3. 실행 루프
# ============================================================

def run_all_cases():
    net = build_net()
    base_load_p = net.load['p_mw'].to_numpy().copy()
    base_load_q = net.load['q_mvar'].to_numpy().copy()
    load_bus = net.load['bus'].to_numpy()
    n_bus = PM.N_BUS

    children, r_pu, x_pu = _build_topology(net)
    order, parent_of, line_of = _topo_order(children, PM.SLACK_BUS)

    sgen_pool = [pp.create_sgen(net, bus=1, p_mw=0.0, q_mvar=0.0, name=f'LDF_TEST_{k}')
                 for k in range(SGEN_POOL_SIZE)]

    cases = build_cases()
    print(f'총 케이스 수: {len(cases)}', flush=True)

    rows = []
    diverged_cases = []
    negative_vsq_cases = []
    t0 = time.time()

    for ci, case in enumerate(cases):
        if ci > 0 and ci % 200 == 0:
            print(f'  진행: {ci}/{len(cases)} ({time.time() - t0:.1f}s)', flush=True)

        scale = PM.LOAD[case['scenario']][case['hour']]
        injections = _case_injections(case)
        vm_ac = _run_ac(net, base_load_p, base_load_q, scale, injections, sgen_pool)
        if vm_ac is None:
            diverged_cases.append(case)
            continue

        own_p, own_q = _own_flow(base_load_p, base_load_q, load_bus, n_bus, scale, injections)
        try:
            v_sq = _lindistflow_v_sq(order, parent_of, line_of, r_pu, x_pu, own_p, own_q, n_bus)
        except AssertionError as e:
            negative_vsq_cases.append(dict(case, error=str(e)))
            continue

        v_ldf = np.sqrt(v_sq)
        err_sq = np.abs(v_sq - vm_ac ** 2)
        err_v = np.abs(v_ldf - vm_ac)
        err_sq[PM.SLACK_BUS] = 0.0
        err_v[PM.SLACK_BUS] = 0.0  # 슬랙 제외(양쪽 고정, item2 "판정 제외")

        mask = np.ones(n_bus, dtype=bool)
        mask[PM.SLACK_BUS] = False
        case_max_err_v = float(np.max(err_v[mask]))
        case_max_err_v_bus = int(np.argmax(np.where(mask, err_v, -1.0)))
        case_max_err_sq = float(np.max(err_sq[mask]))

        row = dict(case)
        row.pop('injections', None)  # bus 리스트는 injection_bus_list 문자열 컬럼으로 이미 남음 - 원시 튜플리스트는 CSV에 부적합
        row.update(dict(
            case_id=ci,
            case_max_err_v=case_max_err_v,
            case_max_err_v_bus=case_max_err_v_bus,
            case_max_err_sq=case_max_err_sq,
            err_v_bus17=float(err_v[17]),
            err_v_bus32=float(err_v[32]),
            err_v_injection_bus=(float(err_v[int(case['ess_bus'])])
                                  if case['ess_bus'] is not None else np.nan),
            vm_pu_max=float(vm_ac.max()),  # 보완명령 item G: 과전압(>1.0) 달성 여부 판정용
        ))
        for b in range(n_bus):
            row[f'errv_bus{b:02d}'] = float(err_v[b])
            row[f'errsq_bus{b:02d}'] = float(err_sq[b])
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f'완료: {len(df)}행 ({time.time() - t0:.1f}s), 발산 {len(diverged_cases)}건, '
          f'v^2<0 {len(negative_vsq_cases)}건', flush=True)
    return df, diverged_cases, negative_vsq_cases, parent_of


# ============================================================
# 4. 오차곡선 CSV (item5-3, 보완명령 3절: x=p_mw 부호 있는 값 그대로 -0.25...+2.4,
#    충전/방전/역주입이 한 좌표계에 - G는 단일버스 P가 아니므로 제외)
# ============================================================

def build_curve_csv(df):
    subset = df[df['section'].isin(
        ['B_visited', 'B_boundary', 'D_reverse', 'E_visited_charge']
    )].copy()
    subset['abs_p_mw'] = subset['p_mw'].abs()
    cols = ['case_id', 'section', 'direction', 'scenario', 'hour', 'ess_bus', 'p_mw', 'abs_p_mw',
            'case_max_err_v', 'case_max_err_v_bus', 'err_v_injection_bus']
    return subset[cols].sort_values(['ess_bus', 'p_mw', 'direction'])


# ============================================================
# main
# ============================================================

def main():
    _check_env()
    section('부록 C.6-2 게이트: LinDistFlow 정식화 정확도 검증')
    print(f'S_BASE_MVA = {S_BASE_MVA} (params.TARGET_MVA에서 파생)', flush=True)
    print(f'Z_BASE_OHM = {Z_BASE_OHM:.6f} Ohm (VN_KV={PM.VN_KV}^2 / S_BASE_MVA)', flush=True)
    print(f'V_SLACK_SQ = {V_SLACK_SQ} (SLACK_VM_PU={PM.SLACK_VM_PU}^2)', flush=True)

    df, diverged, neg_vsq, parent_of = run_all_cases()

    if diverged:
        print(f'\n★ 발산 케이스 {len(diverged)}건 (예시 최대 5개):', flush=True)
        for c in diverged[:5]:
            print(f'  {c}', flush=True)
    if neg_vsq:
        print(f'\n★ LinDistFlow v^2<0 케이스 {len(neg_vsq)}건 (예시 최대 5개):', flush=True)
        for c in neg_vsq[:5]:
            print(f'  {c}', flush=True)

    if len(df) == 0:
        print('★ 유효 케이스가 하나도 없음 - 판정 불가. 위 발산/음수v^2 로그를 먼저 확인할 것.',
              flush=True)
        return

    section('판정 (C.8 - 방문영역 A_base+B_visited+C_reactive+E_visited_charge+F_charge_reactive 대상)')
    judged = df[df['judged']]
    worst = judged.loc[judged['case_max_err_v'].idxmax()]
    global_max_err_v = float(worst['case_max_err_v'])
    global_max_err_sq = float(judged['case_max_err_sq'].max())

    print(f'방문영역(통합) 글로벌 max err_v  = {global_max_err_v:.6f} pu', flush=True)
    print(f'  발생 위치: section={worst["section"]} scenario={worst["scenario"]} '
          f'hour={worst["hour"]} ess_bus={worst["ess_bus"]} p_mw={worst["p_mw"]} '
          f'q_mvar={worst["q_mvar"]:.4f} -> 최대오차 버스={worst["case_max_err_v_bus"]}', flush=True)
    print(f'방문영역(통합) 글로벌 max err_sq = {global_max_err_sq:.6e} pu^2', flush=True)

    # 보완명령 2절: 직전 실행(방전만 판정 - A_base+B_visited+C_reactive)과 이번 신규(충전
    # 판정 - E_visited_charge+F_charge_reactive)를 나란히 비교 - "구멍1"의 크기를 정량화.
    old_judged = judged[judged['section'].isin(['A_base', 'B_visited', 'C_reactive'])]
    new_charge_judged = judged[judged['section'].isin(['E_visited_charge', 'F_charge_reactive'])]
    old_max = float(old_judged['case_max_err_v'].max()) if len(old_judged) else float('nan')
    new_max = float(new_charge_judged['case_max_err_v'].max()) if len(new_charge_judged) else float('nan')
    print(f'\n[비교] 기존 판정(방전만, {len(old_judged)}건) max err_v = {old_max:.6f} pu', flush=True)
    print(f'[비교] 신규 충전 판정(E+F, {len(new_charge_judged)}건) max err_v = {new_max:.6f} pu '
          f'(기존 대비 {new_max / old_max:.2f}배)' if old_max else '', flush=True)

    verdict1 = 'PASS' if global_max_err_v <= GATE_THRESHOLD_PU else 'FAIL'
    print(f'\n1차(상한선) 판정: max err_v {global_max_err_v:.6f} <= {GATE_THRESHOLD_PU} pu ? '
          f'-> {verdict1}', flush=True)

    lo, hi = EXPECTED_RANGE_PU
    if global_max_err_v <= hi:
        verdict2 = f'PASS - 기대범위[{lo},{hi}] 이내, 정식화 건강함'
    elif global_max_err_v <= GATE_THRESHOLD_PU:
        verdict2 = f'경보 - 1차는 통과했으나 기대범위[{lo},{hi}] 초과(기대보다 나쁨, 구조점검 필요)'
    else:
        verdict2 = '해당없음(1차 이미 FAIL)'
    print(f'2차(실질 합격선) 판정: {verdict2}', flush=True)

    section('오차 구조 진단 (C.8 라 - 기계적 폐기 금지, 애매하면 구조부터 볼 것)')
    # ★ 보완명령 2절 정정: "버스별 median이 균일(비율~1.1배)이라 안전"이 아니라, 오차는
    # 슬랙(bus0)에서 bus17로 이어지는 간선을 따라 단조 누적된다(버려진 손실항이 선로를 따라
    # 쌓이는 정상 패턴) - 아래는 그 간선 경로를 직접 출력한다(하드코딩 아님, parent_of로 역추적).
    trunk = [17]
    while parent_of[trunk[-1]] is not None:
        trunk.append(parent_of[trunk[-1]])
    trunk = list(reversed(trunk))  # root(0) -> ... -> 17

    ess_judged = judged[judged['ess_bus'].notna()]
    bus_err_cols = [f'errv_bus{b:02d}' for b in range(PM.N_BUS) if b != PM.SLACK_BUS]
    if len(ess_judged) > 0:
        bus_median = ess_judged[bus_err_cols].median()
        print(f'간선 경로(슬랙->bus17) median err_v 누적 (버스별, 판정케이스 중 ESS 주입 있는 것만):',
              flush=True)
        for b in trunk:
            if b == PM.SLACK_BUS:
                continue
            print(f'  bus{b:02d}: {bus_median[f"errv_bus{b:02d}"]:.6f}', flush=True)
        first_b, last_b = trunk[1], trunk[-1]
        first_v = float(bus_median[f'errv_bus{first_b:02d}'])
        last_v = float(bus_median[f'errv_bus{last_b:02d}'])
        ratio = last_v / first_v if first_v > 0 else float('inf')
        print(f'  -> bus{first_b}({first_v:.6f}) 대비 bus{last_b}({last_v:.6f}): {ratio:.1f}배 '
              '(간선 누적 - 국소 폭발이 아니라 절대값이 작아 안전한 패턴)', flush=True)
        print('버스별 median err_v (내림차순 상위 5개, 참고):', flush=True)
        for col, val in bus_median.sort_values(ascending=False).head(5).items():
            print(f'  {col}: {val:.6f}', flush=True)

    b_visited = judged[judged['section'] == 'B_visited'].copy()
    if len(b_visited) > 1:
        corr = b_visited['p_mw'].abs().corr(b_visited['case_max_err_v'])
        print(f'\nB_visited(방전): |P_ess| vs case_max_err_v 상관계수 = {corr:.4f}', flush=True)
    e_charge = judged[judged['section'] == 'E_visited_charge'].copy()
    if len(e_charge) > 1:
        corr_c = e_charge['p_mw'].abs().corr(e_charge['case_max_err_v'])
        print(f'E_visited_charge(충전): |P_ess| vs case_max_err_v 상관계수 = {corr_c:.4f} '
              '(양수면 "주입 클수록 오차 큼"이라는 1절 예상과 일치 - 충전이 부하에 더해져 '
              '|선로조류|를 키우므로 방전보다 이 상관이 뚜렷할 것으로 예상)', flush=True)

    section('역조류 달성 여부 (G_true_reverse, 판정 제외 - Phase 2 대비 기록)')
    g_rows = df[df['section'] == 'G_true_reverse']
    if len(g_rows):
        achieved = g_rows[g_rows['vm_pu_max'] > 1.0]
        print(f'G 케이스 {len(g_rows)}건 중 vm_pu_max>1.0(과전압 실제 발생) {len(achieved)}건', flush=True)
        best = g_rows.loc[g_rows['vm_pu_max'].idxmax()]
        print(f'  최대 vm_pu_max = {best["vm_pu_max"]:.6f} pu '
              f'(group={best.get("injection_group")}, scenario={best["scenario"]}, '
              f'hour={best["hour"]}, p_mw(버스당)={best["p_mw"]})', flush=True)
        if len(achieved) == 0:
            print('  ★ 이번 격자로도 과전압(>1.0)을 못 만들었음 - 주입량을 더 키워야 함 '
                  '(판정 대상 아니므로 게이트 결론에는 영향 없음, Phase 2 착수 전 재시도 필요).',
                  flush=True)
    else:
        print('G 케이스 없음(생성 확인 필요)', flush=True)

    section('대칭성 점검 (B_boundary 충전(음수) vs D_reverse(양수 발전초과), 같은 |주입|)')
    charge = df[(df['section'] == 'B_boundary') & (df['p_mw'] < 0)
                & (df['ess_bus'].isin(BUSES_D))].copy()
    charge['abs_p_mw'] = charge['p_mw'].abs()
    reverse = df[df['section'] == 'D_reverse'].copy()
    reverse['abs_p_mw'] = reverse['p_mw'].abs()
    merged = charge.merge(reverse, on=['scenario', 'hour', 'ess_bus', 'abs_p_mw'],
                           suffixes=('_charge', '_reverse'))
    if len(merged):
        diff = (merged['case_max_err_v_charge'] - merged['case_max_err_v_reverse']).abs()
        print(f'매칭 {len(merged)}쌍, |오차차이| 평균={diff.mean():.6f} 최대={diff.max():.6f} pu '
              '(작을수록 방향대칭 - 1절 "오차는 |조류|에 의존, 방향에 대칭")', flush=True)
    else:
        print('매칭되는 쌍 없음(케이스 생성 확인 필요)', flush=True)

    # CSV 저장 (item5)
    out_dir = os.path.join(SCRIPT_DIR, 'results', 'test_lindistflow')
    os.makedirs(out_dir, exist_ok=True)
    hostname = socket.gethostname()
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    cases_path = os.path.join(out_dir, f'test_lindistflow_cases_{hostname}_{ts}.csv')
    df.to_csv(cases_path, index=False, encoding='utf-8')
    print(f'\n버스별 오차 분포 CSV 저장: {cases_path} ({len(df)}행)', flush=True)

    curve_path = os.path.join(out_dir, f'test_lindistflow_curve_{hostname}_{ts}.csv')
    curve_df = build_curve_csv(df)
    curve_df.to_csv(curve_path, index=False, encoding='utf-8')
    print(f'오차곡선 CSV 저장: {curve_path} ({len(curve_df)}행)', flush=True)

    section('종합 요약')
    print(f'1차(상한선) 판정: {verdict1}', flush=True)
    print(f'2차(기대성능) 판정: {verdict2}', flush=True)
    print(f'기존판정(방전만) max={old_max:.6f} vs 신규판정(충전 포함) max={global_max_err_v:.6f} '
          f'- 충전을 포함해도 결론이 {"동일" if verdict1 == "PASS" else "뒤집힘"}', flush=True)
    print('다음 단계는 사람이 판단: 통과 시 C.6-3(본체 lower_lp.py 편입) 착수, 애매하면 '
          '위 구조진단(간선누적 패턴/충전vs방전 상관/대칭성)부터 재검토 '
          '(C.8 라, 기계적 폐기 금지).', flush=True)


if __name__ == '__main__':
    main()
