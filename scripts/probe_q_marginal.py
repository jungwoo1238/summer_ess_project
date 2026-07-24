"""Q(무효전력)의 순 한계효과 부호를 k-스윕 귀납이 아니라 계통 파라미터로부터의 부등식으로
확정한다 (확인 전용, 본 파이프라인 미포함 - probe_q_value.py/probe_q_sensitivity.py의 후속).

배경: probe_q_value.py/probe_q_sensitivity.py는 특정 (b,S,E,k) 조합을 조류계산으로 직접
평가해 "이 지점에서 Q가 이득/손해"를 실측했다. 이 스크립트는 반대 방향으로 접근한다 -
"어떤 버스·시각·크기에서도 Q 주입이 순손해"라는 명제를 개별 지점 평가가 아니라
버스 i에서의 한계효과(marginal effect) 부등식으로 직접 유도한다.

## 판정 부등식 (★ 표기 명확화 - 두 Q를 구분한다)
  Q_inj : ESS가 버스 i에 주입하는 무효전력 [Mvar] - 결정변수(ESS 운전점)
  Q_flow(e) : 분기 e를 흐르는 무효조류 [Mvar] - 상태변수(res_line.q_from_mvar에서 읽음)

방사형 망에서 버스 i에 Q_inj를 주입하면 path(i)(슬랙~i 사이의 모든 분기) 위의 모든
Q_flow가 그만큼 정확히 감소하고(방사형이라 경로가 유일함) 경로 밖은 불변이므로:

  선로손실 저감 한계효과(Q_inj=0에서 평가 - 주입하면 Q_flow가 줄어 체감하므로 이 값이 상한):
    -dLoss_line/dQ_inj = 2 * Sum_{e in path(i)} r_e * Q_flow(e) / V_e^2   =: LHS

  PCS 손실 한계증분(benefits.loss_pcs의 미분, ETA_PCS=params.py 상수):
    dLoss_pcs/dQ_inj = (1-ETA_PCS) * Q_inj / sqrt(P_inj^2 + Q_inj^2)      =: RHS
    (Q_inj/sqrt(P_inj^2+Q_inj^2)는 ESS 출력의 무효분 비율, 0~1이므로 RHS <= 1-ETA_PCS = 0.03)

  순손해 조건: LHS < RHS

LHS는 계통 조건(역률·부하·경로)이 정하고 ESS 운전점과 무관하다(Q_inj=0에서 평가하므로).
RHS는 ESS 운전점(P_inj,Q_inj)이 정하고 계통 조건과 무관하다. 두 쪽이 서로 다른 것에
의존하므로 "LHS의 전역 최대값 < RHS의 전역 최대값(0.03)"이면 **어떤 버스·시각·운전점을
골라도** 순손해가 성립한다 - 개별 (b,S,E,k) 지점을 아무리 스윕해도 반증할 수 없는 결론이다.

★ 단위 규약(중요, 틀리면 LHS와 RHS의 자릿수가 안 맞는다): RHS는 무차원 비율(0~0.03)이므로
LHS도 무차원이어야 직접 비교가 성립한다. r_e/Q_flow(e)/V_e를 물리단위(Ohm/MVar/kV) 그대로
넣으면 LHS가 무차원이 되지 않는다 - 반드시 pu로 변환한다(r_e_pu=r_e_ohm/Z_BASE_OHM,
Q_flow_pu=Q_flow_mvar/S_BASE_MVA, V_e는 res_bus.vm_pu가 이미 pu). params.py의
Z_BASE_OHM/S_BASE_MVA(=TARGET_MVA)를 그대로 쓴다(lower_lp.py가 LinDistFlow에 쓰는 것과
동일한 base - 새로 정의하지 않는다).

## Q_flow(e)의 부호 규약 (명시적으로 확인 - 부호가 뒤집히면 결론이 뒤집힌다)
pandapower의 `res_line.q_from_mvar`는 from_bus에서 해당 선로로 "흘러 들어가는" 무효전력이다.
case33bw(및 lower_lp._build_topology가 이미 검증한 구조 - CLAUDE.md 부록C.2, 게이트 실측
"Baran-Wu load-positive 부호규약")에서 모든 선로의 from_bus는 슬랙에 더 가까운(상류) 버스,
to_bus는 하류 버스다(방사형 트리를 from_bus->to_bus 방향으로만 구성해도 32개 선로 전부가
모순 없이 하나의 트리를 이룬다는 사실 자체가 이 가정의 검증이다 - _build_parent_tracking
참조). 부하는 유효/무효 전력을 모두 소비(양수)하므로 정상 상태에서 `q_from_mvar`는 상류에서
하류로 흘러가는 방향(=부하를 향하는 방향)에서 양수다. 즉 `q_from_mvar`가 바로 이 스크립트가
쓰는 Q_flow(e)이고, 그 부호는 "하류 부하가 소비하는 무효전력"과 같은 방향이다 - 버스 i에
ESS가 Q_inj>0(용량성, 부하가 필요로 하는 무효전력의 일부를 대신 공급)을 주입하면 path(i)의
모든 Q_flow(e)가 정확히 Q_inj만큼 줄어든다(그만큼 상류에서 덜 끌어와도 되므로). 이것이
"방사형 망에서 버스 i에 Q_inj를 주입하면 path(i)상의 모든 분기에서 Q_flow가 그만큼 감소"라는
전제와 정확히 일치하는 부호다.

## V_e(대표 전압)의 선택
분기 e의 from_bus(상류측) vm_pu를 쓴다. 근거: 손실공식 Loss_e = r_e*(P_e^2+Q_e^2)/V_e^2에서
V_e는 원래 전류 I_e = S_e/V_e를 정의하는 기준 전압인데, 방사형 배전망은 전압강하가 작아
(1절: 기저 강하폭 0.059pu, 슬랙~말단 전체) from_bus/to_bus 전압 차가 크지 않다. from_bus를
쓰는 이유는 (a) 슬랙(bus 0)에 가장 가까운 쪽이라 pu 기준 전압(1.0 부근)에 더 가깝게 유지되어
분모가 0에 가까워질 위험이 상대적으로 작고, (b) LinDistFlow(lower_lp.py)가 이미 "상류 버스
전압으로 하류 조류를 표현"하는 재귀식(v_j = v_i - 2*(...))을 쓰므로 동일한 관례를 따르면
기존 코드와 비교하기 쉽다. to_bus를 썼어도 결론(LHS의 자릿수·부호)이 바뀔 정도로 전압차가
크지 않다(1절 검증값 참조) - 다만 재현성을 위해 여기서는 from_bus로 고정한다.

## 이 판정의 유효 범위 (★ 명시)
이 부등식은 "선로손실 저감 vs PCS 손실 증가"라는 **손실 채널만** 비교한다. 전압 페널티
(evaluate.py의 LAMBDA_V, lower_lp.py의 mu_volt)가 활성(비영, 그리고 실제로 바인딩)이면
Q_inj가 전압을 끌어올려 그 페널티를 줄이는 **별도의 편익 채널**이 추가되고, 이 부등식의
LHS에는 그 항이 없다. 현재 기저(ESS 없음) 조류에서는 전 시나리오·시각·버스 전압위반이
0이므로(1절 VALIDATION.v_violation_total_scaled=0.0) 이 채널은 지금 비활성이지만, Phase 2
(태양광 역조류로 인한 과전압)에서는 활성화될 수 있다 - 그때는 이 부등식에 전압지원 편익
항을 추가해 재평가해야 하며, 이 스크립트의 결론을 그대로 끌어쓰면 안 된다.

## 재사용
probe_q_value.py의 _build_net_with_pf/_prepare_condition을 그대로 재사용한다(통제 조건을
새로 정의하지 않기 위함 - probe_q_sensitivity.py와 같은 이유).

실행: `python scripts/probe_q_marginal.py`  (★ 이 스크립트는 작성만 하고 실행하지 않는다 -
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

import params as PM
import evaluate

from probe_q_value import (
    BASE_PF_EXPECTED,
    TARGET_PF,
    _build_net_with_pf,
    _prepare_condition,
    section,
    _check_env,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ★ scripts/results (프로젝트 루트 results/가 아니다) - 루트 results/는 main.py 본실험
# 결과 전용이고, 이 스크립트는 일회성 확인용이라 scripts/results에 둔다.
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

BUS_CANDIDATES = list(range(PM.B_BOUNDS[0], PM.B_BOUNDS[1] + 1))  # 1..32 (슬랙 0 제외)

RHS_MAX = 1.0 - PM.ETA_PCS  # RHS = (1-ETA_PCS)*ratio, ratio<=1이므로 RHS의 전역 상한

CSV_FIELDS = ['power_factor', 'bus', 'scenario', 't', 'lhs', 'q_e_sum', 'path_len']


# ============================================================
# 방사형 토폴로지: 부모 추적으로 path(i) 구성 (networkx 없이)
# ============================================================

def _build_parent_tracking(net):
    """in_service 선로만으로 슬랙(bus0)을 루트로 하는 트리를 구성한다. parent_of[bus]=부모버스,
    line_of[bus]=그 버스를 부모에 잇는 pandapower 선로 인덱스(원본 인덱스, 압축하지 않음 -
    net.line/net.res_line을 그대로 인덱싱하기 위함). from_bus가 항상 상류라는 가정은 모듈
    docstring의 "Q_flow(e) 부호 규약" 절에서 검증한다(트리가 모순 없이 구성된다는 사실 자체가
    검증이다 - 만약 어떤 선로의 from_bus가 실제로는 하류였다면 아래 루프에서 그 버스가 두 번
    이상 부모를 갖으려 하거나 슬랙에서 도달 불가능한 고립 버스가 남아 즉시 드러난다)."""
    lines = net.line[net.line['in_service']]
    children = {int(b): [] for b in net.bus.index}
    for idx, row in lines.iterrows():
        i, j = int(row['from_bus']), int(row['to_bus'])
        children[i].append((j, idx))

    parent_of = {PM.SLACK_BUS: None}
    line_of = {}
    stack = [PM.SLACK_BUS]
    while stack:
        u = stack.pop()
        for v, lidx in children[u]:
            assert v not in parent_of, (
                f'버스 {v}가 두 부모를 가짐(비방사형 또는 from/to 방향 가정 위반) - '
                'Q_flow 부호 규약을 재검토할 것'
            )
            parent_of[v] = u
            line_of[v] = lidx
            stack.append(v)

    missing = set(int(b) for b in net.bus.index) - set(parent_of.keys())
    assert not missing, f'슬랙에서 도달 불가능한 버스 존재: {missing} - in_service 선로 확인'
    return parent_of, line_of


def _path_lines(bus, parent_of, line_of):
    """슬랙~bus 사이의 pandapower 선로 인덱스 리스트(하류 순서: bus에 가장 가까운 선로가
    마지막이 아니라 첫 원소 - 순서는 합산에 영향 없으므로 상관없다)."""
    path = []
    u = bus
    while u != PM.SLACK_BUS:
        path.append(line_of[u])
        u = parent_of[u]
    return path


def _line_r_pu(net):
    """in_service 선로별 r_e_pu = r_ohm_per_km*length_km/Z_BASE_OHM. dict[line_idx]->float."""
    lines = net.line[net.line['in_service']]
    r_pu = {}
    for idx, row in lines.iterrows():
        r_pu[idx] = float(row['r_ohm_per_km']) * float(row['length_km']) / PM.Z_BASE_OHM
    return r_pu


# ============================================================
# ALL_DAYS x 24h 기저(Q_inj=0, ESS 없음) 조류계산 - 선로별 원시값 수집
# ============================================================

def _compute_branch_series(net, base_p, base_q):
    """evaluate._compute_base_flow(집계값 p_slack/loss만 반환)를 재사용하지 않는다 - 이
    스크립트가 필요한 것은 집계가 아니라 (시나리오,시각)별 선로별 q_from_mvar과 from_bus
    vm_pu 원시값이기 때문이다. net에는 sgen이 없으므로(build_net() 직후 상태) 이 조류계산은
    이미 Q_inj=0 조건이다.

    반환: dict[scenario] -> dict[t] -> dict[line_idx -> (q_from_mvar, v_from_pu)]
    """
    lines = net.line[net.line['in_service']]
    line_idxs = lines.index.to_numpy()
    from_bus_of = {int(idx): int(net.line.at[idx, 'from_bus']) for idx in line_idxs}

    series = {}
    for s in PM.ALL_DAYS:
        profile = PM.LOAD[s]
        series[s] = {}
        for t in range(PM.TIME_STEPS):
            scale = profile[t]
            net.load['p_mw'] = base_p * scale
            net.load['q_mvar'] = base_q * scale

            ok = evaluate._run_pf_with_retry(net)
            if not ok:
                raise RuntimeError(
                    f'기저(Q_inj=0, ESS 없음) 조류계산 발산: 시나리오={s}, t={t}. '
                    '정상 부하범위에서 기저 발산은 비정상이므로 그대로 보고한다.'
                )

            q_from = net.res_line['q_from_mvar']
            vm = net.res_bus['vm_pu']
            per_t = {}
            for idx in line_idxs:
                idx = int(idx)
                per_t[idx] = (float(q_from.at[idx]), float(vm.at[from_bus_of[idx]]))
            series[s][t] = per_t
    return series


# ============================================================
# LHS 계산 (역률 1개 조건)
# ============================================================

def _run_pf_scan(pf_target, pf_label):
    net, q_scale, p_total, q_total_before = _build_net_with_pf(pf_target)
    base_p, base_q = _prepare_condition(net)

    parent_of, line_of = _build_parent_tracking(net)
    r_pu = _line_r_pu(net)

    # ---- 검산1: path(슬랙)=공집합 ----
    assert _path_lines(PM.SLACK_BUS, parent_of, line_of) == [], (
        'path(bus0)이 비어있지 않음 - 부모추적 로직 오류'
    )

    path_cache = {b: _path_lines(b, parent_of, line_of) for b in BUS_CANDIDATES}
    path_len = {b: len(path_cache[b]) for b in BUS_CANDIDATES}

    # ---- 검산2: 말단 버스(17,32)의 path_len이 상류 버스(1)보다 큼 ----
    assert path_len[17] > path_len[1], (
        f"path_len[17]={path_len[17]} <= path_len[1]={path_len[1]} - 방사형 깊이 계산 오류 의심"
    )
    assert path_len[32] > path_len[1], (
        f"path_len[32]={path_len[32]} <= path_len[1]={path_len[1]} - 방사형 깊이 계산 오류 의심"
    )

    branch_series = _compute_branch_series(net, base_p, base_q)

    rows = []
    lhs_grid = {}
    for s in PM.ALL_DAYS:
        for t in range(PM.TIME_STEPS):
            per_t = branch_series[s][t]
            for bus in BUS_CANDIDATES:
                lhs_sum = 0.0
                q_e_sum = 0.0
                for lidx in path_cache[bus]:
                    q_from_mvar, v_from_pu = per_t[lidx]
                    lhs_sum += r_pu[lidx] * (q_from_mvar / PM.S_BASE_MVA) / (v_from_pu ** 2)
                    q_e_sum += q_from_mvar
                lhs = 2.0 * lhs_sum
                rows.append(dict(power_factor=pf_label, bus=bus, scenario=s, t=t,
                                  lhs=lhs, q_e_sum=q_e_sum, path_len=path_len[bus]))
                lhs_grid[(bus, s, t)] = lhs

    return rows, lhs_grid


# ============================================================
# stdout 요약 (자동판정 없음 - 수치만 제시)
# ============================================================

def _print_summary(pf_label, rows):
    section(f'요약: PF={pf_label:.6f}')

    best = max(rows, key=lambda r: r['lhs'])
    print(f"전역 최대 LHS = {best['lhs']:.6f}  (bus={best['bus']}, scenario={best['scenario']}, "
          f"t={best['t']})", flush=True)
    print(f"RHS 상한(1-ETA_PCS) = {RHS_MAX:.4f}  ->  전역최대LHS/RHS상한 비율 = "
          f"{best['lhs'] / RHS_MAX:.4f}", flush=True)

    over = [r for r in rows if r['lhs'] > RHS_MAX]
    print(f"\nLHS > {RHS_MAX:.4f}(RHS 상한)인 (bus,scenario,t) 조합 수 = {len(over)}", flush=True)
    for r in over[:50]:
        print(f"  bus={r['bus']:2d}  scenario={r['scenario']:12s}  t={r['t']:2d}  "
              f"lhs={r['lhs']:.6f}", flush=True)
    if len(over) > 50:
        print(f"  ... 외 {len(over) - 50}건 생략(CSV 전체 참조)", flush=True)

    print("\n버스별 임계 Q_i/S 비율 (그 버스의 시나리오x시각 중 LHS가 가장 큰=Q에 가장 유리한 "
          "지점 기준. Q/sqrt(P^2+Q^2) 비율이 이 값 미만일 때만 그 지점에서 Q가 이득):",
          flush=True)
    by_bus = {}
    for r in rows:
        by_bus.setdefault(r['bus'], []).append(r)
    for bus in sorted(by_bus):
        r_best = max(by_bus[bus], key=lambda r: r['lhs'])
        lhs_max = r_best['lhs']
        if lhs_max >= RHS_MAX:
            verdict = '모든 Q 수준에서 이득 (LHS_max >= RHS 상한)'
        else:
            crit_ratio = lhs_max / RHS_MAX
            verdict = f'Q/sqrt(P^2+Q^2) < {crit_ratio:.4f}에서만 이득'
        print(f"  bus={bus:2d}: LHS_max={lhs_max:.6f} (scenario={r_best['scenario']}, "
              f"t={r_best['t']})  ->  {verdict}", flush=True)


def _print_interpretation():
    section('해석 지침 (자동판정 없음 - 수치를 보고 사람이 판단할 것)')
    print(
        "- PF=0.95에서 전역 최대 LHS < RHS 상한(0.03)이면: Q는 그 버스의 Q_i/S가 임계비율을\n"
        "  넘는 범위에서만 순손해이고 그 아래에서는 이득이다 - 임계비율과 그 영역의 j_net\n"
        "  기여 크기를 별도로(예: probe_q_sensitivity.py 스타일 조류계산으로) 평가해야 한다.\n"
        "- PF=0.95에서 전역 최대 LHS가 RHS 상한보다 충분히 작으면(예: 절반 이하): 실질적으로\n"
        "  거의 모든 Q 수준·버스·시각에서 순손해로 볼 수 있다.\n"
        "- 이 판정은 전압 페널티(mu_volt/LAMBDA_V)가 비활성인 경우에 한한다(모듈 docstring\n"
        "  '이 판정의 유효 범위' 절 참조). Phase 2에서 과전압이 발생하면 Q에 전압지원 편익이\n"
        "  추가되므로 이 부등식을 그대로 쓰면 안 된다.\n"
        "- 임계값을 코드가 자동으로 판정해 출력하지 않는다(probe_q_value.py/\n"
        "  probe_q_sensitivity.py에서 자동판정이 오판을 낸 전례가 있다 - 위 수치를 직접\n"
        "  보고 판단할 것).",
        flush=True,
    )


# ============================================================
# 메인
# ============================================================

def _make_path():
    hostname = socket.gethostname()
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(RESULTS_DIR, f'probe_q_marginal_{hostname}_{ts}.csv')


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

    section('PF=0.850(원본)')
    rows_base, lhs_base = _run_pf_scan(None, BASE_PF_EXPECTED)

    section('PF=0.95(보상, P 고정)')
    rows_target, lhs_target = _run_pf_scan(TARGET_PF, TARGET_PF)

    # ---- 검산3: PF=0.95의 LHS가 전 조합에서 PF=0.850보다 작음(무효조류가 줄었으므로) ----
    tol = 1e-9
    violations = [
        (key, lhs_base[key], lhs_target[key])
        for key in lhs_base
        if lhs_target[key] > lhs_base[key] + tol
    ]
    assert not violations, (
        f'검산3 실패: PF=0.95의 LHS가 PF=0.850보다 큰 조합이 {len(violations)}건 있음 - '
        f'예: {violations[:5]} - Q_flow 부호 규약(모듈 docstring 참조)을 재검토할 것'
    )

    all_rows = rows_base + rows_target
    _write_csv(_make_path(), all_rows)

    _print_summary(BASE_PF_EXPECTED, rows_base)
    _print_summary(TARGET_PF, rows_target)
    _print_interpretation()
