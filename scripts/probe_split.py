"""분산 설치의 이득 확인 - 총 규모(S_total,E_total)를 n=1 최적값으로 고정한 채 2~3기로
쪼개 배치했을 때 j_net이 단일 설치보다 개선되는지 직접 측정한다 (확인 전용, 본 파이프라인
미포함 - scripts/probe_large_solutions.py, probe_bus_sweep.py와 같은 성격).

배경(CLAUDE.md 7-A절 "★ 미확인: 분산 설치의 이득"): n=1 최적해는 b=15 근방,
S=0.1764 MVA, E=0.4192 MWh, j_net≈3.05e6원/년이다. n=2 dev 실행 3 run 전부 한 기를
S=E=0으로 소멸시켜 사실상 n=1과 동치인 해로 수렴했는데, 이것이 "분산해도 이득이
없어서"인지 "PSO가 못 찾아서"인지는 그 실행만으로는 갈리지 않는다. scripts/probe_bus_sweep.py
실측(CLAUDE.md 8절-5)이 위치 의존성의 근원을 밝혔다 - b_defer(편익의 84%)는 슬랙 피크
1점만 보므로 위치·분할에 거의 무관하고, 위치 차이는 거의 전부 b_loss(I^2R)에서 나온다.
따라서 이 실험이 실제로 재는 것은 b_loss의 차이뿐이며, 그 절대 크기는 단일 설치 기준
약 19만원(j_net의 6%)이다 - PSO 자체의 run간 편차(1.04%, ~3만원)보다는 크지만 여전히
작은 항이다.

PSO를 돌리지 않는다. 지정한 (b,S,E) 조합을 evaluate.evaluate_particle로 직접 평가한다.
evaluate.py/benefits.py/lower_lp.py/params.py는 수정하지 않는다(순수 소비자). 입자 벡터는
evaluate의 4n 시그니처(b_1,S_1,E_1,q_1, b_2,S_2,E_2,q_2, ...)를 그대로 쓴다(q는 전부 0.0 -
CLAUDE.md 7-A절 _expand_to_4n과 같은 규약).

★ 노이즈 기준 (오해하기 쉬운 지점): PSO run간 편차 1.04%(~3만원)를 이 실험의 분해능으로
쓰면 안 된다 - 그건 *탐색* 편차이고 이 실험은 탐색을 하지 않는다. 이 실험의 실제 노이즈는
CLAUDE.md 7절 "수치 잡음의 성격"이 규명한 warm start 이력 잡음뿐이며 상한이 6.91원/년이다.
즉 수천 원 수준의 차이도 유의미하게 검출된다 - "노이즈에 묻혔다"고 쓰기 전에 반드시
차이의 절대값을 6.91원과 비교할 것.

실행: `python scripts/probe_split.py`

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 총 규모 고정값 - n=1 최적해(main.py dev run0, CLAUDE.md 7-A절). 이 실험 전체에서
# 절대 바꾸지 않는다(규모 효과와 분산 효과가 섞이면 6-A절대로 해석 불가능해진다).
TOTAL_S = 0.1764   # MVA
TOTAL_E = 0.4192   # MWh

Q_RATIO = 0.0   # 뼈대 확정값 (CLAUDE.md 2절 "뼈대에서는 Q=0으로 고정")

# 수치 잡음 상한 (CLAUDE.md 7절 "수치 잡음의 성격" 실측 - scripts/probe_noise.py 유래).
# 이 값보다 작은 차이는 warm start 이력 잡음과 구분 불가하므로 "유의미한 차이"의 최소 기준.
NOISE_BOUND_WON = 6.91

# gbest_f 재현 검산 등 금액 항등식 비교의 허용오차 (main.py RUN_CONSISTENCY_ATOL_WON과 같은
# 계보, CLAUDE.md 7절 원칙4 "금액, 기대값=0"). r=0.5 재계산 vs (B) 대응 조합 대조에 쓴다.
CONSISTENCY_ATOL_WON = 10.0

# (A) 기준선 - 단일 설치 5개. probe_bus_sweep.py 실측에서 b=13이 전체 32개 버스 중 최고
# 순위였다(주간선 중간, CLAUDE.md 8절-5). 이 스크립트의 "단일 최적" 기준점으로 쓴다.
SINGLE_BUSES = [13, 14, 12, 15, 31]
SINGLE_BEST_BUS = 13

# (B) 균등 2분할 - (버스쌍, 그룹라벨)
EVEN_SPLIT_PAIRS = (
    [((13, 14), 'same_feeder_adjacent'),
     ((12, 13), 'same_feeder_adjacent'),
     ((12, 14), 'same_feeder_adjacent'),
     ((14, 15), 'same_feeder_adjacent'),
     ((13, 15), 'same_feeder_adjacent')]
    + [((13, 17), 'same_feeder_distant'),
       ((12, 16), 'same_feeder_distant'),
       ((13, 10), 'same_feeder_distant'),
       ((12, 17), 'same_feeder_distant')]
    + [((13, 31), 'cross_feeder'),
       ((13, 32), 'cross_feeder'),
       ((13, 30), 'cross_feeder'),
       ((14, 31), 'cross_feeder'),
       ((12, 31), 'cross_feeder'),
       ((15, 31), 'cross_feeder'),
       ((13, 29), 'cross_feeder')]
    + [((13, 11), 'upper_x_mid'),
       ((13, 9), 'upper_x_mid')]
    + [((13, 18), 'control'),
       ((13, 1), 'control')]
)

# (C) 비균등 2분할 - 상위 조합 3개, 분할비 스윕. r=0.5는 (B)의 대응 조합과 물리적으로
# 동일해야 하므로 재현성 검증용으로 겹쳐 계산한다(측정4).
UNEVEN_PAIRS = [(13, 31), (13, 17), (13, 14)]
RATIO_SWEEP = [0.25, 0.375, 0.5, 0.625, 0.75]

# (D) 3분할 - 상위 조합 2개
TRIPLE_GROUPS = [(13, 14, 31), (13, 31, 32)]

CSV_FIELDS = [
    'case', 'n_units', 'buses', 'split_ratio', 'S_each', 'E_each',
    'j_net', 'b_energy', 'b_defer', 'b_arb', 'b_loss', 'cost',
    'v_violation', 'i_violation', 'diverged', 'delta_vs_single_best',
]


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


def _eval_units(buses, S_list, E_list, q=Q_RATIO):
    """지정한 n기 조합을 evaluate의 4n 시그니처로 조립해 직접 평가한다(PSO 미사용,
    main.py의 _expand_to_4n과 같은 규약). buses/S_list/E_list는 길이가 같아야 한다."""
    n = len(buses)
    assert len(S_list) == n and len(E_list) == n, (buses, S_list, E_list)
    x = np.zeros(4 * n, dtype=float)
    for i in range(n):
        x[4 * i:4 * i + 4] = [float(buses[i]), float(S_list[i]), float(E_list[i]), float(q)]
    return evaluate.evaluate_particle(x, return_detail=True)


# ============================================================
# 단일 설치 캐시 (모든 조합에서 쓰이는 버스 전부 - delta_vs_single_best 기준값)
# ============================================================

def _collect_needed_single_buses():
    needed = set(SINGLE_BUSES)
    needed.add(SINGLE_BEST_BUS)
    for pair, _group in EVEN_SPLIT_PAIRS:
        needed.update(pair)
    for pair in UNEVEN_PAIRS:
        needed.update(pair)
    for triple in TRIPLE_GROUPS:
        needed.update(triple)
    return sorted(needed)


def build_single_cache():
    section('단일 설치 캐시 계산 (모든 조합의 delta_vs_single_best 기준값)')
    needed = _collect_needed_single_buses()
    cache = {}
    for b in needed:
        detail = _eval_units([b], [TOTAL_S], [TOTAL_E])
        cache[b] = detail
        tag = ' <- 단일 최적(기준)' if b == SINGLE_BEST_BUS else ''
        print(f"  b={b:2d} j_net={detail['j_net']:.4e} b_defer={detail['b_defer']:.4e} "
              f"b_energy={detail['b_energy']:.4e} b_loss={detail['b_loss']:.4e}{tag}", flush=True)

    actual_best = max(cache, key=lambda b: cache[b]['j_net'])
    if actual_best != SINGLE_BEST_BUS:
        print(f'  참고: 이번 실행에서는 b={actual_best}가 b={SINGLE_BEST_BUS}보다 근소하게 '
              f'높다(j_net 차이={cache[actual_best]["j_net"] - cache[SINGLE_BEST_BUS]["j_net"]:.2f}원). '
              'probe_bus_sweep.py 32개 전수 스윕 기준으로는 b=13이 최고였다 - 이 스크립트가 '
              '평가한 15개 버스 부분집합 안에서의 참고 사실일 뿐 모순은 아니다.', flush=True)

    return cache


# ============================================================
# 레코드 조립 (CSV_FIELDS + 리포트 전용 부가필드, DictWriter가 extrasaction='ignore'로
# CSV_FIELDS만 골라 쓴다)
# ============================================================

def _split_ratio_str(S_list):
    total = sum(S_list)
    if total <= 0:
        return ''
    return ';'.join(f'{s / total:.4f}' for s in S_list)


def _build_record(case, buses, S_list, E_list, detail, single_cache, extra=None):
    n_units = len(buses)
    buses_str = ';'.join(str(b) for b in buses)
    S_each_str = ';'.join(f'{s:.6f}' for s in S_list)
    E_each_str = ';'.join(f'{e:.6f}' for e in E_list)
    split_ratio_str = _split_ratio_str(S_list)

    record = dict(
        case=case, n_units=n_units, buses=buses_str, split_ratio=split_ratio_str,
        S_each=S_each_str, E_each=E_each_str,
    )

    if detail.get('diverged'):
        record.update(
            j_net='', b_energy='', b_defer='', b_arb='', b_loss='', cost='',
            v_violation='', i_violation='', diverged=True, delta_vs_single_best='',
        )
        return record

    ref_bus = max(buses, key=lambda b: single_cache[b]['j_net'])
    ref_detail = single_cache[ref_bus]
    delta = detail['j_net'] - ref_detail['j_net']

    record.update(
        j_net=detail['j_net'], b_energy=detail['b_energy'], b_defer=detail['b_defer'],
        b_arb=detail['b_arb'], b_loss=detail['b_loss'], cost=detail['cost'],
        v_violation=detail['v_violation'], i_violation=detail['i_violation'],
        diverged=False, delta_vs_single_best=delta,
        # 리포트 전용 부가필드 (CSV_FIELDS에는 없음 - extrasaction='ignore'로 저장 시 제외됨)
        ref_bus=ref_bus,
        delta_b_defer=detail['b_defer'] - ref_detail['b_defer'],
        delta_b_energy=detail['b_energy'] - ref_detail['b_energy'],
        delta_b_loss=detail['b_loss'] - ref_detail['b_loss'],
    )
    if extra:
        record.update(extra)
    return record


# ============================================================
# 측정 A/B/C/D
# ============================================================

def measure_A(single_cache):
    section('(A) 기준선: 단일 설치 5개')
    records = []
    for b in SINGLE_BUSES:
        detail = single_cache[b]
        rec = _build_record('A_single', [b], [TOTAL_S], [TOTAL_E], detail, single_cache)
        records.append(rec)
        print(f"  b={b:2d} j_net={rec['j_net']:.4e}", flush=True)
    return records


def measure_B(single_cache):
    section('(B) 균등 2분할 (S_each=E_each=총량/2)')
    S_each, E_each = TOTAL_S / 2, TOTAL_E / 2
    records = []
    for pair, group in EVEN_SPLIT_PAIRS:
        detail = _eval_units(list(pair), [S_each, S_each], [E_each, E_each])
        rec = _build_record(f'B_even2_{group}', list(pair), [S_each, S_each], [E_each, E_each],
                             detail, single_cache, extra=dict(group=group))
        records.append(rec)
        if detail.get('diverged'):
            print(f'  {pair}({group}): 발산 -> 건너뜀', flush=True)
            continue
        print(f"  {pair}({group:20s}) j_net={rec['j_net']:.4e} "
              f"delta_vs_single_best={rec['delta_vs_single_best']:+.2f}원 "
              f"delta_b_loss={rec['delta_b_loss']:+.2f}원", flush=True)
    return records


def measure_C(single_cache):
    section('(C) 비균등 2분할 - 분할비 스윕 (r in {0.25,0.375,0.5,0.625,0.75})')
    records = []
    for pair in UNEVEN_PAIRS:
        print(f'  조합 {pair}:', flush=True)
        for r in RATIO_SWEEP:
            S1, S2 = r * TOTAL_S, (1 - r) * TOTAL_S
            E1, E2 = r * TOTAL_E, (1 - r) * TOTAL_E
            detail = _eval_units(list(pair), [S1, S2], [E1, E2])
            rec = _build_record('C_ratio_sweep', list(pair), [S1, S2], [E1, E2], detail,
                                 single_cache, extra=dict(pair=pair, ratio_r=r))
            records.append(rec)
            if detail.get('diverged'):
                print(f'    r={r:.3f}: 발산 -> 건너뜀', flush=True)
                continue
            print(f"    r={r:.3f} (S1={S1:.4f},S2={S2:.4f}) j_net={rec['j_net']:.4e} "
                  f"delta_vs_single_best={rec['delta_vs_single_best']:+.2f}원", flush=True)
    return records


def measure_D(single_cache):
    section('(D) 3분할 (S_each=E_each=총량/3)')
    S_each, E_each = TOTAL_S / 3, TOTAL_E / 3
    records = []
    for triple in TRIPLE_GROUPS:
        detail = _eval_units(list(triple), [S_each] * 3, [E_each] * 3)
        rec = _build_record('D_triple', list(triple), [S_each] * 3, [E_each] * 3,
                             detail, single_cache)
        records.append(rec)
        if detail.get('diverged'):
            print(f'  {triple}: 발산 -> 건너뜀', flush=True)
            continue
        print(f"  {triple} j_net={rec['j_net']:.4e} "
              f"delta_vs_single_best={rec['delta_vs_single_best']:+.2f}원", flush=True)
    return records


# ============================================================
# 판정
# ============================================================

def report1_ranked_table(all_records, single_cache):
    section('판정 1) 조합별 j_net 내림차순 (단일 최적 b=%d 대비 차이)' % SINGLE_BEST_BUS)
    ref_j_net = single_cache[SINGLE_BEST_BUS]['j_net']
    valid = [r for r in all_records if not r['diverged']]
    for r in sorted(valid, key=lambda r: -r['j_net']):
        diff = r['j_net'] - ref_j_net
        pct = diff / abs(ref_j_net) * 100 if ref_j_net != 0 else float('nan')
        print(f"  {r['case']:28s} buses={r['buses']:10s} j_net={r['j_net']:.4e}  "
              f"vs b={SINGLE_BEST_BUS}: {diff:+.2f}원({pct:+.4f}%)", flush=True)


def report2_component_variance(all_records):
    section('판정 2) 편익 성분별 분산(단일 대비 delta) - b_defer/b_energy/b_loss')
    non_single = [r for r in all_records if not r['diverged'] and r['case'] != 'A_single']

    for name in ('delta_b_defer', 'delta_b_energy', 'delta_b_loss'):
        vals = np.array([r[name] for r in non_single], dtype=float)
        print(f"  {name:16s}: min={vals.min():+.2f}  max={vals.max():+.2f}  "
              f"mean={vals.mean():+.2f}  std={vals.std():.2f}  (원)", flush=True)

    defer_vals = np.array([r['delta_b_defer'] for r in non_single], dtype=float)
    loss_vals = np.array([r['delta_b_loss'] for r in non_single], dtype=float)
    max_abs_defer = float(np.max(np.abs(defer_vals)))
    max_abs_loss = float(np.max(np.abs(loss_vals)))
    print(f"\n  |delta_b_defer| 최대={max_abs_defer:.2f}원  vs  |delta_b_loss| 최대={max_abs_loss:.2f}원",
          flush=True)
    if max_abs_defer > NOISE_BOUND_WON and max_abs_defer > max_abs_loss * 0.5:
        print('  ★ 예상 밖: b_defer에서도 노이즈 상한을 넘는 유의한 차이가 나왔고, 그 크기가 '
              'b_loss와 같은 자릿수다. "이득은 b_loss에서만 나온다"는 배경 가정이 이번 실행에서는 '
              '깨졌다 - 원인(피크시각 근처 시간대 위치별 손실 차이가 슬랙 피크 자체를 흔들었을 '
              '가능성 등) 추가 확인 필요.', flush=True)
    else:
        print('  b_defer 변화는 노이즈 상한 이하이거나 b_loss 대비 미미하다 - 배경 가정('
          '"이득이 있다면 b_loss에서만 나온다")과 정합.', flush=True)


def report3_ratio_curves(all_records):
    section('판정 3) (C) 분할비 스윕 곡선')
    c_records = [r for r in all_records if r['case'] == 'C_ratio_sweep' and not r['diverged']]
    by_pair = {}
    for r in c_records:
        by_pair.setdefault(r['pair'], []).append(r)

    for pair, recs in by_pair.items():
        recs_sorted = sorted(recs, key=lambda r: r['ratio_r'])
        print(f'  조합 {pair}:', flush=True)
        for r in recs_sorted:
            marker = '  <-- 균등분할(r=0.5)' if abs(r['ratio_r'] - 0.5) < 1e-9 else ''
            print(f"    r={r['ratio_r']:.3f}  j_net={r['j_net']:.4e}{marker}", flush=True)
        best = max(recs_sorted, key=lambda r: r['j_net'])
        if abs(best['ratio_r'] - 0.5) < 1e-9:
            print(f'    -> 최적 r=0.5(균등분할). 곡선이 중앙에서 정점.', flush=True)
        else:
            print(f"    -> 최적 r={best['ratio_r']:.3f} (균등분할이 아니라 {pair[0]}쪽으로 "
                  f"치우침 - 두 버스의 위치 가치가 다르다는 뜻)", flush=True)


def report4_r_half_reproducibility(all_records):
    section('판정 4) r=0.5 재계산 vs (B) 균등분할 재현성 확인 (atol=%.1f원)' % CONSISTENCY_ATOL_WON)
    b_by_pair = {}
    for r in all_records:
        if r['case'].startswith('B_even2_') and not r['diverged']:
            pair = tuple(int(x) for x in r['buses'].split(';'))
            b_by_pair[pair] = r

    all_match = True
    for r in all_records:
        if r['case'] == 'C_ratio_sweep' and abs(r.get('ratio_r', -1) - 0.5) < 1e-9 and not r['diverged']:
            pair = r['pair']
            b_rec = b_by_pair.get(pair)
            if b_rec is None:
                print(f'  {pair}: (B)에 대응 조합이 없음 - 비교 불가', flush=True)
                continue
            diff = abs(r['j_net'] - b_rec['j_net'])
            ok = diff <= CONSISTENCY_ATOL_WON
            all_match &= ok
            print(f"  {pair}: C(r=0.5) j_net={r['j_net']:.4e}  vs  B(균등) j_net={b_rec['j_net']:.4e}  "
                  f"차이={diff:.4f}원  {'일치' if ok else '★불일치'}", flush=True)
    print(f"\n  종합: {'전부 일치 - 재현성 확인됨' if all_match else '★불일치 발견 - 위 상세 참조'}",
          flush=True)


def final_verdict(all_records, single_cache):
    section('종합 판정')
    ref_bus = SINGLE_BEST_BUS
    ref_j_net = single_cache[ref_bus]['j_net']

    non_single = [r for r in all_records if not r['diverged'] and r['case'] != 'A_single']
    best = max(non_single, key=lambda r: r['j_net'])
    diff = best['j_net'] - ref_j_net

    print(f"단일 최적(b={ref_bus}) j_net = {ref_j_net:.4e}원", flush=True)
    print(f"분산 조합 중 최우수: {best['case']} buses={best['buses']} j_net={best['j_net']:.4e}원",
          flush=True)
    print(f"차이 = {diff:+.2f}원 (노이즈 상한 {NOISE_BOUND_WON}원)", flush=True)
    print(f"해당 조합의 b_loss={best['b_loss']:.2f}원 (delta_b_loss vs 참조 단일={best['delta_b_loss']:+.2f}원), "
          f"b_defer={best['b_defer']:.4e}원 (delta_b_defer={best['delta_b_defer']:+.2f}원)", flush=True)

    if diff > NOISE_BOUND_WON:
        print(f"\n판정: ★ 분산 이득 있음. {best['case']}({best['buses']})가 단일 최적보다 "
              f"{diff:.2f}원 크고, 이는 노이즈 상한({NOISE_BOUND_WON}원)을 명확히 상회한다. "
              'PSO가 이 조합을 못 찾았다는 뜻이다 - n=2 이상 탐색을 재검토할 것.', flush=True)
    else:
        print(f"\n판정: 분산 이득 없음. 모든 분산 조합이 단일 최적 이하(또는 노이즈 상한 "
              f"이내의 차이)다. 실제로 이득이 없다는 뜻이며, 최적 기수 1이라는 결론이 강화된다.",
              flush=True)


# ============================================================
# CSV 저장
# ============================================================

def _make_path():
    hostname = socket.gethostname()
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(SCRIPT_DIR, f'probe_split_{hostname}_{ts}.csv')


def _write_csv(path, records):
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

    evaluate.init_worker()  # 1회만 호출, 기저 조류계산 120회 캐싱 재사용

    single_cache = build_single_cache()

    all_records = []
    all_records += measure_A(single_cache)
    all_records += measure_B(single_cache)
    all_records += measure_C(single_cache)
    all_records += measure_D(single_cache)

    _write_csv(_make_path(), all_records)

    report1_ranked_table(all_records, single_cache)
    report2_component_variance(all_records)
    report3_ratio_curves(all_records)
    report4_r_half_reproducibility(all_records)
    final_verdict(all_records, single_cache)
