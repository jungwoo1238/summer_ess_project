"""full 실행 최적해(버스 15로 수렴) 용량을 전 버스(1~32)에 고정하고 전수 스윕 - PSO가
놓친 우월 버스가 있는지 스크리닝 (확인 전용, 본 파이프라인 미포함 - probe_bus_sweep.py,
probe_large_solutions.py 등과 같은 성격).

배경: full(n=1, 30 run) 최적해가 버스 15로 수렴했다(대다수 run). 그러나 PSO가 완전한
전역최적에 도달했다는 보장은 없으므로, 버스 15 최적해의 용량(S,E)을 전 버스에 동일하게
강제하고 전수 평가해 이 고정 용량에서 정말 버스 15가 최고 편익인지 확인한다.

★ 이 스윕의 논리적 범위 (결과 해석 시 반드시 지킬 것): 모든 버스에 동일 (S,E)를
강제하므로, 버스 15가 아닌 버스에 대해서는 그 버스의 진짜 잠재력을 과소평가한다(각
버스의 진짜 최적 용량은 서로 다름 - full 실측에서 버스별 최적 S가 0.1750~0.1792로
갈린다). 따라서 해석은 두 갈래로만 한다:
  1. 버스 15가 1위 -> PSO 위치 탐색이 옳았다는 강화 증거(다른 버스는 자기 최적 용량을
     줘도 15를 못 넘을 것 - 이미 자기 용량 기준에서도 15보다 낮았으므로).
  2. 다른 버스 X가 1위 -> X를 자기 최적 용량으로 재탐색해야 하는 신호(X의 여기 j_net은
     과소평가값이므로 진짜 잠재력은 더 높을 수 있다). PSO가 X를 놓쳤을 가능성 - 별도
     후속 실험 필요.
  즉 "놓친 우월 버스가 있는가"의 스크리닝이지 버스 간 최종 순위표가 아니다.

evaluate.py/benefits.py/lower_lp.py/pso_core.py/main.py는 수정하지 않는다(순수 소비자).
고정 용량(S,E)은 코드에 하드코딩하지 않고, runs.csv 경로를 인자로 받아 버스 15로
수렴한 run들의 x_json에서 직접 읽어 평균한다(재현성·검증가능성 - 다른 머신/다른 full
실행 결과로도 그대로 재사용 가능).

실행: `python scripts/probe_bus_sweep_full.py results/runs_n1_<ts>.csv`

# ------------------------------------------------------------------
# 무엇을 고정했는가 (실행 후 채울 것 - scripts/ 규약)
#   S,E 출처(runs.csv 경로, 표본 run 수):
#   고정값:
#   실행 일시:
#   결론:
# ------------------------------------------------------------------
"""
import os
import sys
import csv
import json
import socket
import argparse
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import params as PM
import evaluate

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

Q_RATIO = 0.0   # 뼈대 확정값 (CLAUDE.md 2절 "뼈대에서는 Q=0으로 고정")

# 고정 용량의 기준이 되는 버스. "값"(S,E)은 runs.csv에서 읽지만 "어느 버스로 수렴한
# run들을 볼 것인가"는 이 스크립트의 설계 질문 자체라 상수로 둔다(CSV 컬럼명
# delta_vs_bus15_*도 이 값에 대응 - 지시사항의 고정 컬럼명 그대로).
REFERENCE_BUS = 15

# 검증1(버스15 자기 j_net) 일치 판정 허용오차. 고정용량은 18개 run 각각의 (미세하게
# 다른) 실제 최적용량이 아니라 그 평균이므로, 순수 warm-start 잡음(7절 실측 상한
# 6.91원)보다 여유를 둔다 - S가 run마다 ~1e-5 MVA씩 달라 그 자체로도 j_net에 미세한
# 차이를 만들 수 있기 때문이다(둘을 구분하지 않고 넉넉한 한계로 "일치"를 판정).
VALIDATION1_ATOL_WON = 100.0


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


def _eval(b, S, E, q=Q_RATIO):
    """n=1 입자를 4n(b,S,E,q_ratio) 형태로 조립해 evaluate_particle 호출 (CLAUDE.md 7-A절
    _expand_to_4n 어댑터와 같은 규약)."""
    x = np.array([float(b), float(S), float(E), float(q)], dtype=float)
    return evaluate.evaluate_particle(x, return_detail=True)


# ============================================================
# runs.csv에서 버스별 (S,E,j_net) 읽기 - 고정 용량은 여기서 파생, 하드코딩 금지
# ============================================================

def load_runs(runs_csv):
    with open(runs_csv, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    parsed = []
    for r in rows:
        x = json.loads(r['x_json'])
        assert len(x) == 3, (
            f"n_ess!=1로 보이는 x_json(len={len(x)}) - 이 스크립트는 단일기(n=1) full 결과 "
            f"전용이다: {r['x_json']}")
        b, S, E = x
        parsed.append(dict(run=int(r['run']), b=int(round(b)), S=float(S), E=float(E),
                            j_net=float(r['j_net'])))
    return parsed


def bus_stats(parsed_runs, bus):
    """설치된(S,E>0) run만 대상으로 그 버스의 (S,E) 평균/표준편차와 j_net median/mean."""
    vals = [r for r in parsed_runs if r['b'] == bus and r['S'] > 1e-9 and r['E'] > 1e-9]
    if not vals:
        return None
    S_arr = np.array([r['S'] for r in vals])
    E_arr = np.array([r['E'] for r in vals])
    j_arr = np.array([r['j_net'] for r in vals])
    return dict(n=len(vals), S_mean=float(S_arr.mean()), S_std=float(S_arr.std()),
                E_mean=float(E_arr.mean()), E_std=float(E_arr.std()),
                j_median=float(np.median(j_arr)), j_mean=float(j_arr.mean()))


# ============================================================
# 전 버스 스윕
# ============================================================

SWEEP_FIELDS = [
    'bus', 'S_fixed', 'E_fixed', 'j_net', 'b_energy', 'b_defer', 'b_arb', 'b_loss', 'cost',
    'v_violation', 'i_violation', 'penalty_v', 'penalty_line', 'diverged',
    'rank', 'delta_vs_bus15_won', 'delta_vs_bus15_pct',
]


def sweep_all_buses(S_fixed, E_fixed):
    section(f'전 버스(1~32) 전수 스윕: S={S_fixed:.6f} MVA, E={E_fixed:.6f} MWh 고정')
    rows = []
    for b in range(PM.B_BOUNDS[0], PM.B_BOUNDS[1] + 1):
        detail = _eval(b, S_fixed, E_fixed)
        if detail.get('diverged'):
            print(f'  b={b:2d}: 조류계산 발산', flush=True)
            rows.append(dict(
                bus=b, S_fixed=S_fixed, E_fixed=E_fixed,
                j_net='', b_energy='', b_defer='', b_arb='', b_loss='', cost='',
                v_violation='', i_violation='', penalty_v='', penalty_line='', diverged=True,
            ))
            continue

        penalty_v = PM.LAMBDA_V * detail['v_violation']
        penalty_line = PM.LAMBDA_LINE * detail['i_violation']
        row = dict(
            bus=b, S_fixed=S_fixed, E_fixed=E_fixed,
            j_net=detail['j_net'], b_energy=detail['b_energy'], b_defer=detail['b_defer'],
            b_arb=detail['b_arb'], b_loss=detail['b_loss'], cost=detail['cost'],
            v_violation=detail['v_violation'], i_violation=detail['i_violation'],
            penalty_v=penalty_v, penalty_line=penalty_line, diverged=False,
        )
        rows.append(row)
        flag = '  *전압위반' if detail['v_violation'] > 0 else ''
        print(f"  b={b:2d} j_net={row['j_net']:.6e} b_defer={row['b_defer']:.4e} "
              f"b_loss={row['b_loss']:.4e} v_viol={row['v_violation']:.6f}{flag}", flush=True)
    return rows


def compute_rank_and_delta(rows, reference_bus):
    """rank: j_net 내림차순(발산 제외). delta_vs_bus15_*: 스윕 자체의 reference_bus 결과
    대비 차이(양수 = 그 버스가 reference_bus를 이김) - "동일 고정용량 기준" 비교이므로
    비교 대상도 같은 스윕 안의 값이어야 정합적이다(외부 runs.csv 값과 비교하지 않음)."""
    valid = [r for r in rows if not r['diverged']]
    for i, r in enumerate(sorted(valid, key=lambda r: -r['j_net'])):
        r['rank'] = i + 1
    for r in rows:
        if r['diverged']:
            r['rank'] = ''

    ref_row = next((r for r in rows if r['bus'] == reference_bus and not r['diverged']), None)
    ref_j = ref_row['j_net'] if ref_row else None

    for r in rows:
        if r['diverged'] or ref_j is None:
            r['delta_vs_bus15_won'] = ''
            r['delta_vs_bus15_pct'] = ''
            continue
        delta = r['j_net'] - ref_j
        r['delta_vs_bus15_won'] = delta
        r['delta_vs_bus15_pct'] = (delta / abs(ref_j) * 100) if ref_j else float('nan')

    return ref_row


# ============================================================
# 판정 출력
# ============================================================

def report_ranking(rows, reference_bus):
    section('판정 1) j_net 내림차순 전체 표')
    valid = [r for r in rows if not r['diverged']]
    for r in sorted(valid, key=lambda r: r['rank']):
        marker = f'  <== 버스 {reference_bus}(기준)' if r['bus'] == reference_bus else ''
        print(f"  {r['rank']:2d}위  b={r['bus']:2d}  j_net={r['j_net']:.6e}  "
              f"delta={r['delta_vs_bus15_won']:+.2f}원({r['delta_vs_bus15_pct']:+.4f}%){marker}",
              flush=True)
    diverged = [r for r in rows if r['diverged']]
    if diverged:
        print(f"  (발산으로 순위 제외: b={[r['bus'] for r in diverged]})", flush=True)


def report_core_judgment(rows, reference_bus):
    section('판정 2) 핵심 판정')
    valid = [r for r in rows if not r['diverged']]
    ref_row = next((r for r in valid if r['bus'] == reference_bus), None)
    if ref_row is None:
        print(f'  버스 {reference_bus} 결과 없음(발산) - 판정 불가', flush=True)
        return

    print(f"  버스 {reference_bus}가 고정 용량 S={ref_row['S_fixed']:.5f}MVA에서 전 버스 중 "
          f"{ref_row['rank']}위 / {len(valid)}", flush=True)
    if ref_row['rank'] == 1:
        print('  판정: PSO 위치 탐색 확증 - 놓친 우월 버스 없음.', flush=True)
    else:
        best = min(valid, key=lambda r: r['rank'])
        print(f"  판정: ★ 경고 - 버스 {best['bus']}가 {reference_bus}를 "
              f"{best['delta_vs_bus15_pct']:.4f}%p 앞섬. 버스 {best['bus']}의 자기최적 용량 "
              f"재탐색 필요.", flush=True)


def report_full_buses_comparison(rows, parsed_runs, reference_bus):
    section('판정 3) full에서 실제 관측된 버스 대조')
    observed_buses = sorted({r['b'] for r in parsed_runs
                              if any(rr['b'] == r['b'] and rr['S'] > 1e-9 for rr in parsed_runs)})
    print(f'  full 실행에서 실제로 나온 버스: {observed_buses}', flush=True)
    valid = {r['bus']: r for r in rows if not r['diverged']}
    for b in observed_buses:
        r = valid.get(b)
        st = bus_stats(parsed_runs, b)
        if r is None or st is None:
            print(f'  b={b:2d}: 스윕 결과 또는 full 자료 없음', flush=True)
            continue
        print(f"  b={b:2d}: 고정용량 스윕 {r['rank']}위(delta={r['delta_vs_bus15_pct']:+.4f}%)  "
              f"| full 자기최적 {st['n']}run, median j_net={st['j_median']:.6e}원", flush=True)


def report_penalty_check(rows):
    section('판정 4) 페널티 컬럼 점검 (전압위반)')
    valid = [r for r in rows if not r['diverged']]
    viol = [r for r in valid if r['v_violation'] > 0]
    if not viol:
        print(f'  전 {len(valid)}개 버스 v_violation=0, penalty_v=0 확인.', flush=True)
    else:
        print(f'  ★ v_violation>0인 버스 {len(viol)}개 발견 - 이 고정 용량이 해당 버스엔 '
              '과대할 수 있음(해석에 반영할 것):', flush=True)
        for r in viol:
            print(f"    b={r['bus']:2d}: v_violation={r['v_violation']:.6f} pu  "
                  f"penalty_v={r['penalty_v']:.4e}원", flush=True)


def run_validations(rows, parsed_runs, reference_bus):
    section('검증 (스윕 경로가 full과 정합하는지)')
    valid = {r['bus']: r for r in rows if not r['diverged']}

    # 검증1: 스윕의 reference_bus 자기 j_net vs full runs.csv의 reference_bus median j_net.
    ref_stats = bus_stats(parsed_runs, reference_bus)
    ref_row = valid.get(reference_bus)
    if ref_row and ref_stats:
        diff = ref_row['j_net'] - ref_stats['j_median']
        ok = abs(diff) < VALIDATION1_ATOL_WON
        print(f"  검증1: 스윕 버스{reference_bus} j_net={ref_row['j_net']:.4f}원 vs "
              f"full median={ref_stats['j_median']:.4f}원  차이={diff:+.4f}원  "
              f"({'일치 - 스윕 경로가 full과 동일함' if ok else f'★차이가 허용오차({VALIDATION1_ATOL_WON}원)를 넘음'})",
              flush=True)
    else:
        print(f'  검증1: 버스{reference_bus} 자료 부족 - 생략', flush=True)

    # 검증2: 버스16(있다면) - 자기최적 용량이 버스15와 근접하므로 고정용량에서도 근접해야 함.
    b16_stats = bus_stats(parsed_runs, 16)
    b16_row = valid.get(16)
    if b16_row and b16_stats:
        diff16 = b16_row['j_net'] - b16_stats['j_median']
        pct16 = diff16 / abs(b16_stats['j_median']) * 100
        print(f"  검증2: 스윕 버스16(고정용량) j_net={b16_row['j_net']:.4f}원 vs "
              f"full 버스16 자기최적 median={b16_stats['j_median']:.4f}원  "
              f"차이={diff16:+.4f}원({pct16:+.4f}%)  "
              '(버스16 자기최적 용량이 버스15와 근접하므로 이 차이도 작아야 함)', flush=True)
    else:
        print('  검증2: 버스16 자료 없음(full에서 버스16으로 수렴한 run이 없었을 수 있음) - 생략',
              flush=True)

    # 검증3: 전 버스 diverged=False.
    n_diverged = sum(1 for r in rows if r['diverged'])
    print(f"  검증3: 발산 버스 수 = {n_diverged}/{len(rows)}  "
          f"({'통과' if n_diverged == 0 else '★ 발산 있음 - 위 목록 확인'})", flush=True)


# ============================================================
# CSV 저장
# ============================================================

def _make_path():
    out_dir = os.path.join(SCRIPT_DIR, 'results', 'probe_bus_sweep_full')
    os.makedirs(out_dir, exist_ok=True)
    hostname = socket.gethostname()
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(out_dir, f'probe_bus_sweep_full_{hostname}_{ts}.csv')


def _write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=SWEEP_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        f.flush()
        os.fsync(f.fileno())
    print(f'\nCSV 저장: {path}', flush=True)


# ============================================================
# 진입점
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=f'full 실행 최적해(버스 {REFERENCE_BUS}) 용량을 전 버스에 고정하고 '
                     '전수 스윕한다(PSO가 놓친 우월 버스 스크리닝).')
    parser.add_argument('runs_csv', help='full 실행 runs_n1_<ts>.csv 경로 (n_ess=1 전용).')
    args = parser.parse_args()

    _check_env()

    parsed_runs = load_runs(args.runs_csv)
    ref_stats = bus_stats(parsed_runs, REFERENCE_BUS)
    assert ref_stats is not None, (
        f'{args.runs_csv}에 버스{REFERENCE_BUS}로 수렴한 run이 없음 - REFERENCE_BUS 상수를 '
        '확인할 것.')

    S_fixed, E_fixed = ref_stats['S_mean'], ref_stats['E_mean']
    section(f'고정 용량 결정 (버스{REFERENCE_BUS}로 수렴한 {ref_stats["n"]}개 run 평균, '
            f'출처: {args.runs_csv})')
    print(f"  S_fixed = {S_fixed:.6f} MVA (std={ref_stats['S_std']:.2e})", flush=True)
    print(f"  E_fixed = {E_fixed:.6f} MWh (std={ref_stats['E_std']:.2e})", flush=True)
    print(f'  E/S = {E_fixed / S_fixed:.4f}', flush=True)

    evaluate.init_worker()

    rows = sweep_all_buses(S_fixed, E_fixed)
    compute_rank_and_delta(rows, REFERENCE_BUS)

    _write_csv(_make_path(), rows)

    report_ranking(rows, REFERENCE_BUS)
    report_core_judgment(rows, REFERENCE_BUS)
    report_full_buses_comparison(rows, parsed_runs, REFERENCE_BUS)
    report_penalty_check(rows)
    run_validations(rows, parsed_runs, REFERENCE_BUS)


if __name__ == '__main__':
    main()