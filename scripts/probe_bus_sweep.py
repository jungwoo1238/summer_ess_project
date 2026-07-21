"""버스(b) 위치 무차별성 확인 - 최적 (S,E)를 고정하고 b=1..32 전수 평가 (확인 전용, 본
파이프라인 미포함 - scripts/probe_large_solutions.py 등과 같은 성격).

배경: CLAUDE.md 8절-5는 "b* 분포"를 후처리 지표로 요구하며, 당초 예상은 "손실 편익이
작아 위치가 무차별할 것"이었다. 그러나 실측 편익 구조는 b_defer가 84%, b_energy(손실
포함)가 16%로 예상과 다르다. probe_large_solutions.py는 PSO가 수렴한 b=32와 b=15
"두 점"이 서로 비슷함을 보였을 뿐이고, 이는 "전체 32개 버스가 무차별"과 다른 주장이다
- 나머지 30개 버스는 평가된 적이 없고, PSO가 그 둘로 수렴했다는 사실 자체가 나머지가
도태됐을 가능성을 시사한다. 한편 전압 위반은 위치에 강하게 의존한다(같은 S=2.4에서
b=15는 0.536pu, b=32는 0.013pu로 40배 차이 - probe_large_solutions.py 실측). 이 스크립트는
b 하나만 바꿔가며 32개 전부를 실제로 평가해 이 질문에 직접 답한다.

evaluate.py/benefits.py/lower_lp.py/params.py는 수정하지 않는다(순수 소비자). n=1 입자를
4n 시그니처(b,S,E,q_ratio)로 조립해 evaluate.evaluate_particle(x, return_detail=True)에
그대로 넣는다(q_ratio=0.0 고정 - CLAUDE.md 7-A절, probe_large_solutions.py와 같은 규약).

실행: `python scripts/probe_bus_sweep.py`

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

Q_RATIO = 0.0   # 뼈대 확정값 (CLAUDE.md 2절 "뼈대에서는 Q=0으로 고정")

# 두 크기 기준점. small: main.py dev run0의 gbest 해(작은 해 - ESS 영향 자체가 작아 위치
# 차이가 안 드러날 수 있음). large: E/S=2.38(PSO 해와 동일 비율) 유지한 채 S=1.5로 확대 -
# probe_large_solutions.py 그리드에서 위반이 막 나타나기 시작하는 크기(S=1.5, b=15에서
# v_violation=0.0063pu 관측)라 위치 의존성이 드러나기 시작하는 지점으로 골랐다.
CASES = [
    dict(label='small', S=0.1764, E=0.4192),
    dict(label='large', S=1.5, E=3.57),
]

# 참고(하드코딩): probe_large_solutions.py 판정에서 나온 실제 PSO run간 편차
# (run0 b=32 j_net~3.053e6 vs run1 b=15 j_net~3.085e6, +1.04%). "무차별 판정"의 비교
# 기준으로 쓴다 - 이 스크립트가 재현하는 값이 아니라 별도 실측값이므로 상수로 고정한다.
PSO_RUN_VARIANCE_PCT = 1.04

# case33bw 위상 구간 (0-indexed). 표준 IEEE 33-bus 토폴로지(주간선 1개 + 분기 3개)를
# 0-indexed 버스번호로 옮긴 것: 주간선 0-17(슬랙에서 뻗는 본선), 분기A 18-21(bus1에서
# 분기), 분기B 22-24(bus2에서 분기), 분기C 25-32(bus5에서 분기). B_BOUNDS가 [1,32]라
# 슬랙(bus0)은 탐색범위 밖이다.
FEEDER_SEGMENTS = [
    ('주간선(0-17)', range(0, 18)),
    ('분기A(18-21)', range(18, 22)),
    ('분기B(22-24)', range(22, 25)),
    ('분기C(25-32)', range(25, 33)),
]


def _segment_of(b):
    for name, rng in FEEDER_SEGMENTS:
        if b in rng:
            return name
    return '?'


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
    _expand_to_4n 어댑터와 같은 규약, probe_large_solutions.py와 동일)."""
    x = np.array([float(b), float(S), float(E), float(q)], dtype=float)
    return evaluate.evaluate_particle(x, return_detail=True)


# ============================================================
# 버스 전수 스윕 (b=1..32, (S,E) 고정)
# ============================================================

FIELDS = [
    'case', 'S', 'E', 'b', 'segment',
    'j_net', 'b_energy', 'b_defer', 'b_arb', 'b_loss', 'cost',
    'v_violation', 'i_violation', 'fitness', 'diverged',
]


def sweep(label, S, E):
    section(f"버스 전수 스윕: {label} (S={S} MVA, E={E} MWh, E/S={E / S:.3f})")

    rows = []
    for b in range(PM.B_BOUNDS[0], PM.B_BOUNDS[1] + 1):
        detail = _eval(b, S, E)
        seg = _segment_of(b)

        if detail.get('diverged'):
            print(f'b={b:2d}({seg}): 조류계산 발산 -> 건너뜀', flush=True)
            rows.append(dict(
                case=label, S=S, E=E, b=b, segment=seg, diverged=True,
                j_net='', b_energy='', b_defer='', b_arb='', b_loss='', cost='',
                v_violation='', i_violation='', fitness=detail['fitness'],
            ))
            continue

        row = dict(
            case=label, S=S, E=E, b=b, segment=seg, diverged=False,
            j_net=detail['j_net'], b_energy=detail['b_energy'], b_defer=detail['b_defer'],
            b_arb=detail['b_arb'], b_loss=detail['b_loss'], cost=detail['cost'],
            v_violation=detail['v_violation'], i_violation=detail['i_violation'],
            fitness=detail['fitness'],
        )
        rows.append(row)
        print(f"b={b:2d}({seg:14s}) j_net={row['j_net']:.4e} b_defer={row['b_defer']:.4e} "
              f"b_energy={row['b_energy']:.4e} v_viol={row['v_violation']:.6f} "
              f"i_viol={row['i_violation']:.6f}", flush=True)

    return rows


# ============================================================
# 판정
# ============================================================

def _distribution_stats(rows):
    valid = [r for r in rows if not r['diverged']]
    j = np.array([r['j_net'] for r in valid], dtype=float)
    best = valid[int(np.argmax(j))]
    worst = valid[int(np.argmin(j))]
    range_pct = (j.max() - j.min()) / abs(j.max()) * 100 if j.max() != 0 else float('nan')
    return dict(
        n_valid=len(valid), max=float(j.max()), min=float(j.min()),
        mean=float(j.mean()), std=float(j.std()),
        best_b=best['b'], worst_b=worst['b'], range_pct=range_pct,
        j_array=j, b_array=np.array([r['b'] for r in valid]),
    )


def final_verdict(results):
    section('판정')

    stats_by_case = {}
    for label, rows in results.items():
        stats_by_case[label] = _distribution_stats(rows)

    print('1) j_net의 b에 대한 분포', flush=True)
    for label, st in stats_by_case.items():
        print(f"   [{label}] max={st['max']:.4e}(b={st['best_b']})  "
              f"min={st['min']:.4e}(b={st['worst_b']})  mean={st['mean']:.4e}  "
              f"std={st['std']:.4e}  (max-min)/|max|={st['range_pct']:.2f}%", flush=True)

    print('\n2) 무차별 판정 (전체 폭 vs PSO run간 편차 참고값)', flush=True)
    for label, st in stats_by_case.items():
        print(f"   [{label}] 폭={st['range_pct']:.2f}%  vs  PSO run간 편차(참고)="
              f"{PSO_RUN_VARIANCE_PCT:.2f}%", flush=True)
        if st['range_pct'] < PSO_RUN_VARIANCE_PCT:
            print(f"     판정: 위치 무차별. b*가 run마다 달라지는 것은 탐색 실패가 아니라 "
                  '지형이 평탄하기 때문이다.', flush=True)
        else:
            print(f"     판정: 위치가 유의미하다(폭이 PSO 자체 편차보다 큼). 상위권 버스를 "
                  '나열한다:', flush=True)
            valid = [r for r in results[label] if not r['diverged']]
            top5 = sorted(valid, key=lambda r: -r['j_net'])[:5]
            for r in top5:
                print(f"       b={r['b']:2d}({r['segment']}) j_net={r['j_net']:.4e} "
                      f"v_viol={r['v_violation']:.6f}", flush=True)

    print('\n3) 버스 번호와 j_net의 관계 (case33bw 위상: 주간선 0-17 / 분기A 18-21 / '
          '분기B 22-24 / 분기C 25-32)', flush=True)
    for label, rows in results.items():
        valid = [r for r in rows if not r['diverged']]
        corr = float(np.corrcoef(
            [r['b'] for r in valid], [r['j_net'] for r in valid])[0, 1])
        print(f"   [{label}] 버스번호-j_net 상관계수 = {corr:+.3f} "
              f"({'번호가 클수록(말단) 나쁨' if corr < -0.3 else '번호가 클수록(말단) 좋음' if corr > 0.3 else '뚜렷한 선형 추세 없음'})",
              flush=True)
        print('     구간별 평균 j_net:', flush=True)
        for seg_name, _seg_range in FEEDER_SEGMENTS:
            seg_vals = [r['j_net'] for r in valid if r['segment'] == seg_name]
            if seg_vals:
                print(f"       {seg_name:14s}: 평균={np.mean(seg_vals):.4e}  "
                      f"n={len(seg_vals)}", flush=True)

    print('\n4) 작은 해 vs 큰 해에서 위치 의존성이 다른가', flush=True)
    if 'small' in stats_by_case and 'large' in stats_by_case:
        small_pct = stats_by_case['small']['range_pct']
        large_pct = stats_by_case['large']['range_pct']
        print(f"   small 폭={small_pct:.2f}%  large 폭={large_pct:.2f}%", flush=True)
        if large_pct > small_pct * 1.5:
            print('   판정: 크기가 커질수록 위치 의존성이 강해진다 - "무차별"은 작은 해에서만 '
                  '성립하는(크기 의존적인) 성질이다.', flush=True)
        elif small_pct > large_pct * 1.5:
            print('   판정: 오히려 작은 해에서 위치 의존성이 더 크다(예상 밖 - 원인 재검토 필요).',
                  flush=True)
        else:
            print('   판정: 두 크기에서 위치 의존성 폭이 비슷하다 - "무차별" 여부는 크기와 '
                  '무관한 성질로 보인다.', flush=True)


# ============================================================
# CSV 저장
# ============================================================

def _make_path():
    """scripts/ 바로 아래가 아니라 scripts/results/probe_bus_sweep/ 밑에 모아 둔다(스크립트별
    결과 CSV가 늘어나며 scripts/가 난잡해지는 것을 막기 위함)."""
    out_dir = os.path.join(SCRIPT_DIR, 'results', 'probe_bus_sweep')
    os.makedirs(out_dir, exist_ok=True)
    hostname = socket.gethostname()
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(out_dir, f'probe_bus_sweep_{hostname}_{ts}.csv')


def _write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())
    print(f'CSV 저장: {path}', flush=True)


if __name__ == '__main__':
    _check_env()

    evaluate.init_worker()  # 1회만 호출, 기저 조류계산 120회 캐싱 재사용

    results = {}
    all_rows = []
    for case in CASES:
        rows = sweep(case['label'], case['S'], case['E'])
        results[case['label']] = rows
        all_rows.extend(rows)

    _write_csv(_make_path(), all_rows)

    final_verdict(results)
