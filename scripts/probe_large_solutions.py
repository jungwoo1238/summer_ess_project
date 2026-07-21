"""main.py dev 실행(슬랙 1.02) 최적해가 작게 나온 원인 판정 (확인 전용, 본 파이프라인
미포함 - scripts/probe_voltage.py, check_balance.py, probe_noise.py, bench_workers.py와
같은 성격).

배경: PSO 최적해가 S=0.176 MVA(상한 2.4의 7.3%), E=0.419 MWh(상한 10.2의 4.1%)로
탐색경계 대비 매우 작게 나왔다(j_net=3.05e6, b_energy=2.12e6, b_defer=1.145e7,
cost=1.051e7 - main.py --diagnose 실측). 이것이 (a) 경제성의 실제 한계인지,
(b) PSO가 큰 해를 못 찾은 탐색 실패인지를, 큰 해를 강제로 evaluate에 넣어 j_net을
직접 비교해 판정한다.

evaluate.py/benefits.py/lower_lp.py/params.py는 수정하지 않는다(순수 소비자). n=1
입자를 4n 시그니처(b,S,E,q_ratio)로 조립해 evaluate.evaluate_particle(x,
return_detail=True)에 그대로 넣는다(q_ratio=0.0 고정 - 뼈대 PSO는 3n이지만 evaluate는
4n 시그니처를 유지한다, CLAUDE.md 7-A절). 워커별 기저 조류계산 캐싱(evaluate._BASE_FLOW)도
init_worker() 1회 호출 후 그대로 재사용한다(재구현 안 함, probe_voltage.py와 같은 패턴).

실행: `python scripts/probe_large_solutions.py`

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

# main.py dev 실행(--diagnose)에서 관측된 PSO 최적해 (비교 기준점, 하드코딩 - 이
# 스크립트의 유일한 목적이 이 값이 탐색실패인지 경제성한계인지 판정하는 것이므로
# probe_voltage.py의 MAIN_DIAGNOSED_V_VIOLATION과 같은 패턴으로 남겨둔다).
PSO_BEST = dict(
    b=32, S=0.176, E=0.419,
    j_net=3.05e6, b_energy=2.12e6, b_defer=1.145e7, cost=1.051e7,
)
# run1이 다른 (S,E)에서 찾은 b=15 해 (참고용 - 이 스크립트가 재현하는 값이 아니라
# 실제 PSO 실행 결과. 측정1의 b=32 vs b=15 그리드 비교와는 별개 사실이다).
PSO_RUN0_B32_J_NET = 3.053e6
PSO_RUN1_B15_J_NET = 3.085e6

BUSES = [32, 15]
S_GRID = [0.176, 0.5, 1.0, 1.5, 2.0, 2.4]          # MVA, 0.176 = PSO 최적해
RATIO_GRID = [2.38, 3.0, 3.81, 4.23]                # E/S. 2.38=PSO해, 3.81=손익분기(6절), 4.23=x_max

PEAK_CURVE_BUS = 32
PEAK_CURVE_RATIO = 2.38
PEAK_CURVE_S_GRID = [0.05, 0.1, 0.176, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.4]

Q_RATIO = 0.0   # 뼈대 확정값 (CLAUDE.md 2절 "뼈대에서는 Q=0으로 고정")


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
# 측정 1: 그리드 스윕 (주 측정)
# ============================================================

GRID_FIELDS = [
    'b', 'S', 'E', 'e_over_s', 'skipped', 'skip_reason',
    'j_net', 'b_energy', 'b_defer', 'b_arb', 'b_loss', 'cost',
    'v_violation', 'i_violation', 'fitness', 'decomposition_ok',
]


def measurement1():
    section('측정 1: 그리드 스윕 (b in {32,15} x S in [0.176..2.4] x E/S in [2.38..4.23])')

    rows = []
    for b in BUSES:
        for S in S_GRID:
            for ratio in RATIO_GRID:
                E = S * ratio
                if E > PM.E_BOUNDS[1]:
                    reason = f'E={E:.3f} MWh > E_BOUNDS 상한 {PM.E_BOUNDS[1]} MWh'
                    print(f'b={b:2d} S={S:.3f} E/S={ratio:.2f} -> E={E:.3f} 건너뜀 ({reason})',
                          flush=True)
                    rows.append(dict(
                        b=b, S=S, E=round(E, 3), e_over_s=round(ratio, 3),
                        skipped=True, skip_reason=reason,
                        j_net='', b_energy='', b_defer='', b_arb='', b_loss='', cost='',
                        v_violation='', i_violation='', fitness='', decomposition_ok='',
                    ))
                    continue

                detail = _eval(b, S, E)
                if detail.get('diverged'):
                    print(f'b={b:2d} S={S:.3f} E={E:.3f}(E/S={ratio:.2f}): 조류계산 발산 -> 건너뜀',
                          flush=True)
                    rows.append(dict(
                        b=b, S=S, E=round(E, 3), e_over_s=round(ratio, 3),
                        skipped=True, skip_reason='조류계산 발산(PENALTY_DIVERGE)',
                        j_net='', b_energy='', b_defer='', b_arb='', b_loss='', cost='',
                        v_violation='', i_violation='', fitness=detail['fitness'], decomposition_ok='',
                    ))
                    continue

                row = dict(
                    b=b, S=S, E=round(E, 3), e_over_s=round(ratio, 3),
                    skipped=False, skip_reason='',
                    j_net=detail['j_net'], b_energy=detail['b_energy'], b_defer=detail['b_defer'],
                    b_arb=detail['b_arb'], b_loss=detail['b_loss'], cost=detail['cost'],
                    v_violation=detail['v_violation'], i_violation=detail['i_violation'],
                    fitness=detail['fitness'], decomposition_ok=detail['decomposition_ok'],
                )
                rows.append(row)
                print(f"b={b:2d} S={S:.3f} E={E:.3f}(E/S={ratio:.3f}) "
                      f"j_net={row['j_net']:.4e} b_energy={row['b_energy']:.4e} "
                      f"b_defer={row['b_defer']:.4e} cost={row['cost']:.4e} "
                      f"v_viol={row['v_violation']:.6f} i_viol={row['i_violation']:.6f}",
                      flush=True)

    return rows


# ============================================================
# 측정 2: 피크 저감 곡선 (원인 규명)
# ============================================================

PEAK_FIELDS = [
    'b', 'S', 'E', 'e_over_s',
    'b_defer', 'peak_reduction_mw',
    'overall_base_max_mw', 'overall_base_scenario', 'overall_base_t',
    'overall_ess_max_mw', 'overall_ess_scenario', 'overall_ess_t',
    'summer_peak_base_max_mw', 'summer_peak_base_t',
    'summer_peak_ess_max_mw', 'summer_peak_ess_t',
    'winter_peak_base_max_mw', 'winter_peak_base_t',
    'winter_peak_ess_max_mw', 'winter_peak_ess_t',
    'j_net', 'cost',
    'marginal_b_defer_won_per_mva', 'marginal_cost_won_per_mva', 'marginal_j_net_won_per_mva',
]


def measurement2():
    section(f'측정 2: 피크 저감 곡선 (b={PEAK_CURVE_BUS} 고정, E/S={PEAK_CURVE_RATIO} 고정, S 스윕)')

    base_p_slack = evaluate._BASE_FLOW['p_slack']  # init_worker()가 이미 채워 둠 (재사용, 재계산 안 함)

    rows = []
    prev = None
    for S in PEAK_CURVE_S_GRID:
        E = S * PEAK_CURVE_RATIO
        detail = _eval(PEAK_CURVE_BUS, S, E)
        if detail.get('diverged'):
            print(f'S={S:.3f} E={E:.3f}: 조류계산 발산 -> 건너뜀', flush=True)
            continue

        per_scenario = {}
        overall_base = (-np.inf, None, None)
        overall_ess = (-np.inf, None, None)
        for s in PM.PEAK_DAYS:
            base_arr = base_p_slack[s]
            ess_arr = detail['p_slack_ess'][s]
            bt = int(np.argmax(base_arr))
            bm = float(base_arr[bt])
            et = int(np.argmax(ess_arr))
            em = float(ess_arr[et])
            per_scenario[s] = dict(base_max=bm, base_t=bt, ess_max=em, ess_t=et)
            if bm > overall_base[0]:
                overall_base = (bm, s, bt)
            if em > overall_ess[0]:
                overall_ess = (em, s, et)

        peak_reduction_mw = detail['b_defer'] / PM.C_CAP_PER_MW_YR

        row = dict(
            b=PEAK_CURVE_BUS, S=S, E=round(E, 3), e_over_s=PEAK_CURVE_RATIO,
            b_defer=detail['b_defer'], peak_reduction_mw=peak_reduction_mw,
            overall_base_max_mw=overall_base[0], overall_base_scenario=overall_base[1],
            overall_base_t=overall_base[2],
            overall_ess_max_mw=overall_ess[0], overall_ess_scenario=overall_ess[1],
            overall_ess_t=overall_ess[2],
            summer_peak_base_max_mw=per_scenario['summer_peak']['base_max'],
            summer_peak_base_t=per_scenario['summer_peak']['base_t'],
            summer_peak_ess_max_mw=per_scenario['summer_peak']['ess_max'],
            summer_peak_ess_t=per_scenario['summer_peak']['ess_t'],
            winter_peak_base_max_mw=per_scenario['winter_peak']['base_max'],
            winter_peak_base_t=per_scenario['winter_peak']['base_t'],
            winter_peak_ess_max_mw=per_scenario['winter_peak']['ess_max'],
            winter_peak_ess_t=per_scenario['winter_peak']['ess_t'],
            j_net=detail['j_net'], cost=detail['cost'],
        )

        if prev is not None:
            dS = S - prev['S']
            row['marginal_b_defer_won_per_mva'] = (row['b_defer'] - prev['b_defer']) / dS
            row['marginal_cost_won_per_mva'] = (row['cost'] - prev['cost']) / dS
            row['marginal_j_net_won_per_mva'] = (row['j_net'] - prev['j_net']) / dS
        else:
            row['marginal_b_defer_won_per_mva'] = ''
            row['marginal_cost_won_per_mva'] = ''
            row['marginal_j_net_won_per_mva'] = ''

        rows.append(row)
        prev = row

        marg_str = (f"{row['marginal_j_net_won_per_mva']:.4e}"
                    if row['marginal_j_net_won_per_mva'] != '' else '(첫 지점)')
        print(f"S={S:.3f} E={E:.3f} b_defer={row['b_defer']:.4e} "
              f"피크저감={peak_reduction_mw * 1000:.1f}kW "
              f"기저최대={overall_base[0]:.4f}MW({overall_base[1]},t={overall_base[2]}) "
              f"ESS최대={overall_ess[0]:.4f}MW({overall_ess[1]},t={overall_ess[2]}) "
              f"j_net={row['j_net']:.4e} 한계j_net={marg_str}",
              flush=True)

    section('측정 2 - 한계 j_net 부호 전환 구간')
    sign_change_found = False
    for i in range(1, len(rows)):
        prev_m = rows[i - 1]['marginal_j_net_won_per_mva']
        cur_m = rows[i]['marginal_j_net_won_per_mva']
        if prev_m == '' or cur_m == '':
            continue
        if prev_m > 0 and cur_m <= 0:
            sign_change_found = True
            print(f"  S={rows[i - 1]['S']:.3f}(한계j_net={prev_m:.4e}) -> "
                  f"S={rows[i]['S']:.3f}(한계j_net={cur_m:.4e}) 구간에서 양(+)->음(-)으로 전환 "
                  '-> 이 구간이 한계수익=한계비용인 최적 S 부근', flush=True)
    if not sign_change_found:
        print('  스윕 범위 내에서 한계 j_net 부호 전환 없음 (전 구간 양수 또는 전 구간 음수).',
              flush=True)

    return rows


# ============================================================
# 측정 3: 페널티 경로 생존 확인 (부수 목적)
# ============================================================

def measurement3(grid_rows):
    section('측정 3: 페널티 경로 생존 확인 (v_violation > 0인 조합이 있는가)')

    valid = [r for r in grid_rows if not r['skipped']]
    viol = [r for r in valid if r['v_violation'] > 0]

    print(f'유효 그리드 조합 {len(valid)}개 중 v_violation > 0: {len(viol)}개', flush=True)

    if viol:
        worst = max(viol, key=lambda r: r['v_violation'])
        penalty_v = PM.LAMBDA_V * worst['v_violation']
        j_net_abs = abs(worst['j_net']) if worst['j_net'] != 0 else float('inf')
        ratio = penalty_v / j_net_abs
        print(f"  최대 v_violation 조합: b={worst['b']} S={worst['S']} E={worst['E']:.3f} "
              f"v_violation={worst['v_violation']:.6f} pu", flush=True)
        print(f"  λ_V x v_violation = {penalty_v:.4e} 원, |j_net| = {j_net_abs:.4e} 원, "
              f"비율 = {ratio:.2f}배", flush=True)
        print('  -> 페널티 경로가 살아 있다. 위 배율로 λ_V가 이 크기의 위반에 적절한지 판단 가능.',
              flush=True)
    else:
        print('  48 조합 전부 v_violation=0. S=2.4 MVA 최대충전으로도 전압 위반이 나지 않는다는 '
              '뜻이다 - 이 그리드 범위 안에서는 페널티 경로의 실동작을 확인할 수 없다 '
              '(CLAUDE.md 3-B절 미결 2).', flush=True)

    return dict(n_valid=len(valid), n_violated=len(viol))


# ============================================================
# 판정
# ============================================================

def final_verdict(grid_rows, peak_rows):
    section('판정')

    valid = [r for r in grid_rows if not r['skipped']]

    print('1) PSO 해가 최적인가', flush=True)
    better = [r for r in valid if r['j_net'] > PSO_BEST['j_net']]
    if better:
        best = max(better, key=lambda r: r['j_net'])
        gain = best['j_net'] - PSO_BEST['j_net']
        pct = gain / PSO_BEST['j_net'] * 100
        print(f"   판정: ★ PSO 탐색 실패. 그리드에 PSO 해보다 큰 j_net이 있다.", flush=True)
        print(f"   최우수: b={best['b']} S={best['S']} E={best['E']:.3f} j_net={best['j_net']:.4e} "
              f"(PSO 대비 +{gain:.4e}원, +{pct:.1f}%)", flush=True)
        print('   그리드 상위 5개:', flush=True)
        for r in sorted(better, key=lambda r: -r['j_net'])[:5]:
            print(f"     b={r['b']:2d} S={r['S']:.3f} E={r['E']:.3f}(E/S={r['e_over_s']:.2f}) "
                  f"j_net={r['j_net']:.4e}", flush=True)
    else:
        print(f"   판정: PSO 해가 타당하다. 48 조합 중 PSO 해(j_net={PSO_BEST['j_net']:.4e})를 "
              '넘는 조합이 없다 - 작은 해는 경제성의 실제 한계다.', flush=True)

    print('\n2) b_defer 포화 여부 (측정 2 참조)', flush=True)
    marg = [r for r in peak_rows if r['marginal_b_defer_won_per_mva'] != '']
    if marg:
        seq = [m['marginal_b_defer_won_per_mva'] for m in marg]
        decreasing = all(seq[i] >= seq[i + 1] - 1e-6 for i in range(len(seq) - 1))
        print(f"   한계 b_defer 추이(S 증가 순): {[f'{v:.3e}' for v in seq]}", flush=True)
        if decreasing:
            first_small = next((m for m in marg if m['marginal_b_defer_won_per_mva'] < seq[0] * 0.1),
                                None)
            where = f"S={first_small['S']:.3f} 부근" if first_small else '(뚜렷한 꺾임점 없음)'
            print(f'   판정: 단조감소(포화 패턴 확인). 한계 b_defer가 초기값의 10% 미만으로 '
                  f'떨어지는 지점: {where}', flush=True)
        else:
            print('   판정: 단조감소가 아니다 - 포화 패턴이 명확하지 않음(추가 확인 필요).', flush=True)
    else:
        print('   판정 불가(유효 지점 부족).', flush=True)

    print('\n3) 버스 의존성 (b=32 vs b=15, 그리드 동일 (S,E/S) 대조)', flush=True)
    by_key = {}
    for r in valid:
        by_key.setdefault((r['S'], r['e_over_s']), {})[r['b']] = r['j_net']
    any_pair = False
    max_abs_pct = 0.0
    for (S, ratio), d in sorted(by_key.items()):
        if 32 in d and 15 in d:
            any_pair = True
            diff = d[15] - d[32]
            pct = diff / abs(d[32]) * 100 if d[32] != 0 else float('nan')
            max_abs_pct = max(max_abs_pct, abs(pct))
            print(f"   S={S:.3f} E/S={ratio:.2f}: j_net(b=32)={d[32]:.4e} j_net(b=15)={d[15]:.4e} "
                  f"차이={diff:+.4e}원({pct:+.2f}%)", flush=True)
    if any_pair:
        print(f'   그리드 대조 최대 절대 차이: {max_abs_pct:.2f}%', flush=True)
    else:
        print('   대조 가능한 (S,E/S) 쌍이 없음.', flush=True)

    pso_diff_pct = (PSO_RUN1_B15_J_NET / PSO_RUN0_B32_J_NET - 1) * 100
    print(f"\n   참고(실제 PSO 결과, main.py --diagnose, 서로 다른 (S,E)에서 나온 값 - "
          f"위 그리드 대조와는 별개 사실): run0 b=32 j_net~{PSO_RUN0_B32_J_NET:.3e}원, "
          f"run1 b=15 j_net~{PSO_RUN1_B15_J_NET:.3e}원 ({pso_diff_pct:+.2f}%)", flush=True)


# ============================================================
# CSV 저장
# ============================================================

def _make_path(suffix):
    hostname = socket.gethostname()
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(SCRIPT_DIR, f'probe_large_solutions_{suffix}_{hostname}_{ts}.csv')


def _write_csv(path, fields, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())
    print(f'CSV 저장: {path}', flush=True)


if __name__ == '__main__':
    _check_env()

    evaluate.init_worker()  # 1회만 호출, 기저 조류계산 120회 캐싱 재사용 (재초기화 안 함)

    grid_rows = measurement1()
    peak_rows = measurement2()
    m3 = measurement3(grid_rows)

    _write_csv(_make_path('grid'), GRID_FIELDS, grid_rows)
    _write_csv(_make_path('peakcurve'), PEAK_FIELDS, peak_rows)

    final_verdict(grid_rows, peak_rows)
