"""후처리 지표 계산 (CLAUDE.md 8절, 부록A). runs.csv/generations.csv를 읽어 편익 분해,
운영·설비 지표, 수렴 진단, b* 분포, 스케줄 유사도 등을 계산·출력한다.

**최적화에 개입하지 않는 사후분석 모듈이다.** PSO·evaluate 경로를 수정하지 않고, 이미
저장된 로그(runs.csv/generations.csv)와 lower_lp/benefits/evaluate의 기존 함수만 재사용한다.
evaluate.py/benefits.py/lower_lp.py/pso_core.py/main.py는 전부 미수정 - 필요한 값이 로그에
없으면 그 사실을 보고할 뿐 main.py를 고치지 않는다(이 게이트의 산출물 그 자체).

**full 실행 진입 전 게이트다.** dev 결과(n=1, n=2, n=3)로 완주 확인이 끝나야 full로 넘어간다.

실행: `python postprocess.py results/runs_n1_<ts>.csv [results/runs_n2_<ts>.csv ...]`
      generations.csv는 파일명 규약(runs_ -> generations_)으로 자동 매칭한다.
      probe_bus_sweep.py 결과가 있으면 --bus-sweep-csv로 넘기거나(없으면
      scripts/results/probe_bus_sweep/ 밑에서 최신 파일을 자동 탐색한다).

# ------------------------------------------------------------------
# 검증 기록 (실행 후 채울 것 - scripts/ 규약과 동일하게 남긴다)
#   실행 일시:
#   대상 파일:
#   결론:
# ------------------------------------------------------------------
"""
import os
import sys
import csv
import json
import math
import glob
import argparse
import datetime
from collections import defaultdict

import numpy as np

import params as PM
import benefits
import lower_lp
import evaluate

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR_DEFAULT = os.path.join(PROJECT_ROOT, 'results')
BUS_SWEEP_DIR_DEFAULT = os.path.join(PROJECT_ROOT, 'scripts', 'results', 'probe_bus_sweep')

# ============================================================
# 임계값·허용오차 (CLAUDE.md 7절 원칙4, 8절 지시사항 그대로 - 숫자를 여기 한 곳에만 둔다)
# ============================================================
DEGENERATE_J_NET_WON = 1e3        # 이 미만이면 "전 기 소멸(탐색 실패)" run (8절-5 신규)
NEG_BDEFER_NOISE_ATOL_WON = 1.0   # |min_bdefer|가 이 미만이면 부동소수 잡음 (7절 6.91원 상한보다 보수적)
MONEY_NEAR_ZERO_WON = 1.0         # 이 미만이면 원칙4의 "기대값=0" 분기(atol=10.0)를 쓴다
DECOMP_RTOL = 1e-9                # 원칙4: 금액·기대값!=0 -> rtol=1e-9, atol=0
DECOMP_ATOL_NEAR_ZERO = 10.0      # 원칙4: 금액·기대값=0 -> atol=10.0 (test_evaluate.py won_atol과 동일 계보)
BOUNDARY_WARN_FRAC = 0.02         # 정규화 위치가 이 미만/((1-이 값) 초과)면 경계 근접 경고
SOC_CYCLE_ATOL_MWH = 1e-4         # SOC[24]==SOC[0] 검증 허용오차 (lower_lp._assert_physics의 tol과 동일)


def section(title):
    print('\n' + '=' * 78, flush=True)
    print(title, flush=True)
    print('=' * 78, flush=True)


def _check_env():
    val = os.environ.get('MKL_THREADING_LAYER')
    print(f'MKL_THREADING_LAYER = {val!r}', flush=True)
    if val != 'SEQUENTIAL':
        print("경고: MKL_THREADING_LAYER가 'SEQUENTIAL'이 아닙니다 - (9)절이 조류계산을 위해 "
              "evaluate.evaluate_particle을 호출하므로 워커 스레드 풀 생성 시점에 무증상 종료 "
              "위험이 있습니다(CLAUDE.md 7절). "
              "`conda env config vars set MKL_THREADING_LAYER=SEQUENTIAL -n ess`로 설정하세요.",
              flush=True)


# ============================================================
# 로딩 (runs.csv / generations.csv / probe_bus_sweep.csv)
# ============================================================

NUMERIC_RUN_FIELDS = [
    'gbest_f', 'wall_time_s', 'n_diverged_total', 'n_negative_bdefer', 'min_bdefer',
    'j_net', 'b_energy', 'b_defer', 'b_arb', 'b_loss', 'cost',
    'v_violation', 'i_violation', 'penalty_v', 'penalty_line',
]


def _to_float(s):
    if s is None or s == '':
        return None
    return float(s)


def _to_bool(s):
    return str(s).strip().lower() in ('true', '1')


def load_runs_csv(path):
    """runs.csv -> (parsed rows, 결측 컬럼 목록). 결측 컬럼은 죽이지 않고 None으로 채워
    해당 지표만 생략한다(main.py를 고치지 말고 보고하라는 지시사항의 핵심 구현)."""
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    missing_cols = set()
    parsed = []
    for row in rows:
        rec = {}
        for k in NUMERIC_RUN_FIELDS:
            if k not in row:
                missing_cols.add(k)
                rec[k] = None
            else:
                rec[k] = _to_float(row[k])
        rec['n_ess'] = int(float(row['n_ess']))
        rec['run'] = int(float(row['run']))
        rec['seed'] = row.get('seed', '')
        rec['x_json'] = row.get('x_json', '')
        rec['x'] = np.array(json.loads(rec['x_json']), dtype=float) if rec['x_json'] else np.array([])
        rec['decomposition_ok'] = _to_bool(row.get('decomposition_ok', ''))
        parsed.append(rec)

    return parsed, sorted(missing_cols)


def _derive_generations_path(runs_path):
    base = os.path.basename(runs_path)
    if not base.startswith('runs_'):
        return None
    return os.path.join(os.path.dirname(runs_path), 'generations_' + base[len('runs_'):])


def load_generations_csv(path):
    if path is None or not os.path.exists(path):
        return None
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ('n_ess', 'run', 'gen'):
            r[k] = int(float(r[k]))
        for k in ('gbest_f', 'mean_f', 'std_f', 'worst_f', 'elapsed_s'):
            r[k] = float(r[k])
    return rows


def _find_latest_bus_sweep_csv():
    candidates = sorted(glob.glob(os.path.join(BUS_SWEEP_DIR_DEFAULT, 'probe_bus_sweep_*.csv')))
    return candidates[-1] if candidates else None


def load_bus_sweep_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    by_case = defaultdict(list)
    for r in rows:
        if str(r.get('diverged', 'False')).strip() == 'True':
            continue
        by_case[r['case']].append(dict(b=int(r['b']), S=float(r['S']), j_net=float(r['j_net'])))
    return dict(by_case)


# ============================================================
# 입자 파싱 (3n PSO 벡터 -> (b,S,E) 유닛 리스트, CLAUDE.md 7-A절)
# ============================================================

def parse_units(x3, n_ess):
    """x(3n, pso_core.optimize()가 반환한 반올림 적용값) -> [(b:int,S:float,E:float), ...].
    b가 정수에 가깝지 않으면 _decode 반올림 가정이 깨진 것이므로 경고한다(지시사항의 '확인할 것')."""
    x3 = np.asarray(x3, dtype=float).reshape(n_ess, 3)
    units = []
    for b_raw, S, E in x3:
        if abs(b_raw - round(b_raw)) > 1e-6:
            print(f'  ★ 경고: b={b_raw}가 정수가 아님 - pso_core._decode가 이미 반올림했을 것이라는 '
                  '가정(CLAUDE.md 7-A절)이 깨졌을 수 있음. x_json이 반올림 전 값을 저장했는지 '
                  'main.py를 확인할 것(이 모듈은 수정하지 않음).', flush=True)
        units.append((int(round(b_raw)), float(S), float(E)))
    return units


def installed_only(units, eps=1e-9):
    return [(b, S, E) for b, S, E in units if S > eps and E > eps]


def _expand_to_4n(units):
    """main.py의 _expand_to_4n과 같은 어댑터 - evaluate.evaluate_particle의 4n 시그니처에 맞춘다."""
    x4 = np.zeros(4 * len(units), dtype=float)
    for i, (b, S, E) in enumerate(units):
        x4[4 * i:4 * i + 4] = [b, S, E, 0.0]
    return x4


# ============================================================
# 통계 유틸
# ============================================================

def _stats(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not values:
        return dict(n=0, median=float('nan'), q1=float('nan'), q3=float('nan'),
                    min=float('nan'), max=float('nan'))
    arr = np.array(values, dtype=float)
    return dict(n=len(arr), median=float(np.median(arr)), q1=float(np.percentile(arr, 25)),
                q3=float(np.percentile(arr, 75)), min=float(arr.min()), max=float(arr.max()))


def _print_stats(name, st, unit=''):
    if st['n'] == 0:
        print(f'    {name:16s}: (자료 없음)', flush=True)
        return
    print(f"    {name:16s}: n={st['n']:3d}  median={st['median']:+.4e}{unit}  "
          f"IQR=[{st['q1']:+.4e}, {st['q3']:+.4e}]  range=[{st['min']:+.4e}, {st['max']:+.4e}]",
          flush=True)


# ============================================================
# (1) 편익 분해 + 검산
# ============================================================

def check_decomposition(b_arb, b_loss, b_energy):
    """b_arb+b_loss ~= b_energy. CLAUDE.md 7절 원칙4: 기대값(b_energy) 크기에 따라 규칙이
    갈린다 - 거의 0(전 기 소멸 run)이면 절대오차, 아니면 상대오차."""
    lhs = b_arb + b_loss
    if abs(b_energy) < MONEY_NEAR_ZERO_WON:
        ok = abs(lhs - b_energy) <= DECOMP_ATOL_NEAR_ZERO
        rule = f'atol={DECOMP_ATOL_NEAR_ZERO}(기대값~0)'
    else:
        ok = bool(np.isclose(lhs, b_energy, rtol=DECOMP_RTOL, atol=0.0))
        rule = f'rtol={DECOMP_RTOL}'
    return ok, rule, lhs - b_energy


def section1_benefit_decomposition(group):
    section(f"(1) 편익 분해 - n_ess={group['n_ess']}")
    runs = group['runs']

    if 'j_net' in group['missing_cols']:
        print('  j_net 등 편익 컬럼이 로그에 없음(구 스키마 - main.py 실행 이전 데이터로 추정) '
              '- 이 섹션 생략', flush=True)
        return None

    degenerate = [r for r in runs if r['j_net'] < DEGENERATE_J_NET_WON]
    normal = [r for r in runs if r['j_net'] >= DEGENERATE_J_NET_WON]
    print(f'  전체 {len(runs)}건 중 탐색실패(전 기 소멸, j_net<{DEGENERATE_J_NET_WON:.0e}원) '
          f'{len(degenerate)}건, 정상 {len(normal)}건', flush=True)

    for label, subset in (('전체(탐색실패 포함)', runs), ('정상만(탐색실패 제외)', normal)):
        print(f'  [{label}, n={len(subset)}]', flush=True)
        for field in ('j_net', 'b_energy', 'b_defer', 'b_arb', 'b_loss', 'cost'):
            _print_stats(field, _stats([r[field] for r in subset]), '원')

    print('  편익 비중(정상 run만, run별 j_net 대비 %):', flush=True)
    for field in ('b_defer', 'b_arb', 'b_loss'):
        ratios = [r[field] / r['j_net'] * 100 for r in normal if r['j_net']]
        _print_stats(f'{field}/j_net', _stats(ratios), '%')
    print('  (예상: b_defer >> b_arb > b_loss)', flush=True)

    bad = []
    for r in runs:
        ok, rule, diff = check_decomposition(r['b_arb'], r['b_loss'], r['b_energy'])
        if not ok:
            bad.append((r['run'], diff, rule))
    if bad:
        print(f'  ★ b_arb+b_loss≈b_energy 위반 {len(bad)}건:', flush=True)
        for run_idx, diff, rule in bad:
            print(f'    run={run_idx}  차이={diff:+.4f}원 ({rule})', flush=True)
    else:
        print(f'  b_arb+b_loss≈b_energy 검산: 전 {len(runs)}건 통과', flush=True)

    return dict(degenerate=degenerate, normal=normal, decomposition_bad=bad)


# ============================================================
# (2) 비용 재계산
# ============================================================

def check_cost(units, cost_reported):
    s_total = sum(u[1] for u in units)
    e_total = sum(u[2] for u in units)
    cost_recomputed = benefits.total_cost(s_total, e_total)
    if abs(cost_reported) < MONEY_NEAR_ZERO_WON:
        ok = abs(cost_recomputed - cost_reported) <= DECOMP_ATOL_NEAR_ZERO
    else:
        ok = bool(np.isclose(cost_recomputed, cost_reported, rtol=DECOMP_RTOL, atol=0.0))
    return ok, cost_recomputed


def section2_cost_check(group):
    section(f"(2) 비용 재계산 검증 - n_ess={group['n_ess']}")
    if 'cost' in group['missing_cols']:
        print('  cost 컬럼 없음(구 스키마) - 생략', flush=True)
        return None

    bad = []
    for r in group['runs']:
        units = parse_units(r['x'], group['n_ess'])
        ok, recomputed = check_cost(units, r['cost'])
        if not ok:
            bad.append((r['run'], r['cost'], recomputed))
    if bad:
        print(f'  ★ cost 재계산 불일치 {len(bad)}건:', flush=True)
        for run_idx, reported, recomputed in bad:
            print(f'    run={run_idx}  runs.csv={reported:.2f}원  재계산={recomputed:.2f}원', flush=True)
    else:
        print(f'  cost 재계산 검증: 전 {len(group["runs"])}건 일치 (benefits.total_cost 회귀 확인)',
              flush=True)
    return dict(ok=(not bad), bad=bad)


# ============================================================
# best run 선택
# ============================================================

def pick_best_run(group):
    valid = [r for r in group['runs'] if r['gbest_f'] is not None]
    if not valid:
        return None
    return min(valid, key=lambda r: r['gbest_f'])


# ============================================================
# (3)-1,2,3 운영·설비 지표 (η_peak, 지속시간, E/S)
# ============================================================

def section3_operating_metrics(group, best):
    section(f"(3) 운영·설비 지표 (최적해, n_ess={group['n_ess']}, "
            f"최우수 run={best['run'] if best else '없음'})")
    if best is None:
        print('  유효 run 없음 - 생략', flush=True)
        return None

    units = parse_units(best['x'], group['n_ess'])
    installed = installed_only(units)
    print(f'  units(전체)={units}', flush=True)
    print(f'  설치된 기(S>0 and E>0)={installed}', flush=True)

    s_total = sum(u[1] for u in units)
    e_total = sum(u[2] for u in units)
    result = dict(units=units, installed=installed, s_total=s_total, e_total=e_total)

    if best['b_defer'] is None:
        print('  b_defer 없음(구 스키마) - eta_peak 등 생략', flush=True)
        return result
    if s_total <= 0:
        print('  s_total=0 (전 기 소멸) - eta_peak/지속시간/E-S비 정의 불가, 생략', flush=True)
        return result

    delta_p_peak_mw = best['b_defer'] / PM.C_CAP_PER_MW_YR
    eta_peak = delta_p_peak_mw / s_total
    duration_h = e_total * PM.DOD / s_total
    e_over_s = e_total / s_total

    print(f'  1) eta_peak = {eta_peak:.4f} (상한 1.0, deltaP_peak={delta_p_peak_mw * 1000:.2f}kW, '
          f'S_total={s_total:.4f}MVA)', flush=True)
    print(f'  2) 지속시간 = {duration_h:.3f} h (손익분기 한계 {PM.BREAKEVEN_DURATION_H_REFERENCE}h '
          f'대비 {"이내" if duration_h < PM.BREAKEVEN_DURATION_H_REFERENCE else "*초과"})', flush=True)
    print(f'  3) E/S 비 = {e_over_s:.4f} MWh/MVA (x_max 참고값 {PM.X_MAX_REFERENCE} 대비 '
          f'{"이내" if e_over_s < PM.X_MAX_REFERENCE else "*초과"})', flush=True)

    result.update(eta_peak=eta_peak, duration_h=duration_h, e_over_s=e_over_s)
    return result


# ============================================================
# lower_lp 재호출 - 스케줄(P_net, soc) 1회 계산, EFC/유사도/원자료 CSV가 공유
# ============================================================

def compute_schedules(installed_units):
    """installed_units: [(b,S,E), ...]. 유닛x시나리오(ALL_DAYS 5개) 조합마다 lower_lp를
    정확히 1회씩만 호출한다 - 이 결과를 (3)-4 EFC, (8) 유사도, (9) 원자료 CSV가 공유한다
    (지시사항 "lower_lp 재호출은 1회만 수행"). pandapower는 호출하지 않는다."""
    base_p_sum = float(evaluate._BASE_P.sum())
    schedules = {s: [] for s in PM.ALL_DAYS}   # scenario -> [(P_net[T], soc[T+1]) per unit]
    for b, S, E in installed_units:
        for s in PM.AVG_DAYS:
            smp = np.asarray(PM.SMP[s])
            P_net, soc = lower_lp.solve_avg(S, E, smp, assert_physics=False)
            schedules[s].append((P_net, soc))
        for s in PM.PEAK_DAYS:
            load_mw = base_p_sum * np.asarray(PM.LOAD[s])
            P_net, soc, _pk = lower_lp.solve_peak(S, E, load_mw, assert_physics=False)
            schedules[s].append((P_net, soc))
    return schedules, base_p_sum


# ============================================================
# (3)-4 EFC
# ============================================================

def _efc(P_net, E_rated_mwh):
    e_usable = E_rated_mwh * PM.DOD
    if e_usable <= 0:
        return 0.0
    return float(np.sum(np.abs(P_net)) * PM.DT_HOURS / (2.0 * e_usable))


def section3_4_efc(group, installed_units, schedules):
    section(f"(3)-4 EFC (등가 전 사이클) - n_ess={group['n_ess']}")
    if not installed_units:
        print('  설치된 기 없음 - 생략', flush=True)
        return None

    results = []
    for i, (b, S, E) in enumerate(installed_units):
        annual_efc = 0.0
        for s in PM.AVG_DAYS:
            P_net, _soc = schedules[s][i]
            annual_efc += PM.N_WEEKDAYS[s] * _efc(P_net, E)
        peak_efc = {s: _efc(schedules[s][i][0], E) for s in PM.PEAK_DAYS}
        worst_case_annual = annual_efc * (365.0 / PM.TOTAL_WEEKDAYS_PER_YEAR)

        print(f'  기 {i}(b={b}, E={E:.4f}MWh):', flush=True)
        print(f"    연간EFC(247일 가중, 주값) = {annual_efc:.2f}회/년  "
              f"(Lazard {PM.LAZARD_CYCLES_PER_YEAR}회/년 대비 "
              f"{'과소=활용저조' if annual_efc < PM.LAZARD_CYCLES_PER_YEAR else '*과다=수명위협 가능'})",
              flush=True)
        print(f"    보조 상한점검(365/247 스케일, 주말=평일 최악가정) = {worst_case_annual:.2f}회/년 "
              f"({'350 미만 - 안전' if worst_case_annual < PM.LAZARD_CYCLES_PER_YEAR else '*350 초과'})",
              flush=True)
        print(f"    PEAK_DAYS(연 2일 상당, 가중 제외 별도보고): "
              f"summer_peak={peak_efc.get('summer_peak', float('nan')):.4f}회  "
              f"winter_peak={peak_efc.get('winter_peak', float('nan')):.4f}회", flush=True)

        results.append(dict(unit_idx=i, b=b, E=E, annual_efc=annual_efc,
                             worst_case_annual_efc=worst_case_annual, peak_efc=peak_efc))
    return results


# ============================================================
# (3)-5 b* 분포
# ============================================================

def section3_5_bus_distribution(group, bus_sweep_data):
    section(f"(3)-5 b* 분포 (독립실행 전체) - n_ess={group['n_ess']}")

    installed_buses = []
    n_dead_total = 0
    for r in group['runs']:
        units = parse_units(r['x'], group['n_ess'])
        for b, S, E in units:
            if S > 1e-9 and E > 1e-9:
                installed_buses.append(b)
            else:
                n_dead_total += 1

    if not installed_buses:
        print('  설치된 기가 전혀 없음(전 run 탐색실패) - 생략', flush=True)
        return None

    hist = defaultdict(int)
    for b in installed_buses:
        hist[b] += 1
    n_distinct = len(hist)
    mode_b = max(hist, key=hist.get)

    print(f'  설치된 기 {len(installed_buses)}개(소멸된 기 {n_dead_total}개), '
          f'서로 다른 버스 {n_distinct}종', flush=True)
    print(f'  최빈값 b={mode_b}({hist[mode_b]}회)', flush=True)
    print('  히스토그램(내림차순):', flush=True)
    for b, cnt in sorted(hist.items(), key=lambda kv: -kv[1]):
        print(f"    b={b:2d}: {'#' * cnt} ({cnt})", flush=True)

    print()
    print('  ★ 해석 규칙(CLAUDE.md 8절-5, 4차 수정 반영): 당초 예상이던 "위치 무차별"은 실측으로 '
          '반증됐다(probe_bus_sweep.py, j_net 폭 34.9%). 보고 문구는 "무차별"이 아니라 '
          '"유의하지만 최적권이 넓다"로 쓴다 - 상위 10개 버스가 0.5% 이내로 촘촘하다는 것이 근거다.',
          flush=True)

    rank_info = None
    if bus_sweep_data is not None:
        rank_info = _rank_against_bus_sweep(sorted(set(installed_buses)), bus_sweep_data)
    else:
        print('  (probe_bus_sweep.py 결과 없음 - 전수평가 순위 대조 생략)', flush=True)

    return dict(hist=dict(hist), n_distinct=n_distinct, mode_b=mode_b, rank_info=rank_info)


def _rank_against_bus_sweep(distinct_buses, by_case):
    """PSO가 찾은 각 b가 probe_bus_sweep.py 전수평가(32개 버스) 기준 몇 위인지 대조한다.
    이 함수는 b만 받으므로, 실제 (S,E)와 무관하게 probe_bus_sweep.py가 평가한 두 케이스
    ('small'~0.1764MVA, 'large'~1.5MVA) 중 PSO 뼈대 최적해 규모(~0.1764)에 가장 가까운
    케이스를 기준으로 삼는다."""
    case_S = {case: rows[0]['S'] for case, rows in by_case.items() if rows}
    if not case_S:
        return None
    case = min(case_S, key=lambda c: abs(case_S[c] - 0.1764))
    ranked = sorted(by_case[case], key=lambda row: -row['j_net'])
    n = len(ranked)

    results = []
    for b in distinct_buses:
        rank = next((i + 1 for i, row in enumerate(ranked) if row['b'] == b), None)
        results.append((b, case, rank, n))
        if rank is not None:
            print(f"  PSO가 찾은 b={b}는 probe_bus_sweep.py '{case}' 케이스(S={case_S[case]}) 기준 "
                  f'{rank}위/{n}', flush=True)
        else:
            print(f"  PSO가 찾은 b={b}는 probe_bus_sweep.py '{case}' 데이터에 없음", flush=True)
    return results


# ============================================================
# (3)-6 q*_ratio
# ============================================================

def section3_6_q_ratio(group):
    section(f"(3)-6 q*_ratio - n_ess={group['n_ess']}")
    print('  현 뼈대는 PSO가 q 차원을 탐색하지 않는다(q=0 고정, CLAUDE.md 7-A절) - 값은 항상 0.',
          flush=True)
    print('  q_ratio 스윕(8절-6)은 별도 스크립트 사안이며 이 모듈에서는 계산하지 않는다.', flush=True)


# ============================================================
# (4) 수렴 진단
# ============================================================

def section4_convergence(group):
    section(f"(4) 수렴 진단 - n_ess={group['n_ess']}")
    if group['gens'] is None:
        print('  generations 파일을 찾지 못함 - 생략', flush=True)
        return None

    by_run = defaultdict(list)
    for row in group['gens']:
        by_run[row['run']].append(row)

    print('  run별 최종 세대 std_f/|gbest_f|:', flush=True)
    for run_idx in sorted(by_run):
        last = max(by_run[run_idx], key=lambda r: r['gen'])
        ratio = last['std_f'] / abs(last['gbest_f']) if last['gbest_f'] != 0 else float('nan')
        print(f"    run={run_idx}: gen={last['gen']}  gbest_f={last['gbest_f']:+.4e}  "
              f"std_f={last['std_f']:.4e}  std/|gbest|={ratio:.4f}", flush=True)

    gbest_finals = [r['gbest_f'] for r in group['runs'] if r['gbest_f'] is not None]
    gbest_stats = _stats(gbest_finals)
    print(f"\n  재시작 분산(run간 최종 gbest_f): median={gbest_stats['median']:+.4e}  "
          f"range=[{gbest_stats['min']:+.4e}, {gbest_stats['max']:+.4e}]", flush=True)
    if gbest_stats['n'] >= 2 and gbest_stats['median'] != 0:
        spread_pct = (gbest_stats['max'] - gbest_stats['min']) / abs(gbest_stats['median']) * 100
        print(f"  gbest_f 폭 = {spread_pct:.2f}%", flush=True)

    print()
    print('  ★ 판정은 곡선의 평탄함이 아니라 재시작 분산(run간 gbest 분포)으로 한다(CLAUDE.md 7절 - '
          '곡선 평탄화는 w 선형감소에 의한 소진수렴의 증거일 뿐 전역최적 도달을 함의하지 않는다).',
          flush=True)
    print('  판정표:', flush=True)
    print('    gbest_f 분산 小 + gbest_x 뭉침  -> 전역최적의 강한 증거', flush=True)
    print('    gbest_f 분산 大                 -> 탐색 부족(입자 수 증설 필요)', flush=True)
    print('    f는 뭉치는데 x는 흩어짐         -> 최적권이 넓음(8절-5)', flush=True)
    print('    f도 흩어짐                      -> 탐색 실패', flush=True)

    return dict(gbest_stats=gbest_stats)


# ============================================================
# (5) 탐색 실패 run 카운터
# ============================================================

def section5_search_failure(group, decomp_result):
    section(f"(5) 탐색 실패 run 카운터 - n_ess={group['n_ess']}")
    if decomp_result is None:
        print('  j_net 데이터 없음 - 생략', flush=True)
        return None

    n_total = len(group['runs'])
    n_deg = len(decomp_result['degenerate'])
    print(f'  전체 {n_total}건 중 탐색실패(j_net<{DEGENERATE_J_NET_WON:.0e}원) {n_deg}건 '
          f'({n_deg / n_total * 100:.1f}%)', flush=True)
    for r in decomp_result['degenerate']:
        print(f"    run={r['run']}  j_net={r['j_net']:.6f}원  x={r['x_json']}", flush=True)
    print('  이 값이 탐색 예산 충분성의 증거다 - 0이면 예산 충분, 지속적으로 나오면 입자/세대 '
          '증설을 검토할 것. (1)절 "전체" vs "정상만" 비교가 이 카운트의 실질적 효과다.',
          flush=True)
    return dict(n_total=n_total, n_degenerate=n_deg, ratio=n_deg / n_total)


# ============================================================
# (6) 음의 B_defer 카운터
# ============================================================

def section6_negative_bdefer(group):
    section(f"(6) 음의 B_defer 카운터 - n_ess={group['n_ess']}")
    if 'n_negative_bdefer' in group['missing_cols']:
        print('  n_negative_bdefer 컬럼 없음(구 스키마) - 생략', flush=True)
        return None

    total_negative = sum(int(r['n_negative_bdefer'] or 0) for r in group['runs'])
    min_vals = [r['min_bdefer'] for r in group['runs'] if r['min_bdefer'] is not None]
    overall_min = min(min_vals) if min_vals else None

    print(f'  전체 run 합산 음의 B_defer 발생 횟수 = {total_negative}', flush=True)
    if overall_min is not None:
        is_noise = abs(overall_min) < NEG_BDEFER_NOISE_ATOL_WON
        print(f"  전체 최소값 min_bdefer = {overall_min:.6f}원 -> "
              f"{'잡음(|min_bdefer|<1원)' if is_noise else '*실재하는 음편익'}", flush=True)
    print('  ★ 해석 규칙(CLAUDE.md 3절): n=1에서 나오는 ~1e-5원 규모는 부동소수점 잡음이며 '
          '"PSO가 피크 증가를 회피했다"는 서사로 쓰면 틀린다. 실질적 음편익 서사는 n>=2에서만 '
          '성립한다(n=2 실측 -2.4e7원).', flush=True)
    return dict(total_negative=total_negative, overall_min=overall_min)


# ============================================================
# (7) 경계 근접 확인
# ============================================================

def section7_boundary(group, operating):
    section(f"(7) 경계 근접 확인 - n_ess={group['n_ess']}")
    if operating is None or not operating.get('installed'):
        print('  설치된 기 없음 - 생략', flush=True)
        return None

    warned = False
    for i, (b, S, E) in enumerate(operating['installed']):
        s_frac = (S - PM.S_BOUNDS[0]) / (PM.S_BOUNDS[1] - PM.S_BOUNDS[0])
        e_frac = (E - PM.E_BOUNDS[0]) / (PM.E_BOUNDS[1] - PM.E_BOUNDS[0])
        s_warn = s_frac < BOUNDARY_WARN_FRAC or s_frac > 1 - BOUNDARY_WARN_FRAC
        e_warn = e_frac < BOUNDARY_WARN_FRAC or e_frac > 1 - BOUNDARY_WARN_FRAC
        warned = warned or s_warn or e_warn
        print(f"  기 {i}(b={b}): S={S:.4f} ({s_frac * 100:.2f}% of range)"
              f"{' *경계근접' if s_warn else ''}  "
              f"E={E:.4f} ({e_frac * 100:.2f}% of range){' *경계근접' if e_warn else ''}", flush=True)

    print(f"  판정: {'*경계 근접 발견 - 탐색범위 재검토 필요' if warned else '경계에서 충분히 떨어져 있음(결과가 탐색범위 상한/하한에 의존하지 않음)'}",
          flush=True)
    return dict(warned=warned)


# ============================================================
# (8) 평균일/최대일 스케줄 유사도
# ============================================================

def section8_schedule_similarity(group, installed_units, schedules):
    section(f"(8) 평균일/최대일 스케줄 유사도 - n_ess={group['n_ess']}")
    if not installed_units:
        print('  설치된 기 없음 - 생략', flush=True)
        return None

    main_pairs = [('summer', 'summer_peak'), ('winter', 'winter_peak')]
    ref_pairs = [('shoulder', 'summer_peak'), ('shoulder', 'winter_peak')]

    print('  보조: SMP-부하 상관계수(회귀 검증 - params.py만 사용, 조류계산 불필요, 0.97 재현 확인용):',
          flush=True)
    for s in PM.ALL_DAYS:
        corr = float(np.corrcoef(PM.LOAD[s], PM.SMP[s])[0, 1])
        print(f'    {s:12s}: corr(LOAD,SMP)={corr:+.4f}', flush=True)

    results = []
    for i, (b, S, E) in enumerate(installed_units):
        print(f'\n  기 {i}(b={b}, S={S:.4f}MVA):', flush=True)
        for s_avg, s_peak in main_pairs:
            P_avg = schedules[s_avg][i][0]
            P_peak = schedules[s_peak][i][0]
            if np.std(P_avg) > 0 and np.std(P_peak) > 0:
                corr = float(np.corrcoef(P_avg, P_peak)[0, 1])
            else:
                corr = float('nan')
            l1 = float(np.mean(np.abs(P_avg / S - P_peak / S))) if S > 0 else float('nan')
            t_dis_avg, t_ch_avg = int(np.argmax(P_avg)), int(np.argmin(P_avg))
            t_dis_peak, t_ch_peak = int(np.argmax(P_peak)), int(np.argmin(P_peak))

            print(f'    {s_avg} vs {s_peak}: 상관계수={corr:+.4f}  정규화L1거리={l1:.4f}', flush=True)
            print(f"      최대방전시각: {s_avg}=t{t_dis_avg}  {s_peak}=t{t_dis_peak}"
                  f"{'  <- 일치' if t_dis_avg == t_dis_peak else '  <- 불일치'}", flush=True)
            print(f"      최대충전시각: {s_avg}=t{t_ch_avg}  {s_peak}=t{t_ch_peak}"
                  f"{'  <- 일치' if t_ch_avg == t_ch_peak else '  <- 불일치'}", flush=True)

            results.append(dict(unit=i, b=b, s_avg=s_avg, s_peak=s_peak, corr=corr, l1=l1,
                                 t_dis_avg=t_dis_avg, t_ch_avg=t_ch_avg,
                                 t_dis_peak=t_dis_peak, t_ch_peak=t_ch_peak))

        print('    (참고, 계절이 다름 - 주 결과 아님) shoulder 대조:', flush=True)
        for s_ref, s_peak in ref_pairs:
            P_ref = schedules[s_ref][i][0]
            P_peak = schedules[s_peak][i][0]
            if np.std(P_ref) > 0 and np.std(P_peak) > 0:
                corr = float(np.corrcoef(P_ref, P_peak)[0, 1])
            else:
                corr = float('nan')
            print(f'      {s_ref} vs {s_peak}: 상관계수={corr:+.4f}', flush=True)

    print()
    print('  ★ 해석 규칙(CLAUDE.md 8절-7, 반드시 이 서술 유지): 유사하다는 결과가 나와도 그 근거를 '
          '"같은 정책"으로 쓰면 틀린다 - 평균일 LP는 가격(SMP)을, 최대일 LP는 부하를 좇으므로 '
          '목적이 다르다. 유사해지는 이유는 SMP-부하 高상관(위 수치, 참고 0.97) 때문이다. '
          '유사하면 "일관된 운영"의 방증, 다르면 "차익 유인 vs 피크저감 유인의 충돌"의 발견 - '
          '이쪽도 결과이지 실패가 아니다.', flush=True)

    return results


# ============================================================
# (9) 스케줄 원자료 저장
# ============================================================

SCHEDULE_FIELDS = [
    'scenario', 't', 'unit', 'b',
    'p_ch', 'p_dis', 'p_net', 'soc', 'soc_frac',
    'load_mw', 'smp_won_per_kwh', 'p_slack_base', 'p_slack_ess',
]


def get_slack_via_evaluate(units_all, n_ess):
    """p_slack_base/p_slack_ess는 조류계산 결과라 lower_lp만으로는 못 얻는다. evaluate.py를
    고치지 않고 evaluate.evaluate_particle(return_detail=True)의 기존 반환값에서 얻을 수
    있는지 확인한다 - 있으면 쓰고, 없으면(예: 발산) 두 컬럼을 생략한다고 보고한다."""
    x4 = _expand_to_4n(units_all)
    detail = evaluate.evaluate_particle(x4, return_detail=True)
    if detail.get('diverged'):
        print('  ★ 최적해 재평가가 발산 - p_slack_base/p_slack_ess 확보 불가, 두 컬럼 생략',
              flush=True)
        return None, None
    print('  p_slack_base/p_slack_ess: evaluate.evaluate_particle(return_detail=True)의 기존 '
          "반환값(_BASE_FLOW['p_slack'], detail['p_slack_ess'])에서 그대로 확보함 - evaluate.py "
          '미수정.', flush=True)
    return evaluate._BASE_FLOW['p_slack'], detail['p_slack_ess']


def validate_schedules(installed_units, schedules):
    print('  검증(저장 전, lower_lp._assert_physics와 별개로 저장 CSV 자체의 일관성 확인):',
          flush=True)
    any_violation = False
    max_ch_dis_product = 0.0
    for s in PM.ALL_DAYS:
        for i, (b, S, E) in enumerate(installed_units):
            P_net, soc = schedules[s][i]
            if abs(soc[-1] - soc[0]) > SOC_CYCLE_ATOL_MWH:
                print(f'    * SOC 사이클 등식 위반: {s} unit{i} SOC[0]={soc[0]:.6f} '
                      f'SOC[24]={soc[-1]:.6f}', flush=True)
                any_violation = True
            lo, hi = PM.SOC_MIN_FRAC * E, PM.SOC_MAX_FRAC * E
            if np.any(soc < lo - SOC_CYCLE_ATOL_MWH) or np.any(soc > hi + SOC_CYCLE_ATOL_MWH):
                print(f'    * SOC 범위 위반: {s} unit{i}', flush=True)
                any_violation = True
            if np.any(np.abs(P_net) > S + SOC_CYCLE_ATOL_MWH):
                print(f'    * |P_net|<=S 위반: {s} unit{i}', flush=True)
                any_violation = True
            p_ch = np.maximum(-P_net, 0.0)
            p_dis = np.maximum(P_net, 0.0)
            max_ch_dis_product = max(max_ch_dis_product, float(np.max(p_ch * p_dis)))

    print(f'    동시충방전 최대곱(p_ch*p_dis) = {max_ch_dis_product:.6e} '
          f"(완전히 0이 아닐 수 있음 - solve_avg의 EPS_REG=1e-6 정규화항 때문, 실패 아님)",
          flush=True)
    print(f"    SOC[24]==SOC[0] / SOC 범위 / |P_net|<=S: "
          f"{'* 위반 있음(위 상세)' if any_violation else '전부 통과'}", flush=True)
    return dict(ok=(not any_violation), max_ch_dis_product=max_ch_dis_product)


def build_schedule_rows(installed_units, schedules, base_p_sum, slack_base, slack_ess):
    rows = []
    for s in PM.ALL_DAYS:
        load_mw_s = base_p_sum * np.asarray(PM.LOAD[s])
        smp_s = np.asarray(PM.SMP[s])
        for t in range(PM.TIME_STEPS):
            p_slack_base_t = float(slack_base[s][t]) if slack_base is not None else ''
            p_slack_ess_t = float(slack_ess[s][t]) if slack_ess is not None else ''
            for i, (b, S, E) in enumerate(installed_units):
                P_net, soc = schedules[s][i]
                p_net_t = float(P_net[t])
                p_dis = max(p_net_t, 0.0)
                p_ch = max(-p_net_t, 0.0)
                soc_t = float(soc[t])
                soc_frac = soc_t / E if E > 0 else float('nan')
                rows.append(dict(
                    scenario=s, t=t, unit=i, b=b,
                    p_ch=p_ch, p_dis=p_dis, p_net=p_net_t, soc=soc_t, soc_frac=soc_frac,
                    load_mw=float(load_mw_s[t]), smp_won_per_kwh=float(smp_s[t]),
                    p_slack_base=p_slack_base_t, p_slack_ess=p_slack_ess_t,
                ))
    return rows


def print_wide_schedule(installed_units, schedules):
    section('스케줄 원자료 (wide 표, 방전(+)/충전(-))')
    for i, (b, S, E) in enumerate(installed_units):
        print(f'\n  기 {i}(b={b}) P_net[MW]  (t=0..23, 방전(+)/충전(-)):', flush=True)
        header = f"{'t':>3s} | " + ' | '.join(f'{s:>13s}' for s in PM.ALL_DAYS)
        print('  ' + header, flush=True)
        for t in range(PM.TIME_STEPS):
            vals = [schedules[s][i][0][t] for s in PM.ALL_DAYS]
            print('  ' + f'{t:>3d} | ' + ' | '.join(f'{v:>+13.4f}' for v in vals), flush=True)

        print(f'\n  기 {i}(b={b}) SOC[MWh]  (t=0..24):', flush=True)
        print('  ' + header, flush=True)
        for t in range(PM.TIME_STEPS + 1):
            vals = [schedules[s][i][1][t] for s in PM.ALL_DAYS]
            print('  ' + f'{t:>3d} | ' + ' | '.join(f'{v:>13.4f}' for v in vals), flush=True)


def section9_schedule_raw(group, best, operating, schedules, base_p_sum, out_dir, ts):
    section(f"(9) 스케줄 원자료 저장 - n_ess={group['n_ess']}")
    installed = operating['installed']

    slack_base, slack_ess = get_slack_via_evaluate(operating['units'], group['n_ess'])
    validate_schedules(installed, schedules)
    print_wide_schedule(installed, schedules)

    print('\n  SOC[24]==SOC[0] 상세(요약):', flush=True)
    for i, (b, S, E) in enumerate(installed):
        for s in PM.ALL_DAYS:
            soc = schedules[s][i][1]
            print(f'    {s:12s} unit{i}(b={b}): SOC[0]={soc[0]:.6f} SOC[24]={soc[-1]:.6f} '
                  f'diff={soc[-1] - soc[0]:+.2e}', flush=True)

    rows = build_schedule_rows(installed, schedules, base_p_sum, slack_base, slack_ess)
    path = os.path.join(out_dir, f"schedule_n{group['n_ess']}_{ts}.csv")
    _write_csv(path, SCHEDULE_FIELDS, rows)
    print(f'\n  CSV 저장: {path} ({len(rows)}행)', flush=True)
    return dict(path=path, n_rows=len(rows), has_slack=(slack_base is not None))


# ============================================================
# CSV 저장 공통
# ============================================================

def _write_csv(path, fields, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        f.flush()
        os.fsync(f.fileno())


# ============================================================
# 그룹(=한 n_ess) 처리 조립
# ============================================================

def load_group(runs_path):
    section(f'로드: {runs_path}')
    runs, missing = load_runs_csv(runs_path)
    if not runs:
        print('  run 없음 - 건너뜀', flush=True)
        return None
    n_ess = runs[0]['n_ess']
    gen_path = _derive_generations_path(runs_path)
    gens = load_generations_csv(gen_path)
    if gens is None:
        print(f'  경고: generations 파일을 찾지 못함({gen_path}) - (4)절 수렴 진단 생략', flush=True)
    print(f'  n_ess={n_ess}  run 수={len(runs)}  결측 컬럼={missing or "없음"}', flush=True)
    return dict(n_ess=n_ess, runs_path=runs_path, gen_path=gen_path, runs=runs, gens=gens,
                missing_cols=missing)


def process_group(runs_path, bus_sweep_data, out_dir, ts):
    group = load_group(runs_path)
    if group is None:
        return None

    decomp = section1_benefit_decomposition(group)
    cost_check = section2_cost_check(group)

    best = pick_best_run(group)
    operating = section3_operating_metrics(group, best)

    schedules = None
    base_p_sum = None
    efc_result = None
    if operating is not None and operating.get('installed'):
        schedules, base_p_sum = compute_schedules(operating['installed'])
        efc_result = section3_4_efc(group, operating['installed'], schedules)

    bus_dist = section3_5_bus_distribution(group, bus_sweep_data)
    section3_6_q_ratio(group)
    convergence = section4_convergence(group)
    search_failure = section5_search_failure(group, decomp)
    neg_bdefer = section6_negative_bdefer(group)
    boundary = section7_boundary(group, operating)

    similarity_result = None
    schedule9_result = None
    if schedules is not None:
        similarity_result = section8_schedule_similarity(group, operating['installed'], schedules)
        schedule9_result = section9_schedule_raw(group, best, operating, schedules, base_p_sum,
                                                  out_dir, ts)

    return dict(
        n_ess=group['n_ess'], group=group, decomp=decomp, cost_check=cost_check, best=best,
        operating=operating, efc=efc_result, bus_dist=bus_dist, convergence=convergence,
        search_failure=search_failure, neg_bdefer=neg_bdefer, boundary=boundary,
        similarity=similarity_result, schedule9=schedule9_result,
    )


# ============================================================
# 기수 간 비교 (기수-순편익 곡선)
# ============================================================

def print_cross_n_curve(all_results):
    section('기수-순편익 곡선 (n_ess 간 비교)')
    valid = [r for r in all_results if r is not None and r['decomp'] is not None]
    if not valid:
        print('  비교할 데이터 없음(편익 컬럼이 있는 그룹이 하나도 없음)', flush=True)
        return

    if any(r['n_ess'] >= 2 for r in valid):
        print('  ★ 주의(CLAUDE.md 2절/7-A절): n>=2 결과는 각 기의 solve_peak이 전체 시스템 부하를 '
              '독립적으로 보고 계산되어 b_defer가 구조적으로 과소평가된 상태다(실측: 2등분 시 '
              '단일 대비 0.53배, probe_split.py 참조). 아래 곡선의 n>=2 지점은 이 한계를 감안해서 '
              '읽을 것 - "최적 기수 1" 결론의 근거로 이 곡선을 쓰지 말 것(그 결론은 6-A절 한계 '
              'j_net 부호전환에서 독립적으로 성립한다).', flush=True)

    print(f"\n  {'n_ess':>6s} | {'run수':>5s} | {'탐색실패':>8s} | {'j_net median(전체)':>20s} | "
          f"{'j_net median(정상만)':>20s}", flush=True)
    for r in sorted(valid, key=lambda r: r['n_ess']):
        all_j = _stats([x['j_net'] for x in r['group']['runs']])
        normal_j = _stats([x['j_net'] for x in r['decomp']['normal']])
        n_deg = len(r['decomp']['degenerate'])
        n_tot = len(r['group']['runs'])
        print(f"  {r['n_ess']:>6d} | {n_tot:>5d} | {f'{n_deg}/{n_tot}':>8s} | "
              f"{all_j['median']:>+20.4e} | {normal_j['median']:>+20.4e}", flush=True)


# ============================================================
# 요약 CSV
# ============================================================

SUMMARY_FIELDS = [
    'n_ess', 'n_runs', 'n_degenerate',
    'j_net_median_all', 'j_net_median_normal',
    'b_defer_share_pct_median', 'b_arb_share_pct_median', 'b_loss_share_pct_median',
    'decomposition_violations', 'cost_check_ok',
    'best_run', 'best_gbest_f', 'best_j_net',
    's_total', 'e_total', 'eta_peak', 'duration_h', 'e_over_s',
    'annual_efc_unit0', 'n_distinct_buses', 'mode_bus',
    'total_negative_bdefer', 'min_bdefer_overall', 'boundary_warning',
]


def build_summary_row(result):
    if result is None:
        return None
    g = result['group']
    decomp = result['decomp']
    cost_check = result['cost_check']
    op = result['operating'] or {}
    efc = result['efc'] or []
    bus_dist = result['bus_dist'] or {}
    neg_bdefer = result['neg_bdefer'] or {}
    boundary = result['boundary'] or {}

    def share(field):
        if not decomp:
            return ''
        ratios = [r[field] / r['j_net'] * 100 for r in decomp['normal'] if r['j_net']]
        return _stats(ratios)['median']

    return dict(
        n_ess=g['n_ess'], n_runs=len(g['runs']),
        n_degenerate=len(decomp['degenerate']) if decomp else '',
        j_net_median_all=_stats([r['j_net'] for r in g['runs']])['median'] if decomp else '',
        j_net_median_normal=_stats([r['j_net'] for r in decomp['normal']])['median'] if decomp else '',
        b_defer_share_pct_median=share('b_defer'),
        b_arb_share_pct_median=share('b_arb'),
        b_loss_share_pct_median=share('b_loss'),
        decomposition_violations=len(decomp['decomposition_bad']) if decomp else '',
        cost_check_ok=cost_check['ok'] if cost_check else '',
        best_run=result['best']['run'] if result['best'] else '',
        best_gbest_f=result['best']['gbest_f'] if result['best'] else '',
        best_j_net=result['best']['j_net'] if result['best'] else '',
        s_total=op.get('s_total', ''), e_total=op.get('e_total', ''),
        eta_peak=op.get('eta_peak', ''), duration_h=op.get('duration_h', ''),
        e_over_s=op.get('e_over_s', ''),
        annual_efc_unit0=efc[0]['annual_efc'] if efc else '',
        n_distinct_buses=bus_dist.get('n_distinct', ''), mode_bus=bus_dist.get('mode_b', ''),
        total_negative_bdefer=neg_bdefer.get('total_negative', ''),
        min_bdefer_overall=neg_bdefer.get('overall_min', ''),
        boundary_warning=boundary.get('warned', ''),
    )


# ============================================================
# CLI
# ============================================================

def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description='postprocess: CLAUDE.md 8절 후처리 지표 계산 (사후분석 전용, 최적화 미개입).',
        epilog='예: python postprocess.py results/runs_n1_<ts>.csv results/runs_n2_<ts>.csv',
    )
    parser.add_argument('runs', nargs='+',
                         help='runs_n<k>_<ts>.csv 경로(들). generations는 파일명 규약으로 자동 매칭.')
    parser.add_argument('--bus-sweep-csv', default=None,
                         help='scripts/probe_bus_sweep.py 결과 CSV 경로. 생략하면 '
                              f'{BUS_SWEEP_DIR_DEFAULT} 밑에서 최신 파일을 자동 탐색(없으면 생략).')
    parser.add_argument('--results-dir', default=RESULTS_DIR_DEFAULT,
                         help='postprocess_*.csv / schedule_*.csv 출력 위치 (기본: results/).')
    return parser


def main():
    args = _build_arg_parser().parse_args()
    _check_env()

    print('evaluate.init_worker() 호출 - 기저 조류계산 120회 1회만 캐싱(이후 (9)절에서 재사용).',
          flush=True)
    evaluate.init_worker()

    bus_sweep_path = args.bus_sweep_csv or _find_latest_bus_sweep_csv()
    bus_sweep_data = None
    if bus_sweep_path and os.path.exists(bus_sweep_path):
        print(f'probe_bus_sweep.py 결과 사용: {bus_sweep_path}', flush=True)
        bus_sweep_data = load_bus_sweep_csv(bus_sweep_path)
    else:
        print('probe_bus_sweep.py 결과 없음 - (3)-5 전수평가 순위 대조 생략.', flush=True)

    os.makedirs(args.results_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    all_results = []
    missing_report = []
    for runs_path in args.runs:
        result = process_group(runs_path, bus_sweep_data, args.results_dir, ts)
        all_results.append(result)
        if result is not None and result['group']['missing_cols']:
            missing_report.append((result['n_ess'], runs_path, result['group']['missing_cols']))

    print_cross_n_curve(all_results)

    n_ess_label = '-'.join(str(r['n_ess']) for r in all_results if r is not None)
    summary_rows = [build_summary_row(r) for r in all_results]
    summary_rows = [r for r in summary_rows if r is not None]
    summary_path = os.path.join(args.results_dir, f'postprocess_n{n_ess_label}_{ts}.csv')
    _write_csv(summary_path, SUMMARY_FIELDS, summary_rows)

    section('종합 요약')
    print(f'요약 CSV: {summary_path}', flush=True)
    for r in all_results:
        if r is None:
            continue
        sched = r.get('schedule9')
        print(f"  n_ess={r['n_ess']}: schedule CSV="
              f"{sched['path'] if sched else '(생성 안 됨)'}", flush=True)

    if missing_report:
        print('\n★ 로그에 없어 생략된 지표(main.py 수정 여부는 이 목록을 보고 사람이 판단할 것):',
              flush=True)
        for n_ess, path, cols in missing_report:
            print(f'  n_ess={n_ess} ({path}): {cols}', flush=True)
    else:
        print('\n모든 입력 파일이 현행 스키마를 갖추고 있어 생략된 지표 없음.', flush=True)


if __name__ == '__main__':
    main()
