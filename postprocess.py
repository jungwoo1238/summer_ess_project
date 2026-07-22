"""후처리 지표 계산 (CLAUDE.md 8절, 부록A). runs.csv/generations.csv를 읽어 편익 분해,
운영·설비 지표, 수렴 진단, b* 분포, 스케줄 유사도 등을 계산·출력한다.

**최적화에 개입하지 않는 사후분석 모듈이다.** PSO·evaluate 경로를 수정하지 않고, 이미
저장된 로그(runs.csv/generations.csv)와 lower_lp/benefits/evaluate의 기존 함수만 재사용한다.
evaluate.py/benefits.py/lower_lp.py/pso_core.py/main.py는 전부 미수정 - 필요한 값이 로그에
없으면 그 사실을 보고할 뿐 main.py를 고치지 않는다(이 게이트의 산출물 그 자체).

**full 실행 진입 전 게이트다.** dev 결과(n=1, n=2, n=3)로 완주 확인이 끝나야 full로 넘어간다.

★ 출력은 기수마다 별도 파일이다(① - 여러 기수를 한 요약 파일에 다행으로 합치지 않는다):
  - `postprocess_n{k}_{ts}.csv`       : run 수준 요약, 기수 전체에 1행.
  - `postprocess_units_n{k}_{ts}.csv` : 기(unit) 수준, 최우수 run의 기마다 1행(소멸 기 포함).
여러 기수를 한 번에 넘겨도 각 기수는 독립적으로 자기 파일 쌍을 낸다. 화면에서 기수 간
비교(선택적 병합 뷰)를 보려면 `--compare`를 추가한다(파일은 만들지 않음, 기본 꺼짐).

★ probe_bus_sweep.py 결과는 이 모듈이 읽지 않는다(④ - 용도 분리). 그 스윕은 어떤 최적해의
(S,E)를 고정하고 버스만 바꾸는 조건부 단면이라, PSO 결과의 품질을 그 스윕에 대조하면 인과가
뒤집힌다(최적해가 달라지면 스윕 순위도 바뀌고, 각 버스의 진짜 최적 (S,E)는 서로 다르다).
"최적권이 넓다"의 근거는 이제 시뮬레이션 자체 - 정상 run들이 고른 서로 다른 b*의 j_net 폭
(`bstar_jnet_spread_pct`)이다.

실행: `python postprocess.py results/runs_n1_<ts>.csv`  (기수 1개당 1회 실행 권장)
      generations.csv는 파일명 규약(runs_ -> generations_)으로 자동 매칭한다.

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
    """③ median을 주값으로 유지하되(heavy-tail·탐색실패 run에 강건) mean도 항상 함께 낸다 -
    median≫mean이면 아래쪽 꼬리(잔존 실패/조율실패 run) 신호가 된다."""
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not values:
        return dict(n=0, mean=float('nan'), median=float('nan'), q1=float('nan'), q3=float('nan'),
                    min=float('nan'), max=float('nan'))
    arr = np.array(values, dtype=float)
    return dict(n=len(arr), mean=float(np.mean(arr)), median=float(np.median(arr)),
                q1=float(np.percentile(arr, 25)), q3=float(np.percentile(arr, 75)),
                min=float(arr.min()), max=float(arr.max()))


def _print_stats(name, st, unit=''):
    if st['n'] == 0:
        print(f'    {name:16s}: (자료 없음)', flush=True)
        return
    print(f"    {name:16s}: n={st['n']:3d}  median={st['median']:+.4e}{unit}  "
          f"mean={st['mean']:+.4e}{unit}  "
          f"IQR=[{st['q1']:+.4e}, {st['q3']:+.4e}]  range=[{st['min']:+.4e}, {st['max']:+.4e}]",
          flush=True)


def _money_isclose(actual, expected):
    """CLAUDE.md 7절 원칙4: 기대값 크기에 따라 규칙이 갈린다 - 거의 0(전 기 소멸 run 등)이면
    절대오차(atol=10), 아니면 상대오차(rtol=1e-9, atol=0). check_decomposition/check_cost/
    compute_gross_shares가 전부 이 규칙 하나를 공유한다(중복 방지)."""
    if abs(expected) < MONEY_NEAR_ZERO_WON:
        return abs(actual - expected) <= DECOMP_ATOL_NEAR_ZERO
    return bool(np.isclose(actual, expected, rtol=DECOMP_RTOL, atol=0.0))


# ============================================================
# (1) 편익 분해 + 검산
# ============================================================

def check_decomposition(b_arb, b_loss, b_energy):
    """b_arb+b_loss ~= b_energy."""
    lhs = b_arb + b_loss
    ok = _money_isclose(lhs, b_energy)
    rule = (f'atol={DECOMP_ATOL_NEAR_ZERO}(기대값~0)' if abs(b_energy) < MONEY_NEAR_ZERO_WON
            else f'rtol={DECOMP_RTOL}')
    return ok, rule, lhs - b_energy


def compute_gross_shares(r):
    """① 분모를 총편익 B_gross=b_defer+b_energy(=b_defer+b_arb+b_loss)로 바꾼다 - j_net(순편익)
    대비 비율은 세 항의 합이 100%가 되지 않아(cost 항이 빠져 있음) 물리적 의미가 없었다.

    두 경로로 계산한 B_gross가 일치하는지(교차검증), 세 share의 합이 100%인지를 hard assert로
    검증한다 - 둘 다 순수 항등식이라 실패하면 코드 버그이지 판단의 문제가 아니다(사후분석
    모듈이므로 여기서 죽어도 최적화 파이프라인에는 영향 없음).

    100%-합 assert는 b_gross가 MONEY_NEAR_ZERO_WON보다 작을 때(전 기 소멸 run 등)는 건너뛴다 -
    분모가 거의 0인 나눗셈은 사소한 절대오차도 백분율로는 크게 증폭되어(예: 4.9e-7원 차이가
    0.0017%p로 보임) 100%±0.01 기준이 원래 의도(진짜 분해 오류 검출)와 무관하게 깨질 수 있다."""
    b_gross_a = r['b_defer'] + r['b_energy']
    b_gross_b = r['b_defer'] + r['b_arb'] + r['b_loss']
    assert _money_isclose(b_gross_b, b_gross_a), (
        f"B_gross 교차검증 실패(run={r.get('run')}): b_defer+b_energy={b_gross_a} vs "
        f"b_defer+b_arb+b_loss={b_gross_b}")
    b_gross = b_gross_a

    if b_gross == 0:
        return dict(b_gross=0.0, defer_pct=float('nan'), arb_pct=float('nan'),
                     loss_pct=float('nan'), cost_pct=float('nan'))

    defer_pct = r['b_defer'] / b_gross * 100
    arb_pct = r['b_arb'] / b_gross * 100
    loss_pct = r['b_loss'] / b_gross * 100
    cost_pct = r['cost'] / b_gross * 100

    if abs(b_gross) > MONEY_NEAR_ZERO_WON:
        total = defer_pct + arb_pct + loss_pct
        assert abs(total - 100.0) < 0.01, (
            f'share 합이 100%±0.01이 아님(run={r.get("run")}): {total}')

    return dict(b_gross=b_gross, defer_pct=defer_pct, arb_pct=arb_pct, loss_pct=loss_pct,
                cost_pct=cost_pct)


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

    # ① 편익 share: 분모는 총편익 B_gross=b_defer+b_energy(=b_defer+b_arb+b_loss)다. j_net(순편익)을
    # 분모로 쓰면 cost 항이 빠져 있어 세 항의 합이 100%가 되지 않는다(물리적 의미 없음 - 이전
    # 버전의 오류). compute_gross_shares가 run마다 assert로 100% 분해를 강제한다.
    gross_normal = [compute_gross_shares(r) for r in normal]
    gross_all = [compute_gross_shares(r) for r in runs]

    print('  편익 비중(①, 분모=B_gross=b_defer+b_energy):', flush=True)
    for label, gross_list in (('정상만(기본값)', gross_normal), ('전체(_all, 참고)', gross_all)):
        print(f'    [{label}]', flush=True)
        for key, name in (('defer_pct', 'b_defer'), ('arb_pct', 'b_arb'),
                           ('loss_pct', 'b_loss'), ('cost_pct', 'cost')):
            _print_stats(f'{name}/B_gross', _stats([g[key] for g in gross_list]), '%')
    print('  (주의: compute_gross_shares의 assert는 run 하나하나가 정확히 100%로 분해되는지를 '
          '보장할 뿐이다. 위에 찍힌 median은 run별 share를 각각 모아 median을 낸 값이라, median의 '
          '합 자체가 100%일 필요는 없다 - median은 선형연산이 아니다.)', flush=True)
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

    return dict(degenerate=degenerate, normal=normal, decomposition_bad=bad,
                gross_normal=gross_normal, gross_all=gross_all)


# ============================================================
# (2) 비용 재계산
# ============================================================

def check_cost(units, cost_reported):
    s_total = sum(u[1] for u in units)
    e_total = sum(u[2] for u in units)
    cost_recomputed = benefits.total_cost(s_total, e_total)
    ok = _money_isclose(cost_recomputed, cost_reported)
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

    print(f'  eta_peak = {eta_peak:.4f} (상한 1.0, deltaP_peak={delta_p_peak_mw * 1000:.2f}kW, '
          f'S_total={s_total:.4f}MVA)', flush=True)
    print(f'  지속시간 = {duration_h:.3f} h (손익분기 한계 {PM.BREAKEVEN_DURATION_H_REFERENCE}h '
          f'대비 {"이내" if duration_h < PM.BREAKEVEN_DURATION_H_REFERENCE else "*초과"})', flush=True)
    print(f'  E/S 비 = {e_over_s:.4f} MWh/MVA (x_max 참고값 {PM.X_MAX_REFERENCE} 대비 '
          f'{"이내" if e_over_s < PM.X_MAX_REFERENCE else "*초과"})', flush=True)
    print('  (② 기별 세부값은 postprocess_units_n<k>_<ts>.csv 참조 - 소멸 기가 NaN으로 나란히 '
          '보인다.)', flush=True)

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


def section3_4_efc(group, all_units, installed_units, schedules):
    section(f"(3)-4 EFC (등가 전 사이클) - n_ess={group['n_ess']}")
    if not installed_units:
        print('  설치된 기 없음 - 생략', flush=True)
        return None

    # ③ 기별 EFC. all_units(소멸 기 포함) 순서를 그대로 훑되, installed_units(=schedules와 같은
    # 순서로 필터링된 목록)에 없는 것(S<=0)은 EFC=NaN으로 두고 active 집계에서 제외한다.
    # 이 per_unit 결과가 그대로 postprocess_units_n<k>.csv의 원천이다(②).
    results = []
    inst_idx = 0
    for full_idx, (b, S, E) in enumerate(all_units):
        is_active = S > 1e-9 and E > 1e-9
        if not is_active:
            results.append(dict(unit_idx=full_idx, b=b, E=E, active=False,
                                 annual_efc=float('nan'), worst_case_annual_efc=float('nan'),
                                 peak_efc={s: float('nan') for s in PM.PEAK_DAYS}))
            continue

        i = inst_idx   # installed_units/schedules 쪽 인덱스 (all_units에서 소멸 기를 건너뛴 순서)
        inst_idx += 1
        annual_efc = 0.0
        for s in PM.AVG_DAYS:
            P_net, _soc = schedules[s][i]
            annual_efc += PM.N_WEEKDAYS[s] * _efc(P_net, E)
        peak_efc = {s: _efc(schedules[s][i][0], E) for s in PM.PEAK_DAYS}
        worst_case_annual = annual_efc * (365.0 / PM.TOTAL_WEEKDAYS_PER_YEAR)
        results.append(dict(unit_idx=full_idx, b=b, E=E, active=True, annual_efc=annual_efc,
                             worst_case_annual_efc=worst_case_annual, peak_efc=peak_efc))

    for r in results:
        if not r['active']:
            print(f"  기 {r['unit_idx']}(b={r['b']}, 소멸): EFC=NaN (② postprocess_units 파일에서 "
                  "is_active=False로 확인 가능)", flush=True)
            continue
        print(f"  기 {r['unit_idx']}(b={r['b']}, E={r['E']:.4f}MWh):", flush=True)
        print(f"    연간EFC(247일 가중, 주값) = {r['annual_efc']:.2f}회/년  "
              f"(Lazard {PM.LAZARD_CYCLES_PER_YEAR}회/년 대비 "
              f"{'과소=활용저조' if r['annual_efc'] < PM.LAZARD_CYCLES_PER_YEAR else '*과다=수명위협 가능'})",
              flush=True)
        print(f"    보조 상한점검(365/247 스케일, 주말=평일 최악가정) = "
              f"{r['worst_case_annual_efc']:.2f}회/년 "
              f"({'350 미만 - 안전' if r['worst_case_annual_efc'] < PM.LAZARD_CYCLES_PER_YEAR else '*350 초과'})",
              flush=True)
        pe = r['peak_efc']
        print(f"    PEAK_DAYS(연 2일 상당, 가중 제외 별도보고): "
              f"summer_peak={pe.get('summer_peak', float('nan')):.4f}회  "
              f"winter_peak={pe.get('winter_peak', float('nan')):.4f}회", flush=True)

    # ★ efc_mean_active는 is_active=True인 기만 평균한다(② 지시사항). n=1처럼 활성 기가
    # 하나뿐이면 min=max=mean이 되어 자기검증이 되고, 다수기에서 활성 기가 여럿이면 이 구분이
    # 비로소 실제로 발동한다(이전까지는 최우수 run에 활성 기가 항상 하나뿐이라 미발동이었다).
    active_annual = [r['annual_efc'] for r in results if r['active']]
    efc_min = min(active_annual) if active_annual else float('nan')
    efc_max = max(active_annual) if active_annual else float('nan')
    efc_mean_active = float(np.mean(active_annual)) if active_annual else float('nan')
    print(f"\n  요약(active 유닛 {len(active_annual)}개 기준): efc_min={efc_min:.2f}  "
          f"efc_max={efc_max:.2f}  efc_mean_active={efc_mean_active:.2f}"
          + ('  (n=1이라 세 값 동일 - 자기검증)' if len(active_annual) == 1 else
             '  (★ efc_mean_active는 is_active=True인 기만 평균 - 다수기에서 발동)'), flush=True)
    print('  (기별 원자료는 postprocess_units_n<k>_<ts>.csv에도 저장됨 - ②)', flush=True)

    return dict(per_unit=results, efc_min=efc_min, efc_max=efc_max, efc_mean_active=efc_mean_active)


# ============================================================
# (3)-5 b* 분포
# ============================================================

def section3_5_bus_distribution(group, decomp):
    section(f"(3)-5 b* 분포 (degenerate run 제외, ⑥) - n_ess={group['n_ess']}")

    if decomp is None:
        print('  j_net 데이터 없음(구 스키마) - degenerate 판별 불가, 생략', flush=True)
        return None

    # ⑥ 소멸 run(전 기 S=0)의 b*는 초기난수 흔적일 뿐 최적위치가 아니므로 b* 분포에서 제외한다.
    # 이로써 bstar_n_runs = n_runs - n_degenerate가 되어 n_runs와 구별되는 독립적 의미
    # (유효 b* 표본 수)를 갖는다.
    normal = decomp['normal']
    n_normal = len(normal)
    n_total = len(group['runs'])

    print(f'  유효 표본(⑥, degenerate 제외) n_runs={n_normal} '
          f'(전체 {n_total}건 중 {n_total - n_normal}건 제외)', flush=True)
    if n_normal < 10:
        print(f'      ★ 표본 부족(유효 n_runs={n_normal}<10) - 분포 폭 해석에 주의할 것.', flush=True)

    # ① 그룹핑 키 = 그 run에서 active(S>0)인 기들의 버스를 정렬한 튜플. 소멸 기는 위치가
    # 무의미하므로 키에서 제외한다. j_net은 전 기 상호작용의 단일 결과이므로 "첫 설치 버스"
    # 하나만으로 그룹핑하면(이전 버전) 다수기의 나머지 기 위치 정보를 버리게 된다. n=1이면
    # 키가 항상 길이 1인 튜플이라 이전 동작과 수치적으로 동일하다(회귀 없음).
    run_keys = {}
    by_key = defaultdict(list)
    for r in normal:
        units = parse_units(r['x'], group['n_ess'])
        installed = installed_only(units)
        key = tuple(sorted(b for b, S, E in installed))
        run_keys[r['run']] = key
        if key:
            by_key[key].append(r['j_net'])

    if not by_key:
        print('      정상 run에 설치된 기가 없음 - 생략', flush=True)
        bstar = dict(n_runs=n_normal, hist={}, n_distinct=0, mode_b=None,
                     sample_warning=(n_normal < 10), jnet_spread_pct=float('nan'),
                     effective_n_units_median=float('nan'), effective_n_units_dist='',
                     spread_scope='')
        return dict(bstar=bstar, n_distinct=0, mode_b=None)

    # ② 다수기에서 실질 기수 분포 - "다수기가 단일기로 붕괴한다"(6-A절 "최적 기수 1")는
    # 결론의 직접 통계 증거다. full에서 다수기(n_ess>=2)를 돌려도 정상 run들의 active 기
    # 개수가 1로 수렴하면, 그건 지표의 결함이 아니라 결론이 실제로 반영된 것이다.
    n_units_list = []
    for k, js in by_key.items():
        n_units_list += [len(k)] * len(js)
    effective_n_units_median = float(np.median(n_units_list))
    n_units_hist = defaultdict(int)
    for n in n_units_list:
        n_units_hist[n] += 1
    effective_n_units_dist = ', '.join(f'{k}:{v}' for k, v in sorted(n_units_hist.items()))

    print(f'\n  실질 기수 분포(②, 정상 run의 active 기 개수): {effective_n_units_dist} '
          f'(median={effective_n_units_median:.1f})', flush=True)
    print('    (6-A절 "최적 기수 1" 결론의 직접 통계 증거 - 다수기 탐색이라도 이 값이 1로 '
          '수렴하면 실제로 단일기로 붕괴한다는 뜻이다.)', flush=True)

    # 위치 조합 히스토그램(①: active 버스 집합 키 기준)
    hist = {k: len(js) for k, js in by_key.items()}
    n_distinct = len(hist)
    mode_key = max(hist, key=hist.get)

    print(f'\n  위치 조합 분포(①, active 버스 집합 기준): 서로 다른 조합 {n_distinct}종, '
          f'최빈값 {mode_key}({hist[mode_key]}회)', flush=True)
    for k, cnt in sorted(hist.items(), key=lambda kv: -kv[1]):
        print(f"    {k}: {'#' * cnt} ({cnt})", flush=True)

    # ③ bstar_jnet_spread_pct("위치를 바꿔도 j_net이 촘촘한가")는 본질적으로 기수가 같은
    # run들 안에서만 의미가 있다 - 기수가 섞이면 규모효과(6-A절 한계 j_net 부호전환)와
    # 위치효과가 뒤섞인다. 최빈 기수(보통 1)의 run들만 골라 그 안에서 위치조합별 median ->
    # 조합간 (max-min)/전체median을 잰다. 다른 기수의 run 수는 bstar_spread_scope에 명시한다.
    mode_n_units = max(n_units_hist, key=n_units_hist.get)
    scoped_keys = {k: js for k, js in by_key.items() if len(k) == mode_n_units}
    n_scoped = sum(len(js) for js in scoped_keys.values())
    j_scoped = [j for js in scoped_keys.values() for j in js]
    overall_median = _stats(j_scoped)['median']
    bus_medians = {k: _stats(js)['median'] for k, js in scoped_keys.items()}
    n_distinct_scoped = len(bus_medians)

    if n_distinct_scoped <= 1:
        jnet_spread_pct = 0.0   # 정의상 - 위치조합이 하나뿐이면 "서로 다른 위치 간 차이"가 없음
    else:
        med_vals = list(bus_medians.values())
        jnet_spread_pct = ((max(med_vals) - min(med_vals)) / abs(overall_median) * 100
                            if overall_median else float('nan'))

    spread_scope = f'{mode_n_units}-unit runs 기준 ({n_scoped} of {n_normal})'

    print(f'\n  위치조합별 j_net median ({spread_scope}):', flush=True)
    for k, med in sorted(bus_medians.items(), key=lambda kv: -kv[1]):
        note = '  <- run 1개뿐이라 그 값 자체' if len(scoped_keys[k]) == 1 else ''
        print(f'    {k}(n={len(scoped_keys[k])}): median j_net={med:+.4e}원{note}', flush=True)

    print(f"\n  bstar_jnet_spread_pct = {jnet_spread_pct:.2f}% ({spread_scope}, "
          f"전체median={overall_median:+.4e}원) - 위치조합이 {n_distinct_scoped}종으로 "
          f"갈렸는데도 이 폭이면 \"최적권이 넓다\"의 직접 증거", flush=True)
    sample_warning = n_scoped < 10
    if sample_warning:
        print(f'      ★ 표본 부족(scope 내 유효 run={n_scoped}<10) - spread는 참고값으로만 볼 것.',
              flush=True)

    print()
    print('  ★ 해석 규칙(CLAUDE.md 8절-5, 재개정 필요 - ④): 근거는 probe_bus_sweep 같은 외부 '
          '고정-(S,E) 스윕이 아니라 이 시뮬레이션 자체다(스윕은 메커니즘 규명용 참고자료일 뿐 '
          'PSO 품질 평가 지표가 아니다). PSO가 서로 다른 위치를 고른 정상 run들의 j_net 폭이 '
          '좁으면 "무차별"이 아니라 "유의하지만 최적권이 넓다"로 쓴다.', flush=True)

    bstar = dict(n_runs=n_normal, hist=hist, n_distinct=n_distinct, mode_b=mode_key,
                 sample_warning=sample_warning, jnet_spread_pct=jnet_spread_pct,
                 effective_n_units_median=effective_n_units_median,
                 effective_n_units_dist=effective_n_units_dist,
                 spread_scope=spread_scope)
    return dict(bstar=bstar, n_distinct=n_distinct, mode_b=mode_key)


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

def section4_convergence(group, decomp):
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

    # ⑤ degenerate(전 기 소멸) run은 gbest_f가 0 근방이라 재시작 분산을 왜곡한다 - 기본값은
    # 정상만(normal), _all은 전체 포함(참고용)으로 병기한다. decomp가 없으면(구 스키마) 구분할
    # 수 없으므로 전체를 그대로 "정상만" 자리에 쓴다(둘이 같아짐 - 저하가 아니라 정보 부족).
    normal_idx = {r['run'] for r in decomp['normal']} if decomp else None
    gbest_all = [r['gbest_f'] for r in group['runs'] if r['gbest_f'] is not None]
    if normal_idx is not None:
        gbest_normal = [r['gbest_f'] for r in group['runs']
                         if r['gbest_f'] is not None and r['run'] in normal_idx]
    else:
        gbest_normal = gbest_all

    stats_normal = _stats(gbest_normal)
    stats_all = _stats(gbest_all)
    print(f"\n  재시작 분산(⑤, 기본값=정상만 제외): median={stats_normal['median']:+.4e}  "
          f"range=[{stats_normal['min']:+.4e}, {stats_normal['max']:+.4e}]", flush=True)
    print(f"  재시작 분산(_all, 전체 포함): median={stats_all['median']:+.4e}  "
          f"range=[{stats_all['min']:+.4e}, {stats_all['max']:+.4e}]", flush=True)

    spread_normal = spread_all = float('nan')
    if stats_normal['n'] >= 2 and stats_normal['median'] != 0:
        spread_normal = (stats_normal['max'] - stats_normal['min']) / abs(stats_normal['median']) * 100
        print(f"  gbest_f 폭(정상만) = {spread_normal:.2f}%", flush=True)
    if stats_all['n'] >= 2 and stats_all['median'] != 0:
        spread_all = (stats_all['max'] - stats_all['min']) / abs(stats_all['median']) * 100
        print(f"  gbest_f 폭(_all) = {spread_all:.2f}%", flush=True)

    print()
    print('  ★ 판정은 곡선의 평탄함이 아니라 재시작 분산(run간 gbest 분포)으로 한다(CLAUDE.md 7절 - '
          '곡선 평탄화는 w 선형감소에 의한 소진수렴의 증거일 뿐 전역최적 도달을 함의하지 않는다).',
          flush=True)
    print('  판정표:', flush=True)
    print('    gbest_f 분산 小 + gbest_x 뭉침  -> 전역최적의 강한 증거', flush=True)
    print('    gbest_f 분산 大                 -> 탐색 부족(입자 수 증설 필요)', flush=True)
    print('    f는 뭉치는데 x는 흩어짐         -> 최적권이 넓음(8절-5)', flush=True)
    print('    f도 흩어짐                      -> 탐색 실패', flush=True)

    return dict(gbest_stats_normal=stats_normal, gbest_stats_all=stats_all,
                spread_pct_normal=spread_normal, spread_pct_all=spread_all)


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
    # ⑤ 예외: 다른 지표와 달리 degenerate(j_net<1e3) run을 여기서는 제외하지 않는다. 소멸 run의
    # min_bdefer는 어차피 잡음(~0)이라 넣어도 무해하지만, "탐색 중 실제로 음의 B_defer가
    # 나왔는가"라는 질문은 개별 값의 크기(NEG_BDEFER_NOISE_ATOL_WON)로 판단하는 것이지
    # j_net 크기로 판단하는 게 아니다 - 살아있는 다수기 run(정상)의 진짜 음편익을 놓치면 안 되므로
    # 전체 run을 그대로 집계한다.
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

    # ② boundary_warning을 불리언만 남기지 않고 실제 정규화 위치(s_boundary_frac/e_boundary_frac)를
    # 컬럼으로 남긴다 - 심사 방어 논거("경계에서 얼마나 떨어져 있는가")로 쓰려면 숫자가 있어야 한다.
    # 설치된 기가 여럿이면 세미콜론으로 이어붙인다(probe_split.py의 다기 컬럼 표기와 동일 관례).
    s_fracs, e_fracs, flags = [], [], []
    n_installed = len(operating['installed'])
    for i, (b, S, E) in enumerate(operating['installed']):
        s_frac = (S - PM.S_BOUNDS[0]) / (PM.S_BOUNDS[1] - PM.S_BOUNDS[0])
        e_frac = (E - PM.E_BOUNDS[0]) / (PM.E_BOUNDS[1] - PM.E_BOUNDS[0])
        s_warn = s_frac < BOUNDARY_WARN_FRAC or s_frac > 1 - BOUNDARY_WARN_FRAC
        e_warn = e_frac < BOUNDARY_WARN_FRAC or e_frac > 1 - BOUNDARY_WARN_FRAC
        s_fracs.append(s_frac)
        e_fracs.append(e_frac)
        prefix = f'unit{i}:' if n_installed > 1 else ''
        if s_warn:
            flags.append(f'{prefix}S={s_frac:.3f}')
        if e_warn:
            flags.append(f'{prefix}E={e_frac:.3f}')
        print(f"  기 {i}(b={b}): S={S:.4f} ({s_frac * 100:.2f}% of range)"
              f"{' *경계근접' if s_warn else ''}  "
              f"E={E:.4f} ({e_frac * 100:.2f}% of range){' *경계근접' if e_warn else ''}", flush=True)

    warned = bool(flags)
    warning_str = ';'.join(flags)
    print(f"  판정: {'*경계 근접 발견(' + warning_str + ') - 탐색범위 재검토 필요' if warned else '경계에서 충분히 떨어져 있음(결과가 탐색범위 상한/하한에 의존하지 않음)'}",
          flush=True)
    return dict(
        warned=warned, warning_str=warning_str,
        s_boundary_frac=';'.join(f'{v:.4f}' for v in s_fracs),
        e_boundary_frac=';'.join(f'{v:.4f}' for v in e_fracs),
    )


# ============================================================
# (8) 평균일/최대일 스케줄 유사도
# ============================================================

_SIMILARITY_SEASON_KEYS = []
for _season in ('summer', 'winter'):
    _SIMILARITY_SEASON_KEYS += [
        f'corr_{_season}_avg_peak', f'l1norm_{_season}',
        f'dis_peak_hours_{_season}_avg', f'dis_peak_hours_{_season}_peak',
    ]
del _season


def _nan_similarity_row():
    """소멸 기(is_active=False)의 유사도 행 - dis_peak_hours류는 문자열이라 빈 문자열,
    나머지는 NaN(② 지시사항 "유사도=NaN")."""
    return {k: ('' if 'dis_peak_hours' in k else float('nan')) for k in _SIMILARITY_SEASON_KEYS}


def _discharge_peak_hours(P_net, peak_frac=0.9):
    """① 방전 봉우리 시각 집합만 남긴다(정보손실 없는 원자료). 파생 단일통계(argmax,
    SMP가중무게중심)는 전부 뺐다 - 둘 다 다봉에서 오도했다: argmax는 이중봉을 단봉인 척
    보고했고(실측: summer_avg 이중봉 0;16의 argmax는 t=0만 집었다), 무게중심은 방전이
    실제로 0인 중간 시각을 반환했다(실측: 이중봉 0;16의 무게중심 t=8, 삼중봉 0;10;11의
    무게중심 t=7 - 둘 다 그 시각엔 방전이 없다). SMP는 schedule CSV에 이미 있으므로
    봉우리 시각과 SMP의 대조는 사후분석에서 이 원자료로 직접 한다."""
    p_dis = np.maximum(P_net, 0.0)
    max_dis = float(np.max(p_dis))
    if max_dis <= 0:
        return ''
    peak_hours = [t for t in range(len(p_dis)) if p_dis[t] >= peak_frac * max_dis]
    return ';'.join(str(t) for t in peak_hours)


def section8_schedule_similarity(group, all_units, installed_units, schedules):
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

    # ② 모든 기(전체 all_units 인덱스 기준)에 대해 유사도를 낸다 - postprocess_units 파일이
    # 기마다 1행이므로 대표 유닛 하나만 뽑지 않는다. 소멸 기는 NaN 행.
    per_unit = {}
    inst_idx = 0
    for full_idx, (b, S, E) in enumerate(all_units):
        is_active = S > 1e-9 and E > 1e-9
        if not is_active:
            per_unit[full_idx] = _nan_similarity_row()
            continue
        i = inst_idx
        inst_idx += 1

        print(f'\n  기 {full_idx}(b={b}, S={S:.4f}MVA):', flush=True)
        row = {}
        for s_avg, s_peak in main_pairs:
            season = 'summer' if s_avg == 'summer' else 'winter'
            P_avg = schedules[s_avg][i][0]
            P_peak = schedules[s_peak][i][0]
            if np.std(P_avg) > 0 and np.std(P_peak) > 0:
                corr = float(np.corrcoef(P_avg, P_peak)[0, 1])
            else:
                corr = float('nan')
            l1 = float(np.mean(np.abs(P_avg / S - P_peak / S))) if S > 0 else float('nan')

            peak_hours_avg = _discharge_peak_hours(P_avg)
            peak_hours_peak = _discharge_peak_hours(P_peak)

            print(f'    {s_avg} vs {s_peak}: 상관계수={corr:+.4f}  정규화L1거리={l1:.4f}', flush=True)
            print(f"      방전피크시각(최댓값 90%이상 전부): {s_avg}=[{peak_hours_avg}]  "
                  f"{s_peak}=[{peak_hours_peak}]  (SMP 대조는 schedule CSV로 직접 할 것)",
                  flush=True)

            row[f'corr_{season}_avg_peak'] = corr
            row[f'l1norm_{season}'] = l1
            row[f'dis_peak_hours_{season}_avg'] = peak_hours_avg
            row[f'dis_peak_hours_{season}_peak'] = peak_hours_peak

        print('    (참고, 계절이 다름 - 주 결과 아님) shoulder 대조:', flush=True)
        for s_ref, s_peak in ref_pairs:
            P_ref = schedules[s_ref][i][0]
            P_peak = schedules[s_peak][i][0]
            if np.std(P_ref) > 0 and np.std(P_peak) > 0:
                corr = float(np.corrcoef(P_ref, P_peak)[0, 1])
            else:
                corr = float('nan')
            print(f'      {s_ref} vs {s_peak}: 상관계수={corr:+.4f}', flush=True)

        per_unit[full_idx] = row

    print()
    print('  ★ ① dis_peak_hours만 남긴다(원자료). argmax/SMP가중무게중심 같은 파생 단일통계는 '
          '다봉에서 오도했다(실측: summer_avg 이중봉 0;16의 무게중심 t=8, winter_avg 삼중봉 '
          '0;10;11의 무게중심 t=7 - 둘 다 방전이 0인 시각이었다). 봉우리 시각과 SMP의 대조는 '
          'schedule CSV(같은 t에 smp_won_per_kwh 컬럼이 있음)로 사후분석에서 직접 할 것.',
          flush=True)
    print('  ★ 해석 규칙(CLAUDE.md 8절-7, 반드시 이 서술 유지): 유사하다는 결과가 나와도 그 근거를 '
          '"같은 정책"으로 쓰면 틀린다 - 평균일 LP는 가격(SMP)을, 최대일 LP는 부하를 좇으므로 '
          '목적이 다르다. 유사해지는 이유는 SMP-부하 高상관(위 수치, 참고 0.97) 때문이다. '
          '유사하면 "일관된 운영"의 방증, 다르면 "차익 유인 vs 피크저감 유인의 충돌"의 발견 - '
          '이쪽도 결과이지 실패가 아니다. 계절을 섞지 않는다(summer<->summer_peak, '
          'winter<->winter_peak만).', flush=True)

    return dict(per_unit=per_unit)


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


def process_group(runs_path, out_dir, ts):
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
        efc_result = section3_4_efc(group, operating['units'], operating['installed'], schedules)

    bus_dist = section3_5_bus_distribution(group, decomp)
    section3_6_q_ratio(group)
    convergence = section4_convergence(group, decomp)
    search_failure = section5_search_failure(group, decomp)
    neg_bdefer = section6_negative_bdefer(group)
    boundary = section7_boundary(group, operating)

    similarity_result = None
    schedule9_result = None
    if schedules is not None:
        similarity_result = section8_schedule_similarity(group, operating['units'],
                                                           operating['installed'], schedules)
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
# ② 요약을 run 수준 / 기(unit) 수준 두 파일로 분리
# ============================================================

def _representative_active_unit_corr(result):
    """③ 대표 유닛(첫 is_active=True 기) 기준 corr_summer_avg_peak/corr_winter_avg_peak을
    찾는다. 대표(0번) 유닛이 소멸 기일 수 있으므로(예: n=3 최우수 run은 unit0/1이 소멸,
    unit2만 활성) 반드시 첫 활성 기를 찾아야 한다."""
    op = result['operating']
    similarity = result['similarity']
    if not op or not similarity:
        return None, None
    units = op['units']
    per_unit = similarity['per_unit']
    for i, (b, S, E) in enumerate(units):
        if S > 1e-9 and E > 1e-9:
            row = per_unit.get(i, {})
            return row.get('corr_summer_avg_peak'), row.get('corr_winter_avg_peak')
    return None, None


def _incentive_conflict_label(corr):
    """③ CLAUDE.md 8절-7 해석 규칙: corr가 높아도 "같은 정책"이 아니다 - 평균일 LP는 SMP를,
    최대일 LP는 부하를 좇는다. 유사해지는 건 SMP-부하 高상관(0.97) 때문이고, 낮으면 그 상관이
    그날 조건에서 깨진 것이다 - 충돌은 실패가 아니라 발견이다."""
    if corr is None or (isinstance(corr, float) and math.isnan(corr)):
        return ''
    if corr >= 0.7:
        return 'aligned'
    if corr >= 0.3:
        return 'partial'
    return 'conflict'


RUN_SUMMARY_FIELDS = [
    'n_ess', 'n_runs', 'n_degenerate',
    # ③ median을 주값으로 유지, mean을 옆에 병기(_mean). degenerate 기본 제외, _all=전체 포함.
    'j_net_median', 'j_net_mean', 'j_net_median_all', 'j_net_mean_all',
    'b_defer_share_of_gross_pct', 'b_defer_share_of_gross_pct_mean',
    'b_defer_share_of_gross_pct_all', 'b_defer_share_of_gross_pct_all_mean',
    'b_arb_share_of_gross_pct', 'b_arb_share_of_gross_pct_mean',
    'b_arb_share_of_gross_pct_all', 'b_arb_share_of_gross_pct_all_mean',
    'b_loss_share_of_gross_pct', 'b_loss_share_of_gross_pct_mean',
    'b_loss_share_of_gross_pct_all', 'b_loss_share_of_gross_pct_all_mean',
    'cost_to_gross_ratio_pct', 'cost_to_gross_ratio_pct_mean',
    'cost_to_gross_ratio_pct_all', 'cost_to_gross_ratio_pct_all_mean',
    'decomposition_violations', 'cost_check_ok',
    'best_run', 'best_gbest_f', 'best_j_net',
    's_boundary_frac', 'e_boundary_frac', 'boundary_warning',
    'gbest_median', 'gbest_mean', 'gbest_median_all', 'gbest_mean_all',
    'gbest_spread_pct', 'gbest_spread_pct_all',
    # ⑥ b* 분포 요약(degenerate 제외 - section3_5_bus_distribution 참조)
    'bstar_distinct_count', 'bstar_jnet_spread_pct', 'bstar_spread_scope',
    'bstar_n_runs', 'bstar_sample_warning',
    # ② 다수기 실질 기수 분포(6-A절 "최적 기수 1" 결론의 직접 통계 증거)
    'effective_n_units_median', 'effective_n_units_dist',
    # ③ 유인충돌 판정(대표 활성 기 기준, 계절 내 corr) - CLAUDE.md 8절-7
    'incentive_conflict_summer', 'incentive_conflict_winter',
    'total_negative_bdefer', 'min_bdefer_overall',
]


def build_run_summary_row(result):
    if result is None:
        return None
    g = result['group']
    decomp = result['decomp']
    cost_check = result['cost_check']
    boundary = result['boundary'] or {}
    convergence = result['convergence'] or {}
    neg_bdefer = result['neg_bdefer'] or {}
    bus_dist = result['bus_dist'] or {}
    bstar = bus_dist.get('bstar') or {}

    def gross_stat(key, gross_list, field):
        if not gross_list:
            return ''
        return _stats([gv[key] for gv in gross_list])[field]

    j_all = _stats([r['j_net'] for r in g['runs']]) if decomp else {}
    j_normal = _stats([r['j_net'] for r in decomp['normal']]) if decomp else {}
    gbest_normal = convergence.get('gbest_stats_normal') or {}
    gbest_all_stats = convergence.get('gbest_stats_all') or {}
    gn = decomp['gross_normal'] if decomp else None
    ga = decomp['gross_all'] if decomp else None
    corr_summer, corr_winter = _representative_active_unit_corr(result)

    return dict(
        n_ess=g['n_ess'], n_runs=len(g['runs']),
        n_degenerate=len(decomp['degenerate']) if decomp else '',
        j_net_median=j_normal.get('median', ''), j_net_mean=j_normal.get('mean', ''),
        j_net_median_all=j_all.get('median', ''), j_net_mean_all=j_all.get('mean', ''),
        b_defer_share_of_gross_pct=gross_stat('defer_pct', gn, 'median') if decomp else '',
        b_defer_share_of_gross_pct_mean=gross_stat('defer_pct', gn, 'mean') if decomp else '',
        b_defer_share_of_gross_pct_all=gross_stat('defer_pct', ga, 'median') if decomp else '',
        b_defer_share_of_gross_pct_all_mean=gross_stat('defer_pct', ga, 'mean') if decomp else '',
        b_arb_share_of_gross_pct=gross_stat('arb_pct', gn, 'median') if decomp else '',
        b_arb_share_of_gross_pct_mean=gross_stat('arb_pct', gn, 'mean') if decomp else '',
        b_arb_share_of_gross_pct_all=gross_stat('arb_pct', ga, 'median') if decomp else '',
        b_arb_share_of_gross_pct_all_mean=gross_stat('arb_pct', ga, 'mean') if decomp else '',
        b_loss_share_of_gross_pct=gross_stat('loss_pct', gn, 'median') if decomp else '',
        b_loss_share_of_gross_pct_mean=gross_stat('loss_pct', gn, 'mean') if decomp else '',
        b_loss_share_of_gross_pct_all=gross_stat('loss_pct', ga, 'median') if decomp else '',
        b_loss_share_of_gross_pct_all_mean=gross_stat('loss_pct', ga, 'mean') if decomp else '',
        cost_to_gross_ratio_pct=gross_stat('cost_pct', gn, 'median') if decomp else '',
        cost_to_gross_ratio_pct_mean=gross_stat('cost_pct', gn, 'mean') if decomp else '',
        cost_to_gross_ratio_pct_all=gross_stat('cost_pct', ga, 'median') if decomp else '',
        cost_to_gross_ratio_pct_all_mean=gross_stat('cost_pct', ga, 'mean') if decomp else '',
        decomposition_violations=len(decomp['decomposition_bad']) if decomp else '',
        cost_check_ok=cost_check['ok'] if cost_check else '',
        best_run=result['best']['run'] if result['best'] else '',
        best_gbest_f=result['best']['gbest_f'] if result['best'] else '',
        best_j_net=result['best']['j_net'] if result['best'] else '',
        s_boundary_frac=boundary.get('s_boundary_frac', ''),
        e_boundary_frac=boundary.get('e_boundary_frac', ''),
        boundary_warning=boundary.get('warning_str', ''),
        gbest_median=gbest_normal.get('median', ''), gbest_mean=gbest_normal.get('mean', ''),
        gbest_median_all=gbest_all_stats.get('median', ''), gbest_mean_all=gbest_all_stats.get('mean', ''),
        gbest_spread_pct=convergence.get('spread_pct_normal', ''),
        gbest_spread_pct_all=convergence.get('spread_pct_all', ''),
        bstar_distinct_count=bstar.get('n_distinct', ''),
        bstar_jnet_spread_pct=bstar.get('jnet_spread_pct', ''),
        bstar_spread_scope=bstar.get('spread_scope', ''),
        bstar_n_runs=bstar.get('n_runs', ''), bstar_sample_warning=bstar.get('sample_warning', ''),
        effective_n_units_median=bstar.get('effective_n_units_median', ''),
        effective_n_units_dist=bstar.get('effective_n_units_dist', ''),
        incentive_conflict_summer=_incentive_conflict_label(corr_summer),
        incentive_conflict_winter=_incentive_conflict_label(corr_winter),
        total_negative_bdefer=neg_bdefer.get('total_negative', ''),
        min_bdefer_overall=neg_bdefer.get('overall_min', ''),
    )


UNITS_FIELDS = [
    'n_ess', 'unit_idx', 'bus', 'S_rated', 'E_rated', 'is_active',
    'eta_peak', 'duration_h', 'e_over_s', 'annual_efc',
    'corr_summer_avg_peak', 'corr_winter_avg_peak',
    'l1norm_summer', 'l1norm_winter',
    # ① dis_hour_*(argmax)와 dis_hour_smp_weighted_*는 삭제 - 봉우리 원자료(dis_peak_hours)만
    # 남긴다(다봉에서 파생 단일통계가 오도했다). SMP 대조는 schedule CSV로 직접 할 것.
    'dis_peak_hours_summer_avg', 'dis_peak_hours_summer_peak',
    'dis_peak_hours_winter_avg', 'dis_peak_hours_winter_peak',
]


def compute_per_unit_operating(units, b_defer):
    """② eta_peak/duration_h/e_over_s를 기(unit)마다 계산한다.
    duration_h_i, e_over_s_i는 그 기 자신의 S_i,E_i만으로 정의되는 자기완결적 비율이라
    모호함이 없다(LP 불필요). eta_peak_i는 시스템 전체 deltaP_peak(b_defer에서 역산)를
    그 기의 S_i로 나눈 값이다.
    ★ 귀속 캐비어트: solve_peak의 "자기 피크 최소화" 목적은 기별로 분리 가능한 형태가 아니다
    (probe_split.py 실측 - 독립적으로 푼 기들의 피크저감을 단순 합하면 한 기가 전체 용량으로
    푸는 것보다 항상 손해였다). 즉 b_defer는 시스템 전체값이지 기별 기여분으로 쪼개지지 않는다.
    현재 실측 데이터는 활성 기가 항상 최대 1개라 이 값이 시스템값과 정확히 같지만, 활성 기가
    여럿인 경우 이 eta_peak_i는 "그 기 혼자 전체 용량이었다면"이라는 가정적 수치이지 그 기의
    실제 기여분이 아니다."""
    if b_defer is None:
        delta_p_peak_mw = None
    else:
        delta_p_peak_mw = b_defer / PM.C_CAP_PER_MW_YR

    results = []
    for i, (b, S, E) in enumerate(units):
        is_active = S > 1e-9 and E > 1e-9
        if not is_active or delta_p_peak_mw is None:
            results.append(dict(unit_idx=i, eta_peak=float('nan'), duration_h=float('nan'),
                                 e_over_s=float('nan')))
            continue
        results.append(dict(unit_idx=i, eta_peak=delta_p_peak_mw / S,
                             duration_h=E * PM.DOD / S, e_over_s=E / S))
    return results


def build_units_rows(result):
    """② 기(unit) 수준 파일의 행들 - 최우수 run의 기마다 1행(소멸 기 포함, n=1이면 1행)."""
    if result is None or result['operating'] is None:
        return []
    n_ess = result['n_ess']
    units = result['operating']['units']
    best = result['best']
    b_defer = best['b_defer'] if best else None

    per_unit_op = {o['unit_idx']: o for o in compute_per_unit_operating(units, b_defer)}
    efc_by_idx = {r['unit_idx']: r for r in (result['efc']['per_unit'] if result['efc'] else [])}
    sim_by_idx = (result['similarity']['per_unit'] if result['similarity'] else {})

    rows = []
    for i, (b, S, E) in enumerate(units):
        is_active = S > 1e-9 and E > 1e-9
        op = per_unit_op.get(i, dict(eta_peak=float('nan'), duration_h=float('nan'),
                                      e_over_s=float('nan')))
        efc_row = efc_by_idx.get(i, {})
        sim_row = _nan_similarity_row()
        sim_row.update(sim_by_idx.get(i, {}))

        row = dict(
            n_ess=n_ess, unit_idx=i, bus=b, S_rated=S, E_rated=E, is_active=is_active,
            eta_peak=op['eta_peak'], duration_h=op['duration_h'], e_over_s=op['e_over_s'],
            annual_efc=efc_row.get('annual_efc', float('nan')),
        )
        for k in UNITS_FIELDS:
            if k not in row:
                row[k] = sim_row.get(k, float('nan'))
        rows.append(row)
    return rows


def write_group_outputs(result, out_dir, ts):
    """① 기수마다 별도 파일 - run 수준 요약 1개(postprocess_n{k}) + 기 수준 1개
    (postprocess_units_n{k})를 즉시 낸다. 다른 기수 결과와 합쳐지지 않는다."""
    n_ess = result['n_ess']
    run_row = build_run_summary_row(result)
    run_path = os.path.join(out_dir, f'postprocess_n{n_ess}_{ts}.csv')
    _write_csv(run_path, RUN_SUMMARY_FIELDS, [run_row] if run_row else [])

    units_rows = build_units_rows(result)
    units_path = os.path.join(out_dir, f'postprocess_units_n{n_ess}_{ts}.csv')
    _write_csv(units_path, UNITS_FIELDS, units_rows)

    print(f"\n[n_ess={n_ess}] run 수준 요약: {run_path}", flush=True)
    print(f"[n_ess={n_ess}] 기 수준 요약({len(units_rows)}행): {units_path}", flush=True)
    if run_row:
        print(f"[n_ess={n_ess}] 유인충돌 판정(③, 대표 활성 기 기준): "
              f"summer={run_row['incentive_conflict_summer'] or '(계산불가)'}  "
              f"winter={run_row['incentive_conflict_winter'] or '(계산불가)'}", flush=True)
    return run_path, units_path


# ============================================================
# CLI
# ============================================================

def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description='postprocess: CLAUDE.md 8절 후처리 지표 계산 (사후분석 전용, 최적화 미개입). '
                    '① 기수마다 별도 파일(run 수준 + 기 수준)을 낸다.',
        epilog='예: python postprocess.py results/runs_n1_<ts>.csv  (기수 1개당 1회 실행 권장)\n'
               '    여러 기수를 한 번에 넘겨도 파일은 기수별로 따로 나온다. --compare는 화면에서만'
               ' 비교 뷰를 추가로 보여준다.',
    )
    parser.add_argument('runs', nargs='+',
                         help='runs_n<k>_<ts>.csv 경로(들). generations는 파일명 규약으로 자동 매칭.')
    parser.add_argument('--compare', action='store_true',
                         help='① 여러 기수를 함께 넘겼을 때 화면에 기수-순편익 비교(선택적 병합 뷰)를 '
                              '추가로 출력한다. 파일은 만들지 않는다(기본 꺼짐).')
    parser.add_argument('--results-dir', default=RESULTS_DIR_DEFAULT,
                         help='postprocess_*.csv / postprocess_units_*.csv / schedule_*.csv 출력 '
                              '위치 (기본: results/).')
    return parser


def main():
    args = _build_arg_parser().parse_args()
    _check_env()

    print('evaluate.init_worker() 호출 - 기저 조류계산 120회 1회만 캐싱(이후 (9)절에서 재사용).',
          flush=True)
    evaluate.init_worker()

    os.makedirs(args.results_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    all_results = []
    missing_report = []
    output_paths = []
    for runs_path in args.runs:
        result = process_group(runs_path, args.results_dir, ts)
        all_results.append(result)
        if result is not None:
            run_path, units_path = write_group_outputs(result, args.results_dir, ts)
            output_paths.append((result['n_ess'], run_path, units_path, result.get('schedule9')))
            if result['group']['missing_cols']:
                missing_report.append((result['n_ess'], runs_path, result['group']['missing_cols']))

    if args.compare:
        print_cross_n_curve(all_results)

    section('종합 요약')
    for n_ess, run_path, units_path, sched in output_paths:
        print(f'  n_ess={n_ess}:', flush=True)
        print(f'    run 수준: {run_path}', flush=True)
        print(f'    기 수준 : {units_path}', flush=True)
        print(f"    스케줄  : {sched['path'] if sched else '(생성 안 됨)'}", flush=True)

    if missing_report:
        print('\n★ 로그에 없어 생략된 지표(main.py 수정 여부는 이 목록을 보고 사람이 판단할 것):',
              flush=True)
        for n_ess, path, cols in missing_report:
            print(f'  n_ess={n_ess} ({path}): {cols}', flush=True)
    else:
        print('\n모든 입력 파일이 현행 스키마를 갖추고 있어 생략된 지표 없음.', flush=True)


if __name__ == '__main__':
    main()
