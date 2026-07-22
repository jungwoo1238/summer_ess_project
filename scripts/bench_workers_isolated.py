"""워커 수 독립효과 벤치마크 (확인 전용, bench_workers.py의 자매품 - 본 파이프라인 미포함).

기존 `scripts/bench_workers.py`는 워커:입자 비를 1:2로 **묶어서** 스윕했다
((10,20)/(12,24)/(16,32) 등). 이 설계는 워커 수의 독립 효과를 입자 수(=작업량) 변화와
교락(confounding)시킨다 - (16,32)가 (10,20)보다 빠를 때 그게 워커가 16이라 빠른 건지
입자가 32라 병렬 활용이 좋아진 건지 분리되지 않는다.

이 스크립트는 **n_particles=32로 고정**하고 워커 수만 {4,8,10,12,16}으로 스윕해 워커 수
하나만의 효과를 분리한다. 이번에는 **측정만** 한다 - 결과에 따라 실제 워커 수(main.py의
DEFAULT_N_WORKERS=16)를 바꿀지는 별도 판단이며 이 스크립트의 범위 밖이다.

★★★ 실행 주체: 이 벤치마크는 사용자가 데스크탑 터미널에서 직접 실행한다. ★★★
Claude Code(VS Code 확장) 세션이 자체적으로 실행하지 않는다 - VS Code 경유로 병렬 워커를
띄우면 메모리 부족으로 데스크탑이 다운되는 문제가 실제로 있었다(main.py full 실행에서
실측된 것과 동일 원인 - CLAUDE.md 7절, "결과CSV/실행 기록" 참조). 파일 하단 "실행 안내"
참조.

MKL_THREADING_LAYER=SEQUENTIAL 전제(conda env 'ess'에 이미 설정 - CLAUDE.md 7절). 이 값이
아니면 MKL 기본 스레딩이 워커 스레드풀을 만드는 시점에 무증상 종료할 수 있어, 시작 전에
확인하고 아니면 즉시 중단한다(bench_workers.py와 동일 방어).

evaluate/benefits/lower_lp/pso_core/main.py는 전혀 수정하지 않는다. main.py가 실제
본실험에서 쓰는 병렬 실행 경로(main.RunObjective + main._build_bounds/_build_int_dims +
pso_core.PSO)를 그대로 import해 재사용한다 - 이 스크립트가 만드는 것은 n_workers를
스윕하는 바깥 루프뿐이다. 그래서 여기서 재는 부하는 "장난감 벤치마크"가 아니라 실제
파이프라인과 동일한 evaluate_particle 호출(n_ess=1, 120 PF/평가)이다.

실행 시간 참고(대략치, ideal 선형 스케일 가정 - 실제로는 오버헤드 때문에 이보다 오래 걸림):
  1회 evaluate_particle ~= 0.685초(CLAUDE.md 7절 실측치) x 32입자 x (10세대+1) ~= 241초(순차).
  워커 수로 나누면 W=4: ~60초/rep, W=16: ~15초/rep. 3 rep x 5개 워커수 조합 총합 대략
  8~12분 수준으로 예상(Pool 생성 오버헤드 별도).
"""
import os
import sys
import csv
import gc
import math
import time
import socket
import datetime
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import params as PM
import evaluate
import pso_core
import main as M  # main.py의 실제 병렬 실행 경로(RunObjective 등)를 그대로 재사용. 수정 안 함.

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ============================================================
# 설정
# ============================================================
N_ESS = 1                    # main.RunObjective가 요구하는 기수 - 벤치마크는 항상 단일기
N_PARTICLES_BENCH = 32       # ★ 전 조합 고정 - 이 실험의 핵심(워커 효과만 분리)
N_ITERS_BENCH = 10           # 짧은 실행으로 워커 효율만 잰다(본실험 100세대까지 갈 필요 없음)
N_REPS = 3                   # 조합당 반복 횟수(타이밍 잡음 축소용 - 시드 고정이라 결과는 결정론적)
BENCH_SEED = 42              # 전 조합·전 rep 동일 시드 - 입자 초기배치·평가부하를 동일하게 유지

# 첫 rep을 워밍업으로 보고 요약(중앙값/최솟값) 계산에서 제외할지. CSV에는 3 rep 전부 남긴다
# (버리는 건 "화면 요약 통계"뿐 - 원자료는 항상 보존).
DISCARD_FIRST_REP_AS_WARMUP = True

# ★ 상한 16 고정(기존 bench_workers.py 실측: 20/24워커에서 시스템 정지). 하한 4는
# "웨이브를 늘리면(32/4=8웨이브) 오히려 빨라지는가"를 보는 대조군. 10/12는 32의 약수가
# 아니라 웨이브 경계에서 유휴 슬롯이 생기는 조합(정수배 vs 비정수배 비교용).
WORKERS_GRID = [4, 8, 10, 12, 16]

MIN_FREE_MB_PER_WORKER = 400  # bench_workers.py와 동일 기준(조합 시작 전 메모리 사전점검)

# full 실행 참고치(CLAUDE.md 7-A절 실행 기록·이번 대화의 사용자 진술) - 이 스크립트의
# 측정치가 같은 자릿수인지 대조하는 용도. 이 스크립트가 산출한 값이 아니라 외부 참고값이므로
# 리터럴로 둔다(단일 진실 원천은 어디까지나 CLAUDE.md).
REFERENCE_FULL_N_WORKERS = 16
REFERENCE_FULL_GEN_TIME_S = 3.0        # ~= 300초 / 100세대
REFERENCE_FULL_SPEEDUP_AT_16 = 7.3     # 이론상한 16배의 46%

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_FIELDS = [
    'n_workers', 'n_particles', 'n_iters_bench', 'rep',
    'wall_time_s', 'eval_throughput', 'n_waves', 'idle_slots', 'completed_ok',
]


def _make_results_path():
    out_dir = os.path.join(SCRIPT_DIR, 'results', 'bench_workers_isolated')
    os.makedirs(out_dir, exist_ok=True)
    hostname = socket.gethostname()
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(out_dir, f'bench_workers_isolated_{hostname}_{ts}.csv')


# ============================================================
# 환경 확인 (bench_workers.py와 동일 방어 - 경고가 아니라 중단)
# ============================================================

def _check_env_or_exit():
    val = os.environ.get('MKL_THREADING_LAYER')
    print(f'MKL_THREADING_LAYER = {val!r}', flush=True)
    if val != 'SEQUENTIAL':
        print(
            "\n[중단] MKL_THREADING_LAYER가 'SEQUENTIAL'이 아닙니다.\n"
            "  MKL 기본 스레딩(Intel OpenMP)이 워커 프로세스의 스레드 풀을 생성하는 시점에\n"
            "  프로세스가 무증상 종료할 수 있습니다(CLAUDE.md 7절 실측 확인 증상).\n"
            "  conda env에 설정한 뒤(코드에서 os.environ으로 설정하지 않음) 새 셸에서 "
            "다시 실행하세요.\n\n"
            "    conda env config vars set MKL_THREADING_LAYER=SEQUENTIAL -n ess\n",
            flush=True,
        )
        sys.exit(1)


def _enough_memory_for(n_workers):
    if not _HAS_PSUTIL:
        return True, None, None
    available_mb = psutil.virtual_memory().available / (1024 ** 2)
    required_mb = n_workers * MIN_FREE_MB_PER_WORKER
    return available_mb >= required_mb, available_mb, required_mb


# ============================================================
# CSV: 조합/rep마다 즉시 append + flush + fsync (bench_workers.py와 동일 패턴 -
# 중간에 죽어도 이미 쓴 행은 보존됨)
# ============================================================

def _init_csv(path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        f.flush()
        os.fsync(f.fileno())


def _append_csv_row(path, row):
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def _waves_and_idle(n_workers, n_particles):
    """이론값(실측 아님) - chunksize=1의 동적 로드밸런싱은 균등한 '웨이브'로 정확히
    떨어지지 않을 수 있음(CLAUDE.md 7절: 입자별 평가시간이 불균등). 여기서는 단순
    ceil(n_particles/n_workers) 정의를 그대로 쓴다(사용자 요청 정의)."""
    n_waves = math.ceil(n_particles / n_workers)
    idle_slots = n_waves * n_workers - n_particles
    return n_waves, idle_slots


def _failed_row(n_workers, rep, n_waves, idle_slots):
    return dict(
        n_workers=n_workers, n_particles=N_PARTICLES_BENCH, n_iters_bench=N_ITERS_BENCH,
        rep=rep, wall_time_s=float('nan'), eval_throughput=float('nan'),
        n_waves=n_waves, idle_slots=idle_slots, completed_ok=False,
    )


# ============================================================
# 조합 1개 실행 (Pool 1회 생성 -> N_REPS회 pso.optimize() 반복, main.py와 동일하게
# Pool을 run들 사이에 재사용 - init_worker의 기저 캐싱 비용을 rep마다 다시 치르지 않음)
# ============================================================

def _run_combo(n_workers, results_csv, gbest_log):
    n_waves, idle_slots = _waves_and_idle(n_workers, N_PARTICLES_BENCH)
    n_evals_per_run = N_PARTICLES_BENCH * (N_ITERS_BENCH + 1)  # 초기평가 1회 + n_iters 루프

    ok, available_mb, required_mb = _enough_memory_for(n_workers)
    if not ok:
        print(f'  -> SKIPPED (메모리 부족): available={available_mb:.0f}MB < '
              f'required={required_mb:.0f}MB', flush=True)
        for rep in range(N_REPS):
            row = _failed_row(n_workers, rep, n_waves, idle_slots)
            _append_csv_row(results_csv, row)
        return

    try:
        pool = mp.Pool(n_workers, initializer=evaluate.init_worker)
    except Exception as exc:
        print(f'  -> FAILED (Pool 생성 실패): {type(exc).__name__}: {exc}', flush=True)
        for rep in range(N_REPS):
            row = _failed_row(n_workers, rep, n_waves, idle_slots)
            _append_csv_row(results_csv, row)
        return

    try:
        for rep in range(N_REPS):
            try:
                bounds = M._build_bounds(N_ESS)
                int_dims = M._build_int_dims(N_ESS)
                objective = M.RunObjective(pool, N_ESS)
                pso = pso_core.PSO(
                    objective=objective,
                    bounds=bounds,
                    n_particles=N_PARTICLES_BENCH,
                    n_iters=N_ITERS_BENCH,
                    w_max=PM.PSO_W_MAX, w_min=PM.PSO_W_MIN,
                    c1=PM.PSO_C1, c2=PM.PSO_C2,
                    v_clamp_k=PM.PSO_V_MAX_RATIO,
                    int_dims=int_dims,
                    seed=BENCH_SEED,  # 전 조합·전 rep 동일 - 입자 궤적을 동일하게 유지
                )

                t0 = time.perf_counter()
                result = pso.optimize()
                wall_time_s = time.perf_counter() - t0

                throughput = n_evals_per_run / wall_time_s
                row = dict(
                    n_workers=n_workers, n_particles=N_PARTICLES_BENCH,
                    n_iters_bench=N_ITERS_BENCH, rep=rep,
                    wall_time_s=wall_time_s, eval_throughput=throughput,
                    n_waves=n_waves, idle_slots=idle_slots, completed_ok=True,
                )
                _append_csv_row(results_csv, row)
                gbest_log.append((n_workers, rep, float(result['f'])))

                print(f'  rep {rep + 1}/{N_REPS}: wall_time={wall_time_s:.2f}s  '
                      f'throughput={throughput:.3f}evals/s  gbest_f={result["f"]:.6e}', flush=True)
            except Exception as exc:
                print(f'  rep {rep + 1}/{N_REPS} FAILED: {type(exc).__name__}: {exc}', flush=True)
                row = _failed_row(n_workers, rep, n_waves, idle_slots)
                _append_csv_row(results_csv, row)
                # 이 조합의 워커풀이 이미 불안정할 수 있으니 남은 rep은 건너뛰고 다음 조합으로.
                break
    finally:
        pool.close()
        pool.join()

    gc.collect()
    time.sleep(2)  # 다음 조합 전 OS가 메모리를 회수할 시간을 준다(bench_workers.py와 동일)


# ============================================================
# 요약 출력
# ============================================================

def _print_summary(rows_by_worker, gbest_log):
    print('\n' + '=' * 100, flush=True)
    print('1) 조합별 wall_time(중앙값) 오름차순', flush=True)
    print('=' * 100, flush=True)
    header = (f"{'workers':>7} {'n_waves':>7} {'idle':>5} {'median_wall(s)':>14} "
              f"{'min_wall(s)':>11} {'median_throughput':>18}")
    print(header, flush=True)
    print('-' * len(header), flush=True)

    summary = []  # (n_workers, median_wall, min_wall, median_throughput, n_waves, idle)
    for n_workers in sorted(rows_by_worker.keys()):
        reps = rows_by_worker[n_workers]
        ok_reps = [r for r in reps if r['completed_ok']]
        if not ok_reps:
            print(f"{n_workers:>7}  (완주한 rep 없음)", flush=True)
            continue
        use_reps = ok_reps[1:] if (DISCARD_FIRST_REP_AS_WARMUP and len(ok_reps) > 1) else ok_reps
        walls = [r['wall_time_s'] for r in use_reps]
        throughputs = [r['eval_throughput'] for r in use_reps]
        median_wall = float(np.median(walls))
        min_wall = float(np.min(walls))
        median_throughput = float(np.median(throughputs))
        n_waves, idle = ok_reps[0]['n_waves'], ok_reps[0]['idle_slots']
        summary.append((n_workers, median_wall, min_wall, median_throughput, n_waves, idle))

    summary.sort(key=lambda t: t[1])
    for n_workers, median_wall, min_wall, median_throughput, n_waves, idle in summary:
        print(f"{n_workers:>7} {n_waves:>7} {idle:>5} {median_wall:>14.3f} "
              f"{min_wall:>11.3f} {median_throughput:>18.3f}", flush=True)

    print(f"\n(요약 통계는 DISCARD_FIRST_REP_AS_WARMUP={DISCARD_FIRST_REP_AS_WARMUP} 설정에 따라 "
          f"{'첫 rep 제외' if DISCARD_FIRST_REP_AS_WARMUP else '전체 rep 포함'}. "
          f"CSV 원자료에는 항상 3 rep 전부 남아 있음)", flush=True)

    # ------------------------------------------------------------------
    # 2) 처리율 기준 최적 워커 수 판정
    # ------------------------------------------------------------------
    print('\n' + '=' * 100, flush=True)
    print('2) 처리율(평가/초) 기준 판정', flush=True)
    print('=' * 100, flush=True)
    if summary:
        best = max(summary, key=lambda t: t[3])
        print(f"이 측정 기준 최고 처리율 워커 수: {best[0]} "
              f"(median_throughput={best[3]:.3f} evals/s)", flush=True)
    print("★ 이는 이번 측정 결과일 뿐이다 - 실제 워커 수(main.py DEFAULT_N_WORKERS) 변경 여부는 "
          "별도 판단 사항이며 이 스크립트의 범위 밖이다.", flush=True)

    # ------------------------------------------------------------------
    # 3) full 실행 참고치와 대조
    # ------------------------------------------------------------------
    print('\n' + '=' * 100, flush=True)
    print('3) full 실행 참고치와 대조', flush=True)
    print('=' * 100, flush=True)
    w16 = next((s for s in summary if s[0] == REFERENCE_FULL_N_WORKERS), None)
    w4 = next((s for s in summary if s[0] == min(WORKERS_GRID)), None)
    if w16 is not None:
        n_evals_per_run = N_PARTICLES_BENCH * (N_ITERS_BENCH + 1)
        gen_time_w16 = w16[1] / (N_ITERS_BENCH + 1)
        print(f"이 측정 W={REFERENCE_FULL_N_WORKERS} 세대당 시간(역산): {gen_time_w16:.3f}s/gen "
              f"(참고: full 실측 ~{REFERENCE_FULL_GEN_TIME_S:.1f}s/gen)", flush=True)
        ratio = gen_time_w16 / REFERENCE_FULL_GEN_TIME_S
        same_order = 0.3 <= ratio <= 3.0
        print(f"  비율 = {ratio:.2f}배 -> {'같은 자릿수(정합)' if same_order else '자릿수 차이 있음(불일치 - 확인 필요)'}",
              flush=True)
        if w4 is not None:
            speedup_vs_w4 = w4[1] / w16[1]
            print(f"이 벤치마크의 W={REFERENCE_FULL_N_WORKERS} vs W={min(WORKERS_GRID)} 가속비: "
                  f"{speedup_vs_w4:.2f}배 "
                  f"(★ W=1을 이 그리드에서 측정하지 않아 W={min(WORKERS_GRID)}를 기준으로 삼음 - "
                  f"full 참고치의 {REFERENCE_FULL_SPEEDUP_AT_16:.1f}배(W=1 대비 W=16)와 기준이 달라 "
                  "직접 비교는 아니고 참고용).", flush=True)
    else:
        print(f"W={REFERENCE_FULL_N_WORKERS} 조합이 완주되지 않아 대조 불가.", flush=True)

    # ------------------------------------------------------------------
    # 4) 정수배 vs 비정수배 처리율 차이
    # ------------------------------------------------------------------
    print('\n' + '=' * 100, flush=True)
    print('4) 웨이브 정수배(8,16) vs 비정수배(10,12) 처리율 비교', flush=True)
    print('=' * 100, flush=True)
    exact = {s[0]: s[3] for s in summary if s[0] in (8, 16)}
    inexact = {s[0]: s[3] for s in summary if s[0] in (10, 12)}
    for w, t in sorted(exact.items()):
        print(f"  정수배  W={w:>2} (idle=0)      : median_throughput={t:.3f} evals/s", flush=True)
    for w, t in sorted(inexact.items()):
        n_waves, idle = _waves_and_idle(w, N_PARTICLES_BENCH)
        print(f"  비정수배 W={w:>2} (idle={idle})    : median_throughput={t:.3f} evals/s", flush=True)

    # ------------------------------------------------------------------
    # 검증: 전 조합이 동일 시드/입자를 받아 gbest_f가 (거의) 같은지
    # ------------------------------------------------------------------
    print('\n' + '=' * 100, flush=True)
    print('검증: 전 조합·전 rep이 동일 결과(gbest_f)에 도달했는가 (처리율만 달라야 함)',
          flush=True)
    print('=' * 100, flush=True)
    if gbest_log:
        ref = gbest_log[0][2]
        # CLAUDE.md 7절 "수치 잡음의 성격": warm-start 이력 차이로 워커별 <1e-8 MW 수준
        # float 잡음이 있을 수 있고, 이게 fitness에서는 <7원/년 규모다. 10세대 누적을 감안해
        # 넉넉한 여유(1000원)를 허용오차로 둔다 - 완전 비트일치를 요구하지 않는다.
        tol_won = 1000.0
        max_diff = max(abs(f - ref) for _w, _r, f in gbest_log)
        all_close = max_diff <= tol_won
        print(f"기준값(첫 조합 rep0) gbest_f = {ref:.6e}", flush=True)
        print(f"전체 조합·rep 중 최대 편차 = {max_diff:.6e} 원 (허용오차 {tol_won:.0f}원)", flush=True)
        print(f"판정: {'통과 - 동일 시드/입자 확인됨' if all_close else '★ 불일치 - 워커 수에 따라 결과 자체가 달라짐(확인 필요)'}",
              flush=True)
        if not all_close:
            for w, r, f in gbest_log:
                diff = f - ref
                if abs(diff) > tol_won:
                    print(f"    workers={w} rep={r}: gbest_f={f:.6e} (diff={diff:+.6e})", flush=True)
    else:
        print('완주된 rep이 없어 검증 불가.', flush=True)


# ============================================================
# 메인
# ============================================================

def main():
    _check_env_or_exit()

    print(f'psutil 사용 가능: {_HAS_PSUTIL}'
          + ('' if _HAS_PSUTIL else ' (메모리 사전점검 생략)'), flush=True)
    print(f'n_particles={N_PARTICLES_BENCH}(고정) n_iters_bench={N_ITERS_BENCH} '
          f'n_reps={N_REPS} seed={BENCH_SEED}(전 조합 동일) workers_grid={WORKERS_GRID}',
          flush=True)

    results_csv = _make_results_path()
    _init_csv(results_csv)
    print(f'결과 CSV: {results_csv} (조합/rep마다 즉시 append)', flush=True)

    ordered_grid = sorted(WORKERS_GRID)  # 오름차순 고정(요구사항)
    rows_by_worker = {}
    gbest_log = []  # (n_workers, rep, gbest_f) - CSV에는 안 남기고 이 스크립트 안에서만 검증용

    for i, n_workers in enumerate(ordered_grid, 1):
        print(f'\n[{i}/{len(ordered_grid)}] workers={n_workers} (particles={N_PARTICLES_BENCH} 고정) 시작',
              flush=True)
        t_combo0 = time.perf_counter()
        _run_combo(n_workers, results_csv, gbest_log)
        print(f'  조합 완료 ({time.perf_counter() - t_combo0:.1f}s 소요)', flush=True)

    # CSV를 다시 읽어 요약(중간에 실패해도 이미 쓰인 행으로 부분 요약이 가능하도록 파일에서 재구성)
    with open(results_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['n_workers'] = int(row['n_workers'])
            row['rep'] = int(row['rep'])
            row['completed_ok'] = row['completed_ok'] == 'True'
            row['wall_time_s'] = float(row['wall_time_s']) if row['wall_time_s'] != 'nan' else float('nan')
            row['eval_throughput'] = float(row['eval_throughput']) if row['eval_throughput'] != 'nan' else float('nan')
            row['n_waves'] = int(row['n_waves'])
            row['idle_slots'] = int(row['idle_slots'])
            rows_by_worker.setdefault(row['n_workers'], []).append(row)

    _print_summary(rows_by_worker, gbest_log)
    print(f'\n결과 CSV: {results_csv}', flush=True)


if __name__ == '__main__':
    main()
