"""워커 수 실측 벤치마크 (확인 전용, 본 파이프라인 미포함 - scripts/check_balance.py,
scripts/probe_noise.py와 같은 성격). main.py 착수 전 미정사항 #1 "워커 수"를 실측으로
정한다(CLAUDE.md 부록A 말미 "★ main.py 착수 전 미정 사항" 참조).

배경:
  - MKL_THREADING_LAYER=SEQUENTIAL이므로 워커당 1스레드다(conda env config vars로 이미
    설정됨 - CLAUDE.md 7절). 따라서 워커 수 = 실제 병렬도이고, oversubscription은
    스레드 경합이 아니라 프로세스 스케줄링 경합으로만 나타난다.
  - 입자 평가 시간은 균등하지 않다: LP 반복횟수·조류계산 뉴턴 반복횟수가 입자마다 다르고,
    발산 케이스(PENALTY_DIVERGE)는 재시도(flat) 경로까지 타 최대 반복까지 돈다. 따라서
    "입자 수가 워커 수의 정수배일 때 최적"이라는 직관이 자명하지 않다 - 불균등이 크면
    오히려 입자가 워커보다 많고 chunksize=1일 때 pool.map의 동적 로드밸런싱이 유리할 수
    있다. 이 스크립트는 그 두 가설(정수배 유리 vs 동적 로드밸런싱 유리)을 실측으로 가른다.
  - ★ 24워커·20워커 조합은 그리드에서 제외했다. 첫 실행에서 5번째 조합 부근에서 화면이
    얼어붙고 python.exe 프로세스가 전부 무증상 종료되는 문제가 발생했다(stdout/stderr
    아무것도 안 남음) - CLAUDE.md 7절에 기록된 "MKL 기본 스레딩이 워커 스레드 풀을 생성하는
    시점에 무증상 종료" 증상과 일치하고, 메모리 부족(워커당 ~325MB 관측, 24워커면
    ~7.8GB - 다른 프로세스와 합쳐 15.7GB 한계에 근접)도 배제할 수 없다. 원인을 확정하기
    전에 재발을 막는 것이 우선이라 물리 16코어를 넘는 워커 수 자체를 그리드에서 뺐다.
    **"24워커는 이 머신에서 실행 불가"라는 것 자체가 유효한 실측 결과다** - 하이퍼스레딩이
    BLAS 연산에 이득이 거의 없다는 CLAUDE.md 부록A의 판단과도 정합적이다.

실행: `python scripts/bench_workers.py` (프로젝트 루트에서, 또는 어디서든 - sys.path를
스스로 보정한다). Windows는 spawn 방식이라 아래 오케스트레이션 전체가 반드시
`if __name__ == '__main__':` 안에 있어야 한다(가드가 없으면 자식 프로세스가 재귀적으로
스크립트를 재실행하며 Pool을 또 만든다).

evaluate.py는 수정하지 않는다 - 이 스크립트는 순수 소비자다. evaluate.evaluate_particle의
현재 시그니처(4n차원 x, 4번째 성분이 q_ratio)에 맞춰 particle을 만든다: 뼈대 단계에서
PSO 탐색공간은 3n(b,S,E)뿐이므로(q_ratio는 Q_RATIO_BOUNDS=(0,0)로 고정될 예정 - CLAUDE.md
2절), 이 스크립트도 3열(b,S,E)로 무작위 생성한 뒤 호출 직전에 q_ratio=0.0을 붙여 4n
벡터로 만든다.

방어적 설계 (무증상 종료 재발 방지, 이번 개정의 핵심):
  1. 그리드에서 20/24워커 제거(위 배경 참조).
  2. 조합을 워커 수 오름차순으로 실행 - 큰 조합에서 죽어도 작은 조합 결과는 이미 남는다.
  3. CSV는 조합이 끝날 때마다 즉시 append+flush+fsync(마지막 일괄저장 아님). 결과 파일은
     실행마다 <호스트명>_<타임스탬프>로 새로 만들어 기존 파일을 덮어쓰지 않는다.
  4. 조합 시작 전 psutil로 가용 메모리를 확인해, n_workers*400MB보다 적으면 그 조합을
     건너뛰고 다음으로 진행한다(psutil 없으면 점검 생략 - 스크립트가 죽으면 안 됨).
  5. MKL_THREADING_LAYER != 'SEQUENTIAL'이면 무증상 종료의 유력 원인이므로 아예 시작하지
     않고 즉시 종료한다(경고가 아니라 중단).
  6. Pool은 with문의 암묵적 terminate()에 맡기지 않고 매 조합마다 close()+join()으로
     명시적으로 회수한 뒤 gc.collect()+2초 대기로 OS가 메모리를 정리할 시간을 준다.
  7. 모든 진행 출력은 print(..., flush=True) - 버퍼링 때문에 죽는 순간의 마지막 메시지가
     유실되면 어디서 죽었는지 알 수 없다.

evaluate.py 수정 금지. .gitignore에서 결과 CSV를 제외하지 않는다 - 벤치마크 결과는
머신 종속이라 재실행으로 복원되지 않으므로 scripts/probe_noise.py 실측값을 CLAUDE.md에
남긴 것과 같은 이유로 커밋 대상이다.

# ------------------------------------------------------------------
# 실행 기록 (실행 후 채울 것 - scripts/probe_noise.py와 같은 규약)
#   실행 일시:
#   머신 사양: 물리 16코어 / 논리 24 프로세서 / 메모리 15.7GB (CLAUDE.md 7절 참고값)
#   결과 요약:
#   비고: 20/24워커 조합은 위 "배경" 사유로 그리드에서 제외됨(그 자체가 결과).
# ------------------------------------------------------------------
"""
import os
import sys
import csv
import gc
import math
import time
import socket
import datetime
import threading
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import params as PM
import evaluate

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ============================================================
# 설정
# ============================================================
N_ESS = 1          # 뼈대 단계: q_ratio는 PSO 변수가 아님 -> 실질 탐색차원 3n, n=1 고정
SEED = 42

# 20/24워커 제외(위 모듈 docstring "배경" 참조 - 무증상 종료 재발 방지 + 메모리 여유 부족).
# 워커 수 오름차순으로 나열(작은 조합부터 실행하기 위함 - main()에서 재차 방어적으로 정렬함).
GRID = [
    # 워커 축 (입자/워커 비=2 고정, 워커수만 스윕)
    (10, 20), (12, 24), (14, 28), (16, 32),
    # 정수배 라인 (워커수=16 고정)
    (16, 48), (16, 64),
    # 비정수배 라인 (워커수=16 고정, 나머지가 남는 입자수)
    (16, 30), (16, 36), (16, 40),
]

N_WARMUP_GENS = 1     # numba JIT + init_worker 캐싱 비용을 측정에서 제외하기 위한 워밍업
N_MEASURED_GENS = 2   # 최솟값 사용(아래 참조) - 3->2로 축소해 전체 실행시간 단축
MEM_POLL_INTERVAL_S = 0.1
MIN_FREE_MB_PER_WORKER = 400  # 조합 시작 전 메모리 사전점검 기준

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_N_PARTICLES = max(n_particles for _, n_particles in GRID)

CSV_FIELDS = [
    'n_workers', 'n_particles', 'rounds',
    'gen_time_best_s', 'gen_time_min_s', 'gen_time_max_s',
    'time_per_particle_s', 'eval_time_std_s', 'n_diverged', 'peak_rss_mb',
    'status', 'error',
]


def _make_results_path():
    """실행마다 새 파일(호스트명+타임스탬프) - 기존 결과를 덮어쓰지 않는다."""
    hostname = socket.gethostname()
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(SCRIPT_DIR, f'bench_workers_{hostname}_{ts}.csv')


# ============================================================
# 환경 확인 - MKL 스레딩 문제는 무증상 종료의 유력 원인이라 경고가 아니라 중단으로 격상
# (설정은 하지 않는다 - CLAUDE.md 7절: 환경 문제는 환경 설정으로 해결하고 코드는 안 건드림)
# ============================================================

def _check_env_or_exit():
    val = os.environ.get('MKL_THREADING_LAYER')
    print(f'MKL_THREADING_LAYER = {val!r}', flush=True)
    if val != 'SEQUENTIAL':
        print(
            "\n[중단] MKL_THREADING_LAYER가 'SEQUENTIAL'이 아닙니다.\n"
            "  원인: MKL 기본 스레딩(Intel OpenMP)이 워커 프로세스의 스레드 풀을 생성하는\n"
            "  시점에 프로세스가 무증상 종료할 수 있습니다(CLAUDE.md 7절 실측 확인 증상).\n"
            "  해결: 아래 명령으로 conda env에 설정한 뒤(코드에서 os.environ으로 설정하지\n"
            "  않음 - 환경 문제는 환경 설정으로 해결), 새 셸에서 다시 실행하세요.\n\n"
            "    conda env config vars set MKL_THREADING_LAYER=SEQUENTIAL -n ess\n",
            flush=True,
        )
        sys.exit(1)


# ============================================================
# 입자 생성 (params.py 경계에서 무작위 추출, 3열: b,S,E)
# ============================================================

def _generate_particles(n, seed=SEED):
    """3n 형태(b,S,E) 무작위 추출. 고정된 좋은 해를 반복평가하면 발산 케이스가 안 걸려
    시간분산을 과소평가하므로 반드시 탐색공간 전체에서 균일추출한다.
    b는 pso_core.PSO._decode와 동일 규약(반올림)으로 정수화한다."""
    rng = np.random.default_rng(seed)
    b_lo, b_hi = PM.B_BOUNDS
    s_lo, s_hi = PM.S_BOUNDS
    e_lo, e_hi = PM.E_BOUNDS
    b = np.round(rng.uniform(b_lo, b_hi, size=n))
    S = rng.uniform(s_lo, s_hi, size=n)
    E = rng.uniform(e_lo, e_hi, size=n)
    return np.stack([b, S, E], axis=1)


_ALL_PARTICLES = None  # 지연 생성 (MAX_N_PARTICLES 기준 1회만) - 모든 조합이 앞부분을 공유


def _particles_for(n_particles):
    """모든 그리드 조합이 동일 시드에서 시작해 같은 입자집합의 앞부분을 공유하도록,
    가장 큰 n_particles로 한 번만 생성해 슬라이스한다(조합 간 비교 오염 방지)."""
    global _ALL_PARTICLES
    if _ALL_PARTICLES is None:
        _ALL_PARTICLES = _generate_particles(MAX_N_PARTICLES)
    assert n_particles <= len(_ALL_PARTICLES)
    return _ALL_PARTICLES[:n_particles]


# ============================================================
# 평가 래퍼 (evaluate.py는 건드리지 않고 이 스크립트 안에서만 감쌈)
# ============================================================

def _timed_eval(row):
    """row = (b,S,E) 3성분. evaluate.evaluate_particle의 실제 시그니처(4n차원, 4번째 성분
    q_ratio)에 맞춰 q_ratio=0.0을 붙여 호출한다(N_ESS=1이므로 벡터 길이 4).
    (fitness, elapsed) 튜플을 반환해 개별 평가시간을 측정할 수 있게 한다."""
    b, S, E = row
    x = np.array([b, S, E, 0.0])
    t0 = time.perf_counter()
    fitness = evaluate.evaluate_particle(x)
    elapsed = time.perf_counter() - t0
    return fitness, elapsed


# ============================================================
# 메모리 사전점검 (조합 시작 전) + 메모리 모니터 (조합 실행 중 배경 폴링)
# ============================================================

def _enough_memory_for(n_workers):
    """가용 메모리가 n_workers*MIN_FREE_MB_PER_WORKER보다 적으면 False.
    psutil이 없으면 점검을 생략하고 True(스크립트가 죽으면 안 됨)."""
    if not _HAS_PSUTIL:
        return True, None, None
    available_mb = psutil.virtual_memory().available / (1024 ** 2)
    required_mb = n_workers * MIN_FREE_MB_PER_WORKER
    return available_mb >= required_mb, available_mb, required_mb


class _MemoryMonitor:
    """프로세스 트리(부모+자식) RSS 합의 최댓값을 배경 스레드로 폴링."""

    def __init__(self, interval=MEM_POLL_INTERVAL_S):
        self.interval = interval
        self.peak_rss_mb = float('nan')
        self._stop_event = None
        self._thread = None

    def start(self):
        if not _HAS_PSUTIL:
            return
        self._stop_event = threading.Event()
        self.peak_rss_mb = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        proc = psutil.Process(os.getpid())
        while not self._stop_event.is_set():
            try:
                total = proc.memory_info().rss
                for child in proc.children(recursive=True):
                    try:
                        total += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                self.peak_rss_mb = max(self.peak_rss_mb, total / (1024 ** 2))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            self._stop_event.wait(self.interval)

    def stop(self):
        if self._thread is None:
            return self.peak_rss_mb
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        return self.peak_rss_mb


# ============================================================
# 조합 1개 실행
# ============================================================

def _run_combo(n_workers, n_particles):
    particles = _particles_for(n_particles)
    rounds = math.ceil(n_particles / n_workers)

    monitor = _MemoryMonitor()
    gen_times = []
    all_elapsed = []
    n_diverged = None

    monitor.start()
    pool = mp.Pool(n_workers, initializer=evaluate.init_worker)
    try:
        # 워밍업: numba JIT 컴파일 + init_worker 캐싱 비용을 측정 밖으로 덜어냄
        for _ in range(N_WARMUP_GENS):
            pool.map(_timed_eval, particles, chunksize=1)

        for gen_idx in range(N_MEASURED_GENS):
            t0 = time.perf_counter()
            results = pool.map(_timed_eval, particles, chunksize=1)
            gen_times.append(time.perf_counter() - t0)
            all_elapsed.extend(elapsed for _fitness, elapsed in results)
            if gen_idx == 0:
                # particle 집합이 세대마다 동일(고정 평가, PSO 이동 없음)하므로 결정론적 -
                # 첫 측정세대의 발산 수만 대표값으로 남긴다.
                n_diverged = sum(1 for fitness, _e in results if fitness >= PM.PENALTY_DIVERGE)
    finally:
        # with문의 암묵적 __exit__은 terminate()라 워커를 강제로 죽인다 - 여기서는
        # close()(신규 작업 차단) + join()(정상 종료 대기)으로 명시적으로 회수한다.
        pool.close()
        pool.join()
        peak_rss_mb = monitor.stop()

    gen_times = np.asarray(gen_times)
    all_elapsed = np.asarray(all_elapsed)
    # 배경 프로세스 간섭(OS 스케줄링, 다른 앱)은 세대 시간을 늘리는 방향으로만 작용하므로
    # 최솟값이 워커 수/입자 수 조합 자체의 성능을 가장 깨끗하게 반영하는 추정치다(중앙값보다).
    gen_time_best = float(np.min(gen_times))

    return dict(
        n_workers=n_workers,
        n_particles=n_particles,
        rounds=rounds,
        gen_time_best_s=gen_time_best,
        gen_time_min_s=float(np.min(gen_times)),
        gen_time_max_s=float(np.max(gen_times)),
        time_per_particle_s=gen_time_best / n_particles,
        eval_time_std_s=float(np.std(all_elapsed)),
        n_diverged=int(n_diverged),
        peak_rss_mb=peak_rss_mb,
        status='ok',
        error='',
    )


def _empty_row(n_workers, n_particles, status, error):
    return dict(
        n_workers=n_workers,
        n_particles=n_particles,
        rounds=math.ceil(n_particles / n_workers),
        gen_time_best_s=float('nan'),
        gen_time_min_s=float('nan'),
        gen_time_max_s=float('nan'),
        time_per_particle_s=float('nan'),
        eval_time_std_s=float('nan'),
        n_diverged=-1,
        peak_rss_mb=float('nan'),
        status=status,
        error=error,
    )


# ============================================================
# CSV: 조합마다 즉시 append + flush + fsync (일괄저장 금지 - 중간에 죽어도 앞부분 보존)
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


# ============================================================
# 결과 표 출력
# ============================================================

def _print_summary_table(rows):
    ok_rows = [r for r in rows if r['status'] == 'ok']
    ok_rows.sort(key=lambda r: r['time_per_particle_s'])

    print('\n' + '=' * 100, flush=True)
    print('요약 (time_per_particle_s 오름차순)', flush=True)
    print('=' * 100, flush=True)
    header = (f"{'workers':>7} {'particles':>9} {'rounds':>6} {'time/particle(s)':>17} "
              f"{'gen_best(s)':>11} {'std(s)':>8} {'diverged':>8} {'peak_rss(MB)':>12}")
    print(header, flush=True)
    print('-' * len(header), flush=True)
    for r in ok_rows:
        peak_str = f"{r['peak_rss_mb']:.0f}" if not math.isnan(r['peak_rss_mb']) else 'N/A'
        print(f"{r['n_workers']:>7} {r['n_particles']:>9} {r['rounds']:>6} "
              f"{r['time_per_particle_s']:>17.4f} {r['gen_time_best_s']:>11.3f} "
              f"{r['eval_time_std_s']:>8.4f} {r['n_diverged']:>8} {peak_str:>12}", flush=True)

    other_rows = [r for r in rows if r['status'] != 'ok']
    if other_rows:
        print('\n실패/건너뜀:', flush=True)
        for r in other_rows:
            print(f"  workers={r['n_workers']} particles={r['n_particles']} "
                  f"status={r['status']}: {r['error']}", flush=True)


# ============================================================
# 메인
# ============================================================

def main():
    _check_env_or_exit()  # SEQUENTIAL 아니면 여기서 sys.exit(1)

    print(f'psutil 사용 가능: {_HAS_PSUTIL}'
          + ('' if _HAS_PSUTIL else ' (peak_rss_mb=NaN, 메모리 사전점검 생략)'), flush=True)

    # 워커 수 오름차순 방어적 재정렬(요구사항 #2) - GRID를 나중에 편집해도 순서가 깨지지 않게.
    ordered_grid = sorted(GRID, key=lambda t: t[0])

    results_csv = _make_results_path()
    _init_csv(results_csv)
    print(f'결과 CSV: {results_csv} (조합마다 즉시 append)', flush=True)
    print(f'그리드: {len(ordered_grid)}개 조합, 워밍업 {N_WARMUP_GENS}세대 + 측정 {N_MEASURED_GENS}세대씩 '
          '(세대 시간은 최솟값 사용)', flush=True)
    print(f'MAX_N_PARTICLES={MAX_N_PARTICLES} 기준 입자 1회 생성(시드={SEED}), 조합마다 앞부분 슬라이스',
          flush=True)

    rows = []
    for i, (n_workers, n_particles) in enumerate(ordered_grid, 1):
        print(f'\n[{i}/{len(ordered_grid)}] workers={n_workers} particles={n_particles} 시작', flush=True)

        ok, available_mb, required_mb = _enough_memory_for(n_workers)
        if not ok:
            reason = (f'available={available_mb:.0f}MB < required={required_mb:.0f}MB '
                       f'({n_workers}workers x {MIN_FREE_MB_PER_WORKER}MB)')
            row = _empty_row(n_workers, n_particles, 'skipped: insufficient memory', reason)
            print(f'  -> SKIPPED (메모리 부족): {reason}', flush=True)
            rows.append(row)
            _append_csv_row(results_csv, row)
            continue

        t_combo0 = time.perf_counter()
        try:
            row = _run_combo(n_workers, n_particles)
        except Exception as exc:  # 한 조합이 실패해도 나머지 그리드는 계속 진행
            row = _empty_row(n_workers, n_particles, 'failed', f'{type(exc).__name__}: {exc}')
        combo_elapsed = time.perf_counter() - t_combo0

        if row['status'] == 'ok':
            peak_str = f"{row['peak_rss_mb']:.0f}MB" if not math.isnan(row['peak_rss_mb']) else 'N/A'
            print(f"  -> OK ({combo_elapsed:.1f}s)  time/particle={row['time_per_particle_s']:.4f}s  "
                  f"gen_best={row['gen_time_best_s']:.3f}s  std={row['eval_time_std_s']:.4f}s  "
                  f"diverged={row['n_diverged']}  peak_rss={peak_str}", flush=True)
        else:
            print(f"  -> FAILED ({combo_elapsed:.1f}s): {row['error']}", flush=True)

        rows.append(row)
        _append_csv_row(results_csv, row)

        # 다음 조합 전 OS가 메모리를 회수할 시간을 준다 (요구사항 #6).
        gc.collect()
        time.sleep(2)

    _print_summary_table(rows)
    print(f'\n결과 CSV: {results_csv}', flush=True)


if __name__ == '__main__':
    main()
