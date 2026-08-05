"""파이프라인 단일 진입점 (CLAUDE.md 부록A "구현 순서" 4번째 단계: lower_lp+test_lp ->
benefits -> evaluate -> **main** -> postprocess).

역할: lshade_core.LSHADEBench에 evaluate를 목적함수로 꽂고, 단일 기수(n_ess)에 대해 독립실행(run)
루프를 돌리며 결과를 CSV 로그로 남긴다. postprocess.py가 그 로그를 읽는다. main은
조립·실행·기록만 하며 최적화 로직 자체는 손대지 않는다(evaluate.py 등 미수정).

★ 실행 단위는 "기수 1개"다. CLAUDE.md 2절은 "1기부터 하나씩 늘리며 순편익이 꺾이는
지점을 탐색"한다고 규정하는데, 꺾이는 지점 판단은 사람이 n=1 결과를 보고 λ·입자 수를
조정한 뒤 n=2로 넘어가는 식이라 기본은 한 번에 한 기수만 돈다. --n-ess에 콤마 리스트를
명시하면 여러 기수를 순차 실행할 수 있지만 기본값은 항상 단일 정수 1이다.

L-SHADE 탐색 차원은 3n(b,S,E)이다. ★ C.6-3(LinDistFlow 편입) 이후 evaluate.py 자신이 3n
시그니처로 바뀌어(Q는 이제 하위 LP의 시변 변수라 상위 최적화기에 q_ratio가 없다 - 부록C.4-(3))
main의 _expand_to_4n 어댑터는 제거했다. main은 lshade_core가 반환하는 3n 배열을
evaluate.evaluate_particle에 그대로 넘긴다.

★ runs.csv의 편익 분해·페널티 컬럼(j_net/b_energy/.../penalty_v/penalty_line)은 --diagnose
여부와 무관하게 항상 기록한다. full 프로파일(30 run)에서 --diagnose가 stdout에 30줄을
흘려보내고 끝나면 postprocess.py가 사후 분석할 자료가 남지 않기 때문이다. run당 gbest
해에 대해 evaluate_particle(return_detail=True, collect_diagnostics=True)를 1회 더 호출한다.
--diagnose는 이제 "CSV에 이미 남는 값을 화면에서도 보기"로 역할이 축소됐다 -
재평가를 또 하지 않고 CSV 기록에 쓴 detail을 그대로 출력한다.

★ 확정 사항 3) postprocess.py 자동 실행. runs.csv/generations.csv는 기존처럼 run마다
즉시 append하지만(변경 없음), postprocess_n{k}/postprocess_units_n{k}/schedule_n{k}
3종은 그 기수의 n_runs개 run이 전부 끝난 직후(후처리 지표는 최종 결과가 있어야 계산
가능하므로) 자동으로 만든다. 다섯 파일(runs/generations/postprocess/postprocess_units/
schedule)이 같은 타임스탬프를 공유해 파일명만 보고도 한 세트임을 알 수 있다.
--no-postprocess로 끌 수 있다(postprocess.py를 손으로 나중에 따로 돌리고 싶을 때).
# 실행 기록 (실행 후 채울 것 - scripts/probe_noise.py, bench_workers.py와 같은 규약)
#   실행 일시:
#   머신 사양: 물리 16코어 / 논리 24 프로세서 / 메모리 15.7GB (CLAUDE.md 7절 참고값)
#   결과 요약:
# ------------------------------------------------------------------
"""
import os
import csv
import json
import math
import time
import argparse
import datetime
import multiprocessing as mp

import numpy as np

import params as PM
import benefits
import evaluate
import loss_coeffs
import lshade_core
import postprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

DEFAULT_BASE_SEED = 42
DEFAULT_N_WORKERS = 16   # scripts/bench_workers.py 실측 확정값 (아래 "확정 사항 1)" 참조)

# L-SHADE 초기 개체군은 12*n_dims로 결정한다. 프로파일은 평가예산 계수와 run 수만 정의한다.
# eval_budget = budget_coef * N_init (N_init=12*n_dims): dev 30배, full 100배.
PROFILES = {
    'dev':  dict(budget_coef=30,  n_runs=3),
    'full': dict(budget_coef=100, n_runs=20),
}

GEN_FIELDS = [
    'n_ess', 'run', 'gen', 'batch_size', 'eval_cumulative',
    'gbest_f', 'mean_f', 'std_f', 'worst_f',
    'elapsed_s', 'coef_cache_hit_rate_cumulative',
]
RUN_FIELDS = [
    'n_ess', 'run', 'seed', 'gbest_f', 'x_json', 'wall_time_s',
    'n_diverged_total', 'n_negative_bdefer', 'min_bdefer',
    # ★ gbest 해의 편익 분해·페널티 (--diagnose 무관 상시 기록, 아래 "확정 사항 2)" 참조).
    'j_net', 'b_energy', 'b_defer', 'b_arb', 'b_loss_avg',
    'b_arb_total', 'b_loss_total', 'cost',
    'capex_power_annual', 'capex_energy_annual', 'opex',
    'v_violation', 'i_violation', 'penalty_v', 'penalty_line', 'decomposition_ok',
    'recompute_jnet_delta', 'recompute_convergence_ratio', 'recompute_max_p_shift',
    'cross_effect_pct_avg_max', 'cross_effect_pct_peak_max',
    # 컬럼명은 명령안 호환을 위해 mw를 유지하지만 evaluate의 cross_effect 단위는 MWh다.
    'cross_effect_mw_avg_total', 'cross_effect_mw_peak_total',
    'cross_effect_won_avg_annual',
    'coef_cache_hit_rate', 'coef_runpp_total', 'coef_unique_bs_count',
    'coef_measure_wall_s',
]

# gbest_f(L-SHADE가 이미 추적 중인 값) vs -j_net+penalty_v+penalty_line(gbest 해를 사후
# 재평가해 얻은 값)의 검산 허용오차. n=2 우수해(Q가 PCS 한계까지 binding,
# poly_binding_frac~0.53)에서 QP solver 재현 오차 10.07원이 관측되어 15원으로 둔다.
# 물리량이 아닌 solver 재현 잔차의 상한이며, 수백~수천 원 규모의 버그 탐지력은 유지한다.
RUN_CONSISTENCY_ATOL_WON = 15.0


# ============================================================
# 확정 사항 1) 워커/입자 수 (scripts/bench_workers.py 실측, 2026-07-20, 데스크탑, 유효 3회 실행)
# ============================================================
# - 조합 간 차이가 실행 간 재현 노이즈에 묻혀 유의한 차이 없음(세 실행의 최우수 조합이
#   각각 16/32, 16/64, 12/24로 전부 달랐다).
# - 워커 10~16 구간에서 평가당 시간 0.09~0.10초로 일정 -> 병렬 효율 저하 없음.
# - 입자 수를 워커 수의 정수배로 맞출 필요 없음(평가시간 표준편차가 평균의 2배 수준이라
#   chunksize=1의 동적 로드밸런싱이 라운드 정합 효과를 상쇄).
# - 메모리 워커당 약 288MB(16워커=약 4.6GB). 20워커 이상에서 시스템 정지 2회 발생 ->
#   20·24워커 사용 금지(scripts/bench_workers.py 그리드에서도 이미 제외됨).
# -> N_WORKERS=16, N_PARTICLES=32 기본값(CLI로 덮어쓰기 가능).


# ============================================================
# 확정 사항 2) runs.csv 편익 분해·페널티 컬럼 상시 기록
# ============================================================
# - 배경: --diagnose는 stdout 출력뿐이라 full 프로파일(30 run)에서 화면을 스쳐 지나가고
#   사라진다. postprocess.py가 편익 분해(8절)를 다루려면 파일에 남아 있어야 한다.
# - run당 gbest 해 1개에 대해서만 evaluate_particle(return_detail=True)를 추가 호출한다
#   (전 입자 기록은 CLAUDE.md 7-A절 "중간 저장" 원칙과 충돌 - 비용 대비 이득 없음).
#   명령 G 이후 gbest 상세 경로는 X0/X2·전압 진단을 추가로 수행하지만 run당 1회뿐이며,
#   L-SHADE의 전 입자 평가에는 collect_diagnostics=False라 이 추가 비용이 없다.
# - 검산(-j_net+penalty_v+penalty_line == gbest_f)은 경고만 하고 죽이지 않는다. 본실험
#   30 run 중간에 assert로 죽으면 그 기수 전체를 다시 돌려야 해 손해가 훨씬 크다.


# ============================================================
# 확정 사항 3) run_for_n_ess 종료 직후 postprocess.py 자동 실행
# ============================================================
# - postprocess.process_group()이 evaluate._BASE_P/_BASE_FLOW(모듈 전역)를 직접 참조하므로
#   메인 프로세스 자신의 evaluate.init_worker()가 먼저 호출돼 있어야 한다. Pool 워커들은
#   별도 프로세스라 이 초기화와 무관하다(각자 자기 프로세스에서 이미 따로 초기화됨) - 메인
#   프로세스 쪽 초기화는 그것과 별개로 한 번 더 필요하다.
# - 여러 기수를 한 번에 돌려도(--n-ess "1,2,3") 1회만 하면 되므로 모듈 전역 플래그로 가드.

_postprocess_ready = False


def _ensure_postprocess_ready():
    global _postprocess_ready
    if not _postprocess_ready:
        print('[postprocess] evaluate.init_worker() 호출(메인 프로세스, 1회) - '
              '기저 조류계산 120회 캐싱.', flush=True)
        evaluate.init_worker()
        _postprocess_ready = True


# ============================================================
# 환경 확인 (설정은 하지 않는다 - CLAUDE.md 7절: 환경 문제는 환경 설정으로 해결)
# ============================================================

def _check_env():
    val = os.environ.get('MKL_THREADING_LAYER')
    print(f'MKL_THREADING_LAYER = {val!r}', flush=True)
    if val != 'SEQUENTIAL':
        print("경고: MKL_THREADING_LAYER가 'SEQUENTIAL'이 아닙니다. 워커 스레드 풀 생성 "
              "시점에 무증상 종료가 발생할 수 있습니다(CLAUDE.md 7절). "
              "`conda env config vars set MKL_THREADING_LAYER=SEQUENTIAL -n ess`로 설정하세요.",
              flush=True)


# ============================================================
# 3n 경계 구성 (evaluate.py도 3n 시그니처 - C.6-3 이후 어댑터 불필요)
# ============================================================

def _build_bounds(n_ess):
    """(b,S,E) x n_ess -> (3*n_ess, 2) 경계 배열. q_ratio는 여기 없음(PSO 변수가 아님)."""
    unit_bounds = [PM.B_BOUNDS, PM.S_BOUNDS, PM.E_BOUNDS]
    return np.array(unit_bounds * n_ess, dtype=float)


def _build_int_dims(n_ess):
    """b가 있는 위치([0, 3, 6, ...]) - 한 기당 (b,S,E) 3칸 중 첫 칸."""
    return [3 * i for i in range(n_ess)]


# ============================================================
# 워커에서 실행되는 평가 함수 (Pool.map 대상 - 반드시 모듈 최상위, 피클 가능해야 함)
# ============================================================

def _eval_for_optimizer(x3n):
    """evaluate.evaluate_particle을 return_detail=True로 호출해 (fitness, diverged, b_defer)
    3튜플만 돌려준다. 전체 dict를 그대로 IPC로 돌려보내면 p_slack_ess/loss_ess/unit_p/unit_q
    등 이 스크립트가 쓰지 않는 배열까지 매 평가마다 pickling되므로, 필요한 스칼라만 추린다.
    return_detail=True를 쓰는 이유: CLAUDE.md 8절이 요구하는 "음의 B_defer 발생 카운터"와
    발산 누적은 탐색 전체에 걸쳐 기록해야 사후 복원이 가능하다(최적해 하나만으로는 탐색 중
    발생 여부를 알 수 없음) - fitness 스칼라만 반환하는 경로로는 이 정보에 접근할 수 없다.
    """
    detail = evaluate.evaluate_particle(x3n, return_detail=True)
    # 각 Pool 워커의 프로세스 로컬 캐시 통계를 평가 결과에 편승시킨다.
    # 실패해도 fitness 반환은 유지하여 진단 계측이 L-SHADE를 중단시키지 않게 한다.
    try:
        cache_info = dict(pid=os.getpid(), **loss_coeffs.cache_info())
    except Exception as exc:
        cache_info = dict(pid=os.getpid(), error=repr(exc))
    if detail.get('diverged'):
        return detail['fitness'], True, None, cache_info
    return detail['fitness'], False, detail['b_defer'], cache_info


# ============================================================
# L-SHADE가 기대하는 objective(X)->fitness 배열 인터페이스 + 부수 진단 수집
# ============================================================

class RunObjective:
    """L-SHADE는 objective(X: (batch_size, n_dims)) -> (batch_size,)만 요구한다.
    이 클래스는 그 계약을 만족하면서, Pool로 분배한 평가 결과에서 발산·음의 B_defer를
    부수적으로 누적한다(evaluate.py는 건드리지 않음 - return_detail=True가 이미 주는
    필드를 읽기만 한다). LPSR로 호출별 배치 크기가 달라지므로 호출별 최솟값을 함께
    기록하고, 실행 루프에서 그 누적 최솟값을 gbest_f로 만든다.
    """

    _CACHE_COUNTERS = (
        'hits', 'misses', 'runpp_attempts', 'measure_wall_s', 'size',
    )

    def __init__(self, pool, n_ess, expected_workers):
        self.pool = pool
        self.n_ess = n_ess
        self.expected_workers = int(expected_workers)
        self.n_diverged_total = 0
        self.n_negative_bdefer = 0
        self.min_bdefer = float('inf')
        self.gen_rows = []  # objective 호출별 배치 통계(best_f_batch는 CSV 기록 전에 제거)
        self._gen_idx = 0
        self._eval_cumulative = 0
        self._cache_first_by_pid = {}
        self._cache_last_by_pid = {}
        self._cache_collection_failed = False

    def _record_cache_info(self, info):
        """PID별 이 run의 첫/마지막 누적 카운터를 보관한다.

        첫 스냅샷은 해당 워커의 첫 평가 *후* 값이므로 차분에서 워커당 첫
        평가 1건이 빠진다. 워커 강제 살포 없이 실제 평가 워커를 빠짐없이
        관측하기 위한 의도된 상한 n_workers건의 근사다.
        """
        try:
            if not info or info.get('error'):
                raise ValueError(info.get('error', 'cache_info missing') if info else 'cache_info missing')
            pid = int(info['pid'])
            snap = {name: float(info[name]) for name in self._CACHE_COUNTERS}
            self._cache_first_by_pid.setdefault(pid, snap)
            self._cache_last_by_pid[pid] = snap
        except Exception as exc:
            if not self._cache_collection_failed:
                print(f'  경고: 워커 캐시 통계 수집 실패({exc!r}); 이 run의 캐시 통계는 NaN 처리합니다.',
                      flush=True)
            self._cache_collection_failed = True

    def _cache_deltas(self):
        if self._cache_collection_failed or not self._cache_last_by_pid:
            return None
        totals = {name: 0.0 for name in self._CACHE_COUNTERS}
        for pid, last in self._cache_last_by_pid.items():
            first = self._cache_first_by_pid[pid]
            for name in self._CACHE_COUNTERS:
                totals[name] += max(0.0, last[name] - first[name])
        return totals

    def _cache_hit_rate_cumulative(self):
        totals = self._cache_deltas()
        if totals is None:
            return float('nan')
        requests = totals['hits'] + totals['misses']
        return totals['hits'] / requests if requests else float('nan')

    def cache_run_fields(self):
        """전 평가 워커의 run 증분 합을 runs.csv 필드로 변환한다.

        coef_unique_bs_count는 정확한 전역 (b,S) 고유수가 아니다. 워커별
        cache size 증가량의 합으로, 워커 간 중복을 포함하는 상한 프록시다.
        coef_measure_wall_s도 병렬 워커의 측정시간 합(CPU-time 성격)이다.
        """
        totals = self._cache_deltas()
        if totals is None:
            return dict(
                coef_cache_hit_rate=float('nan'),
                coef_runpp_total=float('nan'),
                coef_unique_bs_count=float('nan'),
                coef_measure_wall_s=float('nan'),
            )
        observed = len(self._cache_last_by_pid)
        if observed < self.expected_workers:
            print(f'  경고: 캐시 통계에서 {observed}/{self.expected_workers}개 워커 PID만 관측됨. '
                  '실제로 평가를 반환한 워커만 집계합니다.', flush=True)
        requests = totals['hits'] + totals['misses']
        return dict(
            coef_cache_hit_rate=totals['hits'] / requests if requests else float('nan'),
            coef_runpp_total=int(round(totals['runpp_attempts'])),
            coef_unique_bs_count=int(round(totals['size'])),
            coef_measure_wall_s=totals['measure_wall_s'],
        )

    def __call__(self, X):
        X = np.asarray(X, dtype=float)
        particles_3n = [np.asarray(row, dtype=float) for row in X]

        t0 = time.perf_counter()
        results = self.pool.map(_eval_for_optimizer, particles_3n, chunksize=1)
        elapsed_s = time.perf_counter() - t0

        fitness = np.empty(len(results), dtype=float)
        for i, (fit, diverged, b_defer, cache_info) in enumerate(results):
            fitness[i] = fit
            self._record_cache_info(cache_info)
            if diverged:
                self.n_diverged_total += 1
            else:
                if b_defer < 0:
                    self.n_negative_bdefer += 1
                if b_defer < self.min_bdefer:
                    self.min_bdefer = b_defer

        batch_size = len(fitness)
        self._eval_cumulative += batch_size
        self.gen_rows.append(dict(
            gen=self._gen_idx,
            batch_size=batch_size,
            eval_cumulative=self._eval_cumulative,
            best_f_batch=float(np.min(fitness)),
            mean_f=float(np.mean(fitness)),
            std_f=float(np.std(fitness)),
            worst_f=float(np.max(fitness)),
            elapsed_s=elapsed_s,
            coef_cache_hit_rate_cumulative=self._cache_hit_rate_cumulative(),
        ))
        self._gen_idx += 1

        return fitness


# ============================================================
# CSV: 즉시 append + flush + fsync (bench_workers.py와 같은 방어 패턴)
# ============================================================

def _init_csv(path, fields):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        f.flush()
        os.fsync(f.fileno())


def _append_csv_rows(path, fields, rows):
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        for row in rows:
            writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


# ============================================================
# gbest 해의 편익 분해·페널티 (runs.csv에 상시 기록 - 확정 사항 2) 참조)
# ============================================================

def _evaluate_gbest_detail(gbest_x_3n):
    """run 종료 후 gbest 해 1개에 대해서만 evaluate_particle(return_detail=True)를 호출한다
    (전 입자 기록은 안 함). 부록B의 LAMBDA_V/LAMBDA_LINE/PENALTY_DIVERGE가 전부 "잠정값,
    첫 실행 로그 보고 조정"이므로 이 값을 볼 수단이 CSV에도 항상 남아 있어야 한다."""
    x = np.asarray(gbest_x_3n, dtype=float)
    detail = evaluate.evaluate_particle(
        x, return_detail=True, collect_diagnostics=True
    )
    return dict(detail)


def _annual_cost_components(gbest_x_3n):
    """gbest의 총 S·E로 runs.csv용 연간 비용 3분해를 계산한다."""
    units = np.asarray(gbest_x_3n, dtype=float).reshape(-1, 3)
    s_total = float(np.sum(units[:, 1]))
    e_total = float(np.sum(units[:, 2]))
    return dict(
        capex_power_annual=PM.CRF_20 * PM.C_KW_CAPEX_PER_MVA * s_total,
        capex_energy_annual=PM.CRF_20 * PM.C_KWH_CAPEX_PER_MWH * e_total,
        opex=benefits.opex(s_total, e_total),
    )


def _gbest_detail_to_run_fields(detail, gbest_f):
    """detail(evaluate_particle 반환) -> runs.csv에 넣을 편익 분해·페널티 필드 dict.
    gbest 해가 발산 상태면(이론상 거의 불가능 - 발산 페널티 1e15가 L-SHADE gbest가 되려면
    전 입자가 발산해야 함) 빈 문자열로 채운다(CSV 스키마는 유지하되 값 없음을 표시).
    검산(-j_net+penalty_v+penalty_line == gbest_f)은 경고만 하고 죽이지 않는다(확정 사항 2) -
    본실험 도중 assert로 죽으면 손해가 훨씬 크다)."""
    if detail.get('diverged'):
        print('  경고: gbest 해가 발산 상태 - 편익 분해 불가(전 입자 발산 등 비정상 상황).',
              flush=True)
        return dict(
            j_net='', b_energy='', b_defer='', b_arb='', b_loss_avg='',
            b_arb_total='', b_loss_total='', cost='',
            capex_power_annual='', capex_energy_annual='', opex='',
            v_violation='', i_violation='', penalty_v='', penalty_line='', decomposition_ok='',
            recompute_jnet_delta='', recompute_convergence_ratio='',
            recompute_max_p_shift='', cross_effect_pct_avg_max='',
            cross_effect_pct_peak_max='', cross_effect_mw_avg_total='',
            cross_effect_mw_peak_total='', cross_effect_won_avg_annual='',
            coef_cache_hit_rate='',
            coef_runpp_total='', coef_unique_bs_count='', coef_measure_wall_s='',
        )

    penalty_v = PM.LAMBDA_V * detail['v_violation']
    penalty_line = PM.LAMBDA_LINE * detail['i_violation']

    recomputed_fitness = -detail['j_net'] + penalty_v + penalty_line
    diff = abs(recomputed_fitness - gbest_f)
    if diff > RUN_CONSISTENCY_ATOL_WON:
        print(f'  경고: gbest_f 재현 불일치 - gbest_f(L-SHADE)={gbest_f:.6e} vs '
              f'재계산(-j_net+penalty_v+penalty_line)={recomputed_fitness:.6e} '
              f'(차이={diff:.4e}원 > 허용오차 {RUN_CONSISTENCY_ATOL_WON}원). '
              'evaluate.py 로직 변경이나 편익 분해 코드의 버그를 의심할 것.', flush=True)

    qp_diag = detail.get('qp_diag_loss_vs_ac') or {}
    required_scenarios = set(PM.ALL_DAYS)
    if not required_scenarios.issubset(qp_diag):
        cross_fields = dict(
            cross_effect_pct_avg_max='', cross_effect_pct_peak_max='',
            cross_effect_mw_avg_total='', cross_effect_mw_peak_total='',
            cross_effect_won_avg_annual='',
        )
    else:
        cross_fields = dict(
            cross_effect_pct_avg_max=max(
                abs(float(qp_diag[s]['cross_effect_pct']))
                for s in PM.AVG_DAYS
            ),
            cross_effect_pct_peak_max=max(
                abs(float(qp_diag[s]['cross_effect_pct']))
                for s in PM.PEAK_DAYS
            ),
            # cross_effect = sum_t(loss_MW)*DT_HOURS, 따라서 실제 단위는 MWh.
            cross_effect_mw_avg_total=sum(
                float(qp_diag[s]['cross_effect']) for s in PM.AVG_DAYS
            ),
            cross_effect_mw_peak_total=sum(
                float(qp_diag[s]['cross_effect']) for s in PM.PEAK_DAYS
            ),
            cross_effect_won_avg_annual=sum(
                float(
                    np.sum(
                        (
                            np.asarray(qp_diag[s]['ac_actual_loss_mw'], dtype=float)
                            - np.asarray(qp_diag[s]['qp_diag_loss_mw'], dtype=float)
                        )
                        * np.asarray(PM.SMP_PER_MWH[s], dtype=float)
                    )
                    * PM.DT_HOURS
                    * PM.N_WEEKDAYS[s]
                )
                for s in PM.AVG_DAYS
            ),
        )

    return dict(
        j_net=detail['j_net'], b_energy=detail['b_energy'], b_defer=detail['b_defer'],
        b_arb=detail['b_arb'], b_loss_avg=detail['b_loss'],
        b_arb_total=detail['b_arb_total'], b_loss_total=detail['b_loss_total'],
        cost=detail['cost'],
        v_violation=detail['v_violation'], i_violation=detail['i_violation'],
        penalty_v=penalty_v, penalty_line=penalty_line,
        decomposition_ok=detail['decomposition_ok'],
        recompute_jnet_delta=detail.get('recompute_jnet_delta', float('nan')),
        recompute_convergence_ratio=detail.get(
            'recompute_convergence_ratio', float('nan')
        ),
        recompute_max_p_shift=detail.get('recompute_max_p_shift', float('nan')),
        **cross_fields,
        coef_cache_hit_rate=detail.get('coef_cache_hit_rate', float('nan')),
        coef_runpp_total=detail.get('coef_runpp_total', float('nan')),
        coef_unique_bs_count=detail.get('coef_unique_bs_count', float('nan')),
        coef_measure_wall_s=detail.get('coef_measure_wall_s', float('nan')),
    )


def _print_diagnosis(n_ess, run_idx, detail, run_fields):
    """--diagnose 전용 화면 출력. CSV에 이미 기록한 값(run_fields)을 그대로 보여줄 뿐 재평가는
    하지 않는다(확정 사항 2) - CSV가 사본의 원본이고 화면은 그 사본을 보는 창일 뿐)."""
    print(f'  [진단] n_ess={n_ess} run={run_idx} gbest 해 편익 분해 (runs.csv와 동일):', flush=True)
    if detail.get('diverged'):
        print('    gbest 해가 발산 상태입니다 - 편익 분해 불가(정상 평가가 아님).', flush=True)
        return

    print(f"    j_net={run_fields['j_net']:.4e}  b_energy={run_fields['b_energy']:.4e}  "
          f"b_defer={run_fields['b_defer']:.4e}  cost={run_fields['cost']:.4e}", flush=True)
    print(f"    v_violation={run_fields['v_violation']:.6f} pu  -> "
          f"penalty_v={run_fields['penalty_v']:.4e} 원", flush=True)
    print(f"    i_violation={run_fields['i_violation']:.6f}     -> "
          f"penalty_line={run_fields['penalty_line']:.4e} 원", flush=True)


# ============================================================
# 기수 1개 실행 (독립실행 n_runs회, Pool은 이 기수 안에서 재사용)
# ============================================================

def run_for_n_ess(n_ess, profile, n_workers, base_seed, diagnose, run_postprocess=True):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    gens_path = os.path.join(RESULTS_DIR, f'generations_n{n_ess}_{timestamp}.csv')
    runs_path = os.path.join(RESULTS_DIR, f'runs_n{n_ess}_{timestamp}.csv')
    _init_csv(gens_path, GEN_FIELDS)
    _init_csv(runs_path, RUN_FIELDS)

    bounds = _build_bounds(n_ess)
    int_dims = _build_int_dims(n_ess)
    n_dims = len(bounds)
    n_init = 12 * n_dims
    eval_budget = profile['budget_coef'] * n_init

    print(f'\n[n_ess={n_ess}] 시작: N_init={n_init} eval_budget={eval_budget} '
          f'n_runs={profile["n_runs"]} n_workers={n_workers}', flush=True)
    print(f'[n_ess={n_ess}] 로그: {gens_path}\n              {runs_path}', flush=True)

    best_gbest_f = math.inf

    # 기수마다 독립된 가상의 부모 시퀀스(spawn_key=(n_ess,)) - 어떤 다른 n_ess 값들과
    # 같은 호출에서 함께 실행되든, spawn 호출 순서와 무관하게 항상 같은 run별 시드가
    # 나오게 한다(base_seed+run_idx 같은 저품질 시드 대신 SeedSequence 계보를 명시적으로
    # 분리한다). L-SHADE의 default_rng(seed)는 SeedSequence 객체를 직접 받는다.
    base_seq = np.random.SeedSequence(entropy=base_seed, spawn_key=(n_ess,))
    run_seed_seqs = base_seq.spawn(profile['n_runs'])

    # ★ Pool은 run마다 새로 만들지 않고 이 기수 안에서 재사용한다(init_worker의 기저
    # 조류계산 120회 캐싱 비용을 run마다 다시 치르지 않기 위함). 기수가 바뀌면(차원이
    # 바뀌므로) run_for_n_ess가 다시 호출되며 새 Pool이 만들어진다.
    pool = mp.Pool(n_workers, initializer=evaluate.init_worker)
    try:
        for run_idx in range(profile['n_runs']):
            objective = RunObjective(pool, n_ess, n_workers)
            optimizer = lshade_core.LSHADEBench(
                objective=objective,
                bounds=bounds,
                n_particles=n_init,
                n_iters=eval_budget,
                int_dims=int_dims,
                seed=run_seed_seqs[run_idx],
                eval_budget=eval_budget,
            )

            t_run0 = time.perf_counter()
            result = optimizer.optimize()
            wall_time_s = time.perf_counter() - t_run0

            gen_rows_full = []
            running_best = float('inf')
            for source_row in objective.gen_rows:
                row = dict(source_row)
                running_best = min(running_best, row.pop('best_f_batch'))
                gen_rows_full.append(dict(
                    n_ess=n_ess,
                    run=run_idx,
                    gbest_f=running_best,
                    **row,
                ))
            _append_csv_rows(gens_path, GEN_FIELDS, gen_rows_full)

            # seed 컬럼에는 이 run에 실제로 쓰인 자식 SeedSequence를 대표하는 정수 지문을
            # 남긴다(generate_state로 상태 1워드). base_seed 자체는 --seed 인자/stdout에
            # 남으므로, "이 run이 어떤 난수열로 돌았는가"를 구분하는 데는 이 지문이 직접적이다.
            # 재현하려면 --seed 값과 n_ess, run_idx를 함께 알아야 한다.
            seed_repr = int(run_seed_seqs[run_idx].generate_state(1)[0])
            min_bdefer = objective.min_bdefer if math.isfinite(objective.min_bdefer) else ''
            gbest_f = float(result['f'])

            # ★ --diagnose 여부와 무관하게 항상 계산·기록 (확정 사항 2)).
            gbest_detail = _evaluate_gbest_detail(result['x'])
            benefit_fields = _gbest_detail_to_run_fields(gbest_detail, gbest_f)
            benefit_fields.update(_annual_cost_components(result['x']))
            # G의 메인 프로세스 gbest 1회 통계 대신, L-SHADE를 수행한 전 워커의
            # PID별 누적 카운터 baseline 차분 합으로 runs.csv를 채운다.
            benefit_fields.update(objective.cache_run_fields())

            run_row = dict(
                n_ess=n_ess, run=run_idx, seed=seed_repr,
                gbest_f=gbest_f,
                x_json=json.dumps(result['x'].tolist()),
                wall_time_s=wall_time_s,
                n_diverged_total=objective.n_diverged_total,
                n_negative_bdefer=objective.n_negative_bdefer,
                min_bdefer=min_bdefer,
                **benefit_fields,
            )
            _append_csv_rows(runs_path, RUN_FIELDS, [run_row])

            best_gbest_f = min(best_gbest_f, gbest_f)

            print(f'[n_ess={n_ess}] run {run_idx + 1}/{profile["n_runs"]} 완료 '
                  f'({wall_time_s:.1f}s)  gbest_f={result["f"]:.6e}  '
                  f'diverged={objective.n_diverged_total}  '
                  f'neg_bdefer={objective.n_negative_bdefer}', flush=True)

            if diagnose:
                _print_diagnosis(n_ess, run_idx, gbest_detail, benefit_fields)
    finally:
        pool.close()
        pool.join()

    print(f'[n_ess={n_ess}] 완료. {profile["n_runs"]}개 run 중 최우수 gbest_f={best_gbest_f:.6e}',
          flush=True)

    if run_postprocess:
        _ensure_postprocess_ready()
        print(f'\n[n_ess={n_ess}] postprocess.py 자동 실행 ({profile["n_runs"]}개 run 전부 완료 후 '
              f'1회) - {runs_path}', flush=True)
        try:
            pp_result = postprocess.process_group(runs_path, RESULTS_DIR, timestamp)
            if pp_result is not None:
                postprocess.write_group_outputs(pp_result, RESULTS_DIR, timestamp)
            else:
                print(f'[n_ess={n_ess}] postprocess 생략(그룹 없음) - runs.csv를 확인할 것.',
                      flush=True)
        except Exception as e:
            # runs.csv/generations.csv는 이미 저장되어 안전하다 - postprocess 실패로 L-SHADE
            # 결과가 유실되면 안 되므로 예외를 삼키고 경고만 남긴다(재실행은 명령 하나로 가능).
            print(f'[n_ess={n_ess}] ★ 경고: postprocess 자동 실행 중 오류 발생({e!r}). '
                  f'runs.csv/generations.csv는 이미 저장되어 있으니 수동으로 재실행할 것: '
                  f'python postprocess.py {runs_path}', flush=True)


# ============================================================
# CLI
# ============================================================

def _parse_n_ess_list(s):
    values = [int(v.strip()) for v in s.split(',') if v.strip()]
    assert values, f'--n-ess 파싱 결과가 비었음: {s!r}'
    for v in values:
        assert v >= 1, f'n_ess는 1 이상이어야 함: {v}'
    return values


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description='ESS 배치·용량 산정 L-SHADE 파이프라인 진입점 (CLAUDE.md 부록A).',
        epilog='예: python main.py --n-ess 1 --profile dev',
    )
    parser.add_argument('--n-ess', type=str, default='1',
                         help='기수(정수) 또는 콤마구분 리스트(예: "1" 또는 "1,2,3"). 기본값 1.')
    parser.add_argument('--profile', choices=list(PROFILES.keys()), default='dev',
                         help='dev(기본, 첫 실행용) 또는 full(본실험, 명시 지정 필요).')
    parser.add_argument('--n-workers', type=int, default=None,
                         help=f'기본 {DEFAULT_N_WORKERS}(실측 확정값, "확정 사항 1)" 참조).')
    parser.add_argument('--budget-coef', type=int, default=None,
                         help='평가예산 계수(eval_budget=budget_coef*12*n_dims)를 덮어씀.')
    parser.add_argument('--n-runs', type=int, default=None, help='프로파일 기본값을 덮어씀.')
    parser.add_argument('--seed', type=int, default=DEFAULT_BASE_SEED,
                         help='BASE_SEED. run별 시드는 SeedSequence(entropy=BASE_SEED, '
                              'spawn_key=(n_ess,)).spawn(n_runs)에서 파생하며, runs.csv에는 '
                              '각 자식 시드의 정수 지문을 기록한다.')
    parser.add_argument('--diagnose', action='store_true',
                         help='각 run 종료 후 gbest 해의 페널티 성분을 사후 재계산해 출력.')
    parser.add_argument('--no-postprocess', action='store_true',
                         help='기수 완료 직후 postprocess.py 자동 실행(기본 켜짐)을 끈다 - '
                              '나중에 손으로 python postprocess.py <runs.csv>를 돌리고 싶을 때.')
    return parser


def _resolve_profile(args):
    profile = dict(PROFILES[args.profile])
    if args.budget_coef is not None:
        profile['budget_coef'] = args.budget_coef
    if args.n_runs is not None:
        profile['n_runs'] = args.n_runs
    n_workers = args.n_workers if args.n_workers is not None else DEFAULT_N_WORKERS
    return profile, n_workers


def main():
    args = _build_arg_parser().parse_args()
    _check_env()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    n_ess_list = _parse_n_ess_list(args.n_ess)
    profile, n_workers = _resolve_profile(args)

    print(f'프로파일: {args.profile} -> {profile}', flush=True)
    print(f'워커 수: {n_workers}', flush=True)
    print(f'기수 목록: {n_ess_list}', flush=True)

    for n_ess in n_ess_list:
        run_for_n_ess(n_ess, profile, n_workers, args.seed, args.diagnose,
                      run_postprocess=not args.no_postprocess)


if __name__ == '__main__':
    main()
