"""확인 전용 스크립트 (evaluate.py/test_evaluate.py 작성 후 실측).

test_evaluate.py의 S=0∧E=0 케이스에서 p_slack_ess가 base_flow와 ~1e-9 MW 차이가 나는
현상의 원인을 실험 1~4로 가른다. 두 가설:
  가설1: sgen(P=Q=0) 객체의 '존재' 자체가 노드 전력방정식/야코비안을 바꾼다.
  가설2: 비교 대상 두 계산의 초기값 이력(warm start 경로)이 달라 뉴턴법이
         tolerance 안의 서로 다른 점에서 멈춘다.
기존 파일은 건드리지 않는다(build_net, params, lower_lp 등은 import만 한다).
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandapower as pp

from build_net import build_net
import params as PM
from lower_lp import solve_avg


def section(title):
    print('\n' + '=' * 70)
    print(title)
    print('=' * 70)


# ------------------------------------------------------------------
# 실험 1: sgen 존재 여부의 순수 효과 (init='flat'으로 초기값 이력을 통제)
# ------------------------------------------------------------------
def experiment1():
    section("실험 1: sgen 존재 여부의 순수 효과 (init='flat'으로 통제)")

    net_a = build_net()
    pp.runpp(net_a, numba=True, init='flat')
    p_slack_a = net_a.res_ext_grid.p_mw.sum()
    iters_a = net_a._ppc['iterations']

    net_b = build_net()
    pp.create_sgen(net_b, bus=10, p_mw=0.0, q_mvar=0.0, name='ESS_zero')
    pp.runpp(net_b, numba=True, init='flat')
    p_slack_b = net_b.res_ext_grid.p_mw.sum()
    iters_b = net_b._ppc['iterations']

    diff = p_slack_b - p_slack_a
    print(f"(a) sgen 없음       : P_slack = {p_slack_a:.15f} MW, iterations={iters_a}")
    print(f"(b) sgen(P=Q=0) 있음: P_slack = {p_slack_b:.15f} MW, iterations={iters_b}")
    print(f'차이 (b-a): {diff:.6e} MW')
    print(f'res_ext_grid (b):\n{net_b.res_ext_grid}')
    return diff


# ------------------------------------------------------------------
# 실험 2: warm start 이력의 효과 (sgen 없음으로 통제)
# ------------------------------------------------------------------
def experiment2(tolerance_mva=1e-8, verbose=False):
    """target 부하상태(scale=1.0)를 (a) flat 직접계산 (b) 다른 부하상태를 먼저 계산해
    이력을 남긴 뒤 init='results'로 복귀 계산, 두 경로로 구해 비교한다."""
    net_a = build_net()
    pp.runpp(net_a, numba=True, init='flat', tolerance_mva=tolerance_mva)
    p_slack_a = net_a.res_ext_grid.p_mw.sum()

    net_b = build_net()
    base_p = net_b.load['p_mw'].copy()
    base_q = net_b.load['q_mvar'].copy()

    other_scale = 0.6
    net_b.load['p_mw'] = base_p * other_scale
    net_b.load['q_mvar'] = base_q * other_scale
    pp.runpp(net_b, numba=True, init='flat', tolerance_mva=tolerance_mva)

    net_b.load['p_mw'] = base_p
    net_b.load['q_mvar'] = base_q
    pp.runpp(net_b, numba=True, init='results', tolerance_mva=tolerance_mva)
    p_slack_b = net_b.res_ext_grid.p_mw.sum()

    diff = p_slack_b - p_slack_a
    if verbose:
        print(f"(a) flat 직접계산   : P_slack = {p_slack_a:.15f} MW")
        print(f"(b) warm-start 복귀 : P_slack = {p_slack_b:.15f} MW")
        print(f'차이 (b-a): {diff:.6e} MW')
    return p_slack_a, p_slack_b, diff


# ------------------------------------------------------------------
# 실험 3: 잡음이 S>0 실제 ESS 주입 상태에서도 공통인지
# ------------------------------------------------------------------
def experiment3():
    section('실험 3: S=1.0MVA/E=4MWh 실제 스케줄에서 flat독립계산 vs warm-start체인')

    smp = np.asarray(PM.SMP['summer'])
    P_net, _ = solve_avg(1.0, 4.0, smp, assert_physics=False)

    bus = 17
    T = PM.TIME_STEPS

    # 경로 (a): 매 시각 독립적으로 flat 초기값에서 계산 (이력 없음)
    p_slack_flat = np.zeros(T)
    for t in range(T):
        net = build_net()
        base_p = net.load['p_mw'].copy()
        base_q = net.load['q_mvar'].copy()
        scale = PM.LOAD['summer'][t]
        net.load['p_mw'] = base_p * scale
        net.load['q_mvar'] = base_q * scale
        pp.create_sgen(net, bus=bus, p_mw=float(P_net[t]), q_mvar=0.0, name='ESS')
        pp.runpp(net, numba=True, init='flat')
        p_slack_flat[t] = net.res_ext_grid.p_mw.sum()

    # 경로 (b): warm start 체인 (evaluate.py와 동일한 패턴)
    net_ws = build_net()
    base_p = net_ws.load['p_mw'].copy()
    base_q = net_ws.load['q_mvar'].copy()
    pp.create_sgen(net_ws, bus=bus, p_mw=0.0, q_mvar=0.0, name='ESS')
    p_slack_ws = np.zeros(T)
    for t in range(T):
        scale = PM.LOAD['summer'][t]
        net_ws.load['p_mw'] = base_p * scale
        net_ws.load['q_mvar'] = base_q * scale
        net_ws.sgen.at[0, 'p_mw'] = float(P_net[t])
        net_ws.sgen.at[0, 'q_mvar'] = 0.0
        pp.runpp(net_ws, numba=True, init='results')
        p_slack_ws[t] = net_ws.res_ext_grid.p_mw.sum()

    diff = p_slack_ws - p_slack_flat
    max_abs = float(np.max(np.abs(diff)))
    argmax_t = int(np.argmax(np.abs(diff)))
    print(f'summer 24h: flat독립계산 vs warm-start체인 최대 절대차 = {max_abs:.6e} MW (t={argmax_t})')
    print(f'시각별 차이(MW): {np.array2string(diff, precision=3, floatmode="fixed")}')
    return max_abs


# ------------------------------------------------------------------
# 실험 4: tolerance_mva 의존성
# ------------------------------------------------------------------
def experiment4():
    section('실험 4: tolerance_mva 의존성 (실험2를 tolerance 바꿔가며 반복)')

    tolerances = [1e-8, 1e-10, 1e-12]
    results = []
    for tol in tolerances:
        p_a, p_b, diff = experiment2(tolerance_mva=tol)
        results.append((tol, p_a, p_b, diff))
        print(f'tolerance_mva={tol:.0e}: P_slack_a={p_a:.15f}, P_slack_b={p_b:.15f}, 차이={diff:.6e} MW')

    section('실험 4 부속: tolerance별 runpp 시간 비용 (warm start, 100회 평균)')
    net = build_net()
    pp.runpp(net, numba=True, init='flat')  # 워밍업(numba jit)
    timing = {}
    for tol in tolerances:
        n_reps = 100
        t0 = time.perf_counter()
        for _ in range(n_reps):
            pp.runpp(net, numba=True, init='results', tolerance_mva=tol)
        elapsed = (time.perf_counter() - t0) / n_reps
        timing[tol] = elapsed
        print(f'tolerance_mva={tol:.0e}: 평균 {elapsed * 1000:.4f} ms/회')

    return results, timing


if __name__ == '__main__':
    diff1 = experiment1()

    section("실험 2: warm start 이력의 효과 (sgen 없음, tolerance 기본값 1e-8)")
    p_a2, p_b2, diff2 = experiment2(verbose=True)

    diff3 = experiment3()

    results4, timing4 = experiment4()

    section('요약')
    print(f'실험1 (sgen 유무, flat으로 이력 통제)   diff = {diff1:.6e} MW')
    print(f'실험2 (warm-start 이력, sgen 없음)       diff = {diff2:.6e} MW')
    print(f'실험3 (S=1,E=4 실사용, summer 24h 최대)  diff = {diff3:.6e} MW')
    print('실험4 (tolerance별 실험2 차이 및 시간비용):')
    for tol, p_a, p_b, diff in results4:
        print(f'  tolerance_mva={tol:.0e}: diff={diff:.6e} MW, {timing4[tol] * 1000:.4f} ms/회')

    # (c) B_energy에 미치는 최대 금전 영향 추정: 실험3 최대 P_slack 차이(MW) 기준,
    # 가장 비싼 SMP(winter_peak 최대값)로 환산 + N_WEEKDAYS 최대치를 곱해 상한을 잡는다.
    max_smp_per_mwh = max(np.max(v) for v in PM.SMP_PER_MWH.values())
    max_nwd = max(PM.N_WEEKDAYS.values())
    worst_case_krw_per_hour = diff3 * max_smp_per_mwh * PM.DT_HOURS
    worst_case_krw_annual = worst_case_krw_per_hour * max_nwd * PM.TIME_STEPS
    print(f'\n(c) 참고: 실험3 최대 MW차이({diff3:.3e}) 단독 1시간 환산 = {worst_case_krw_per_hour:.6f} 원')
    print(f'    24시간x최대가중치({max_nwd:.1f}) 극단 상한 = {worst_case_krw_annual:.4f} 원 (실제 B_energy는 여러 항 상쇄로 이보다 작음)')
