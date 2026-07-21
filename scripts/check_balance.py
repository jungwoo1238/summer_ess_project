"""확인 전용 스크립트 (benefits.py 작성 전 실측).

CLAUDE.md 3절의 항등식 P_slack = SigmaLoad + Loss - P_ESS 가 실제로 성립하는지,
성립하지 않는다면 어떤 항이 빠졌는지, atol을 얼마로 잡아야 하는지 확인한다.
기존 파일은 건드리지 않는다 (build_net, params 등은 import만 한다).
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pandapower as pp

from build_net import build_net
from params import LOAD, VALIDATION


def section(title):
    print('\n' + '=' * 70)
    print(title)
    print('=' * 70)


def main():
    # ------------------------------------------------------------------
    # 1. 구조 점검
    # ------------------------------------------------------------------
    section('1. 구조 점검')
    net = build_net()

    g_us = net.line['g_us_per_km'].unique()
    print(f'net.line.g_us_per_km 고유값: {g_us}')

    print(f'net.shunt 비어있는가: {net.shunt.empty}')
    if not net.shunt.empty:
        print(net.shunt[['bus', 'p_mw', 'q_mvar', 'in_service']])

    c_nf = net.line['c_nf_per_km'].unique()
    print(f'net.line.c_nf_per_km 고유값 (참고용): {c_nf}')

    n_in_service = int(net.line['in_service'].sum())
    print(f'in_service=True 선로 개수: {n_in_service} (32개여야 함)')

    # ------------------------------------------------------------------
    # 2. 기저 조류계산
    # ------------------------------------------------------------------
    section('2. 기저 조류계산 (0번 검증 재현 + 수지 확인)')
    net = build_net()
    pp.runpp(net, numba=True)

    p_slack = net.res_ext_grid.p_mw.sum()
    sum_load = net.res_load.p_mw.sum()
    loss = net.res_line.pl_mw.sum()
    residual = p_slack - (sum_load + loss)

    print(f'P_slack       = {p_slack:.10f} MW')
    print(f'SigmaLoad     = {sum_load:.10f} MW')
    print(f'Loss(pl_mw)   = {loss:.10f} MW')
    print(f'잔차 r = P_slack - (SigmaLoad + Loss) = {residual:.10e} MW')

    loss_alt = net.res_line.p_from_mw.sum() + net.res_line.p_to_mw.sum()
    print(f'\n손실 정의 대조:')
    print(f'  p_from_mw.sum() + p_to_mw.sum() = {loss_alt:.10f} MW')
    print(f'  pl_mw.sum()                     = {loss:.10f} MW')
    print(f'  차이                             = {loss_alt - loss:.10e} MW')

    total_loss_kw = loss * 1000
    vmin = net.res_bus.vm_pu.min()
    vmin_bus = net.res_bus.vm_pu.idxmin()
    line0_current_a = net.res_line.at[0, 'i_ka'] * 1000
    slack_mw = net.res_ext_grid.at[0, 'p_mw']

    # 검증값은 params.VALIDATION을 그대로 참조한다(리터럴로 다시 박지 않음 - build_net.py
    # __main__과 동일 원칙). 슬랙 전압이 바뀌면 이 스크립트도 값을 다시 타이핑할 필요 없이
    # 자동으로 새 기준값을 따라간다.
    print(f'\n0번 검증값 재현:')
    print(f"  총손실: {total_loss_kw:.2f} kW (검증값 {VALIDATION['loss_kw_scaled']} kW)")
    print(f"  Vmin: {vmin:.4f} pu, bus {vmin_bus} "
          f"(검증값 {VALIDATION['vmin_pu_scaled']} pu, bus {VALIDATION['vmin_bus']})")
    print(f"  주간선 전류(line 0): {line0_current_a:.2f} A (검증값 {VALIDATION['line0_current_a_scaled']} A)")
    print(f'  슬랙 유입: {slack_mw:.4f} MW (참고용 - 부하 프로파일·슬랙 전압에 따라 달라져 '
          '단일 검증값으로 쓰지 않음, CLAUDE.md 1절)')

    # ------------------------------------------------------------------
    # 3. ESS 주입 상태에서 재확인
    # ------------------------------------------------------------------
    section('3. ESS 주입 크기별 잔차 (bus 17)')
    ess_bus = 17
    p_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    rows = []
    for p_mw in p_values:
        net = build_net()
        pp.create_sgen(net, bus=ess_bus, p_mw=p_mw, q_mvar=0.0, name='ESS_test')
        pp.runpp(net, numba=True)

        p_slack_i = net.res_ext_grid.p_mw.sum()
        sum_load_i = net.res_load.p_mw.sum()
        loss_i = net.res_line.pl_mw.sum()
        p_sgen_i = net.res_sgen.p_mw.sum()
        # P_slack = SigmaLoad + Loss - P_sgen  (sgen: 발전기준 양수. 방전 +, 충전 -)
        residual_i = p_slack_i - (sum_load_i + loss_i - p_sgen_i)
        rows.append(dict(
            p_sgen_mw=p_mw,
            p_slack_mw=p_slack_i,
            sum_load_mw=sum_load_i,
            loss_mw=loss_i,
            residual_mw=residual_i,
        ))

    df3 = pd.DataFrame(rows)
    with pd.option_context('display.float_format', lambda x: f'{x:.10f}'):
        print(df3.to_string(index=False))

    residual_abs = df3['residual_mw'].abs()
    print(f'\n잔차 절대값 range: min={residual_abs.min():.3e}, max={residual_abs.max():.3e} MW')
    # 크기 의존 여부: |P_sgen| 대비 상관/추세로 판정
    corr = np.corrcoef(df3['p_sgen_mw'].abs(), residual_abs)[0, 1] if residual_abs.std() > 0 else float('nan')
    print(f'|P_sgen| vs |잔차| 상관계수: {corr}')

    # ------------------------------------------------------------------
    # 4. summer_peak 24시간 순차 조류계산
    # ------------------------------------------------------------------
    section('4. summer_peak 24시간 시각별 잔차 (부하 프로파일 적용)')
    net = build_net()
    base_p = net.load['p_mw'].copy()
    base_q = net.load['q_mvar'].copy()
    profile = LOAD['summer_peak']

    residuals = []
    for t in range(24):
        scale = profile[t]
        net.load['p_mw'] = base_p * scale
        net.load['q_mvar'] = base_q * scale
        pp.runpp(net, numba=True, init='results')

        p_slack_t = net.res_ext_grid.p_mw.sum()
        sum_load_t = net.res_load.p_mw.sum()
        loss_t = net.res_line.pl_mw.sum()
        residual_t = p_slack_t - (sum_load_t + loss_t)
        residuals.append(dict(t=t, scale=scale, p_slack=p_slack_t, sum_load=sum_load_t,
                               loss=loss_t, residual=residual_t))

    df4 = pd.DataFrame(residuals)
    with pd.option_context('display.float_format', lambda x: f'{x:.10f}'):
        print(df4.to_string(index=False))

    max_abs = df4['residual'].abs().max()
    mean_abs = df4['residual'].abs().mean()
    signs = np.sign(df4['residual'])
    sign_consistent = (signs == signs.iloc[0]).all() or (df4['residual'].abs() < 1e-12).all()

    print(f'\n최대 절대잔차: {max_abs:.6e} MW')
    print(f'평균 절대잔차: {mean_abs:.6e} MW')
    print(f'부호 일관성: {"일관됨" if sign_consistent else "일관되지 않음"} (부호들: {sorted(set(signs))})')

    # ------------------------------------------------------------------
    # 최종 요약
    # ------------------------------------------------------------------
    section('요약')
    print(f'(a) g_us_per_km 고유값: {g_us}, net.shunt 비어있음: {net.shunt.empty}')
    print(f'(b) 3번 결과의 |P_sgen| vs |잔차| 상관계수: {corr}')
    print(f'    잔차 range (3번): [{residual_abs.min():.3e}, {residual_abs.max():.3e}] MW')
    print(f'(c) 4번 최대 절대잔차: {max_abs:.6e} MW, 평균 절대잔차: {mean_abs:.6e} MW')


if __name__ == '__main__':
    main()
