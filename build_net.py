import pandapower.networks as nw

from params import VN_KV, K_SCALE, LINE_RATINGS_A, LINE_RATING_DEFAULT_A


def build_net():
    """IEEE 33-bus(case33bw) -> 22.9kV/10MVA 스케일 계통 준비. CLAUDE.md 1절."""
    net = nw.case33bw()

    # 전압 스케일
    net.bus['vn_kv'] = VN_KV
    net.ext_grid['vm_pu'] = 1.0

    # 부하 스케일 (유효·무효 동일 계수 K -> 역률 보존)
    net.load['p_mw'] *= K_SCALE
    net.load['q_mvar'] *= K_SCALE

    # 선로 정격 주입 (max_i_ka 더미(99999) 대체). tie 선로(in_service=False)는 제약 미대상이라 default만 적용.
    for idx, row in net.line.iterrows():
        pair = (row['from_bus'], row['to_bus'])
        rating_a = LINE_RATINGS_A.get(pair, LINE_RATING_DEFAULT_A)
        net.line.at[idx, 'max_i_ka'] = rating_a / 1000.0

    return net


if __name__ == '__main__':
    import pandapower as pp

    net = build_net()
    pp.runpp(net, numba=True)

    total_loss_kw = net.res_line.pl_mw.sum() * 1000
    vmin = net.res_bus.vm_pu.min()
    vmin_bus = net.res_bus.vm_pu.idxmin()
    line0_current_a = net.res_line.at[0, 'i_ka'] * 1000
    slack_mw = net.res_ext_grid.at[0, 'p_mw']

    print(f'총손실: {total_loss_kw:.2f} kW (검증값 310.06 kW)')
    print(f'Vmin: {vmin:.4f} pu, bus {vmin_bus} (검증값 0.9407 pu, bus 17)')
    print(f'주간선 전류(line 0): {line0_current_a:.2f} A (검증값 261.51 A)')
    print(f'슬랙 유입: {slack_mw:.4f} MW (검증값 8.8125 MW)')
