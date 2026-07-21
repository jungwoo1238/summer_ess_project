import pandapower.networks as nw

import params as PM
from params import VN_KV, K_SCALE, LINE_RATINGS_A, LINE_RATING_DEFAULT_A


def build_net(slack_vm_pu=None):
    """IEEE 33-bus(case33bw) -> 22.9kV/10MVA 스케일 계통 준비. CLAUDE.md 1절.

    slack_vm_pu: None이면 params.SLACK_VM_PU(현행 1.02) 사용. 구 값(1.0) 재현이 필요하면
    명시적으로 build_net(slack_vm_pu=1.0)으로 호출한다(CLAUDE.md 1절 "변경이 슬랙 전압에만
    기인함"의 회귀 확인 - test_evaluate.py::test_base_violation_at_legacy_slack 참조).
    """
    if slack_vm_pu is None:
        slack_vm_pu = PM.SLACK_VM_PU

    net = nw.case33bw()

    # 전압 스케일
    net.bus['vn_kv'] = VN_KV
    net.ext_grid['vm_pu'] = slack_vm_pu

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

    net = build_net()  # slack_vm_pu 생략 -> PM.SLACK_VM_PU(현행) 기본값 적용
    pp.runpp(net, numba=True)

    total_loss_kw = net.res_line.pl_mw.sum() * 1000
    vmin = net.res_bus.vm_pu.min()
    vmin_bus = net.res_bus.vm_pu.idxmin()
    line0_current_a = net.res_line.at[0, 'i_ka'] * 1000
    slack_mw = net.res_ext_grid.at[0, 'p_mw']

    # 검증값은 params.VALIDATION을 그대로 참조한다 - 리터럴로 다시 박으면 값이 두 곳에
    # 흩어져 나중에 슬랙 전압을 또 바꿀 때 여기만 업데이트 누락되는 문제가 재발한다.
    print(f'슬랙 전압(vm_pu) = {PM.SLACK_VM_PU}')
    print(f"총손실: {total_loss_kw:.2f} kW (검증값 {PM.VALIDATION['loss_kw_scaled']} kW)")
    print(f"Vmin: {vmin:.4f} pu, bus {vmin_bus} "
          f"(검증값 {PM.VALIDATION['vmin_pu_scaled']} pu, bus {PM.VALIDATION['vmin_bus']})")
    print(f"주간선 전류(line 0): {line0_current_a:.2f} A "
          f"(검증값 {PM.VALIDATION['line0_current_a_scaled']} A)")
    print(f'슬랙 유입: {slack_mw:.4f} MW (참고용 - 부하 프로파일·슬랙 전압에 따라 달라져 '
          '단일 검증값으로 쓰지 않음, CLAUDE.md 1절)')
