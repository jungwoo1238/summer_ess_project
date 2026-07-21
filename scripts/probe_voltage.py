"""전압 위반의 구조적 원인 확인 (확인 전용, 본 파이프라인 미포함 - scripts/check_balance.py,
probe_noise.py, bench_workers.py와 같은 성격).

배경: main.py dev 프로파일 진단(--diagnose)에서 v_violation=0.334 pu가 세 run 모두
소수 5자리까지 동일하게 관측됐다. 이는 탐색 실패가 아니라 "ESS로 제거 불가능한 구조적
잔여 위반"으로 추정된다. CLAUDE.md 1절의 기저 Vmin=0.9407 pu(<0.95, bus 17)가 원인
후보다. 이 스크립트는 그 추정을 실측으로 확인하고(측정1), 슬랙(변전소) 전압을 올려
(OLTC 조정에 해당) 해소 가능한지 진단하고(측정2), CLAUDE.md 1절 검증표가 어느 조건에서
재현되는지 명확히 하는 것(측정3)까지 세 갈래로 확인한다.

build_net.py / evaluate.py / params.py는 수정하지 않는다(순수 소비자). 전압 위반 공식은
evaluate.py가 이미 쓰는 것(CLAUDE.md 7절: Σ[max(0,V-V_MAX)+max(0,V_MIN-V)], ALL_DAYS x
24h x 전 버스)을 그대로 재사용한다 - 두 가지 방식으로:
  1) 실제 함수 재사용: evaluate.init_worker()가 내부적으로 돌리는 evaluate._compute_base_flow의
     합산치(evaluate._BASE_FLOW['v_violation'])를 "공식값"으로 가져와 교차검증 기준으로 쓴다.
     evaluate.py가 노출하는 값은 시나리오/버스/시각별로 이미 합산된 스칼라뿐이라, 이 스크립트가
     필요로 하는 분해(어느 버스·시각·시나리오가 위반에 기여하는지)는 얻을 수 없다.
  2) 상세분해가 필요한 부분은 evaluate._run_pf_with_retry(조류계산+재시도 로직 재사용,
     재구현 안 함)로 직접 시나리오/시각별 조류계산을 돌리되, 위반 공식 자체는
     params.V_MIN/V_MAX 상수를 그대로 써서 evaluate.py와 정의가 갈리지 않게 한다. 이렇게
     구한 합계를 1)의 공식값과 대조해(둘 다 독립적으로 새로 build한 net에서 시작하므로
     완전히 동일한 실행은 아니지만, scripts/probe_noise.py 실측대로 그 차이는 1e-8 MW
     스케일의 부동소수 잡음 수준이라 pu 단위 위반량 비교에서는 무시 가능하다) 재사용이
     제대로 됐는지 확인한다.

실행: `python scripts/probe_voltage.py`

# ------------------------------------------------------------------
# 실행 기록 (실행 후 채울 것 - scripts/probe_noise.py, bench_workers.py와 같은 규약)
#   실행 일시:
#   머신 사양: 물리 16코어 / 논리 24 프로세서 / 메모리 15.7GB (CLAUDE.md 7절 참고값)
#   결과 요약:
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
from build_net import build_net
import evaluate

# main.py --diagnose에서 관측된 값 (측정1 판정 기준점). 하드코딩이지만 이 스크립트의
# 유일한 목적이 "그 값이 왜 나왔는가"를 확인하는 것이므로 비교 기준으로 남겨둔다.
MAIN_DIAGNOSED_V_VIOLATION = 0.334

VM_PU_SWEEP = [1.00, 1.01, 1.02, 1.03, 1.04, 1.05]

# CLAUDE.md 1절 검증표 조건과 동일한 스냅샷(측정2에서 "총손실"/"line0 전류"를 이 지점
# 기준으로도 같이 보여주기 위함 - 5절 주석 "부하 정규화 1.0 = 여름최대일 t=18 = case33bw
# 기본부하" 참조). scale=1.0인 지점.
SNAPSHOT_SCENARIO = 'summer_peak'
SNAPSHOT_T = 18

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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


# ============================================================
# 공통: ALL_DAYS x 24h 조류계산 (evaluate._run_pf_with_retry 재사용)
# ============================================================

def _run_all_days_detailed(net, base_p, base_q):
    """evaluate.py의 조류계산 규약(init='results' -> 실패시 'flat' 재시도)을 그대로 쓴다
    (evaluate._run_pf_with_retry 재사용, 재구현 안 함). evaluate._compute_base_flow와
    달리 시나리오/시각/버스별 vm_pu 원자료를 전부 남겨 이 스크립트가 필요로 하는
    분해(측정1)와 스윕 지표(측정2)를 만들 수 있게 한다.

    호출부가 매 스윕지점(vm_pu 변경 시)마다 net을 새로 build해서 넘기면 워밍스타트
    이력이 스윕 지점 간에 섞이지 않는다(CLAUDE.md 7절 "수치 잡음의 성격" 참조 - 이
    스크립트는 그 방식을 택함, 지시사항이 허용하는 두 방식 중 하나).
    """
    n_scen = len(PM.ALL_DAYS)
    n_bus = PM.N_BUS
    line_in_service = net.line['in_service'].to_numpy()
    line_rating = net.line['max_i_ka'].to_numpy()
    n_line = len(net.line)

    vm_pu = np.zeros((n_scen, PM.TIME_STEPS, n_bus))
    p_slack = {s: np.zeros(PM.TIME_STEPS) for s in PM.ALL_DAYS}
    loss = {s: np.zeros(PM.TIME_STEPS) for s in PM.ALL_DAYS}
    i_ratio = np.zeros((n_scen, PM.TIME_STEPS, n_line))

    for si, s in enumerate(PM.ALL_DAYS):
        profile = PM.LOAD[s]
        for t in range(PM.TIME_STEPS):
            scale = profile[t]
            net.load['p_mw'] = base_p * scale
            net.load['q_mvar'] = base_q * scale

            ok = evaluate._run_pf_with_retry(net)
            if not ok:
                raise RuntimeError(
                    f'조류계산 발산: 시나리오={s}, t={t} (ESS 없는 기저 상태 발산은 비정상)'
                )

            vm_pu[si, t, :] = net.res_bus.vm_pu.to_numpy()
            p_slack[s][t] = net.res_ext_grid.p_mw.sum()
            loss[s][t] = net.res_line.pl_mw.sum()
            i_ratio[si, t, line_in_service] = (
                net.res_line.i_ka.to_numpy()[line_in_service] / line_rating[line_in_service]
            )

    return dict(
        vm_pu=vm_pu, p_slack=p_slack, loss=loss, i_ratio=i_ratio,
        n_line=n_line, line_in_service=line_in_service,
    )


def _violation_components(vm_pu):
    """max(0, V-V_MAX)[상한] + max(0, V_MIN-V)[하한]. evaluate.py와 완전히 동일한 정의
    (params.V_MIN/V_MAX 상수를 그대로 쓰므로 재구현이되 정의가 갈릴 여지가 없다)."""
    over = np.maximum(0.0, vm_pu - PM.V_MAX)
    under = np.maximum(0.0, PM.V_MIN - vm_pu)
    return under, over


# ============================================================
# 측정 1: 기저 상태(ESS 없음) 전압 위반 상세분해
# ============================================================

def measurement1():
    section('측정 1: 기저 상태(ESS 없음) 전압 위반 상세분해')

    # 재사용 경로: evaluate.py 자신의 공식 합산치를 교차검증 기준으로 가져온다.
    evaluate.init_worker()
    official_total = evaluate._BASE_FLOW['v_violation']
    print(f'[재사용] evaluate._compute_base_flow의 공식 v_violation 합계 = '
          f'{official_total:.6f} pu', flush=True)

    # 상세분해: 별도 net에서 직접 수집(evaluate.py가 분해를 노출하지 않으므로).
    net = build_net()
    base_p = net.load['p_mw'].to_numpy().copy()
    base_q = net.load['q_mvar'].to_numpy().copy()
    data = _run_all_days_detailed(net, base_p, base_q)

    under, over = _violation_components(data['vm_pu'])
    total_under = float(np.sum(under))
    total_over = float(np.sum(over))
    total = total_under + total_over

    print(f'[직접계산] 총 위반량 = {total:.6f} pu (하한 {total_under:.6f} + 상한 {total_over:.6f})',
          flush=True)
    diff = abs(total - official_total)
    match = diff < 1e-6
    print(f'evaluate.py 공식값과의 차이 = {diff:.3e} pu '
          f'({"일치(부동소수 잡음 수준) - 재사용 검증 통과" if match else "★불일치 - 확인 필요"})',
          flush=True)

    print('\n시나리오별:', flush=True)
    for si, s in enumerate(PM.ALL_DAYS):
        u = float(np.sum(under[si]))
        o = float(np.sum(over[si]))
        print(f'  {s:12s}: 총 {u + o:.6f}  (하한 {u:.6f} / 상한 {o:.6f})', flush=True)

    print('\n버스별 기여 (전 시나리오·시각 합산, 0보다 큰 것만, 내림차순):', flush=True)
    bus_total = under.sum(axis=(0, 1)) + over.sum(axis=(0, 1))  # (n_bus,)
    for b in np.argsort(-bus_total):
        if bus_total[b] <= 0:
            continue
        print(f'  bus {b:2d}: {bus_total[b]:.6f} pu', flush=True)
    if not np.any(bus_total > 0):
        print('  (위반 버스 없음)', flush=True)

    print('\n시각별 기여 (전 시나리오·버스 합산):', flush=True)
    time_total = under.sum(axis=(0, 2)) + over.sum(axis=(0, 2))  # (T,)
    for t in range(PM.TIME_STEPS):
        marker = '  <-- 위반' if time_total[t] > 0 else ''
        print(f'  t={t:2d}: {time_total[t]:.6f} pu{marker}', flush=True)

    section('측정 1 판정: main.py 진단값(0.334 pu)과 기저 위반량 비교')
    print(f'기저(ESS 없음) 위반량 = {total:.6f} pu, main.py 진단값 = {MAIN_DIAGNOSED_V_VIOLATION} pu',
          flush=True)
    if total > MAIN_DIAGNOSED_V_VIOLATION:
        print(f'  판정: 기저({total:.6f}) > ESS 적용 후 관측값({MAIN_DIAGNOSED_V_VIOLATION}) '
              '-> ESS가 위반을 줄이고는 있으나 완전히 없애지는 못한다.', flush=True)
    elif abs(total - MAIN_DIAGNOSED_V_VIOLATION) < 1e-3:
        print(f'  판정: 기저({total:.6f}) ≈ ESS 적용 후 관측값({MAIN_DIAGNOSED_V_VIOLATION}) '
              '-> "구조적 잔여 위반" 가설과 정합. ESS가 이 위반에 거의 손을 못 댄다는 뜻.',
              flush=True)
    else:
        print(f'  판정: 기저({total:.6f}) < ESS 적용 후 관측값({MAIN_DIAGNOSED_V_VIOLATION}) '
              '-> ESS가 오히려 위반을 악화시키고 있다(예상 밖 - 배치·충방전 스케줄 재검토 필요).',
              flush=True)

    return dict(total=total, official_total=official_total, match=match)


# ============================================================
# 측정 2: 슬랙 전압(vm_pu) 스윕
# ============================================================

def measurement2():
    section('측정 2: 슬랙 전압(vm_pu) 스윕 - 변전소 OLTC 조정으로 기저 위반 해소 가능한가')

    snap_idx = PM.ALL_DAYS.index(SNAPSHOT_SCENARIO)
    rows = []

    for vm_pu in VM_PU_SWEEP:
        # 스윕 지점마다 net을 새로 build -> 이전 vm_pu의 워밍스타트 이력과 완전히 분리.
        net = build_net()
        net.ext_grid['vm_pu'] = vm_pu
        base_p = net.load['p_mw'].to_numpy().copy()
        base_q = net.load['q_mvar'].to_numpy().copy()
        data = _run_all_days_detailed(net, base_p, base_q)

        under, over = _violation_components(data['vm_pu'])
        total_under = float(np.sum(under))
        total_over = float(np.sum(over))

        vmin = float(data['vm_pu'].min())
        vmin_si, vmin_t, vmin_bus = np.unravel_index(np.argmin(data['vm_pu']), data['vm_pu'].shape)
        vmax = float(data['vm_pu'].max())
        vmax_si, vmax_t, vmax_bus = np.unravel_index(np.argmax(data['vm_pu']), data['vm_pu'].shape)

        max_util = float(data['i_ratio'].max())
        mu_si, mu_t, mu_line = np.unravel_index(np.argmax(data['i_ratio']), data['i_ratio'].shape)

        # CLAUDE.md 1절 검증표와 같은 스냅샷(summer_peak, t=18=scale 1.0) 기준 손실/line0 전류.
        loss_kw_snapshot = float(data['loss'][SNAPSHOT_SCENARIO][SNAPSHOT_T]) * 1000
        line0_a_snapshot = float(data['i_ratio'][snap_idx, SNAPSHOT_T, 0]
                                  * net.line.at[0, 'max_i_ka'] * 1000)

        row = dict(
            vm_pu=vm_pu,
            vmin=vmin, vmin_scenario=PM.ALL_DAYS[vmin_si], vmin_bus=int(vmin_bus), vmin_t=int(vmin_t),
            vmax=vmax, vmax_scenario=PM.ALL_DAYS[vmax_si], vmax_bus=int(vmax_bus), vmax_t=int(vmax_t),
            total_violation=total_under + total_over,
            under_violation=total_under, over_violation=total_over,
            loss_kw_snapshot=loss_kw_snapshot, line0_current_a_snapshot=line0_a_snapshot,
            max_line_utilization=max_util, max_util_scenario=PM.ALL_DAYS[mu_si],
            max_util_line=int(mu_line), max_util_t=int(mu_t),
        )
        rows.append(row)

        print(f'vm_pu={vm_pu:.2f}: Vmin={vmin:.4f}pu(bus{vmin_bus},{PM.ALL_DAYS[vmin_si]},t={vmin_t}) '
              f'Vmax={vmax:.4f}pu(bus{vmax_bus},{PM.ALL_DAYS[vmax_si]},t={vmax_t}) '
              f'위반=하한{total_under:.4f}+상한{total_over:.4f} '
              f'손실(스냅샷)={loss_kw_snapshot:.2f}kW line0(스냅샷)={line0_a_snapshot:.2f}A '
              f'최대선로이용률={max_util:.4f}({PM.ALL_DAYS[mu_si]},line{mu_line},t={mu_t})',
              flush=True)

    section('측정 2 판정')
    zero_rows = [r for r in rows if r['total_violation'] <= 1e-9]
    if zero_rows:
        best = min(zero_rows, key=lambda r: r['vm_pu'])
        print(f"위반량이 0이 되는 최소 vm_pu = {best['vm_pu']:.2f}", flush=True)
        print(f"  그 지점 상한(1.05) 위반 = {best['over_violation']:.6f} "
              f"({'새로 생기지 않음' if best['over_violation'] <= 1e-9 else '★새로 생김'})", flush=True)
        print(f"  그 지점 최대 선로 이용률 = {best['max_line_utilization']:.4f} "
              f"({'여유 있음(<1.0)' if best['max_line_utilization'] < 1.0 else '★정격 초과'})", flush=True)
    else:
        print(f'스윕 범위 {VM_PU_SWEEP} 안에서는 위반량이 0이 되는 지점이 없음.', flush=True)
        min_row = min(rows, key=lambda r: r['total_violation'])
        print(f"  범위 내 최소 위반: vm_pu={min_row['vm_pu']:.2f}, "
              f"위반량={min_row['total_violation']:.6f}", flush=True)

    return rows


# ============================================================
# 측정 3: CLAUDE.md 1절 검증표 재현 확인
# ============================================================

def measurement3():
    section('측정 3: CLAUDE.md 1절 검증표 재현 확인 (vm_pu=1.0)')

    print('조건 확인: build_net.py의 __main__ 자체검증은 net을 만든 뒤 부하를 전혀 스케일링', flush=True)
    print("하지 않고(즉 scale=1.0) pp.runpp를 돌린다. 5절 주석 \"부하 정규화 1.0 = 여름최대일 ", flush=True)
    print("t=18 = case33bw 기본부하\"에 따르면 이는 LOAD['summer_peak'][18]과 수학적으로 동일한", flush=True)
    print('지점이다(이미 test_evaluate.py::test_base_reproduces_validation에서 검증된 사실 - ', flush=True)
    print('여기서는 evaluate._run_pf_with_retry 경로로 재확인한다).', flush=True)

    scale_at_snapshot = PM.LOAD[SNAPSHOT_SCENARIO][SNAPSHOT_T]
    print(f"\nLOAD['{SNAPSHOT_SCENARIO}'][{SNAPSHOT_T}] = {scale_at_snapshot} (1.0이어야 함)", flush=True)
    scenario_matches = np.isclose(scale_at_snapshot, 1.0)
    if not scenario_matches:
        print('★ scale이 1.0이 아님 - 위 "조건 확인" 전제가 깨졌다. 아래 재현 시도는 무의미할 수 있음.',
              flush=True)

    net = build_net()  # net.ext_grid.vm_pu=1.0 (build_net.py 기본값)
    ok = evaluate._run_pf_with_retry(net)  # 부하 스케일링 없음(build_net()이 만든 값 그대로)
    assert ok, '기저 조류계산 발산 - 정상 부하범위에서 비정상'

    loss_kw = net.res_line.pl_mw.sum() * 1000
    vmin = float(net.res_bus.vm_pu.min())
    vmin_bus = int(net.res_bus.vm_pu.idxmin())
    line0_a = net.res_line.at[0, 'i_ka'] * 1000
    slack_mw = net.res_ext_grid.p_mw.sum()

    checks = [
        ('총손실 kW', loss_kw, PM.VALIDATION['loss_kw_scaled']),
        ('Vmin pu', vmin, PM.VALIDATION['vmin_pu_scaled']),
        ('line0 전류 A', line0_a, PM.VALIDATION['line0_current_a_scaled']),
        ('슬랙 유입 MW', slack_mw, PM.VALIDATION['slack_import_mw_scaled']),
    ]
    all_ok = True
    for name, actual, expected in checks:
        atol = max(abs(expected) * 1e-4, 1e-3)
        ok_flag = bool(np.isclose(actual, expected, atol=atol))
        all_ok &= ok_flag
        print(f'  {name}: 실측={actual:.4f}  검증표={expected}  '
              f'{"OK" if ok_flag else "*** 불일치 ***"}', flush=True)
    bus_ok = (vmin_bus == PM.VALIDATION['vmin_bus'])
    all_ok &= bus_ok
    print(f'  Vmin 버스: 실측={vmin_bus}  검증표={PM.VALIDATION["vmin_bus"]}  '
          f'{"OK" if bus_ok else "*** 불일치 ***"}', flush=True)

    print(f'\n재현 판정: {"전부 일치" if all_ok else "*** 불일치 발견 - 위 표 참조, 숨기지 않고 보고함 ***"}',
          flush=True)
    return all_ok


# ============================================================
# CSV 저장 (측정 2 스윕 결과 - 가장 자연스러운 표 형태 데이터)
# ============================================================

CSV_FIELDS = [
    'vm_pu', 'vmin', 'vmin_scenario', 'vmin_bus', 'vmin_t',
    'vmax', 'vmax_scenario', 'vmax_bus', 'vmax_t',
    'total_violation', 'under_violation', 'over_violation',
    'loss_kw_snapshot', 'line0_current_a_snapshot',
    'max_line_utilization', 'max_util_scenario', 'max_util_line', 'max_util_t',
]


def _make_results_path():
    """scripts/ 바로 아래가 아니라 scripts/results/probe_voltage/ 밑에 모아 둔다(스크립트별
    결과 CSV가 늘어나며 scripts/가 난잡해지는 것을 막기 위함)."""
    out_dir = os.path.join(SCRIPT_DIR, 'results', 'probe_voltage')
    os.makedirs(out_dir, exist_ok=True)
    hostname = socket.gethostname()
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(out_dir, f'probe_voltage_{hostname}_{ts}.csv')


def _write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())
    print(f'\n측정 2 스윕 결과 CSV 저장: {path}', flush=True)


if __name__ == '__main__':
    _check_env()

    m1 = measurement1()
    m2_rows = measurement2()
    m3_ok = measurement3()

    results_path = _make_results_path()
    _write_csv(results_path, m2_rows)

    section('종합 요약')
    print(f'측정1 - 기저 위반량: {m1["total"]:.6f} pu '
          f'(evaluate.py 공식값과 {"일치" if m1["match"] else "불일치(확인 필요)"})', flush=True)
    print(f'측정1 - main.py 0.334pu 대비: '
          f'{"기저값이 더 큼(ESS가 일부 완화)" if m1["total"] > MAIN_DIAGNOSED_V_VIOLATION else "기저값과 근접/이하"}',
          flush=True)
    zero_vm_pu = [r['vm_pu'] for r in m2_rows if r['total_violation'] <= 1e-9]
    print(f'측정2 - 위반 0이 되는 vm_pu: '
          f'{min(zero_vm_pu) if zero_vm_pu else "스윕 범위 내 없음"}', flush=True)
    print(f'측정3 - CLAUDE.md 1절 검증표 재현: {"성공" if m3_ok else "실패(위 상세 참조)"}', flush=True)
