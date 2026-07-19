"""편익·비용 순수함수 모듈 (CLAUDE.md 3절, 8절, 부록A).

조류계산도 LP도 호출하지 않는다 - 숫자(dict[str, np.ndarray] 또는 스칼라) in -> 원(KRW) out만 한다.
pandapower, cvxpy는 import하지 않는다.

자료구조: 시나리오별 시계열은 전부 dict[str, np.ndarray(shape=(T,))]다. 배열+인덱스 매핑을
쓰지 않는 이유는 CLAUDE.md 3절 표대로 집계 대상이 에너지=AVG_DAYS/이연=PEAK_DAYS/제약=ALL_DAYS로
갈리는데, 인덱스 방식은 어긋나도 조용히 잘못된 답을 내기 때문이다. dict는 키 검증으로 즉시 터진다.
각 함수는 자신이 쓰는 그룹(AVG_DAYS/PEAK_DAYS/ALL_DAYS) 기준으로 키 커버리지를 확인하고,
그 그룹은 params.py의 리스트를 순회한다(하드코딩된 길이 없음 - Phase 2 시나리오 확장 대비).

단위: 이 모듈 전체가 MW/MWh/원 단일 단위다. 함수 안에 ×1000이 등장하지 않는다 - 대신
params.py에 미리 만들어 둔 MW/MWh 기준 상수(C_CAP_PER_MW_YR 등)만 쓴다.
"""

import numpy as np

import params as PM


def _assert_keys(data, required_keys, name):
    missing = set(required_keys) - set(data.keys())
    assert not missing, f'{name}에 필수 시나리오 키 누락: {missing}'


def assert_slack_balance(p_slack, load_sum, loss, p_ess, scenarios=None, atol=1e-7, enabled=True):
    """수지 항등식 P_slack = SigmaLoad + Loss - P_ESS 를 시나리오·시각별로 검증한다.

    atol=1e-7 MW, rtol=0 (확정값, scripts/check_balance.py 실측 근거): 0번 계통·ESS 주입
    스윕(-2~+2MW)·summer_peak 24h 전체에서 관측된 최대 절대잔차는 4.44e-15~7.11e-15 MW로
    순수 float64 반올림 잡음(machine epsilon 수준)이었고 ESS 주입 크기와 무관했다. atol=1e-7은
    그 잡음 바닥보다 6자릿수 여유를 둬 다른 시나리오에서도 오탐 없이 통과시키되, 만약 실제로
    수지식에 빠진 항이 있다면(그 경우 잔차는 kW 스케일, 1e-3 MW 이상) 확실히 걸러낸다.
    rtol을 쓰지 않는 이유: 기본 rtol=1e-5를 두면 8.8MW 근방에서 허용오차가 8.8e-5로 커져
    의도한 절대기준이 무너진다 - 반드시 rtol=0으로 명시한다.

    enabled=False로 끌 수 있다 (lower_lp._assert_physics와 같은 패턴, 본실험 속도용).
    """
    if not enabled:
        return
    if scenarios is None:
        scenarios = PM.ALL_DAYS

    for data, name in ((p_slack, 'p_slack'), (load_sum, 'load_sum'), (loss, 'loss'), (p_ess, 'p_ess')):
        _assert_keys(data, scenarios, name)

    for s in scenarios:
        residual = p_slack[s] - (load_sum[s] + loss[s] - p_ess[s])
        if not np.allclose(residual, 0.0, atol=atol, rtol=0.0):
            bad_t = np.where(np.abs(residual) > atol)[0]
            detail = ', '.join(f't={t}(r={residual[t]:.3e}MW)' for t in bad_t)
            raise AssertionError(
                f'슬랙 수지 불일치: 시나리오={s}, atol={atol}MW, 위반 시각: {detail}'
            )


def b_energy(p_slack_base, p_slack_ess, smp_mwh, n_weekdays):
    """B_energy (CLAUDE.md 3절): 손실 편익 + 조달비를 이중계상 없이 함께 잡는다.
    Sigma_{s in AVG_DAYS} N_WD[s] * Sigma_t (P_slack_base[s,t] - P_slack_ess[s,t]) * SMP[s,t] * dt

    AVG_DAYS만 순회한다 - PEAK_DAYS를 넣으면 이중계상이다(최대일 데이터는 평균일 통계에
    이미 녹아 있음 - CLAUDE.md 3절).
    """
    for data, name in ((p_slack_base, 'p_slack_base'), (p_slack_ess, 'p_slack_ess'),
                        (smp_mwh, 'smp_mwh'), (n_weekdays, 'n_weekdays')):
        _assert_keys(data, PM.AVG_DAYS, name)

    total_won = 0.0
    for s in PM.AVG_DAYS:
        delta_mw = p_slack_base[s] - p_slack_ess[s]
        day_won = float(np.sum(delta_mw * smp_mwh[s])) * PM.DT_HOURS
        total_won += n_weekdays[s] * day_won
    return total_won


def b_defer(p_slack_base, p_slack_ess):
    """B_defer (CLAUDE.md 3절): 설비 이연. PEAK_DAYS 전체를 통합해 단일 max를 취한다
    (연중 최대 슬랙피크 1점, 일수가중 없음).

    ★ pk_s(solve_peak이 반환하는 부하단 대리 피크)는 인자로 받지 않는다 - 받을 수 없는
    시그니처라 오용이 구조적으로 불가능하다. 여기 들어오는 값은 반드시 실제 조류계산으로
    얻은 슬랙 유입 배열이어야 한다(CLAUDE.md 2절 경고, solve_peak docstring 참조).

    음수를 클리핑하지 않는다: ESS가 피크시각에 충전해 슬랙 피크를 오히려 키우면 B_defer<0이
    되는데, 이는 설비 투자를 앞당기는 실질 손해라 물리적으로 옳다. 0으로 클립하면 나쁜 해가
    PSO에게 "무해"로 보여 탐색 지형이 평탄해진다(PSO는 fitness 차이로 움직이므로 평탄 영역은
    방향을 잃는 구간이 된다) - CLAUDE.md 설계원칙.
    """
    for data, name in ((p_slack_base, 'p_slack_base'), (p_slack_ess, 'p_slack_ess')):
        _assert_keys(data, PM.PEAK_DAYS, name)

    peak_base_mw = max(float(np.max(p_slack_base[s])) for s in PM.PEAK_DAYS)
    peak_ess_mw = max(float(np.max(p_slack_ess[s])) for s in PM.PEAK_DAYS)

    return PM.C_CAP_PER_MW_YR * (peak_base_mw - peak_ess_mw)


def capex(s_mva, e_mwh):
    """CAPEX = C_KW_CAPEX_PER_MVA*S + C_KWH_CAPEX_PER_MWH*E (CLAUDE.md 3절)."""
    return PM.C_KW_CAPEX_PER_MVA * s_mva + PM.C_KWH_CAPEX_PER_MWH * e_mwh


def opex(s_mva, e_mwh):
    """OPEX = C_KWH_OPEX_PER_MWH_YR*E + WARRANTY_RATE*CAPEX (CLAUDE.md 3절)."""
    return PM.C_KWH_OPEX_PER_MWH_YR * e_mwh + PM.WARRANTY_RATE * capex(s_mva, e_mwh)


def total_cost(s_mva, e_mwh):
    """Cost = CRF_20*CAPEX + OPEX.

    ★ CRF_20(ESS 달력수명 20년, 4.5%)을 쓴다. CRF_30(30년)은 c_cap 산출에 이미 반영된
    값이라 여기서 다시 쓰면 이중 반영이다(CLAUDE.md 4절 CRF 구분 표).
    """
    return PM.CRF_20 * capex(s_mva, e_mwh) + opex(s_mva, e_mwh)


def j_net(b_energy_val, b_defer_val, s_mva, e_mwh):
    """J_net = B_energy + B_defer - Cost (CLAUDE.md 3절). [원/년]"""
    return b_energy_val + b_defer_val - total_cost(s_mva, e_mwh)


# ============================================================
# 후처리용 편익 분해 (CLAUDE.md 8절) - 최적화에 개입하지 않는 순수 계산
# ============================================================

def b_arb(p_ch, p_dis, smp_mwh, n_weekdays):
    """B_arb (CLAUDE.md 8절): 조달비만 분리. ESS 순방전 x SMP.
    Sigma_s N_WD[s] Sigma_t (P_dis[s,t] - P_ch[s,t]) * SMP[s,t] * dt   (AVG_DAYS)

    p_ch/p_dis: 실제로 sgen에 주입된 값 기준 충/방전(MW, 둘 다 비음수 규약).
    """
    for data, name in ((p_ch, 'p_ch'), (p_dis, 'p_dis'), (smp_mwh, 'smp_mwh'), (n_weekdays, 'n_weekdays')):
        _assert_keys(data, PM.AVG_DAYS, name)

    total_won = 0.0
    for s in PM.AVG_DAYS:
        net_mw = p_dis[s] - p_ch[s]
        day_won = float(np.sum(net_mw * smp_mwh[s])) * PM.DT_HOURS
        total_won += n_weekdays[s] * day_won
    return total_won


def b_loss(loss_base, loss_ess, smp_mwh, n_weekdays):
    """B_loss (CLAUDE.md 8절): 손실만 분리.
    Sigma_s N_WD[s] Sigma_t (Loss_base[s,t] - Loss_ess[s,t]) * SMP[s,t] * dt   (AVG_DAYS)

    음수가 나올 수 있다 - 오류가 아니다. LP가 조달비(슬랙 유입 최소화)를 좇아 심야에
    대량 충전하면 그 시각 손실(전류^2에 비례)이 오히려 늘어날 수 있다(CLAUDE.md 8절).
    """
    for data, name in ((loss_base, 'loss_base'), (loss_ess, 'loss_ess'), (smp_mwh, 'smp_mwh'), (n_weekdays, 'n_weekdays')):
        _assert_keys(data, PM.AVG_DAYS, name)

    total_won = 0.0
    for s in PM.AVG_DAYS:
        delta_mw = loss_base[s] - loss_ess[s]
        day_won = float(np.sum(delta_mw * smp_mwh[s])) * PM.DT_HOURS
        total_won += n_weekdays[s] * day_won
    return total_won


def check_b_energy_decomposition(b_arb_val, b_loss_val, b_energy_val, atol=1e-6):
    """검산: B_arb + B_loss ~= B_energy (CLAUDE.md 8절).

    이 항등식은 P_slack = SigmaLoad + Loss - P_ESS 에서 나오며, P_ESS가 "실제로 sgen에
    주입된 값"일 때만 성립한다(LP가 반환한 원본 스케줄이 아니라 주입값이어야 한다 -
    부호 규약·버스별 분배·단위 변환이 어긋나는 대표적 경로이므로 여기서 갈라진다면
    이 셋 중 하나를 의심할 것).
    """
    return bool(np.isclose(b_arb_val + b_loss_val, b_energy_val, atol=atol, rtol=0.0))
