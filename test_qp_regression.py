"""QP 손실항 편입 회귀 기준(CLAUDE.md 부록 A).

실제 손실계수 측정을 포함하므로 느린 테스트다. Codex 환경에서는 실행하지 않고,
사용자 데스크탑 터미널에서 pytest 또는 이 파일 직접 실행으로 검증한다.
"""

from functools import lru_cache

import numpy as np
import pytest

import loss_coeffs
import lower_lp as LP
import params as PM


BUS = 15
S_MVA = 0.20
E_MWH = 0.42
T = PM.TIME_STEPS
AVG_SCENARIO = PM.AVG_DAYS[0]
PEAK_SCENARIO = PM.PEAK_DAYS[0]


def _zero_coeffs():
    return {
        name: np.zeros((1, T), dtype=float)
        for name in loss_coeffs.COEFF_NAMES
    }


@lru_cache(maxsize=None)
def _measured_coeffs(scenario):
    grid = loss_coeffs.measure_coeffs_grid(
        BUS,
        S_MVA,
        scenarios=(scenario,),
        times=range(T),
        p_center=0.0,
        strict=True,
    )
    coeffs = {
        name: np.asarray(grid[name], dtype=float).copy()
        for name in loss_coeffs.COEFF_NAMES
    }
    assert all(value.shape == (1, T) for value in coeffs.values())
    return coeffs


def _solve_avg_with_objective(coeffs, force_q_zero):
    smp = np.asarray(PM.SMP_PER_MWH[AVG_SCENARIO], dtype=float)
    profile = np.asarray(PM.LOAD[AVG_SCENARIO], dtype=float)
    p_net, q, soc = LP.solve_avg(
        S_mva=[S_MVA],
        E_mwh=[E_MWH],
        bus_idx=[BUS],
        smp=smp,
        profile=profile,
        coeffs=coeffs,
        mu_volt=0.0,
        force_q_zero=force_q_zero,
    )
    normalized = LP._normalize_coeffs(coeffs, 1, T)
    entry = LP._get_problem(
        'avg',
        1,
        force_q_zero,
        PM.ETA_C,
        PM.ETA_D,
        PM.SELF_DISCHARGE_HOURLY,
        PM.SOC_INIT_FRAC,
        PM.SOC_MIN_FRAC,
        PM.SOC_MAX_FRAC,
        0.0,
        normalized,
    )
    return p_net, q, soc, float(entry['problem'].value)


@pytest.mark.slow
def test_layer1_force_q_zero_p_loss_is_bounded():
    """Q=0에서 실계수와 무손실 기준의 차이는 P 손실항 기여로만 제한된다."""
    measured = _measured_coeffs(AVG_SCENARIO)
    p_real, q_real, _, _ = _solve_avg_with_objective(measured, True)
    p_zero, q_zero, _, _ = _solve_avg_with_objective(_zero_coeffs(), True)

    max_p_shift = float(np.max(np.abs(p_real - p_zero)))
    assert np.allclose(q_real, 0.0, atol=1e-7)
    assert np.allclose(q_zero, 0.0, atol=1e-7)
    assert np.isfinite(max_p_shift)
    # 원 단위 2,100~2,600원 게이트는 evaluate 통합 경로가 필요하다.
    # 이 파일에서는 스케줄이 정격의 25% 이상 바뀌는 구조적 오염을 하드 실패로 둔다.
    assert max_p_shift <= 0.25 * S_MVA, max_p_shift
    print(f'layer1 max|P_real-P_zero|={max_p_shift:.12g} MW')


@pytest.mark.slow
def test_layer2_q_enabled_objective_is_monotone():
    """Q=0이 feasible subset이므로 Q 허용 QP의 solver 목적값은 더 나쁠 수 없다."""
    measured = _measured_coeffs(AVG_SCENARIO)
    _, q_free, _, objective_free = _solve_avg_with_objective(measured, False)
    _, q_zero, _, objective_zero = _solve_avg_with_objective(measured, True)

    tolerance = 1e-6 * max(1.0, abs(objective_zero))
    assert objective_free <= objective_zero + tolerance, (
        objective_free,
        objective_zero,
        tolerance,
    )
    assert np.allclose(q_zero, 0.0, atol=1e-7)
    assert np.all(np.isfinite(q_free))
    print(
        'layer2 objective_free/objective_zero=',
        objective_free,
        objective_zero,
    )


@pytest.mark.slow
def test_peak_loss_reduction_sign_with_measured_coeffs():
    """실측 PEAK 계수로 방전 시 -a_P*P가 음수가 아님을 명시적으로 재확인한다."""
    coeffs = _measured_coeffs(PEAK_SCENARIO)
    assert np.all(coeffs['a_P'] < 0.0)
    assert np.all(coeffs['a_Q'] < 0.0)

    profile = np.asarray(PM.LOAD[PEAK_SCENARIO], dtype=float)
    load_total = 10.0 * profile
    p_net, _, _, _ = LP.solve_peak(
        S_mva=[S_MVA],
        E_mwh=[E_MWH],
        bus_idx=[BUS],
        load_total=load_total,
        profile=profile,
        coeffs=coeffs,
        mu_volt=0.0,
        force_q_zero=False,
    )
    discharge = p_net[0] > 1e-9
    assert np.any(discharge), '부호를 검증할 방전 시각이 없음'
    p_loss_reduction = -coeffs['a_P'][0, discharge] * p_net[0, discharge]
    assert np.all(p_loss_reduction > -1e-9), p_loss_reduction
    print(f'peak min(-a_P*P)={float(np.min(p_loss_reduction)):.12g} MW')


@pytest.mark.skip(
    reason='P 고정 프로토타입 및 원 단위 j_net 재현은 evaluate 통합 회귀로 후속 분리'
)
def test_layer3_fixed_p_prototype_reference():
    """부록 A 층 3 참고 기준의 자리표시자(하드 게이트 아님)."""


if __name__ == '__main__':
    test_layer1_force_q_zero_p_loss_is_bounded()
    test_layer2_q_enabled_objective_is_monotone()
    test_peak_loss_reduction_sign_with_measured_coeffs()
    print('layer3 skipped: evaluate 통합 원 단위/P 고정 회귀는 후속 분리')
    print('all implemented QP regression tests passed')
