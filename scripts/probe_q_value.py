# Compare the same ESS placement with Q enabled and force_q_zero=True.
# The Q-disabled case re-optimizes P and is not a fixed-P post-hoc counterfactual.

import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

import evaluate
import params as PM


X_OPT = np.array(
    [32.0, 0.21358313449208763, 0.4407054133130149], dtype=float
)

_TABLE_METRICS = (
    "j_net",
    "b_energy",
    "b_defer",
    "b_arb",
    "b_loss",
    "cost",
    "v_violation",
    "i_violation",
)
_MONEY_METRICS = {
    "j_net", "b_energy", "b_defer", "b_arb", "b_loss", "cost"
}


def _number(value):
    return float(np.asarray(value, dtype=float))


def _table_cell(value, *, money=False):
    value = _number(value)
    if money:
        return f"{value:.6g}({value:.0f})"
    return f"{value:.6g}"


def _print_benefit_table(detail_q, detail_q0):
    print("[A]")
    print("metric,q_allowed,q_forbidden,diff_allowed_minus_forbidden")
    for metric in _TABLE_METRICS:
        q_value = _number(detail_q[metric])
        q0_value = _number(detail_q0[metric])
        diff = q_value - q0_value
        money = metric in _MONEY_METRICS
        print(
            f"{metric},{_table_cell(q_value, money=money)},"
            f"{_table_cell(q0_value, money=money)},"
            f"{_table_cell(diff, money=money)}"
        )


def _print_q_value(detail_q, detail_q0):
    j_q = _number(detail_q["j_net"])
    dj_net = j_q - _number(detail_q0["j_net"])
    percent = 100.0 * dj_net / j_q if abs(j_q) > 0.0 else float("nan")
    print("[B]")
    print("dj_net_won,dj_net_won_rounded,dj_net_over_q_allowed_pct")
    print(f"{dj_net:.6g},{dj_net:.0f},{percent:.6g}")


def _print_channels(detail_q, detail_q0):
    print("[C]")
    print("metric,diff_allowed_minus_forbidden_won,diff_rounded_won")
    for metric in ("b_defer", "b_arb", "b_loss"):
        diff = _number(detail_q[metric]) - _number(detail_q0[metric])
        print(f"d{metric},{diff:.6g},{diff:.0f}")


def _print_schedule_differences(detail_q, detail_q0):
    print("[D]")
    if "unit_p" not in detail_q or "unit_p" not in detail_q0:
        print("schedule_keys_missing")
        return
    print("scenario,max_abs_pnet_q0_minus_q_MW")
    for scenario in PM.ALL_DAYS:
        p_q = np.asarray(detail_q["unit_p"][scenario], dtype=float)
        p_q0 = np.asarray(detail_q0["unit_p"][scenario], dtype=float)
        max_abs = float(np.max(np.abs(p_q0 - p_q)))
        print(f"{scenario},{max_abs:.12e}")


def main():
    evaluate.init_worker()
    detail_q = evaluate.evaluate_particle(
        X_OPT, return_detail=True, force_q_zero=False
    )
    detail_q0 = evaluate.evaluate_particle(
        X_OPT, return_detail=True, force_q_zero=True
    )

    _print_benefit_table(detail_q, detail_q0)
    _print_q_value(detail_q, detail_q0)
    _print_channels(detail_q, detail_q0)
    _print_schedule_differences(detail_q, detail_q0)


if __name__ == "__main__":
    main()
