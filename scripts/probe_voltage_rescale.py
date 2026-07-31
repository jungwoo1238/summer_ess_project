r"""새 부하 스케일링(S=10 MVA, 총 역률 0.95)의 기저 전압을 계측한다.

본코드는 수정하지 않는다. 실행은 사용자가 직접 수행한다.

    python C:\Users\PowerSL\summer_ess_project\scripts\probe_voltage_rescale.py
"""

from __future__ import annotations

import datetime as datetime_
import importlib.metadata
import os
import platform
import socket
import sys
import time
from typing import Any

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import evaluate
import params as PM
from build_net import build_net


VM_PU_SWEEP = (1.02, 1.03, 1.04, 1.05, 1.06)
ORIGINAL_P_MW = 3.715
ORIGINAL_Q_MVAR = 2.300
TARGET_S_MVA = 10.0
TARGET_PF = 0.95
TARGET_P_MW = TARGET_S_MVA * TARGET_PF
TARGET_Q_MVAR = TARGET_S_MVA * np.sqrt(1.0 - TARGET_PF**2)
K_P_EXACT = TARGET_P_MW / ORIGINAL_P_MW
K_Q_EXACT = TARGET_Q_MVAR / ORIGINAL_Q_MVAR
K_P = 2.557201
K_Q = 1.357609
ZERO_TOL = 1e-9

LEGACY_REFERENCE = {
    "load_case": "legacy_reference",
    "vm_pu": 1.02,
    "vmin_pu": 0.96199,
    "vmin_bus": "",
    "vmin_scenario": "",
    "vmin_t": "",
    "lower_violation_l1_pu": 0.0,
    "vmax_pu": "",
    "vmax_bus": "",
    "vmax_scenario": "",
    "vmax_t": "",
    "upper_violation_l1_pu": "",
    "total_line_loss_kw_sum": "",
    "mean_line_loss_kw": "",
    "max_line_loss_kw": "",
    "max_line_utilization_pu": "",
    "max_util_line": "",
    "max_util_scenario": "",
    "max_util_t": "",
    "converged_cases": 120,
    "diverged_cases": 0,
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def _paths() -> str:
    stamp = datetime_.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(
        RESULTS_DIR, f"probe_voltage_rescale_{stamp}_report.md"
    )


def _md_table(rows: list[dict[str, Any]]) -> list[str]:
    columns = list(rows[0]) if rows else []
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, (float, np.floating)):
                value = f"{float(value):.12g}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _write_report(
    path: str,
    table0: list[dict[str, Any]],
    table1: list[dict[str, Any]] | None = None,
    table2: list[dict[str, Any]] | None = None,
    environment: list[dict[str, Any]] | None = None,
) -> None:
    lines = ["# probe_voltage_rescale", "", "## Table 0", ""]
    lines.extend(_md_table(table0))
    if table1 is not None:
        lines.extend(["", "## Table 1", ""])
        lines.extend(_md_table(table1))
    if table2 is not None:
        lines.extend(["", "## Table 2", ""])
        lines.extend(_md_table(table2))
    if environment is not None:
        lines.extend(["", "## Environment", ""])
        lines.extend(_md_table(environment))
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
        file.flush()
        os.fsync(file.fileno())


def _print_table(label: str, rows: list[dict[str, Any]]) -> None:
    print(label, flush=True)
    for row in rows:
        print(
            " ".join(f"{key}={value}" for key, value in row.items()),
            flush=True,
        )


def _new_scaled_net(vm_pu: float):
    net = build_net(slack_vm_pu=vm_pu)

    built_p = net.load["p_mw"].to_numpy(dtype=float).copy()
    built_q = net.load["q_mvar"].to_numpy(dtype=float).copy()
    original_p = built_p / float(PM.K_P)
    original_q = built_q / float(PM.K_Q)

    original_p_sum = float(np.sum(original_p))
    original_q_sum = float(np.sum(original_q))
    if not np.isclose(original_p_sum, ORIGINAL_P_MW, atol=1e-10):
        raise RuntimeError(
            f"original_p_sum_mismatch={original_p_sum:.12g}"
        )
    if not np.isclose(original_q_sum, ORIGINAL_Q_MVAR, atol=1e-10):
        raise RuntimeError(
            f"original_q_sum_mismatch={original_q_sum:.12g}"
        )

    net.load["p_mw"] = original_p * K_P
    net.load["q_mvar"] = original_q * K_Q
    return net, {
        "built_net_p_sum_before_inverse_mw": float(np.sum(built_p)),
        "built_net_q_sum_before_inverse_mvar": float(np.sum(built_q)),
        "original_p_sum_after_inverse_mw": original_p_sum,
        "original_q_sum_after_inverse_mvar": original_q_sum,
    }


def _table0(net, recovery: dict[str, float]) -> list[dict[str, Any]]:
    p_sum = float(net.load["p_mw"].sum())
    q_sum = float(net.load["q_mvar"].sum())
    s_sum = float(np.hypot(p_sum, q_sum))
    pf = p_sum / s_sum
    return [{
        **recovery,
        "K_P_applied": K_P,
        "K_Q_applied": K_Q,
        "K_P_formula_unrounded": K_P_EXACT,
        "K_Q_formula_unrounded": K_Q_EXACT,
        "applied_p_sum_mw": p_sum,
        "target_p_sum_mw": TARGET_P_MW,
        "p_sum_error_mw": p_sum - TARGET_P_MW,
        "applied_q_sum_mvar": q_sum,
        "target_q_sum_mvar": TARGET_Q_MVAR,
        "q_sum_error_mvar": q_sum - TARGET_Q_MVAR,
        "applied_s_sum_mva": s_sum,
        "target_s_sum_mva": TARGET_S_MVA,
        "s_sum_error_mva": s_sum - TARGET_S_MVA,
        "applied_total_pf": pf,
        "target_total_pf": TARGET_PF,
        "pf_error": pf - TARGET_PF,
    }]


def _run_vm(vm_pu: float) -> tuple[dict[str, Any], int]:
    net, _recovery = _new_scaled_net(vm_pu)
    base_p = net.load["p_mw"].to_numpy(dtype=float).copy()
    base_q = net.load["q_mvar"].to_numpy(dtype=float).copy()

    ext_grid_buses = {
        int(bus) for bus in net.ext_grid.loc[
            net.ext_grid["in_service"].astype(bool), "bus"
        ].to_numpy()
    }
    non_slack_buses = np.array(
        [int(bus) for bus in net.bus.index if int(bus) not in ext_grid_buses],
        dtype=int,
    )
    line_in_service = net.line["in_service"].to_numpy(dtype=bool)
    line_rating = net.line["max_i_ka"].to_numpy(dtype=float)

    vmin = np.inf
    vmax = -np.inf
    vmin_location: tuple[int, str, int] | None = None
    vmax_location: tuple[int, str, int] | None = None
    lower_l1 = 0.0
    upper_l1 = 0.0
    losses_kw: list[float] = []
    max_util = -np.inf
    max_util_location: tuple[int, str, int] | None = None
    calls = 0
    diverged: list[tuple[str, int]] = []

    for scenario in PM.ALL_DAYS:
        for t in range(PM.TIME_STEPS):
            scale = float(PM.LOAD[scenario][t])
            net.load["p_mw"] = base_p * scale
            net.load["q_mvar"] = base_q * scale
            calls += 1
            ok = evaluate._run_pf_with_retry(net)
            if not ok:
                diverged.append((scenario, t))
                print(
                    f"pf_diverged vm_pu={vm_pu} scenario={scenario} t={t}",
                    flush=True,
                )
                continue

            vm = net.res_bus.vm_pu.loc[non_slack_buses].to_numpy(dtype=float)
            local_min_index = int(np.argmin(vm))
            local_max_index = int(np.argmax(vm))
            local_min = float(vm[local_min_index])
            local_max = float(vm[local_max_index])
            if local_min < vmin:
                vmin = local_min
                vmin_location = (
                    int(non_slack_buses[local_min_index]), scenario, t
                )
            if local_max > vmax:
                vmax = local_max
                vmax_location = (
                    int(non_slack_buses[local_max_index]), scenario, t
                )

            lower_l1 += float(np.sum(np.maximum(0.0, PM.V_MIN - vm)))
            upper_l1 += float(np.sum(np.maximum(0.0, vm - PM.V_MAX)))
            losses_kw.append(float(net.res_line.pl_mw.sum()) * 1000.0)

            utilization = np.full(len(net.line), np.nan)
            utilization[line_in_service] = (
                net.res_line.i_ka.to_numpy(dtype=float)[line_in_service]
                / line_rating[line_in_service]
            )
            local_line = int(np.nanargmax(utilization))
            local_util = float(utilization[local_line])
            if local_util > max_util:
                max_util = local_util
                max_util_location = (local_line, scenario, t)

    vmin_bus, vmin_scenario, vmin_t = (
        vmin_location if vmin_location is not None else ("", "", "")
    )
    vmax_bus, vmax_scenario, vmax_t = (
        vmax_location if vmax_location is not None else ("", "", "")
    )
    util_line, util_scenario, util_t = (
        max_util_location if max_util_location is not None else ("", "", "")
    )
    loss_array = np.asarray(losses_kw, dtype=float)
    row = {
        "load_case": "new_pf_0p95",
        "vm_pu": vm_pu,
        "vmin_pu": vmin if np.isfinite(vmin) else np.nan,
        "vmin_bus": vmin_bus,
        "vmin_scenario": vmin_scenario,
        "vmin_t": vmin_t,
        "lower_violation_l1_pu": lower_l1,
        "vmax_pu": vmax if np.isfinite(vmax) else np.nan,
        "vmax_bus": vmax_bus,
        "vmax_scenario": vmax_scenario,
        "vmax_t": vmax_t,
        "upper_violation_l1_pu": upper_l1,
        "total_line_loss_kw_sum": (
            float(np.sum(loss_array)) if len(loss_array) else np.nan
        ),
        "mean_line_loss_kw": (
            float(np.mean(loss_array)) if len(loss_array) else np.nan
        ),
        "max_line_loss_kw": (
            float(np.max(loss_array)) if len(loss_array) else np.nan
        ),
        "max_line_utilization_pu": (
            max_util if np.isfinite(max_util) else np.nan
        ),
        "max_util_line": util_line,
        "max_util_scenario": util_scenario,
        "max_util_t": util_t,
        "converged_cases": len(loss_array),
        "diverged_cases": len(diverged),
    }
    return row, calls


def _table2(table1_new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    zero_rows = [
        row for row in table1_new
        if row["diverged_cases"] == 0
        and float(row["lower_violation_l1_pu"]) <= ZERO_TOL
    ]
    if not zero_rows:
        return [{
            "minimum_vm_pu_with_zero_lower_violation": "",
            "vmax_pu_at_minimum_vm": "",
            "upper_margin_1p05_minus_vmax_pu": "",
            "upper_violation_l1_pu": "",
            "zero_lower_violation_found": False,
        }]
    best = min(zero_rows, key=lambda row: float(row["vm_pu"]))
    return [{
        "minimum_vm_pu_with_zero_lower_violation": best["vm_pu"],
        "vmax_pu_at_minimum_vm": best["vmax_pu"],
        "upper_margin_1p05_minus_vmax_pu": (
            PM.V_MAX - float(best["vmax_pu"])
        ),
        "upper_violation_l1_pu": best["upper_violation_l1_pu"],
        "zero_lower_violation_found": True,
    }]


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return ""


def main() -> int:
    started = time.perf_counter()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = _paths()

    validation_net, recovery = _new_scaled_net(VM_PU_SWEEP[0])
    table0 = _table0(validation_net, recovery)
    _write_report(report_path, table0)
    _print_table("TABLE_0", table0)

    expected_calls = (
        len(VM_PU_SWEEP) * len(PM.ALL_DAYS) * PM.TIME_STEPS
    )
    print(f"expected_power_flow_cases={expected_calls}", flush=True)

    new_rows = []
    total_calls = 0
    for vm_pu in VM_PU_SWEEP:
        row, calls = _run_vm(vm_pu)
        new_rows.append(row)
        total_calls += calls
        _write_report(report_path, table0, [LEGACY_REFERENCE, *new_rows])
        print(
            "TABLE_1_ROW "
            + " ".join(f"{key}={value}" for key, value in row.items()),
            flush=True,
        )

    table1 = [LEGACY_REFERENCE, *new_rows]
    table2 = _table2(new_rows)
    _write_report(report_path, table0, table1, table2)
    _print_table("TABLE_2", table2)

    environment = [
        {"item": "python_version", "value": platform.python_version()},
        {"item": "pandapower_version", "value": _version("pandapower")},
        {"item": "numpy_version", "value": _version("numpy")},
        {"item": "hostname", "value": socket.gethostname()},
        {"item": "all_days_count", "value": len(PM.ALL_DAYS)},
        {"item": "time_steps", "value": PM.TIME_STEPS},
        {"item": "non_slack_bus_count", "value": PM.N_BUS - 1},
        {"item": "power_flow_case_calls", "value": total_calls},
        {
            "item": "diverged_case_count",
            "value": sum(int(row["diverged_cases"]) for row in new_rows),
        },
        {
            "item": "total_execution_time_s",
            "value": time.perf_counter() - started,
        },
        {
            "item": "violation_definition",
            "value": "sum(max(0,0.95-Vbus)); non-slack buses; ALL_DAYS x 24",
        },
        {
            "item": "legacy_reference_p_sum_mw",
            "value": 8.502,
        },
        {
            "item": "legacy_reference_q_sum_mvar",
            "value": 5.264,
        },
        {
            "item": "legacy_reference_total_pf",
            "value": 0.850241,
        },
    ]
    _write_report(report_path, table0, table1, table2, environment)
    _print_table("ENVIRONMENT", environment)
    print(f"report={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
