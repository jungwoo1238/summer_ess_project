"""두 기 동시주입 손실의 교차항을 기 간 전기적 거리별로 측정한다.

대각 근사(기별 손실의 독립 측정·합산)의 정당성 판정에 쓸 수치만 출력한다.
"""

import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import heapq
import math
import statistics
import sys
from pathlib import Path

import pandapower as pp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import loss_coeffs as LC
from build_net import build_net


B1 = 31
B2_CANDIDATES = (30, 25, 18, 15, 10, 5, 2)
INJECTION_POINTS = (
    (0.18, 0.10),
    (0.15, 0.15),
    (-0.15, 0.10),
    (0.00, 0.18),
)
SCENARIO = "summer"
TIME_INDEX = 18
PF_TOLERANCE_MVA = LC.PF_TOLERANCE_DEFAULT_MVA
DENOM_TOL_MW = 1e-15


def _fmt(value: float) -> str:
    return "%.6g" % float(value)


def _path_metrics(net, source: int, target: int) -> tuple[int, float]:
    """Return active-line shortest-path hops and summed |Z| in ohms."""
    adjacency: dict[int, list[tuple[int, float]]] = {
        int(bus): [] for bus in net.bus.index
    }
    for _, line in net.line.iterrows():
        if "in_service" in net.line.columns and not bool(line["in_service"]):
            continue
        from_bus = int(line["from_bus"])
        to_bus = int(line["to_bus"])
        length_km = float(line["length_km"])
        z_ohm = length_km * math.hypot(
            float(line["r_ohm_per_km"]), float(line["x_ohm_per_km"])
        )
        adjacency[from_bus].append((to_bus, z_ohm))
        adjacency[to_bus].append((from_bus, z_ohm))

    queue: list[tuple[float, int, int]] = [(0.0, 0, int(source))]
    best: dict[int, tuple[float, int]] = {int(source): (0.0, 0)}
    while queue:
        distance, hops, bus = heapq.heappop(queue)
        if (distance, hops) != best.get(bus):
            continue
        if bus == int(target):
            return hops, distance
        for next_bus, edge_distance in adjacency[bus]:
            candidate = (distance + edge_distance, hops + 1)
            if candidate < best.get(next_bus, (math.inf, sys.maxsize)):
                best[next_bus] = candidate
                heapq.heappush(queue, (candidate[0], candidate[1], next_bus))
    raise RuntimeError(f"no active-line path between bus {source} and bus {target}")


def _set_probe(
    net,
    sgen1_idx: int,
    sgen2_idx: int,
    b2: int,
    p1_mw: float,
    q1_mvar: float,
    p2_mw: float,
    q2_mvar: float,
) -> None:
    net.sgen.at[sgen1_idx, "p_mw"] = float(p1_mw)
    net.sgen.at[sgen1_idx, "q_mvar"] = float(q1_mvar)
    net.sgen.at[sgen2_idx, "bus"] = int(b2)
    net.sgen.at[sgen2_idx, "p_mw"] = float(p2_mw)
    net.sgen.at[sgen2_idx, "q_mvar"] = float(q2_mvar)


def _measure_loss_mw(net, stats: LC._RunStats) -> float:
    if not LC._run_pf_with_retry(net, stats, PF_TOLERANCE_MVA):
        raise RuntimeError("power flow failed after results/flat retries")
    return LC._network_loss_mw(net)


def _relative_pct(numerator_mw: float, denominator_mw: float) -> float:
    if abs(denominator_mw) <= DENOM_TOL_MW:
        return float("nan")
    return numerator_mw / denominator_mw * 100.0


def main() -> None:
    net = build_net()
    base_p = net.load["p_mw"].to_numpy(dtype=float).copy()
    base_q = net.load["q_mvar"].to_numpy(dtype=float).copy()
    LC._set_load(net, base_p, base_q, SCENARIO, TIME_INDEX)

    # pandapower sgen convention: positive P is generation/discharge and
    # positive Q is reactive injection, matching loss_coeffs._measure_loss.
    sgen1_idx = int(
        pp.create_sgen(
            net, bus=B1, p_mw=0.0, q_mvar=0.0, name="CROSS_TERM_PROBE_1"
        )
    )
    sgen2_idx = int(
        pp.create_sgen(
            net,
            bus=B2_CANDIDATES[0],
            p_mw=0.0,
            q_mvar=0.0,
            name="CROSS_TERM_PROBE_2",
        )
    )
    stats = LC._RunStats()

    # L0 is invariant over bus pairs and injection points at this fixed load.
    _set_probe(net, sgen1_idx, sgen2_idx, B2_CANDIDATES[0], 0.0, 0.0, 0.0, 0.0)
    l0_mw = _measure_loss_mw(net, stats)

    distances = {
        b2: _path_metrics(net, B1, b2) for b2 in B2_CANDIDATES
    }
    ordered_b2 = sorted(
        B2_CANDIDATES, key=lambda b2: (distances[b2][1], distances[b2][0], b2)
    )
    rows: list[dict[str, float | int]] = []

    for b2 in ordered_b2:
        path_hops, path_impedance_ohm = distances[b2]
        for p_inj_mw, q_inj_mvar in INJECTION_POINTS:
            _set_probe(
                net,
                sgen1_idx,
                sgen2_idx,
                b2,
                p_inj_mw,
                q_inj_mvar,
                0.0,
                0.0,
            )
            l1_mw = _measure_loss_mw(net, stats) - l0_mw

            _set_probe(
                net,
                sgen1_idx,
                sgen2_idx,
                b2,
                0.0,
                0.0,
                p_inj_mw,
                q_inj_mvar,
            )
            l2_mw = _measure_loss_mw(net, stats) - l0_mw

            _set_probe(
                net,
                sgen1_idx,
                sgen2_idx,
                b2,
                p_inj_mw,
                q_inj_mvar,
                p_inj_mw,
                q_inj_mvar,
            )
            l12_mw = _measure_loss_mw(net, stats) - l0_mw

            cross_c_mw = l12_mw - l1_mw - l2_mw
            cross_rel_pct = _relative_pct(cross_c_mw, l12_mw)
            diag_approx_err_pct = _relative_pct(abs(cross_c_mw), abs(l12_mw))
            rows.append(
                {
                    "b1": B1,
                    "b2": b2,
                    "path_hops": path_hops,
                    "path_impedance_ohm": path_impedance_ohm,
                    "p_inj_mw": p_inj_mw,
                    "q_inj_mvar": q_inj_mvar,
                    "l0_mw": l0_mw,
                    "l1_mw": l1_mw,
                    "l2_mw": l2_mw,
                    "l12_mw": l12_mw,
                    "cross_c_mw": cross_c_mw,
                    "cross_rel_pct": cross_rel_pct,
                    "diag_approx_err_pct": diag_approx_err_pct,
                }
            )

    print("[CROSS_TERM_BY_DISTANCE]")
    print(
        "b1,b2,path_hops,path_impedance_ohm,P_inj_MW,Q_inj_Mvar,"
        "L0_MW,L1_MW,L2_MW,L12_MW,cross_C_MW,cross_rel_pct,"
        "diag_approx_err_pct"
    )
    for row in rows:
        print(
            ",".join(
                (
                    str(row["b1"]),
                    str(row["b2"]),
                    str(row["path_hops"]),
                    _fmt(row["path_impedance_ohm"]),
                    _fmt(row["p_inj_mw"]),
                    _fmt(row["q_inj_mvar"]),
                    _fmt(row["l0_mw"]),
                    _fmt(row["l1_mw"]),
                    _fmt(row["l2_mw"]),
                    _fmt(row["l12_mw"]),
                    _fmt(row["cross_c_mw"]),
                    _fmt(row["cross_rel_pct"]),
                    _fmt(row["diag_approx_err_pct"]),
                )
            )
        )

    print("[CROSS_TERM_SUMMARY]")
    for p_inj_mw, q_inj_mvar in INJECTION_POINTS:
        point_rows = [
            row
            for row in rows
            if row["p_inj_mw"] == p_inj_mw
            and row["q_inj_mvar"] == q_inj_mvar
        ]
        nearest = point_rows[0]
        farthest = point_rows[-1]
        print(
            ",".join(
                (
                    "P_inj_MW=" + _fmt(p_inj_mw),
                    "Q_inj_Mvar=" + _fmt(q_inj_mvar),
                    "nearest_b2=" + str(nearest["b2"]),
                    "nearest_path_impedance_ohm="
                    + _fmt(nearest["path_impedance_ohm"]),
                    "nearest_cross_rel_pct=" + _fmt(nearest["cross_rel_pct"]),
                    "farthest_b2=" + str(farthest["b2"]),
                    "farthest_path_impedance_ohm="
                    + _fmt(farthest["path_impedance_ohm"]),
                    "farthest_cross_rel_pct=" + _fmt(farthest["cross_rel_pct"]),
                )
            )
        )

    absolute_relative = [
        abs(float(row["cross_rel_pct"]))
        for row in rows
        if math.isfinite(float(row["cross_rel_pct"]))
    ]
    print(
        "max_abs_cross_rel_pct=" + _fmt(max(absolute_relative))
        + ",median_abs_cross_rel_pct="
        + _fmt(statistics.median(absolute_relative))
    )


if __name__ == "__main__":
    main()
