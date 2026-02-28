"""
Baselines — 非 RL 路由基线策略，用于对比评估。

Non-RL routing strategies for comparison.

实现的策略 / Implemented Strategies:
  1. dijkstra: 全局 Dijkstra 最短时延路由 (忽略域边界, 上界参考)
  2. greedy:   域感知贪心路由 (每个域边界选利用率最低的链路)
  3. random:   随机路由 (随机选域间链路 + 随机选 K 路径, 下界参考)

公平性保证 / Fairness:
  所有基线使用与 HRL 相同的:
  - 流生成器 (Poisson, 相同种子)
  - 带宽分配机制 (原子分配)
  - 流生命周期管理 (到期释放)
  - SimEngine 实例 (相同拓扑和时隙设置)
"""

from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np

from core.flow import FlowRequest
from env.sim_engine import NetworkSimEngine


# ==================================================================
#  Global Dijkstra baseline
# ==================================================================
def route_flow_dijkstra(
    sim: NetworkSimEngine,
    flow: FlowRequest,
) -> tuple[bool, float, list[int]]:
    """全局 Dijkstra 路由 (忽略域边界, 上界参考)。

    Route a flow using global Dijkstra (ignoring domain boundaries).
    代表“理想情况下”的最优解, HRL 应尽量接近。

    Returns
    -------
    tuple[bool, float, list[int]]
        (success, delay_seconds, satellite_path)。
    """
    G = sim.topology.build_nx_graph(
        sim.current_timeslot,
        bw_manager=sim.bandwidth,
        bw_weight_alpha=sim.bw_weight_alpha,
    )
    try:
        path = nx.dijkstra_path(G, flow.src_sat, flow.dst_sat, weight="weight")
    except nx.NetworkXNoPath:
        return False, float("inf"), []

    # Check bandwidth feasibility
    if not sim.bandwidth.check_feasibility(path, flow.bandwidth):
        return False, float("inf"), path

    delay = sum(
        sim.topology.get_link_delay(path[i], path[i + 1], sim.current_timeslot)
        for i in range(len(path) - 1)
    )
    ok = sim.bandwidth.allocate(path, flow.bandwidth, flow.flow_id)
    return ok, delay, path


# ==================================================================
#  Greedy per-domain baseline
# ==================================================================
def route_flow_greedy_domain(
    sim: NetworkSimEngine,
    flow: FlowRequest,
    K: int = 4,
) -> tuple[bool, float, list[int]]:
    """域感知贪心路由。

    Route a flow using greedy per-domain strategy.
    在每个域边界，选择当前利用率最低的域间链路。
    域内使用最短路径 (K=1)。

    Returns
    -------
    tuple[bool, float, list[int]]
        (success, delay_seconds, satellite_path)。
    """
    domain_path = sim.get_domain_sequence(flow.src_sat, flow.dst_sat)
    full_path: list[int] = []
    total_delay = 0.0
    current_entry = flow.src_sat

    for i, domain_id in enumerate(domain_path):
        if i < len(domain_path) - 1:
            next_domain = domain_path[i + 1]
            links = sim.domain.get_inter_domain_links(domain_id, next_domain)
            # Pick the junction with lowest utilization
            best_idx = 0
            best_util = float("inf")
            for idx, (sa, sb) in enumerate(links):
                u = sim.bandwidth.get_utilization(sa, sb)
                if u < best_util:
                    best_util = u
                    best_idx = idx
            exit_sat, entry_next = links[best_idx]
        else:
            exit_sat = flow.dst_sat
            entry_next = None

        # Shortest path within domain
        paths = sim.compute_k_shortest(domain_id, current_entry, exit_sat, K=1)
        if not paths:
            return False, float("inf"), []

        seg_path = paths[0]
        ok, seg_delay = sim.try_allocate_segment(seg_path, flow.bandwidth, flow.flow_id)
        if not ok:
            return False, float("inf"), []

        if full_path and seg_path and full_path[-1] == seg_path[0]:
            full_path.extend(seg_path[1:])
        else:
            full_path.extend(seg_path)
        total_delay += seg_delay

        # Inter-domain link bandwidth + delay
        if entry_next is not None:
            inter_ok = sim.bandwidth.allocate(
                [exit_sat, entry_next], flow.bandwidth, flow.flow_id)
            if not inter_ok:
                return False, float("inf"), []
            total_delay += sim.topology.get_link_delay(
                exit_sat, entry_next, sim.current_timeslot)
            current_entry = entry_next

    return True, total_delay, full_path


# ==================================================================
#  Random baseline
# ==================================================================
def route_flow_random(
    sim: NetworkSimEngine,
    flow: FlowRequest,
    K: int = 4,
    rng: np.random.Generator | None = None,
) -> tuple[bool, float, list[int]]:
    """随机路由 (下界参考)。

    Route a flow with random junction selection + random K-path choice.
    域间链路随机选择 + 域内路径随机选择。
    HRL 应显著优于此基线。

    Returns
    -------
    tuple[bool, float, list[int]]
        (success, delay_seconds, satellite_path)。
    """
    if rng is None:
        rng = np.random.default_rng()

    domain_path = sim.get_domain_sequence(flow.src_sat, flow.dst_sat)
    full_path: list[int] = []
    total_delay = 0.0
    current_entry = flow.src_sat

    for i, domain_id in enumerate(domain_path):
        if i < len(domain_path) - 1:
            next_domain = domain_path[i + 1]
            links = sim.domain.get_inter_domain_links(domain_id, next_domain)
            idx = int(rng.integers(0, len(links)))
            exit_sat, entry_next = links[idx]
        else:
            exit_sat = flow.dst_sat
            entry_next = None

        paths = sim.compute_k_shortest(domain_id, current_entry, exit_sat, K=K)
        if not paths:
            return False, float("inf"), []

        sel = int(rng.integers(0, len(paths)))
        seg_path = paths[sel]
        ok, seg_delay = sim.try_allocate_segment(seg_path, flow.bandwidth, flow.flow_id)
        if not ok:
            return False, float("inf"), []

        if full_path and seg_path and full_path[-1] == seg_path[0]:
            full_path.extend(seg_path[1:])
        else:
            full_path.extend(seg_path)
        total_delay += seg_delay

        if entry_next is not None:
            inter_ok = sim.bandwidth.allocate(
                [exit_sat, entry_next], flow.bandwidth, flow.flow_id)
            if not inter_ok:
                return False, float("inf"), []
            total_delay += sim.topology.get_link_delay(
                exit_sat, entry_next, sim.current_timeslot)
            current_entry = entry_next

    return True, total_delay, full_path


# ==================================================================
#  Baseline episode runner
# ==================================================================
def run_baseline_episode(
    sim: NetworkSimEngine,
    strategy: str = "dijkstra",
    T_episode: int = 100,
    K: int = 4,
    rng: np.random.Generator | None = None,
) -> dict:
    """使用基线策略运行一个回合。

    Run an episode using a baseline strategy, collecting metrics.
    与 HRL 使用相同的流生命周期和带宽分配机制。

    Parameters
    ----------
    strategy : str
        ``"dijkstra"`` | ``"greedy"`` | ``"random"``。
    T_episode : int
        回合时隙数。
    K : int
        K 最短路候选数 (greedy/random 使用)。

    Returns
    -------
    dict
        包含 success_count, fail_count, delays, success_rate 等。
    """
    if rng is None:
        rng = np.random.default_rng()

    sim.reset()

    successes = 0
    failures = 0
    delays: list[float] = []

    for t in range(T_episode):
        sim.release_expired_flows()
        sim.advance_timeslot()
        new_flows = sim.generate_flows()

        for flow in new_flows:
            if strategy == "dijkstra":
                ok, delay, path = route_flow_dijkstra(sim, flow)
            elif strategy == "greedy":
                ok, delay, path = route_flow_greedy_domain(sim, flow, K)
            elif strategy == "random":
                ok, delay, path = route_flow_random(sim, flow, K, rng)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            if ok and path:
                sim.register_active_flow(flow, path, delay)
                delays.append(delay)
                successes += 1
            else:
                sim.record_failure()
                failures += 1

    total = successes + failures
    return {
        "strategy": strategy,
        "success_count": successes,
        "fail_count": failures,
        "success_rate": successes / max(total, 1),
        "total_delay": sum(delays),
        "avg_delay": np.mean(delays) if delays else float("inf"),
        "p95_delay": float(np.percentile(delays, 95)) if delays else float("inf"),
        "delays": delays,
    }
