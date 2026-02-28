"""
NetworkSimEngine — 中心网络仿真引擎。

本模块是 HRL 卫星网络路由系统的核心状态机，**不是** Gym 环境。
训练循环 (rollout.py) 通过调用其公开接口来驱动仿真。

它拥有并管理所有核心模块:
  Topology       → +Grid 拓扑、卫星位置、链路时延
  DomainManager  → 域划分、域间链路、域路径
  BandwidthManager → 链路带宽资源管理
  FlowManager    → 活跃流生命周期管理
  FlowGenerator  → Poisson 流生成器

公开接口分类 / Public API Categories:
  回合生命周期: reset(), advance_timeslot(), release_expired_flows(), generate_flows()
  域查询: get_domain(), get_domain_sequence(), junction_sat_from_action()
  K最短路: compute_k_shortest()
  带宽分配: try_allocate_segment()
  流注册: register_active_flow(), record_failure()
  观测构建: get_upper_obs(), get_lower_kpath_obs(), get_lower_hop_obs()
"""

from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np

from core.topology import Topology
from core.domain import DomainManager
from core.bandwidth import BandwidthManager
from core.flow import FlowRequest, FlowManager
from core.flow_generator import FlowGenerator


class NetworkSimEngine:
    """中心网络仿真引擎。

    Central network simulation engine.

    初始化流程 / Initialization:
      1. 加载 StarPerf XML 星座 (约 30-60s, 会被缓存到 HDF5)
      2. 构建 +Grid 拓扑 (Topology)
      3. 域划分 (DomainManager)
      4. 初始化带宽管理器 (BandwidthManager)
      5. 初始化流管理器 + 流生成器

    Parameters
    ----------
    constellation_name : str
        StarPerf XML 星座名称 (e.g., "Starlink", "Kuiper", "OneWeb")。
    shell_idx : int
        轨道壳层索引 (0-based)。
    dT : int
        时隙间隔 (秒)，决定时间粒度。
    n_domains : int
        域数量，用于分层管理。
    domain_mode : str
        域划分插件名称 (e.g., "by_orbit_group")。
    link_capacity_mbps : float
        每条 ISL 链路容量 (Mbps)。
    arrival_rate : float
        Poisson 流到达率 (每时隙平均流数)。
    bw_range : tuple[float, float]
        流带宽需求范围 (Mbps)。
    delay_range : tuple[float, float]
        流时延预算范围 (秒)。
    duration_range : tuple[int, int]
        流持续时间范围 (时隙数)。
    bw_weight_alpha : float
        K最短路边权重中的带宽惩罚系数。
    seed : int | None
        随机种子。
    """

    def __init__(
        self,
        constellation_name: str = "Starlink",
        shell_idx: int = 0,
        dT: int = 60,
        n_domains: int = 6,
        domain_mode: str = "by_orbit_group",
        link_capacity_mbps: float = 2000.0,
        arrival_rate: float = 5.0,
        bw_range: tuple[float, float] = (10.0, 200.0),
        delay_range: tuple[float, float] = (0.02, 0.10),
        duration_range: tuple[int, int] = (5, 30),
        bw_weight_alpha: float = 1.0,
        seed: int | None = None,
    ):
        self.rng = np.random.default_rng(seed)
        self.bw_weight_alpha = bw_weight_alpha

        # ── Core modules ──
        print("[SimEngine] Initialising topology …")
        self.topology = Topology(constellation_name, shell_idx, dT)

        print("[SimEngine] Partitioning into domains …")
        self.domain = DomainManager(self.topology.shell, n_domains, domain_mode)

        print("[SimEngine] Creating bandwidth manager …")
        self.bandwidth = BandwidthManager(self.topology, link_capacity_mbps)

        print("[SimEngine] Creating flow manager …")
        self.flow_manager = FlowManager(self.bandwidth)

        print("[SimEngine] Creating flow generator …")
        self.flow_generator = FlowGenerator(
            self.topology, self.flow_manager,
            arrival_rate=arrival_rate,
            bw_range=bw_range,
            delay_range=delay_range,
            duration_range=duration_range,
            rng=self.rng,
        )

        # Convenience aliases
        self.n_domains = n_domains
        self.sats_per_orbit = self.topology.sats_per_orbit
        self.num_sats = self.topology.num_sats
        self.num_timeslots = self.topology.num_timeslots

        # ── Episode state ──
        self.current_timeslot: int = 1

        # ── Statistics ──
        self.stats_success: int = 0
        self.stats_fail: int = 0

        print(f"[SimEngine] Ready: {self.num_sats} sats, {n_domains} domains, "
              f"{self.num_timeslots} timeslots")

    # ==================================================================
    #  回合生命周期 (Episode Lifecycle)
    # ==================================================================
    def reset(self, start_timeslot: int = 1) -> None:
        """重置回合状态 — 清空带宽、流、统计计数器。

        Reset episode state. Called at the start of each training episode.
        """
        self.current_timeslot = start_timeslot
        self.bandwidth.reset()
        self.flow_manager.reset()
        self.stats_success = 0
        self.stats_fail = 0

    def advance_timeslot(self) -> None:
        """推进仿真时钟一个时隙。

        超过轨道周期时回绕到 1。卫星位置和链路时延随时隙变化。
        """
        self.current_timeslot += 1
        if self.current_timeslot > self.num_timeslots:
            self.current_timeslot = 1

    def release_expired_flows(self) -> list[int]:
        """Release flows whose duration has elapsed.  Returns expired ids."""
        return self.flow_manager.tick()

    def generate_flows(self) -> list[FlowRequest]:
        """Generate new flow requests for the current timeslot."""
        return self.flow_generator.generate(self.current_timeslot)

    # ==================================================================
    #  Domain queries (delegated)
    # ==================================================================
    def get_domain(self, sat_id: int) -> int:
        return self.domain.get_domain(sat_id)

    def get_domain_sequence(self, src_sat: int, dst_sat: int) -> list[int]:
        """Compute domain-level shortest path between the domains of src/dst."""
        d_src = self.domain.get_domain(src_sat)
        d_dst = self.domain.get_domain(dst_sat)
        return self.domain.get_domain_path(d_src, d_dst)

    def junction_sat_from_action(self, d_from: int, d_to: int,
                                 action_idx: int) -> tuple[int, int]:
        """(exit_sat_in_d_from, entry_sat_in_d_to)"""
        return self.domain.junction_sat_from_action(d_from, d_to, action_idx)

    def get_peer_sat(self, exit_sat: int, next_domain: int) -> int:
        """Given an exit satellite, return the entry satellite in next_domain
        that is connected to it via the inter-orbit ISL."""
        for nb in self.topology.adj[exit_sat]:
            if nb != -1 and self.domain.get_domain(nb) == next_domain:
                return nb
        raise ValueError(f"exit_sat {exit_sat} has no neighbor in domain {next_domain}")

    # ==================================================================
    #  K-shortest-path computation
    # ==================================================================
    def compute_k_shortest(self, domain_id: int,
                           entry_sat: int, exit_sat: int,
                           K: int = 4) -> list[list[int]]:
        """Compute K diverse shortest paths within *domain_id* using
        bandwidth-aware edge weights.

        Uses iterative edge-penalty Dijkstra (much faster than Yen's
        algorithm on +Grid topologies with hundreds of nodes).

        Returns up to K paths (may be fewer).
        """
        sat_ids = self.domain.get_domain_sats(domain_id)
        G = self.topology.build_nx_subgraph(
            sat_ids, self.current_timeslot,
            bw_manager=self.bandwidth,
            bw_weight_alpha=self.bw_weight_alpha,
        )
        if entry_sat not in G or exit_sat not in G:
            return []

        paths: list[list[int]] = []
        penalty_factor = 2.0  # multiply edge weight after each use

        for _ in range(K):
            try:
                p = nx.dijkstra_path(G, entry_sat, exit_sat, weight="weight")
            except nx.NetworkXNoPath:
                break
            paths.append(p)
            # Penalise edges on this path to encourage diversity
            for j in range(len(p) - 1):
                u, v = p[j], p[j + 1]
                if G.has_edge(u, v):
                    G[u][v]["weight"] *= penalty_factor
        return paths

    # ==================================================================
    #  Segment allocation
    # ==================================================================
    def try_allocate_segment(self, path: list[int], bw: float,
                             flow_id: int) -> tuple[bool, float]:
        """Try to allocate bandwidth on *path*.

        Returns (success, total_delay_seconds).
        """
        delay = sum(
            self.topology.get_link_delay(path[i], path[i + 1], self.current_timeslot)
            for i in range(len(path) - 1)
        )
        ok = self.bandwidth.allocate(path, bw, flow_id)
        return ok, delay

    def register_active_flow(self, request: FlowRequest, path: list[int],
                             delay: float) -> None:
        """Register a successfully-routed flow."""
        self.flow_manager.register(request, path, delay)
        self.stats_success += 1

    def record_failure(self) -> None:
        """Record a routing failure (for stats)."""
        self.stats_fail += 1

    # ==================================================================
    #  Observation builders
    # ==================================================================
    def get_upper_obs(self, d_from: int, d_to: int,
                      flow: FlowRequest,
                      domain_path: list[int],
                      hop_index: int) -> np.ndarray:
        """Build the upper-agent observation vector.

        Observation layout (≈77 dims for 6 domains):
          [22 inter-domain links × 3 features]   = 66
          + [bw_demand, delay_budget, remaining_hops,
             src_domain/n_dom, dst_domain/n_dom]  =  5
          + [n_domains × avg_utilization]          =  n_domains

        Features per inter-domain link:
          utilization, delay, peer-domain avg utilization
        """
        links = self.domain.get_inter_domain_links(d_from, d_to)

        link_features = []
        for (sa, sb) in links:
            util = self.bandwidth.get_utilization(sa, sb)
            delay = self.topology.get_link_delay(sa, sb, self.current_timeslot)
            # Peer domain average utilization
            peer_domain_sats = self.domain.get_domain_sats(d_to)
            peer_util = self.bandwidth.get_domain_avg_utilization(peer_domain_sats)
            link_features.extend([util, delay * 100.0, peer_util])  # scale delay

        # Flow features
        remaining_hops = len(domain_path) - 1 - hop_index
        d_src = self.get_domain(flow.src_sat)
        d_dst = self.get_domain(flow.dst_sat)
        flow_features = [
            flow.bandwidth / 200.0,      # normalise ~[0, 1]
            flow.max_delay * 10.0,        # normalise ~[0, 1]
            remaining_hops / max(self.n_domains - 1, 1),
            d_src / max(self.n_domains - 1, 1),
            d_dst / max(self.n_domains - 1, 1),
        ]

        # Global per-domain average utilisation
        global_utils = []
        for d in range(self.n_domains):
            sats = self.domain.get_domain_sats(d)
            global_utils.append(self.bandwidth.get_domain_avg_utilization(sats))

        obs = np.array(link_features + flow_features + global_utils,
                       dtype=np.float32)
        return obs

    def get_upper_obs_dim(self) -> int:
        """Observation dimension for the upper agent."""
        return self.sats_per_orbit * 3 + 5 + self.n_domains

    def get_lower_kpath_obs(self, domain_id: int,
                            k_paths: list[list[int]],
                            flow: FlowRequest,
                            entry_sat: int,
                            exit_sat: int,
                            K: int = 4) -> np.ndarray:
        """Build the lower-agent K-path observation vector.

        Layout (≈22 dims for K=4):
          [K paths × 4 features]   = 16
          + [bw_demand, delay_budget,
             entry_lon, entry_lat, exit_lon, exit_lat] = 6

        Features per path:
          total_delay, min_remaining_bw, hop_count, bottleneck_utilisation
        """
        t = self.current_timeslot
        path_features: list[float] = []

        for pi in range(K):
            if pi < len(k_paths):
                p = k_paths[pi]
                total_delay = sum(
                    self.topology.get_link_delay(p[j], p[j + 1], t)
                    for j in range(len(p) - 1)
                )
                min_bw = min(
                    (self.bandwidth.get_remaining_bw(p[j], p[j + 1])
                     for j in range(len(p) - 1)),
                    default=self.bandwidth.capacity,
                )
                bottleneck_util = max(
                    (self.bandwidth.get_utilization(p[j], p[j + 1])
                     for j in range(len(p) - 1)),
                    default=0.0,
                )
                hops = len(p) - 1
                path_features.extend([
                    total_delay * 100.0,             # scale delay
                    min_bw / self.bandwidth.capacity, # normalised
                    hops / 30.0,                     # normalise (~max hops)
                    bottleneck_util,
                ])
            else:
                # Pad missing paths with worst-case values
                path_features.extend([1.0, 0.0, 1.0, 1.0])

        # Sub-task features
        e_lon, e_lat, _ = self.topology.get_sat_position(entry_sat, t)
        x_lon, x_lat, _ = self.topology.get_sat_position(exit_sat, t)
        sub_features = [
            flow.bandwidth / 200.0,
            flow.max_delay * 10.0,
            e_lon / 180.0, e_lat / 90.0,
            x_lon / 180.0, x_lat / 90.0,
        ]

        obs = np.array(path_features + sub_features, dtype=np.float32)
        return obs

    def get_lower_kpath_obs_dim(self, K: int = 4) -> int:
        """Observation dimension for the lower K-path agent."""
        return K * 4 + 6

    def get_lower_hop_obs(self, current_sat: int, exit_sat: int,
                          domain_id: int,
                          accumulated_delay: float,
                          step: int, max_steps: int,
                          flow: FlowRequest) -> np.ndarray:
        """Build the lower-agent hop-by-hop observation vector.

        Layout (≈23 dims):
          [cur_lon, cur_lat, cur_alt]           = 3
          + [exit_lon, exit_lat, exit_alt]       = 3
          + [dist_to_exit]                       = 1
          + [accumulated_delay]                  = 1
          + [step_progress]                      = 1
          + [bw_demand]                          = 1
          + [remaining_delay_budget]             = 1
          + [4 neighbors × 3 features]           = 12
                                              total 23

        Neighbor features:
          [distance_to_exit, remaining_bw, in_domain_flag]
        """
        t = self.current_timeslot
        domain_sats = set(self.domain.get_domain_sats(domain_id))

        c_lon, c_lat, c_alt = self.topology.get_sat_position(current_sat, t)
        e_lon, e_lat, e_alt = self.topology.get_sat_position(exit_sat, t)
        dist = self.topology.sat_distance_km(current_sat, exit_sat, t)

        obs_parts: list[float] = [
            c_lon / 180.0, c_lat / 90.0, c_alt / 1000.0,
            e_lon / 180.0, e_lat / 90.0, e_alt / 1000.0,
            min(dist / 20000.0, 1.0),
            min(accumulated_delay * 100.0, 1.0),
            step / max(max_steps, 1),
            flow.bandwidth / 200.0,
            max(0.0, (flow.max_delay - accumulated_delay)) * 10.0,
        ]

        # 4 neighbors
        neighbors = self.topology.adj[current_sat]
        for nb in neighbors:
            if nb == -1:
                obs_parts.extend([1.0, 0.0, 0.0])
            else:
                nb_dist = self.topology.sat_distance_km(nb, exit_sat, t)
                nb_bw = self.bandwidth.get_remaining_bw(current_sat, nb)
                in_domain = 1.0 if nb in domain_sats else 0.0
                obs_parts.extend([
                    min(nb_dist / 20000.0, 1.0),
                    nb_bw / self.bandwidth.capacity,
                    in_domain,
                ])

        return np.array(obs_parts, dtype=np.float32)

    def get_lower_hop_obs_dim(self) -> int:
        """Observation dimension for the lower hop-by-hop agent."""
        return 23
