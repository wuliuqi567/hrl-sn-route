"""
BandwidthManager — 每链路带宽资源管理。

职责 / Responsibilities:
  - 追踪每条 ISL 链路上已分配的带宽
  - 支持原子分配 (拒绝超容): allocate(path, bw, flow_id)
  - 支持流过期释放: release(flow_id)
  - 提供利用率/剩余带宽查询 (用于观测向量构建)

内部数据结构 / Internal Data:
  _used : dict[(min_id, max_id), float]
    每条链路已使用的带宽 (Mbps)。链路键为小编号在前，避免重复。
  _flow_allocs : dict[flow_id, list[(link_key, bw)]]
    每个流占用的链路及带宽，用于 release 时回滚。
    支持同一 flow_id 多次 allocate (append 而非覆盖)，
    用于跨域流分段分配。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.topology import Topology


class BandwidthManager:
    """Per-link bandwidth resource manager.

    Parameters
    ----------
    topology : Topology
        The +Grid topology instance (used to enumerate links).
    link_capacity_mbps : float
        Capacity of each ISL in Mbps (default 2000 ≈ 2 Gbps).
    """

    def __init__(self, topology: "Topology", link_capacity_mbps: float = 2000.0):
        self.topology = topology
        self.capacity = link_capacity_mbps

        # _used[(min_id, max_id)] = total allocated bandwidth (Mbps)
        self._used: dict[tuple[int, int], float] = {}

        # _flow_allocs[flow_id] = [(link_key, bw), ...] for rollback
        self._flow_allocs: dict[int, list[tuple[tuple[int, int], float]]] = {}

        # Initialise all links to 0 usage
        for sid, neighbors in topology.adj.items():
            for nb in neighbors:
                if nb != -1:
                    key = self._link_key(sid, nb)
                    if key not in self._used:
                        self._used[key] = 0.0

    # ------------------------------------------------------------------
    # Key helper
    # ------------------------------------------------------------------
    @staticmethod
    def _link_key(sat1: int, sat2: int) -> tuple[int, int]:
        return (min(sat1, sat2), max(sat1, sat2))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def check_feasibility(self, path: list[int], bw: float) -> bool:
        """Return True if *bw* Mbps can be allocated on every hop of *path*."""
        for i in range(len(path) - 1):
            key = self._link_key(path[i], path[i + 1])
            if self._used.get(key, 0.0) + bw > self.capacity:
                return False
        return True

    def allocate(self, path: list[int], bw: float, flow_id: int) -> bool:
        """Allocate *bw* Mbps on each hop of *path* for *flow_id*.

        Returns True on success, False if any link would exceed capacity
        (no partial allocation — atomic).
        """
        if not self.check_feasibility(path, bw):
            return False

        allocs: list[tuple[tuple[int, int], float]] = []
        for i in range(len(path) - 1):
            key = self._link_key(path[i], path[i + 1])
            self._used[key] = self._used.get(key, 0.0) + bw
            allocs.append((key, bw))

        # Append (not overwrite) — a cross-domain flow may allocate
        # multiple segments under the same flow_id.
        if flow_id in self._flow_allocs:
            self._flow_allocs[flow_id].extend(allocs)
        else:
            self._flow_allocs[flow_id] = allocs
        return True

    def release(self, flow_id: int) -> None:
        """Release all bandwidth previously allocated to *flow_id*."""
        allocs = self._flow_allocs.pop(flow_id, [])
        for key, bw in allocs:
            self._used[key] = max(0.0, self._used.get(key, 0.0) - bw)

    def get_remaining_bw(self, sat1: int, sat2: int) -> float:
        """Remaining bandwidth (Mbps) on the link between sat1 and sat2."""
        key = self._link_key(sat1, sat2)
        return self.capacity - self._used.get(key, 0.0)

    def get_utilization(self, sat1: int, sat2: int) -> float:
        """Link utilization in [0, 1]."""
        key = self._link_key(sat1, sat2)
        return self._used.get(key, 0.0) / self.capacity

    def get_domain_avg_utilization(self, domain_sats: list[int]) -> float:
        """Average utilization of all links whose both endpoints are in *domain_sats*."""
        sat_set = set(domain_sats)
        total = 0.0
        count = 0
        topo = self.topology
        for sid in domain_sats:
            for nb in topo.adj[sid]:
                if nb != -1 and nb in sat_set and nb > sid:  # avoid double count
                    total += self.get_utilization(sid, nb)
                    count += 1
        return total / max(count, 1)

    def get_link_utilizations_array(self, link_list: list[tuple[int, int]]) -> np.ndarray:
        """Return utilizations for a specific list of links as a numpy array."""
        return np.array([self.get_utilization(a, b) for a, b in link_list],
                        dtype=np.float32)

    def reset(self) -> None:
        """Clear all allocations (start of new episode)."""
        for key in self._used:
            self._used[key] = 0.0
        self._flow_allocs.clear()
