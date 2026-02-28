"""
DomainManager — 封装 StarPerf 域划分插件，提供域级查询。

职责 / Responsibilities:
  - 调用域划分插件 (e.g., by_orbit_group) 获取 DomainInfo
  - 提供 sat→domain 映射 和 domain→sats 列表
  - 计算域级最短路径 (域图为环形/链形拓扑)
  - 将上层智能体的离散动作映射到具体的域间卫星对

域划分逻辑 (by_orbit_group):
  将星座的 N 个轨道均匀分成 n_domains 组，
  每组包含连续的 N/n_domains 个轨道及其上的所有卫星。
  域间链路为相邻域边界轨道上的 sats_per_orbit 条 ISL。
"""

from __future__ import annotations

from typing import Optional

import networkx as nx


class DomainManager:
    """封装 StarPerf 域划分插件结果。

    Wraps a StarPerf domain partition plugin result.

    Parameters
    ----------
    shell : StarPerf shell object
        待划分的 shell 对象。
    n_domains : int
        域数量。
    mode : str
        划分插件名称 (默认 ``"by_orbit_group"``)。
    """

    def __init__(self, shell, n_domains: int, mode: str = "by_orbit_group"):
        from src.XML_constellation.constellation_domain.domain_partition_plugin_manager import (
            domain_partition_plugin_manager,
        )
        mgr = domain_partition_plugin_manager()
        mgr.set_partition_mode(mode)
        self.info = mgr.execute_partition(shell, n_domains)

        self.n_domains = self.info.n_domains
        self.sats_per_orbit = self.info.sats_per_orbit

        # Build a domain-level NetworkX graph for shortest-path queries
        self._domain_graph = nx.Graph()
        for d in range(self.n_domains):
            self._domain_graph.add_node(d)
        for (da, db) in self.info.inter_domain_links:
            self._domain_graph.add_edge(da, db)

    # ------------------------------------------------------------------
    # Basic lookups
    # ------------------------------------------------------------------
    def get_domain(self, sat_id: int) -> int:
        """Return the domain id that *sat_id* belongs to."""
        return self.info.domain_of_sat[sat_id]

    def get_domain_sats(self, domain_id: int) -> list[int]:
        """Return all satellite ids in *domain_id*."""
        return self.info.domain_sats[domain_id]

    def get_domain_orbits(self, domain_id: int) -> list[int]:
        """Return 1-based orbit indices belonging to *domain_id*."""
        return self.info.domain_orbits[domain_id]

    # ------------------------------------------------------------------
    # Inter-domain link queries
    # ------------------------------------------------------------------
    def get_inter_domain_links(self, d1: int, d2: int) -> list[tuple[int, int]]:
        """Return list of ``(sat_in_d1, sat_in_d2)`` ISL pairs.

        The links are stored as ``(d_smaller, d_larger)`` in the ring direction.
        This method handles lookup in both directions.
        """
        if (d1, d2) in self.info.inter_domain_links:
            return self.info.inter_domain_links[(d1, d2)]
        elif (d2, d1) in self.info.inter_domain_links:
            return [(b, a) for a, b in self.info.inter_domain_links[(d2, d1)]]
        else:
            raise ValueError(f"No inter-domain links between domain {d1} and {d2}")

    def junction_sat_from_action(self, d_from: int, d_to: int,
                                 action_idx: int) -> tuple[int, int]:
        """Map an upper-agent action index to a concrete junction satellite pair.

        Parameters
        ----------
        d_from, d_to : int
            Adjacent domain ids.
        action_idx : int
            Index in ``[0, sats_per_orbit)``.

        Returns
        -------
        (exit_sat, entry_sat) : tuple[int, int]
            exit_sat is in d_from, entry_sat is in d_to.
        """
        links = self.get_inter_domain_links(d_from, d_to)
        idx = min(action_idx, len(links) - 1)
        return links[idx]

    # ------------------------------------------------------------------
    # Domain-level path queries
    # ------------------------------------------------------------------
    def get_domain_path(self, src_domain: int, dst_domain: int) -> list[int]:
        """Shortest domain sequence from src_domain to dst_domain.

        For a ring topology (Walker-Delta) the shortest path is at most
        ``n_domains // 2`` hops. Returns list including both endpoints,
        e.g. ``[0, 1, 2, 3]``.
        """
        if src_domain == dst_domain:
            return [src_domain]
        return nx.shortest_path(self._domain_graph, src_domain, dst_domain)

    def domain_distance(self, d1: int, d2: int) -> int:
        """Number of domain hops between d1 and d2."""
        if d1 == d2:
            return 0
        return nx.shortest_path_length(self._domain_graph, d1, d2)
