"""
Topology — 封装 StarPerf 星座，提供 +Grid 拓扑查询接口。

职责 / Responsibilities:
  - 加载 StarPerf XML 星座数据 (轨道、卫星、位置时间序列)
  - 构建 +Grid 邻接字典: adj[sat_id] = [up, down, left, right]
    • up/down: 轨内链路 (intra-orbit ISL), 环形连接
    • left/right: 轨间链路 (inter-orbit ISL), 极轨不连端
  - 提供时变查询: 卫星位置、链路时延、卫星间距离
  - 构建 NetworkX 子图 (K最短路 / Dijkstra 使用)

数据结构 / Data Structures:
  adj : dict[int, list[int]]
    sat_id → [up, down, left, right]。
    -1 表示无链路 (极轨星座边缘轨道)。
  _sat_by_id : dict[int, Satellite]
    sat_id → StarPerf 卫星对象 (包含经纬度/高度时间序列)。

sat_id 编号规则:
  sat_id = (orbit_index - 1) * sats_per_orbit + sat_index
  orbit_index ∈ [1, num_orbits], sat_index ∈ [1, sats_per_orbit]
  即 1-based 连续编号。
"""

from __future__ import annotations

import os
from math import radians, cos, sin, asin, sqrt
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

_STARPERF_ROOT = str(Path(__file__).resolve().parent.parent / "StarPerf_Simulator")


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float,
                  radius: float = 6371.0) -> float:
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * asin(sqrt(a)) * radius


class Topology:
    """封装 StarPerf 星座 shell，提供 +Grid 拓扑查询。

    Wraps a StarPerf constellation shell and exposes +Grid topology queries.

    初始化流程:
      1. 切换到 StarPerf 目录 → 加载 XML 星座配置
      2. 建立 +Grid 连接 (positive_Grid 插件)
      3. 构建 adj 邻接字典

    Parameters
    ----------
    constellation_name : str
        XML 星座名称, e.g. "Starlink", "Kuiper"。
    shell_idx : int
        0-based 轨道壳层索引。
    dT : int
        时隙间隔 (秒)。
    """

    def __init__(self, constellation_name: str = "Starlink",
                 shell_idx: int = 0, dT: int = 60):
        self.constellation_name = constellation_name
        self.shell_idx = shell_idx
        self.dT = dT

        # ---------- Load constellation via StarPerf ----------
        original_cwd = os.getcwd()
        os.chdir(_STARPERF_ROOT)
        try:
            from src.constellation_generation.by_XML.constellation_configuration import (
                constellation_configuration,
            )
            from src.XML_constellation.constellation_connectivity.connectivity_mode_plugin_manager import (
                connectivity_mode_plugin_manager,
            )

            self.constellation = constellation_configuration(dT, constellation_name)
            conn_mgr = connectivity_mode_plugin_manager()
            conn_mgr.set_connection_mode("positive_Grid")
            conn_mgr.execute_connection_policy(self.constellation, dT)
        finally:
            os.chdir(original_cwd)

        self.shell = self.constellation.shells[shell_idx]
        self.num_sats: int = self.shell.number_of_satellites
        self.num_orbits: int = self.shell.number_of_orbits
        self.sats_per_orbit: int = self.shell.number_of_satellite_per_orbit
        self.num_timeslots: int = int(self.shell.orbit_cycle / dT) + 1
        self.inclination: float = self.shell.inclination

        # sat_id → satellite object (fast lookup)
        self._sat_by_id: dict = {}
        for orbit in self.shell.orbits:
            for sat in orbit.satellites:
                self._sat_by_id[sat.id] = sat

        # ---------- Build +Grid adjacency ----------
        self.adj: dict[int, list[int]] = {}  # sat_id → [up, down, left, right]
        self._build_adjacency()

    # ------------------------------------------------------------------
    # +Grid 邻接关系构建 (Adjacency Construction)
    # ------------------------------------------------------------------
    def _build_adjacency(self):
        """构建 +Grid 邻接字典。

        +Grid 拓扑规则:
          - 轨内链路 (up/down): 同一轨道内的相邻卫星，环形连接
          - 轨间链路 (left/right): 相邻轨道同编号卫星连接
          - 极轨星座 (inclination ≈ 90°): 第一/最后轨道无左/右链路 (-1)
          - 非极轨星座: 第一和最后轨道环形连接
        """
        n_orbits = self.num_orbits
        n_per = self.sats_per_orbit
        is_polar = 80 < self.inclination < 100

        for oi in range(1, n_orbits + 1):
            for si in range(1, n_per + 1):
                sid = (oi - 1) * n_per + si

                # intra-orbit: up / down (ring within orbit)
                up = sid + 1 if si < n_per else (oi - 1) * n_per + 1
                down = sid - 1 if si > 1 else oi * n_per

                # inter-orbit: left / right
                if oi > 1:
                    left = (oi - 2) * n_per + si
                elif not is_polar:
                    left = (n_orbits - 1) * n_per + si
                else:
                    left = -1

                if oi < n_orbits:
                    right = oi * n_per + si
                elif not is_polar:
                    right = si
                else:
                    right = -1

                self.adj[sid] = [up, down, left, right]

    # ------------------------------------------------------------------
    # 时变查询 (Time-varying Queries)
    # 卫星位置和链路时延随时隙变化，因为卫星在轨道上运动。
    # ------------------------------------------------------------------
    def get_sat_position(self, sat_id: int, timeslot: int) -> tuple[float, float, float]:
        """Return (longitude, latitude, altitude_km) at *timeslot* (1-based)."""
        sat = self._sat_by_id[sat_id]
        idx = timeslot - 1
        return sat.longitude[idx], sat.latitude[idx], sat.altitude[idx]

    def get_link_delay(self, sat1: int, sat2: int, timeslot: int) -> float:
        """Propagation delay in seconds between two adjacent satellites."""
        lon1, lat1, alt1 = self.get_sat_position(sat1, timeslot)
        lon2, lat2, alt2 = self.get_sat_position(sat2, timeslot)
        avg_r = 6371.0 + (alt1 + alt2) / 2.0
        dist = _haversine_km(lon1, lat1, lon2, lat2, radius=avg_r)
        return dist / 300_000.0  # speed of light

    def sat_distance_km(self, sat1: int, sat2: int, timeslot: int) -> float:
        """Great-circle distance (km) between two satellites at *timeslot*."""
        lon1, lat1, alt1 = self.get_sat_position(sat1, timeslot)
        lon2, lat2, alt2 = self.get_sat_position(sat2, timeslot)
        avg_r = 6371.0 + (alt1 + alt2) / 2.0
        return _haversine_km(lon1, lat1, lon2, lat2, radius=avg_r)

    # ------------------------------------------------------------------
    # NetworkX graph builders
    # ------------------------------------------------------------------
    def build_nx_graph(self, timeslot: int, *,
                       bw_manager=None, bw_weight_alpha: float = 0.0) -> nx.Graph:
        """Build a full-shell NetworkX graph with delay-based edge weights.

        Parameters
        ----------
        timeslot : int
            1-based timeslot index.
        bw_manager : BandwidthManager, optional
            If provided and *bw_weight_alpha* > 0, edge weight includes a
            bandwidth-aware penalty: ``w = delay + alpha / remaining_bw``.
        bw_weight_alpha : float
            Bandwidth penalty coefficient.
        """
        G = nx.Graph()
        for sid in self._sat_by_id:
            G.add_node(sid)
        for sid, neighbors in self.adj.items():
            for nb in neighbors:
                if nb != -1 and not G.has_edge(sid, nb):
                    delay = self.get_link_delay(sid, nb, timeslot)
                    w = delay
                    if bw_manager is not None and bw_weight_alpha > 0:
                        rem = bw_manager.get_remaining_bw(sid, nb)
                        w += bw_weight_alpha / max(rem, 1.0)
                    G.add_edge(sid, nb, weight=w, delay=delay)
        return G

    def build_nx_subgraph(self, sat_ids, timeslot: int, *,
                          bw_manager=None, bw_weight_alpha: float = 0.0) -> nx.Graph:
        """Build a subgraph restricted to *sat_ids*.

        Parameters are the same as :meth:`build_nx_graph`.
        """
        sat_set = set(sat_ids)
        G = nx.Graph()
        for sid in sat_set:
            G.add_node(sid)
        for sid in sat_set:
            for nb in self.adj[sid]:
                if nb != -1 and nb in sat_set and not G.has_edge(sid, nb):
                    delay = self.get_link_delay(sid, nb, timeslot)
                    w = delay
                    if bw_manager is not None and bw_weight_alpha > 0:
                        rem = bw_manager.get_remaining_bw(sid, nb)
                        w += bw_weight_alpha / max(rem, 1.0)
                    G.add_edge(sid, nb, weight=w, delay=delay)
        return G

    def nearest_sat(self, lon: float, lat: float, timeslot: int) -> int:
        """Return the sat_id closest to ground point (lon, lat)."""
        best_id = -1
        best_d = float("inf")
        for sid, sat in self._sat_by_id.items():
            d = _haversine_km(lon, lat, sat.longitude[timeslot - 1],
                              sat.latitude[timeslot - 1])
            if d < best_d:
                best_d = d
                best_id = sid
        return best_id
