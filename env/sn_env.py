"""
SatelliteNetworkEnv — 基于 StarPerf 的卫星网络路由强化学习环境

兼容 Gymnasium / Stable-Baselines3 接口。

RL 任务：给定源地面站和目标地面站，智能体在卫星网络拓扑上逐跳选择
         下一跳卫星，目标是以最小时延到达距目标地面站最近的卫星。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import networkx as nx
from gymnasium import spaces
from math import radians, cos, sin, asin, sqrt

# ---------------------------------------------------------------------------
# StarPerf 工程根目录（用于切换工作目录，兼容其相对路径读取配置/数据）
# ---------------------------------------------------------------------------
_STARPERF_ROOT = str(Path(__file__).resolve().parent.parent / "StarPerf_Simulator")


# ========================== 工具函数 ==========================

def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float,
                  radius: float = 6371.0) -> float:
    """地面 Haversine 距离 (km)，默认地球半径 6371 km。"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * asin(sqrt(a)) * radius


def _sat_distance_km(lon1, lat1, alt1, lon2, lat2, alt2) -> float:
    """考虑卫星轨道高度的距离 (km)。"""
    avg_alt = (alt1 + alt2) / 2.0
    return _haversine_km(lon1, lat1, lon2, lat2, radius=6371.0 + avg_alt)


# ========================== 环境定义 ==========================

class SatelliteNetworkEnv(gym.Env):
    """
    卫星网络逐跳路由 RL 环境（Gymnasium 接口）。

    ── 核心思路 ──
    • 基于 StarPerf 的 XML 星座生成和 +Grid ISL 拓扑。
    • 每个 episode: 随机选择 (源, 目标) 地面站对和时间片，
      智能体从源卫星出发，每步选择一个邻居作为下一跳，直到到达目标卫星或超时。
    • 奖励信号：跳转时延的负值（鼓励低延迟路径），到达目标给予正奖励。

    ── 状态空间 ──
    观测向量 (连续, Box):
        [当前卫星 lon, lat, alt,          (3)
         目标卫星 lon, lat, alt,          (3)
         到目标的归一化距离,              (1)
         已累计时延 (归一化),             (1)
         已走步数 / 最大步数,             (1)
         当前卫星的 4 个邻居到目标的归一化距离  (4)]
                                      共 13 维

    ── 动作空间 ──
    Discrete(4): 对应 +Grid 拓扑中最多 4 条 ISL（同轨前/后 + 邻轨左/右）。
                 如果某个 slot 不存在（极轨缝隙），该动作映射为"原地不动"并给惩罚。

    Parameters
    ----------
    constellation_name : str
        XML 星座名称，如 "Starlink", "Kuiper"。
    shell_idx : int
        使用的 shell 索引 (0-based)，默认 0（第一层壳）。
    dT : int
        时间片间隔 (秒)，默认 60。
    max_steps : int
        每个 episode 最大步数，默认 50。
    gs_pairs : list[tuple[tuple, tuple]] | None
        地面站 (源, 目标) 经纬度对列表。若为 None 则使用默认城市对。
    seed : int | None
        随机种子。
    """

    metadata = {"render_modes": ["human"]}

    # ─── 默认地面站对 (经度, 纬度) ────────────────────────
    DEFAULT_GS_PAIRS: list[tuple[tuple[float, float], tuple[float, float]]] = [
        # (北京, 纽约)
        ((116.4, 39.9), (-74.0, 40.7)),
        # (伦敦, 东京)
        ((-0.1, 51.5), (139.7, 35.7)),
        # (上海, 旧金山)
        ((121.5, 31.2), (-122.4, 37.8)),
        # (悉尼, 洛杉矶)
        ((151.2, -33.9), (-118.2, 34.1)),
        # (新加坡, 法兰克福)
        ((103.8, 1.4), (8.7, 50.1)),
        # (巴西利亚, 开普敦)
        ((-47.9, -15.8), (18.4, -33.9)),
        # (莫斯科, 孟买)
        ((37.6, 55.8), (72.9, 19.1)),
        # (迪拜, 首尔)
        ((55.3, 25.3), (127.0, 37.6)),
    ]

    def __init__(
        self,
        constellation_name: str = "Starlink",
        shell_idx: int = 0,
        dT: int = 60,
        max_steps: int = 50,
        gs_pairs: Optional[list] = None,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.constellation_name = constellation_name
        self.shell_idx = shell_idx
        self.dT = dT
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.gs_pairs = gs_pairs or self.DEFAULT_GS_PAIRS

        # ── 生成/加载星座 ──
        self._init_constellation()

        # ── 构建 +Grid 邻接表 (不随时间变化的拓扑结构) ──
        self._build_adjacency()

        # ── Gym spaces ──
        # 动作: 4 个方向 (同轨前, 同轨后, 邻轨左, 邻轨右)
        self.action_space = spaces.Discrete(4)

        # 观测: 13 维连续向量
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(13,), dtype=np.float32
        )

        # ── Episode 状态 ──
        self._current_sat_id: int = -1
        self._target_sat_id: int = -1
        self._source_gs: tuple = (0.0, 0.0)
        self._target_gs: tuple = (0.0, 0.0)
        self._current_timeslot: int = 1
        self._accumulated_delay: float = 0.0
        self._step_count: int = 0
        self._visited: set[int] = set()

        # 用于归一化
        self._max_possible_delay = 0.1  # ~30,000 km / 300,000 km/s
        self._earth_circumference = 40075.0  # km

    # ================================================================
    #                    初始化：加载星座
    # ================================================================
    def _init_constellation(self):
        """调用 StarPerf 生成星座并建立 +Grid ISL 连接。"""
        # 保存/恢复工作目录，因为 StarPerf 使用相对路径
        original_cwd = os.getcwd()
        os.chdir(_STARPERF_ROOT)

        try:
            from src.constellation_generation.by_XML.constellation_configuration import (
                constellation_configuration,
            )
            from src.XML_constellation.constellation_connectivity.connectivity_mode_plugin_manager import (
                connectivity_mode_plugin_manager,
            )

            print(f"[SatelliteNetworkEnv] 正在生成星座 '{self.constellation_name}' ...")
            self.constellation = constellation_configuration(self.dT, self.constellation_name)

            print("[SatelliteNetworkEnv] 正在建立 +Grid ISL 拓扑 ...")
            conn_manager = connectivity_mode_plugin_manager()
            conn_manager.set_connection_mode("positive_Grid")
            conn_manager.execute_connection_policy(self.constellation, self.dT)

            self.shell = self.constellation.shells[self.shell_idx]
            self.num_sats = self.shell.number_of_satellites
            self.num_orbits = self.shell.number_of_orbits
            self.sats_per_orbit = self.shell.number_of_satellite_per_orbit
            self.num_timeslots = int(self.shell.orbit_cycle / self.dT) + 1

            # 建立 id → satellite 对象的快速索引
            self._sat_by_id: dict = {}
            for orbit in self.shell.orbits:
                for sat in orbit.satellites:
                    self._sat_by_id[sat.id] = sat

            print(f"[SatelliteNetworkEnv] 星座就绪: {self.num_sats} 颗卫星, "
                  f"{self.num_orbits} 条轨道, {self.num_timeslots} 个时间片")
        finally:
            os.chdir(original_cwd)

    # ================================================================
    #                    构建 +Grid 邻接表
    # ================================================================
    def _build_adjacency(self):
        """
        为每颗卫星构建 4 方向邻居映射:
            adj[sat_id] = [up_id, down_id, left_id, right_id]
        不存在的邻居用 -1 表示。
        """
        n_orbits = self.num_orbits
        n_per_orbit = self.sats_per_orbit
        inclination = self.shell.inclination
        is_polar = 80 < inclination < 100

        self.adj: dict[int, list[int]] = {}

        for oi in range(1, n_orbits + 1):
            for si in range(1, n_per_orbit + 1):
                sat_id = (oi - 1) * n_per_orbit + si

                # ── 同轨: 前(up) / 后(down) ──
                up_id = sat_id + 1 if si < n_per_orbit else (oi - 1) * n_per_orbit + 1
                down_id = sat_id - 1 if si > 1 else oi * n_per_orbit

                # ── 邻轨: 左(left=上一轨道) / 右(right=下一轨道) ──
                if oi > 1:
                    left_id = (oi - 2) * n_per_orbit + si
                elif not is_polar:
                    left_id = (n_orbits - 1) * n_per_orbit + si  # 环绕
                else:
                    left_id = -1  # 极轨无环绕

                if oi < n_orbits:
                    right_id = oi * n_per_orbit + si
                elif not is_polar:
                    right_id = si  # 环绕
                else:
                    right_id = -1  # 极轨无环绕

                self.adj[sat_id] = [up_id, down_id, left_id, right_id]

    # ================================================================
    #                    Gym 核心接口
    # ================================================================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # 随机选择地面站对
        pair_idx = self.np_random.integers(0, len(self.gs_pairs))
        self._source_gs, self._target_gs = self.gs_pairs[pair_idx]

        # 随机选择时间片
        self._current_timeslot = int(self.np_random.integers(1, self.num_timeslots + 1))

        # 找到源/目标最近卫星
        self._current_sat_id = self._nearest_sat(*self._source_gs, self._current_timeslot)
        self._target_sat_id = self._nearest_sat(*self._target_gs, self._current_timeslot)

        # 确保源 ≠ 目标
        if self._current_sat_id == self._target_sat_id:
            # 用 Dijkstra 距离最远的卫星当目标
            self._target_sat_id = self._farthest_sat(self._current_sat_id)

        self._accumulated_delay = 0.0
        self._step_count = 0
        self._visited = {self._current_sat_id}

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        self._step_count += 1

        neighbors = self.adj[self._current_sat_id]
        next_sat_id = neighbors[action]

        reward = 0.0
        terminated = False
        truncated = False

        if next_sat_id == -1:
            # ── 无效动作 (极轨缝隙) ──
            reward = -0.5
        else:
            # ── 计算跳转时延 ──
            hop_delay = self._get_link_delay(self._current_sat_id, next_sat_id, self._current_timeslot)
            self._accumulated_delay += hop_delay

            # 每跳的时延惩罚 (归一化)
            reward = -hop_delay / self._max_possible_delay

            # 访问重复节点额外惩罚
            if next_sat_id in self._visited:
                reward -= 0.3

            self._visited.add(next_sat_id)
            self._current_sat_id = next_sat_id

            # ── 到达目标？ ──
            if self._current_sat_id == self._target_sat_id:
                terminated = True
                # 到达奖励 = 正比于效率 (步数越少越好)
                reward += 10.0 * (1.0 - self._step_count / self.max_steps)

        # ── 超时截断 ──
        if self._step_count >= self.max_steps and not terminated:
            truncated = True
            # 未到达的距离惩罚
            dist = self._sat_to_target_distance(self._current_sat_id)
            reward -= 2.0 * dist / self._earth_circumference

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    # ================================================================
    #                    观测构建
    # ================================================================
    def _get_obs(self) -> np.ndarray:
        t = self._current_timeslot
        cur = self._sat_by_id[self._current_sat_id]
        tgt = self._sat_by_id[self._target_sat_id]

        # 当前卫星位置 (归一化)
        cur_lon = cur.longitude[t - 1] / 180.0
        cur_lat = cur.latitude[t - 1] / 90.0
        cur_alt = cur.altitude[t - 1] / 1000.0  # ~0.5-0.6 范围

        # 目标卫星位置 (归一化)
        tgt_lon = tgt.longitude[t - 1] / 180.0
        tgt_lat = tgt.latitude[t - 1] / 90.0
        tgt_alt = tgt.altitude[t - 1] / 1000.0

        # 到目标归一化距离
        dist = self._sat_to_target_distance(self._current_sat_id)
        norm_dist = min(dist / (self._earth_circumference / 2), 1.0)

        # 累计时延 (归一化)
        norm_delay = min(self._accumulated_delay / self._max_possible_delay, 1.0)

        # 进度
        progress = self._step_count / self.max_steps

        # 4 个邻居到目标的距离 (归一化)
        neighbor_dists = []
        for nb_id in self.adj[self._current_sat_id]:
            if nb_id == -1:
                neighbor_dists.append(1.0)  # 不存在 → 最大距离
            else:
                d = self._sat_to_target_distance(nb_id)
                neighbor_dists.append(min(d / (self._earth_circumference / 2), 1.0))

        obs = np.array(
            [cur_lon, cur_lat, cur_alt,
             tgt_lon, tgt_lat, tgt_alt,
             norm_dist, norm_delay, progress]
            + neighbor_dists,
            dtype=np.float32,
        )
        return obs

    def _get_info(self) -> dict:
        return {
            "current_sat_id": self._current_sat_id,
            "target_sat_id": self._target_sat_id,
            "accumulated_delay_s": self._accumulated_delay,
            "step_count": self._step_count,
            "timeslot": self._current_timeslot,
            "source_gs": self._source_gs,
            "target_gs": self._target_gs,
            "dijkstra_delay_s": self._dijkstra_delay(),
        }

    # ================================================================
    #                    辅助方法
    # ================================================================
    def _nearest_sat(self, lon: float, lat: float, t: int) -> int:
        """找到距离地面点 (lon, lat) 最近的卫星 ID。"""
        best_id = -1
        best_dist = float("inf")
        for sat_id, sat in self._sat_by_id.items():
            d = _haversine_km(lon, lat, sat.longitude[t - 1], sat.latitude[t - 1])
            if d < best_dist:
                best_dist = d
                best_id = sat_id
        return best_id

    def _farthest_sat(self, from_id: int) -> int:
        """返回距 from_id 最远的卫星（备用目标）。"""
        t = self._current_timeslot
        src = self._sat_by_id[from_id]
        best_id = from_id
        best_dist = 0.0
        for sat_id, sat in self._sat_by_id.items():
            d = _sat_distance_km(
                src.longitude[t - 1], src.latitude[t - 1], src.altitude[t - 1],
                sat.longitude[t - 1], sat.latitude[t - 1], sat.altitude[t - 1],
            )
            if d > best_dist:
                best_dist = d
                best_id = sat_id
        return best_id

    def _sat_to_target_distance(self, sat_id: int) -> float:
        """当前时间片下，卫星 sat_id 到目标卫星的地面距离 (km)。"""
        t = self._current_timeslot
        sat = self._sat_by_id[sat_id]
        tgt = self._sat_by_id[self._target_sat_id]
        return _sat_distance_km(
            sat.longitude[t - 1], sat.latitude[t - 1], sat.altitude[t - 1],
            tgt.longitude[t - 1], tgt.latitude[t - 1], tgt.altitude[t - 1],
        )

    def _get_link_delay(self, sat1_id: int, sat2_id: int, t: int) -> float:
        """两颗直连卫星之间在时间片 t 的传播时延 (秒)。"""
        sat1 = self._sat_by_id[sat1_id]
        sat2 = self._sat_by_id[sat2_id]
        dist = _sat_distance_km(
            sat1.longitude[t - 1], sat1.latitude[t - 1], sat1.altitude[t - 1],
            sat2.longitude[t - 1], sat2.latitude[t - 1], sat2.altitude[t - 1],
        )
        return dist / 300000.0  # 光速 300,000 km/s

    def _dijkstra_delay(self) -> float:
        """计算当前时间片下源→目标的 Dijkstra 最短时延 (秒)，用于对比。"""
        try:
            G = nx.Graph()
            for sat_id in self._sat_by_id:
                G.add_node(sat_id)
            for sat_id, neighbors in self.adj.items():
                for nb_id in neighbors:
                    if nb_id != -1 and not G.has_edge(sat_id, nb_id):
                        delay = self._get_link_delay(sat_id, nb_id, self._current_timeslot)
                        G.add_edge(sat_id, nb_id, weight=delay)
            return nx.dijkstra_path_length(
                G, source=self._current_sat_id, target=self._target_sat_id
            )
        except nx.NetworkXNoPath:
            return float("inf")

    # ================================================================
    #                    渲染 (可选)
    # ================================================================
    def render(self):
        if self.render_mode == "human":
            t = self._current_timeslot
            cur = self._sat_by_id[self._current_sat_id]
            tgt = self._sat_by_id[self._target_sat_id]
            dist = self._sat_to_target_distance(self._current_sat_id)
            print(
                f"[Step {self._step_count:3d}] "
                f"Sat {self._current_sat_id:4d} "
                f"({cur.longitude[t-1]:7.2f}°, {cur.latitude[t-1]:6.2f}°) "
                f"→ Target {self._target_sat_id:4d} "
                f"({tgt.longitude[t-1]:7.2f}°, {tgt.latitude[t-1]:6.2f}°) "
                f"| dist={dist:.0f}km | delay={self._accumulated_delay*1000:.2f}ms"
            )


# ========================== 环境注册 ==========================
gym.register(
    id="SatelliteNetwork-v0",
    entry_point="env.sn_env:SatelliteNetworkEnv",
)


# ========================== 快速自测 ==========================
if __name__ == "__main__":
    print("=" * 60)
    print("  SatelliteNetworkEnv 自测")
    print("=" * 60)

    # 使用较小星座快速测试
    env = SatelliteNetworkEnv(
        constellation_name="Kuiper",  # 比 Starlink 小，初始化更快
        shell_idx=0,
        dT=120,
        max_steps=30,
        render_mode="human",
    )

    obs, info = env.reset(seed=42)
    print(f"\n初始观测维度: {obs.shape}")
    print(f"Dijkstra 最优时延: {info['dijkstra_delay_s']*1000:.2f} ms")
    print()

    total_reward = 0.0
    for step_i in range(30):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        if terminated or truncated:
            status = "到达目标!" if terminated else "超时截断"
            print(f"\n{status} | 累计奖励: {total_reward:.3f} "
                  f"| RL时延: {info['accumulated_delay_s']*1000:.2f}ms "
                  f"| Dijkstra: {info['dijkstra_delay_s']*1000:.2f}ms")
            break

    env.close()
    print("\n✓ 环境自测完成")
