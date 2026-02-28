"""
LowerKPathEnvShell — 下层智能体 (K 最短路模式) 的虚拟 Gymnasium 环境。

仅用于 SB3 模型初始化，实际交互通过 rollout.py 进行。

空间定义 / Space Definitions:
  observation_space : Box(K*4 + 6,)   ≈ 22 dims for K=4
    - K 条路径 × 4 特征: [时延, 最小带宽, 跳数, 瓶颈利用率]
    - 6 全局特征: [带宽需求, 时延预算, 入口经纬度, 出口经纬度]
  action_space      : Discrete(K)     ≈ 4
    - 选择第几条候选路径
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
from gymnasium import spaces

if TYPE_CHECKING:
    from env.sim_engine import NetworkSimEngine


class LowerKPathEnvShell(gym.Env):
    """下层 K-path 智能体的 SB3 初始化用 Shell 环境。

    Parameters
    ----------
    sim : NetworkSimEngine
        用于获取观测空间维度。
    K : int
        K 最短路候选数，决定 obs 维度和 action 空间大小。
    """
    metadata = {"render_modes": []}

    def __init__(self, sim: "NetworkSimEngine", K: int = 4):
        super().__init__()
        self._obs_dim = sim.get_lower_kpath_obs_dim(K)
        self._K = K

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self._obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(K)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self._obs_dim, dtype=np.float32), {}

    def step(self, action):
        return (np.zeros(self._obs_dim, dtype=np.float32),
                0.0, True, False, {})
