"""
LowerHopEnvShell — 下层智能体 (逐跳模式) 的虚拟 Gymnasium 环境。

仅用于 SB3 模型初始化，实际交互通过 rollout.py 进行。

空间定义 / Space Definitions:
  observation_space : Box(23,)
    - 自身位置(3) + 目标位置(3) + 距离(1) + 累积时延(1)
    + 步数进度(1) + 带宽需求(1) + 剩余时延预算(1)
    + 4邻居 × 3特征(12) = 23维
  action_space      : Discrete(4)
    - 4 个方向: up(0), down(1), left(2), right(3)
    对应 +Grid 拓扑中的 轨内前/后、轨间左/右 ISL
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
from gymnasium import spaces

if TYPE_CHECKING:
    from env.sim_engine import NetworkSimEngine


class LowerHopEnvShell(gym.Env):
    """下层逐跳智能体的 SB3 初始化用 Shell 环境。

    Parameters
    ----------
    sim : NetworkSimEngine
        用于获取观测空间维度 (23 维)。
    """
    metadata = {"render_modes": []}

    def __init__(self, sim: "NetworkSimEngine"):
        super().__init__()
        self._obs_dim = sim.get_lower_hop_obs_dim()

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self._obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self._obs_dim, dtype=np.float32), {}

    def step(self, action):
        return (np.zeros(self._obs_dim, dtype=np.float32),
                0.0, False, False, {})
