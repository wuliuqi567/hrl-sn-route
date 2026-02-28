"""
UpperEnvShell — 上层智能体的虚拟 Gymnasium 环境。

本环境仅用于初始化 SB3 DQN 模型，提供:
  - observation_space: 观测空间定义 (Box)
  - action_space: 动作空间定义 (Discrete)

实际的状态转换和奖励计算在 train/rollout.py 中完成，
不通过此 Shell 环境的 step()/reset() 方法。

这种设计模式的原因:
  - HRL 中多智能体共享同一个环境状态，无法装入单个 Gym 环境
  - SB3 要求模型初始化时提供一个符合 Gym API 的环境
  - Shell 环境满足此需求，但不参与实际交互

Dummy Gymnasium environment used only to initialise the
SB3 DQN model for the upper (cross-domain) agent.

The real interaction happens in train/rollout.py via
  model.predict()  /  replay_buffer.add()  /  model.train()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
from gymnasium import spaces

if TYPE_CHECKING:
    from env.sim_engine import NetworkSimEngine


class UpperEnvShell(gym.Env):
    """上层智能体的 SB3 初始化用 Shell 环境。

    Shell environment for initialising the upper SB3 DQN model.

    Attributes
    ----------
    observation_space : spaces.Box
        观测向量空间，维度 = sats_per_orbit * 3 + 5 + n_domains。
        包含: 域间链路特征 + 流需求特征 + 全局利用率。
    action_space : spaces.Discrete
        动作空间 = sats_per_orbit (选择第几条域间链路)。
        例如 Starlink: Discrete(22), Kuiper: Discrete(34)。
    """

    metadata = {"render_modes": []}

    def __init__(self, sim: "NetworkSimEngine"):
        super().__init__()
        self._obs_dim = sim.get_upper_obs_dim()
        self._n_actions = sim.sats_per_orbit

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self._obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self._n_actions)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self._obs_dim, dtype=np.float32), {}

    def step(self, action):
        return (np.zeros(self._obs_dim, dtype=np.float32),
                0.0, True, False, {})
