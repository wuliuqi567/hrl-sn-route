"""
模型工厂 — 算法无关的 RL 模型创建、加载、训练接口。

Model Factory — Algorithm-agnostic RL model creation, loading, and training.

设计目标 / Design Goals:
  - 将算法选择从训练脚本中解耦，变为配置级决策
  - 支持 SB3 所有 off-policy 离散动作算法: DQN, DDQN (内置), QRDQN
  - 提供统一的 create / load / add_experience / train_step 接口
  - 训练脚本只需调用工厂方法，无需了解底层算法细节

支持的算法 / Supported Algorithms:
  ┌────────────┬────────────────────────┬──────────────────────────────┐
  │ 算法名     │ 来源                   │ 备注                         │
  ├────────────┼────────────────────────┼──────────────────────────────┤
  │ DQN        │ stable_baselines3      │ 标准 DQN (内含 Double-Q)     │
  │ QRDQN      │ sb3_contrib            │ 分位数回归 DQN              │
  │ (未来)     │ ...                    │ 可扩展更多 off-policy 算法   │
  └────────────┴────────────────────────┴──────────────────────────────┘

架构位置 / Architecture Position:
  ┌─────────────────────────────────────────────┐
  │ configs/default.yaml                        │
  │   training.algorithm: "DQN"                 │
  │   training.algorithm_kwargs: {}             │
  │                 ↓                           │
  │ agents/model_factory.py  ← 本模块          │
  │   create_model() / load_model()             │
  │                 ↓                           │
  │ train_*.py  (算法无关的训练循环)             │
  │   factory.add_experience() / train_step()   │
  │                 ↓                           │
  │ rollout.py  (算法无关的回合执行)             │
  │   model.predict()                           │
  └─────────────────────────────────────────────┘

使用示例 / Usage:
    from agents.model_factory import ModelFactory

    factory = ModelFactory(algorithm="DQN", algo_kwargs={})

    model = factory.create_model(env_shell, dqn_cfg, seed=42)
    model = factory.load_model("checkpoints/model.zip", env_shell)

    factory.add_experience(model, experience)
    factory.train_step(model, gradient_steps=4)
    factory.save_model(model, "checkpoints/model")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from train.rollout import Experience


# =====================================================================
#  算法注册表 (Algorithm Registry)
#  将算法名称映射到 SB3 类和对应的默认参数过滤器
# =====================================================================
_ALGORITHM_REGISTRY: dict[str, dict[str, Any]] = {}


def _register_algorithms():
    """延迟注册所有支持的算法，避免启动时的 import 开销。"""
    global _ALGORITHM_REGISTRY

    if _ALGORITHM_REGISTRY:
        return  # 已注册

    # ── DQN (stable-baselines3 内置) ──
    try:
        from stable_baselines3 import DQN
        _ALGORITHM_REGISTRY["DQN"] = {
            "class": DQN,
            "type": "off_policy",
            # DQN 接受的超参数键 (用于从 dqn_cfg 中过滤)
            "supported_params": {
                "learning_rate", "buffer_size", "batch_size", "gamma",
                "tau", "target_update_interval", "exploration_fraction",
                "exploration_final_eps", "learning_starts",
                "gradient_steps", "train_freq",
            },
        }
    except ImportError:
        pass

    # ── QRDQN (sb3-contrib) ──
    try:
        from sb3_contrib import QRDQN
        _ALGORITHM_REGISTRY["QRDQN"] = {
            "class": QRDQN,
            "type": "off_policy",
            "supported_params": {
                "learning_rate", "buffer_size", "batch_size", "gamma",
                "tau", "target_update_interval", "exploration_fraction",
                "exploration_final_eps", "learning_starts",
                "gradient_steps", "train_freq",
                # QRDQN 特有参数
                "n_quantiles",
            },
        }
    except ImportError:
        pass


def get_supported_algorithms() -> list[str]:
    """返回当前环境中可用的算法列表。

    Returns
    -------
    list[str]
        可用算法名称列表, e.g. ["DQN", "QRDQN"]。
    """
    _register_algorithms()
    return list(_ALGORITHM_REGISTRY.keys())


# =====================================================================
#  ModelFactory — 工厂类
# =====================================================================
class ModelFactory:
    """算法无关的模型工厂。

    Algorithm-agnostic model factory for creating, loading, and managing
    SB3 off-policy models.

    Parameters
    ----------
    algorithm : str
        算法名称, e.g. "DQN", "QRDQN"。不区分大小写。
    algo_kwargs : dict | None
        算法特有的额外参数, e.g. {"n_quantiles": 200} for QRDQN。
        这些参数会覆盖 dqn_cfg 中的同名参数。

    Raises
    ------
    ValueError
        如果算法名称不在注册表中。

    示例 / Example:
        factory = ModelFactory("DQN")
        model = factory.create_model(env, cfg, seed=42)
    """

    def __init__(self, algorithm: str = "DQN", algo_kwargs: dict | None = None):
        _register_algorithms()

        # 不区分大小写匹配
        algo_key = algorithm.upper()
        if algo_key not in _ALGORITHM_REGISTRY:
            available = ", ".join(sorted(_ALGORITHM_REGISTRY.keys()))
            raise ValueError(
                f"不支持的算法: '{algorithm}'。"
                f"可用算法: [{available}]。"
                f"若需 QRDQN, 请安装 sb3-contrib: pip install sb3-contrib"
            )

        self._algo_info = _ALGORITHM_REGISTRY[algo_key]
        self._algo_class = self._algo_info["class"]
        self._algo_type = self._algo_info["type"]  # "off_policy"
        self._supported_params = self._algo_info["supported_params"]
        self._algo_kwargs = algo_kwargs or {}
        self.algorithm_name = algo_key

    @property
    def is_off_policy(self) -> bool:
        """当前算法是否为 off-policy 类型。"""
        return self._algo_type == "off_policy"

    def _filter_params(self, dqn_cfg: dict) -> dict:
        """从通用 dqn_cfg 中过滤出当前算法支持的参数。

        Parameters
        ----------
        dqn_cfg : dict
            配置文件中 training.dqn 段的内容。

        Returns
        -------
        dict
            仅包含当前算法支持的参数键值对。
        """
        filtered = {}
        for key, val in dqn_cfg.items():
            if key in self._supported_params:
                filtered[key] = val

        # 用算法特有的 kwargs 覆盖
        for key, val in self._algo_kwargs.items():
            if key in self._supported_params:
                filtered[key] = val

        return filtered

    def create_model(
        self,
        env: gym.Env,
        dqn_cfg: dict,
        seed: int = 42,
        policy: str = "MlpPolicy",
        policy_kwargs: dict | None = None,
    ):
        """创建一个新的 RL 模型。

        Create a new RL model with the configured algorithm.

        Parameters
        ----------
        env : gym.Env
            Shell 环境 (提供 observation_space 和 action_space)。
        dqn_cfg : dict
            配置文件中 training.dqn 段的超参数字典。
        seed : int
            随机种子。
        policy : str
            策略网络类型, 默认 "MlpPolicy"。
        policy_kwargs : dict | None
            策略网络额外参数 (e.g., features_extractor_class)。

        Returns
        -------
        SB3 BaseAlgorithm
            创建好的模型实例。
        """
        params = self._filter_params(dqn_cfg)

        # 确保 learning_starts 至少为 batch_size
        if "learning_starts" not in params and "batch_size" in params:
            params["learning_starts"] = params["batch_size"]

        kwargs: dict[str, Any] = {
            **params,
            "verbose": 0,
            "seed": seed,
        }
        if policy_kwargs is not None:
            kwargs["policy_kwargs"] = policy_kwargs

        model = self._algo_class(policy, env, **kwargs)
        print(f"  [Factory] Created {self.algorithm_name} model "
              f"(obs={env.observation_space.shape}, act={env.action_space})")
        return model

    def load_model(
        self,
        checkpoint_path: str,
        env: gym.Env,
    ):
        """从检查点加载模型。

        Load a model from a checkpoint file.

        Parameters
        ----------
        checkpoint_path : str
            模型检查点路径 (.zip 文件)。
        env : gym.Env
            Shell 环境 (用于验证 obs/act 空间)。

        Returns
        -------
        SB3 BaseAlgorithm
            加载的模型实例。
        """
        model = self._algo_class.load(checkpoint_path, env=env)
        print(f"  [Factory] Loaded {self.algorithm_name} model from {checkpoint_path}")
        return model

    def add_experience(self, model, exp: Experience) -> None:
        """将一条经验插入模型的回放缓冲区。

        Add a single experience to the model's replay buffer.
        This is the unified off-policy API, compatible with DQN/QRDQN/etc.

        Parameters
        ----------
        model : SB3 off-policy model
            目标模型。
        exp : Experience
            经验元组 (obs, next_obs, action, reward, done, truncated, info)。
        """
        model.replay_buffer.add(
            exp.obs.reshape(1, -1),
            exp.next_obs.reshape(1, -1),
            np.array([[exp.action]]),
            np.array([exp.reward]),
            np.array([exp.done or exp.truncated]),
            [{}],
        )

    def add_experiences(self, model, exps: list[Experience]) -> int:
        """批量插入经验到回放缓冲区。

        Batch insert experiences into the replay buffer.

        Parameters
        ----------
        model : SB3 off-policy model
            目标模型。
        exps : list[Experience]
            经验列表。

        Returns
        -------
        int
            实际插入的经验数量。
        """
        for exp in exps:
            self.add_experience(model, exp)
        return len(exps)

    def train_step(self, model, gradient_steps: int = 1) -> None:
        """执行若干步梯度更新。

        Perform gradient update steps using samples from the replay buffer.

        Parameters
        ----------
        model : SB3 off-policy model
            目标模型。
        gradient_steps : int
            梯度更新步数。
        """
        model.train(gradient_steps=gradient_steps)

    def should_train(self, model, total_experiences: int) -> bool:
        """判断是否应该开始训练。

        Check if enough experiences have been collected to start training.

        Parameters
        ----------
        model : SB3 off-policy model
            目标模型。
        total_experiences : int
            已收集的总经验数。

        Returns
        -------
        bool
            True 表示可以开始训练。
        """
        return total_experiences >= model.batch_size

    @staticmethod
    def save_model(model, path: str) -> str:
        """保存模型到磁盘。

        Save model checkpoint.

        Parameters
        ----------
        model : SB3 BaseAlgorithm
            待保存的模型。
        path : str
            保存路径 (不含 .zip 后缀)。

        Returns
        -------
        str
            实际保存的路径。
        """
        save_dir = Path(path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(path))
        print(f"  [Factory] Model saved to {path}")
        return str(path)

    @staticmethod
    def set_training_mode(model, training: bool = True) -> None:
        """设置模型的训练/推理模式。

        Set the model's policy to training or evaluation mode.
        Affects BatchNorm / Dropout layers.

        Parameters
        ----------
        model : SB3 BaseAlgorithm
            目标模型。
        training : bool
            True=训练模式, False=推理模式 (冻结)。
        """
        model.policy.set_training_mode(training)


# =====================================================================
#  便捷函数 (Convenience Functions)
# =====================================================================
def create_factory_from_config(cfg: dict) -> ModelFactory:
    """从配置字典创建 ModelFactory 实例。

    Create a ModelFactory from the training config dict.

    Parameters
    ----------
    cfg : dict
        完整配置字典。读取以下键:
        - training.algorithm (str): 算法名称, 默认 "DQN"
        - training.algorithm_kwargs (dict): 算法特有参数, 默认 {}

    Returns
    -------
    ModelFactory
        配置好的工厂实例。

    示例 / Example:
        cfg = yaml.safe_load(open("configs/default.yaml"))
        factory = create_factory_from_config(cfg)
    """
    algo = cfg.get("training", {}).get("algorithm", "DQN")
    algo_kwargs = cfg.get("training", {}).get("algorithm_kwargs", {})
    return ModelFactory(algorithm=algo, algo_kwargs=algo_kwargs)
