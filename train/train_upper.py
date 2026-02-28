"""
Phase 2: 上层智能体训练 (下层智能体冻结)

Upper agent training with lower agent frozen.

加载 Phase 1 训练好的下层模型，冻结其参数，训练上层 DQN 模型。
上层智能体负责在跨域流中选择域间中继链路（出口/入口卫星对）。

训练流程 / Training Flow:
  1. 初始化 SimEngine
  2. 加载下层模型 (DQN.load) 并冻结 (set_training_mode(False))
  3. 创建新的上层 DQN 模型
  4. 循环 upper_train_episodes 个回合:
     a. run_episode() → 上层选链路 + 下层域内路由
     b. 只将上层经验插入上层回放缓冲区 (下层经验丢弃)
     c. 仅训练上层模型
  5. 保存模型

设计动机 / Design Rationale:
  分阶段训练可缓解 HRL 中多智能体同时学习的不稳定性。
  冻结下层后，上层面对的是一个稳定的域内路由环境，
  可以专注学习域间链路选择策略。

Usage:
    python -m train.train_upper --config configs/default.yaml \
           --lower-ckpt checkpoints/lower_k_path_pretrained.zip
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from env.sim_engine import NetworkSimEngine
from env.upper_env_shell import UpperEnvShell
from env.lower_kpath_env_shell import LowerKPathEnvShell
from env.lower_hop_env_shell import LowerHopEnvShell
from agents.model_factory import create_factory_from_config
from train.rollout import run_episode
from train.logger import TrainingLogger


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train_upper(cfg: dict, lower_ckpt: str):
    """Phase 2: 训练上层智能体，下层冻结。

    Train the upper agent with frozen lower.

    Parameters
    ----------
    cfg : dict
        完整配置字典。
        关键字段: training.dqn.*, training.upper_train_episodes,
        routing.lower_mode, constellation.* 等。
    lower_ckpt : str
        Phase 1 训练好的下层模型检查点路径 (.zip)。

    Returns
    -------
    SB3 DQN model
        训练好的上层模型。
    """
    # ── 创建模型工厂 ──
    factory = create_factory_from_config(cfg)

    seed = cfg["training"]["seed"]
    lower_mode = cfg["routing"]["lower_mode"]
    K = cfg["routing"]["K"]
    dqn_cfg = cfg["training"]["dqn"]
    T_episode = cfg["training"]["T_episode"]

    # ── Build simulation engine ──
    sim = NetworkSimEngine(
        constellation_name=cfg["constellation"]["name"],
        shell_idx=cfg["constellation"]["shell_idx"],
        dT=cfg["constellation"]["dT"],
        n_domains=cfg["domain"]["n_domains"],
        domain_mode=cfg["domain"]["mode"],
        link_capacity_mbps=cfg["network"]["link_capacity_mbps"],
        arrival_rate=cfg["traffic"]["arrival_rate"],
        bw_range=tuple(cfg["traffic"]["bw_range"]),
        delay_range=tuple(cfg["traffic"]["delay_range"]),
        duration_range=tuple(cfg["traffic"]["duration_range"]),
        bw_weight_alpha=cfg["routing"]["bw_weight_alpha"],
        seed=seed,
    )

    # ── 加载已训练的下层模型并冻结 ──
    # 冻结意味着下层参数不再更新，但仍可 predict()
    if lower_mode == "k_path":
        lower_env = LowerKPathEnvShell(sim, K=K)
    else:
        lower_env = LowerHopEnvShell(sim)

    lower_model = factory.load_model(lower_ckpt, env=lower_env)
    factory.set_training_mode(lower_model, False)  # 冻结
    print(f"  Lower model frozen")

    # ── 创建新的上层模型 ──
    upper_env = UpperEnvShell(sim)
    upper_model = factory.create_model(
        upper_env, dqn_cfg, seed=seed,
    )

    n_episodes = cfg["training"]["upper_train_episodes"]

    print(f"\n{'='*60}")
    print(f"  Phase 2: Upper training (lower frozen)")
    print(f"  Episodes: {n_episodes}, T_episode: {T_episode}")
    print(f"{'='*60}\n")

    logger = TrainingLogger(
        log_dir="runs", prefix="phase2_upper",
        console_interval=max(1, n_episodes // 20),
    )

    total_upper_exps = 0
    t0 = time.time()

    for ep in range(n_episodes):
        # 运行完整回合: 上层选链路 + 下层域内路由 (下层冻结, 仅推理)
        upper_exps, _, ep_metrics = run_episode(
            upper_model, lower_model, sim,
            T_episode=T_episode,
            lower_mode=lower_mode,
            K=K,
            deterministic=False,  # 训练模式: 带 epsilon-greedy 探索
        )

        # ── 仅将上层经验插入上层回放缓冲区 ──
        # 下层经验 (_) 被丢弃，因为下层已冻结
        for exp in upper_exps:
            factory.add_experience(upper_model, exp)
        total_upper_exps += len(upper_exps)

        # ── 训练上层模型 ──
        if factory.should_train(upper_model, total_upper_exps) and upper_exps:
            grad_steps = max(1, len(upper_exps) // 4)
            factory.train_step(upper_model, gradient_steps=grad_steps)

        # Logging
        avg_delay = (np.mean(ep_metrics.delays)
                     if ep_metrics.delays else float("inf"))
        logger.log_episode(ep, n_episodes, {
            "success_rate": ep_metrics.success_rate,
            "avg_delay": avg_delay,
            "upper_exps": len(upper_exps),
            "success_count": ep_metrics.success_count,
            "fail_count": ep_metrics.fail_count,
        })

    # Save
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / "upper_trained"
    factory.save_model(upper_model, str(save_path))
    logger.close()
    print(f"\n  ✓ Upper model saved to {save_path}")
    print(f"  Total upper experiences: {total_upper_exps}")
    print(f"  Total time: {time.time() - t0:.0f}s")

    return upper_model


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Upper agent training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--lower-ckpt", type=str,
                        default="checkpoints/lower_k_path_pretrained.zip")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_upper(cfg, args.lower_ckpt)


if __name__ == "__main__":
    main()
