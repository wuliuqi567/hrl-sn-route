"""
Phase 3: HRL 联合微调 + 完整训练管线 (Joint Fine-tuning & Full Pipeline)

本模块有两个主要功能:
  1. finetune_hrl(): Phase 3 联合微调 — 上层和下层智能体同时可训练
  2. full_pipeline(): 完整 3 阶段管线 (Phase 1 → 2 → 3)

联合微调的设计动机 / Design Rationale:
  - Phase 1、2 分层训练后，上下层策略可能不够协调
  - Phase 3 解冻双方，在完整跨域任务中联合优化
  - 同时训练上层 (域间链路选择) 和下层 (域内路由)
  - 双方的回放缓冲区独立，梯度更新也独立

full_pipeline 的优势:
  - SimEngine 仅初始化一次 (星座加载耗时约 60s)
  - 三个阶段共享同一个 sim 实例
  - 无需手动管理检查点传递

Usage:
    # 完整管线 (Phase 1 → 2 → 3):
    python -m train.train_hrl --config configs/default.yaml --full-pipeline

    # 仅 Phase 3 (从检查点加载):
    python -m train.train_hrl --config configs/default.yaml \
           --upper-ckpt checkpoints/upper_trained.zip \
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
from train.rollout import run_episode
from train.logger import TrainingLogger


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_sim(cfg: dict) -> NetworkSimEngine:
    """从配置字典构建仿真引擎。

    Construct the simulation engine from config dict.
    封装了 SimEngine 的所有初始化参数，避免在多处重复构造。

    Parameters
    ----------
    cfg : dict
        完整配置字典 (从 YAML 加载)。

    Returns
    -------
    NetworkSimEngine
        初始化完成的仿真引擎。
    """
    return NetworkSimEngine(
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
        seed=cfg["training"]["seed"],
    )


def finetune_hrl(
    cfg: dict,
    upper_ckpt: str | None = None,
    lower_ckpt: str | None = None,
    sim: NetworkSimEngine | None = None,
    upper_model=None,
    lower_model=None,
):
    """Phase 3: 上下层智能体联合微调。

    Joint fine-tuning of both agents.

    训练流程 / Training Flow:
      1. 加载或接收上下层模型 (也可新建)
      2. 设置双方为训练模式 (set_training_mode(True))
      3. for each episode:
         a. run_episode() → 收集上层和下层经验
         b. 上层经验 → 上层 replay buffer; 下层经验 → 下层 replay buffer
         c. 分别训练两个模型
      4. 保存双方模型

    Parameters
    ----------
    cfg : dict
        完整配置字典。
    upper_ckpt : str | None
        上层模型检查点路径 (可选)。
    lower_ckpt : str | None
        下层模型检查点路径 (可选)。
    sim : NetworkSimEngine | None
        若由 full_pipeline 调用，传入已有的 sim 避免重复初始化。
    upper_model, lower_model
        若由 full_pipeline 调用，传入前两阶段训练好的模型。

    Returns
    -------
    tuple[SB3 model, SB3 model]
        (微调后的上层模型, 微调后的下层模型)。
    """
    from stable_baselines3 import DQN

    seed = cfg["training"]["seed"]
    lower_mode = cfg["routing"]["lower_mode"]
    K = cfg["routing"]["K"]
    dqn_cfg = cfg["training"]["dqn"]
    T_episode = cfg["training"]["T_episode"]

    # ── 构建 sim (若未由调用者提供) ──
    if sim is None:
        sim = build_sim(cfg)

    # ── 构建 / 加载模型 ──
    # 支持三种来源: (1) 传入的模型 (2) 从检查点加载 (3) 新建
    if lower_mode == "k_path":
        lower_env = LowerKPathEnvShell(sim, K=K)
    else:
        lower_env = LowerHopEnvShell(sim)
    upper_env = UpperEnvShell(sim)

    if upper_model is None:
        if upper_ckpt:
            upper_model = DQN.load(upper_ckpt, env=upper_env)
            print(f"  Upper model loaded from {upper_ckpt}")
        else:
            upper_model = DQN(
                "MlpPolicy", upper_env,
                learning_rate=dqn_cfg["learning_rate"],
                buffer_size=dqn_cfg["buffer_size"],
                batch_size=dqn_cfg["batch_size"],
                gamma=dqn_cfg["gamma"],
                tau=dqn_cfg["tau"],
                target_update_interval=dqn_cfg["target_update_interval"],
                exploration_fraction=dqn_cfg["exploration_fraction"],
                exploration_final_eps=dqn_cfg["exploration_final_eps"],
                learning_starts=dqn_cfg["batch_size"],
                verbose=0, seed=seed,
            )

    if lower_model is None:
        if lower_ckpt:
            lower_model = DQN.load(lower_ckpt, env=lower_env)
            print(f"  Lower model loaded from {lower_ckpt}")
        else:
            lower_model = DQN(
                "MlpPolicy", lower_env,
                learning_rate=dqn_cfg["learning_rate"],
                buffer_size=dqn_cfg["buffer_size"],
                batch_size=dqn_cfg["batch_size"],
                gamma=dqn_cfg["gamma"],
                tau=dqn_cfg["tau"],
                target_update_interval=dqn_cfg["target_update_interval"],
                exploration_fraction=dqn_cfg["exploration_fraction"],
                exploration_final_eps=dqn_cfg["exploration_final_eps"],
                learning_starts=dqn_cfg["batch_size"],
                verbose=0, seed=seed,
            )

    # ── 确保双方都在训练模式 (区别于 Phase 2 的冻结) ──
    upper_model.policy.set_training_mode(True)
    lower_model.policy.set_training_mode(True)

    n_episodes = cfg["training"]["finetune_episodes"]

    print(f"\n{'='*60}")
    print(f"  Phase 3: Joint HRL fine-tuning")
    print(f"  Episodes: {n_episodes}, T_episode: {T_episode}")
    print(f"{'='*60}\n")

    logger = TrainingLogger(
        log_dir="runs", prefix="phase3_finetune",
        console_interval=max(1, n_episodes // 20),
    )

    total_upper = 0
    total_lower = 0
    t0 = time.time()

    for ep in range(n_episodes):
        upper_exps, lower_exps, ep_metrics = run_episode(
            upper_model, lower_model, sim,
            T_episode=T_episode,
            lower_mode=lower_mode,
            K=K,
            deterministic=False,
        )

        # ── 分别插入两个独立的回放缓冲区 ──
        # 上下层模型各自维护独立的 replay buffer，
        # 因为 obs/action 空间不同，不能混用。
        for exp in upper_exps:
            upper_model.replay_buffer.add(
                exp.obs.reshape(1, -1),
                exp.next_obs.reshape(1, -1),
                np.array([[exp.action]]),
                np.array([exp.reward]),
                np.array([exp.done or exp.truncated]),
                [{}],
            )
        for exp in lower_exps:
            lower_model.replay_buffer.add(
                exp.obs.reshape(1, -1),
                exp.next_obs.reshape(1, -1),
                np.array([[exp.action]]),
                np.array([exp.reward]),
                np.array([exp.done or exp.truncated]),
                [{}],
            )
        total_upper += len(upper_exps)
        total_lower += len(lower_exps)

        # ── 分别训练两个模型 ──
        # 上下层独立梯度更新，互不干扰
        if total_upper >= upper_model.batch_size and upper_exps:
            upper_model.train(gradient_steps=max(1, len(upper_exps) // 4))
        if total_lower >= lower_model.batch_size and lower_exps:
            lower_model.train(gradient_steps=max(1, len(lower_exps) // 4))

        # Logging
        avg_delay = (np.mean(ep_metrics.delays)
                     if ep_metrics.delays else float("inf"))
        logger.log_episode(ep, n_episodes, {
            "success_rate": ep_metrics.success_rate,
            "avg_delay": avg_delay,
            "upper_exps": len(upper_exps),
            "lower_exps": len(lower_exps),
            "success_count": ep_metrics.success_count,
            "fail_count": ep_metrics.fail_count,
        })

    logger.close()

    # ── Save both models ──
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)

    upper_path = save_dir / "upper_finetuned"
    lower_path = save_dir / f"lower_{lower_mode}_finetuned"
    upper_model.save(str(upper_path))
    lower_model.save(str(lower_path))

    print(f"\n  ✓ Upper model saved to {upper_path}")
    print(f"  ✓ Lower model saved to {lower_path}")
    print(f"  Total: upper_exps={total_upper}, lower_exps={total_lower}")
    print(f"  Total time: {time.time() - t0:.0f}s")

    return upper_model, lower_model


def full_pipeline(cfg: dict):
    """运行完整的 3 阶段训练管线。

    Run the complete 3-phase training pipeline:
      Phase 1: 下层预训练 (随机域内子任务)
      Phase 2: 上层训练 (下层冻结, 完整回合)
      Phase 3: 联合微调 (双方可训练)

    优势:
      - SimEngine 仅初始化一次，避免约 60s 的重复加载
      - 模型在阶段间无缝传递，无需手动管理检查点
      - 统一的日志和进度输出

    Parameters
    ----------
    cfg : dict
        完整配置字典。
    """
    from stable_baselines3 import DQN
    from train.train_lower import (
        pretrain_lower_episode_kpath,
        pretrain_lower_episode_hop,
    )

    print("\n" + "=" * 60)
    print("  HRL Satellite Routing — Full Training Pipeline")
    print("=" * 60)

    seed = cfg["training"]["seed"]
    rng = np.random.default_rng(seed)
    lower_mode = cfg["routing"]["lower_mode"]
    K = cfg["routing"]["K"]
    dqn_cfg = cfg["training"]["dqn"]
    T_episode = cfg["training"]["T_episode"]

    # ── Build SimEngine once ──
    sim = build_sim(cfg)

    # ── Phase 1: Lower pre-training ──
    from env.lower_kpath_env_shell import LowerKPathEnvShell
    from env.lower_hop_env_shell import LowerHopEnvShell

    if lower_mode == "k_path":
        lower_env = LowerKPathEnvShell(sim, K=K)
    else:
        lower_env = LowerHopEnvShell(sim)

    lower_model = DQN(
        "MlpPolicy", lower_env,
        learning_rate=dqn_cfg["learning_rate"],
        buffer_size=dqn_cfg["buffer_size"],
        batch_size=dqn_cfg["batch_size"],
        gamma=dqn_cfg["gamma"],
        tau=dqn_cfg["tau"],
        target_update_interval=dqn_cfg["target_update_interval"],
        exploration_fraction=dqn_cfg["exploration_fraction"],
        exploration_final_eps=dqn_cfg["exploration_final_eps"],
        learning_starts=dqn_cfg["batch_size"],
        verbose=0, seed=seed,
    )

    n_pretrain = cfg["training"]["pretrain_episodes"]
    n_subtasks = cfg["training"]["pretrain_subtasks_per_episode"]

    print(f"\n{'='*60}")
    print(f"  Phase 1: Lower pre-training ({lower_mode})")
    print(f"  Episodes: {n_pretrain}, Sub-tasks/ep: {n_subtasks}")
    print(f"{'='*60}\n")

    total_lower_exps = 0
    t0 = time.time()
    for ep in range(n_pretrain):
        sim.reset(start_timeslot=int(rng.integers(1, sim.num_timeslots + 1)))
        for _ in range(int(rng.integers(0, 10))):
            sim.advance_timeslot()

        if lower_mode == "k_path":
            exps = pretrain_lower_episode_kpath(lower_model, sim, n_subtasks, K, rng)
        else:
            exps = pretrain_lower_episode_hop(lower_model, sim, n_subtasks, rng)

        for exp in exps:
            lower_model.replay_buffer.add(
                exp.obs.reshape(1, -1), exp.next_obs.reshape(1, -1),
                np.array([[exp.action]]), np.array([exp.reward]),
                np.array([exp.done or exp.truncated]), [{}],
            )
        total_lower_exps += len(exps)

        if total_lower_exps >= lower_model.batch_size and exps:
            lower_model.train(gradient_steps=max(1, len(exps) // 4))

        if (ep + 1) % max(1, n_pretrain // 20) == 0 or ep == 0:
            print(f"  [P1 Ep {ep+1:4d}/{n_pretrain}] "
                  f"exps={total_lower_exps:6d}  time={time.time()-t0:.0f}s")

    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    lower_model.save(str(save_dir / f"lower_{lower_mode}_pretrained"))
    print(f"  ✓ Phase 1 complete ({time.time()-t0:.0f}s)")

    # ── Phase 2: Upper training (lower frozen) ──
    from env.upper_env_shell import UpperEnvShell

    lower_model.policy.set_training_mode(False)
    upper_env = UpperEnvShell(sim)
    upper_model = DQN(
        "MlpPolicy", upper_env,
        learning_rate=dqn_cfg["learning_rate"],
        buffer_size=dqn_cfg["buffer_size"],
        batch_size=dqn_cfg["batch_size"],
        gamma=dqn_cfg["gamma"],
        tau=dqn_cfg["tau"],
        target_update_interval=dqn_cfg["target_update_interval"],
        exploration_fraction=dqn_cfg["exploration_fraction"],
        exploration_final_eps=dqn_cfg["exploration_final_eps"],
        learning_starts=dqn_cfg["batch_size"],
        verbose=0, seed=seed,
    )

    n_upper_ep = cfg["training"]["upper_train_episodes"]
    print(f"\n{'='*60}")
    print(f"  Phase 2: Upper training (lower frozen)")
    print(f"  Episodes: {n_upper_ep}, T_episode: {T_episode}")
    print(f"{'='*60}\n")

    total_upper_exps = 0
    t1 = time.time()
    for ep in range(n_upper_ep):
        upper_exps, _, ep_metrics = run_episode(
            upper_model, lower_model, sim,
            T_episode=T_episode, lower_mode=lower_mode,
            K=K, deterministic=False,
        )
        for exp in upper_exps:
            upper_model.replay_buffer.add(
                exp.obs.reshape(1, -1), exp.next_obs.reshape(1, -1),
                np.array([[exp.action]]), np.array([exp.reward]),
                np.array([exp.done or exp.truncated]), [{}],
            )
        total_upper_exps += len(upper_exps)

        if total_upper_exps >= upper_model.batch_size and upper_exps:
            upper_model.train(gradient_steps=max(1, len(upper_exps) // 4))

        if (ep + 1) % max(1, n_upper_ep // 20) == 0 or ep == 0:
            sr = ep_metrics.success_rate
            print(f"  [P2 Ep {ep+1:4d}/{n_upper_ep}] "
                  f"upper={len(upper_exps):3d}  SR={sr:.2%}  "
                  f"time={time.time()-t1:.0f}s")

    upper_model.save(str(save_dir / "upper_trained"))
    print(f"  ✓ Phase 2 complete ({time.time()-t1:.0f}s)")

    # ── Phase 3: Joint fine-tuning ──
    upper_model.policy.set_training_mode(True)
    lower_model.policy.set_training_mode(True)

    finetune_hrl(
        cfg, sim=sim,
        upper_model=upper_model,
        lower_model=lower_model,
    )

    print("\n" + "=" * 60)
    print("  ✓ Full pipeline complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Joint HRL fine-tuning / Full pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--upper-ckpt", type=str, default=None)
    parser.add_argument("--lower-ckpt", type=str, default=None)
    parser.add_argument("--full-pipeline", action="store_true",
                        help="Run full 3-phase pipeline from scratch")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.full_pipeline:
        full_pipeline(cfg)
    else:
        finetune_hrl(cfg,
                     upper_ckpt=args.upper_ckpt,
                     lower_ckpt=args.lower_ckpt)


if __name__ == "__main__":
    main()
