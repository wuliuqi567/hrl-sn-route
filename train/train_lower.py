"""
Phase 1: 下层智能体预训练 (Lower Agent Pre-training)

在无上层智能体参与的情况下，仅训练下层域内路由智能体。
随机生成域内子任务 (entry → exit)，让下层智能体学会:
  - K-path 模式: 在 K 条候选路径中选择最优的一条
  - hop-by-hop 模式: 逐跳导航到达目标出口卫星

训练流程 / Training Flow:
  1. 初始化 SimEngine (加载星座 + 拓扑 + 域划分)
  2. 创建 SB3 DQN 模型 (MlpPolicy)
  3. 循环 pretrain_episodes 个回合:
     a. 重置仿真环境 (随机起始时隙)
     b. 生成 n_subtasks 个随机域内子任务
     c. 收集 Experience → 插入回放缓冲区
     d. 调用 model.train() 进行梯度更新
  4. 保存模型到 checkpoints/

注意: 当前硬编码使用 SB3 DQN，后续将通过工厂模式重构为算法无关。

Usage:
    python -m train.train_lower --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from core.flow import FlowRequest
from env.sim_engine import NetworkSimEngine
from env.lower_kpath_env_shell import LowerKPathEnvShell
from env.lower_hop_env_shell import LowerHopEnvShell
from agents.model_factory import create_factory_from_config
from train.rollout import (
    route_intra_kpath, route_intra_hop,
    Experience,
)
from train.logger import TrainingLogger


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def pretrain_lower_episode_kpath(
    lower_model,
    sim: NetworkSimEngine,
    n_subtasks: int,
    K: int,
    rng: np.random.Generator,
) -> list[Experience]:
    """生成随机域内子任务并收集 K-path 经验。

    Generate random intra-domain sub-tasks and collect K-path experiences.
    Each sub-task is a random (entry, exit) pair within a random domain.

    逻辑流程 / Logic:
      for each subtask:
        1. 随机选择一个域 (domain_id)
        2. 随机选择域内两个卫星作为 entry 和 exit
        3. 创建合成流请求 (随机带宽/时延需求)
        4. 调用 route_intra_kpath() 收集经验
        5. 若成功，注册流以产生真实拥塞效果

    Parameters
    ----------
    lower_model : SB3 BaseAlgorithm
        下层智能体模型。
    sim : NetworkSimEngine
        网络仿真引擎。
    n_subtasks : int
        每回合生成的子任务数量。
    K : int
        K 最短路候选数。
    rng : np.random.Generator
        随机数生成器。

    Returns
    -------
    list[Experience]
        本回合所有子任务产生的经验列表。
    """
    exps: list[Experience] = []

    for _ in range(n_subtasks):
        # Pick a random domain
        domain_id = int(rng.integers(0, sim.n_domains))
        sats = sim.domain.get_domain_sats(domain_id)
        if len(sats) < 2:
            continue
        entry, exit_ = rng.choice(sats, size=2, replace=False)
        entry, exit_ = int(entry), int(exit_)

        # Create a synthetic flow
        fid = sim.flow_manager.next_flow_id()
        flow = FlowRequest(
            flow_id=fid,
            src_sat=entry,
            dst_sat=exit_,
            bandwidth=float(rng.uniform(10, 200)),
            max_delay=float(rng.uniform(0.02, 0.10)),
            duration=int(rng.integers(5, 31)),
            arrival_timeslot=sim.current_timeslot,
        )

        seg_exps = route_intra_kpath(
            lower_model, sim, entry, exit_, flow, domain_id, K,
            deterministic=False,
        )
        exps.extend(seg_exps)

        # If allocation succeeded, register to create realistic congestion;
        # otherwise don't (failed allocation isn't stored in bandwidth mgr)
        seg_info = seg_exps[-1].info if seg_exps else {}
        if seg_info.get("segment_ok") and seg_info.get("segment_path"):
            sim.flow_manager.register(
                flow, seg_info["segment_path"], seg_info["segment_delay"])

    return exps


def pretrain_lower_episode_hop(
    lower_model,
    sim: NetworkSimEngine,
    n_subtasks: int,
    rng: np.random.Generator,
) -> list[Experience]:
    """生成随机域内子任务并收集 hop-by-hop 经验。

    Generate random intra-domain sub-tasks and collect hop-by-hop
    experiences. Same logic as kpath version but uses route_intra_hop.

    Parameters
    ----------
    lower_model : SB3 BaseAlgorithm
        下层智能体模型。
    sim : NetworkSimEngine
        网络仿真引擎。
    n_subtasks : int
        每回合子任务数量。
    rng : np.random.Generator
        随机数生成器。

    Returns
    -------
    list[Experience]
        本回合的逐跳经验列表（每个子任务可能产生多条）。
    """
    exps: list[Experience] = []

    for _ in range(n_subtasks):
        domain_id = int(rng.integers(0, sim.n_domains))
        sats = sim.domain.get_domain_sats(domain_id)
        if len(sats) < 2:
            continue
        entry, exit_ = rng.choice(sats, size=2, replace=False)
        entry, exit_ = int(entry), int(exit_)

        fid = sim.flow_manager.next_flow_id()
        flow = FlowRequest(
            flow_id=fid,
            src_sat=entry,
            dst_sat=exit_,
            bandwidth=float(rng.uniform(10, 200)),
            max_delay=float(rng.uniform(0.02, 0.10)),
            duration=int(rng.integers(5, 31)),
            arrival_timeslot=sim.current_timeslot,
        )

        seg_exps = route_intra_hop(
            lower_model, sim, entry, exit_, flow, domain_id,
            deterministic=False,
        )
        exps.extend(seg_exps)

        seg_info = seg_exps[-1].info if seg_exps else {}
        if seg_info.get("segment_ok") and seg_info.get("segment_path"):
            sim.flow_manager.register(
                flow, seg_info["segment_path"], seg_info["segment_delay"])

    return exps


def train_lower(cfg: dict):
    """Phase 1: 下层智能体预训练主函数。

    Pre-train the lower agent using random intra-domain sub-tasks.

    训练循环详解 / Training Loop Detail:
      1. 初始化 SimEngine 和 DQN 模型
      2. for each episode:
         a. 重置 sim (随机起始时隙 + 随机推进若干步)
         b. 生成 n_subtasks 个子任务 → 收集 Experience
         c. 手动将 Experience 插入 SB3 replay_buffer
            → replay_buffer.add(obs, next_obs, action, reward, done, infos)
         d. 当缓冲区样本数 >= batch_size 时:
            → model.train(gradient_steps=N)
      3. 保存模型

    Note: 此处使用 SB3 的 **手动训练模式** (而非 model.learn()),
    因为我们的环境不是标准 Gym 环境，而是通过 rollout.py 手动驱动的。

    Parameters
    ----------
    cfg : dict
        从 YAML 配置文件加载的完整配置字典。
        关键字段: training.dqn.* (DQN 超参), training.pretrain_episodes,
        routing.lower_mode, routing.K, constellation.*, etc.

    Returns
    -------
    SB3 DQN model
        训练好的下层模型（同时保存到磁盘）。
    """
    # ── 创建模型工厂 (算法由配置决定) ──
    factory = create_factory_from_config(cfg)

    seed = cfg["training"]["seed"]
    rng = np.random.default_rng(seed)
    lower_mode = cfg["routing"]["lower_mode"]
    K = cfg["routing"]["K"]
    dqn_cfg = cfg["training"]["dqn"]

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

    # ── 根据路由模式创建对应的 Shell 环境 ──
    # Shell 环境是仅用于 SB3 模型初始化的虚拟 Gym 环境，
    # 提供 observation_space 和 action_space 定义。
    # 实际交互通过 rollout.py 进行。
    if lower_mode == "k_path":
        lower_env = LowerKPathEnvShell(sim, K=K)
    else:
        lower_env = LowerHopEnvShell(sim)

    # ── 创建 DQN 模型 ──
    # 工厂根据 cfg["training"]["algorithm"] 自动选择算法 (DQN/QRDQN/...)
    lower_model = factory.create_model(
        lower_env, dqn_cfg, seed=seed,
    )

    n_episodes = cfg["training"]["pretrain_episodes"]
    n_subtasks = cfg["training"]["pretrain_subtasks_per_episode"]

    print(f"\n{'='*60}")
    print(f"  Phase 1: Lower pre-training ({lower_mode})")
    print(f"  Episodes: {n_episodes}, Sub-tasks/ep: {n_subtasks}")
    print(f"{'='*60}\n")

    logger = TrainingLogger(
        log_dir="runs", prefix=f"phase1_lower_{lower_mode}",
        console_interval=max(1, n_episodes // 20),
    )

    total_exps = 0
    t0 = time.time()

    for ep in range(n_episodes):
        # 重置 sim: 清空带宽 + 流, 随机选择起始时隙 (获得不同卫星位置)
        sim.reset(start_timeslot=int(rng.integers(1, sim.num_timeslots + 1)))

        # 随机推进若干时隙 → 增加卫星位置多样性
        for _ in range(int(rng.integers(0, 10))):
            sim.advance_timeslot()

        # 生成子任务并收集经验
        if lower_mode == "k_path":
            exps = pretrain_lower_episode_kpath(
                lower_model, sim, n_subtasks, K, rng)
        else:
            exps = pretrain_lower_episode_hop(
                lower_model, sim, n_subtasks, rng)

        # ── 手动插入 SB3 回放缓冲区 ──
        # 工厂封装了 off-policy 算法通用的 replay_buffer.add() 调用，
        # 使训练脚本无需关心底层算法差异。
        n_added = 0
        for exp in exps:
            factory.add_experience(lower_model, exp)
            n_added += 1
        total_exps += n_added

        # ── 梯度更新 ──
        # 当累积样本数 >= batch_size 时，从 replay buffer 采样进行训练。
        # gradient_steps 正比于本回合新增经验数，避免过度/不足训练。
        if factory.should_train(lower_model, total_exps):
            grad_steps = max(1, n_added // 4)
            factory.train_step(lower_model, gradient_steps=grad_steps)

        # Logging
        n_success = sum(1 for e in exps if e.info.get("segment_ok", False))
        n_fail = len([e for e in exps if e.info.get("segment_ok") is not None]) - n_success
        avg_reward = np.mean([e.reward for e in exps]) if exps else 0.0
        logger.log_episode(ep, n_episodes, {
            "lower_exps": n_added,
            "avg_reward": avg_reward,
            "success_count": n_success,
            "fail_count": max(0, n_fail),
            "total_exps": total_exps,
        })

    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f"lower_{lower_mode}_pretrained"
    factory.save_model(lower_model, str(save_path))
    logger.close()
    print(f"\n  ✓ Lower model saved to {save_path}")
    print(f"  Total experiences: {total_exps}")
    print(f"  Total time: {time.time() - t0:.0f}s")

    return lower_model


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Lower agent pre-training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_lower(cfg)


if __name__ == "__main__":
    main()
