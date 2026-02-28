"""
Evaluate — HRL 智能体 vs 基线策略对比评估。

Compare trained HRL agents against baseline routing strategies.

评估指标 / Metrics:
  - 路由成功率 (Success Rate)
  - 平均时延 (Average Delay)
  - P95 时延 (95th Percentile Delay)
  - 成功/失败流计数

评估流程 / Evaluation Flow:
  1. 构建 SimEngine
  2. 加载训练好的 HRL 模型 (确定性决策, deterministic=True)
  3. 运行多个回合，收集指标
  4. 同样条件下运行基线策略 (dijkstra/greedy/random)
  5. 打印对比表格

Usage:
    python -m eval.evaluate --config configs/default.yaml \
           --upper-ckpt checkpoints/upper_finetuned.zip \
           --lower-ckpt checkpoints/lower_k_path_finetuned.zip
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
from train.rollout import run_episode, EpisodeMetrics
from eval.baselines import run_baseline_episode


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate_hrl(
    sim: NetworkSimEngine,
    upper_model,
    lower_model,
    cfg: dict,
    n_episodes: int,
) -> dict:
    """评估训练好的 HRL 智能体。

    Evaluate trained HRL agents over multiple episodes.
    使用 deterministic=True 确保可复现、无探索噪声。

    Parameters
    ----------
    sim : NetworkSimEngine
        仿真引擎。
    upper_model, lower_model
        训练好的 SB3 模型。
    cfg : dict
        配置字典。
    n_episodes : int
        评估回合数。

    Returns
    -------
    dict
        包含 success_rate, avg_delay, p95_delay 等指标。
    """
    lower_mode = cfg["routing"]["lower_mode"]
    K = cfg["routing"]["K"]
    T_episode = cfg["training"]["T_episode"]

    all_delays: list[float] = []
    total_success = 0
    total_fail = 0

    for ep in range(n_episodes):
        upper_exps, lower_exps, ep_metrics = run_episode(
            upper_model, lower_model, sim,
            T_episode=T_episode,
            lower_mode=lower_mode,
            K=K,
            deterministic=True,
        )
        total_success += ep_metrics.success_count
        total_fail += ep_metrics.fail_count
        all_delays.extend(ep_metrics.delays)

    total = total_success + total_fail
    return {
        "strategy": f"HRL ({lower_mode})",
        "success_count": total_success,
        "fail_count": total_fail,
        "success_rate": total_success / max(total, 1),
        "avg_delay": float(np.mean(all_delays)) if all_delays else float("inf"),
        "p95_delay": float(np.percentile(all_delays, 95)) if all_delays else float("inf"),
        "n_episodes": n_episodes,
    }


def evaluate_baselines(
    sim: NetworkSimEngine,
    cfg: dict,
    n_episodes: int,
) -> list[dict]:
    """Evaluate all baselines."""
    K = cfg["routing"]["K"]
    T_episode = cfg["training"]["T_episode"]
    rng = np.random.default_rng(cfg["training"]["seed"])

    results = []
    for strategy in ["dijkstra", "greedy", "random"]:
        print(f"  Evaluating baseline: {strategy} ...")
        ep_results: list[dict] = []
        for ep in range(n_episodes):
            r = run_baseline_episode(
                sim, strategy=strategy, T_episode=T_episode, K=K, rng=rng)
            ep_results.append(r)

        # Aggregate
        all_delays = []
        total_succ = 0
        total_fail = 0
        for r in ep_results:
            total_succ += r["success_count"]
            total_fail += r["fail_count"]
            all_delays.extend(r["delays"])

        total = total_succ + total_fail
        results.append({
            "strategy": strategy,
            "success_count": total_succ,
            "fail_count": total_fail,
            "success_rate": total_succ / max(total, 1),
            "avg_delay": float(np.mean(all_delays)) if all_delays else float("inf"),
            "p95_delay": float(np.percentile(all_delays, 95)) if all_delays else float("inf"),
            "n_episodes": n_episodes,
        })

    return results


def print_results_table(results: list[dict]):
    """Pretty-print comparison table."""
    print(f"\n{'='*72}")
    print(f"  {'Strategy':<20s} {'SR':>8s} {'Avg Delay':>12s} "
          f"{'P95 Delay':>12s} {'Succ':>6s} {'Fail':>6s}")
    print(f"  {'-'*66}")
    for r in results:
        sr = f"{r['success_rate']:.2%}"
        avg = f"{r['avg_delay']*1000:.2f} ms" if r['avg_delay'] < 1 else "N/A"
        p95 = f"{r['p95_delay']*1000:.2f} ms" if r['p95_delay'] < 1 else "N/A"
        print(f"  {r['strategy']:<20s} {sr:>8s} {avg:>12s} "
              f"{p95:>12s} {r['success_count']:>6d} {r['fail_count']:>6d}")
    print(f"{'='*72}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate HRL vs baselines")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--upper-ckpt", type=str, default=None)
    parser.add_argument("--lower-ckpt", type=str, default=None)
    parser.add_argument("--n-episodes", type=int, default=None,
                        help="Override eval_episodes from config")
    parser.add_argument("--baselines-only", action="store_true",
                        help="Only run baseline evaluations")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_episodes = args.n_episodes or cfg["evaluation"]["eval_episodes"]

    # Build sim
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
        seed=cfg["training"]["seed"],
    )

    all_results: list[dict] = []

    # ── HRL evaluation ──
    if not args.baselines_only and args.upper_ckpt and args.lower_ckpt:
        factory = create_factory_from_config(cfg)

        lower_mode = cfg["routing"]["lower_mode"]
        K = cfg["routing"]["K"]

        upper_env = UpperEnvShell(sim)
        if lower_mode == "k_path":
            lower_env = LowerKPathEnvShell(sim, K=K)
        else:
            lower_env = LowerHopEnvShell(sim)

        upper_model = factory.load_model(args.upper_ckpt, env=upper_env)
        lower_model = factory.load_model(args.lower_ckpt, env=lower_env)

        print(f"\n  Evaluating HRL ({lower_mode}) ...")
        hrl_result = evaluate_hrl(sim, upper_model, lower_model, cfg, n_episodes)
        all_results.append(hrl_result)

    # ── Baselines ──
    baseline_results = evaluate_baselines(sim, cfg, n_episodes)
    all_results.extend(baseline_results)

    print_results_table(all_results)


if __name__ == "__main__":
    main()
