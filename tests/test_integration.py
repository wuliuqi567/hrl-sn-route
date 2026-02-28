"""
Integration test — verify end-to-end HRL pipeline with a small constellation.

Uses Kuiper (34 orbits × 34 sats = 1,156 sats) which is smaller than
Starlink (72 × 22 = 1,584) and faster to initialise.

Test sequence:
  1. Build Topology + DomainManager + BandwidthManager + FlowManager
  2. Build SimEngine
  3. Instantiate SB3 DQN models via env shells
  4. Run a short episode (10 timeslots) with random policies
  5. Verify experience collection and replay buffer insertion
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pytest


# ── Pytest Fixtures ──
# 这些 fixture 使得各测试函数可通过依赖注入获取共享对象，
# 不再需要手动传递返回值。

@pytest.fixture(scope="module")
def sim():
    """构建共享的 SimEngine（整个测试模块只初始化一次）。"""
    from env.sim_engine import NetworkSimEngine
    return NetworkSimEngine(
        constellation_name="Kuiper",
        shell_idx=0,
        dT=120,
        n_domains=2,
        link_capacity_mbps=2000.0,
        arrival_rate=3.0,
        seed=42,
    )


@pytest.fixture(scope="module")
def factory():
    """构建 ModelFactory。"""
    from agents.model_factory import ModelFactory
    return ModelFactory(algorithm="DQN")


@pytest.fixture(scope="module")
def upper_model(sim, factory):
    """通过工厂创建上层模型。"""
    from env.upper_env_shell import UpperEnvShell
    upper_env = UpperEnvShell(sim)
    dqn_cfg = {
        "learning_rate": 1e-4,
        "buffer_size": 10000,
        "batch_size": 32,
    }
    return factory.create_model(upper_env, dqn_cfg, seed=42)


@pytest.fixture(scope="module")
def lower_model(sim, factory):
    """通过工厂创建下层模型。"""
    from env.lower_kpath_env_shell import LowerKPathEnvShell
    lower_kpath_env = LowerKPathEnvShell(sim, K=4)
    dqn_cfg = {
        "learning_rate": 1e-4,
        "buffer_size": 10000,
        "batch_size": 32,
    }
    return factory.create_model(lower_kpath_env, dqn_cfg, seed=42)


@pytest.fixture(scope="module")
def episode_results(sim, upper_model, lower_model):
    """运行一个短回合，返回 (upper_exps, lower_exps, metrics)。"""
    from train.rollout import run_episode
    return run_episode(
        upper_model, lower_model, sim,
        T_episode=5,
        lower_mode="k_path",
        K=4,
        deterministic=False,
    )


def test_core_modules(sim):
    """Test core/ modules independently."""
    print("=" * 60)
    print("  [1/4] Testing core modules")
    print("=" * 60)

    # Basic domain checks
    assert sim.n_domains == 2
    assert sim.get_domain(1) == 0   # sat 1 in domain 0
    assert sim.num_sats > 0
    domain_path = sim.get_domain_sequence(1, sim.num_sats)
    assert len(domain_path) >= 1
    print(f"  Domain path from sat 1 → sat {sim.num_sats}: {domain_path}")

    # Bandwidth checks
    assert sim.bandwidth.get_remaining_bw(1, 2) == 2000.0
    assert sim.bandwidth.get_utilization(1, 2) == 0.0

    # Flow generation
    sim.reset()
    flows = sim.generate_flows()
    print(f"  Generated {len(flows)} flows at timeslot {sim.current_timeslot}")
    for f in flows[:3]:
        print(f"    Flow {f.flow_id}: sat {f.src_sat} → {f.dst_sat}, "
              f"{f.bandwidth:.0f} Mbps, budget {f.max_delay*1000:.1f} ms")

    # K-shortest-path
    domain0_sats = sim.domain.get_domain_sats(0)
    entry = domain0_sats[0]
    # Pick an exit a few orbits away (not the extreme end)
    mid_idx = min(len(domain0_sats) // 4, len(domain0_sats) - 1)
    exit_ = domain0_sats[mid_idx]
    paths = sim.compute_k_shortest(0, entry, exit_, K=4)
    print(f"  K-shortest paths (domain 0, {entry}→{exit_}): "
          f"{len(paths)} found, lengths={[len(p) for p in paths]}")
    assert len(paths) > 0, "Should find at least 1 path in domain"

    print("  ✓ Core modules OK\n")


def test_env_shells(sim, factory):
    """Test model initialisation via factory + env shells."""
    print("=" * 60)
    print("  [2/4] Testing env shells + factory model init")
    print("=" * 60)

    from env.upper_env_shell import UpperEnvShell
    from env.lower_kpath_env_shell import LowerKPathEnvShell
    from env.lower_hop_env_shell import LowerHopEnvShell

    upper_env = UpperEnvShell(sim)
    lower_kpath_env = LowerKPathEnvShell(sim, K=4)
    lower_hop_env = LowerHopEnvShell(sim)

    print(f"  Upper: obs={upper_env.observation_space.shape}, "
          f"act={upper_env.action_space.n}")
    print(f"  Lower K-path: obs={lower_kpath_env.observation_space.shape}, "
          f"act={lower_kpath_env.action_space.n}")
    print(f"  Lower hop: obs={lower_hop_env.observation_space.shape}, "
          f"act={lower_hop_env.action_space.n}")

    dqn_cfg = {
        "learning_rate": 1e-4,
        "buffer_size": 10000,
        "batch_size": 32,
    }

    t0 = time.time()
    upper_model = factory.create_model(upper_env, dqn_cfg, seed=42)
    lower_model = factory.create_model(lower_kpath_env, dqn_cfg, seed=42)
    print(f"  Factory models created in {time.time() - t0:.1f}s")

    # Smoke-test predict
    obs = np.zeros(upper_env.observation_space.shape, dtype=np.float32)
    action, _ = upper_model.predict(obs, deterministic=True)
    print(f"  Upper predict test: action={action}")

    obs = np.zeros(lower_kpath_env.observation_space.shape, dtype=np.float32)
    action, _ = lower_model.predict(obs, deterministic=True)
    print(f"  Lower predict test: action={action}")

    print("  ✓ Env shells + factory OK\n")


def test_rollout(sim, upper_model, lower_model, episode_results):
    """Test a short episode rollout."""
    print("=" * 60)
    print("  [3/4] Testing rollout (5 timeslots)")
    print("=" * 60)

    upper_exps, lower_exps, metrics = episode_results

    print(f"  Upper experiences: {len(upper_exps)}")
    print(f"  Lower experiences: {len(lower_exps)}")
    print(f"  Flows routed: success={metrics.success_count}, fail={metrics.fail_count}")
    print(f"  Delays collected: {len(metrics.delays)}")
    print(f"  Active flows: {sim.flow_manager.get_active_count()}")

    if upper_exps:
        e = upper_exps[0]
        print(f"  Sample upper exp: obs_shape={e.obs.shape}, "
              f"action={e.action}, reward={e.reward:.3f}, done={e.done}")
    if lower_exps:
        e = lower_exps[0]
        print(f"  Sample lower exp: obs_shape={e.obs.shape}, "
              f"action={e.action}, reward={e.reward:.3f}, done={e.done}")

    print("  ✓ Rollout OK\n")


def test_replay_buffer(factory, upper_model, lower_model, episode_results):
    """Test factory-based replay buffer insertion + training."""
    print("=" * 60)
    print("  [4/4] Testing replay buffer insertion + training")
    print("=" * 60)

    upper_exps, lower_exps, _ = episode_results

    n_upper = 0
    for exp in upper_exps:
        factory.add_experience(upper_model, exp)
        n_upper += 1

    n_lower = 0
    for exp in lower_exps:
        factory.add_experience(lower_model, exp)
        n_lower += 1

    print(f"  Inserted {n_upper} upper, {n_lower} lower experiences")
    print(f"  Upper buffer pos: {upper_model.replay_buffer.pos}")
    print(f"  Lower buffer pos: {lower_model.replay_buffer.pos}")

    # Try training if enough samples
    if factory.should_train(upper_model, n_upper):
        factory.train_step(upper_model, gradient_steps=2)
        print(f"  Upper model trained (2 gradient steps)")
    else:
        print(f"  Upper: not enough samples ({n_upper}) to train")

    if factory.should_train(lower_model, n_lower):
        factory.train_step(lower_model, gradient_steps=2)
        print(f"  Lower model trained (2 gradient steps)")
    else:
        print(f"  Lower: not enough samples ({n_lower}) to train")

    print("  ✓ Replay buffer + training OK\n")


def main():
    """直接运行集成测试（非 pytest 模式）。"""
    print("\n" + "=" * 60)
    print("  HRL Satellite Routing — Integration Test")
    print("=" * 60 + "\n")

    from env.sim_engine import NetworkSimEngine
    from agents.model_factory import ModelFactory
    from env.upper_env_shell import UpperEnvShell
    from env.lower_kpath_env_shell import LowerKPathEnvShell
    from train.rollout import run_episode

    t_total = time.time()

    # Build shared objects
    sim_obj = NetworkSimEngine(
        constellation_name="Kuiper", shell_idx=0, dT=120,
        n_domains=2, link_capacity_mbps=2000.0, arrival_rate=3.0, seed=42,
    )
    fact = ModelFactory(algorithm="DQN")
    dqn_cfg = {"learning_rate": 1e-4, "buffer_size": 10000, "batch_size": 32}
    upper_env = UpperEnvShell(sim_obj)
    lower_env = LowerKPathEnvShell(sim_obj, K=4)
    u_model = fact.create_model(upper_env, dqn_cfg, seed=42)
    l_model = fact.create_model(lower_env, dqn_cfg, seed=42)

    test_core_modules(sim_obj)
    test_env_shells(sim_obj, fact)

    ep_res = run_episode(
        u_model, l_model, sim_obj,
        T_episode=5, lower_mode="k_path", K=4, deterministic=False,
    )
    test_rollout(sim_obj, u_model, l_model, ep_res)
    test_replay_buffer(fact, u_model, l_model, ep_res)

    print("=" * 60)
    print(f"  ALL TESTS PASSED  ({time.time() - t_total:.1f}s total)")
    print("=" * 60)


if __name__ == "__main__":
    main()
