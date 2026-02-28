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


def test_core_modules():
    """Test core/ modules independently."""
    print("=" * 60)
    print("  [1/4] Testing core modules")
    print("=" * 60)

    from env.sim_engine import NetworkSimEngine

    t0 = time.time()
    sim = NetworkSimEngine(
        constellation_name="Kuiper",
        shell_idx=0,
        dT=120,
        n_domains=2,          # 34 orbits / 2 = 17 orbits per domain
        link_capacity_mbps=2000.0,
        arrival_rate=3.0,
        seed=42,
    )
    print(f"  SimEngine init: {time.time() - t0:.1f}s")

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
    return sim


def test_env_shells(sim):
    """Test SB3 model initialisation via env shells."""
    print("=" * 60)
    print("  [2/4] Testing env shells + SB3 model init")
    print("=" * 60)

    from stable_baselines3 import DQN
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

    t0 = time.time()
    upper_model = DQN(
        "MlpPolicy", upper_env,
        learning_rate=1e-4,
        buffer_size=10000,
        batch_size=32,
        learning_starts=50,
        verbose=0,
    )
    lower_model = DQN(
        "MlpPolicy", lower_kpath_env,
        learning_rate=1e-4,
        buffer_size=10000,
        batch_size=32,
        learning_starts=50,
        verbose=0,
    )
    print(f"  SB3 DQN models created in {time.time() - t0:.1f}s")

    # Smoke-test predict
    obs = np.zeros(upper_env.observation_space.shape, dtype=np.float32)
    action, _ = upper_model.predict(obs, deterministic=True)
    print(f"  Upper predict test: action={action}")

    obs = np.zeros(lower_kpath_env.observation_space.shape, dtype=np.float32)
    action, _ = lower_model.predict(obs, deterministic=True)
    print(f"  Lower predict test: action={action}")

    print("  ✓ Env shells + SB3 OK\n")
    return upper_model, lower_model


def test_rollout(sim, upper_model, lower_model):
    """Test a short episode rollout."""
    print("=" * 60)
    print("  [3/4] Testing rollout (5 timeslots)")
    print("=" * 60)

    from train.rollout import run_episode

    t0 = time.time()
    upper_exps, lower_exps, metrics = run_episode(
        upper_model, lower_model, sim,
        T_episode=5,
        lower_mode="k_path",
        K=4,
        deterministic=False,
    )
    elapsed = time.time() - t0

    print(f"  Episode done in {elapsed:.1f}s")
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
    return upper_exps, lower_exps


def test_replay_buffer(upper_model, lower_model, upper_exps, lower_exps):
    """Test manual insertion into SB3 replay buffers."""
    print("=" * 60)
    print("  [4/4] Testing replay buffer insertion + training")
    print("=" * 60)

    import torch as th

    n_upper = 0
    for exp in upper_exps:
        obs = exp.obs.reshape(1, -1)
        next_obs = exp.next_obs.reshape(1, -1)
        action = np.array([[exp.action]])
        reward = np.array([exp.reward])
        done = np.array([exp.done or exp.truncated])
        infos = [{}]
        upper_model.replay_buffer.add(obs, next_obs, action, reward, done, infos)
        n_upper += 1

    n_lower = 0
    for exp in lower_exps:
        obs = exp.obs.reshape(1, -1)
        next_obs = exp.next_obs.reshape(1, -1)
        action = np.array([[exp.action]])
        reward = np.array([exp.reward])
        done = np.array([exp.done or exp.truncated])
        infos = [{}]
        lower_model.replay_buffer.add(obs, next_obs, action, reward, done, infos)
        n_lower += 1

    print(f"  Inserted {n_upper} upper, {n_lower} lower experiences")
    print(f"  Upper buffer pos: {upper_model.replay_buffer.pos}")
    print(f"  Lower buffer pos: {lower_model.replay_buffer.pos}")

    # Try training if enough samples
    min_samples = max(upper_model.batch_size, lower_model.batch_size)
    if n_upper >= min_samples:
        upper_model.train(gradient_steps=2)
        print(f"  Upper model trained (2 gradient steps)")
    else:
        print(f"  Upper: not enough samples ({n_upper}/{min_samples}) to train")

    if n_lower >= min_samples:
        lower_model.train(gradient_steps=2)
        print(f"  Lower model trained (2 gradient steps)")
    else:
        print(f"  Lower: not enough samples ({n_lower}/{min_samples}) to train")

    print("  ✓ Replay buffer + training OK\n")


def main():
    print("\n" + "=" * 60)
    print("  HRL Satellite Routing — Integration Test")
    print("=" * 60 + "\n")

    t_total = time.time()

    sim = test_core_modules()
    upper_model, lower_model = test_env_shells(sim)
    upper_exps, lower_exps = test_rollout(sim, upper_model, lower_model)
    test_replay_buffer(upper_model, lower_model, upper_exps, lower_exps)

    print("=" * 60)
    print(f"  ALL TESTS PASSED  ({time.time() - t_total:.1f}s total)")
    print("=" * 60)


if __name__ == "__main__":
    main()
