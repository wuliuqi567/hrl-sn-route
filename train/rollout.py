"""
Rollout — 分层强化学习 (HRL) 卫星路由的回合执行逻辑。

本模块是训练循环和 RL 智能体之间的 **算法无关桥梁层**：
  - 只通过 model.predict(obs, deterministic) 与 SB3 模型交互
  - 不直接引用 DQN 或任何特定算法的 API
  - 返回通用 Experience 元组，由训练脚本插入对应算法的回放缓冲区

Architecture / 架构角色:
  ┌──────────────────────────────────────┐
  │ train_*.py  (算法特定：DQN / DDQN …) │  ← 创建模型 + 训练循环
  │       ↓  model  ↓                    │
  │   rollout.py  (算法无关)             │  ← 回合执行 + 经验收集
  │       ↓  sim   ↓                    │
  │   sim_engine.py (环境状态机)         │  ← 网络仿真
  └──────────────────────────────────────┘

Public API / 公开接口:
  - run_episode()            完整回合（T 时隙，Poisson 到达）
  - route_cross_domain()     跨域流路由（上层 + 下层智能体协作）
  - route_intra_domain()     域内流路由分发器（仅下层智能体）
  - route_intra_kpath()      K 最短路选择模式（单步 contextual bandit）
  - route_intra_hop()        逐跳导航模式（多步 MDP）

数据流 / Data Flow:
  1. run_episode 循环 T 时隙
  2. 每时隙生成新流 → 判断跨域/域内
  3. 跨域: route_cross_domain → 上层选中继链路 + 下层域内路由
  4. 域内: route_intra_domain → K-path 或 hop-by-hop
  5. 收集 Experience 元组返回给训练脚本
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.flow import FlowRequest
from env.sim_engine import NetworkSimEngine


# ── Experience tuple / 经验元组 ───────────────────────────────────
@dataclass
class Experience:
    """一次状态转移的记录，用于 SB3 回放缓冲区 (replay buffer)。

    A single (s, a, r, s', done) transition compatible with SB3's
    off-policy replay buffer API.

    Attributes
    ----------
    obs : np.ndarray
        当前观测向量 s。
    next_obs : np.ndarray
        下一步观测向量 s'。对于 done=True 的终止步，next_obs 可为全零。
    action : int
        智能体选取的离散动作索引。
    reward : float
        即时奖励 r。
    done : bool
        是否自然终止（到达目的地 / 路由失败 / K-path 单步结束）。
    truncated : bool
        是否因步数超限被截断（仅 hop-by-hop 模式使用）。
    info : dict
        附加信息，如 ``segment_path``、``segment_ok``、``segment_delay``，
        供训练脚本和指标收集使用。
    """
    obs: np.ndarray
    next_obs: np.ndarray
    action: int
    reward: float
    done: bool          # terminated / 自然终止
    truncated: bool = False
    info: dict = field(default_factory=dict)


# =====================================================================
#  奖励系数 (Reward Coefficients)
#  这些系数控制各智能体的奖励信号强度，可通过调参优化策略学习效果。
# =====================================================================

# ── 上层智能体 (Upper Agent: 跨域中继链路选择) ──
UPPER_UTIL_COEFF = -1.0       # 链路利用率惩罚：鼓励选择低负载链路
UPPER_DELAY_COEFF = -5.0      # 链路时延惩罚：鼓励选择低时延链路
UPPER_SUCCESS_BONUS = 5.0     # 端到端路由成功奖励
UPPER_FAIL_PENALTY = -5.0     # 路由失败惩罚
UPPER_DELAY_NORM = 0.1        # 最终奖励中时延归一化分母 (秒)

# ── 下层 K-path 智能体 (Lower K-path Agent: K 条候选路径选择) ──
KPATH_BW_COEFF = 1.0          # 瓶颈带宽奖励：鼓励选择带宽充裕的路径
KPATH_DELAY_COEFF = -10.0     # 路径时延惩罚
KPATH_FAIL_PENALTY = -3.0     # 带宽不足 / 无路径惩罚

# ── 下层逐跳智能体 (Lower Hop-by-hop Agent: 逐步导航到出口) ──
HOP_DELAY_COEFF = -20.0       # 每跳时延惩罚
HOP_REVISIT_PENALTY = -0.3    # 重复访问节点惩罚：防止环路
HOP_INVALID_PENALTY = -0.5    # 无效动作惩罚（链路不存在 / 出域）
HOP_BW_PENALTY = -0.5         # 带宽不足惩罚
HOP_ARRIVE_BONUS = 5.0        # 到达出口奖励（乘以剩余步数比例）
HOP_TRUNCATE_PENALTY = -2.0   # 步数耗尽未到达的截断惩罚


# ==================================================================
#  回合运行器 (Episode Runner)
# ==================================================================
@dataclass
class EpisodeMetrics:
    """一个回合的聚合指标。

    Aggregated metrics for one episode, used for logging and evaluation.

    Attributes
    ----------
    success_count : int
        本回合成功路由的流数量。
    fail_count : int
        本回合路由失败的流数量。
    delays : list[float]
        所有成功路由流的端到端时延列表（秒）。
    """
    success_count: int = 0
    fail_count: int = 0
    delays: list[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """路由成功率 = 成功数 / (成功数 + 失败数)。"""
        total = self.success_count + self.fail_count
        return self.success_count / max(total, 1)


def run_episode(
    upper_model,
    lower_model,
    sim: NetworkSimEngine,
    T_episode: int = 100,
    lower_mode: str = "k_path",
    K: int = 4,
    deterministic: bool = False,
) -> tuple[list[Experience], list[Experience], EpisodeMetrics]:
    """运行一个完整的回合，包含 T_episode 个时隙。

    Run a full episode of T_episode timeslots. This is the main entry
    point called by all training scripts (Phase 1-3).

    运行流程 / Execution Flow:
      for each timeslot t in [0, T_episode):
        1. 释放已过期的流 (release expired flows)
        2. 推进时隙 (advance timeslot → 更新卫星位置和链路时延)
        3. 生成新流请求 (Poisson arrival)
        4. 对每个新流判断:
           - 同域流 → route_intra_domain (仅下层智能体)
           - 跨域流 → route_cross_domain (上层 + 下层协作)
        5. 收集 Experience 元组

    Parameters
    ----------
    upper_model : SB3 BaseAlgorithm
        上层智能体模型，用于跨域中继链路选择。
        仅调用 model.predict(obs, deterministic) — 算法无关。
    lower_model : SB3 BaseAlgorithm
        下层智能体模型，用于域内路由。
        仅调用 model.predict(obs, deterministic) — 算法无关。
    sim : NetworkSimEngine
        网络仿真引擎，管理拓扑、带宽、流等状态。
    T_episode : int
        每回合的时隙数量，默认 100。
    lower_mode : str
        下层路由模式: ``"k_path"`` (K最短路选择) 或 ``"hop_by_hop"`` (逐跳导航)。
    K : int
        K 最短路候选数量（仅 k_path 模式使用）。
    deterministic : bool
        若 True，模型贪心决策（评估模式）；否则带探索（训练模式）。

    Returns
    -------
    tuple[list[Experience], list[Experience], EpisodeMetrics]
        (上层经验列表, 下层经验列表, 回合指标)。
        训练脚本负责将经验插入对应模型的回放缓冲区。
    """
    # ── 重置仿真环境 (清空带宽、流、统计计数器) ──
    sim.reset()
    upper_exps: list[Experience] = []   # 上层智能体经验列表
    lower_exps: list[Experience] = []   # 下层智能体经验列表
    metrics = EpisodeMetrics()

    for t in range(T_episode):
        # 1. 释放已过期的流 → 归还带宽资源
        sim.release_expired_flows()

        # 2. 推进时隙 → 卫星位置变化 → 链路时延更新
        sim.advance_timeslot()

        # 3. 按 Poisson 过程生成新流请求
        new_flows = sim.generate_flows()

        # 4. 逐流路由决策
        for flow in new_flows:
            src_domain = sim.get_domain(flow.src_sat)
            dst_domain = sim.get_domain(flow.dst_sat)

            if src_domain == dst_domain:
                # ── 同域流: 仅下层智能体参与 ──
                l_exps = route_intra_domain(
                    lower_model, sim, flow.src_sat, flow.dst_sat,
                    flow, src_domain, lower_mode, K, deterministic,
                )
                lower_exps.extend(l_exps)
                # 从最后一条经验的 info 字典中提取路由结果
                if l_exps:
                    seg_info = l_exps[-1].info
                    if seg_info.get("segment_ok"):
                        metrics.success_count += 1
                        metrics.delays.append(seg_info.get("segment_delay", 0.0))
                    else:
                        metrics.fail_count += 1
            else:
                # ── 跨域流: 上层 + 下层协作 ──
                # 上层选择域间中继链路，下层在每个域内寻路
                u_exps, l_exps = route_cross_domain(
                    upper_model, lower_model, sim, flow,
                    lower_mode, K, deterministic,
                )
                upper_exps.extend(u_exps)
                lower_exps.extend(l_exps)
                # 跨域结果由 sim 的权威计数器追踪 (见下方同步)

    # ── 与 sim 权威计数器同步 ──
    # route_cross_domain 内部调用了 sim.register_active_flow / record_failure,
    # 所以以 sim 的计数为准，避免重复计算。
    metrics.success_count = sim.stats_success
    metrics.fail_count = sim.stats_fail
    # 收集本回合所有成功路由流的时延（含已过期释放的流）
    metrics.delays = sim.flow_manager.get_all_delays()

    return upper_exps, lower_exps, metrics


# ==================================================================
#  跨域路由 (Cross-domain Routing)
#  上层智能体选择域间中继卫星，下层智能体在每段域内寻路
# ==================================================================
def route_cross_domain(
    upper_model,
    lower_model,
    sim: NetworkSimEngine,
    flow: FlowRequest,
    lower_mode: str = "k_path",
    K: int = 4,
    deterministic: bool = False,
) -> tuple[list[Experience], list[Experience]]:
    """路由一个跨越多个域的流请求。

    Route a flow that crosses one or more domain boundaries.

    算法流程 / Algorithm:
      1. 计算域级最短路径序列: [D_src, D_1, D_2, ..., D_dst]
      2. 对序列中每对相邻域 (D_i, D_{i+1}):
         a. 上层智能体观测 → predict → 选择域间中继链路 (exit_sat, entry_sat)
         b. 下层智能体在 D_i 内从 current_entry → exit_sat 寻路
         c. 分配域间链路带宽
         d. 收集上层/下层经验
      3. 最后一段域内: exit_sat = dst_sat (无需上层决策)
      4. 全部成功 → 注册活跃流; 任一段失败 → 记录失败

    Parameters
    ----------
    upper_model : SB3 BaseAlgorithm
        上层跨域智能体 (仅调用 predict)。
    lower_model : SB3 BaseAlgorithm
        下层域内路由智能体 (仅调用 predict)。
    sim : NetworkSimEngine
        网络仿真引擎。
    flow : FlowRequest
        待路由的流请求。
    lower_mode : str
        下层路由模式: ``"k_path"`` 或 ``"hop_by_hop"``。
    K : int
        K 最短路候选数。
    deterministic : bool
        是否贪心决策。

    Returns
    -------
    tuple[list[Experience], list[Experience]]
        (上层经验列表, 下层经验列表)。
    """
    # 计算域级路径: 源域 → … → 目的域
    domain_path = sim.get_domain_sequence(flow.src_sat, flow.dst_sat)

    upper_exps: list[Experience] = []  # 上层经验 (每跨一个域边界产生一条)
    lower_exps: list[Experience] = []  # 下层经验 (每段域内路由产生若干条)
    full_path: list[int] = []          # 累积的端到端卫星路径
    total_delay = 0.0                  # 累积的端到端时延
    success = True                     # 是否全部段都成功

    current_entry_sat = flow.src_sat   # 当前段的入口卫星

    for i, current_domain in enumerate(domain_path):
        # ── 确定当前段的出口卫星 ──
        if i < len(domain_path) - 1:
            # 非最后一段: 上层智能体选择域间中继链路
            next_domain = domain_path[i + 1]

            # Upper agent decides which inter-domain link to use
            upper_obs = sim.get_upper_obs(
                current_domain, next_domain, flow, domain_path, i)
            action_np, _ = upper_model.predict(upper_obs, deterministic=deterministic)
            upper_action = int(action_np)

            exit_sat, entry_sat_next = sim.junction_sat_from_action(
                current_domain, next_domain, upper_action)
        else:
            # Last domain — exit = destination
            exit_sat = flow.dst_sat
            entry_sat_next = None

        # ── Lower agent routes within current domain ──
        seg_exps = route_intra_domain(
            lower_model, sim, current_entry_sat, exit_sat,
            flow, current_domain, lower_mode, K, deterministic,
        )
        lower_exps.extend(seg_exps)

        # Check if intra-domain routing succeeded
        # The last experience's info carries 'segment_path' and 'segment_ok'
        seg_info = seg_exps[-1].info if seg_exps else {}
        seg_path = seg_info.get("segment_path")
        seg_ok = seg_info.get("segment_ok", False)
        seg_delay = seg_info.get("segment_delay", float("inf"))

        if not seg_ok or seg_path is None:
            success = False
            # Record upper experience with fail reward
            if i < len(domain_path) - 1:
                next_obs = np.zeros_like(upper_obs)
                upper_exps.append(Experience(
                    obs=upper_obs, next_obs=next_obs,
                    action=upper_action,
                    reward=UPPER_FAIL_PENALTY,
                    done=True,
                ))
            break

        # Append segment to full path (avoid duplicate junction nodes)
        if full_path and seg_path and full_path[-1] == seg_path[0]:
            full_path.extend(seg_path[1:])
        else:
            full_path.extend(seg_path)
        total_delay += seg_delay

        # ── Collect upper experience (intermediate reward) ──
        if i < len(domain_path) - 1:
            link_util = sim.bandwidth.get_utilization(exit_sat, entry_sat_next)
            link_delay = sim.topology.get_link_delay(
                exit_sat, entry_sat_next, sim.current_timeslot)
            total_delay += link_delay  # add inter-domain link delay

            # Allocate bandwidth on the inter-domain link itself
            inter_link_ok = sim.bandwidth.allocate(
                [exit_sat, entry_sat_next], flow.bandwidth, flow.flow_id)
            if not inter_link_ok:
                success = False
                next_obs = np.zeros_like(upper_obs)
                upper_exps.append(Experience(
                    obs=upper_obs, next_obs=next_obs,
                    action=upper_action,
                    reward=UPPER_FAIL_PENALTY,
                    done=True,
                ))
                break

            step_reward = (UPPER_UTIL_COEFF * link_util
                           + UPPER_DELAY_COEFF * link_delay)

            # For next_obs: peek at next boundary (or zeros if last)
            if i + 1 < len(domain_path) - 1:
                next_next_domain = domain_path[i + 2]
                next_upper_obs = sim.get_upper_obs(
                    next_domain, next_next_domain, flow, domain_path, i + 1)
            else:
                next_upper_obs = np.zeros_like(upper_obs)

            is_last_boundary = (i == len(domain_path) - 2)
            upper_exps.append(Experience(
                obs=upper_obs,
                next_obs=next_upper_obs,
                action=upper_action,
                reward=step_reward,
                done=is_last_boundary,
            ))

        # Move to next domain
        if entry_sat_next is not None:
            current_entry_sat = entry_sat_next

    # ── Final reward adjustment ──
    if success and upper_exps:
        final_reward = (UPPER_SUCCESS_BONUS
                        - abs(UPPER_DELAY_COEFF) * total_delay / UPPER_DELAY_NORM)
        # Add final reward to the last upper experience
        upper_exps[-1] = Experience(
            obs=upper_exps[-1].obs,
            next_obs=upper_exps[-1].next_obs,
            action=upper_exps[-1].action,
            reward=upper_exps[-1].reward + final_reward,
            done=True,
        )

    # Register or record failure
    if success and full_path:
        sim.register_active_flow(flow, full_path, total_delay)
    else:
        sim.record_failure()

    return upper_exps, lower_exps


# ==================================================================
#  统一域内路由分发器 (Intra-domain Routing Dispatcher)
# ==================================================================
def route_intra_domain(
    lower_model,
    sim: NetworkSimEngine,
    entry_sat: int,
    exit_sat: int,
    flow: FlowRequest,
    domain_id: int,
    mode: str = "k_path",
    K: int = 4,
    deterministic: bool = False,
) -> list[Experience]:
    """在单个域内路由，根据 mode 分发到 K-path 或 hop-by-hop。

    Route within a single domain. Dispatches to the appropriate
    routing strategy based on the ``mode`` parameter.

    Parameters
    ----------
    lower_model : SB3 BaseAlgorithm
        下层智能体模型。
    sim : NetworkSimEngine
        网络仿真引擎。
    entry_sat : int
        域内入口卫星 ID。
    exit_sat : int
        域内出口卫星 ID。
    flow : FlowRequest
        流请求（用于构建观测向量和带宽分配）。
    domain_id : int
        域编号。
    mode : str
        ``"k_path"`` 或 ``"hop_by_hop"``。
    K : int
        K 最短路候选数（仅 k_path 模式）。
    deterministic : bool
        是否贪心决策。

    Returns
    -------
    list[Experience]
        域内路由产生的经验列表。最后一条经验的 ``info`` 字典包含:
        - ``segment_path``: 成功时为卫星路径列表, 失败时为 None
        - ``segment_ok``: bool, 是否成功
        - ``segment_delay``: float, 段内时延(秒)
    """
    if mode == "k_path":
        return route_intra_kpath(
            lower_model, sim, entry_sat, exit_sat, flow, domain_id, K, deterministic)
    elif mode == "hop_by_hop":
        return route_intra_hop(
            lower_model, sim, entry_sat, exit_sat, flow, domain_id, deterministic)
    else:
        raise ValueError(f"Unknown lower_mode: {mode}")


# ==================================================================
#  K 最短路选择模式 (K-path Selection Mode)
#  建模为 contextual bandit (单步决策, done=True)
# ==================================================================
def route_intra_kpath(
    lower_model,
    sim: NetworkSimEngine,
    entry_sat: int,
    exit_sat: int,
    flow: FlowRequest,
    domain_id: int,
    K: int = 4,
    deterministic: bool = False,
) -> list[Experience]:
    """K 最短路选择 — 单步 contextual bandit。

    K-shortest-path selection modeled as a contextual bandit:
    - 观测: K 条候选路径的特征 + 流需求特征
    - 动作: 选择第几条路径 (Discrete(K))
    - 奖励: 基于瓶颈带宽和路径时延
    - 终止: 始终 done=True (单步决策，无后续状态)

    算法流程 / Algorithm:
      1. 计算 entry→exit 的 K 条带宽感知最短路
      2. 构建观测向量 (每条路径: 时延/带宽/跳数/瓶颈利用率)
      3. 下层智能体 predict → 选择一条路径
      4. 尝试在选中路径上分配带宽
      5. 计算奖励并返回单条 Experience

    Parameters
    ----------
    lower_model : SB3 BaseAlgorithm
        下层智能体 (仅调用 predict)。
    sim : NetworkSimEngine
        网络仿真引擎。
    entry_sat, exit_sat : int
        域内入口/出口卫星 ID。
    flow : FlowRequest
        流请求。
    domain_id : int
        域编号。
    K : int
        候选路径数。
    deterministic : bool
        是否贪心。

    Returns
    -------
    list[Experience]
        长度为 1 的列表，info 中包含 segment_path / segment_ok / segment_delay。
    """
    # Step 1: 计算域内 K 条带宽感知最短路
    k_paths = sim.compute_k_shortest(domain_id, entry_sat, exit_sat, K)

    if not k_paths:
        # 无路径可达 — 生成一条负奖励的虚拟经验
        obs = np.zeros(sim.get_lower_kpath_obs_dim(K), dtype=np.float32)
        return [Experience(
            obs=obs, next_obs=obs, action=0,
            reward=KPATH_FAIL_PENALTY, done=True,
            info={"segment_path": None, "segment_ok": False, "segment_delay": 0.0},
        )]

    # Step 2: 构建观测向量 (K×4 路径特征 + 6 流/位置特征)
    obs = sim.get_lower_kpath_obs(domain_id, k_paths, flow, entry_sat, exit_sat, K)

    # Step 3: 智能体选择路径 (算法无关的 predict 调用)
    action_np, _ = lower_model.predict(obs, deterministic=deterministic)
    action = int(action_np)
    # 安全索引: 若 action >= len(k_paths), 取最后一条
    selected_path = k_paths[min(action, len(k_paths) - 1)]

    # Step 4: 尝试在选中路径上原子地分配带宽
    ok, delay = sim.try_allocate_segment(selected_path, flow.bandwidth, flow.flow_id)

    # ── 奖励计算 (Reward Shaping) ──
    if ok:
        min_bw = min(
            (sim.bandwidth.get_remaining_bw(selected_path[j], selected_path[j + 1])
             for j in range(len(selected_path) - 1)),
            default=sim.bandwidth.capacity,
        )
        reward = (KPATH_BW_COEFF * np.exp((min_bw - sim.bandwidth.capacity)
                                          / sim.bandwidth.capacity)
                  + KPATH_DELAY_COEFF * delay)
    else:
        reward = KPATH_FAIL_PENALTY

    next_obs = np.zeros_like(obs)  # done=True → next_obs unused by DQN
    return [Experience(
        obs=obs, next_obs=next_obs, action=action,
        reward=float(reward), done=True,
        info={
            "segment_path": selected_path if ok else None,
            "segment_ok": ok,
            "segment_delay": delay,
        },
    )]


# ==================================================================
#  逐跳导航模式 (Hop-by-hop Navigation Mode)
#  建模为多步 MDP: 智能体逐步选择下一跳，直到到达出口或步数耗尽
# ==================================================================
def route_intra_hop(
    lower_model,
    sim: NetworkSimEngine,
    entry_sat: int,
    exit_sat: int,
    flow: FlowRequest,
    domain_id: int,
    deterministic: bool = False,
) -> list[Experience]:
    """域内逐跳导航 — 多步 MDP。

    Hop-by-hop navigation within a domain modeled as a multi-step MDP:
    - 状态: 当前卫星位置 + 目标出口 + 邻居特征
    - 动作: 选择四个方向之一 (up/down/left/right, Discrete(4))
    - 奖励: 每跳时延惩罚 + 到达奖励 / 截断惩罚 / 环路惩罚
    - 终止: 到达出口 (done=True) 或步数耗尽 (truncated=True)

    算法流程 / Algorithm:
      1. 从 entry_sat 出发，最多走 max_steps 步
      2. 每步: 构建观测 → predict → 选择下一跳
      3. 检查: 无效动作 / 重复访问 / 带宽不足 → 惩罚
      4. 到达 exit_sat → 成功奖励; 步数耗尽 → 截断惩罚
      5. 到达后验证: 对完整路径原子分配带宽

    Parameters
    ----------
    lower_model : SB3 BaseAlgorithm
        下层智能体 (仅调用 predict)。
    sim : NetworkSimEngine
        网络仿真引擎。
    entry_sat, exit_sat : int
        域内入口/出口卫星 ID。
    flow : FlowRequest
        流请求。
    domain_id : int
        域编号。
    deterministic : bool
        是否贪心。

    Returns
    -------
    list[Experience]
        逐跳产生的经验列表，最后一条 info 包含路由结果。
    """
    domain_sats = set(sim.domain.get_domain_sats(domain_id))  # 当前域内所有卫星
    max_steps = max(len(domain_sats) // 4, 10)  # 最大步数 ≈ 域大小/4

    current_sat = entry_sat     # 当前所在卫星
    path = [current_sat]        # 累积路径
    visited = {current_sat}     # 已访问卫星集合 (用于检测环路)
    acc_delay = 0.0             # 累积时延
    experiences: list[Experience] = []

    for step in range(max_steps):
        # 构建逐跳观测: 自身位置 + 目标位置 + 距离 + 邻居特征 (共23维)
        obs = sim.get_lower_hop_obs(
            current_sat, exit_sat, domain_id,
            acc_delay, step, max_steps, flow)

        # 智能体选择方向 (0=up, 1=down, 2=left, 3=right)
        action_np, _ = lower_model.predict(obs, deterministic=deterministic)
        action = int(action_np)
        neighbors = sim.topology.adj[current_sat]  # [up, down, left, right]
        next_sat = neighbors[action]

        # ── 无效动作: 链路不存在(-1) 或下一跳出域 ──
        if next_sat == -1 or next_sat not in domain_sats:
            reward = HOP_INVALID_PENALTY
            terminated = False
            truncated = (step == max_steps - 1)
            next_obs = obs  # stay in place

        else:
            hop_delay = sim.topology.get_link_delay(
                current_sat, next_sat, sim.current_timeslot)
            acc_delay += hop_delay

            has_bw = (sim.bandwidth.get_remaining_bw(current_sat, next_sat)
                      >= flow.bandwidth)

            current_sat = next_sat
            path.append(current_sat)

            terminated = (current_sat == exit_sat)
            truncated = (step == max_steps - 1) and not terminated

            # ── Reward shaping ──
            reward = HOP_DELAY_COEFF * hop_delay
            if current_sat in visited:
                reward += HOP_REVISIT_PENALTY
            if not has_bw:
                reward += HOP_BW_PENALTY
            if terminated:
                reward += HOP_ARRIVE_BONUS * (1.0 - step / max_steps)
            if truncated:
                dist = sim.topology.sat_distance_km(
                    current_sat, exit_sat, sim.current_timeslot)
                reward += HOP_TRUNCATE_PENALTY * dist / 20000.0

            visited.add(current_sat)
            next_obs = sim.get_lower_hop_obs(
                current_sat, exit_sat, domain_id,
                acc_delay, step + 1, max_steps, flow)

        experiences.append(Experience(
            obs=obs, next_obs=next_obs, action=action,
            reward=float(reward),
            done=terminated,
            truncated=truncated,
        ))

        if terminated or truncated:
            break

    # ── 后验证: 对完整路径原子分配带宽 ──
    # 逐跳过程中不分配带宽 (避免部分分配), 到达后统一分配
    reached = (current_sat == exit_sat)
    if reached:
        ok = sim.bandwidth.allocate(path, flow.bandwidth, flow.flow_id)
    else:
        ok = False

    # 在最后一条经验的 info 中标记路由结果，供上层调用者使用
    if experiences:
        experiences[-1].info = {
            "segment_path": path if ok else None,
            "segment_ok": ok,
            "segment_delay": acc_delay,
        }

    return experiences
