
# 分层强化学习卫星网络路由 — 修订设计方案 v2

> **约束前提（用户确认）**：
> 1. 仅单 shell（LEO），不需要物理 MEO 层，上层智能体作为逻辑全局决策者
> 2. 总共只有 **2 个 SB3 模型**（upper_model + lower_model），下层智能体所有域共享参数
> 3. 域划分作为 StarPerf 的插件实现
> 4. 上层 → 选择具体域间链路 → 下层域内路由 → 奖励反馈给上层
> 5. 需要明确 episode 定义、rollout 流程、经验收集方式

---

## 一、可行性分析

### 1.1 ✅ 可行的部分

| 设计点 | 可行性 | 原因 |
|--------|--------|------|
| 仅 2 个 SB3 模型 | ✅ 完全可行 | 上层 1 个 + 下层 1 个（共享权重处理所有域），SB3 原生支持 |
| 下层参数共享 | ✅ 可行 | 等分轨道 → 所有域结构相同（卫星数、ISL 数一致），观测/动作空间维度统一 |
| 上层选域间链路 | ✅ 可行 | 等分时相邻域边界固定为 `sats_per_orbit` 条 inter-orbit ISL，动作空间 = `Discrete(sats_per_orbit)` |
| 域划分为 StarPerf 插件 | ✅ 可行 | 完全遵循 `connectivity_mode_plugin_manager` 的 importlib 模式 |
| 单 shell + 逻辑上层 | ✅ 可行 | 上层不需要物理卫星，只是一个可访问全网状态的决策模型 |
| 下层反馈奖励给上层 | ✅ 可行 | 上层 env 内部调用下层 model.predict()，下层执行结果直接作为上层的奖励信号 |

### 1.2 ⚠️ 需要解决的关键问题

#### 问题 A：跨多域流量的序列决策

**场景**：流量从域 0 到域 5，需经过域 0→1→2→3→4→5 共 5 个域边界。

**矛盾**：上层智能体是"选择一条域间链路"，但一条流可能需要选择多条域间链路。

**方案**：将上层建模为 **序列决策**——每条跨域流被分解为多步：
- Step 1：在域 0→1 边界选择一条域间链路
- Step 2：在域 1→2 边界选择一条域间链路
- ...
- Step N：在域 (N-1)→N 边界选择最后一条域间链路

每步之间，下层智能体执行当前域的域内路由（从入口到上层选定的出口）。一条流的路由完成后计算总奖励。

**上层 episode 一步 = 一个域边界决策 + 对应的域内路由执行**。

> 这与论文的设计一致：论文中每个域的上层智能体独立标记 junction link，这里改为同一个模型依次处理每个域边界。

#### 问题 B：域间路由序列的确定

**问题**：上层选择的是"用哪条域间链路"，但"经过哪些域"的顺序本身需要事先确定。

**方案**：域间路由序列由 **域级最短路径** 预先确定，不作为 RL 决策：
- 将域抽象为节点、域间有链路的域对抽象为边，构建域级图
- 对每条流，用 Dijkstra 在域级图上求最短域序列
- 上层 RL 只决定"在每个域边界用哪条具体链路"，不决定域序列

**合理性**：域序列选择的组合空间小（相邻域通常只有 1 条最短路径），域间链路选择才是关键决策点（影响负载均衡和拥塞）。

#### 问题 C：上层奖励的延迟反馈

**问题**：上层在域边界 0→1 选了链路后，必须等下层在域 0 内完成路由，然后才能进入域 1，再做下一个决策。最终奖励要等整条流路由完成。

**方案**：使用 **分步即时奖励 + 终局奖励** 的混合设计：
- 每个域边界决策后，立即给一个 **中间奖励**（基于所选域间链路的利用率、拥塞度）
- 整条流路由完成后，给一个 **终局奖励**（端到端时延、是否成功）
- 上层的总回报 = Σ 中间奖励 + 终局奖励

#### 问题 D：下层 K 最短路径的实时性

**问题**：域内 K 条候选路径需要考虑当前带宽状态，但 K-shortest-paths 通常基于静态权重。

**方案**：使用 **带宽感知权重** 计算 K 最短路径：
- 权重 = `delay + α × (1 / remaining_bandwidth)`，拥塞链路权重增大
- 每次调用下层前重新计算（域内规模小，计算开销可控：264 节点的 K 最短路 ≈ ms 级）

#### 问题 E：等分轨道在极轨星座的边界问题

**问题**：Starlink shell 1 倾角 53°（非极轨），但 OneWeb/Kuiper 有极轨配置（80°-100°）。极轨星座的第一轨道和最后轨道之间无 inter-orbit ISL（+Grid 开缝），导致域划分可能产生首尾不连通的域。

**方案**：
- 非极轨（倾角 ≤ 80°）：轨道平面形成环，可均匀切分，首尾域也有域间链路
- 极轨（80° < 倾角 < 100°）：开缝处不切域边界，将首尾轨道分到相邻域中
- 域划分插件中通过 `inclination` 参数自动处理

---

## 二、修订后的系统架构

### 2.1 总体架构

```
                     ┌────────────────────────────────┐
                     │         HRL 训练协调器           │
                     │    (自定义训练循环, 非 SB3 内置)   │
                     └───────────┬────────────────────┘
                                 │
                ┌────────────────┼────────────────┐
                │                │                │
        ┌───────▼───────┐ ┌─────▼─────┐ ┌────────▼────────┐
        │  Upper Model  │ │  Lower    │ │  Network Sim    │
        │  (SB3 DQN)    │ │  Model    │ │  Engine         │
        │               │ │  (SB3 DQN)│ │                 │
        │ obs: 全网+流   │ │ obs: 域内  │ │ · Topology      │
        │ act: 域间链路   │ │ act: 双模式│ │ · Bandwidth     │
        │               │ │  K路径选择 │ │ · Flow lifecycle│
        └───────────────┘ │  或逐跳导航│ │ · Domain        │
                          │ 共享权重   │ └─────────────────┘
                          └───────────┘
```

### 2.2 上下层交互时序（单条跨域流）

```
流量请求到达: src_domain=0, dst_domain=3
域级最短路径预计算: [0, 1, 2, 3]  (经过 3 个域边界)

─── 域边界 0→1 ───
  Upper.observe(全网状态 + 流特征 + 边界0→1的22条域间链路状态)
  Upper.act() → 选择域间链路 j₁ (出口卫星: orbit_boundary × sats_per_orbit + j₁)
  Lower.observe(域0内链路状态 + 子任务: src_sat → exit_sat_j₁)
  Lower.act() → 选择 K 条候选路径中的第 k₁ 条
  执行: 检查带宽 → 分配 → 计算域内时延 d₁
  Upper 获得中间奖励 r₁

─── 域边界 1→2 ───
  Upper.observe(更新后全网状态 + 边界1→2的域间链路状态)
  Upper.act() → 选择域间链路 j₂
  Lower.observe(域1内链路状态 + 子任务: entry_sat_j₁ → exit_sat_j₂)
  Lower.act() → 选择第 k₂ 条路径
  执行: 检查带宽 → 分配 → 计算域内时延 d₂
  Upper 获得中间奖励 r₂

─── 域边界 2→3 ───
  Upper.observe(更新后全网状态 + 边界2→3的域间链路状态)
  Upper.act() → 选择域间链路 j₃
  Lower.observe(域2内链路状态 + 子任务: entry_sat_j₂ → exit_sat_j₃)
  Lower.act() → 选择第 k₃ 条路径
  执行 ...
  Lower.observe(域3内链路状态 + 子任务: entry_sat_j₃ → dst_sat)
  Lower.act() → 选择第 k₄ 条路径
  执行 ...

─── 流路由完成 ───
  总时延 = d₁ + d₂ + d₃ + d₄ + 域间链路延迟
  检查 QoS 约束 → 成功/失败
  Upper 获得终局奖励 R_final
  完整路径上所有链路分配带宽，持续 duration 个 timeslot
```

### 2.3 两个 SB3 模型的规格

#### Upper Model

| 项目 | 规格 |
|------|------|
| **算法** | DQN（off-policy，支持 replay buffer 手动添加经验） |
| **观测空间** | `Box(obs_dim,)` |
| **观测组成** | 当前域边界 22 条域间链路的 `[利用率, 延迟, 对端域平均利用率]` = 22×3=66 维 + 流特征 `[bw_demand, delay_budget, remaining_hops, src_domain, dst_domain]` = 5 维 + 全局 `[各域平均利用率]` = n_domains 维 ≈ **77 维**（6 域时） |
| **动作空间** | `Discrete(22)` — 22 条 inter-orbit ISL 中选一条（`sats_per_orbit`） |
| **奖励** | 中间: $r_{step} = -\alpha \cdot utilization_{selected} - \beta \cdot delay_{selected}$；终局: $R_{final} = \gamma \cdot \mathbb{1}_{success} - \delta \cdot \frac{total\_delay}{max\_delay}$ |

#### Lower Model（双模式）

下层智能体支持 **两种工作模式**，通过配置切换：

##### 模式 A：K 最短路径选择（k_path）

| 项目 | 规格 |
|------|------|
| **算法** | DQN |
| **观测空间** | `Box(22,)` |
| **观测组成** | K 条候选路径的特征 `[总延迟, 最小剩余带宽, 跳数, 瓶颈利用率]` = K×4=16 维（K=4）+ 子任务特征 `[bw_demand, delay_budget, entry_pos(2), exit_pos(2)]` = 6 维 ≈ **22 维** |
| **动作空间** | `Discrete(K)` — K=4 条预计算最短路径中选一条 |
| **奖励** | $r = \lambda_1 \cdot e^{(bw_{remaining}-bw_{max})} + \lambda_2 \cdot e^{-delay_{path}} - \lambda_3 \cdot \mathbb{1}_{fail}$ |
| **Episode 长度** | **1 步**（单步决策，无序列性） |
| **done** | **始终 True**（每次决策即为完整 episode） |
| **权重共享** | 所有域使用同一个 model，不同域的输入通过观测区分 |

##### 模式 B：逐跳导航（hop_by_hop）

| 项目 | 规格 |
|------|------|
| **算法** | DQN |
| **观测空间** | `Box(23,)` |
| **观测组成** | 当前卫星位置 `[lon, lat, alt]` (3) + 出口卫星位置 `[lon, lat, alt]` (3) + 到出口距离 (1) + 累计时延 (1) + 步数进度 (1) + 带宽需求 (1) + 剩余时延预算 (1) + 4 邻居特征 `[到出口距离, 链路剩余带宽, 域内标志]` (4×3=12) ≈ **23 维** |
| **动作空间** | `Discrete(4)` — +Grid 四方向（同轨前/后 + 邻轨左/右） |
| **奖励** | 每跳: $r_{hop} = -\alpha \cdot delay_{hop} - \beta \cdot \mathbb{1}_{revisit} - \gamma \cdot \mathbb{1}_{invalid}$；到达: $r_{arrive} = +R_{success} \cdot (1 - steps/max\_steps)$ |
| **Episode 长度** | **多步**（1 ~ max_domain_steps，域内卫星数/4 为上限） |
| **done** | `terminated`: 到达出口卫星；`truncated`: 超过 max_domain_steps |
| **权重共享** | 同模式 A |

##### 两种模式的对比

| 维度 | K 路径选择 | 逐跳导航 |
|------|-----------|----------|
| 决策粒度 | 粗 — 选完整路径 | 细 — 逐节点选下一跳 |
| 训练难度 | 低 — 类似 contextual bandit | 高 — 需要序列决策 + 信用分配 |
| 路径质量 | 受限于预计算 K 条路径 | 可探索任意路径组合 |
| 带宽感知 | 预计算时可加权，但路径固定 | 每跳实时感知带宽状态 |
| 训练速度 | 快（单步回合） | 慢（多步回合 + 更大搜索空间） |
| 适用阶段 | 先跑通全流程 | 性能提升阶段 |

---

## 三、Episode / Rollout / 经验收集 设计

### 3.1 Episode 定义

一个 **episode = 一个时间窗口**（例如 T_episode = 100 个 timeslot），在此期间：

1. 仿真时钟从 timeslot $t_0$ 推进到 $t_0 + T_{episode}$
2. 每个 timeslot：
   - 释放到期流量（归还带宽）
   - 按 Poisson 过程生成新的流量请求
   - 对每条新流量请求执行上下层路由决策
   - 更新链路延迟（读取下一 timeslot 的 StarPerf 预计算数据）
3. Episode 结束条件：到达 $T_{episode}$ 个 timeslot

```python
# Episode 伪代码
def run_episode(upper_model, lower_model, sim_engine, T_episode=100):
    sim_engine.reset()                    # 重置带宽状态、清空活跃流
    upper_experiences = []
    lower_experiences = []
    
    for t in range(T_episode):
        # 1. 释放到期流
        sim_engine.release_expired_flows()
        
        # 2. 更新拓扑延迟
        sim_engine.advance_timeslot()
        
        # 3. 生成新流
        new_flows = sim_engine.generate_flows()
        
        # 4. 路由每条流
        for flow in new_flows:
            src_domain = sim_engine.get_domain(flow.src_sat)
            dst_domain = sim_engine.get_domain(flow.dst_sat)
            
            if src_domain == dst_domain:
                # 域内流: 仅下层决策
                lower_exp = route_intra_domain(lower_model, sim_engine, flow)
                lower_experiences.extend(lower_exp)
            else:
                # 跨域流: 上层+下层协作
                upper_exp, lower_exp = route_cross_domain(
                    upper_model, lower_model, sim_engine, flow)
                upper_experiences.extend(upper_exp)
                lower_experiences.extend(lower_exp)
    
    return upper_experiences, lower_experiences
```

### 3.2 Rollout 流程（单条跨域流）

```python
def route_cross_domain(upper_model, lower_model, sim, flow):
    domain_path = sim.get_domain_sequence(flow.src_sat, flow.dst_sat)
    # domain_path 例: [0, 1, 2, 3]
    
    upper_exps = []
    lower_exps = []
    full_path = []
    total_delay = 0.0
    success = True
    
    current_entry_sat = flow.src_sat
    
    for i in range(len(domain_path)):
        current_domain = domain_path[i]
        
        if i < len(domain_path) - 1:
            # ── 上层决策: 选域间链路 ──
            next_domain = domain_path[i + 1]
            upper_obs = sim.get_upper_obs(current_domain, next_domain, flow)
            upper_action, _ = upper_model.predict(upper_obs, deterministic=False)
            exit_sat = sim.junction_sat_from_action(
                current_domain, next_domain, upper_action)
        else:
            # 最后一个域: 出口 = 目标卫星
            exit_sat = flow.dst_sat
        
        # ── 下层决策: 域内路由 ──
        k_paths = sim.compute_k_shortest(current_domain, current_entry_sat, exit_sat, K=4)
        if not k_paths:
            success = False
            break
        
        lower_obs = sim.get_lower_obs(current_domain, k_paths, flow)
        lower_action, _ = lower_model.predict(lower_obs, deterministic=False)
        selected_path = k_paths[min(lower_action, len(k_paths) - 1)]
        
        # ── 执行: 检查+分配带宽 ──
        seg_ok, seg_delay = sim.try_allocate_segment(selected_path, flow.bandwidth)
        if not seg_ok:
            success = False
            break
        
        full_path.extend(selected_path)
        total_delay += seg_delay
        
        # ── 收集下层经验 ──
        lower_reward = compute_lower_reward(seg_delay, seg_ok, flow)
        lower_exps.append((lower_obs, lower_action, lower_reward, ...))
        
        # ── 收集上层中间经验 ──
        if i < len(domain_path) - 1:
            upper_reward_step = compute_upper_step_reward(exit_sat, sim)
            upper_exps.append((upper_obs, upper_action, upper_reward_step, ...))
        
        # 下一段的入口 = 域间链路对端
        if i < len(domain_path) - 1:
            current_entry_sat = sim.get_peer_sat(exit_sat, next_domain)
    
    # ── 终局: 补充上层终局奖励到最后一步 ──
    if upper_exps:
        final_reward = compute_upper_final_reward(success, total_delay, flow)
        upper_exps[-1] = update_reward(upper_exps[-1], final_reward)
    
    # 若成功, 注册活跃流
    if success:
        sim.register_active_flow(flow, full_path)
    
    return upper_exps, lower_exps
```

### 3.3 经验收集与训练

采用 **离线收集 + 批量训练** 的方式（off-policy DQN 天然支持）：

```python
def train_hrl(config):
    sim = NetworkSimEngine(config)
    
    upper_model = DQN("MlpPolicy", UpperEnvShell(sim), **upper_hparams)
    lower_model = DQN("MlpPolicy", LowerEnvShell(sim), **lower_hparams)
    
    # ════════ 阶段 1: 下层预训练 ════════
    # 随机生成域内子任务 (entry→exit)，无需上层
    for episode in range(pretrain_episodes):
        lower_exps = pretrain_lower_episode(lower_model, sim)
        for exp in lower_exps:
            lower_model.replay_buffer.add(*exp)
        lower_model.train(gradient_steps=len(lower_exps))
    
    # ════════ 阶段 2: 上层训练 (下层冻结) ════════
    lower_model.policy.set_training_mode(False)  # 冻结
    for episode in range(upper_train_episodes):
        upper_exps, lower_exps = run_episode(upper_model, lower_model, sim)
        for exp in upper_exps:
            upper_model.replay_buffer.add(*exp)
        upper_model.train(gradient_steps=len(upper_exps))
    
    # ════════ 阶段 3: 联合微调 ════════
    lower_model.policy.set_training_mode(True)   # 解冻
    for episode in range(finetune_episodes):
        upper_exps, lower_exps = run_episode(upper_model, lower_model, sim)
        for exp in upper_exps:
            upper_model.replay_buffer.add(*exp)
        for exp in lower_exps:
            lower_model.replay_buffer.add(*exp)
        upper_model.train(gradient_steps=len(upper_exps))
        lower_model.train(gradient_steps=len(lower_exps))
```

### 3.4 SB3 兼容性方案

**核心思路**：不用 `model.learn()`，而是用 `model.predict()` + `model.replay_buffer.add()` + `model.train()`。

SB3 的 DQN 支持以下低级 API：
- `model.predict(obs)` — 用当前策略选动作
- `model.replay_buffer.add(obs, next_obs, action, reward, done, info)` — 手动添加经验
- `model.train(gradient_steps=N)` — 执行 N 步梯度更新

**UpperEnvShell / LowerEnvShell**：仅用于初始化 SB3 模型（定义 observation_space、action_space），不实际调用 `step()`。真正的环境交互在自定义训练循环中完成。

```python
class UpperEnvShell(gym.Env):
    """仅用于 SB3 模型初始化的空壳环境"""
    def __init__(self, sim):
        super().__init__()
        self.observation_space = spaces.Box(-1, 1, shape=(77,), dtype=np.float32)
        self.action_space = spaces.Discrete(sim.sats_per_orbit)  # 22
    def reset(self, **kw): return np.zeros(77, dtype=np.float32), {}
    def step(self, a): return np.zeros(77, dtype=np.float32), 0, True, False, {}

# ── K路径模式 ──
class LowerKPathEnvShell(gym.Env):
    """K路径选择模式的空壳环境"""
    def __init__(self, sim):
        super().__init__()
        self.observation_space = spaces.Box(-1, 1, shape=(22,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # K=4 条候选路径
    def reset(self, **kw): return np.zeros(22, dtype=np.float32), {}
    def step(self, a): return np.zeros(22, dtype=np.float32), 0, True, False, {}

# ── 逐跳模式 ──
class LowerHopEnvShell(gym.Env):
    """逐跳导航模式的空壳环境"""
    def __init__(self, sim):
        super().__init__()
        self.observation_space = spaces.Box(-1, 1, shape=(23,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # +Grid 四方向
    def reset(self, **kw): return np.zeros(23, dtype=np.float32), {}
    def step(self, a): return np.zeros(23, dtype=np.float32), 0, False, False, {}
```

### 3.5 下层智能体训练机制详解

#### 3.5.1 K 路径选择模式的训练

K 路径选择本质上是 **contextual bandit（上下文赌博机）**，不是序列决策：

```
┌───────────────────────────────────────────────────┐
│  子任务到达 (entry_sat → exit_sat, bw, delay)      │
│       ↓                                            │
│  计算 K=4 条候选路径 (带宽感知权重)                  │
│       ↓                                            │
│  构建观测 obs (22 维)                               │
│       ↓                                            │
│  lower_model.predict(obs) → action ∈ {0,1,2,3}    │
│       ↓                                            │
│  执行 selected_path, 获得 reward                    │
│       ↓                                            │
│  存入 replay_buffer:                                │
│    (obs, next_obs=零向量, action, reward, done=True)│
│                                                     │
│  ★ done 始终为 True → DQN 目标退化为:               │
│    Q(s,a) ← r    (无未来折扣项)                     │
│    即 γ·max_a' Q(s',a')·(1-done) = 0               │
│    模型本质上学习 Q(s,a) ≈ E[R(s,a)]               │
└───────────────────────────────────────────────────┘
```

**为什么没有 done 概念也能训练？**

因为 DQN 的 TD-target 公式为：
$$y = r + \gamma \cdot \max_{a'} Q_{target}(s', a') \cdot (1 - done)$$

当 $done = 1$ 时，$y = r$，即 Q 值直接回归到即时奖励。这在 SB3 中完全支持——每次调用 `replay_buffer.add()` 时传入 `done=True`，DQN 就会正确处理单步回合。

**训练效果**：模型学习到的是"在给定网络状态和子任务下，选哪条路径期望奖励最高"，本质是一个条件回归问题。收敛通常比多步 RL 更快更稳定。

#### 3.5.2 逐跳导航模式的训练

逐跳模式是标准的 **多步 MDP**，有明确的 terminated/truncated 概念：

```python
def route_intra_domain_hop(lower_model, sim, entry_sat, exit_sat, flow, domain_id):
    """逐跳模式: 下层智能体在域内逐步导航"""
    domain_sats = set(sim.domain.get_domain_sats(domain_id))
    max_steps = len(domain_sats) // 4  # 域内卫星数/4 作为步数上限
    
    current_sat = entry_sat
    path = [current_sat]
    visited = {current_sat}
    accumulated_delay = 0.0
    experiences = []
    
    for step in range(max_steps):
        # ── 观测 ──
        obs = build_hop_observation(
            sim, current_sat, exit_sat, domain_id, domain_sats,
            accumulated_delay, step, max_steps, flow)
        
        # ── 动作 ──
        action, _ = lower_model.predict(obs, deterministic=False)
        neighbors = sim.topology.adj[current_sat]  # [up, down, left, right]
        next_sat = neighbors[action]
        
        # ── 判断动作有效性 ──
        if next_sat == -1 or next_sat not in domain_sats:
            # 无效动作: 边界外或不存在
            reward = -0.5
            terminated = False
            truncated = (step == max_steps - 1)
            next_obs = obs  # 原地不动
        else:
            hop_delay = sim.topology.get_link_delay(
                current_sat, next_sat, sim.current_timeslot)
            accumulated_delay += hop_delay
            
            # 检查该跳链路带宽
            has_bw = sim.bandwidth.get_remaining_bw(
                current_sat, next_sat) >= flow.bandwidth
            
            current_sat = next_sat
            path.append(current_sat)
            
            # ── 终止条件 ──
            terminated = (current_sat == exit_sat)
            truncated = (step == max_steps - 1) and not terminated
            
            # ── 奖励 ──
            reward = -hop_delay / 0.05  # 归一化时延惩罚
            if current_sat in visited:
                reward -= 0.3  # 重访惩罚
            if not has_bw:
                reward -= 0.5  # 带宽不足惩罚
            if terminated:
                reward += 5.0 * (1.0 - step / max_steps)  # 到达奖励
            if truncated:
                dist = sim.topology.sat_distance(current_sat, exit_sat)
                reward -= 2.0 * dist / 20000  # 未到达距离惩罚
            
            visited.add(current_sat)
            next_obs = build_hop_observation(
                sim, current_sat, exit_sat, domain_id, domain_sats,
                accumulated_delay, step + 1, max_steps, flow)
        
        # ── 收集经验 ──
        experiences.append((obs, next_obs, action, reward, terminated, truncated))
        
        if terminated or truncated:
            break
    
    success = (current_sat == exit_sat)
    return path if success else None, accumulated_delay, experiences
```

#### 3.5.3 逐跳模式的域边界处理

逐跳模式的核心难点是 **域边界约束**——智能体不能跳出当前域：

```
域 d 的卫星集合: domain_sats = {sat_1, sat_2, ..., sat_264}
+Grid 邻居: adj[sat_id] = [up, down, left, right]

对于边界轨道上的卫星:
  · 域右边界: right 邻居 ∉ domain_sats → 动作 3 (right) 无效
  · 域左边界: left 邻居 ∉ domain_sats → 动作 2 (left) 无效
  · 同轨 up/down: 始终域内有效

处理方式: 在观测中通过 "域内标志" 向智能体提供信息
  neighbor_features[i] = [..., in_domain_flag]
  in_domain_flag = 1.0 if 邻居 ∈ domain_sats else 0.0

当智能体选择域外方向时:
  → 原地不动 + 惩罚 reward = -0.5
  → 不消耗步数 (或消耗, 增加 truncation 压力)
```

#### 3.5.4 逐跳模式下的带宽分配策略

逐跳导航与带宽分配有时序矛盾：逐跳时路径尚不完整，无法预先检查全路径带宽。

**方案：乐观路由 + 事后验证**

```
1. 逐跳导航阶段: 智能体在每一跳观测邻居链路的剩余带宽，
   但 不做带宽预留（仅读取，不修改）
2. 路径完成后: 整条路径一次性检查+分配带宽
3. 如果某条链路带宽不足: 路由失败, 给下层负奖励

为什么不边走边预留?
  · 如果中途失败需要回滚已预留的带宽，增加复杂度
  · 在同一 timeslot 内多条流并发时可能产生死锁
  · 乐观路由更简单且与 K 路径模式一致（都是完整路径后分配）
```

#### 3.5.5 两种模式与上层交互的统一接口

```python
def route_intra_domain(lower_model, sim, entry_sat, exit_sat, flow, domain_id, mode):
    """统一接口: 根据模式调用不同的域内路由"""
    if mode == "k_path":
        # 计算 K 条带宽感知路径
        k_paths = sim.compute_k_shortest(domain_id, entry_sat, exit_sat, K=4)
        if not k_paths:
            return None, float('inf'), []
        obs = build_kpath_observation(sim, k_paths, flow)
        action, _ = lower_model.predict(obs, deterministic=False)
        path = k_paths[min(action, len(k_paths) - 1)]
        ok, delay = sim.try_allocate_segment(path, flow.bandwidth)
        reward = compute_kpath_reward(delay, ok, flow)
        # 单步经验: done=True
        exps = [(obs, np.zeros_like(obs), action, reward, True, False)]
        return path if ok else None, delay, exps
    
    elif mode == "hop_by_hop":
        # 逐跳导航
        path, delay, exps = route_intra_domain_hop(
            lower_model, sim, entry_sat, exit_sat, flow, domain_id)
        if path is not None:
            ok = sim.bandwidth.allocate(path, flow.bandwidth, flow.flow_id)
            if not ok:
                path = None  # 带宽不足
        return path, delay, exps
```

---

## 四、域划分 StarPerf 插件设计

### 4.0 星座类型支持策略

| 星座类型 | 代表 | 倾角 | +Grid 特性 | 域环拓扑 | 当前状态 |
|----------|------|------|-----------|---------|----------|
| **Walker-Delta** | Starlink (53°), Kuiper (51.9°) | ≤ 80° | 首尾轨道环绕（有 inter-orbit ISL） | **环形**：域 0↔1↔...↔(n-1)↔0 | ✅ 当前实现 |
| **Walker-Star** | OneWeb (87.9°), Polar (90°) | 80°~100° | 首尾轨道开缝（无 inter-orbit ISL） | **链形**：域 0↔1↔...↔(n-1) | 🔮 未来扩展 |

**当前设计决策**：先仅实现 Walker-Delta（非极轨）类型星座的域划分。代码中保留 `is_polar` 判断分支但以 `NotImplementedError` 拦截，确保未来可无缝扩展到 Walker-Star。

Walker-Delta 环形域拓扑的重要推论：
- 域级图为 **环形图**，任意两域间最短路径 ≤ `n_domains / 2` 跳
- 每对相邻域之间有 `sats_per_orbit` 条域间链路（Starlink: 22 条）
- 域内卫星的 up/down 邻居始终在域内，left/right 邻居在边界处跨域

### 4.1 插件架构

遵循 StarPerf 的 `connectivity_mode_plugin_manager` 模式：

```
StarPerf_Simulator/src/XML_constellation/
├── constellation_connectivity/           # 已有
│   ├── connectivity_mode_plugin_manager.py
│   └── connectivity_plugin/
│       ├── positive_Grid.py
│       └── bent_pipe.py
└── constellation_domain/                 # 新增
    ├── domain_partition_plugin_manager.py
    └── domain_partition_plugin/
        ├── by_orbit_group.py            # 按轨道组划分（Walker-Delta 适用）
        └── by_orbit_group_polar.py      # 🔮 未来: Walker-Star 极轨划分
```

### 4.2 插件管理器

```python
# domain_partition_plugin_manager.py
import importlib, os

class domain_partition_plugin_manager:
    def __init__(self):
        self.plugins = {}
        package_name = "src.XML_constellation.constellation_domain.domain_partition_plugin"
        plugins_path = package_name.replace(".", os.path.sep)
        for plugin_name in os.listdir(plugins_path):
            if plugin_name.endswith(".py"):
                plugin_name = plugin_name[:-3]
                plugin = importlib.import_module(package_name + "." + plugin_name)
                if hasattr(plugin, plugin_name) and callable(getattr(plugin, plugin_name)):
                    self.plugins[plugin_name] = getattr(plugin, plugin_name)
        self.current_partition_mode = "by_orbit_group"

    def set_partition_mode(self, plugin_name):
        self.current_partition_mode = plugin_name

    def execute_partition(self, shell, n_domains):
        """返回 DomainInfo 对象"""
        function = self.plugins[self.current_partition_mode]
        return function(shell, n_domains)
```

### 4.3 核心划分插件

```python
# by_orbit_group.py
from dataclasses import dataclass, field

@dataclass
class DomainInfo:
    n_domains: int
    domain_of_sat: dict[int, int]          # sat_id → domain_id
    domain_sats: dict[int, list[int]]      # domain_id → [sat_ids]
    domain_orbits: dict[int, list[int]]    # domain_id → [orbit_indices (1-based)]
    inter_domain_links: dict[tuple[int,int], list[tuple[int,int]]]
        # (domain_a, domain_b) → [(sat_in_a, sat_in_b), ...]
    orbits_per_domain: int
    sats_per_orbit: int

def by_orbit_group(shell, n_domains):
    """按轨道组等分域
    
    当前仅支持 Walker-Delta (倾角 ≤ 80°, 非极轨) 星座。
    轨道环绕: 首尾轨道之间有 inter-orbit ISL, 域形成环形拓扑。
    
    Walker-Star (极轨, 80°< inc <100°) 支持预留:
    - is_polar 分支已保留但当前会 raise NotImplementedError
    - 未来可实现 by_orbit_group_polar.py 插件单独处理
    """
    n_orbits = shell.number_of_orbits
    sats_per_orbit = shell.number_of_satellite_per_orbit
    is_polar = 80 < shell.inclination < 100
    
    # ── 当前仅支持 Walker-Delta ──
    if is_polar:
        raise NotImplementedError(
            f"by_orbit_group 当前仅支持 Walker-Delta (inclination ≤ 80°), "
            f"当前星座倾角 = {shell.inclination}°。"
            f"Walker-Star 极轨星座请使用 by_orbit_group_polar 插件 (待实现)。"
        )
    
    assert n_orbits % n_domains == 0, \
        f"n_orbits({n_orbits}) must be divisible by n_domains({n_domains})"
    orbits_per_domain = n_orbits // n_domains
    
    domain_of_sat = {}
    domain_sats = {d: [] for d in range(n_domains)}
    domain_orbits = {d: [] for d in range(n_domains)}
    
    for oi in range(1, n_orbits + 1):
        domain_id = (oi - 1) // orbits_per_domain
        domain_orbits[domain_id].append(oi)
        for si in range(1, sats_per_orbit + 1):
            sat_id = (oi - 1) * sats_per_orbit + si
            domain_of_sat[sat_id] = domain_id
            domain_sats[domain_id].append(sat_id)
    
    # 识别域间链路: 相邻域边界轨道的 inter-orbit ISL
    # Walker-Delta: 环形拓扑, 所有相邻域对 (包括 n-1 ↔ 0) 都有域间链路
    inter_domain_links = {}
    for d in range(n_domains):
        d_next = (d + 1) % n_domains  # 环形: 最后一域连回第一域
        right_orbit = domain_orbits[d][-1]   # 域 d 最右轨道
        left_orbit = domain_orbits[d_next][0] # 域 d+1 最左轨道
        links = []
        for si in range(1, sats_per_orbit + 1):
            sat_a = (right_orbit - 1) * sats_per_orbit + si
            sat_b = (left_orbit - 1) * sats_per_orbit + si
            links.append((sat_a, sat_b))
        inter_domain_links[(d, d_next)] = links
    
    return DomainInfo(
        n_domains=n_domains,
        domain_of_sat=domain_of_sat,
        domain_sats=domain_sats,
        domain_orbits=domain_orbits,
        inter_domain_links=inter_domain_links,
        orbits_per_domain=orbits_per_domain,
        sats_per_orbit=sats_per_orbit,
    )
```

---

## 五、修订后的代码架构

### 5.1 目录结构

```
hrl-sn-route/
├── StarPerf_Simulator/          # 第三方仿真器（pip install -e 安装）
│   └── src/XML_constellation/
│       └── constellation_domain/         # ★ 新增: 域划分插件
│           ├── domain_partition_plugin_manager.py
│           └── domain_partition_plugin/
│               ├── by_orbit_group.py
│               └── by_orbit_plane.py
├── docs/
│   ├── plan.md                  # 本文档
│   ├── paper_analysis.md        # 论文分析
│   └── starperf.md              # StarPerf 分析
├── core/                        # 核心仿真模块
│   ├── __init__.py
│   ├── topology.py              # +Grid 拓扑管理 (封装 StarPerf)
│   ├── domain.py                # 域划分封装 (调用 StarPerf 插件)
│   ├── bandwidth.py             # 链路带宽资源管理器
│   ├── flow.py                  # 流量请求 & 生命周期管理
│   └── flow_generator.py        # 泊松流量生成器
├── env/                         # RL 环境
│   ├── __init__.py
│   ├── sim_engine.py            # 网络仿真引擎 (非 Gym, 纯状态机)
│   ├── upper_env_shell.py       # 上层 SB3 模型初始化用空壳
│   ├── lower_kpath_env_shell.py # 下层 K路径模式空壳
│   └── lower_hop_env_shell.py   # 下层逐跳模式空壳
├── agents/                      # RL 智能体封装
│   ├── __init__.py
│   └── gnn_extractor.py         # GNN 特征提取器 (DGL + SB3, 可选)
├── train/                       # 训练脚本
│   ├── train_lower.py           # 阶段 1: 下层预训练
│   ├── train_upper.py           # 阶段 2: 上层训练
│   ├── train_hrl.py             # 阶段 3: 联合微调
│   └── rollout.py               # 单 episode rollout 逻辑
├── eval/                        # 评估
│   ├── evaluate.py
│   └── baselines.py             # Dijkstra / 贪心 / 单层 RL 基线
├── configs/
│   └── default.yaml
└── env/sn_env.py                # 旧版单智能体环境 (保留参考)
```

### 5.2 核心模块接口

#### `core/topology.py`

```python
class Topology:
    """封装 StarPerf 星座, 提供 +Grid 拓扑查询"""
    def __init__(self, constellation_name, shell_idx, dT): ...
    
    # 拓扑结构 (不随时间变化)
    adj: dict[int, list[int]]     # sat_id → [up, down, left, right]
    
    # 时变数据
    def get_link_delay(self, sat1, sat2, timeslot) -> float: ...
    def get_sat_position(self, sat_id, timeslot) -> tuple[float,float,float]: ...
    
    # 子图
    def build_nx_subgraph(self, sat_ids, timeslot, bw_manager=None) -> nx.Graph: ...
    
    # 全局图
    def build_nx_graph(self, timeslot) -> nx.Graph: ...
```

#### `core/domain.py`

```python
class DomainManager:
    """封装域划分插件, 提供域级查询"""
    def __init__(self, shell, n_domains, mode="by_orbit_group"):
        # 调用 StarPerf domain_partition_plugin_manager
        self.info: DomainInfo = ...
    
    def get_domain(self, sat_id) -> int: ...
    def get_domain_sats(self, domain_id) -> list[int]: ...
    def get_inter_domain_links(self, d1, d2) -> list[tuple[int,int]]: ...
    def get_domain_path(self, src_domain, dst_domain) -> list[int]: ...
    def junction_sat_from_action(self, d1, d2, action_idx) -> tuple[int,int]: ...
```

#### `core/bandwidth.py`

```python
class BandwidthManager:
    """链路级带宽资源管理"""
    def __init__(self, topology: Topology, link_capacity_mbps=2000.0): ...
    
    def allocate(self, path: list[int], bw: float, flow_id: int) -> bool: ...
    def release(self, flow_id: int) -> None: ...
    def check_feasibility(self, path: list[int], bw: float) -> bool: ...
    def get_utilization(self, sat1: int, sat2: int) -> float: ...
    def get_remaining_bw(self, sat1: int, sat2: int) -> float: ...
    def get_all_utilizations(self) -> np.ndarray: ...
```

#### `core/flow.py`

```python
@dataclass
class FlowRequest:
    flow_id: int
    src_sat: int                     # 源接入卫星
    dst_sat: int                     # 目标接入卫星
    bandwidth: float                 # Mbps
    max_delay: float                 # 秒
    duration: int                    # timeslots
    arrival_timeslot: int

@dataclass
class ActiveFlow:
    request: FlowRequest
    path: list[int]
    remaining_duration: int
    actual_delay: float

class FlowManager:
    """活跃流生命周期管理"""
    active_flows: dict[int, ActiveFlow]
    
    def register(self, request, path, delay) -> None: ...
    def tick(self) -> list[int]:
        """推进一步, 返回到期的 flow_id 列表"""
    def get_active_count(self) -> int: ...
```

---

## 六、关键设计决策（已确定）

| 决策项 | 选择 | 理由 |
|--------|------|------|
| Shell 数量 | 1（仅 LEO） | 用户要求，简化实现 |
| SB3 模型数量 | 2（upper + lower） | 用户要求 |
| 下层参数共享 | 所有域共享一个 lower_model | 域结构等同 + 用户要求 |
| 域划分方式 | 按轨道组等分 | 稳定、+Grid 兼容 |
| 域间路由序列 | 预计算（非 RL 决策） | 组合空间小，非关键决策 |
| 上层动作 | 选具体域间链路 | 用户要求 |
| 下层动作 | **双模式**：K 路径选择 / 逐跳导航 | K 路径先跑通，逐跳后续提升 |
| 下层默认模式 | K 路径选择（k_path） | 训练快、调试简单、先验证流程 |
| 星座类型 | Walker-Delta（非极轨, ≤ 80°） | 当前聚焦 Starlink shell 1 (53°) |
| Walker-Star 支持 | 代码预留 `is_polar` 分支 | 未来扩展 OneWeb 等极轨星座 |
| Episode 定义 | 固定时间窗口（100 timeslot） | 捕获流量动态 + 资源竞争 |
| 训练方式 | 自定义循环 + SB3 低级 API | 兼容分层协调 |
| GNN | 先 MLP 跑通，后续替换 | 降低调试难度 |

---

## 七、实施优先级

| 优先级 | 模块 | 依赖 | 工作量 |
|--------|------|------|--------|
| P0 | 域划分插件 (`constellation_domain/`) | StarPerf 已有架构 | 小 |
| P0 | `core/topology.py` | StarPerf constellation | 小 |
| P0 | `core/domain.py` | 域划分插件 | 小 |
| P0 | `core/bandwidth.py` | topology | 中 |
| P0 | `core/flow.py` + `core/flow_generator.py` | 无 | 小 |
| P1 | `env/sim_engine.py` | 所有 core 模块 | 大 |
| P1 | `train/rollout.py` | sim_engine | 大 |
| P1 | `env/upper_env_shell.py` + `lower_*_env_shell.py` | sim_engine | 小 |
| P2 | `train/train_lower.py` | rollout + lower_env | 中 |
| P2 | `train/train_upper.py` | rollout + upper_env | 中 |
| P2 | `train/train_hrl.py` | 上面两个 | 中 |
| P3 | `eval/baselines.py` + `eval/evaluate.py` | sim_engine | 中 |
| P3 | `agents/gnn_extractor.py` | DGL 已安装 | 中 |

---

## 八、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| SB3 DQN 的 replay_buffer.add() 手动调用可能与内部状态冲突 | 训练不收敛 | 使用 SB3 的 `_store_transition()` 或直接用 `ReplayBuffer` 类独立管理 |
| K 最短路径计算在大域（264 节点）上可能较慢 | 训练速度 | 缓存已计算路径，仅在带宽状态显著变化时重新计算 |
| 上层奖励延迟（等所有域段完成）可能影响信用分配 | 上层学习困难 | 使用分步中间奖励 + GAE(λ) 或 n-step return |
| 下层预训练与实际跨域子任务分布不匹配 | 下层泛化差 | 预训练时使用多样化的 (entry, exit) 对，覆盖域内各种路由场景 |
| 极轨星座首尾域间无连接，导致某些域对不可达 | 路由失败 | 域级图连通性检查 + 不可达流直接标记失败（当前版本不涉及） |
| 逐跳模式域边界约束导致动作空间受限 | 边界卫星学习效率低 | 通过观测中的 in_domain_flag 提供先验，让智能体学会避开 |
| 逐跳模式乐观路由事后验证可能浪费算力 | 路由失败率高 | 逐跳观测中加入每跳剩余带宽信息，引导智能体选宽裕链路 |
| K 路径与逐跳两种模式训练曲线不可比 | 对比困难 | 使用相同流量负载、统一评估指标（成功率 + 端到端延迟） |