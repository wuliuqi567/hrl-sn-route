# 论文仿真设计分析与基于流的网络仿真实现方案

> 论文：Li et al., "Stigmergy and Hierarchical Learning for Routing Optimization in Multi-Domain Collaborative Satellite Networks", IEEE JSAC, Vol. 42, No. 5, May 2024
>
> 结合项目：**StarPerf Simulator** + **Adaptive (DQN Routing)**

---

## 一、论文核心思想

### 1.1 问题定义

论文针对的是**大规模多域协作卫星网络**中的路由优化问题。核心挑战：
- 集中式路由在大规模网络中扩展性差（SDN控制器负载高）
- 分布式路由缺乏全局信息，难以达到全局最优
- 跨域路由的决策复杂度随域数增加急剧上升

### 1.2 解决方案

提出 **Stigmergy Multi-Agent Hierarchical DRL Routing** 算法：

| 层级 | 负责 | 机制 | 目标 |
|------|------|------|------|
| **上层智能体 (Upper Agent)** | 跨域路由决策 | DQN + 信息素(Pheromone)协同 | 选择跨域链路（子目标节点） |
| **下层智能体 (Lower Agent)** | 域内路由优化 | GNN + DQN (消息传递神经网络) | 从 k 条预计算路径中选最优域内路径 |

### 1.3 系统模型

卫星网络 $D = \{1, 2, \ldots, d\}$，每个域 $d$ 包含：
- 节点集 $N^d = \{1, 2, \ldots, n\}$
- 域内链路集 $E^d = \{e^d_{i,j}\}$
- 跨域链路集 $E^{d,d'} = \{e^{d,d'}_{i,j}\}$
- 带宽资源 $B^d, B^{d,d'}$

**路由代价指标**：

$$u_n = u_n^{utl} + u_n^{delay}$$

其中链路利用率 $u_n^{utl}$ 和端到端延迟 $u_n^{delay}$ 分别为：

$$u_n^{utl} = \sum_{d,d' \in D} \sum_{i,j \in p_n} \left(\frac{b^{used,d}_{i,j}}{b^{all,d}_{i,j}} + \frac{b^{used,d,d'}_{i,j}}{b^{all,d,d'}_{i,j}}\right)$$

$$u_n^{delay} = \sum_{d,d' \in D} \sum_{i,j \in p_n} \left(delay^d_{i,j} + delay^{d,d'}_{i,j}\right)$$

**每跳延迟**：$delay_{i,j} = delay^{proc}_{i,j} + delay^{trans}_{i,j}$

### 1.4 约束条件

| 约束 | 含义 | 表达式 |
|------|------|--------|
| C1 | 延迟 QoS | 路径总延迟 ≤ $QoS_n^{delay}$ (200-280ms) |
| C2 | 带宽 QoS | 路径上每条链路剩余带宽 ≥ $QoS_n^{band}$ (10-40Mbps) |
| C3 | ISL 持续时间 QoS | 链路维持时间 ≥ $QoS_n^{dur}$ |
| C4 | 多普勒频移 | $\Delta\lambda_{i,j}(t) \leq \Delta\lambda_{th} = 3.22 \times 10^{-11}$ m |

---

## 二、论文仿真设计详解

### 2.1 星座拓扑生成

| 参数 | 值 |
|------|-----|
| 仿真工具 | MATLAB 2021b 卫星工具箱 |
| 星座类型 | Walker星座（3种规模） |
| 星座规模 | 96颗 (8轨×12星), 144颗 (12轨×12星), 192颗 (16轨×12星) |
| 拓扑快照 | 取单个运行周期的快照 |
| 导入方式 | 导入 Python 3.8, 用 NetworkX 构建图 |

### 2.2 网络参数 (TABLE I)

| 类别 | 参数 | 值 |
|------|------|-----|
| **链路** | 链路带宽 | 2000 Mbps |
| **链路** | 多普勒频移阈值 | $3.22 \times 10^{-11}$ m |
| **业务** | 流量需求 | 10-40 Mbps |
| **业务** | 到达率分布 | Poisson, $\lambda$ = 10, 15, 20 |
| **业务** | 服务持续时间 | 6-10 秒 |
| **业务** | 延迟容忍度 | 200-280 ms |
| **GNN** | 消息传递迭代次数 T | 4 |
| **GNN** | Dropout率 | 0.01 |
| **GNN** | Readout单元数 | 35 |

### 2.3 智能体配置

| 参数 | 上层智能体 | 下层智能体 |
|------|-----------|-----------|
| 网络类型 | DQN (全连接) | GNN + DQN (消息传递NN) |
| 状态空间 | $(BW, Dis, Dop, Phi, Flag, Hid)$ | $(BW, Dis, Dop, Flag)$ |
| 动作空间 | 标记跨域链路 Flag | 从 k 条预计算路径中选择 |
| 奖励函数 | $\alpha \cdot e^{(\phi_{min}-\phi_{sel})} + \beta \cdot r_{hid}$ | $\lambda \cdot e^{(BW_{sel}-BW_{max})} + \gamma \cdot e^{-delay}$ |
| 协同机制 | 信息素(Pheromone) | 消息传递(Message Passing) |
| 训练回合数 | 1500 episodes | 1500 episodes |
| 策略 | ε-greedy | ε-greedy |

### 2.4 信息素机制

信息素分布在跨域链路(Junction)上，强度规则：

$$ph^{d,d'}_{i,j}(t+\Delta t) = \rho_{(t+\Delta t)} \cdot \left[ph^{d,d'}_{i,j}(t) + \sum_n \tau^{d,d'}_{(i,j),n}(t+\Delta t)\right]$$

条件函数 $\tau$ 的四个来源：

| 条件 | 信息素变化原因 |
|------|-------------|
| $\tau_1$：相邻域路由历史 | 经过域 $d_m$ 的路由请求数变化 → 信息素累积 |
| $\tau_2$：跨域链路路由历史 | 经过特定跨域链路的路由请求数变化 → 信息素累积 |
| $\tau_3$：相邻链路多普勒超阈值 | 邻近链路 $\Delta\lambda > \Delta\lambda_{th}$ → 增加 $\mu$ |
| $\tau_4$：相邻链路维持时间为零 | 邻近链路断裂 → 增加 $\omega$ |

蒸发机制：$\rho(t) = \alpha$ 在 $t = t_e$ 时刻（防止无限累积）。

### 2.5 基线对比算法

| 算法 | 类型 | 关键参数 |
|------|------|---------|
| SR-C | 集中式最短路径 | Dijkstra |
| SR-D | 分布式最短路径 | 各域独立 Dijkstra |
| ACO | 蚁群路由优化 | 贪心前向概率=0.9, 蒸发率=0.3, 迭代50次 |
| DQN | 集中式 DQN | 与上层智能体参数一致 |

### 2.6 仿真流程

```
1. MATLAB 生成 Walker 星座 → 取单周期拓扑快照
2. 导入 Python (NetworkX)
3. 星座图按域分区（每个轨道面对应一个域或多个轨道面一个域）
4. 建立域间跨域链路 (Junctions)
5. 初始化上层/下层智能体的 Q-网络和目标 Q-网络
6. for m = 1...M (1500 episodes):
   a. 重置环境，获取初始状态
   b. 生成路由请求（Poisson到达）
   c. 对每个上层智能体：
      ① 更新信息素（Algorithm 2）
      ② ε-greedy选择跨域链路（给下层智能体子目标）
   d. 对每个下层智能体：
      ① GNN消息传递（Algorithm 1, T=4次迭代）
      ② 从 k 条预计算路径中选择域内路径
      ③ 执行路由，观察奖励
   e. 存储经验，采样mini-batch，梯度下降更新 Q-网络
   f. 每 T episode 更新目标网络
7. 测试：150个时间步，评估成功率/延迟/链路利用率
```

### 2.7 性能指标

| 指标 | 定义 |
|------|------|
| **通信成功率** | 满足约束 C1-C4 的路由请求数 / 总路由请求数 |
| **通信延迟** | 路由路径上所有链路延迟之和（ms） |
| **链路利用率** | 链路承载流量 / 最大承载容量 |

### 2.8 关键实验结论

- **收敛性**：所有三种规模星座上下层智能体奖励均在600-1500 episodes内收敛
- **参数敏感性**：最优参数为 k=4条备选路径，蒸发系数 ρ=0.6
- **成功率**：192-Walker上提升高达45.6%（vs基线）
- **延迟**：与SR-C算法相当（~60ms），显著优于ACO（>90ms）
- **链路利用率**：在保持高成功率的同时实现最高链路利用率（负载均衡）

---

## 三、论文仿真设计 vs 两个项目的对比

### 3.1 三者仿真架构对比

| 维度 | 论文 (Stigmergy HRL) | StarPerf Simulator | Adaptive (DQN) |
|------|---------------------|--------------------|----------------|
| **星座生成** | MATLAB卫星工具箱 Walker星座 | SGP4/Skyfield 从XML/TLE生成 | MATLAB+STK 生成 Starlink |
| **拓扑规模** | 96/144/192颗 | 1584颗 (Starlink Shell1) | 1584颗(CSV), 实跑500节点 |
| **网络建模** | 多域划分 NetworkX图 | 单域 NetworkX图 + HDF5 | 单域 NetworkX图 |
| **流量模型** | 基于流：Poisson到达, 10-40Mbps, 6-10s持续 | 基于流/连接：人口密度加权, 0.5Mbps/条 | 基于包：均匀随机注入 2500包 |
| **路由算法** | 层次DRL (GNN+DQN) | +Grid环面贪心/最短路 | 单层DQN逐跳决策 |
| **带宽建模** | 有：2000Mbps链路带宽, 10-40Mbps需求 | 有：GSL下行链路带宽限制 | 无：仅队列容量 |
| **延迟建模** | 处理延迟+传输延迟 | ISL传播延迟(距离/光速) | 抽象edge_delay时隙 |
| **动态性** | ISL断裂(极区)+多普勒频移 | 静态拓扑每时隙独立计算 | 边随机删除+正弦波动 |
| **多智能体** | ✅ 上下层分层多智能体 | ❌ 无RL | ❌ 单智能体(per-dest DQN) |
| **评估指标** | 成功率/延迟/链路利用率 | ISL/GSL链路负载 | 投递时间/队列长度 |
| **训练硬件** | i7-13700K + RTX4080 + 32GB | 无训练 | 未明确 |
| **框架** | PyTorch 1.12 + NetworkX | 纯Python + Skyfield | PyTorch + NetworkX + Gym |

### 3.2 关键差异分析

#### A. 流量模型差异（最核心）

**论文的"基于流"模型**：
```
- 每条流 = 一个服务请求，具有：
  · 源节点、目的节点
  · 带宽需求 (10-40 Mbps)
  · 服务持续时间 (6-10 秒)
  · 延迟容忍度 (200-280 ms)
  · Poisson到达率 (λ=10/15/20)
- 路由决策为：为每条流找到一条端到端路径
- 路径需同时满足 C1-C4 四项约束
- 流持续占用路径上每条链路的带宽资源
- 服务结束后释放带宽
```

**Adaptive的"基于包"模型**：
```
- 每个包 = 1 单位（无带宽概念）
- 逐跳决策（每个节点选下一跳）
- 动态性来自队列拥塞
- 无链路带宽容量、无QoS约束
```

**StarPerf的"链路负载"模型**：
```
- 统计每条ISL/GSL的负载（连接数或Mbps）
- 不做逐跳路由决策
- 关注宏观流量分布
```

#### B. 域划分差异

论文的核心创新在于**多域协作**，但两个项目都**没有实现域划分**：

| 项目 | 域划分 | 域控制器 |
|------|--------|---------|
| 论文 | 有：8/12/16个域（对应轨道面数） | 有：每域一个SDN控制器 |
| StarPerf | 无 | 无 |
| Adaptive | 无 | 无 |

#### C. GNN 差异

论文使用**消息传递神经网络 (MPNN)** 作为下层智能体的核心：

```
输入：链路隐藏状态 h⁰(padding from observation)
迭代T次：
  1. 计算消息: M = Σ m(h_neighbor, h_self)
  2. 更新隐藏状态: h_new = t(h_old, M)
输出：最终隐藏状态 → readout → Q值
```

Adaptive 仅使用全连接 DQN（3层FC），无图结构感知能力。

---

## 四、基于两个项目实现论文仿真的方案

### 4.1 总体架构设计

```
┌─────────────────────────────────────────────────────────┐
│                    HRL Satellite Router                   │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────────┐    ┌───────────────────────────┐   │
│  │  物理层 (StarPerf) │    │   路由层 (Adaptive扩展)     │   │
│  │                    │    │                             │   │
│  │ · Walker星座生成   │    │ · 多域划分模块              │   │
│  │ · ISL拓扑计算      │───→│ · 上层智能体 (DQN+Pheromone)│   │
│  │ · 延迟矩阵         │    │ · 下层智能体 (GNN+DQN)     │   │
│  │ · 多普勒频移       │    │ · 流量生成器 (Poisson)      │   │
│  │ · ISL持续时间      │    │ · 带宽资源管理              │   │
│  └──────────────────┘    └───────────────────────────┘   │
│                                                           │
│  ┌──────────────────────────────────────────────────┐    │
│  │              仿真引擎 (新建)                        │    │
│  │                                                      │    │
│  │ · 时隙驱动的事件循环                                  │    │
│  │ · 流管理 (创建/路由/释放)                             │    │
│  │ · 约束检查 (C1-C4)                                   │    │
│  │ · 性能统计                                            │    │
│  └──────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### 4.2 模块复用计划

#### 可复用 StarPerf 的模块

| StarPerf 模块 | 用途 | 改造需求 |
|---------------|------|---------|
| `constellation_generation/by_XML/` | Walker星座生成 | 低：已支持多种星座参数配置 |
| `constellation_entity/satellite.py` | 卫星实体（位置、速度） | 低：直接使用 |
| `constellation_connectivity/` | ISL拓扑生成 | 中：需增加跨轨ISL极区断裂判定 |
| Skyfield 轨道传播 | 卫星位置随时间变化 | 低：直接使用 |
| HDF5 数据存储 | 仿真数据持久化 | 低：直接使用 |

**StarPerf 无法提供但论文需要的**：
- 多普勒频移计算（需从卫星相对速度推导）
- ISL 持续时间函数 $l_{i,j}(t)$（需从轨道参数推导）
- Walker 星座相位因子 F 的处理
- 链路级带宽资源管理

#### 可复用 Adaptive 的模块

| Adaptive 模块 | 用途 | 改造需求 |
|---------------|------|---------|
| `DQN.py` | 上层智能体网络基础 | 中：需扩展输入维度（加入Phi/Hid） |
| `neural_network.py` | DQN训练框架（policy+target） | 中：需增加GNN版本 |
| `replay_memory.py` | 经验回放缓冲区 | 低：直接使用 |
| `our_agent.py` | ε-greedy策略框架 | 中：需支持层次化动作选择 |
| `our_env.py` | Gym环境框架 | 高：核心重构为多域+流 |
| `dynetwork.py` | 动态网络骨架 | 高：需改为带宽资源模型 |
| `UpdateEdges.py` | 边动态变化思路 | 高：需改为物理ISL断裂模型 |

**Adaptive 无法提供但论文需要的**：
- GNN 消息传递神经网络
- 信息素 (Pheromone) 机制
- 多域划分与跨域路由
- 流量模型（从包到流）
- 带宽资源管理

### 4.3 需要新建的核心模块

#### 模块 1：域划分器 (DomainPartitioner)

```python
class DomainPartitioner:
    """将卫星网络图划分为多个域"""
    
    def __init__(self, constellation_graph: nx.Graph, num_domains: int):
        self.full_graph = constellation_graph
        self.num_domains = num_domains
        self.domains = {}           # {domain_id: subgraph}
        self.junctions = {}         # {(d1,d2): [(node_i, node_j), ...]}
    
    def partition_by_orbital_plane(self):
        """按轨道面划分域（论文的方式）
        
        Walker 星座: num_planes 个轨道面, 每个轨道面 sats_per_plane 颗卫星
        域数量 = 轨道面数量（或轨道面的分组）
        """
        for plane_idx in range(self.num_planes):
            domain_nodes = [plane_idx * self.sats_per_plane + j 
                          for j in range(self.sats_per_plane)]
            self.domains[plane_idx] = self.full_graph.subgraph(domain_nodes)
        
        # 识别跨域链路 (Junctions)
        for (u, v) in self.full_graph.edges():
            domain_u = self.get_domain(u)
            domain_v = self.get_domain(v)
            if domain_u != domain_v:
                key = (min(domain_u, domain_v), max(domain_u, domain_v))
                self.junctions.setdefault(key, []).append((u, v))
    
    def get_domain(self, node_id: int) -> int:
        return node_id // self.sats_per_plane
    
    def get_intra_domain_graph(self, domain_id: int) -> nx.Graph:
        return self.domains[domain_id]
    
    def get_cross_domain_links(self, d1: int, d2: int) -> list:
        return self.junctions.get((min(d1, d2), max(d1, d2)), [])
```

#### 模块 2：流量生成器 (FlowGenerator)

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class ServiceFlow:
    """服务流实体（替代 Adaptive 的 Packet 类）"""
    flow_id: int
    source: int              # 源卫星节点
    destination: int         # 目的卫星节点
    bandwidth_demand: float  # 带宽需求 (Mbps), 10-40
    duration: float          # 服务持续时间 (秒), 6-10
    delay_tolerance: float   # 延迟容忍度 (ms), 200-280
    arrival_time: float      # 到达时间
    path: list = None        # 分配的路由路径
    status: str = 'pending'  # pending/active/completed/failed

class FlowGenerator:
    """Poisson流量生成器"""
    
    def __init__(self, arrival_rate: float = 15, 
                 bw_range=(10, 40), 
                 duration_range=(6, 10),
                 delay_range=(200, 280)):
        self.arrival_rate = arrival_rate  # λ
        self.bw_range = bw_range
        self.duration_range = duration_range
        self.delay_range = delay_range
        self.flow_counter = 0
    
    def generate_flows(self, time_step: float, nodes: list) -> list:
        """在给定时隙生成一批新流"""
        # Poisson 到达数量
        num_arrivals = np.random.poisson(self.arrival_rate)
        flows = []
        for _ in range(num_arrivals):
            src, dst = np.random.choice(nodes, size=2, replace=False)
            flow = ServiceFlow(
                flow_id=self.flow_counter,
                source=src,
                destination=dst,
                bandwidth_demand=np.random.uniform(*self.bw_range),
                duration=np.random.uniform(*self.duration_range),
                delay_tolerance=np.random.uniform(*self.delay_range),
                arrival_time=time_step
            )
            flows.append(flow)
            self.flow_counter += 1
        return flows
```

#### 模块 3：带宽资源管理器 (BandwidthManager)

```python
class BandwidthManager:
    """链路带宽资源管理"""
    
    def __init__(self, graph: nx.Graph, link_capacity: float = 2000.0):
        self.graph = graph
        self.link_capacity = link_capacity  # Mbps
        # 记录每条链路的已用带宽
        for u, v in graph.edges():
            graph[u][v]['capacity'] = link_capacity
            graph[u][v]['used_bandwidth'] = 0.0
            graph[u][v]['active_flows'] = []
    
    def allocate(self, path: list, flow: 'ServiceFlow') -> bool:
        """为流在路径上分配带宽"""
        # 检查路径上所有链路是否有足够剩余带宽
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            remaining = self.graph[u][v]['capacity'] - self.graph[u][v]['used_bandwidth']
            if remaining < flow.bandwidth_demand:
                return False  # 不满足约束 C2
        
        # 分配带宽
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            self.graph[u][v]['used_bandwidth'] += flow.bandwidth_demand
            self.graph[u][v]['active_flows'].append(flow.flow_id)
        return True
    
    def release(self, path: list, flow: 'ServiceFlow'):
        """流结束后释放带宽"""
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            self.graph[u][v]['used_bandwidth'] -= flow.bandwidth_demand
            if flow.flow_id in self.graph[u][v]['active_flows']:
                self.graph[u][v]['active_flows'].remove(flow.flow_id)
    
    def get_utilization(self, u: int, v: int) -> float:
        """获取链路利用率"""
        return self.graph[u][v]['used_bandwidth'] / self.graph[u][v]['capacity']
```

#### 模块 4：信息素管理器 (PheromoneManager)

```python
class PheromoneManager:
    """跨域链路信息素管理"""
    
    def __init__(self, junctions: dict, evaporation_rate: float = 0.6):
        self.pheromone = {}  # {(u, v): float}
        self.evaporation_rate = evaporation_rate
        self.routing_history = {}  # {domain_id: int} 路由经过次数
        self.link_routing_history = {}  # {(u,v): int}
        
        # 初始化信息素为0
        for (d1, d2), links in junctions.items():
            for (u, v) in links:
                self.pheromone[(u, v)] = 0.0
                self.link_routing_history[(u, v)] = 0
    
    def update(self, junction_link: tuple, 
               observation: dict, 
               routing_domain: int,
               doppler_shifts: dict,
               isl_durations: dict,
               doppler_threshold: float = 3.22e-11):
        """更新信息素 (Algorithm 2)"""
        u, v = junction_link
        delta_tau = 0.0
        
        # τ1: 相邻域路由历史变化
        delta_tau += self.routing_history.get(routing_domain, 0)
        
        # τ2: 该跨域链路自身路由历史变化
        delta_tau += self.link_routing_history.get((u, v), 0)
        
        # τ3: 相邻链路多普勒超阈值
        for neighbor_link, doppler in doppler_shifts.items():
            if doppler > doppler_threshold:
                delta_tau += 1.0  # μ
        
        # τ4: 相邻链路断裂
        for neighbor_link, duration in isl_durations.items():
            if duration == 0:
                delta_tau += 1.0  # ω
        
        self.pheromone[(u, v)] += delta_tau
    
    def evaporate(self):
        """信息素蒸发"""
        for key in self.pheromone:
            self.pheromone[key] *= self.evaporation_rate
    
    def get_pheromone(self, u: int, v: int) -> float:
        return self.pheromone.get((u, v), 0.0)
```

#### 模块 5：GNN 消息传递网络 (MessagePassingNetwork)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MessagePassingNetwork(nn.Module):
    """下层智能体的消息传递神经网络 (Algorithm 1)"""
    
    def __init__(self, input_dim: int, hidden_dim: int, 
                 readout_dim: int = 35, T: int = 4, dropout: float = 0.01):
        super().__init__()
        self.T = T  # 消息传递迭代次数
        
        # 消息函数 m(h_neighbor, h_self) -> message
        self.message_fn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 更新函数 t(h_old, M_aggregated) -> h_new
        self.update_fn = nn.GRUCell(hidden_dim, hidden_dim)
        
        # 初始嵌入: observation -> hidden state
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Readout: hidden states -> Q values
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, readout_dim),
            nn.ReLU(),
            nn.Linear(readout_dim, 1)
        )
    
    def forward(self, edge_features: torch.Tensor, 
                adjacency: torch.Tensor, 
                num_paths: int) -> torch.Tensor:
        """
        Args:
            edge_features: [num_edges, input_dim] 链路特征
            adjacency: [num_edges, num_edges] 链路邻接关系
            num_paths: k 条备选路径数
        Returns:
            q_values: [num_paths] 每条路径的 Q 值
        """
        # 初始化隐藏状态
        h = self.embedding(edge_features)  # [num_edges, hidden_dim]
        
        # T 次消息传递迭代
        for t in range(self.T):
            messages = []
            for i in range(h.shape[0]):
                # 收集邻居消息
                neighbors = adjacency[i].nonzero(as_tuple=True)[0]
                if len(neighbors) > 0:
                    neighbor_h = h[neighbors]  # [num_neighbors, hidden_dim]
                    self_h = h[i].unsqueeze(0).expand_as(neighbor_h)
                    # 消息 = m(h_neighbor, h_self)
                    msg_input = torch.cat([neighbor_h, self_h], dim=-1)
                    msg = self.message_fn(msg_input)
                    # 聚合消息 (sum)
                    aggregated = msg.sum(dim=0)
                else:
                    aggregated = torch.zeros(h.shape[1], device=h.device)
                messages.append(aggregated)
            
            M = torch.stack(messages)  # [num_edges, hidden_dim]
            # 更新隐藏状态
            h = self.update_fn(M, h)
        
        # Readout -> Q values for k paths
        q_vals = self.readout(h)  # [num_edges, 1]
        # 路径级Q值需要将路径上链路的Q值聚合
        return q_vals
```

#### 模块 6：层次化环境 (HierarchicalSatEnv)

```python
import gymnasium as gym
import networkx as nx

class HierarchicalSatEnv(gym.Env):
    """层次化多域卫星网络路由环境"""
    
    def __init__(self, config: dict):
        self.config = config
        self.constellation = None      # StarPerf 星座对象
        self.domain_partitioner = None
        self.bandwidth_manager = None
        self.pheromone_manager = None
        self.flow_generator = None
        self.active_flows = {}         # {flow_id: ServiceFlow}
        self.time_step = 0
        self.stats = {
            'success': 0, 'failure': 0,
            'total_delay': 0.0, 'link_utilizations': []
        }
    
    def reset(self):
        """重置环境"""
        # 1. 从 StarPerf 生成星座拓扑
        self.graph = self._build_constellation()
        
        # 2. 域划分
        self.domain_partitioner = DomainPartitioner(
            self.graph, self.config['num_domains'])
        self.domain_partitioner.partition_by_orbital_plane()
        
        # 3. 初始化资源管理
        self.bandwidth_manager = BandwidthManager(
            self.graph, self.config['link_bandwidth'])
        
        # 4. 初始化信息素
        self.pheromone_manager = PheromoneManager(
            self.domain_partitioner.junctions,
            self.config['evaporation_rate'])
        
        # 5. 初始化流量生成器
        self.flow_generator = FlowGenerator(
            arrival_rate=self.config['arrival_rate'],
            bw_range=self.config['bw_range'],
            duration_range=self.config['duration_range'],
            delay_range=self.config['delay_range'])
        
        self.time_step = 0
        self.active_flows.clear()
        return self._get_observations()
    
    def step(self, upper_actions: dict, lower_actions: dict):
        """执行一个时间步
        
        Args:
            upper_actions: {agent_id: junction_link} 上层选择的跨域链路
            lower_actions: {agent_id: path_index} 下层选择的路径索引
        """
        rewards_upper = {}
        rewards_lower = {}
        
        # 1. 生成新流
        new_flows = self.flow_generator.generate_flows(
            self.time_step, list(self.graph.nodes()))
        
        # 2. 对每条新流执行层次化路由
        for flow in new_flows:
            src_domain = self.domain_partitioner.get_domain(flow.source)
            dst_domain = self.domain_partitioner.get_domain(flow.destination)
            
            if src_domain == dst_domain:
                # 域内路由: 仅用下层智能体
                success = self._route_intra_domain(flow, src_domain, lower_actions)
            else:
                # 跨域路由: 先上层选跨域链路, 再下层选域内路径
                success = self._route_cross_domain(
                    flow, src_domain, dst_domain, 
                    upper_actions, lower_actions)
            
            if success:
                self.active_flows[flow.flow_id] = flow
                self.stats['success'] += 1
            else:
                self.stats['failure'] += 1
        
        # 3. 清理到期流（释放带宽）
        self._expire_flows()
        
        # 4. 更新链路状态（ISL断裂、多普勒变化）
        self._update_link_states()
        
        # 5. 信息素蒸发
        if self.time_step % self.config['evaporation_period'] == 0:
            self.pheromone_manager.evaporate()
        
        self.time_step += 1
        
        obs = self._get_observations()
        done = self.time_step >= self.config['max_steps']
        return obs, rewards_upper, rewards_lower, done, {}
    
    def _route_cross_domain(self, flow, src_domain, dst_domain,
                             upper_actions, lower_actions) -> bool:
        """跨域路由流程"""
        # 路径: src -> [域内路径] -> [跨域链路] -> [域内路径] -> dst
        # 可能经过多个域
        
        # 1. 获取域间路由序列 (简化: 选最短域间路径)
        domain_path = self._find_domain_path(src_domain, dst_domain)
        
        full_path = []
        for i, domain in enumerate(domain_path):
            if i < len(domain_path) - 1:
                next_domain = domain_path[i + 1]
                # 上层智能体选择跨域链路
                junction = upper_actions.get((domain, next_domain))
                # 下层智能体选择域内路径到junction
                intra_path = lower_actions.get((domain, junction))
                full_path.extend(intra_path)
        
        # 2. 检查约束 C1-C4
        if not self._check_constraints(full_path, flow):
            return False
        
        # 3. 分配带宽
        if not self.bandwidth_manager.allocate(full_path, flow):
            return False
        
        flow.path = full_path
        flow.status = 'active'
        return True
    
    def _check_constraints(self, path: list, flow: 'ServiceFlow') -> bool:
        """检查路径是否满足 QoS 约束 C1-C4"""
        total_delay = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge_data = self.graph[u][v]
            
            # C1: 延迟约束
            total_delay += edge_data.get('delay', 0)
            
            # C2: 带宽约束 (在 allocate 中检查)
            
            # C3: ISL持续时间约束
            if edge_data.get('isl_duration', float('inf')) < flow.duration:
                return False
            
            # C4: 多普勒约束
            if edge_data.get('doppler', 0) > 3.22e-11:
                return False
        
        if total_delay > flow.delay_tolerance:
            return False
        
        return True
    
    def _get_observations(self) -> dict:
        """获取上下层智能体的观测"""
        obs = {'upper': {}, 'lower': {}}
        
        for domain_id in range(self.domain_partitioner.num_domains):
            # 上层观测: 跨域链路状态
            cross_links = []
            for (d1, d2), links in self.domain_partitioner.junctions.items():
                if d1 == domain_id or d2 == domain_id:
                    for (u, v) in links:
                        cross_links.append({
                            'bw': self.bandwidth_manager.get_utilization(u, v),
                            'dis': self.graph[u][v].get('distance', 0),
                            'dop': self.graph[u][v].get('doppler', 0),
                            'phi': self.pheromone_manager.get_pheromone(u, v),
                            'flag': self.graph[u][v].get('flag', 0),
                        })
            obs['upper'][domain_id] = cross_links
            
            # 下层观测: 域内链路状态
            subgraph = self.domain_partitioner.get_intra_domain_graph(domain_id)
            intra_links = []
            for (u, v) in subgraph.edges():
                intra_links.append({
                    'bw': self.bandwidth_manager.get_utilization(u, v),
                    'dis': self.graph[u][v].get('distance', 0),
                    'dop': self.graph[u][v].get('doppler', 0),
                    'flag': self.graph[u][v].get('flag', 0),
                })
            obs['lower'][domain_id] = intra_links
        
        return obs
```

### 4.4 训练主循环 (类比 Algorithm 3)

```python
class HRLTrainer:
    """Stigmergy Multi-Agent Hierarchical DRL 训练器"""
    
    def __init__(self, env: HierarchicalSatEnv, config: dict):
        self.env = env
        self.num_upper = config['num_domains']
        self.num_lower = config['num_domains']
        self.num_episodes = config.get('num_episodes', 1500)
        self.target_update_freq = config.get('target_update', 10)
        self.gamma = config.get('gamma', 0.99)
        
        # 初始化上层Q网络 (每个域一个)
        self.upper_q_nets = [DQN(upper_state_dim, upper_action_dim) 
                            for _ in range(self.num_upper)]
        self.upper_target_nets = [DQN(upper_state_dim, upper_action_dim) 
                                 for _ in range(self.num_upper)]
        self.upper_replay = [ReplayMemory(10000) for _ in range(self.num_upper)]
        
        # 初始化下层Q网络 (每个域一个 GNN+DQN)
        self.lower_q_nets = [MessagePassingNetwork(lower_input_dim, hidden_dim)
                            for _ in range(self.num_lower)]
        self.lower_target_nets = [MessagePassingNetwork(lower_input_dim, hidden_dim)
                                 for _ in range(self.num_lower)]
        self.lower_replay = [ReplayMemory(10000) for _ in range(self.num_lower)]
    
    def train(self):
        """主训练循环 (Algorithm 3)"""
        for episode in range(self.num_episodes):
            obs = self.env.reset()
            
            for t in range(self.env.config['max_steps']):
                upper_actions = {}
                lower_actions = {}
                
                # 上层智能体决策
                for domain_id in range(self.num_upper):
                    # 更新信息素
                    self.env.pheromone_manager.update(...)
                    
                    # ε-greedy 选择跨域链路
                    state_upper = self._encode_upper_state(obs['upper'][domain_id])
                    action_upper = self._select_action_upper(
                        self.upper_q_nets[domain_id], state_upper)
                    upper_actions[domain_id] = action_upper
                
                # 下层智能体决策
                for domain_id in range(self.num_lower):
                    # GNN 消息传递
                    state_lower = self._encode_lower_state(obs['lower'][domain_id])
                    action_lower = self._select_action_lower(
                        self.lower_q_nets[domain_id], state_lower)
                    lower_actions[domain_id] = action_lower
                
                # 执行动作
                next_obs, r_upper, r_lower, done, info = self.env.step(
                    upper_actions, lower_actions)
                
                # 存储经验并学习
                for domain_id in range(self.num_upper):
                    self.upper_replay[domain_id].push(
                        obs['upper'][domain_id], upper_actions[domain_id],
                        r_upper.get(domain_id, 0), next_obs['upper'][domain_id])
                    self._learn_upper(domain_id)
                
                for domain_id in range(self.num_lower):
                    self.lower_replay[domain_id].push(
                        obs['lower'][domain_id], lower_actions[domain_id],
                        r_lower.get(domain_id, 0), next_obs['lower'][domain_id])
                    self._learn_lower(domain_id)
                
                obs = next_obs
                if done:
                    break
            
            # 更新目标网络
            if episode % self.target_update_freq == 0:
                for i in range(self.num_upper):
                    self.upper_target_nets[i].load_state_dict(
                        self.upper_q_nets[i].state_dict())
                for i in range(self.num_lower):
                    self.lower_target_nets[i].load_state_dict(
                        self.lower_q_nets[i].state_dict())
```

### 4.5 数据流水线

```
StarPerf 物理层                        路由仿真层
┌─────────────┐                    ┌──────────────┐
│ Walker XML   │──→ constellation  │ DomainPartit. │
│ 配置文件     │    generation     │ 域划分        │
└─────────────┘        │          └──────┬───────┘
                       ↓                 │
              ┌──────────────┐           │
              │ 卫星位置计算   │           │
              │ (Skyfield)   │           │
              └──────┬───────┘           │
                     ↓                   ↓
              ┌──────────────┐   ┌──────────────┐
              │ ISL拓扑计算    │──→│ HierarchicalEnv│
              │ 延迟矩阵      │   │ 多域环境      │
              │ 多普勒频移    │   └──────┬───────┘
              │ ISL持续时间   │          │
              └──────────────┘          ↓
                                 ┌──────────────┐
                                 │ FlowGenerator │
                                 │ Poisson流生成  │
                                 └──────┬───────┘
                                        ↓
                              ┌──────────────────┐
                              │  HRL Trainer      │
                              │ ┌──────────────┐  │
                              │ │Upper Agent   │  │
                              │ │DQN+Pheromone │  │
                              │ └──────┬───────┘  │
                              │        ↓ subgoal  │
                              │ ┌──────────────┐  │
                              │ │Lower Agent   │  │
                              │ │GNN+DQN       │  │
                              │ └──────────────┘  │
                              └────────┬──────────┘
                                       ↓
                              ┌──────────────────┐
                              │ 性能评估          │
                              │ 成功率/延迟/利用率 │
                              └──────────────────┘
```

---

## 五、关键实现挑战与解决方案

### 5.1 Walker 星座生成

**挑战**：论文用 MATLAB 卫星工具箱，项目中 StarPerf 用 SGP4/XML，Adaptive 用 MATLAB/STK。

**方案**：使用 StarPerf 的 `by_XML` 模块。需要创建 Walker 星座的 XML 配置文件：

```xml
<!-- config/XML_constellation/Walker96.xml -->
<constellation name="Walker96">
    <shell>
        <altitude>550</altitude>
        <inclination>53</inclination>
        <num_of_orbit>8</num_of_orbit>
        <num_of_satellite_per_orbit>12</num_of_satellite_per_orbit>
    </shell>
</constellation>
```

StarPerf 的 `constellation_generation/by_XML/` 可以读取此配置并生成星座，但需要验证是否支持 Walker 相位因子 F。

### 5.2 ISL 持续时间与多普勒频移

**挑战**：两个项目都未实现这些物理量的精确计算。

**方案**：
1. **ISL 持续时间**：利用 StarPerf 的 SGP4 传播器计算卫星相对位置，根据论文公式 (5)-(7) 实现 $l_{i,j}(t)$
2. **多普勒频移**：从 SGP4 获取两颗卫星的相对速度，计算 $\Delta\lambda = \frac{v_r}{c} \cdot \lambda_0$

```python
def compute_doppler_shift(sat_i_pos, sat_i_vel, sat_j_pos, sat_j_vel, wavelength=1550e-9):
    """计算星间链路的多普勒频移"""
    relative_pos = sat_j_pos - sat_i_pos
    relative_vel = sat_j_vel - sat_i_vel
    distance = np.linalg.norm(relative_pos)
    # 径向速度分量
    radial_velocity = np.dot(relative_vel, relative_pos) / distance
    doppler_shift = (radial_velocity / 3e8) * wavelength
    return abs(doppler_shift)

def compute_isl_duration(phase_factor, num_orbits, sats_per_orbit, 
                         orbital_period, polar_boundary_angle):
    """计算ISL持续时间 (论文公式5-7)"""
    delta_omega_f = 2 * np.pi * phase_factor / (num_orbits * sats_per_orbit)
    T_d = (np.pi - 2 * polar_boundary_angle + delta_omega_f) / (2 * np.pi / orbital_period)
    return orbital_period / 2 - T_d  # 有效ISL时间
```

### 5.3 从"基于包"到"基于流"的转换

**挑战**：Adaptive 是包级仿真（逐跳转发），论文是流级仿真（端到端路径分配）。

**关键区别**：

| 包级仿真 (Adaptive) | 流级仿真 (论文) |
|---------------------|----------------|
| 包逐跳移动 | 流一次性分配端到端路径 |
| 每时隙每个包独立决策 | 流到达时一次性决策 |
| 队列容量做拥塞控制 | 链路带宽做拥塞控制 |
| 包到达即释放 | 流持续占用带宽直到服务结束 |
| 无带宽概念 | 每条流有明确带宽需求 |

**方案**：需要**根本性重构** Adaptive 的环境，从逐跳包转发改为端到端流路径分配。核心变化：

1. `Packet` → `ServiceFlow`（增加 bandwidth_demand, duration, delay_tolerance）
2. `dynetwork.sending_queue` → `BandwidthManager.active_flows`
3. `DQN逐跳选下一跳` → `上层选跨域链路 + 下层从k条路径选一条`
4. `队列容量限制` → `链路带宽容量限制`
5. 每个时隙不再move packets，而是：生成新流 → 路由决策 → 分配带宽 → 到期释放

### 5.4 多智能体训练

**挑战**：Adaptive 是单智能体 (per-destination DQN)，论文是多智能体 (per-domain upper+lower)。

**方案**：
- 上层：每个域一个 DQN 智能体，独立训练，通过信息素间接协作
- 下层：每个域一个 GNN+DQN 智能体，独立训练
- 数量：域数 = 轨道面数 (8/12/16 个智能体)，远少于 Adaptive 的 500 个

### 5.5 预计算 k 条路径

**挑战**：下层智能体不是逐跳路由，而是从 k 条预计算路径中选择。

**方案**：使用 NetworkX 的 `k_shortest_paths`：

```python
import networkx as nx

def precompute_k_paths(graph, source, target, k=4, weight='delay'):
    """预计算域内 k 条最短路径"""
    paths = list(nx.shortest_simple_paths(graph, source, target, weight=weight))
    return paths[:k]
```

---

## 六、实现路线图

### Phase 1：基础设施（1-2周）

| 任务 | 来源 | 工作量 |
|------|------|--------|
| Walker 星座 XML 配置文件 | 新建 | 小 |
| StarPerf 星座生成适配 | 复用StarPerf | 小 |
| ISL 拓扑 + 延迟矩阵计算 | 复用StarPerf | 小 |
| 多普勒频移计算模块 | 新建 | 中 |
| ISL 持续时间计算模块 | 新建 | 中 |
| 域划分器 (DomainPartitioner) | 新建 | 中 |

### Phase 2：仿真环境（2-3周）

| 任务 | 来源 | 工作量 |
|------|------|--------|
| ServiceFlow 流实体类 | 改造 Adaptive Packet | 小 |
| FlowGenerator (Poisson) | 新建 | 小 |
| BandwidthManager | 新建 | 中 |
| PheromoneManager | 新建 | 中 |
| HierarchicalSatEnv (Gym) | 重构 Adaptive our_env | 大 |
| 约束检查 C1-C4 | 新建 | 中 |

### Phase 3：智能体网络（2-3周）

| 任务 | 来源 | 工作量 |
|------|------|--------|
| 上层 DQN 网络 | 复用 Adaptive DQN + 扩展 | 小 |
| 下层 GNN 消息传递网络 | 新建 | 大 |
| 多智能体训练循环 | 重构 Adaptive DeepQSimulation | 大 |
| 上层奖励函数（含信息素） | 新建 | 中 |
| 下层奖励函数（BW+delay） | 新建 | 中 |
| 经验回放（复用） | 复用 Adaptive replay_memory | 小 |

### Phase 4：评估与对比（1-2周）

| 任务 | 来源 | 工作量 |
|------|------|--------|
| 基线算法实现 (SR-C, SR-D, ACO, DQN) | 新建 | 中 |
| 150步性能测试框架 | 新建 | 中 |
| 成功率/延迟/链路利用率统计 | 新建 | 小 |
| 收敛曲线绘制 | 复用 Adaptive plot_different_algs | 小 |

---

## 七、技术栈建议

```
Python >= 3.10
├── PyTorch >= 1.12         (DQN + GNN)
├── PyTorch Geometric       (消息传递 GNN, 可选)
├── NetworkX                (图数据结构)
├── Gymnasium               (强化学习环境)
├── Skyfield / SGP4         (轨道传播, 来自 StarPerf)
├── NumPy / SciPy           (科学计算)
├── h5py                    (HDF5 数据存储)
└── Matplotlib              (绘图)
```

---

## 八、总结

论文提出的仿真系统本质上是一个**基于流的多域协作卫星网络路由仿真器**，与两个现有项目相比：

- **StarPerf** 提供了物理层支撑（星座生成、轨道传播、ISL计算），但缺乏路由智能和流量动态仿真
- **Adaptive** 提供了 DQN 路由框架和训练流水线，但缺乏多域划分、GNN、信息素、以及基于流的带宽管理

实现论文仿真的核心工作量在于：
1. **从包到流的范式转换**（Adaptive 的根本重构）
2. **多域划分与跨域路由**（全新模块）
3. **GNN 消息传递网络**（替代全连接 DQN）
4. **信息素协同机制**（全新模块）
5. **物理约束计算**（多普勒、ISL持续时间）

预估总工作量：**6-10 周**（单人全职），其中约 40% 可从现有项目复用。
