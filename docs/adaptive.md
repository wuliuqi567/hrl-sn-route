## Adaptive Routing Mechanism for LEO Satellite Network 项目分析

### 一、项目概述

**Adaptive** 是一个基于 **Deep Q-Learning (DQN)** 的 LEO 卫星网络自适应路由仿真项目。项目核心思路是：将卫星网络中的数据包路由问题建模为强化学习问题，通过 DQN 智能体学习在动态变化的网络拓扑中做出最优路由决策。

项目分为两个子模块：

| 子模块 | 功能 | 技术栈 |
|--------|------|--------|
| `StarPerf/` | 构建卫星星座几何场景，生成物理参数（ISL距离/延迟矩阵） | MATLAB + STK v12.2 |
| `Deep_routing/` | 基于 DQN 的动态路由仿真引擎 | Python 3 + PyTorch + NetworkX + OpenAI Gym |

**数据流水线**：STK 生成星座拓扑 → MATLAB 计算延迟矩阵/邻接矩阵 → 导出 CSV → Python DQN 路由仿真

---

### 二、架构总览

```
Adaptive/
├── StarPerf/                          # 卫星星座物理层仿真（MATLAB/STK）
│   ├── constellation_data/            # TLE 数据（Starlink/OneWeb）
│   ├── countries_json/                # 地面站位置 JSON
│   └── matlab_code/                   # MATLAB 主程序
│       ├── build_constellation.m      # 主入口：创建星座 → 计算延迟
│       ├── Create_LEO.m              # 在 STK 中创建 LEO 卫星
│       ├── Create_Fac.m             # 创建地面设施
│       ├── Create_delay.m           # 计算 LEO 间延迟矩阵
│       ├── Create_location.m        # 导出卫星位置
│       └── Lla2Cbf.m                # 经纬高→笛卡尔坐标转换
│
└── Deep_routing/                      # DQN 路由仿真引擎（Python）
    ├── DeepQSimulation.py            # 主入口：训练 + 测试流程
    ├── our_env.py                    # Gym 环境：网络拓扑 + 数据包路由
    ├── our_agent.py                  # DQN 智能体：ε-greedy 策略 + 学习
    ├── dynetwork.py                  # 动态网络类：数据包生成与管理
    ├── Packet.py                     # 数据包实体类
    ├── DQN.py                        # 神经网络结构（3层全连接）
    ├── neural_network.py             # 神经网络管理器（policy + target）
    ├── replay_memory.py              # 经验回放缓冲区
    ├── UpdateEdges.py                # 边动态变化：删除/恢复/权重正弦波动
    ├── satellite_graphy.py           # 从 CSV 邻接矩阵生成 NetworkX 图
    ├── Setting.json                  # 全局配置参数
    ├── plot_different_algs.py        # 多算法对比绘图
    └── starlink_AER/                 # STK 导出的卫星邻接矩阵
        ├── result1.csv               # 时隙1：1584×1584 邻接矩阵
        ├── result2.csv               # 时隙2
        └── result3.csv               # 时隙3
```

---

### 三、核心模块详解

#### 1. 卫星星座生成（StarPerf/matlab_code/）

**依赖**：MATLAB 2019 + AGI STK v12.2

| 文件 | 功能 |
|------|------|
| `build_constellation.m` | 主流程：创建 STK 场景 → 生成星座 → 计算延迟矩阵 |
| `Create_LEO.m` | 从 Excel 配置读取星座参数（高度、倾角、轨道面数、每轨卫星数），在 STK 中用 J4 摄动传播器创建所有卫星 |
| `Create_Fac.m` | 在 STK 中创建地面设施（从 countries_json 读取位置） |
| `Create_delay.m` | 计算 LEO-LEO 间的 ISL 延迟矩阵，基于 +Grid 拓扑（同轨相邻 + 异轨相邻） |
| `Create_location.m` | 导出卫星各时隙的位置（LLA → CBF 笛卡尔坐标） |
| `Lla2Cbf.m` | 经纬度高度到笛卡尔坐标的转换 |

**ISL 连接方式**（Create_delay.m）：标准 +Grid 拓扑
- 同轨内：卫星 j 与卫星 j+1（首尾环形连接）
- 异轨间：轨道 i 的卫星 j 与轨道 i+1 的卫星 j
- 极轨（倾角 80°~100°）：不建立跨轨连接（极区回避）

**延迟计算**：

$$delay(i,j) = \frac{\sqrt{(x_i - x_j)^2 + (y_i - y_j)^2 + (z_i - z_j)^2}}{3 \times 10^5 \text{ km/s}}$$

**输出**：1584×1584 的邻接/延迟矩阵，导出为 CSV 文件（`starlink_AER/result{t}.csv`）。值为 0 表示无连接，非零值表示两卫星间距离（可推算延迟）。

#### 2. 卫星拓扑导入 (`satellite_graphy.py`)

将 MATLAB/STK 导出的 CSV 邻接矩阵转换为 NetworkX 图：

```python
class Network_input():
    def __init__(self, satellite_time):
        self.matrix = ny.loadtxt('starlink_AER/result' + str(satellite_time) + '.csv',
                                  delimiter=",", skiprows=0)
    def graphy_generate(self):
        G = nx.Graph()
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix)):
                if self.matrix[i][j] > 1:  # 阈值过滤：距离>1 才建边
                    G.add_edge(i, j)
        return G
```

> **注意**：当前代码中硬编码了 Windows 路径 `G:/code/...`，实际使用需修改。

#### 3. 动态网络模型 (`dynetwork.py`)

**DynamicNetwork 类** 是整个网络仿真的数据载体：

| 属性 | 类型 | 描述 |
|------|------|------|
| `_network` | `nx.Graph` | NetworkX 图对象（节点含队列，边含延迟） |
| `adjacency_matrix` | `numpy.matrix` | 邻接矩阵（供 DQN 计算使用） |
| `_packets` | `Packets` | 当前网络中所有数据包的集合 |
| `_max_initializations` | `int` | 允许注入的最大额外数据包数 |
| `_deliveries` | `int` | 已成功投递的数据包总数 |
| `_rejections` | `int` | 被拒绝转发的次数 |
| `_delivery_times` | `list[int]` | 每个已投递包的投递耗时列表 |
| `_purgatory` | `list` | 等待重新注入的数据包缓冲队列 |

**节点属性**（为 NetworkX 图的每个节点设置）：

| 属性 | 默认值 | 描述 |
|------|--------|------|
| `sending_queue` | `[]` | 发送队列：等待路由转发的数据包列表 |
| `receiving_queue` | `[]` | 接收队列：刚收到、需等待 edge_delay 个时隙后才可转发 |
| `max_send_capacity` | 20 | 每时隙最大发送数据包数 |
| `max_receive_capacity` | 75 | 队列最大容量（发送+接收总和） |
| `congestion_measure` | 75 | 拥塞度量 |
| `growth` | 0 | 接收队列增长量 |

**边属性**：

| 属性 | 描述 |
|------|------|
| `edge_delay` | 传输延迟（时隙数），随时间正弦波动 |
| `sine_state` | 正弦波动状态，每时隙增加 π/6 |
| `initial_weight` | 初始延迟权重 |
| `num_traversed` | 被数据包经过的次数 |

#### 4. 数据包模型 (`Packet.py`)

```python
class Packet:
    _startPos   # 起始节点 ID
    _endPos     # 目标节点 ID
    _curPos     # 当前所在节点 ID
    _index      # 数据包唯一 ID
    _weight     # 权重（未使用）
    _time       # 已存活时间（用于计算投递延迟）

class Packets:
    packetList  # {packet_id: Packet} 字典，存储网络中所有数据包
    num_Packets # 数据包总数
```

#### 5. DQN 神经网络 (`DQN.py` + `neural_network.py`)

**网络结构**（DQN.py）：3 层全连接网络

```
输入层: num_states + num_extra_params (节点one-hot编码 + 可选队列长度)
  ↓ Linear → Tanh
隐藏层1: 2 × num_states
  ↓ Linear → Tanh
隐藏层2: 2 × num_states
  ↓ Linear（无激活）
输出层: num_states (每个节点一个Q值)
```

**关键设计：每个目标节点一个独立的神经网络**

```python
# our_env.py
def init_dqns(self):
    temp_dqns = []
    for i in range(self.nnodes):             # 500 个节点 → 500 个神经网络
        temp_dqn = NeuralNetwork(i, self.nnodes, self.input_q_size)
        temp_dqns.append(temp_dqn)
    return temp_dqns
```

每个 `NeuralNetwork` 包含：
- `policy_net`：策略网络（用于选择动作）
- `target_net`：目标网络（用于计算 TD-target）
- `replay_memory`：独立经验回放缓冲区（容量 1000）
- `optimizer`：SGD 优化器（lr=0.005）

> 即 **对于每个可能的目标节点，训练一个专门的路由策略网络**。当数据包的目的地为节点 d 时，使用 `self.dqn[d]` 来决定下一跳。

#### 6. 边动态更新 (`UpdateEdges.py`)

模拟卫星网络的动态性（链路中断、延迟波动）：

| 函数 | 机制 | 描述 |
|------|------|------|
| `Delete()` | 随机删边 | 每时隙随机删除 0~10 条边（模拟 ISL 断链） |
| `Restore()` | 随机恢复 | 从已删除的边中随机恢复部分（模拟链路重建） |
| `Sinusoidal()` | 正弦波动 | 边权重按正弦规律波动：$w(t) = \max(1, w_0 \cdot (1 + 0.5\sin(\theta)))$，$\theta$ 每时隙增加 $\pi/6$ |
| `Random_Walk()` | 随机游走 | 边权重随机 ±2 变化（最小为 1） |

#### 7. 经验回放 (`replay_memory.py`)

支持三种采样策略（通过 Setting.json 配置，三选一）：

| 策略 | 方法 | 描述 |
|------|------|------|
| `sample` | `random.sample()` | 均匀随机采样 |
| `take_recent` | `memory[-batch_size:]` | 取最近的经验 |
| `take_priority` | `random.choices(weights=...)` | 优先经验回放（TD-error 越大，采样概率越高） |

---

### 四、流量仿真方式详解

#### 阶段 1：网络拓扑初始化

```python
# our_env.py __init__()
if self.network_type == 'barabasi-albert':
    satellite_net = satellite_graphy.Network_input(satellite_time)
    network = satellite_net.graphy_generate()    # 从 CSV 生成卫星拓扑图
else:
    network = nx.gnm_random_graph(self.nnodes, self.nedges)
```

支持两种拓扑：
1. **卫星拓扑模式**：从 `starlink_AER/result{t}.csv` 读取 1584×1584 的邻接矩阵，生成真实 Starlink 拓扑
2. **随机图模式**：生成 Barabási-Albert 或 GNM 随机图

但 Setting.json 中配置的 `number nodes: 500` 表明实际仿真使用 500 节点（可能是对 1584 的子网截取或使用随机图）。

#### 阶段 2：数据包注入（流量生成）

**初始批量注入**：

```python
dynetwork.randomGeneratePackets(num_packets_to_generate=2500)
```

每个数据包：
- **起始节点**：从未满节点中 **均匀随机** 选取
- **目标节点**：**均匀随机** 选取（不等于起始节点）
- **初始放置**：放入起始节点的 `sending_queue`

**容量约束**：如果节点的 `sending_queue + receiving_queue >= max_receive_capacity(75)`，该节点不可作为起始节点。

**后续补充注入**：当数据包到达目标后，经随机等待 0~5 个时隙，在随机位置重新生成一个同 ID 的新数据包（新的起终点对），实现 **持续流量负载**：

```python
# send_packet() 中数据包到达目标后
self.dynetwork.GeneratePacket(self.packet, random.randint(0, 5))  # 等待0~5时隙后重新注入
```

额外限制：总注入数不超过 `max_additional_packets`（默认 500），即整个仿真最多产生 2500 + 500 = 3000 个数据包。

#### 阶段 3：每时隙网络更新

每个时隙调用 `updateWhole()`，按顺序执行：

```python
def updateWhole(self, agent, learn=True):
    self.change_network()   # 1. 边动态变化（删除/恢复/权重波动）
    self.purgatory()        # 2. 重新注入等待中的数据包
    self.update_queues()    # 3. 接收队列 → 发送队列（延迟到期的包转移）
    self.update_time()      # 4. 所有包的存活时间 +1
    self.router(agent, learn) # 5. 路由所有数据包
```

#### 阶段 4：数据包路由（DQN 决策）

`router()` 函数遍历每个节点的 `sending_queue`，对每个数据包做路由决策：

```python
for nodeIdx in self.dynetwork._network.nodes:
    for i in range(queue_size):
        if sendctr == sending_capacity:  # 每节点每时隙最多发 20 个包
            break
        
        pkt_state = (当前节点, 目标节点)
        nlist = sorted(当前节点的邻居列表)
        
        # 当前状态编码：one-hot(当前节点) [+ 队列长度]
        cur_state = F.one_hot(当前节点, num_nodes)
        
        # DQN 决策下一跳
        action = agent.act(self.dqn[目标节点], cur_state, nlist)
        
        # 执行转发
        reward, ... = self.step(action, 当前节点)
```

**ε-greedy 策略**（our_agent.py）：

```python
def act(self, neural_network, state, neighbor):
    if random.uniform(0, 1) < epsilon:  # 初始 ε=0.7，逐步衰减
        next_step = random.choice(neighbor)  # 探索：随机选邻居
    else:
        qvals = neural_network.policy_net(state.float())  # 利用：DQN 输出 Q 值
        next_step_idx = qvals[:, neighbor].argmax().item() # 从邻居中选 Q 值最大的
        next_step = neighbor[next_step_idx]
    return next_step
```

#### 阶段 5：奖励函数与转发

**转发成功**（send_packet）：

```python
def send_packet(self, next_step):
    pkt.set_curPos(next_step)
    pkt.set_time(pkt.get_time() + edge_delay)  # 累加传播延迟
    
    if pkt.get_curPos() == dest_node:
        # 到达目标！
        reward = 20 * num_nodes    # 大正奖励（=10000 for 500节点）
        # 重新生成数据包（持续流量）
        self.dynetwork.GeneratePacket(self.packet, random.randint(0, 5))
    else:
        # 中间节点：基于拥塞的负奖励
        q = len(sending_queue) + len(receiving_queue)  # 下一跳节点的队列总长
        q_eq = 0.8 * max_queue                         # 平衡点 = 60
        w = 5                                           # 增长权重
        growth = next_node['growth']                    # 接收队列增长量
        reward = -(q - q_eq + w * growth)
```

**奖励函数解读**：

$$R = -(q - q_{eq} + w \cdot g)$$

其中：
- $q$ = 下一跳节点的总队列长度
- $q_{eq} = 0.8 \times 75 = 60$（队列平衡点）
- $w = 5$（增长惩罚权重）
- $g$ = 下一跳节点的接收队列增长量

**奖励设计意图**：
- 当下一跳队列长度 < 60 且不增长时，奖励为正 → 鼓励转发到空闲节点
- 当下一跳队列接近满载时，奖励为负 → 惩罚向拥塞节点转发
- 到达目标获得大正奖励 → 鼓励尽快投递

**拒绝转发的情况**：
1. 当前节点无邻居（`action == None`）
2. 下一跳节点队列已满（`is_capacity()` 返回 True）

被拒绝的数据包会被放回 `sending_queue` 的前端，下个时隙重试。

#### 阶段 6：DQN 学习更新

```python
# our_agent.py learn()
agent.learn(self.dqn[目标节点], cur_state, action, reward, next_state)

# 核心更新
current_q_values = policy_net(states).gather(actions)     # Q(s,a)
next_q_values = target_net(next_states).max()              # max_a' Q'(s',a')
target_q_values = reward + γ * next_q_values               # TD-target

loss = MSE(current_q_values, target_q_values)              # 损失函数
loss.backward()                                             # 反向传播
optimizer.step()                                            # SGD 更新
```

目标网络每隔 `TARGET_UPDATE=10` 个时隙同步一次策略网络的权重。

---

### 五、端到端流量构建

```
┌──────────┐     放入sending_queue     ┌──────────┐   DQN选下一跳    ┌──────────┐         ┌──────────┐
│ 随机源节点 │ ──────────────────────→ │ 当前节点   │ ──────────────→ │ 下一跳节点 │ → ... →│ 目标节点  │
│(startPos) │   初始/重新注入          │ (curPos)  │  ε-greedy      │ (action)  │         │(endPos)  │
└──────────┘                          └──────────┘                  └──────────┘         └──────────┘
                                           ↑                            |
                                           |   receiving_queue          | edge_delay 时隙后
                                           |   等待 edge_delay 后       | 进入 sending_queue
                                           └────────────────────────────┘
```

**流量模型特点**：

| 特性 | 实现方式 |
|------|---------|
| **流量源分布** | 均匀随机：起终点均匀选取，无地理/人口权重 |
| **流量方向** | 双向对等（P2P）：任意节点→任意节点 |
| **流量大小** | 每个数据包为 1 单位（无带宽建模） |
| **流量持续性** | 到达后重新生成：保持网络始终有 ~2500 个活跃包 |
| **路由方式** | DQN 智能体逐跳决策（vs 传统最短路径基线） |
| **拥塞控制** | 队列容量硬限制（75），超限拒绝转发 |
| **网络动态性** | 边随机删除/恢复 + 权重正弦波动 |
| **对等/层次** | 纯对等网络：无地面站、无接入层概念 |

---

### 六、训练与测试流程

#### 训练阶段

```python
for i_episode in range(10):                    # 10 轮 episode
    env.reset(max(network_load))               # 重置网络，注入 2500 包
    for t in range(100):                       # 每 episode 最多 100 个时隙
        env.updateWhole(agent)                 # 更新网络 + 路由所有包
        if t % 10 == 0:
            env.update_target_weights()        # 更新目标网络
        if all_delivered:
            break
    # 收敛判断：连续5个episode奖励变化 < 5% 则提前停止
```

**超参数**（Setting.json）：

| 参数 | 值 | 描述 |
|------|-----|------|
| `epsilon` | 0.7 | 初始探索率 |
| `decay_epsilon_rate` | 0.99998 | ε 衰减率（每次利用后衰减） |
| `gamma` | 0.6 | 折扣因子 |
| `memory_batch_size` | 16 | 经验回放批大小 |
| `memory_bank_size` | 1000 | 回放缓冲区容量 |
| `optimizer_learning_rate` | 0.005 | SGD 学习率 |

#### 测试阶段

训练完成后，$\varepsilon$ 设为 0.0001（几乎纯利用），在不同网络负载（500~2500，步长500）下各测试 3 次：

```python
agent.config['epsilon'] = 0.0001  # 几乎不探索
for curLoad in [500, 1000, 1500, 2000, 2500]:
    for trial in range(3):
        env.reset(curLoad)
        for t in range(100):
            env.updateWhole(agent, learn=False)  # 不更新网络
```

---

### 七、性能评估指标

| 指标 | 计算方式 | 含义 |
|------|---------|------|
| **平均投递时间** | `sum(delivery_times) / num_delivered` | 数据包从源到目标的平均时隙数 |
| **最大队列长度** | 所有节点所有时隙的最大队列长度 | 拥塞程度峰值 |
| **平均队列长度** | 非空节点的平均队列长度 | 网络平均负载水平 |
| **满载节点比例** | 队列满的节点数 / 有包节点数 × 100% | 网络拥塞覆盖率 |
| **空闲节点比例** | 无包节点数 / 总节点数 × 100% | 网络利用不均衡程度 |
| **平均空闲时间** | 拒绝次数 / 总包数 | 数据包等待/被拒绝的频率 |

支持与 **最短路径 (SP)** 和 **Q-Learning** 基线的对比绘图（`plot_different_algs.py`）。

---

### 八、与 StarPerf 流量仿真的对比

| 维度 | Adaptive (DQN) | StarPerf |
|------|----------------|----------|
| **网络层级** | 纯卫星对等网络，无地面站/地面用户 | 完整的 用户→卫星→GS 三层模型 |
| **流量源建模** | 均匀随机注入，无地理权重 | 1°×1° 全球人口密度加权抽样 |
| **流量方向** | 双向 P2P（任意卫星→任意卫星） | 单向上行（用户→GS 着陆） |
| **流量粒度** | 离散数据包（计数型），无带宽概念 | 连续流量（0.5 Mbps/条），有链路带宽容量 |
| **路由方式** | DQN 强化学习逐跳决策 | +Grid 环面贪心曼哈顿距离 |
| **网络动态性** | 边删除/恢复 + 权重正弦波动 | 静态拓扑（每时隙独立计算） |
| **拥塞建模** | 节点队列容量硬限制 + 发送速率限制 | 下行链路带宽硬限制 + 溢出回滚 |
| **评估指标** | 投递时间、队列长度、满载率、空闲率 | ISL/GSL 链路级流量、接入映射 |
| **时间尺度** | 离散时隙（抽象时间单位） | 1秒/时隙，物理时间 |
| **卫星规模** | 配置 500 节点（实际 CSV 为 1584） | Starlink Shell1 = 1584 颗 |
| **仿真目标** | 验证 DQN 路由策略的收敛性和优越性 | 生成网络级流量负载数据供攻击分析 |
| **物理真实度** | 低（抽象图模型，无地球物理） | 高（SGP4 轨道传播，经纬度建模） |

---

### 九、关键设计评价

| 维度 | 评价 |
|------|------|
| **创新点** | 将卫星路由建模为 RL 问题，每个目标节点独立 DQN，考虑了拥塞感知奖励 |
| **可扩展性** | 差：500 节点需要 500 个神经网络，内存和训练时间随节点数线性增长 |
| **流量真实性** | 低：均匀随机注入，缺乏地理分布、时间动态、流量大小分布 |
| **网络动态性** | 中等：边权正弦波动模拟了 ISL 延迟变化，随机删边模拟了链路中断 |
| **代码质量** | 中等：结构清晰但硬编码路径多（Windows 绝对路径），Setting.json 配置与代码默认值不一致 |
| **工程化程度** | 低：无日志、无单元测试、无模块化导入、路径硬编码 |

### 十、潜在改进方向

1. **流量建模增强**：引入人口密度权重（如 StarPerf）、流大小分布（Pareto/Lognormal）、时间周期性
2. **可扩展路由**：用图神经网络（GNN）替代 per-destination DQN，实现 $O(1)$ 网络数量
3. **物理层整合**：将 StarPerf 的 SGP4 轨道传播与 DQN 路由结合，实现端到端仿真
4. **多目标优化**：在奖励函数中加入能耗、丢包率等多维指标
5. **层次化模型**：增加用户→卫星接入层和卫星→地面站回程层

---

### 十一、总结

Adaptive 项目的核心思路是：**将卫星网络路由问题抽象为"在动态图上做序贯决策"的强化学习问题**。通过 STK 生成真实星座拓扑，然后在 Python 中构建内含队列管理、边动态变化的 Gym 环境，用 DQN 智能体学习拥塞感知的逐跳路由策略。

与 StarPerf 的 **宏观流量注入仿真**（关注链路级负载分布）不同，Adaptive 更关注 **微观路由决策**（关注单个数据包的最优路径选择和投递性能）。两者处于不同的抽象层级，StarPerf 是"网络规划"视角，Adaptive 是"路由策略"视角。
