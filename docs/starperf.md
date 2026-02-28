

## StarPerf Simulator 项目分析

### 一、项目概述

**StarPerf 2.0** 是一个由清华大学团队开发的 **大型LEO卫星星座网络性能仿真平台**，用于评估 Starlink、OneWeb、Kuiper、Boeing、Telesat 等新兴巨型星座的网络性能。纯 Python 实现（≥3.10），无需依赖 STK 等第三方轨道工具。

- **版本**: 2.0.0 | **许可证**: BSD-2-Clause
- **包管理**: 使用 [uv](https://github.com/astral-sh/uv)

---

### 二、架构总览

项目采用 **插件化（Plugin Manager）** 架构，核心模块之间通过插件管理器解耦：

```
StarPerf_Simulator/
├── config/          # 星座配置文件（XML/JSON/TLE）
├── data/            # 仿真输出数据（HDF5 格式）
├── src/             # 核心源代码
│   ├── constellation_generation/   # 星座生成（4种模式）
│   ├── XML_constellation/          # 基于XML配置的仿真引擎
│   ├── TLE_constellation/          # 基于TLE数据的仿真引擎
│   └── standalone_module/          # 独立计算模块
├── samples/         # 测试用例与示例脚本
├── kits/            # 工具脚本集
└── visualization/   # Cesium 3D 可视化
```

---

### 三、核心模块详解

#### 1. 星座生成模块 (`src/constellation_generation/`)

| 子模块 | 功能 |
|--------|------|
| `by_XML/` | 从 XML 配置文件读取星座参数（轨道高度、倾角、卫星数等），通过 SGP4/Skyfield 计算卫星轨迹 |
| `by_TLE/` | 从 TLE（两行轨道根数）数据生成真实星座，支持在线下载最新 TLE |
| `by_duration/` | 自定义仿真时长和采样间隔的星座生成 |
| `by_manual/` | 从 JSON 手动配置生成星座 |

#### 2. 实体模型 (`constellation_entity/`)

**核心数据类**（XML_constellation 和 TLE_constellation 各有一套）：

| 类 | 描述 |
|----|------|
| `constellation` | 星座对象：名称 + shells 列表 |
| `shell` | 壳层：高度、轨道数、每轨卫星数、倾角、轨道周期 |
| `orbit` | 轨道：包含一组卫星 |
| `satellite` | 卫星：经度/纬度/高度时序列表、ISL 连接、真近点角、SGP4 卫星对象 |
| `ISL` | 星间链路：连接两颗卫星，记录延迟 |
| `ground_station` | 地面站：经纬度 |
| `POP` | PoP 点：接入点 |
| `user` | 用户/终端 |

#### 3. 连接模式 (`constellation_connectivity/`)

通过 **插件管理器** 动态加载连接模式：

- **+Grid 模式** (positive_Grid.py)：每颗卫星建立 4 条 ISL（同轨前后 + 邻轨左右），典型的网格拓扑
- **Bent-pipe 模式** (`bent_pipe.py`)：无 ISL，所有通信经地面站中继
- **N-nearest 模式**（TLE 独有）：连接 N 个最近邻卫星
- **Ring 模式**（TLE 独有）：环形拓扑

#### 4. 性能评估 (`constellation_evaluation/`)

分为 **有ISL** 和 **无ISL** 两个子目录：

| 指标 | 说明 |
|------|------|
| delay.py | 基于 Dijkstra 算法计算端到端最短延迟，使用 NetworkX 构建时变拓扑图 |
| `bandwidth.py` | 计算两地之间可用带宽 |
| `betweeness.py` | 计算星座的介数中心性（识别瓶颈链路） |
| `coverage.py` | 使用 H3 六边形网格计算星座覆盖率 |

#### 5. 路由策略 (`constellation_routing/`)

- **shortest_path** — 最短路径（Dijkstra，基于延迟权重）
- **second_shortest_path** — 次短路径
- **least_hop_path** — 最少跳数路径

每种路由：先找源/目标最近卫星 → 构建 NetworkX 图 → 计算路径

#### 6. 流量仿真 (`constellation_traffic/`)

positive_grid_traffic.py — 基于人口密度权重生成地理分布式流量，建模：
- 卫星-地面接入映射
- 星间路由路径
- ISL/上行/下行链路负载分配

#### 7. 能耗仿真 (`constellation_energy/`)

communication_link.py — 建模通信链路能耗：
- GSL 发射/接收功耗（active/idle/tail 三态）
- ISL 发射/接收功耗
- 参数化配置（空闲功率、激活功率、每流量单位功率等）

#### 8. 安全攻击仿真 (`constellation_attack/`)

| 攻击 | 描述 |
|------|------|
| positive_grid_icarus.py | **ICARUS 链路洪泛攻击**（NDSS '25），基于人口密度部署恶意终端，拥塞 GSL 链路 |
| `positive_grid_energy_drain.py` | **能量耗尽攻击**（IWQoS '23），长期模拟使卫星通信子系统持续处于高功耗状态 |
| `icarus_single_link_attack.py` | 单链路 ICARUS 攻击 |

#### 9. 高可生存性 (`constellation_highsurvivability/`)

- **natural_failure_satellites** — 模拟卫星自然故障后的性能退化
- **sunstorm_damaged_satellites** — 模拟太阳风暴区域性损毁卫星后的性能

#### 10. 独立模块 (`standalone_module/`)

| 模块 | 功能 |
|------|------|
| polar_constellation_coverage.py | 极轨星座覆盖率蒙特卡洛计算 |
| `satellite_orbital_period.py` | 卫星轨道周期计算 |
| `satellite_redundancy.py` | 卫星冗余度分析 |
| `satellite_visibility_time.py` | 地面可见时间计算 |

---

### 四、数据流与处理流程

```
XML/TLE/JSON 配置
       ↓
  星座生成 (constellation_configuration)
       ↓
  卫星轨迹计算 → 写入 HDF5 (data/XML_constellation/*.h5)
       ↓
  连接建立 (positive_Grid / bent_pipe / ...)
       ↓
  延迟矩阵计算 → 写入 HDF5 (delay group)
       ↓
  ┌──────────────┬──────────────┬──────────────┐
  性能评估        路由计算        流量/能耗/攻击仿真
  (delay/BW/     (shortest/     (traffic →
   coverage/      least_hop)     energy →
   betweenness)                  attack)
       ↓
  Cesium 3D 可视化 (HTML)
```

### 五、支持的星座

在 config/XML_constellation/ 中已配置：

| 星座 | 配置示例 |
|------|---------|
| **Starlink** | 4层shell，550/570/560/540km，72×22 + 36×20 + 10×52 + 72×22 |
| **Kuiper** | Amazon 的星座方案 |
| **OneWeb** | 低轨通信星座 |
| **Boeing** | Boeing V-band 星座 |
| **Telesat** | Telesat Lightspeed |
| **Polar** | 极轨覆盖星座 |

### 六、关键依赖

| 库 | 作用 |
|----|------|
| `skyfield` + `sgp4` | 卫星轨道传播与位置计算 |
| `poliastro` + `astropy` | 天体力学与轨道分析 |
| `networkx` | 图论建模（路由、介数中心性） |
| `h5py` | HDF5 数据存储（位置、延迟矩阵） |
| `h3` | Uber H3 六边形网格（覆盖率） |
| `numpy` + `pandas` | 数值计算 |
| `pyecharts` | 图表可视化 |
| `global-land-mask` | 陆地/海洋判定 |
| `jenkspy` | 自然间断法分类 |
| `ruptures` | 变点检测 |

### 七、总结评价

| 维度 | 评价 |
|------|------|
| **功能完整性** | 覆盖星座生成、连接建模、性能评估、路由、流量、能耗、攻击、可视化全链路 |
| **学术价值** | 复现了 NDSS'25 和 IWQoS'23 论文的攻击模型 |
| **架构设计** | 插件化 Manager 模式，易于扩展新的连接/路由/攻击策略 |
| **代码风格** | 注释详尽，但存在 XML/TLE 两套引擎大量重复代码，可进一步抽象 |
| **数据管理** | 使用 HDF5 分层存储，时间片粒度的位置和延迟数据 |
| **潜在改进** | 可引入基类/继承减少 XML/TLE 重复；增加类型提示；流量/攻击模块使用全局变量较多 |

User: 详细分析其流量方针方式，以及一般情况下如何仿真网络中的流量，如何构建端到端的流量



## StarPerf 流量仿真方式详细分析

### 一、流量仿真总体架构

流量仿真采用 **插件化架构**，由 `traffic_plugin_manager` 管理，当前实现了 `positive_grid_traffic` 插件：

```
traffic_plugin_manager.py          ← 插件管理器，动态加载/切换流量模型
└── traffic_plugin/
    └── positive_grid_traffic.py   ← +Grid 拓扑下的流量生成实现 (506行)
```

调用方式（见 traffic_generation_cases.py）：

```python
Traffic = traffic_plugin_manager.traffic_plugin_manager()
for t in range(1, duration + 1):
    Traffic.execute_traffic_policy(constellation, t)  # 逐时隙生成
```

---

### 二、流量仿真的完整流水线（6 个阶段）

#### 阶段 1：地理网格划分与卫星位置加载

**地面用户建模**：将地球表面按 **1°×1° 网格** 划分为 `inclination×2×360` 个块（block）。以 Starlink shell1（倾角 53°）为例：
- 纬度范围：[-53°, +53°]，共 106 行
- 经度范围：[-180°, +180°]，共 360 列
- 总计 **38,160 个地面块**，每个块中心作为用户位置

```python
for lat in range(inclination, -inclination, -1):
    for lon in range(-180, 180, 1):
        user_pos_car.append(cir_to_car_np(lat - 0.5, lon + 0.5, 0))
```

**卫星位置**：从预计算的 HDF5 文件加载指定时隙的卫星经纬度高度，转换为笛卡尔坐标。以 Starlink shell1 为例，72 轨道 × 22 颗/轨 = **1,584 颗卫星**。

#### 阶段 2：基于人口密度的流量权重

从 data/starlink_count.txt 加载全球人口密度数据：
- **180 行 × 360 列**（1°×1° 全球网格），覆盖 [-90°, +90°] 纬度
- 截取倾角范围内的行（如 53° 倾角取第 37~142 行）
- 人口密度值作为每个 block 的 **流量生成权重**

```python
for row in range(90 - inclination, 90 + inclination):
    pop_count.extend([float(x) for x in lines[row].split(' ')[:-1]] + [0])
```

> 数据统计：180×360 = 64,800 格，其中 26,195 格有非零人口值。值域示例：从 90,645（低密度）到 487,015（高密度）。

#### 阶段 3：用户-卫星接入映射

为每个地面块找到 **最近的可见卫星**：

```python
for user_id in range(inclination * 2 * 360):
    dis2 = np.sqrt(np.sum(np.square(sat_pos_car - user_pos_car[user_id]), axis=1))
    if min(dis2) > bound:          # 超出最低仰角限制
        user_connect_sat.append(-1) # 无法接入
    else:
        min_dis_sat = np.argmin(dis2)
        user_connect_sat.append(min_dis_sat)
```

**可见性约束**：基于最低仰角（默认 25°）计算最大接入距离：

$$bound = \sqrt{(R + h)^2 - (R \cos\theta_{min})^2} - R \sin\theta_{min}$$

其中 $R = 6371$ km（地球半径），$h$ 为轨道高度，$\theta_{min}$ 为最低仰角。

#### 阶段 4：卫星-地面站接入映射

类似地，为每颗卫星找到最近的可见地面站（GS）：

```python
for sat_id in range(sat_num):
    dis2 = np.sqrt(np.sum(np.square(GS_pos_car - sat_pos_car[sat_id]), axis=1))
    if min(dis2) > bound:
        sat_connect_gs.append(-1)   # 该卫星无法连接任何GS
    else:
        sat_connect_gs.append(np.argmin(dis2))
```

GS 数据来自 data/GS.json，共 **165 个 Starlink 地面站**，含经纬度、天线数、容量等。

#### 阶段 5：路由计算（改进 Floyd 算法）

这是流量仿真的核心路由逻辑。在 +Grid 拓扑中，每颗卫星有 6 条链路：

| 链路索引 | 类型 | 描述 |
|----------|------|------|
| `sat_id × 6 + 0` | ISL | 同轨下一颗（intra-orbit forward） |
| `sat_id × 6 + 1` | ISL | 邻轨（inter-orbit right） |
| `sat_id × 6 + 2` | ISL | 同轨上一颗（intra-orbit backward） |
| `sat_id × 6 + 3` | ISL | 邻轨（inter-orbit left） |
| `sat_id × 6 + 4` | GSL | 下行链路（downlink） |
| `sat_id × 6 + 5` | GSL | 上行链路（uplink） |

**路由策略**：`floyd()` 函数为每颗卫星计算到达最近 **着陆卫星**（landing satellite，即有 GS 连接的卫星）的下一跳：

```python
def floyd():
    for orbit_id in range(num_of_orbit):
        for sat_id in range(sat_of_orbit):
            next_sat, sat_index = find_next_sat(orbit_id, sat_id, ...)
            path[sat_index] = next_sat  # path[i] = 下一跳卫星
```

`find_next_sat()` 使用 **曼哈顿距离**（轨道差 + 轨内卫星差）在环面拓扑上找最近着陆卫星，并确定贪心方向：

```python
orbit_diff = min(|orbit_i - orbit_j|, num_of_orbit - |orbit_i - orbit_j|)
sat_diff   = min(|sat_i - sat_j|, sat_per_orbit - |sat_i - sat_j|)
hops = orbit_diff + sat_diff  # 曼哈顿距离（环面拓扑）
```

> 这是一种 **贪心最短路径路由**：不使用 Dijkstra，而是基于 +Grid 的规则拓扑直接计算最优方向，效率更高。

#### 阶段 6：流量注入与链路负载计算

**流量源选择**（基于人口密度的加权随机抽样）：

```python
def init_flows():
    for block in range(inclination * 2 * 360):
        if pop_count[block] == 0: continue
        weight = math.ceil(pop_count[block])
        flows.append([block, weight])
        flows_cumulate_weight.append(flows_sum_weight)  # 累积权重
```

使用 **二分查找** 实现 $O(\log n)$ 的加权随机抽样：

```python
def get_one_flow(cumulate_weight, num, sum_weight):
    rand = random.randint(1, sum_weight)
    # 二分查找 rand 在累积权重数组中的位置
```

> 人口越密集的区域，被选为流量源的概率越高。

**端到端流量注入** (`add_flow()`)：

```
用户Block → [上行链路] → 接入卫星 → [ISL跳转...] → 着陆卫星 → [下行链路] → 地面站
```

每条流大小固定为 `Flow_size`（默认 0.5 Mbps），注入过程：

1. 从源 block 找到接入卫星 `src_sat`
2. 沿 `path[]` 逐跳转发到着陆卫星
3. 沿途累加每条链路的流量
4. **容量检查**：若下行链路超过 `downlink_capacity`（默认 4096 Mbps），则 **回滚** 该流（撤销所有链路累加）

```python
for add_flow_times in range(2000000):  # 尝试注入 200万条流
    flow_id = get_one_flow(...)        # 加权随机选源
    res = add_flow(flows[flow_id][0], Flow_size)
    if res == -1: continue             # 容量溢出，跳过
```

---

### 三、端到端流量构建的关键设计

```
┌──────────┐     上行GSL      ┌──────────┐    ISL×N跳     ┌──────────┐     下行GSL     ┌──────┐
│ 地面Block │ ──────────────→ │ 接入卫星  │ ──────────→ │ 着陆卫星  │ ──────────────→ │  GS  │
│(src_block)│  uplink 4Gbps   │ (src_sat) │  20Gbps/ISL  │(land_sat)│  downlink 4Gbps │      │
└──────────┘                  └──────────┘               └──────────┘                 └──────┘
```

**流量模型特点**：

| 特性 | 实现方式 |
|------|---------|
| **流量源分布** | 基于全球 1°×1° 人口密度网格的加权随机抽样 |
| **流量粒度** | 固定大小流（默认 0.5 Mbps/条），共尝试注入 200 万条 |
| **流量方向** | 单向：用户 → 卫星 → 地面站（下行为主） |
| **路由方式** | +Grid 环面贪心最短路径（曼哈顿距离） |
| **容量约束** | 下行链路硬约束（4096 Mbps），溢出即回滚 |
| **链路负载统计** | ISL 发送/接收、上行/下行分别记录 |

---

### 四、链路容量参数

| 链路类型 | 默认容量 | 说明 |
|----------|---------|------|
| ISL | 20,480 Mbps (~20 Gbps) | 星间激光链路 |
| 上行 GSL | 4,096 Mbps (~4 Gbps) | 用户→卫星 |
| 下行 GSL | 4,096 Mbps (~4 Gbps) | 卫星→地面站 |
| 链路利用率 | 50%（ratio=0.5） | 实际可用 = 容量 × ratio |

---

### 五、输出数据

每个时隙生成以下文件到 `data/{cons_name}_link_traffic_data/{time_slot}/`：

| 文件 | 内容 | 维度 |
|------|------|------|
| `isl_traffic.txt` | 每颗卫星的 ISL 总流量 | 1584×1 |
| `isl_sender_traffic.txt` | ISL 发送流量 | 1584×1 |
| `isl_receiver_traffic.txt` | ISL 接收流量 | 1584×1 |
| `downlink_traffic.txt` | 下行流量 | 1584×1 |
| `uplink_traffic.txt` | 上行流量 | 1584×1 |
| `sat_connect_gs.txt` | 卫星-GS 映射 | 1584×1 |
| `user_connect_sat.txt` | 用户-卫星映射 | 38160×1 |
| `gsl_occurrence_num.txt` | 每个 GSL 服务的用户块数 | 1584×1 |

---

### 六、与一般网络流量仿真的对比

| 维度 | StarPerf 的实现 | 一般做法/可改进方向 |
|------|----------------|-------------------|
| **流量矩阵生成** | 基于人口密度的单源加权抽样 | 可用 gravity model（引力模型）生成 OD 对间流量：$T_{ij} = k \cdot \frac{P_i \cdot P_j}{d_{ij}^\alpha}$ |
| **流量大小分布** | 固定大小（0.5 Mbps） | 可用重尾分布（Pareto/Lognormal）模拟真实互联网流量 |
| **流量方向** | 单向（用户→GS） | 可增加双向/P2P 流量、回程流量 |
| **时间特性** | 各时隙独立生成 | 可引入日变化模式（diurnal pattern）、突发流量模型 |
| **路由** | 贪心静态路由 | 可使用 ECMP/OSPF/Segment Routing 等动态路由 |
| **拥塞控制** | 下行硬限制 + 回滚 | 可模拟 TCP 拥塞窗口、队列调度 |
| **流量类型** | 不区分业务类型 | 可分视频/网页/IoT 等不同 QoS 等级 |

---

### 七、总结

StarPerf 的流量仿真核心思路是：**人口密度驱动的端到端流量注入 + +Grid 环面贪心路由 + 容量约束下的蒙特卡洛采样**。它通过 200 万次随机流注入试验，构建出一个与真实 Starlink 网络地理分布相关的流量矩阵，最终输出每颗卫星每条链路的负载数据，可供后续能耗分析和攻击仿真使用。该方法简洁高效，但在流量建模的真实性（流大小分布、时间动态、双向流量）方面仍有提升空间。

User: 为什么是采用用户接入卫星，然后卫星选择最近的着陆地面站这种方式

GitHub Copilot: 

User: @agent 重试

GitHub Copilot: 

 这是由 **LEO 卫星互联网的实际通信架构** 决定的。以 Starlink 为例：

### 1. 架构原因：Bent-pipe + ISL 混合回传

LEO 卫星星座的互联网接入本质上是一种 **"空中基站"** 模式。用户的目标不是与另一个用户直接通信，而是 **接入互联网骨干网**：

```
用户终端 → 上行GSL → 接入卫星 → [ISL转发...] → 着陆卫星 → 下行GSL → 地面站(GS) → 互联网骨干
```

地面站（PoP/GS）是卫星网络与地面互联网的 **唯一出口**。无论用户访问什么服务（网页、视频、云），流量最终都必须落地到某个 GS，再经地面光纤到达数据中心。这不是设计选择，而是 **物理约束**。

### 2. 为什么选最近的着陆卫星？

**不是所有卫星都连着地面站。** 以代码为例：

```python
# 只有少数卫星能看到GS
for sat_id in range(sat_num):        # 1584颗卫星
    if min(dis2) > bound:
        sat_connect_gs.append(-1)     # 大量卫星没有GS → 无法着陆
    else:
        sat_connect_gs.append(...)    # 少数卫星有GS连接 → "着陆卫星"
```

Starlink 在全球仅有 **165 个地面站**，主要分布在北美/欧洲/澳洲等陆地区域。1584 颗卫星中，**大部分时刻都在海洋上空**，根本没有可见的地面站。所以：

- **接入卫星**（用户头顶最近的卫星）≠ **着陆卫星**（有 GS 的卫星）
- 两者之间需要通过 ISL 中继，自然选择 **跳数最少（曼哈顿距离最短）** 的着陆卫星

### 3. 为什么不直接用户→卫星→地面站？

因为用户头顶的卫星大概率 **看不到任何地面站**。例如：
- 一个太平洋中间的用户，头顶卫星方圆几千公里内没有任何 GS
- 必须通过 ISL 把流量转发到有 GS 的陆地上空卫星

### 4. 现实对照

这正是 Starlink **V2 卫星（带激光ISL）** 的实际工作方式：

| 阶段 | Starlink 实际 | 代码模型 |
|------|--------------|---------|
| 用户接入 | 用户终端天线锁定最低仰角以上的最佳卫星 | `user_connect_sat[block] = argmin(距离)` |
| ISL 转发 | 通过 4 条激光 ISL 在网格中逐跳转发 | `path[sat] → 下一跳 → ... → 着陆卫星` |
| 落地回传 | 到达有 GS 的卫星，下行回传至地面 | `sat_connect_gs[landing_sat] → GS` |
| 入骨干网 | GS 连地面光纤进入互联网 | 模型边界（不再模拟） |

### 5. 为什么不建模端到端用户-用户？

代码建模的是 **接入流量（access traffic）** 而非 **对等流量（peer-to-peer traffic）**。这是因为：

- 现实中 >95% 的互联网流量是 **用户-数据中心** 模式（看视频/刷网页/云服务），数据中心都在地面
- 卫星-卫星之间的对等通信（如两个 Starlink 用户直接通话）需要 **上行→ISL→下行** 两段 GSL + 中间 ISL，但整体流量占比极小
- 该仿真的主要目的是为后续 **攻击分析**（ICARUS 链路洪泛、能量耗尽）提供负载基线，关注的是链路级负载而非端到端 QoS

**简言之：这种"用户→接入星→ISL→着陆星→GS"的流量模式，是 LEO 卫星互联网回传到地面骨干网的物理必然，不是设计选择。**