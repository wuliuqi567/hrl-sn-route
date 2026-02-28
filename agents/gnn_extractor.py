"""
GNN 特征提取器 — 可选的 GNN 策略主干网络，可插入 SB3 DQN。

GNN Feature Extractor — optional GNN-based policy backbone for SB3.

架构角色 / Architecture Role:
  - SB3 支持自定义 features_extractor_class 插入策略网络前端
  - 本模块提供基于 DGL (Deep Graph Library) 的 GNN 层
  - 将平坦观测向量重塑为图结构 → GNN 消息传递 → 图池化 → 特征向量

适用场景:
  - hop-by-hop 模式: 邻居拓扑关系对决策有显著影响
  - K-path 模式: 候选路径可视为图中的节点 (实验性)
  - 默认 MlpPolicy 在大多数情况下已足够

使用方式 / Usage with SB3:
    from agents.gnn_extractor import GNNFeaturesExtractor

    policy_kwargs = dict(
        features_extractor_class=GNNFeaturesExtractor,
        features_extractor_kwargs=dict(
            features_dim=64,
            n_gnn_layers=2,
            hidden_dim=64,
        ),
    )
    model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, ...)

兼容性 / Compatibility:
  - 兼容所有使用 BaseFeaturesExtractor 的 SB3 算法
    (DQN, DDQN, QRDQN, PPO, A2C, SAC 等)
  - 需要安装 DGL: pip install dgl
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import torch
import torch.nn as nn

try:
    import dgl
    from dgl.nn.pytorch import GraphConv
    HAS_DGL = True
except ImportError:
    HAS_DGL = False

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GNNFeaturesExtractor(BaseFeaturesExtractor):
    """SB3 兼容的 GNN 特征提取器。

    SB3-compatible feature extractor that uses GNN message passing.

    数据流 / Data Flow:
      1. 观测向量切分为: 节点特征 (前 n_nodes×node_feat_dim 维) + 全局特征 (剩余维)
      2. 节点特征 reshape 为 (batch, n_nodes, node_feat_dim)
      3. 经过 n_gnn_layers 层 GraphConv 消息传递
      4. 均值池化 (mean-pool) → (batch, hidden_dim)
      5. 拼接全局特征 → 线性层投影到 features_dim

    Parameters
    ----------
    observation_space : gym.spaces.Box
        观测空间。
    features_dim : int
        输出特征向量维度。
    n_nodes : int
        图中节点数 (K-path: K个路径; hop: 4个邻居)。
    node_feat_dim : int
        每个节点的特征维度。
    global_feat_dim : int
        末尾全局特征数 (不经 GNN 处理)。
    n_gnn_layers : int
        GraphConv 层数。
    hidden_dim : int
        GNN 隐藏层维度。
    adj_list : list[tuple[int, int]] | None
        图的边列表。None 表示全连接图。
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 64,
        n_nodes: int = 4,
        node_feat_dim: int = 4,
        global_feat_dim: int = 6,
        n_gnn_layers: int = 2,
        hidden_dim: int = 64,
        adj_list: Optional[list[tuple[int, int]]] = None,
    ):
        super().__init__(observation_space, features_dim)

        if not HAS_DGL:
            raise ImportError(
                "DGL is required for GNNFeaturesExtractor. "
                "Install with: pip install dgl"
            )

        self.n_nodes = n_nodes
        self.node_feat_dim = node_feat_dim
        self.global_feat_dim = global_feat_dim

        # ── Build static graph ──
        if adj_list is not None:
            src = [e[0] for e in adj_list]
            dst = [e[1] for e in adj_list]
            # Add reverse edges for undirected graph
            src_bi = src + dst
            dst_bi = dst + src
            self._graph = dgl.graph((src_bi, dst_bi), num_nodes=n_nodes)
        else:
            # Fully connected (for flexibility — less graph-specific)
            src, dst = [], []
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j:
                        src.append(i)
                        dst.append(j)
            self._graph = dgl.graph((src, dst), num_nodes=n_nodes)

        # ── GNN layers ──
        self.gnn_layers = nn.ModuleList()
        in_dim = node_feat_dim
        for _ in range(n_gnn_layers):
            self.gnn_layers.append(GraphConv(in_dim, hidden_dim, allow_zero_in_degree=True))
            in_dim = hidden_dim

        # ── Projection head ──
        # After mean-pooling: hidden_dim features from GNN + global_feat_dim
        combined_dim = hidden_dim + global_feat_dim
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract features from batch of observations.

        Parameters
        ----------
        observations : Tensor of shape (batch, obs_dim)

        Returns
        -------
        Tensor of shape (batch, features_dim)
        """
        batch_size = observations.shape[0]
        n_node_feats = self.n_nodes * self.node_feat_dim

        # Split observation into node features and global features
        node_obs = observations[:, :n_node_feats]  # (B, n_nodes * node_feat_dim)
        global_obs = observations[:, n_node_feats:]  # (B, global_feat_dim)

        # Reshape to (B, n_nodes, node_feat_dim)
        node_feats = node_obs.view(batch_size, self.n_nodes, self.node_feat_dim)

        # Batch graphs: create B copies of the base graph
        batched_graph = dgl.batch([self._graph.to(observations.device)] * batch_size)

        # Flatten node features: (B * n_nodes, node_feat_dim)
        h = node_feats.reshape(batch_size * self.n_nodes, self.node_feat_dim)

        # GNN message passing
        for layer in self.gnn_layers:
            h = torch.relu(layer(batched_graph, h))

        # Reshape back: (B, n_nodes, hidden_dim)
        h = h.view(batch_size, self.n_nodes, -1)

        # Mean pooling over nodes: (B, hidden_dim)
        h_pool = h.mean(dim=1)

        # Concatenate with global features and project
        combined = torch.cat([h_pool, global_obs], dim=1)
        return self.fc(combined)


# ── Factory helpers for common configurations ──

def make_kpath_gnn_extractor_kwargs(K: int = 4) -> dict:
    """Return ``features_extractor_kwargs`` for the K-path lower agent.

    In K-path mode, each path is treated as a "node" with 4 features:
      (total_delay, min_remaining_bw, hop_count, bottleneck_util)
    and the 6 global features are:
      (bw_demand, delay_budget, entry_lon, entry_lat, exit_lon, exit_lat)
    """
    return dict(
        features_dim=64,
        n_nodes=K,
        node_feat_dim=4,
        global_feat_dim=6,
        n_gnn_layers=2,
        hidden_dim=64,
        adj_list=None,  # fully-connected among K path "nodes"
    )


def make_hop_gnn_extractor_kwargs(adj_list: list[tuple[int, int]]) -> dict:
    """Return ``features_extractor_kwargs`` for the hop-by-hop lower agent.

    In hop mode, the 4 neighbors are nodes with 3 features each:
      (distance_to_exit, remaining_bw, in_domain_flag)
    and the 11 global features are the ego-centric features.
    """
    return dict(
        features_dim=64,
        n_nodes=4,
        node_feat_dim=3,
        global_feat_dim=11,
        n_gnn_layers=1,
        hidden_dim=32,
        # 4-node star graph: center node 0 connected to neighbors 1,2,3
        # but since all are neighbors, use fully-connected
        adj_list=None,
    )
