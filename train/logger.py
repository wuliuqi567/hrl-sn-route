"""
训练日志器 — TensorBoard + 控制台统一日志接口。

Training logger — unified TensorBoard + console logging for HRL training.

架构角色 / Architecture Role:
  - 算法无关: 不依赖任何特定 RL 算法
  - 被所有训练脚本共用 (Phase 1/2/3)
  - TensorBoard 可选: 若未安装 tensorboard 则自动降级为仅控制台输出

使用方式 / Usage:
    logger = TrainingLogger(log_dir="runs", prefix="phase1_lower")
    for ep in range(n_episodes):
        metrics = {...}  # success_rate, avg_delay, etc.
        logger.log_episode(ep, n_episodes, metrics)
    logger.close()
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False


class TrainingLogger:
    """HRL 训练指标统一日志器。

    Unified logger for HRL training metrics.
    Logs to TensorBoard (if available) and optionally to console.

    Parameters
    ----------
    log_dir : str
        TensorBoard 日志目录 (默认 "runs")。
    prefix : str
        指标名前缀，用于区分不同训练阶段
        (e.g., "phase1_lower_k_path", "phase2_upper", "phase3_finetune")。
    console_interval : int
        每隔 N 个回合打印一次控制台输出。
    enabled : bool
        若 False，所有日志操作被静默跳过。
    """

    def __init__(
        self,
        log_dir: str = "runs",
        prefix: str = "",
        console_interval: int = 10,
        enabled: bool = True,
    ):
        self.prefix = prefix
        self.console_interval = console_interval
        self.enabled = enabled
        self._writer: Optional[SummaryWriter] = None
        self._step = 0
        self._t0 = time.time()

        if enabled and HAS_TB:
            log_path = Path(log_dir) / prefix if prefix else Path(log_dir)
            self._writer = SummaryWriter(str(log_path))

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """记录一个标量值到 TensorBoard。

        Log a scalar value to TensorBoard.

        Parameters
        ----------
        tag : str
            指标名称 (e.g., "success_rate", "avg_delay")。
        value : float
            指标值。
        step : int | None
            训练步数，默认使用内部计数器。
        """
        if not self.enabled:
            return
        s = step if step is not None else self._step
        if self._writer is not None:
            full_tag = f"{self.prefix}/{tag}" if self.prefix else tag
            self._writer.add_scalar(full_tag, value, s)

    def log_episode(
        self,
        episode: int,
        n_episodes: int,
        metrics: dict,
    ):
        """记录回合级别指标。

        Log episode-level metrics to both TensorBoard and console.

        Parameters
        ----------
        episode : int
            当前回合号 (0-indexed)。
        n_episodes : int
            总回合数。
        metrics : dict
            指标字典，支持的键:
            - success_rate: float, 路由成功率
            - avg_delay: float, 平均时延(秒)
            - upper_exps: int, 上层经验数
            - lower_exps: int, 下层经验数
            - success_count: int, 成功流数
            - fail_count: int, 失败流数
            - 其他任意 int/float 键也会被记录到 TensorBoard
        """
        if not self.enabled:
            return

        self._step = episode

        # TensorBoard
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                self.log_scalar(key, val, episode)

        # Console
        if (episode + 1) % self.console_interval == 0 or episode == 0:
            elapsed = time.time() - self._t0
            sr = metrics.get("success_rate", 0.0)
            upper = metrics.get("upper_exps", 0)
            lower = metrics.get("lower_exps", 0)
            avg_d = metrics.get("avg_delay", float("nan"))

            parts = [f"[Ep {episode+1:4d}/{n_episodes}]"]
            if "success_rate" in metrics:
                parts.append(f"SR={sr:.2%}")
            if upper or lower:
                parts.append(f"exp(u={upper},l={lower})")
            if avg_d != float("nan") and avg_d < float("inf"):
                parts.append(f"delay={avg_d*1000:.1f}ms")
            s = metrics.get("success_count", 0)
            f = metrics.get("fail_count", 0)
            if s or f:
                parts.append(f"({s}✓/{f}✗)")
            parts.append(f"t={elapsed:.0f}s")
            print("  " + "  ".join(parts))

    def close(self):
        """Flush and close the TensorBoard writer."""
        if self._writer is not None:
            self._writer.close()
