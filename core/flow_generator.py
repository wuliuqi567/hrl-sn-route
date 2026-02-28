"""
FlowGenerator — Poisson 流到达生成器。

每个时隙按 Poisson(λ) 生成若干条新流请求，
带宽/时延/持续时间均服从均匀分布，源目的卫星随机选取。

设计考量 / Design Notes:
  - Poisson 到达是电信流量的标准建模方式
  - 均匀分布的带宽/时延范围可配置，模拟不同业务场景
  - 源目的随机选取可产生跨域/同域流的自然比例
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from core.flow import FlowRequest

if TYPE_CHECKING:
    from core.topology import Topology
    from core.flow import FlowManager


class FlowGenerator:
    """Poisson traffic flow generator.

    Parameters
    ----------
    topology : Topology
        Used to sample random source/destination satellites.
    flow_manager : FlowManager
        Provides flow-id allocation.
    arrival_rate : float
        Mean number of new flows per timeslot (Poisson λ).
    bw_range : tuple[float, float]
        (min, max) bandwidth demand per flow in Mbps.
    delay_range : tuple[float, float]
        (min, max) delay budget per flow in seconds.
    duration_range : tuple[int, int]
        (min, max) flow duration in timeslots.
    rng : np.random.Generator | None
        Numpy random generator.  If None, a default one is created.
    """

    def __init__(
        self,
        topology: "Topology",
        flow_manager: "FlowManager",
        arrival_rate: float = 5.0,
        bw_range: tuple[float, float] = (10.0, 200.0),
        delay_range: tuple[float, float] = (0.02, 0.10),
        duration_range: tuple[int, int] = (5, 30),
        rng: np.random.Generator | None = None,
    ):
        self.topology = topology
        self.flow_manager = flow_manager
        self.arrival_rate = arrival_rate
        self.bw_range = bw_range
        self.delay_range = delay_range
        self.duration_range = duration_range
        self.rng = rng or np.random.default_rng()

        # Satellite id range (1-based)
        self._sat_ids = list(topology.adj.keys())

    def generate(self, timeslot: int) -> list[FlowRequest]:
        """Sample new flow requests for the given *timeslot*.

        Returns a (possibly empty) list of FlowRequest objects.
        """
        n_flows = self.rng.poisson(self.arrival_rate)
        requests: list[FlowRequest] = []

        for _ in range(n_flows):
            src, dst = self.rng.choice(self._sat_ids, size=2, replace=False)
            bw = self.rng.uniform(*self.bw_range)
            delay = self.rng.uniform(*self.delay_range)
            dur = int(self.rng.integers(self.duration_range[0],
                                        self.duration_range[1] + 1))
            fid = self.flow_manager.next_flow_id()
            requests.append(FlowRequest(
                flow_id=fid,
                src_sat=int(src),
                dst_sat=int(dst),
                bandwidth=float(bw),
                max_delay=float(delay),
                duration=dur,
                arrival_timeslot=timeslot,
            ))
        return requests
