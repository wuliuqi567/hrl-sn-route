"""
Flow — 流量流请求与生命周期管理。

数据结构 / Data Structures:
  - FlowRequest:  不可变的流需求描述符 (frozen dataclass)
    包含: 源/目的卫星, 带宽需求, 时延预算, 持续时间, 到达时隙
  - ActiveFlow:   当前占用带宽的活跃流
    包含: 原始请求, 实际路径, 剩余持续时间, 实际时延
  - FlowManager:  活跃流生命周期管理器
    负责 注册/到期释放/ID分配

生命周期 / Lifecycle:
  FlowRequest 生成 → 路由成功 → register() 成为 ActiveFlow
  → 每时隙 tick() 递减 remaining_duration
  → 到期后 release 带宽并删除
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.bandwidth import BandwidthManager


@dataclass(frozen=True)
class FlowRequest:
    """Immutable descriptor of a traffic flow request."""
    flow_id: int
    src_sat: int           # source access satellite
    dst_sat: int           # destination access satellite
    bandwidth: float       # demanded bandwidth in Mbps
    max_delay: float       # QoS delay budget in seconds
    duration: int          # number of timeslots the flow lasts
    arrival_timeslot: int  # timeslot at which the flow arrives


@dataclass
class ActiveFlow:
    """A flow currently being routed through the network."""
    request: FlowRequest
    path: list[int]
    remaining_duration: int
    actual_delay: float    # end-to-end delay achieved (seconds)


class FlowManager:
    """Manages the lifecycle of active flows.

    Flows are registered when successfully routed and released (bandwidth
    freed) when their remaining duration reaches 0.

    Parameters
    ----------
    bw_manager : BandwidthManager
        Used to release bandwidth when flows expire.
    """

    def __init__(self, bw_manager: "BandwidthManager"):
        self.bw_manager = bw_manager
        self.active_flows: dict[int, ActiveFlow] = {}
        self._next_flow_id: int = 1
        self._all_delays: list[float] = []  # delays from ALL routed flows (including expired)

    def next_flow_id(self) -> int:
        """Return and increment the global flow-id counter."""
        fid = self._next_flow_id
        self._next_flow_id += 1
        return fid

    def register(self, request: FlowRequest, path: list[int],
                 actual_delay: float) -> None:
        """Register a successfully-routed flow as active."""
        self.active_flows[request.flow_id] = ActiveFlow(
            request=request,
            path=path,
            remaining_duration=request.duration,
            actual_delay=actual_delay,
        )
        self._all_delays.append(actual_delay)

    def tick(self) -> list[int]:
        """Advance one timeslot.  Returns list of expired flow_ids."""
        expired: list[int] = []
        for fid, af in list(self.active_flows.items()):
            af.remaining_duration -= 1
            if af.remaining_duration <= 0:
                expired.append(fid)
        for fid in expired:
            self.bw_manager.release(fid)
            del self.active_flows[fid]
        return expired

    def get_active_count(self) -> int:
        return len(self.active_flows)

    def get_all_delays(self) -> list[float]:
        """Return delays from all flows routed during this episode."""
        return list(self._all_delays)

    def reset(self) -> None:
        """Clear all active flows (start of new episode)."""
        self.active_flows.clear()
        self._next_flow_id = 1
        self._all_delays.clear()
