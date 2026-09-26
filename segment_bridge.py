from __future__ import annotations

import dataclasses
from typing import Any

import torch


@dataclasses.dataclass
class SegmentSummary:
    """单段采样后的摘要，用于段间桥接。"""
    mode: str = "text"
    canvas: tuple[int, int] = (1344, 768)
    frame_count: int = 124
    render_frames: int = 124
    tail_overlap_frames: int = 0
    tail_frame_pixel: torch.Tensor | None = None  # [1,H,W,C]
    tail_audio_latent: torch.Tensor | None = None
    identity_refs: list[dict] = dataclasses.field(default_factory=list)
    style_refs: list[dict] = dataclasses.field(default_factory=list)
    audio_policy: str = "generate_new"
    prompt: str = ""


@dataclasses.dataclass
class TransitionPlan:
    """段间过渡方案。

    v2.7.0: 删除 audio_continuity / suggested_first_frame 死字段。
    """
    keyframe_shift: int = 0
    auto_inherit_tail: bool = False
    inherit_refs: bool = False


@dataclasses.dataclass
class KeyframeSpec:
    """关键帧规格（相对坐标）。"""
    role: str = "intermediate"  # first_frame | last_frame | intermediate
    relative_index: int | float = 0  # 0=首, -1=尾, 0~1=比例
    latent: Any = None
    latent_h: int = 0
    latent_w: int = 0
    latent_t: int = 0
    label: str = ""