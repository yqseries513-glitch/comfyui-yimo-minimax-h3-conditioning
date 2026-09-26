from __future__ import annotations

from .segment_bridge import SegmentSummary, TransitionPlan


def resolve_transition(
    prev_summary: SegmentSummary | None,
    curr_mode: str,
    overlap_frames: int,
    video_continuity: str = "context_inject",
    audio_continuity: str = "context_inject",
    auto_inherit_tail: bool = True,
    inherit_identity: bool = True,
) -> TransitionPlan:
    """v2.7.0: 段间过渡完全由用户参数控制，不再使用 transition table。

    视频连续性（video_continuity）:
        context_inject - 注入前段视频 tail latent，拼接删除重叠帧
        hard_concat    - 不注入，各段视频独立

    音频连续性（audio_continuity）:
        由调用方直接控制（sequence_sample / single_sampler），
        不写入 TransitionPlan。

    关键帧移位（keyframe_shift）:
        仅当 video_continuity=context_inject 且 overlap_frames>0 时应用。
        补偿拼接时删除的前 overlap_frames 帧。
        first_frame / intermediate 移位；last_frame 不移位。

    尾帧继承（auto_inherit_tail）:
        仅当 video_continuity=hard_concat 或 overlap_frames<=0 时生效。

    参考继承（inherit_refs）:
        仅对 references / hybrid 模式生效，其余模式跳过。
    """
    plan = TransitionPlan()

    if prev_summary is None:
        plan.keyframe_shift = 0
        plan.auto_inherit_tail = False
        plan.inherit_refs = False
        return plan

    # 关键帧移位：仅视频注入且有 overlap
    if video_continuity == "context_inject" and overlap_frames > 0:
        plan.keyframe_shift = int(overlap_frames)
    else:
        plan.keyframe_shift = 0

    # 尾帧继承：仅硬拼接或 overlap=0
    if (video_continuity == "hard_concat" or overlap_frames <= 0) and auto_inherit_tail:
        plan.auto_inherit_tail = True
    else:
        plan.auto_inherit_tail = False

    if plan.auto_inherit_tail and prev_summary.tail_frame_pixel is None:
        plan.auto_inherit_tail = False

    # 参考继承：仅 references / hybrid 支持
    support_refs = curr_mode in {"references", "hybrid"}
    plan.inherit_refs = bool(inherit_identity and support_refs)
    if plan.inherit_refs and not (prev_summary.identity_refs or prev_summary.style_refs):
        plan.inherit_refs = False

    return plan