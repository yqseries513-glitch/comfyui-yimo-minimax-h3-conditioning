# -*- coding: utf-8 -*-
"""Yimo H3 AV双时钟序列采样器

节点 ID: YimoH3SequenceSampler
显示名: Yimo H3 AV双时钟序列采样器
分类:   Yimo/MiniMax H3/音频
"""

from __future__ import annotations

import sys
import traceback
import logging
import os
import time
from datetime import datetime
from typing import Callable, Sequence

import torch

import comfy.model_management
import comfy.nested_tensor
import comfy.samplers
import comfy.utils
from comfy_api.latest import io

from .core import (
    sorted_autogrow_items,
    nested_av_parts,
    is_connected_value,
    get_sequence_sampler_export_dir,
    get_yimo_output_dir,
    sanitize_filename,
    get_video_frame_count_from_latent,
)
from .face_restore import build_identity_bank, restore_sampled_av
from .i18n import t
from .sampler_core import (
    sequence_sample,
    _isolate_payload_cond,
    _extract_tail_context,
    _inject_context,
    _apply_transition_to_conditioning,
    _extract_segment_summary,
    _overlap_to_latent,
    _resolve_pair_overlaps,
    _concatenate_av_latents,
    _serialize_sampled,
    _deserialize_sampled,
    _summary_to_dict,
    _dict_to_summary,
    _soft_empty_cache,
)
from .transition_policies import resolve_transition
from .color_match_utils import apply_color_match
from .segment_bridge import SegmentSummary
from .dual_clock_core import (
    DEFAULT_SAMPLER_NAME as DUAL_CLOCK_SAMPLER_NAME,
    DEFAULT_SCHEDULER_NAME as DUAL_CLOCK_SCHEDULER_NAME,
    DEFAULT_SHIFT_AUDIO,
    DEFAULT_SHIFT_VIDEO,
    SCHEDULER_OPTIONS as DUAL_CLOCK_SCHEDULER_OPTIONS,
    setup_dual_clock_sampling,
    run_dual_clock_sample,
)

logger = logging.getLogger("YimoH3")

CATEGORY = t("category")

_CHUNK_MEMORY_THRESHOLD_DEFAULT_MB = 4096
_DC_CACHE_SUBDIR = "sequence_sampler/_segment_cache_dc"


# =============================================================================
# 双时钟缓存路径
# =============================================================================

def _dc_cache_dir(chain_id: str) -> str:
    safe_id = sanitize_filename(chain_id) if chain_id else "default"
    path = os.path.join(get_yimo_output_dir(_DC_CACHE_SUBDIR), safe_id)
    os.makedirs(path, exist_ok=True)
    return path


def _dc_cache_path(chain_id: str, segment_index: int) -> str:
    return os.path.join(_dc_cache_dir(chain_id), f"seg_{segment_index:04d}.pt")


def _dc_save_segment_cache(chain_id, segment_index, sampled, summary) -> str:
    if not chain_id:
        return ""
    path = _dc_cache_path(chain_id, segment_index)
    try:
        torch.save({
            "sampled_payload": _serialize_sampled(sampled),
            "summary": _summary_to_dict(summary),
            "segment_index": segment_index,
            "cache_format": "v2",
            "node": "YimoH3SequenceSampler",
        }, path)
        logger.debug("Saved DC segment cache: %s", path)
        return path
    except Exception as e:
        logger.warning("Failed to save DC segment cache to %s: %s", path, e)
        return ""


def _dc_load_segment_cache(chain_id, segment_index, device):
    path = _dc_cache_path(chain_id, segment_index)
    if not os.path.exists(path):
        return None
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        logger.warning(
            "Failed to load DC segment cache from %s: %s. "
            "Refusing unsafe fallback; treat as cache miss.", path, e,
        )
        return None
    try:
        if data.get("cache_format") == "v2" and "sampled_payload" in data:
            sampled = _deserialize_sampled(data["sampled_payload"], device)
        else:
            logger.warning("DC segment cache %s missing sampled_payload; cache miss.", path)
            return None
        if isinstance(data.get("summary"), dict):
            summary = _dict_to_summary(data["summary"])
        else:
            summary = data.get("summary")
        return {
            "sampled": sampled,
            "summary": summary,
            "segment_index": data.get("segment_index", segment_index),
        }
    except Exception as e:
        logger.error("Failed to decode DC segment cache from %s: %s", path, e)
        return None


# =============================================================================
# 双时钟序列采样核心
# =============================================================================

def dual_clock_sequence_sample(
    model,
    segments: list[dict],
    sampler_name: str = DUAL_CLOCK_SAMPLER_NAME,
    scheduler: str = DUAL_CLOCK_SCHEDULER_NAME,
    steps: int = 20,
    shift_video: float = DEFAULT_SHIFT_VIDEO,
    shift_audio: float = DEFAULT_SHIFT_AUDIO,
    cfg: float = 1.0,
    denoise: float = 1.0,
    seed: int = 0,
    overlap_frames: int | Sequence[int] = 22,
    transition_mode: str = "context_inject",
    audio_continuity: str = "context_inject",
    callback: Callable | None = None,
    video_vae=None,
    auto_inherit_tail: bool = True,
    inherit_identity: bool = True,
    resample_segment: int = -1,
    segment_seeds: list[int | None] | None = None,
    chain_id: str = "",
    stop_after_segment: int = -1,
    face_restore: bool = False,
    face_bank=None,
    face_params: dict | None = None,
    previewer=None,
    enable_ref_curve: bool = False,
    color_match_mode: str = "off",
    color_match_strength: float = 0.7,
    color_match_reference_frames: int = 1,
) -> tuple[dict, list[str], list[dict], list[SegmentSummary]]:
    if not segments:
        raise ValueError("At least one segment is required")

    if sampler_name != DUAL_CLOCK_SAMPLER_NAME:
        raise ValueError(
            f"dual_clock_sequence_sample 只支持 "
            f"sampler_name={DUAL_CLOCK_SAMPLER_NAME!r}，"
            f"当前 sampler_name={sampler_name!r}。"
        )

    total_segments = len(segments)

    n_pairs = max(0, total_segments - 1)
    pair_overlaps = _resolve_pair_overlaps(overlap_frames, n_pairs)

    pair_effective = []
    for olap in pair_overlaps:
        if olap <= 0:
            pair_effective.append(0)
        else:
            eff, _, _ = _overlap_to_latent(olap)
            pair_effective.append(eff)

    for olap in pair_overlaps:
        if olap < 0:
            raise ValueError(f"overlap_frames cannot be negative: {olap}")

    effective_total_segments = total_segments
    if stop_after_segment >= 0:
        effective_total_segments = min(stop_after_segment + 1, total_segments)

    total_steps = effective_total_segments * steps

    total_video_frames = 0
    for seg in segments:
        total_video_frames += seg.get("render_frames", seg.get("frame_count", 124))
    for eff in pair_effective:
        if eff > 0:
            total_video_frames -= eff
    total_video_frames = max(1, total_video_frames)

    report_lines = [
        "=== Yimo H3 AV双时钟序列采样器 v5.0.0 ===",
        f"total_segments={total_segments}",
        f"effective_segments={effective_total_segments}",
        f"pair_overlaps(requested)={pair_overlaps}",
        f"pair_overlaps(effective 17n+5)={pair_effective}",
        f"total_video_frames={total_video_frames}",
        f"sampler_name={sampler_name}, scheduler={scheduler}",
        f"steps={steps}, cfg={cfg}, denoise={denoise}, base_seed={seed}",
        f"shift_video={shift_video}, shift_audio={shift_audio} (生效)",
        f"transition_mode(video)={transition_mode}",
        f"audio_continuity={audio_continuity}",
        f"auto_inherit_tail={auto_inherit_tail}",
        f"inherit_identity={inherit_identity}",
        f"resample_segment={resample_segment}",
        f"stop_after_segment={stop_after_segment}",
        f"face_restore={face_restore}",
        f"enable_ref_curve={enable_ref_curve}",
        f"color_match_mode={color_match_mode}, strength={color_match_strength}",
        f"chain_id={chain_id}",
        f"cache_dir={_DC_CACHE_SUBDIR}/{sanitize_filename(chain_id) if chain_id else 'default'}",
    ]

    device = comfy.model_management.intermediate_device()

    sampled_segments: list[dict] = []
    segment_summaries: list[SegmentSummary] = []
    prev_context = None
    prev_summary = None
    start_idx = 0
    cumulative_frame_offset = 0

    if resample_segment >= 0 and chain_id:
        if resample_segment >= total_segments:
            logger.warning(
                "resample_segment %d >= total_segments %d, ignoring",
                resample_segment, total_segments,
            )
            resample_segment = -1
        else:
            cache_loaded = True
            for ci in range(resample_segment):
                data = _dc_load_segment_cache(chain_id, ci, device)
                if data is None:
                    cache_loaded = False
                    logger.warning(
                        "DC segment cache missing for index %d, "
                        "falling back to full resample", ci,
                    )
                    break
                sampled_segments.append(data["sampled"])
                segment_summaries.append(data["summary"])
                cumulative_frame_offset += segments[ci].get(
                    "render_frames", segments[ci].get("frame_count", 124)
                )
                if ci < len(pair_effective):
                    cumulative_frame_offset -= pair_effective[ci]

            if cache_loaded and resample_segment > 0:
                prev_summary = segment_summaries[-1]
                if resample_segment < total_segments:
                    next_olap = (
                        pair_overlaps[resample_segment - 1]
                        if resample_segment - 1 < len(pair_overlaps) else 0
                    )
                    need_context = (
                        next_olap > 0
                        and (
                            transition_mode == "context_inject"
                            or audio_continuity == "context_inject"
                        )
                    )
                    if need_context:
                        prev_context = _extract_tail_context(
                            sampled_segments[-1], next_olap
                        )
                start_idx = resample_segment
                report_lines.append(
                    f"Loaded {resample_segment} cached DC segments from chain "
                    f"'{chain_id}', resampling from segment {resample_segment}"
                )
            elif resample_segment == 0:
                start_idx = 0
                report_lines.append("Resampling from segment 0 (full resample)")

    loop_end = total_segments
    if stop_after_segment >= 0:
        loop_end = min(stop_after_segment + 1, total_segments)

    for i in range(start_idx, loop_end):
        seg = segments[i]
        report_lines.append(f"--- segment {i}/{total_segments} ---")

        positive = seg["positive"]
        negative = seg["negative"]
        latent = seg["latent"]
        frame_count = seg.get("frame_count", 124)
        render_frames = seg.get("render_frames", frame_count)
        mode = seg.get("mode", "text")
        kf_specs = seg.get("keyframe_specs", [])
        identity_refs = seg.get("identity_refs", [])
        style_refs = seg.get("style_refs", [])

        current_overlap = pair_overlaps[i - 1] if i > 0 else 0
        current_effective_overlap = pair_effective[i - 1] if i > 0 else 0

        plan = resolve_transition(
            prev_summary=prev_summary,
            curr_mode=mode,
            overlap_frames=current_effective_overlap,
            video_continuity=transition_mode,
            audio_continuity=audio_continuity,
            auto_inherit_tail=auto_inherit_tail,
            inherit_identity=inherit_identity,
        )

        prev_identity_refs = prev_summary.identity_refs if prev_summary else []
        prev_style_refs = prev_summary.style_refs if prev_summary else []

        positive, trans_report = _apply_transition_to_conditioning(
            positive, kf_specs, frame_count,
            plan.keyframe_shift, plan.auto_inherit_tail,
            prev_summary, video_vae, plan.inherit_refs,
            prev_identity_refs, prev_style_refs,
            curr_mode=mode,
        )
        report_lines.append(f"transition: {trans_report}")

        latent, ctx_report = _inject_context(
            latent, prev_context,
            video_continuity=transition_mode,
            audio_continuity=audio_continuity,
        )
        report_lines.append(ctx_report)

        seg_callback = None
        if callback is not None or previewer is not None:
            def _make_seg_callback(seg_idx, base_idx):
                _step_start = [time.time()]
                def _cb(step, x0, x, seg_total_steps):
                    now = time.time()
                    elapsed = now - _step_start[0]
                    _step_start[0] = now
                    global_step = (seg_idx - base_idx) * steps + step + 1
                    global_step = min(global_step, total_steps)
                    logger.info(
                        "[YimoH3DualClockSeq] segment %d step %d/%d "
                        "(global %d/%d, %.2fs/step)",
                        seg_idx, step + 1, seg_total_steps,
                        global_step, total_steps, elapsed,
                    )
                    preview = None
                    if previewer is not None:
                        try:
                            if hasattr(x0, "is_nested") and x0.is_nested:
                                parts = x0.unbind()
                                preview = previewer.decode_latent_to_preview_image(parts[0])
                            else:
                                preview = previewer.decode_latent_to_preview_image(x0)
                        except Exception:
                            pass
                    if callback is not None:
                        callback(global_step, x0, x, total_steps)
                    return preview
                return _cb
            seg_callback = _make_seg_callback(i, start_idx)

        if segment_seeds is not None and i < len(segment_seeds) and segment_seeds[i] is not None:
            seg_seed = int(segment_seeds[i])
            report_lines.append(f"segment {i}: using explicit seed {seg_seed}")
        else:
            seg_seed = seed + i * 12345

        if enable_ref_curve:
            seg_offset_arg = cumulative_frame_offset
            total_frames_arg = total_video_frames
        else:
            seg_offset_arg = 0
            total_frames_arg = None

        # ====================================================================
        # 双时钟 setup（每段独立）
        # ====================================================================
        try:
            patched_model, sampler_obj, dc_sigmas, diag, original_sampling = (
                setup_dual_clock_sampling(
                    model=model,
                    segment_latent=latent,
                    steps=steps,
                    shift_video=shift_video,
                    shift_audio=shift_audio,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                )
            )
        except Exception as e:
            logger.error("[DualClockSeq] segment %d setup failed: %s", i, e)
            report_lines.append(
                f"segment {i} dual clock setup failed: {type(e).__name__}: {e}"
            )
            raise

        report_lines.append("--- dual clock diagnostics ---")
        report_lines.append(
            f"use_native_av={diag['use_native_av']}, "
            f"audio_velocity_is_raw={diag['audio_velocity_is_raw']}"
        )
        report_lines.append(f"protocol={diag['protocol']['protocol_hint']}")
        report_lines.append(
            f"sigmas_length={diag['sigmas_length']}, "
            f"sigma_first={diag['sigma_first']:.4f}, "
            f"sigma_last={diag['sigma_last']:.4f}"
        )

        # ====================================================================
        # 采样
        # ====================================================================
        noise_mask = latent.get("noise_mask")

        try:
            sampled_samples = run_dual_clock_sample(
                model=patched_model,
                sampler=sampler_obj,
                sigmas=dc_sigmas,
                positive=positive,
                negative=negative,
                latent=latent,
                cfg=cfg,
                seed=seg_seed,
                denoise=denoise,
                noise_mask=noise_mask,
                callback=seg_callback,
                disable_pbar=False,
                device=device,
                original_sampling=original_sampling,
            )
        except Exception as e:
            logger.error(
                "[DualClockSeq] segment %d standard sampling failed: %s", i, e
            )
            report_lines.append(
                f"segment {i} dual clock sampling failed: "
                f"{type(e).__name__}: {e}"
            )
            raise

        sampled = {"samples": sampled_samples}
        if noise_mask is not None:
            sampled["noise_mask"] = noise_mask
        report_lines.append("dual clock standard sampling complete")

        if face_restore and video_vae is not None:
            try:
                sampled, frep = restore_sampled_av(
                    sampled, video_vae, face_bank,
                    **(face_params or {})
                )
                report_lines.append(f"segment {i} face_restore: {frep}")
            except Exception as e:
                logger.warning("Face restore failed for segment %d: %s", i, e)
                report_lines.append(f"segment {i} face_restore failed: {e}")

        color_match_info = {
            "mode": color_match_mode,
            "strength": float(color_match_strength),
            "reference_frames": int(color_match_reference_frames),
            "status": "skipped (first segment)",
            "reference_source": "none",
            "matched_frames": 0,
            "match_level": "none",
        }
        if i > 0 and len(sampled_segments) > 0 and color_match_mode != "off":
            reference = sampled_segments[-1]
            sampled, color_match_info = apply_color_match(
                sampled=sampled,
                reference=reference,
                mode=color_match_mode,
                strength=color_match_strength,
                reference_frames=color_match_reference_frames,
                video_vae=video_vae,
            )
            color_match_info["reference_source"] = f"segment_{i - 1} tail"
        elif color_match_mode == "off":
            color_match_info["status"] = "disabled"

        report_lines.append("--- color match ---")
        report_lines.append(f"mode={color_match_info['mode']}")
        report_lines.append(f"strength={color_match_info['strength']:.2f}")
        report_lines.append(f"reference_frames={color_match_info['reference_frames']}")
        report_lines.append(f"status={color_match_info['status']}")
        report_lines.append(f"match_level={color_match_info['match_level']}")
        report_lines.append(f"matched_frames={color_match_info['matched_frames']}")

        # 统一 video/audio device，避免下游 pack_latents 因 device 分叉报错。
        try:
            _v, _a = nested_av_parts(sampled)
            if _v.device != _a.device:
                sampled = dict(sampled)
                sampled["samples"] = comfy.nested_tensor.NestedTensor(
                    (_v, _a.to(device=_v.device, dtype=_v.dtype))
                )
        except Exception:
            pass

        sampled_segments.append(sampled)

        summary = _extract_segment_summary(
            sampled, video_vae, mode, frame_count, render_frames, current_overlap
        )
        if identity_refs:
            summary.identity_refs = identity_refs
        if style_refs:
            summary.style_refs = style_refs
        summary.audio_policy = seg.get("audio_policy", "generate_new")
        segment_summaries.append(summary)
        prev_summary = summary

        cache_path = ""
        if chain_id:
            cache_path = _dc_save_segment_cache(chain_id, i, sampled, summary)
        if cache_path:
            report_lines.append(f"segment {i}: SAMPLING COMPLETE -> cache={cache_path}")
        else:
            report_lines.append(f"segment {i}: SAMPLING COMPLETE")

        cumulative_frame_offset += render_frames
        if i < len(pair_effective):
            cumulative_frame_offset -= pair_effective[i]

        if i < len(pair_overlaps):
            next_olap = pair_overlaps[i]
            need_context = (
                next_olap > 0
                and (
                    transition_mode == "context_inject"
                    or audio_continuity == "context_inject"
                )
            )
            prev_context = _extract_tail_context(sampled, next_olap) if need_context else None
        else:
            prev_context = None

        _soft_empty_cache()

    if stop_after_segment >= 0 and loop_end < total_segments:
        report_lines.append(
            f"--- EARLY STOP after segment {stop_after_segment} ---"
            f" (skipped segments {loop_end}~{total_segments - 1})"
        )

    n_conc = max(0, len(sampled_segments) - 1)
    video_overlaps_for_concat: list[int] = []
    audio_overlaps_for_concat: list[int] = []
    for j in range(n_conc):
        eff = pair_effective[j] if j < len(pair_effective) else 0
        video_overlaps_for_concat.append(eff if transition_mode == "context_inject" else 0)
        audio_overlaps_for_concat.append(eff if audio_continuity == "context_inject" else 0)

    final_latent = _concatenate_av_latents(
        sampled_segments,
        video_overlaps_for_concat,
        audio_overlaps_for_concat,
    )

    final_video, final_audio = nested_av_parts(final_latent)
    total_latent_t = final_video.shape[2]
    total_frames = get_video_frame_count_from_latent(final_video)

    report_lines.append(
        f"=== final output: latent_t={total_latent_t}, "
        f"estimated_frames={total_frames} ==="
    )

    return final_latent, report_lines, sampled_segments, segment_summaries


# =============================================================================
# 节点
# =============================================================================

def _build_sampler_options() -> list[str]:
    options = [DUAL_CLOCK_SAMPLER_NAME]
    native = list(getattr(comfy.samplers, "SAMPLER_NAMES", []) or [])
    for name in native:
        if name not in options:
            options.append(name)
    return options


def _build_scheduler_options() -> list[str]:
    options = list(DUAL_CLOCK_SCHEDULER_OPTIONS)
    native = list(getattr(comfy.samplers, "SCHEDULER_NAMES", []) or [])
    for name in native:
        if name not in options:
            options.append(name)
    return options


try:
    class YimoH3SequenceSampler(io.ComfyNode):

        @classmethod
        def define_schema(cls):
            sampler_options = _build_sampler_options()
            scheduler_options = _build_scheduler_options()

            return io.Schema(
                node_id="YimoH3SequenceSampler",
                display_name=t("seqsampler_display_name"),
                description=(
                    "MiniMax H3 音视频序列采样器（含双时钟）。\n"
                    "sampler_name=dual_clock_euler → 双时钟（video/audio 独立 sigma 轨迹，\n"
                    "默认 shift_video=12, shift_audio=3）。\n"
                    "sampler_name=其它 → 自动降级到单时钟（ComfyUI 原生 AV 调度）。\n"
                    "段间逻辑（上下文注入 / 尾帧继承 / 身份继承 / 面部修复 / 颜色匹配 / "
                    "断点续采 / 参考曲线 / images 模式 / 自动导出）全部保留。"
                ),
                category=CATEGORY,
                inputs=[
                    io.Model.Input("model", tooltip="MiniMax H3 DiT model."),
                    io.Vae.Input("video_vae", optional=True, tooltip=t("video_vae_tooltip")),

                    io.Autogrow.Input("segments", optional=False, template=io.Autogrow.TemplatePrefix(
                        input=io.Conditioning.Input("segment", optional=True,
                            tooltip="Segment bundle from YimoH3Conditioning.segment"),
                        prefix="segment_", min=1, max=8),
                        tooltip="1~8 个段数据包。每个来自一个 YimoH3Conditioning 的「段数据包」输出。"),

                    io.Combo.Input(
                        "sampler_name",
                        options=sampler_options,
                        default=DUAL_CLOCK_SAMPLER_NAME,
                        tooltip=(
                            "dual_clock_euler = 双时钟 Euler（推荐）。\n"
                            "其它 ComfyUI 原生采样器（euler / dpmpp / ...）= 单时钟，\n"
                            "自动降级到 ComfyUI 原生 AV 调度。\n"
                            "选原生采样器时，shift_video / shift_audio 无效。"
                        ),
                    ),
                    io.Combo.Input(
                        "scheduler",
                        options=scheduler_options,
                        default=DUAL_CLOCK_SCHEDULER_NAME,
                        tooltip=(
                            "native_flow = 双时钟专用 shift 网格（推荐）。\n"
                            "beta57 = 双时钟 beta 调度器。\n"
                            "其它 = ComfyUI 内置调度器。"
                        ),
                    ),
                    io.Int.Input(
                        "steps", default=20, min=1, max=100, step=1,
                        tooltip="采样步数。无加速 LoRA 时推荐 20~30。",
                    ),
                    io.Float.Input(
                        "shift_video", default=DEFAULT_SHIFT_VIDEO,
                        min=0.01, max=100.0, step=0.01,
                        tooltip="视频 sigma shift。默认 12.0。\n"
                                "仅 sampler_name=dual_clock_euler 时生效。",
                    ),
                    io.Float.Input(
                        "shift_audio", default=DEFAULT_SHIFT_AUDIO,
                        min=0.01, max=100.0, step=0.01,
                        tooltip="音频 sigma shift。默认 3.0。\n"
                                "仅 sampler_name=dual_clock_euler 时生效。",
                    ),
                    io.Float.Input(
                        "cfg", default=1.0, min=0.0, max=100.0, step=0.1,
                        tooltip="CFG 引导强度。加速 LoRA 下 1.0；无加速 LoRA 时 4~7。",
                    ),
                    io.Float.Input(
                        "denoise", default=1.0, min=0.0, max=1.0, step=0.01,
                        tooltip="降噪强度。1.0=完全重绘。",
                    ),
                    io.Int.Input(
                        "seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, step=1,
                        tooltip="基础种子。每段派生 seed + i * 12345；"
                                "如已设 segment_seeds 则按逐段覆盖。",
                    ),

                    io.Int.Input(
                        "overlap_frames", default=22, min=0, max=200, step=1,
                        tooltip="默认段间重叠帧数。仅在 overlap_list 留空或全部为负数时生效。",
                    ),
                    io.String.Input(
                        "overlap_list", default="", advanced=True,
                        tooltip=t("overlap_list_tooltip")
                                + "\n注意：填负数会被视为「未设置」，回退到 overlap_frames；0 仍是硬切标记。",
                    ),
                    io.Combo.Input(
                        "transition_mode",
                        options=["context_inject", "hard_concat"],
                        default="context_inject",
                        tooltip="视频连续性：context_inject 注入前段 tail；hard_concat 硬切。",
                    ),
                    io.Combo.Input(
                        "audio_continuity",
                        options=["context_inject", "break"],
                        default="context_inject",
                        advanced=False,
                        tooltip="音频连续性：context_inject 注入前段音频 tail；break 独立。",
                    ),

                    io.Int.Input(
                        "resample_segment", default=-1, min=-1, max=7, step=1, advanced=True,
                        tooltip="指定从第几段开始重新采样（0-based）。-1=全部重新采样。",
                    ),
                    io.String.Input(
                        "segment_seeds", default="", advanced=True,
                        tooltip="逗号分隔的每段独立种子，如：0,123,456,none,789。",
                    ),
                    io.String.Input(
                        "chain_id", default="default", advanced=True,
                        tooltip="缓存标识符。双时钟缓存目录独立（_segment_cache_dc）。",
                    ),
                    io.Int.Input(
                        "stop_after_segment", default=-1, min=-1, max=7, step=1, advanced=True,
                        tooltip="在采样完指定段后提前停止。-1=不提前停止。",
                    ),

                    io.Boolean.Input(
                        "auto_inherit_tail", default=True, advanced=True,
                        tooltip=t("auto_inherit_tail_tooltip"),
                    ),
                    io.Boolean.Input(
                        "inherit_identity", default=True, advanced=True,
                        tooltip=t("inherit_identity_tooltip"),
                    ),

                    io.Boolean.Input(
                        "enable_ref_curve", default=False, advanced=True,
                        tooltip="启用参考强度时间曲线。",
                    ),
                    io.Boolean.Input(
                        "override_ref_curve", default=False, advanced=True,
                        tooltip="覆盖所有分段的参考曲线配置。",
                    ),
                    io.Combo.Input(
                        "ref_curve_direction_override",
                        options=["inherit", "constant", "concept_at_start",
                                 "concept_at_end", "concept_at_middle", "concept_at_ends"],
                        default="inherit", advanced=True,
                        tooltip="统一覆盖曲线方向。inherit=使用各分段自身配置。",
                    ),
                    io.Combo.Input(
                        "ref_curve_shape_override",
                        options=["inherit", "linear", "ease", "sigmoid",
                                 "exponential", "quadratic", "cubic"],
                        default="inherit", advanced=True,
                        tooltip="统一覆盖曲线形状。inherit=使用各分段自身配置。",
                    ),

                    io.Image.Input("face_images", optional=True,
                        tooltip="面部参考图。支持单图/多图/批次。"),
                    io.Boolean.Input("face_restore", default=False, advanced=True,
                        tooltip="启用采样中面部修复。"),
                    io.Combo.Input("face_restore_backend",
                        options=["auto", "gfpgan", "insightface", "opencv"],
                        default="auto", advanced=True,
                        tooltip="面部修复后端。"),
                    io.Float.Input("face_restore_strength", default=0.7, min=0.0, max=1.0, step=0.05,
                        advanced=True, tooltip="面部修复强度。"),
                    io.Float.Input("face_similarity_threshold", default=0.45, min=0.0, max=1.0, step=0.01,
                        advanced=True, tooltip="ArcFace 余弦匹配阈值。"),

                    io.Combo.Input("color_match_mode",
                        options=["off", "low", "medium", "high", "max"],
                        default="off", advanced=True,
                        tooltip=t("color_match_mode_tooltip")),
                    io.Float.Input("color_match_strength",
                        default=0.7, min=0.0, max=1.0, step=0.05, advanced=True,
                        tooltip=t("color_match_strength_tooltip")),
                    io.Int.Input("color_match_reference_frames",
                        default=1, min=1, max=17, step=1, advanced=True,
                        tooltip=t("color_match_reference_frames_tooltip")),

                    io.Combo.Input("output_format",
                        options=["latent", "images"],
                        default="latent", advanced=True,
                        tooltip="latent=AV Latent；images=IMAGE 序列（需 video_vae）。"),
                    io.Int.Input("chunk_memory_threshold_mb",
                        default=_CHUNK_MEMORY_THRESHOLD_DEFAULT_MB,
                        min=512, max=32768, step=256, advanced=True,
                        tooltip="images 模式解码 chunk 的内存阈值（MB）。"),
                    io.Boolean.Input("auto_export_final", default=False, advanced=True,
                        tooltip="是否自动导出最终拼接结果。"),
                    io.String.Input("export_name", default="", advanced=True,
                        tooltip="导出文件名（不含扩展名）。"),
                ],
                outputs=[
                    io.Latent.Output(display_name=t("seqsampler_output_latent")),
                    io.String.Output(display_name=t("seqsampler_output_report")),
                    io.Latent.Output(display_name="分段0AV潜变量"),
                    io.Latent.Output(display_name="分段1AV潜变量"),
                    io.Latent.Output(display_name="分段2AV潜变量"),
                    io.Latent.Output(display_name="分段3AV潜变量"),
                    io.Latent.Output(display_name="分段4AV潜变量"),
                    io.Latent.Output(display_name="分段5AV潜变量"),
                    io.Latent.Output(display_name="分段6AV潜变量"),
                    io.Latent.Output(display_name="分段7AV潜变量"),
                    io.Image.Output(display_name="长视频图像序列 (images 模式)"),
                ],
            )

        @classmethod
        def execute(
            cls, model, video_vae=None,
            segments=None,
            sampler_name=DUAL_CLOCK_SAMPLER_NAME,
            scheduler=DUAL_CLOCK_SCHEDULER_NAME,
            steps=20,
            shift_video=DEFAULT_SHIFT_VIDEO,
            shift_audio=DEFAULT_SHIFT_AUDIO,
            cfg=1.0,
            denoise=1.0,
            seed=0,
            overlap_frames=22,
            overlap_list="",
            transition_mode="context_inject",
            audio_continuity="context_inject",
            resample_segment=-1,
            segment_seeds="",
            chain_id="default",
            stop_after_segment=-1,
            auto_inherit_tail=True,
            inherit_identity=True,
            enable_ref_curve=False,
            override_ref_curve=False,
            ref_curve_direction_override="inherit",
            ref_curve_shape_override="inherit",
            face_images=None,
            face_restore=False,
            face_restore_backend="auto",
            face_restore_strength=0.7,
            face_similarity_threshold=0.45,
            color_match_mode="off",
            color_match_strength=0.7,
            color_match_reference_frames=1,
            output_format="latent",
            chunk_memory_threshold_mb=_CHUNK_MEMORY_THRESHOLD_DEFAULT_MB,
            auto_export_final=False,
            export_name="",
            **kwargs,
        ):
            if model is None:
                raise ValueError("model 未连接")

            use_dual_clock = (sampler_name == DUAL_CLOCK_SAMPLER_NAME)

            native_samplers = set(getattr(comfy.samplers, "SAMPLER_NAMES", []) or [])
            if use_dual_clock:
                native_sampler_name = None
            elif sampler_name in native_samplers:
                native_sampler_name = sampler_name
            else:
                raise ValueError(
                    f"未知的 sampler_name: {sampler_name!r}。\n"
                    f"合法值：{DUAL_CLOCK_SAMPLER_NAME} 或 ComfyUI 原生采样器。"
                )

            # ------------------------------------------------------------------
            # 解析 segments
            # ------------------------------------------------------------------
            segment_items = sorted_autogrow_items(segments)
            if not segment_items:
                raise ValueError(t("err_no_segments"))

            parsed_segments = []
            for ordinal, seg_cond in segment_items:
                if seg_cond is None:
                    raise ValueError(f"段_{ordinal} 为 None。")
                if not isinstance(seg_cond, list) or len(seg_cond) == 0:
                    raise ValueError(t("err_invalid_segment", ordinal=ordinal))
                if not isinstance(seg_cond[0], (list, tuple)):
                    raise ValueError(
                        f"段_{ordinal} 格式异常：不是标准 conditioning 列表。"
                    )
                if len(seg_cond[0]) < 2:
                    raise ValueError(
                        f"段_{ordinal} 格式异常：缺少 conditioning dict。"
                    )
                cond_dict = seg_cond[0][1]
                if not isinstance(cond_dict, dict):
                    raise ValueError(
                        f"段_{ordinal} 格式异常：conditioning 的第二项不是 dict。"
                    )
                if not cond_dict.get("_yimo_segment"):
                    raise ValueError(t("err_not_segment_bundle", ordinal=ordinal))

                data = cond_dict.get("_yimo_data", {})
                parsed_segments.append({
                    "positive": data.get("positive"),
                    "negative": data.get("negative"),
                    "latent": data.get("latent"),
                    "audio": data.get("audio"),
                    "frame_count": data.get("frame_count", 124),
                    "render_frames": data.get("render_frames", 124),
                    "prompt": data.get("prompt", ""),
                    "mode": data.get("mode", "text"),
                    "keyframe_specs": data.get("keyframe_specs", []),
                    "ref_blocks": data.get("ref_blocks", []),
                    "identity_refs": data.get("identity_refs", []),
                    "style_refs": data.get("style_refs", []),
                    "identity_ordinals": data.get("identity_ordinals", []),
                    "audio_policy": data.get("audio_policy", "generate_new"),
                    "width": data.get("width", 1344),
                    "height": data.get("height", 768),
                    "ref_curve": data.get("ref_curve", {}),
                })

            if len(parsed_segments) == 0:
                raise ValueError(t("err_no_valid_segments"))

            # ------------------------------------------------------------------
            # override_ref_curve
            # ------------------------------------------------------------------
            override_report = ""
            if override_ref_curve:
                overridden_count = 0
                skipped_count = 0
                for seg in parsed_segments:
                    positive = seg.get("positive")
                    if not isinstance(positive, list) or len(positive) == 0:
                        continue
                    first_item = positive[0]
                    if not isinstance(first_item, (list, tuple)) or len(first_item) < 2:
                        continue
                    cond_tensor, cond_dict = first_item[0], first_item[1]
                    if not isinstance(cond_dict, dict):
                        continue
                    new_cond_dict = dict(cond_dict)
                    new_cond_dict = _isolate_payload_cond(new_cond_dict)
                    curve_cfg = dict(new_cond_dict.get("_yimo_ref_curve", {}))
                    changed = False
                    if ref_curve_direction_override != "inherit":
                        curve_cfg["direction"] = ref_curve_direction_override
                        changed = True
                    if ref_curve_shape_override != "inherit":
                        curve_cfg["shape"] = ref_curve_shape_override
                        changed = True
                    new_cond_dict["_yimo_ref_curve"] = curve_cfg
                    seg["positive"] = [[cond_tensor, new_cond_dict]] + list(positive[1:])
                    if changed:
                        overridden_count += 1
                    else:
                        skipped_count += 1
                override_report = (
                    f"override_ref_curve=True: overrode {overridden_count} segment(s); "
                    f"skipped {skipped_count} (both inherit)"
                )

            # ------------------------------------------------------------------
            # 空间尺寸校验
            # ------------------------------------------------------------------
            base_latent = parsed_segments[0]["latent"]
            base_video = None
            try:
                base_video, _ = nested_av_parts(base_latent)
            except Exception:
                pass

            for i, seg in enumerate(parsed_segments):
                if seg["positive"] is None or seg["latent"] is None:
                    raise ValueError(t("err_segment_incomplete", index=i))
                if base_video is not None:
                    try:
                        v, _ = nested_av_parts(seg["latent"])
                        if tuple(v.shape[-2:]) != tuple(base_video.shape[-2:]):
                            raise ValueError(
                                f"Spatial mismatch: segment_0 {tuple(base_video.shape[-2:])} "
                                f"vs segment_{i} {tuple(v.shape[-2:])}"
                            )
                    except Exception as e:
                        if "Spatial mismatch" in str(e):
                            raise
                        raise ValueError(t("err_segment_invalid", index=i, error=str(e)))

            # ------------------------------------------------------------------
            # overlap_list
            # ------------------------------------------------------------------
            effective_overlaps = None
            if overlap_list and str(overlap_list).strip():
                try:
                    parts = [p.strip() for p in str(overlap_list).split(",") if p.strip()]
                    parsed = [int(p) for p in parts]
                    if all(p < 0 for p in parsed):
                        effective_overlaps = None
                    else:
                        default_olap = int(overlap_frames)
                        effective_overlaps = [default_olap if p < 0 else p for p in parsed]
                except ValueError as e:
                    raise ValueError(t("err_overlap_list_parse", error=e))

            # ------------------------------------------------------------------
            # segment_seeds
            # ------------------------------------------------------------------
            parsed_segment_seeds = None
            if segment_seeds and str(segment_seeds).strip():
                parts = [p.strip() for p in str(segment_seeds).split(",") if p.strip()]
                parsed = []
                for p in parts:
                    if p.lower() == "none":
                        parsed.append(None)
                        continue
                    try:
                        parsed.append(int(p))
                    except ValueError:
                        parsed.append(None)
                if any(s is not None for s in parsed):
                    parsed_segment_seeds = parsed

            # ------------------------------------------------------------------
            # face_bank
            # ------------------------------------------------------------------
            face_bank = None
            if face_restore and is_connected_value(face_images):
                face_bank = build_identity_bank(face_images)

            # ------------------------------------------------------------------
            # 进度条 / previewer
            # ------------------------------------------------------------------
            actual_segments = len(parsed_segments)
            if stop_after_segment >= 0:
                actual_segments = min(stop_after_segment + 1, actual_segments)
            if resample_segment >= 0 and resample_segment < len(parsed_segments):
                actual_segments = actual_segments - resample_segment
            total_steps = max(1, actual_segments * steps)
            pbar = comfy.utils.ProgressBar(total_steps)

            device = comfy.model_management.get_torch_device()
            previewer = None
            try:
                latent_format = getattr(getattr(model, "model", None), "latent_format", None)
                if latent_format is None:
                    latent_format = getattr(model, "latent_format", None)
                if latent_format is not None:
                    previewer = comfy.utils.get_previewer(device, latent_format)
            except Exception:
                pass

            def _progress_callback(step, x0, x, total):
                preview_bytes = None
                if previewer is not None:
                    try:
                        if hasattr(x0, "is_nested") and x0.is_nested:
                            parts = x0.unbind()
                            preview_bytes = previewer.decode_latent_to_preview_image(parts[0])
                        else:
                            preview_bytes = previewer.decode_latent_to_preview_image(x0)
                    except Exception:
                        pass
                pbar.update_absolute(step, total, preview_bytes)
                return preview_bytes

            face_params = {
                "strength": face_restore_strength,
                "similarity_threshold": face_similarity_threshold,
                "repair_backend": face_restore_backend,
            }

            # ==================================================================
            # 采样分发
            # ==================================================================
            effective_overlaps_arg = (
                effective_overlaps if effective_overlaps is not None
                else overlap_frames
            )

            if use_dual_clock:
                final_latent, report_lines, segment_latents, segment_summaries = (
                    dual_clock_sequence_sample(
                        model=model,
                        segments=parsed_segments,
                        sampler_name=sampler_name,
                        scheduler=scheduler,
                        steps=steps,
                        shift_video=shift_video,
                        shift_audio=shift_audio,
                        cfg=cfg,
                        denoise=denoise,
                        seed=seed,
                        overlap_frames=effective_overlaps_arg,
                        transition_mode=transition_mode,
                        audio_continuity=audio_continuity,
                        callback=_progress_callback,
                        video_vae=video_vae,
                        auto_inherit_tail=auto_inherit_tail,
                        inherit_identity=inherit_identity,
                        resample_segment=resample_segment,
                        segment_seeds=parsed_segment_seeds,
                        chain_id=chain_id,
                        stop_after_segment=stop_after_segment,
                        face_restore=face_restore and video_vae is not None,
                        face_bank=face_bank,
                        face_params=face_params,
                        previewer=previewer,
                        enable_ref_curve=enable_ref_curve,
                        color_match_mode=color_match_mode,
                        color_match_strength=color_match_strength,
                        color_match_reference_frames=color_match_reference_frames,
                    )
                )
                report_lines.insert(0, "backend=dual_clock_euler")

            else:
                final_latent, report_lines, segment_latents, segment_summaries = sequence_sample(
                    model=model,
                    segments=parsed_segments,
                    overlap_frames=effective_overlaps_arg,
                    sampler_name=native_sampler_name,
                    scheduler=scheduler,
                    steps=steps,
                    cfg=cfg,
                    seed=seed,
                    denoise=denoise,
                    transition_mode=transition_mode,
                    audio_continuity=audio_continuity,
                    callback=_progress_callback,
                    video_vae=video_vae,
                    auto_inherit_tail=auto_inherit_tail,
                    inherit_identity=inherit_identity,
                    resample_segment=resample_segment,
                    segment_seeds=parsed_segment_seeds,
                    chain_id=chain_id,
                    stop_after_segment=stop_after_segment,
                    face_restore=face_restore and video_vae is not None,
                    face_bank=face_bank,
                    face_params=face_params,
                    previewer=previewer,
                    enable_ref_curve=enable_ref_curve,
                    color_match_mode=color_match_mode,
                    color_match_strength=color_match_strength,
                    color_match_reference_frames=color_match_reference_frames,
                )
                report_lines.insert(0, f"backend=native_{native_sampler_name}")

            if override_report:
                report_lines.append(override_report)

            # ------------------------------------------------------------------
            # images 输出
            # ------------------------------------------------------------------
            final_images = None
            if output_format == "images":
                if video_vae is None:
                    report_lines.append(
                        "output_format=images 需要 video_vae，已自动回退到 latent 输出。"
                    )
                else:
                    try:
                        final_images = cls._smart_chunked_decode(
                            final_latent, parsed_segments,
                            effective_overlaps_arg, overlap_frames,
                            transition_mode, video_vae, report_lines,
                            chunk_memory_threshold_mb=chunk_memory_threshold_mb,
                        )
                    except Exception as e:
                        logger.warning("smart chunked decode failed: %s", e)
                        report_lines.append(f"smart chunked decode failed: {e}")
                        final_images = None

            # ------------------------------------------------------------------
            # 自动导出
            # ------------------------------------------------------------------
            if auto_export_final and final_latent is not None:
                try:
                    export_dir = get_sequence_sampler_export_dir()
                    if export_name.strip():
                        filename = f"{sanitize_filename(export_name)}.pt"
                    else:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"sequence_{sanitize_filename(chain_id)}_{timestamp}.pt"
                    filepath = os.path.join(export_dir, filename)

                    export_data = {
                        "final_latent": final_latent,
                        "metadata": {
                            "sampler_name": sampler_name,
                            "use_dual_clock": bool(use_dual_clock),
                            "scheduler": scheduler,
                            "steps": steps,
                            "shift_video": float(shift_video),
                            "shift_audio": float(shift_audio),
                            "cfg": float(cfg),
                            "denoise": float(denoise),
                            "chain_id": chain_id,
                            "total_segments": len(parsed_segments),
                            "segments_sampled": len(segment_latents),
                            "seed": seed,
                            "timestamp": datetime.now().isoformat(),
                            "version": "5.0.0",
                        },
                    }
                    export_data["segments_summary"] = [
                        {
                            "index": idx,
                            "mode": s.mode,
                            "frame_count": s.frame_count,
                            "render_frames": s.render_frames,
                            "tail_overlap_frames": s.tail_overlap_frames,
                            "audio_policy": s.audio_policy,
                        }
                        for idx, s in enumerate(segment_summaries)
                    ]
                    torch.save(export_data, filepath)
                    report_lines.append(f"💾 最终结果已导出: {filepath}")
                except Exception as e:
                    logger.warning("自动导出失败: %s", e)
                    report_lines.append(f"⚠️ 自动导出失败: {e}")

            report = "\n".join(report_lines)
            if output_format == "images" and final_images is not None:
                padded = [None] * 8
                return io.NodeOutput(None, report, *padded, final_images)
            padded = segment_latents + [None] * (8 - len(segment_latents))
            return io.NodeOutput(final_latent, report, *padded, None)

        @staticmethod
        def _smart_chunked_decode(
            final_latent, parsed_segments, pair_overlaps_source,
            overlap_frames, transition_mode, video_vae, report_lines,
            chunk_memory_threshold_mb=_CHUNK_MEMORY_THRESHOLD_DEFAULT_MB,
        ):
            import os as _os
            import shutil as _shutil
            import tempfile as _tempfile
            import comfy.model_management as _mm

            threshold_bytes = int(chunk_memory_threshold_mb) * 1024 * 1024

            n_pairs = max(0, len(parsed_segments) - 1)
            if isinstance(pair_overlaps_source, list):
                pair_overlaps = list(pair_overlaps_source)[:n_pairs]
                while len(pair_overlaps) < n_pairs:
                    pair_overlaps.append(0)
            else:
                pair_overlaps = [int(pair_overlaps_source)] * n_pairs

            if transition_mode != "context_inject":
                pair_overlaps = [0] * n_pairs

            chunks = []
            start = 0
            for i, ov in enumerate(pair_overlaps):
                if ov == 0:
                    chunks.append((start, i + 1))
                    start = i + 1
            chunks.append((start, n_pairs + 1))

            report_lines.append("=== v5.0.0 smart_chunked_decode ===")
            report_lines.append(f"n_chunks={len(chunks)} boundaries={chunks}")
            report_lines.append(f"memory_threshold_mb={chunk_memory_threshold_mb}")

            seg_latent_ts = []
            for seg in parsed_segments:
                fc = seg.get("render_frames", seg.get("frame_count", 124))
                if fc <= 5:
                    seg_latent_ts.append(2)
                else:
                    seg_latent_ts.append(((fc - 5) // 17) * 5 + 2)

            trimmed_latent_ts = []
            for i, lt in enumerate(seg_latent_ts):
                if i == 0:
                    trimmed_latent_ts.append(lt)
                else:
                    ov = pair_overlaps[i - 1]
                    _, v_olap_t, _ = _overlap_to_latent(ov) if ov > 0 else (0, 0, 0)
                    v_olap_t = min(v_olap_t, lt)
                    trimmed_latent_ts.append(lt - v_olap_t)

            cum = [0]
            for lt in trimmed_latent_ts:
                cum.append(cum[-1] + lt)

            final_video, _ = nested_av_parts(final_latent)

            tmp_root = _os.path.join(get_yimo_output_dir(), "_seq_chunk_cache")
            _os.makedirs(tmp_root, exist_ok=True)
            tmp_dir = None

            chunk_storage: list[tuple[str, object]] = []
            in_memory_bytes = 0
            spilled = False

            def _ensure_tmp_dir():
                nonlocal tmp_dir
                if tmp_dir is None:
                    tmp_dir = _tempfile.mkdtemp(prefix="yimo_seq_", dir=tmp_root)
                return tmp_dir

            try:
                for ci, (cs, ce) in enumerate(chunks):
                    c_t0 = cum[cs]
                    c_t1 = cum[ce]
                    if c_t1 <= c_t0:
                        continue
                    chunk_video = final_video[:, :, c_t0:c_t1].contiguous()
                    decoded = video_vae.decode(chunk_video)
                    if decoded.ndim == 5:
                        if decoded.shape[0] == 1:
                            decoded = decoded.squeeze(0)
                        elif decoded.shape[1] == 1:
                            decoded = decoded.squeeze(1)
                    if decoded.ndim != 4:
                        raise ValueError(f"chunk_{ci} decode shape 异常: {tuple(decoded.shape)}")
                    if decoded.shape[-1] > 3:
                        decoded = decoded[..., :3]
                    decoded = decoded.clamp(0.0, 1.0)

                    decoded_cpu = decoded.detach().to("cpu").contiguous()
                    chunk_bytes = decoded_cpu.numel() * decoded_cpu.element_size()

                    if in_memory_bytes + chunk_bytes <= threshold_bytes:
                        chunk_storage.append(("mem", decoded_cpu))
                        in_memory_bytes += chunk_bytes
                        report_lines.append(
                            f"chunk_{ci}: latent_t=[{c_t0},{c_t1}) frames={decoded_cpu.shape[0]} "
                            f"kept_in_memory ({chunk_bytes / 1024**2:.1f} MB)"
                        )
                    else:
                        _ensure_tmp_dir()
                        path = _os.path.join(tmp_dir, f"chunk_{ci:03d}.pt")
                        torch.save(decoded_cpu, path)
                        chunk_storage.append(("disk", path))
                        spilled = True
                        report_lines.append(
                            f"chunk_{ci}: latent_t=[{c_t0},{c_t1}) frames={decoded_cpu.shape[0]} "
                            f"spilled_to_disk ({chunk_bytes / 1024**2:.1f} MB)"
                        )

                    del decoded, decoded_cpu, chunk_video
                    try:
                        _mm.soft_empty_cache()
                    except Exception:
                        pass

                pixel_parts = []
                for kind, payload in chunk_storage:
                    if kind == "mem":
                        pixel_parts.append(payload)
                    else:
                        pixel_parts.append(
                            torch.load(payload, map_location="cpu", weights_only=True)
                        )
                final_images = torch.cat(pixel_parts, dim=0)
                report_lines.append(
                    f"final_images_frames={final_images.shape[0]} "
                    f"(in_memory_chunks={sum(1 for k, _ in chunk_storage if k == 'mem')}, "
                    f"disk_chunks={sum(1 for k, _ in chunk_storage if k == 'disk')})"
                )
                return final_images

            finally:
                if spilled and tmp_dir is not None:
                    try:
                        _shutil.rmtree(tmp_dir, ignore_errors=True)
                    except Exception:
                        pass

except Exception as _e:
    print("=" * 70, file=sys.stderr)
    print("YimoH3SequenceSampler MODULE LOAD FAILED:", _e, file=sys.stderr)
    traceback.print_exc()
    print("=" * 70, file=sys.stderr)
    raise