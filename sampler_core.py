# sampler_core.py
# v2.7.1:
# - 采样循环、PDD 双采、分块采样均加入显存清理。
# - _sample_single_segment_pdd_two_stage 放大后显式释放 LOW 阶段中间态。
# - sequence_sample 每段采样完成后 soft_empty_cache。
# v2.7.0:
# - PDD 双阶段采样（LOW → 插值放大 → HIGH）内部实现。
# - 段间帧数反推统一使用 core.get_video_frame_count_from_latent。
# v2.6.4:
# - _concatenate_av_latents 支持视频和音频使用独立的 overlap 列表。

from __future__ import annotations

import copy
import inspect
import logging
import os
import time
from typing import Callable, Sequence

import torch

import comfy.model_management
import comfy.nested_tensor
import comfy.sample

from .core import (
    compute_reference_curve,
    nested_av_parts,
    temporal_shape,
    align_frame_count,
    align_keyframe_position,
    get_sequence_sampler_cache_dir,
    get_video_frame_count_from_latent,
    video_latent_t,
    FPS,
    AUDIO_LATENT_FPS,
)
from .segment_bridge import SegmentSummary
from .transition_policies import resolve_transition
from .face_restore import restore_sampled_av
from .color_match_utils import apply_color_match


logger = logging.getLogger("YimoH3")


def _soft_empty_cache():
    """v2.7.1: 安全地请求 ComfyUI 清空缓存。

    不抛异常，失败时静默跳过。用于采样循环、分块、PDD 中间态等位置，
    减少长视频多段场景下的显存累积。
    """
    try:
        comfy.model_management.soft_empty_cache()
    except Exception:
        pass


# =============================================================================
# PDD sigmas generation
# =============================================================================

def _shifted_sigma(shift: float, sigma: torch.Tensor) -> torch.Tensor:
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def _get_model_sigma_bounds(model):
    sigma_min = 0.0
    sigma_max = 1.0

    if model is None:
        return sigma_min, sigma_max

    ms = None
    try:
        getter = getattr(model, "get_model_object", None)
        if callable(getter):
            ms = getter("model_sampling")
    except Exception:
        ms = None

    if ms is None:
        ms = getattr(model, "model_sampling", None)

    if ms is None:
        inner = getattr(model, "model", None)
        if inner is not None:
            try:
                getter = getattr(inner, "get_model_object", None)
                if callable(getter):
                    ms = getter("model_sampling")
            except Exception:
                ms = None
            if ms is None:
                ms = getattr(inner, "model_sampling", None)

    if ms is not None:
        try:
            sigma_min = float(getattr(ms, "sigma_min", sigma_min))
        except Exception:
            pass
        try:
            sigma_max = float(getattr(ms, "sigma_max", sigma_max))
        except Exception:
            pass

    if not (sigma_min < sigma_max):
        sigma_min, sigma_max = 0.0, 1.0

    return sigma_min, sigma_max


def build_pdd_sigmas(
    model=None,
    num_steps: int = 32,
    block_size: int = 4,
    shift: float = 12.0,
    device=None,
) -> torch.Tensor:
    if device is None:
        device = comfy.model_management.intermediate_device()

    sigma_min, sigma_max = _get_model_sigma_bounds(model)

    sigma = torch.linspace(
        sigma_max, sigma_min,
        num_steps + 1,
        dtype=torch.float64,
        device=device,
    )

    shifted = _shifted_sigma(shift, sigma)

    shifted[0] = sigma_max
    shifted[-1] = 0.0

    return shifted.to(dtype=torch.float32)


def build_pdd_two_stage_sigmas(
    model=None,
    low_steps: int = 4,
    high_steps: int = 4,
    shift: float = 12.0,
    device=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if device is None:
        device = comfy.model_management.intermediate_device()

    total_blocks = low_steps + high_steps
    total_internal_steps = total_blocks * 4

    full_sigmas = build_pdd_sigmas(
        model=model,
        num_steps=total_internal_steps,
        block_size=4,
        shift=shift,
        device=device,
    )

    split_idx = low_steps * 4

    low_sigmas = full_sigmas[: split_idx + 1].clone()
    high_sigmas = full_sigmas[split_idx:].clone()
    high_sigmas[-1] = 0.0

    return low_sigmas, high_sigmas


# =============================================================================
# PDD two-stage helpers
# =============================================================================

def _split_pdd_sigmas(sigmas, split_step: int):
    if sigmas is None:
        raise ValueError("PDD two-stage 需要提供 sigmas（来自 build_pdd_sigmas 或外部 PDD Apply）")
    n = len(sigmas)
    if split_step <= 0 or split_step >= n - 1:
        raise ValueError(
            f"split_step={split_step} 超出有效范围。"
            f"sigmas 共 {n} 个值（{n - 1} 步），split_step 应在 1~{n - 2} 之间。"
        )
    return sigmas[: split_step + 1], sigmas[split_step:]


def _upsample_video_latent_bilinear(
    video_latent: torch.Tensor,
    target_h: int,
    target_w: int,
    dtype,
) -> torch.Tensor:
    B, C, T, H, W = video_latent.shape
    video_bt = video_latent.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    upsampled_bt = torch.nn.functional.interpolate(
        video_bt.to(torch.float32),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    ).to(dtype)
    return (
        upsampled_bt.reshape(B, T, C, target_h, target_w)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )


# =============================================================================
# Overlap → latent mapping
# =============================================================================

def _overlap_to_latent(overlap_frames: int) -> tuple[int, int, int]:
    overlap_frames = int(overlap_frames)
    if overlap_frames <= 0:
        return 0, 0, 0
    if overlap_frames <= 5:
        effective = 5
    else:
        n = max(0, round((overlap_frames - 5) / 17.0))
        effective = 17 * n + 5
    latent_t = video_latent_t(effective)
    audio_t = round(effective / FPS * AUDIO_LATENT_FPS)
    return effective, latent_t, audio_t


# =============================================================================
# Segment-level cache
# =============================================================================

def _get_segment_cache_dir(chain_id: str) -> str:
    return get_sequence_sampler_cache_dir(chain_id)


def _segment_cache_path(chain_id: str, segment_index: int) -> str:
    return os.path.join(_get_segment_cache_dir(chain_id), f"seg_{segment_index:04d}.pt")


def _summary_to_dict(summary: SegmentSummary) -> dict:
    return {
        "mode": summary.mode,
        "canvas": tuple(summary.canvas),
        "frame_count": int(summary.frame_count),
        "render_frames": int(summary.render_frames),
        "tail_overlap_frames": int(summary.tail_overlap_frames),
        "tail_frame_pixel": summary.tail_frame_pixel,
        "tail_audio_latent": summary.tail_audio_latent,
        "identity_refs": list(summary.identity_refs),
        "style_refs": list(summary.style_refs),
        "audio_policy": summary.audio_policy,
        "prompt": summary.prompt,
    }


def _dict_to_summary(d: dict) -> SegmentSummary:
    return SegmentSummary(
        mode=d.get("mode", "text"),
        canvas=tuple(d.get("canvas", (1344, 768))),
        frame_count=int(d.get("frame_count", 124)),
        render_frames=int(d.get("render_frames", 124)),
        tail_overlap_frames=int(d.get("tail_overlap_frames", 0)),
        tail_frame_pixel=d.get("tail_frame_pixel"),
        tail_audio_latent=d.get("tail_audio_latent"),
        identity_refs=list(d.get("identity_refs", [])),
        style_refs=list(d.get("style_refs", [])),
        audio_policy=d.get("audio_policy", "generate_new"),
        prompt=d.get("prompt", ""),
    )


def _serialize_sampled(sampled: dict) -> dict:
    samples = sampled.get("samples")
    payload: dict = {}
    if getattr(samples, "is_nested", False):
        parts = tuple(samples.unbind())
        payload["_nested_parts"] = [p.detach().cpu() for p in parts]
    elif isinstance(samples, torch.Tensor):
        payload["_tensor"] = samples.detach().cpu()
    else:
        raise ValueError(f"Unsupported samples type: {type(samples)}")

    noise_mask = sampled.get("noise_mask")
    if noise_mask is not None:
        if getattr(noise_mask, "is_nested", False):
            mask_parts = tuple(noise_mask.unbind())
            payload["_noise_mask_nested"] = [p.detach().cpu() for p in mask_parts]
        elif isinstance(noise_mask, torch.Tensor):
            payload["_noise_mask_tensor"] = noise_mask.detach().cpu()
    return payload


def _deserialize_sampled(payload: dict, device) -> dict:
    if "_nested_parts" in payload:
        parts = [p.to(device) for p in payload["_nested_parts"]]
        samples = comfy.nested_tensor.NestedTensor(tuple(parts))
    elif "_tensor" in payload:
        samples = payload["_tensor"].to(device)
    else:
        raise ValueError("Serialized sampled payload missing samples")

    out: dict = {"samples": samples}

    if "_noise_mask_nested" in payload:
        mask_parts = [p.to(device) for p in payload["_noise_mask_nested"]]
        out["noise_mask"] = comfy.nested_tensor.NestedTensor(tuple(mask_parts))
    elif "_noise_mask_tensor" in payload:
        out["noise_mask"] = payload["_noise_mask_tensor"].to(device)

    return out


def _save_segment_cache(chain_id: str, segment_index: int, sampled: dict, summary: SegmentSummary):
    if not chain_id:
        return ""
    path = _segment_cache_path(chain_id, segment_index)
    try:
        torch.save({
            "sampled_payload": _serialize_sampled(sampled),
            "summary": _summary_to_dict(summary),
            "segment_index": segment_index,
            "cache_format": "v2",
        }, path)
        logger.debug("Saved segment cache: %s", path)
        return path
    except Exception as e:
        logger.warning("Failed to save segment cache to %s: %s", path, e)
        return ""


def _load_segment_cache(chain_id: str, segment_index: int, device):
    path = _segment_cache_path(chain_id, segment_index)
    if not os.path.exists(path):
        return None

    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        logger.warning(
            "Failed to load segment cache from %s with weights_only=True: %s. "
            "Refusing unsafe fallback; treat as cache miss.",
            path, e,
        )
        return None

    try:
        if data.get("cache_format") == "v2" and "sampled_payload" in data:
            sampled = _deserialize_sampled(data["sampled_payload"], device)
        elif "sampled" in data:
            logger.warning(
                "Segment cache %s uses legacy format without sampled_payload; "
                "treat as cache miss to avoid unsafe load.", path,
            )
            return None
        else:
            logger.warning("Segment cache %s missing samples; cache miss.", path)
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
        logger.error("Failed to decode segment cache from %s: %s", path, e)
        return None


# =============================================================================
# Tail extraction / context injection
# =============================================================================

def _extract_tail_context(av_latent: dict, overlap_frames: int) -> dict | None:
    if overlap_frames <= 0:
        return None
    video, audio = nested_av_parts(av_latent)

    effective_frames, target_latent_t, target_audio_t = _overlap_to_latent(overlap_frames)
    actual_latent_t = min(target_latent_t, video.shape[2])
    actual_audio_t = min(target_audio_t, audio.shape[-1])

    if actual_latent_t <= 0 or actual_audio_t <= 0:
        return None

    if actual_latent_t >= target_latent_t and actual_audio_t >= target_audio_t:
        actual_overlap = effective_frames
    else:
        if actual_latent_t <= 2:
            actual_overlap = 5
        else:
            n = (actual_latent_t - 2) // 5
            actual_overlap = 17 * n + 5
        if actual_overlap > effective_frames:
            actual_overlap = effective_frames
        logger.debug(
            "Tail truncated: requested=%d, effective=%d, available_latent_t=%d, "
            "actual_overlap=%d frames",
            overlap_frames, effective_frames, video.shape[2], actual_overlap
        )

    return {
        "video_tail": video[:, :, -actual_latent_t:, :, :].detach().clone(),
        "audio_tail": audio[:, :, :, -actual_audio_t:].detach().clone(),
        "overlap_frames": actual_overlap,
        "requested_overlap_frames": int(overlap_frames),
        "effective_overlap_frames": effective_frames,
        "tail_latent_t": actual_latent_t,
        "tail_audio_t": actual_audio_t,
        "video_shape": list(video.shape),
        "audio_shape": list(audio.shape),
    }


def _inject_context(
    av_latent: dict,
    context: dict | None,
    video_continuity: str = "context_inject",
    audio_continuity: str = "context_inject",
) -> tuple[dict, str]:
    video, audio = nested_av_parts(av_latent)
    device = video.device
    dtype = video.dtype

    orig_vm, orig_am = None, None
    existing_mask = av_latent.get("noise_mask")
    if existing_mask is not None and getattr(existing_mask, "is_nested", False):
        parts = tuple(existing_mask.unbind())
        if len(parts) == 2:
            orig_vm, orig_am = parts

    video_mask = orig_vm.clone() if orig_vm is not None else torch.ones_like(video)
    audio_mask = orig_am.clone() if orig_am is not None else torch.ones_like(audio)

    new_video = video.clone()
    new_audio = audio.clone()
    reports: list[str] = []

    if video_continuity == "context_inject":
        if context is not None:
            video_tail = context["video_tail"]
            tail_vt = video_tail.shape[2]
            if video.shape[-2:] != video_tail.shape[-2:]:
                raise ValueError(
                    f"Video canvas spatial mismatch: "
                    f"current {tuple(video.shape[-2:])}, tail {tuple(video_tail.shape[-2:])}"
                )
            if tail_vt > video.shape[2]:
                raise ValueError(
                    f"Video tail larger than current segment: "
                    f"tail_t={tail_vt}, curr_t={video.shape[2]}"
                )
            new_video[:, :, :tail_vt, :, :] = video_tail.to(device=device, dtype=dtype)
            video_mask[:, :, :tail_vt, :, :] = 0.0

            if tail_vt < 5:
                semantic_tag = "SHORT: reference-frame semantics, not time-prefix"
                logger.warning(
                    "[YimoH3] Video tail injection is very short (%d latent steps). "
                    "DiT may treat it as a global reference template rather than a "
                    "time prefix, locking the visual style (outfit/scene) of the "
                    "previous segment. For outfit/style changes, use overlap_frames >= 22. "
                    "requested=%s, effective=%s",
                    tail_vt,
                    context.get("requested_overlap_frames", "?"),
                    context.get("effective_overlap_frames", "?"),
                )
            else:
                semantic_tag = "PREFIX: time-prefix semantics"
            reports.append(f"video_inject={tail_vt}/{video.shape[2]} [{semantic_tag}]")
        else:
            reports.append("video: no previous context (first segment)")
    else:
        reports.append("video: hard_concat (no injection)")

    if audio_continuity == "context_inject":
        if context is not None:
            audio_tail = context["audio_tail"]
            tail_at = audio_tail.shape[-1]
            if audio.shape[1:-1] != audio_tail.shape[1:-1]:
                raise ValueError(
                    f"Audio channel mismatch: "
                    f"current {tuple(audio.shape[1:-1])}, tail {tuple(audio_tail.shape[1:-1])}"
                )
            if tail_at > audio.shape[-1]:
                raise ValueError(
                    f"Audio tail larger than current segment: "
                    f"tail_at={tail_at}, curr_at={audio.shape[-1]}"
                )
            new_audio[:, :, :, :tail_at] = audio_tail.to(device=device, dtype=dtype)
            audio_mask[:, :, :, :tail_at] = 0.0
            reports.append(f"audio_inject={tail_at}/{audio.shape[-1]}")
        else:
            reports.append("audio: no previous context (first segment)")
    else:
        reports.append("audio: break (no injection)")

    out = av_latent.copy()
    out["samples"] = comfy.nested_tensor.NestedTensor((new_video, new_audio))
    out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
    return out, "; ".join(reports)


# =============================================================================
# Noise / sample kwargs
# =============================================================================

def _make_noise_for_latent(latent: dict, seed: int):
    device = comfy.model_management.get_torch_device()
    generator = torch.Generator(device=device).manual_seed(seed)

    samples = latent.get("samples")
    if getattr(samples, "is_nested", False):
        parts = tuple(samples.unbind())
        noise_parts = []
        for p in parts:
            noise = torch.randn(p.shape, generator=generator, device=device, dtype=p.dtype)
            noise_parts.append(noise)
        return comfy.nested_tensor.NestedTensor(tuple(noise_parts))
    else:
        noise = torch.randn(
            samples.shape,
            generator=generator,
            device=device,
            dtype=samples.dtype,
        )
        return noise


_SAMPLE_SIG_CACHE = None


def _get_sample_kwargs(
    model,
    noise,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    positive,
    negative,
    latent_image,
    noise_mask,
    denoise: float,
    seed: int,
    callback=None,
    disable_pbar=True,
    sigmas=None,
) -> dict:
    global _SAMPLE_SIG_CACHE
    if _SAMPLE_SIG_CACHE is None:
        try:
            _SAMPLE_SIG_CACHE = set(inspect.signature(comfy.sample.sample).parameters.keys())
        except Exception:
            _SAMPLE_SIG_CACHE = {
                "model", "noise", "steps", "cfg", "sampler_name", "scheduler",
                "positive", "negative", "latent_image", "noise_mask", "denoise",
                "seed", "callback", "sigmas",
            }

    kwargs = {
        "model": model,
        "noise": noise,
        "steps": steps,
        "cfg": cfg,
        "sampler_name": sampler_name,
        "scheduler": scheduler,
        "positive": positive,
        "negative": negative,
        "latent_image": latent_image,
        "denoise": denoise,
        "seed": seed,
    }
    if noise_mask is not None and "noise_mask" in _SAMPLE_SIG_CACHE:
        kwargs["noise_mask"] = noise_mask
    if callback is not None and "callback" in _SAMPLE_SIG_CACHE:
        kwargs["callback"] = callback
    if "disable_pbar" in _SAMPLE_SIG_CACHE:
        kwargs["disable_pbar"] = disable_pbar
    if sigmas is not None and "sigmas" in _SAMPLE_SIG_CACHE:
        kwargs["sigmas"] = sigmas

    return {k: v for k, v in kwargs.items() if k in _SAMPLE_SIG_CACHE}


# =============================================================================
# Debug helpers
# =============================================================================

def _debug_dump_cond_dict(positive, tag: str):
    try:
        if not positive or not isinstance(positive, list) or len(positive) == 0:
            logger.info("[YimoH3-DEBUG/%s] positive is empty", tag)
            return
        first_item = positive[0]
        if not isinstance(first_item, (list, tuple)) or len(first_item) < 2:
            logger.info("[YimoH3-DEBUG/%s] positive[0] malformed", tag)
            return
        cd = first_item[1]
        if not isinstance(cd, dict):
            logger.info("[YimoH3-DEBUG/%s] positive[0][1] is not dict (got %s)", tag, type(cd).__name__)
            return
        keys = list(cd.keys())
        vcna_top = cd.get("minimax_visual_cond_noise_aug", "MISSING")
        payload = cd.get("minimax_payload")
        vcna_in_payload = "N/A"
        if payload is not None and hasattr(payload, "cond") and isinstance(payload.cond, dict):
            vcna_in_payload = payload.cond.get("visual_cond_noise_aug", "MISSING_IN_PAYLOAD")
        elif isinstance(payload, dict):
            vcna_in_payload = payload.get("visual_cond_noise_aug", "MISSING_IN_PAYLOAD")
        refs_count = len(cd.get("minimax_refs", [])) if isinstance(cd.get("minimax_refs"), list) else "N/A"
        kfs_count = len(cd.get("minimax_keyframes", [])) if isinstance(cd.get("minimax_keyframes"), list) else "N/A"
        logger.info(
            "[YimoH3-DEBUG/%s] cond_dict keys(top10)=%s | visual_cond_noise_aug(top)=%s | "
            "visual_cond_noise_aug(payload)=%s | refs=%s | keyframes=%s",
            tag, keys[:10], vcna_top, vcna_in_payload, refs_count, kfs_count,
        )
    except Exception as e:
        logger.info("[YimoH3-DEBUG/%s] debug dump failed: %s", tag, e)


# =============================================================================
# Minimal-cost isolation helpers
# =============================================================================

def _isolate_payload_cond(cond_dict: dict) -> dict:
    if not isinstance(cond_dict, dict):
        return cond_dict

    payload = cond_dict.get("minimax_payload")
    if payload is None:
        return cond_dict

    if not hasattr(payload, "cond"):
        return cond_dict

    inner_cond = getattr(payload, "cond", None)
    if not isinstance(inner_cond, dict):
        return cond_dict

    try:
        new_payload = copy.copy(payload)
        new_payload.cond = dict(inner_cond)
        cond_dict["minimax_payload"] = new_payload
    except Exception as e:
        logger.debug("[YimoH3] _isolate_payload_cond failed: %s", e)

    return cond_dict


def _isolate_ref_dicts(refs: list) -> list:
    if not isinstance(refs, list):
        return refs

    isolated = []
    for r in refs:
        if isinstance(r, dict):
            isolated.append(dict(r))
        else:
            isolated.append(r)
    return isolated


# =============================================================================
# Single segment sampling
# =============================================================================

def _sample_single_segment(
    model,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    positive,
    negative,
    latent: dict,
    denoise: float = 1.0,
    callback=None,
    disable_pbar: bool = True,
    segment_offset: int = 0,
    total_video_frames: int | None = None,
    sigmas=None,
):
    _debug_dump_cond_dict(positive, "before_curve")

    curve_config = None
    if positive and isinstance(positive, list) and len(positive) > 0:
        first_item = positive[0]
        if isinstance(first_item, (list, tuple)) and len(first_item) >= 2:
            cond_dict = first_item[1]
            if isinstance(cond_dict, dict):
                curve_config = cond_dict.get("_yimo_ref_curve")

    if curve_config:
        direction = curve_config.get("direction", "constant")
        base = float(curve_config.get("base_strength", 1.0))
        shape = curve_config.get("shape", "linear")
        fc = curve_config.get("frame_count", 124)

        if direction == "constant":
            logger.info(
                "[YimoH3] Reference curve: direction=constant, base_strength=%.3f "
                "(constant intensity, no per-frame modification)",
                base,
            )
        else:
            if total_video_frames is not None and total_video_frames > 0:
                abs_frame = segment_offset + fc // 2
                total = total_video_frames
            else:
                abs_frame = fc // 2
                total = fc

            abs_frame = min(max(0, abs_frame), max(0, total - 1))
            total = max(1, total)

            current_strength = compute_reference_curve(
                abs_frame, total, direction, shape, base
            )
            logger.info(
                "[YimoH3] Applying reference curve: direction=%s shape=%s "
                "abs_frame=%d/%d base=%.3f -> strength=%.3f",
                direction, shape, abs_frame, total, base, current_strength
            )

            new_positive = []
            for item in positive:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    new_positive.append(item)
                    continue
                cond_tensor, cond_dict = item[0], item[1]
                if isinstance(cond_dict, dict):
                    new_dict = dict(cond_dict)
                    new_dict = _isolate_payload_cond(new_dict)
                    new_dict["minimax_visual_cond_noise_aug"] = float(current_strength)
                    new_dict["_yimo_current_ref_strength"] = float(current_strength)
                    payload = new_dict.get("minimax_payload")
                    if payload is not None:
                        try:
                            if hasattr(payload, "cond") and isinstance(payload.cond, dict):
                                payload.cond["visual_cond_noise_aug"] = float(current_strength)
                            elif isinstance(payload, dict):
                                payload["visual_cond_noise_aug"] = float(current_strength)
                        except Exception as e:
                            logger.debug("[YimoH3] Failed to sync visual_cond_noise_aug into payload: %s", e)
                else:
                    new_dict = cond_dict
                new_positive.append([cond_tensor, new_dict])
            positive = new_positive

    _debug_dump_cond_dict(positive, "after_curve")

    samples_nested = latent.get("samples")
    noise = _make_noise_for_latent(latent, seed)

    noise_mask = latent.get("noise_mask")
    video_noise_mask = None
    audio_noise_mask = None

    if noise_mask is not None and getattr(noise_mask, "is_nested", False):
        mask_parts = tuple(noise_mask.unbind())
        if len(mask_parts) >= 1:
            video_noise_mask = mask_parts[0]
        if len(mask_parts) >= 2:
            audio_noise_mask = mask_parts[1]
    else:
        video_noise_mask = noise_mask

    source_audio = None
    audio_mask_for_restore = None
    if getattr(samples_nested, "is_nested", False):
        try:
            sample_parts = tuple(samples_nested.unbind())
            if len(sample_parts) >= 2:
                source_audio = sample_parts[1].clone()
                if audio_noise_mask is not None:
                    audio_mask_for_restore = audio_noise_mask.clone()
                    logger.info(
                        "[YimoH3] Audio mask range before sampling: [%.3f, %.3f]",
                        float(audio_mask_for_restore.min()),
                        float(audio_mask_for_restore.max()),
                    )
        except Exception as e:
            logger.debug("[YimoH3] audio mask detection failed: %s", e)

    kwargs = _get_sample_kwargs(
        model=model,
        noise=noise,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler_name,
        scheduler=scheduler,
        positive=positive,
        negative=negative,
        latent_image=samples_nested,
        noise_mask=video_noise_mask,
        denoise=denoise,
        seed=seed,
        callback=callback,
        disable_pbar=disable_pbar,
        sigmas=sigmas,
    )

    sampled_samples = comfy.sample.sample(**kwargs)

    if (
        source_audio is not None
        and audio_mask_for_restore is not None
        and getattr(sampled_samples, "is_nested", False)
    ):
        try:
            sampled_parts = tuple(sampled_samples.unbind())
            if len(sampled_parts) >= 2:
                sampled_audio = sampled_parts[1]
                src = source_audio.to(device=sampled_audio.device, dtype=sampled_audio.dtype)
                m = audio_mask_for_restore.to(device=sampled_audio.device, dtype=sampled_audio.dtype)
                if m.shape != sampled_audio.shape:
                    m = m.expand_as(sampled_audio)
                restored_audio = sampled_audio * m + src * (1.0 - m)
                sampled_samples = comfy.nested_tensor.NestedTensor(
                    (sampled_parts[0], restored_audio)
                )
                logger.info(
                    "[YimoH3] Audio per-latent restore applied "
                    "(mask range [%.3f, %.3f], weight=sampled*mask + source*(1-mask))",
                    float(m.min()), float(m.max()),
                )
        except Exception as e:
            logger.warning("[YimoH3] Failed to post-process audio mask: %s", e)

    out = {"samples": sampled_samples}
    if noise_mask is not None:
        out["noise_mask"] = noise_mask
    return out


# =============================================================================
# PDD two-stage sampling
# =============================================================================

def _sample_single_segment_pdd_two_stage(
    model,
    seed: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    positive,
    negative,
    latent: dict,
    full_sigmas,
    split_step: int = 4,
    scale_by: float = 1.5,
    denoise: float = 1.0,
    callback=None,
    disable_pbar: bool = True,
    segment_offset: int = 0,
    total_video_frames: int | None = None,
):
    """PDD 双阶段采样：LOW → 插值放大 → HIGH。

    v2.7.1: 放大后显式释放 LOW 阶段中间态并 soft_empty_cache，
    避免长视频多段采样时显存累积。
    """
    noise_end_sigmas, clean_end_sigmas = _split_pdd_sigmas(full_sigmas, split_step)
    n_low_steps = len(noise_end_sigmas) - 1
    n_high_steps = len(clean_end_sigmas) - 1

    logger.info(
        "[YimoH3-PDD2Stage] LOW=%d steps, HIGH=%d steps, split_step=%d, scale_by=%.2f",
        n_low_steps, n_high_steps, split_step, scale_by,
    )

    samples_nested = latent.get("samples")
    if getattr(samples_nested, "is_nested", False):
        device = tuple(samples_nested.unbind())[0].device
    else:
        device = samples_nested.device

    # ---- 1. LOW 阶段 ----
    low_sigmas_tensor = (
        noise_end_sigmas.to(device)
        if isinstance(noise_end_sigmas, torch.Tensor) else noise_end_sigmas
    )
    low_latent = _sample_single_segment(
        model=model,
        seed=seed,
        steps=n_low_steps,
        cfg=cfg,
        sampler_name=sampler_name,
        scheduler=scheduler,
        positive=positive,
        negative=negative,
        latent=latent,
        denoise=denoise,
        callback=callback,
        disable_pbar=disable_pbar,
        segment_offset=segment_offset,
        total_video_frames=total_video_frames,
        sigmas=low_sigmas_tensor,
    )

    # ---- 2. 插值放大 ----
    low_video, low_audio = nested_av_parts(low_latent)
    B, C, T, H, W = low_video.shape
    target_h = max(2, int(round(H * scale_by / 2)) * 2)
    target_w = max(2, int(round(W * scale_by / 2)) * 2)

    logger.info(
        "[YimoH3-PDD2Stage] upscale video latent: %dx%d -> %dx%d (scale=%.2f)",
        H, W, target_h, target_w, scale_by,
    )

    up_video = _upsample_video_latent_bilinear(
        low_video, target_h, target_w, low_video.dtype
    )
    upscaled_latent = {
        "samples": comfy.nested_tensor.NestedTensor((up_video, low_audio))
    }

    # v2.7.1: 释放 LOW 阶段中间态，避免长视频多段时显存累积
    del low_latent, low_video, low_audio
    _soft_empty_cache()

    # ---- 3. HIGH 阶段 ----
    high_sigmas_tensor = (
        clean_end_sigmas.to(up_video.device)
        if isinstance(clean_end_sigmas, torch.Tensor) else clean_end_sigmas
    )
    final_latent = _sample_single_segment(
        model=model,
        seed=seed,
        steps=n_high_steps,
        cfg=cfg,
        sampler_name=sampler_name,
        scheduler=scheduler,
        positive=positive,
        negative=negative,
        latent=upscaled_latent,
        denoise=denoise,
        callback=callback,
        disable_pbar=disable_pbar,
        segment_offset=segment_offset,
        total_video_frames=total_video_frames,
        sigmas=high_sigmas_tensor,
    )

    return final_latent


# =============================================================================
# Keyframe resolution
# =============================================================================

def _resolve_keyframes_absolute(keyframe_specs, frame_count, shift=0):
    if not keyframe_specs:
        return []

    result = []
    for spec in keyframe_specs:
        role = spec.get("role", "intermediate")
        rel = spec.get("relative_index", 0)

        if role == "first_frame":
            abs_pos = shift
        elif role == "last_frame":
            abs_pos = frame_count - 1
        elif role == "intermediate":
            if isinstance(rel, float) and 0 <= rel <= 1:
                abs_pos = round(rel * (frame_count - 1))
            else:
                abs_pos = int(rel)
            abs_pos += shift
        else:
            abs_pos = int(rel)
            if role != "last_frame":
                abs_pos += shift

        abs_pos = max(0, min(abs_pos, frame_count - 1))
        if 0 < abs_pos < frame_count - 1:
            abs_pos = align_keyframe_position(abs_pos, frame_count)
            if shift > 0 and abs_pos < shift:
                n = max(0, (shift - 5) // 17 + 1)
                candidate = 17 * n + 5
                if candidate < frame_count - 1:
                    abs_pos = candidate

        result.append({
            "resolved_frame_index": abs_pos,
            "latent": spec["latent"],
            "latent_h": spec["latent_h"],
            "latent_w": spec["latent_w"],
            "latent_t": spec["latent_t"],
        })
    return result


# =============================================================================
# Segment summary
# =============================================================================

def _extract_segment_summary(sampled, video_vae, audio_vae, mode, frame_count, render_frames=0, overlap_frames=0):
    video, audio = nested_av_parts(sampled)
    h, w = video.shape[-2] * 16, video.shape[-1] * 16

    actual_render_frames = render_frames if render_frames > 0 else frame_count
    tail_frame_pixel = None
    if video_vae is not None and video.shape[2] > 0:
        try:
            tail_latent = video[:, :, -1:, :, :]
            tail_frame_pixel = video_vae.decode(tail_latent)
            if isinstance(tail_frame_pixel, torch.Tensor) and tail_frame_pixel.ndim == 5:
                if tail_frame_pixel.shape[1] == 1:
                    tail_frame_pixel = tail_frame_pixel.squeeze(1)
                else:
                    tail_frame_pixel = tail_frame_pixel[:, 0]
        except Exception as e:
            logger.warning("Failed to decode tail frame: %s", e)

    tail_audio = None
    if audio_vae is not None and audio.shape[-1] > 0:
        try:
            tail_len = min(40, audio.shape[-1])
            tail_audio = audio[..., -tail_len:].detach().clone()
        except Exception:
            pass

    return SegmentSummary(
        mode=mode,
        canvas=(w, h),
        frame_count=frame_count,
        render_frames=actual_render_frames,
        tail_overlap_frames=overlap_frames,
        tail_frame_pixel=tail_frame_pixel,
        tail_audio_latent=tail_audio,
    )


def _build_prev_summary_from_latent(
    prev_latent,
    video_vae,
    audio_vae,
    mode: str = "text",
    frame_count: int = 124,
):
    if prev_latent is None:
        return None
    try:
        prev_video, prev_audio = nested_av_parts(prev_latent)
    except Exception as e:
        logger.debug("Failed to parse prev_sampled_latent: %s", e)
        return None

    prev_meta: dict = {}
    if isinstance(prev_latent, dict):
        meta = prev_latent.get("_yimo_prev_meta")
        if isinstance(meta, dict):
            prev_meta = meta

    tail_frame_pixel = None
    if video_vae is not None:
        try:
            tail_latent = prev_video[:, :, -1:, :, :]
            tail_frame_pixel = video_vae.decode(tail_latent)
            if isinstance(tail_frame_pixel, torch.Tensor) and tail_frame_pixel.ndim == 5:
                if tail_frame_pixel.shape[1] == 1:
                    tail_frame_pixel = tail_frame_pixel.squeeze(1)
                else:
                    tail_frame_pixel = tail_frame_pixel[:, 0]
        except Exception as e:
            logger.warning("Failed to decode prev tail frame: %s", e)

    tail_audio = None
    if audio_vae is not None and prev_audio.shape[-1] > 0:
        try:
            tail_len = min(40, prev_audio.shape[-1])
            tail_audio = prev_audio[..., -tail_len:].detach().clone()
        except Exception:
            pass

    prev_fc = get_video_frame_count_from_latent(prev_video)

    summary = SegmentSummary(
        mode=prev_meta.get("mode", mode),
        canvas=(prev_video.shape[-1] * 16, prev_video.shape[-2] * 16),
        frame_count=prev_meta.get("frame_count", prev_fc),
        render_frames=prev_meta.get("render_frames", prev_fc),
        tail_overlap_frames=0,
        tail_frame_pixel=tail_frame_pixel,
        tail_audio_latent=tail_audio,
    )
    summary.identity_refs = list(prev_meta.get("identity_refs", []))
    summary.style_refs = list(prev_meta.get("style_refs", []))
    summary.audio_policy = prev_meta.get("audio_policy", "generate_new")
    summary.prompt = prev_meta.get("prompt", "")
    return summary


# =============================================================================
# Transition application
# =============================================================================

def _apply_transition_to_conditioning(
    positive,
    keyframe_specs,
    frame_count,
    shift,
    auto_inherit_tail,
    prev_summary,
    video_vae,
    inherit_refs,
    identity_refs,
    style_refs,
    curr_mode: str = "text",
):
    if positive is None or not isinstance(positive, list) or len(positive) == 0:
        return positive, "no conditioning to modify"

    if not isinstance(positive[0], (list, tuple)) or len(positive[0]) < 2:
        return positive, "conditioning tuple malformed"

    cond_dict = positive[0][1]
    if not isinstance(cond_dict, dict):
        return positive, "conditioning dict missing"

    new_cond_dict = cond_dict.copy()
    reports: list[str] = []

    resolved_kfs = _resolve_keyframes_absolute(keyframe_specs, frame_count, shift=shift)

    existing_keyframes = cond_dict.get("minimax_keyframes") or []
    user_has_first_frame = (
        any(kf.get("role") == "first_frame" for kf in (keyframe_specs or []))
        or any(kf.get("resolved_frame_index") == 0 for kf in existing_keyframes)
        or any(kf.get("resolved_frame_index") == 0 for kf in resolved_kfs)
    )

    if (
        auto_inherit_tail
        and prev_summary is not None
        and prev_summary.tail_frame_pixel is not None
        and video_vae is not None
        and not user_has_first_frame
    ):
        try:
            tail_enc = video_vae.encode(prev_summary.tail_frame_pixel)
            inherited_kf = {
                "resolved_frame_index": 0,
                "latent": tail_enc,
                "latent_h": tail_enc.shape[-2],
                "latent_w": tail_enc.shape[-1],
                "latent_t": int(tail_enc.shape[2]),
            }
            resolved_kfs = [kf for kf in resolved_kfs if kf["resolved_frame_index"] != 0]
            resolved_kfs.insert(0, inherited_kf)
            resolved_kfs.sort(key=lambda x: x["resolved_frame_index"])
            reports.append("inherited tail frame as first_frame@0")
        except Exception as e:
            logger.warning("Failed to inherit tail frame: %s", e)
            reports.append(f"tail inherit failed: {e}")
    elif auto_inherit_tail and user_has_first_frame and prev_summary is not None:
        reports.append("user-provided first_frame takes priority over auto tail inheritance")

    if inherit_refs:
        support_refs = curr_mode in {"references", "hybrid"}
        if support_refs:
            existing_refs = new_cond_dict.get("minimax_refs", [])
            merged_refs = []
            if identity_refs:
                merged_refs.extend(_isolate_ref_dicts(identity_refs))
                reports.append(f"inherited {len(identity_refs)} identity refs")
            if style_refs:
                merged_refs.extend(_isolate_ref_dicts(style_refs))
                reports.append(f"inherited {len(style_refs)} style refs")
            if merged_refs:
                new_cond_dict["minimax_refs"] = list(existing_refs) + merged_refs
        else:
            reports.append(f"refs inheritance skipped: mode={curr_mode} does not support refs")

    if resolved_kfs:
        new_cond_dict["minimax_keyframes"] = resolved_kfs
        new_cond_dict["minimax_frame_count"] = frame_count

    if shift > 0:
        reports.append(f"keyframe_shift={shift}")

    new_positive = [[positive[0][0], new_cond_dict]] + list(positive[1:])
    return new_positive, "; ".join(reports) if reports else "no modifications"


# =============================================================================
# Overlap resolution / concatenation
# =============================================================================

def _resolve_pair_overlaps(overlap_frames, n_pairs):
    if isinstance(overlap_frames, int):
        v = max(0, int(overlap_frames))
        return [v] * n_pairs
    else:
        raw = list(overlap_frames)
        pair_overlaps = []
        for o in raw[:n_pairs]:
            pair_overlaps.append(max(0, int(o)))
        while len(pair_overlaps) < n_pairs:
            pair_overlaps.append(pair_overlaps[-1] if pair_overlaps else 0)
        return pair_overlaps[:n_pairs]


def _concatenate_av_latents(
    samples_list: list,
    video_overlaps: Sequence[int],
    audio_overlaps: Sequence[int] | None = None,
) -> dict:
    if not samples_list:
        raise ValueError("No latents to concatenate")
    if len(samples_list) == 1:
        s = samples_list[0]
        return {"samples": s["samples"] if isinstance(s, dict) else s}

    n_pairs = len(samples_list) - 1

    if audio_overlaps is None:
        audio_overlaps = video_overlaps

    video_pair = _resolve_pair_overlaps(video_overlaps, n_pairs)
    audio_pair = _resolve_pair_overlaps(audio_overlaps, n_pairs)

    video_parts = []
    audio_parts = []

    for i, sample in enumerate(samples_list):
        if isinstance(sample, dict):
            sample = sample.get("samples")

        if getattr(sample, "is_nested", False):
            parts = tuple(sample.unbind())
            video, audio = parts[0], parts[1]
        else:
            raise ValueError(f"Expected NestedTensor, got {type(sample)}")

        if i == 0:
            video_parts.append(video)
            audio_parts.append(audio)
        else:
            v_olap = video_pair[i - 1]
            a_olap = audio_pair[i - 1]

            if v_olap == 0:
                video_parts.append(video)
            else:
                _, v_latent_t, _ = _overlap_to_latent(v_olap)
                actual_v_t = min(v_latent_t, video.shape[2])
                if actual_v_t >= video.shape[2]:
                    logger.warning(
                        f"Segment {i} video latent_t ({video.shape[2]}) <= overlap_latent_t "
                        f"({v_latent_t}); cannot trim. Using hard concat for this segment's video."
                    )
                    video_parts.append(video)
                else:
                    video_parts.append(video[:, :, actual_v_t:, :, :])

            if a_olap == 0:
                audio_parts.append(audio)
            else:
                _, _, a_latent_t = _overlap_to_latent(a_olap)
                actual_a_t = min(a_latent_t, audio.shape[-1])
                if actual_a_t >= audio.shape[-1]:
                    logger.warning(
                        f"Segment {i} audio latent_t ({audio.shape[-1]}) <= overlap_audio_t "
                        f"({a_latent_t}); cannot trim. Using hard concat for this segment's audio."
                    )
                    audio_parts.append(audio)
                else:
                    audio_parts.append(audio[:, :, :, actual_a_t:])

    final_video = torch.cat(video_parts, dim=2)
    final_audio = torch.cat(audio_parts, dim=-1)

    return {"samples": comfy.nested_tensor.NestedTensor((final_video, final_audio))}


# =============================================================================
# Main sequence sampling
# =============================================================================

def sequence_sample(
    model,
    segments: list[dict],
    overlap_frames: int | Sequence[int],
    sampler_name: str,
    scheduler: str,
    steps: int,
    cfg: float,
    seed: int,
    denoise: float = 1.0,
    transition_mode: str = "context_inject",
    audio_continuity: str = "context_inject",
    callback: Callable | None = None,
    video_vae=None,
    audio_vae=None,
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
    sigmas=None,
    color_match_mode: str = "off",
    color_match_strength: float = 0.7,
    color_match_reference_frames: int = 1,
    sampling_mode: str = "standard",
    pdd_split_step: int = 4,
    pdd_scale_by: float = 1.5,
) -> tuple[dict, list[str], list[dict], list[SegmentSummary]]:
    if not segments:
        raise ValueError("At least one segment is required")

    use_pdd_two_stage = (
        sampling_mode == "pdd_two_stage"
        and sigmas is not None
        and isinstance(sigmas, torch.Tensor)
        and len(sigmas) > 2
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
        "=== Yimo H3 Sequence Sampler v3.0.0 ===",
        f"total_segments={total_segments}",
        f"effective_segments={effective_total_segments}",
        f"pair_overlaps(requested)={pair_overlaps}",
        f"pair_overlaps(effective 17n+5)={pair_effective}",
        f"total_video_frames={total_video_frames}",
        f"transition_mode(video)={transition_mode}",
        f"audio_continuity={audio_continuity}",
        f"auto_inherit_tail={auto_inherit_tail}",
        f"inherit_identity={inherit_identity}",
        f"resample_segment={resample_segment}",
        f"stop_after_segment={stop_after_segment}",
        f"face_restore={face_restore}",
        f"enable_ref_curve={enable_ref_curve}",
        f"sampling_mode={sampling_mode}",
        f"external_sigmas={'yes' if sigmas is not None else 'no'}",
        f"pdd_two_stage={'active' if use_pdd_two_stage else 'inactive'}",
        f"pdd_split_step={pdd_split_step}",
        f"pdd_scale_by={pdd_scale_by}",
        f"color_match_mode={color_match_mode}",
        f"color_match_strength={color_match_strength}",
        f"color_match_reference_frames={color_match_reference_frames}",
        f"chain_id={chain_id}",
        f"sampler={sampler_name}/{scheduler}",
        f"steps_per_segment={steps}, total_steps={total_steps}, cfg={cfg}, denoise={denoise}, base_seed={seed}",
    ]

    device = comfy.model_management.intermediate_device()

    sampled_segments = []
    segment_summaries = []
    prev_context = None
    prev_summary = None
    start_idx = 0
    cumulative_frame_offset = 0

    if resample_segment >= 0 and chain_id:
        if resample_segment >= total_segments:
            logger.warning("resample_segment %d >= total_segments %d, ignoring", resample_segment, total_segments)
            resample_segment = -1
        else:
            cache_loaded = True
            for ci in range(resample_segment):
                data = _load_segment_cache(chain_id, ci, device)
                if data is None:
                    cache_loaded = False
                    logger.warning("Segment cache missing for index %d, falling back to full resample", ci)
                    break
                sampled_segments.append(data["sampled"])
                segment_summaries.append(data["summary"])
                cumulative_frame_offset += segments[ci].get("render_frames", segments[ci].get("frame_count", 124))
                if ci < len(pair_effective):
                    cumulative_frame_offset -= pair_effective[ci]

            if cache_loaded and resample_segment > 0:
                prev_summary = segment_summaries[-1]
                if resample_segment < total_segments:
                    next_olap = pair_overlaps[resample_segment - 1] if resample_segment - 1 < len(pair_overlaps) else 0
                    need_context = (
                        next_olap > 0
                        and (transition_mode == "context_inject" or audio_continuity == "context_inject")
                    )
                    if need_context:
                        prev_context = _extract_tail_context(sampled_segments[-1], next_olap)
                start_idx = resample_segment
                report_lines.append(f"Loaded {resample_segment} cached segments from chain '{chain_id}', resampling from segment {resample_segment}")
            elif resample_segment == 0:
                start_idx = 0
                report_lines.append(f"Resampling from segment 0 (full resample)")

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
            positive,
            kf_specs,
            frame_count,
            plan.keyframe_shift,
            plan.auto_inherit_tail,
            prev_summary,
            video_vae,
            plan.inherit_refs,
            prev_identity_refs,
            prev_style_refs,
            curr_mode=mode,
        )
        report_lines.append(f"transition: {trans_report}")

        latent, ctx_report = _inject_context(
            latent,
            prev_context,
            video_continuity=transition_mode,
            audio_continuity=audio_continuity,
        )
        report_lines.append(ctx_report)

        seg_callback = None
        if callback is not None or previewer is not None:
            def _make_seg_callback(seg_idx: int, base_idx: int):
                _step_start = [time.time()]
                def _cb(step: int, x0, x, seg_total_steps: int):
                    now = time.time()
                    elapsed = now - _step_start[0]
                    _step_start[0] = now
                    global_step = (seg_idx - base_idx) * steps + step + 1
                    global_step = min(global_step, total_steps)
                    logger.info(
                        "[YimoH3SequenceSampler] segment %d step %d/%d (global %d/%d, %.2fs/step)",
                        seg_idx, step + 1, seg_total_steps, global_step, total_steps, elapsed,
                    )
                    preview = None
                    if previewer is not None:
                        try:
                            if hasattr(x0, "is_nested") and x0.is_nested:
                                parts = x0.unbind()
                                video_latent = parts[0]
                                preview = previewer.decode_latent_to_preview_image(video_latent)
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

        if use_pdd_two_stage:
            sampled = _sample_single_segment_pdd_two_stage(
                model=model,
                seed=seg_seed,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent=latent,
                full_sigmas=sigmas,
                split_step=pdd_split_step,
                scale_by=pdd_scale_by,
                denoise=denoise,
                callback=seg_callback,
                disable_pbar=False,
                segment_offset=seg_offset_arg,
                total_video_frames=total_frames_arg,
            )
        else:
            sampled = _sample_single_segment(
                model, seg_seed, steps, cfg,
                sampler_name, scheduler,
                positive, negative, latent,
                denoise=denoise,
                callback=seg_callback,
                disable_pbar=False,
                segment_offset=seg_offset_arg,
                total_video_frames=total_frames_arg,
                sigmas=sigmas,
            )

        if face_restore and face_bank and video_vae is not None:
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

        sampled_segments.append(sampled)

        summary = _extract_segment_summary(
            sampled, video_vae, audio_vae, mode, frame_count, render_frames, current_overlap
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
            cache_path = _save_segment_cache(chain_id, i, sampled, summary)

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
                and (transition_mode == "context_inject" or audio_continuity == "context_inject")
            )
            prev_context = _extract_tail_context(sampled, next_olap) if need_context else None
        else:
            prev_context = None

        # v2.7.1: 段间显存清理
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

    report_lines.append(f"=== final output: latent_t={total_latent_t}, estimated_frames={total_frames} ===")

    return final_latent, report_lines, sampled_segments, segment_summaries