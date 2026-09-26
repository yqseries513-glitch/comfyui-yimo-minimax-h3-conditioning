# -*- coding: utf-8 -*-
"""高分辨率重采样核心逻辑（视频部分）

v2.7.1:
- tile 循环、时间 chunk 循环末尾加入 soft_empty_cache。

v2.7.0:
- resample_video 签名删除未使用的 face_images 参数。

v2.6.6:
- 修复 target_h / target_w 对齐逻辑：改为四舍五入到 2 倍数。

v2.6.5:
- 修复 PyTorch 2.12 下 F.interpolate 对 5D 输入报错的问题。
"""

from __future__ import annotations

import math
import logging

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.utils

from .core import (
    nested_av_parts,
    temporal_shape,
    align_frame_count,
    CANVAS_MULTIPLE,
    FPS,
    AUDIO_LATENT_FPS,
)

logger = logging.getLogger("YimoH3")


def _soft_empty_cache():
    """v2.7.1: 安全地请求 ComfyUI 清空缓存。"""
    try:
        comfy.model_management.soft_empty_cache()
    except Exception:
        pass


# =============================================================================
# H3 时间网格工具
# =============================================================================

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0


def frames_for_tokens(n: int) -> int:
    return sum(FRAME_PER_TOKEN[i % 5] for i in range(n))


def tokens_for_frames(f: int) -> int:
    n, acc = 0, 0
    while acc < f:
        acc += FRAME_PER_TOKEN[n % 5]
        n += 1
    return n


def compute_temporal_bounds(latent_t: int, chunk_frames: int, overlap_frames: int):
    chunk_frames = align_frame_count(chunk_frames)
    overlap_frames = align_frame_count(overlap_frames) if overlap_frames > 0 else 0

    chunk_tokens = tokens_for_frames(chunk_frames)
    overlap_tokens = tokens_for_frames(overlap_frames) if overlap_frames > 0 else 0

    if overlap_tokens >= chunk_tokens:
        overlap_tokens = max(0, chunk_tokens - 1)

    if chunk_tokens >= latent_t:
        return [(0, 0, latent_t, frames_for_tokens(latent_t))], frames_for_tokens(latent_t)

    hop = max(1, chunk_tokens - overlap_tokens)
    bounds = []
    prev_k0 = -1
    i = 0
    while True:
        k0 = i * hop
        if k0 + chunk_tokens >= latent_t:
            k1 = latent_t
            k0 = max(k0, latent_t - chunk_tokens)
            if prev_k0 >= 0 and k0 <= prev_k0:
                k0 = prev_k0 + 1
            if k0 >= latent_t:
                break
            bounds.append((k0, frames_for_tokens(k0), k1, frames_for_tokens(k1)))
            break
        bounds.append((k0, frames_for_tokens(k0), k0 + chunk_tokens, frames_for_tokens(k0 + chunk_tokens)))
        prev_k0 = k0
        i += 1

    return bounds, frames_for_tokens(latent_t)


# =============================================================================
# 空间分块网格
# =============================================================================

def _grid_1d(size: int, tile: int, ol: int):
    if size <= tile:
        return [0], [size], [0]
    sh = max(1, tile - ol)
    n = math.ceil((size - ol) / sh)
    if (n - 1) * sh + tile < size:
        n += 1
    starts = [i * sh for i in range(n)]
    if starts[-1] + tile > size:
        starts[-1] = size - tile
    sizes = [min(tile, size - s) for s in starts]
    ovls = [0] * n
    for i in range(1, n):
        ovls[i] = max(0, starts[i - 1] + sizes[i - 1] - starts[i])
    return starts, sizes, ovls


def compute_spatial_grid(h: int, w: int, tile_h: int, tile_w: int, ol_h: int, ol_w: int):
    rows, tile_hs, row_ovls = _grid_1d(h, tile_h, ol_h)
    cols, tile_ws, col_ovls = _grid_1d(w, tile_w, ol_w)
    return rows, cols, tile_hs, tile_ws, row_ovls, col_ovls


# =============================================================================
# 分阶段 sigma
# =============================================================================

def build_staged_sigmas(base_denoise: float = 0.5, stages: int = 3, device=None):
    if device is None:
        device = comfy.model_management.intermediate_device()

    if stages == 1:
        return torch.linspace(base_denoise, 0.0, 9, device=device)

    if stages == 3:
        s1 = torch.linspace(base_denoise, base_denoise * 0.7, 3, device=device)[:-1]
        s2 = torch.linspace(base_denoise * 0.7, base_denoise * 0.3, 3, device=device)[:-1]
        s3 = torch.linspace(base_denoise * 0.3, 0.0, 4, device=device)
        return torch.cat([s1, s2, s3])

    return torch.linspace(base_denoise, 0.0, 9, device=device)


# =============================================================================
# 5D latent 双线性上采样
# =============================================================================

def _upsample_video_latent_5d(video_latent: torch.Tensor, target_h: int, target_w: int, dtype) -> torch.Tensor:
    B, C, T, H, W = video_latent.shape
    video_bt = video_latent.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    upsampled_bt = F.interpolate(
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
# 全局面部检测
# =============================================================================

def detect_global_face_regions(
    video_latent: torch.Tensor,
    video_vae,
    det_threshold: float = 0.5,
    expand_ratio: float = 1.2,
):
    if video_vae is None:
        return []
    try:
        from .face_restore import detect_faces_insightface, detect_faces_opencv
        import numpy as np
        import cv2
    except ImportError:
        logger.debug("[HighRes] face detect dependencies missing")
        return []

    try:
        first = video_vae.decode(video_latent[:, :, :1])
        if first.ndim == 5:
            first = first[0]
        if first.shape[-1] > 3:
            first = first[..., :3]
        frame_np = (first[0].detach().cpu().numpy() * 255).astype(np.uint8)
        bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.debug("[HighRes] global face detection frame decode failed: %s", e)
        return []

    faces = []
    try:
        faces = detect_faces_insightface(bgr, det_threshold)
        if not faces:
            faces = detect_faces_opencv(bgr, det_threshold)
    except Exception as e:
        logger.debug("[HighRes] global face detection failed: %s", e)
        return []

    if not faces:
        return []

    H_latent = video_latent.shape[-2]
    W_latent = video_latent.shape[-1]
    H_pixel = bgr.shape[0]
    W_pixel = bgr.shape[1]

    regions = []
    for f in faces:
        bbox = f["bbox"]
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = (bbox[2] - bbox[0]) * expand_ratio
        h = (bbox[3] - bbox[1]) * expand_ratio
        y0 = max(0, int((cy - h / 2) * H_latent / H_pixel))
        y1 = min(H_latent, int((cy + h / 2) * H_latent / H_pixel))
        x0 = max(0, int((cx - w / 2) * W_latent / W_pixel))
        x1 = min(W_latent, int((cx + w / 2) * W_latent / W_pixel))
        if y1 > y0 and x1 > x0:
            regions.append((y0, y1, x0, x1))

    logger.info("[HighRes] global face regions: %d", len(regions))
    return regions


# =============================================================================
# AV NestedTensor 构造
# =============================================================================

def _make_av_nested(video_chunk: torch.Tensor, audio_chunk: torch.Tensor):
    return comfy.nested_tensor.NestedTensor((video_chunk, audio_chunk))


def _make_av_noise(latent_av, seed: int):
    device = comfy.model_management.get_torch_device()
    gen = torch.Generator(device=device).manual_seed(seed)
    parts = tuple(latent_av.unbind())
    noise_parts = [torch.randn(p.shape, generator=gen, device=device, dtype=p.dtype) for p in parts]
    return comfy.nested_tensor.NestedTensor(tuple(noise_parts))


def _make_placeholder_audio(frames: int, device, dtype):
    _, _, audio_t = temporal_shape(frames)
    return torch.zeros((1, 32, 2, audio_t), device=device, dtype=dtype)


def _extract_audio_slice(audio_latent, f0: int, f1: int, device, dtype):
    if audio_latent is None:
        return None
    a0 = int(round(f0 * FRAME_RESCALE))
    a1 = int(round(f1 * FRAME_RESCALE))
    a0 = max(0, min(a0, audio_latent.shape[-1]))
    a1 = max(a0, min(a1, audio_latent.shape[-1]))
    if a1 <= a0:
        return None
    return audio_latent[:, :, :, a0:a1].to(device=device, dtype=dtype).contiguous()


# =============================================================================
# 第二遍身份注入
# =============================================================================

def inject_identity_refs(positive, segment_bundle, model_refs_max: int = 9):
    if segment_bundle is None:
        return positive, 0
    try:
        cond_dict = segment_bundle[0][1] if isinstance(segment_bundle, list) else None
        if not isinstance(cond_dict, dict):
            return positive, 0

        yimo_data = cond_dict.get("_yimo_data", {})
        identity_refs = yimo_data.get("identity_refs", [])
        if not identity_refs:
            return positive, 0

        identity_refs = identity_refs[: min(len(identity_refs), 3)]

        new_positive = []
        for item in positive:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                new_positive.append(item)
                continue
            cond_tensor, cd = item[0], item[1]
            new_cd = dict(cd) if isinstance(cd, dict) else cd
            if isinstance(new_cd, dict):
                existing_refs = new_cd.get("minimax_refs", [])
                if not isinstance(existing_refs, list):
                    existing_refs = []
                existing_ids = {id(r) for r in existing_refs}
                for ref in identity_refs:
                    if id(ref) not in existing_ids:
                        existing_refs.append(ref)
                        existing_ids.add(id(ref))
                new_cd["minimax_refs"] = existing_refs[:model_refs_max]
            new_positive.append([cond_tensor, new_cd])

        return new_positive, len(identity_refs)
    except Exception as e:
        logger.warning("[HighRes] identity injection failed: %s", e)
        return positive, 0


# =============================================================================
# 颜色匹配
# =============================================================================

def color_match_to_reference(
    current_video: torch.Tensor,
    reference_video: torch.Tensor,
    mode: str = "medium",
    strength: float = 0.7,
) -> torch.Tensor:
    if mode == "off" or strength <= 0:
        return current_video

    from .color_match_utils import (
        match_color_latent_mean,
        match_color_latent_mean_std,
    )

    try:
        if mode == "low":
            return match_color_latent_mean(current_video, reference_video, strength)
        elif mode == "medium":
            return match_color_latent_mean_std(current_video, reference_video, strength)
        else:
            return match_color_latent_mean_std(current_video, reference_video, strength)
    except Exception as e:
        logger.warning("[HighRes] color match failed: %s", e)
        return current_video


# =============================================================================
# 质量检查
# =============================================================================

def quality_check_video(
    original_latent: torch.Tensor,
    resampled_latent: torch.Tensor,
    video_vae,
) -> dict:
    metrics = {"verdict": "unknown", "face_similarity": None, "color_drift": None}
    if video_vae is None:
        metrics["verdict"] = "skipped (no video_vae)"
        return metrics

    try:
        import numpy as np
        import cv2
        from .face_restore import detect_faces_insightface

        orig_frame = video_vae.decode(original_latent[:, :, :1])
        new_frame = video_vae.decode(resampled_latent[:, :, :1])
        if orig_frame.ndim == 5:
            orig_frame = orig_frame[0]
        if new_frame.ndim == 5:
            new_frame = new_frame[0]

        orig_np = (orig_frame[0, ..., :3].detach().cpu().numpy() * 255).astype(np.uint8)
        new_np = (new_frame[0, ..., :3].detach().cpu().numpy() * 255).astype(np.uint8)
        orig_bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
        new_bgr = cv2.cvtColor(new_np, cv2.COLOR_RGB2BGR)

        orig_faces = detect_faces_insightface(orig_bgr)
        new_faces = detect_faces_insightface(new_bgr)
        if orig_faces and new_faces:
            oe = orig_faces[0].get("embedding")
            ne = new_faces[0].get("embedding")
            if oe is not None and ne is not None:
                sim = float(np.dot(oe, ne) / (np.linalg.norm(oe) * np.linalg.norm(ne) + 1e-9))
                metrics["face_similarity"] = sim

        color_drift = float(np.abs(orig_np.astype(np.float32) - new_np.astype(np.float32)).mean() / 255.0)
        metrics["color_drift"] = color_drift

        if metrics["face_similarity"] is not None:
            if metrics["face_similarity"] < 0.5:
                metrics["verdict"] = "identity_drift"
            elif color_drift > 0.15:
                metrics["verdict"] = "color_drift"
            else:
                metrics["verdict"] = "pass"
        else:
            metrics["verdict"] = "no_face_detected"
    except Exception as e:
        logger.debug("[HighRes] quality check failed: %s", e)
        metrics["verdict"] = f"error ({e})"

    return metrics


# =============================================================================
# 采样核心：单 chunk 无分块
# =============================================================================

def _sample_chunk_single(
    chunk_video: torch.Tensor,
    chunk_audio: torch.Tensor | None,
    model,
    positive,
    negative,
    sampler_name: str,
    scheduler: str,
    cfg: float,
    seed: int,
    chunk_sigmas,
    noise_mask_video: torch.Tensor,
):
    device = chunk_video.device
    dtype = chunk_video.dtype

    if chunk_audio is None:
        _, _, audio_t = temporal_shape(frames_for_tokens(chunk_video.shape[2]))
        chunk_audio = torch.zeros((1, 32, 2, audio_t), device=device, dtype=dtype)

    av_latent = _make_av_nested(chunk_video, chunk_audio)
    av_noise = _make_av_noise(av_latent, seed)

    try:
        sampled = comfy.sample.sample(
            model=model,
            noise=av_noise,
            steps=max(len(chunk_sigmas) - 1, 1),
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=av_latent,
            noise_mask=noise_mask_video,
            denoise=1.0,
            seed=seed,
            sigmas=chunk_sigmas.to(device),
            disable_pbar=False,
        )
        if getattr(sampled, "is_nested", False):
            sampled_v = sampled.unbind()[0]
        else:
            sampled_v = sampled
        return sampled_v
    except Exception as e:
        logger.error("[HighRes] chunk sampling failed: %s", e)
        return chunk_video


# =============================================================================
# 采样核心：单 chunk 空间分块
# =============================================================================

def _build_tile_mask(
    tile_h: int,
    tile_w: int,
    T: int,
    ovh: int,
    ovw: int,
    done_top: bool,
    done_left: bool,
    base_denoise: float,
    face_denoise: float,
    face_regions: list,
    r0: int,
    c0: int,
    device,
    dtype,
):
    mask = torch.full((1, 1, T, tile_h, tile_w), base_denoise, device=device, dtype=torch.float32)

    if done_left and ovw > 0:
        fade = torch.linspace(0.0, 1.0, ovw, device=device, dtype=torch.float32)
        mask[..., :ovw] = fade.view(1, 1, 1, 1, ovw) * base_denoise

    if done_top and ovh > 0:
        fade = torch.linspace(0.0, 1.0, ovh, device=device, dtype=torch.float32)
        mask[..., :ovh, :] = torch.minimum(
            mask[..., :ovh, :],
            fade.view(1, 1, 1, ovh, 1) * base_denoise,
        )

    if face_denoise < base_denoise and face_regions:
        for (fy0, fy1, fx0, fx1) in face_regions:
            ty0 = max(0, fy0 - r0)
            ty1 = min(tile_h, fy1 - r0)
            tx0 = max(0, fx0 - c0)
            tx1 = min(tile_w, fx1 - c0)
            if ty1 > ty0 and tx1 > tx0:
                mask[..., ty0:ty1, tx0:tx1] = torch.minimum(
                    mask[..., ty0:ty1, tx0:tx1],
                    torch.tensor(face_denoise, device=device, dtype=torch.float32),
                )

    return mask


def _sample_chunk_tiled(
    chunk_video: torch.Tensor,
    chunk_audio: torch.Tensor | None,
    model,
    positive,
    negative,
    sampler_name: str,
    scheduler: str,
    cfg: float,
    seed: int,
    chunk_sigmas,
    base_denoise: float,
    face_denoise: float,
    face_regions: list,
    tile_size: int,
    tile_overlap: int,
    chunk_id: int,
):
    B, C, T, H, W = chunk_video.shape
    device = chunk_video.device
    dtype = chunk_video.dtype

    rows, cols, tile_hs, tile_ws, row_ovls, col_ovls = compute_spatial_grid(
        H, W, tile_size, tile_size, tile_overlap, tile_overlap
    )
    nrows = len(rows)
    ncols = len(cols)

    logger.info(
        "[HighRes] chunk %d spatial grid: %d x %d tiles (tile=%d, overlap=%d)",
        chunk_id, nrows, ncols, tile_size, tile_overlap,
    )

    output = chunk_video.clone()

    for ri in range(nrows):
        for cj in range(ncols):
            r0 = rows[ri]
            c0 = cols[cj]
            tr = tile_hs[ri]
            tc = tile_ws[cj]
            ovh = row_ovls[ri]
            ovw = col_ovls[cj]

            tile_in = output[:, :, :, r0:r0 + tr, c0:c0 + tc].clone()

            tile_mask = _build_tile_mask(
                tile_h=tr, tile_w=tc, T=T,
                ovh=ovh, ovw=ovw,
                done_top=(ri > 0), done_left=(cj > 0),
                base_denoise=base_denoise,
                face_denoise=face_denoise,
                face_regions=face_regions,
                r0=r0, c0=c0,
                device=device, dtype=dtype,
            )

            if chunk_audio is None:
                _, _, audio_t = temporal_shape(frames_for_tokens(T))
                tile_audio = torch.zeros((1, 32, 2, audio_t), device=device, dtype=dtype)
            else:
                tile_audio = chunk_audio

            av_latent = _make_av_nested(tile_in, tile_audio)
            av_noise = _make_av_noise(
                av_latent,
                seed + chunk_id * 1000000 + ri * 1000 + cj,
            )

            try:
                sampled = comfy.sample.sample(
                    model=model,
                    noise=av_noise,
                    steps=max(len(chunk_sigmas) - 1, 1),
                    cfg=cfg,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    positive=positive,
                    negative=negative,
                    latent_image=av_latent,
                    noise_mask=tile_mask,
                    denoise=1.0,
                    seed=seed,
                    sigmas=chunk_sigmas.to(device),
                    disable_pbar=False,
                )
                if getattr(sampled, "is_nested", False):
                    sampled_tile = sampled.unbind()[0]
                else:
                    sampled_tile = sampled
            except Exception as e:
                logger.error("[HighRes] tile (%d,%d) failed: %s", ri, cj, e)
                sampled_tile = tile_in

            output[:, :, :, r0:r0 + tr, c0:c0 + tc] = sampled_tile

            # v2.7.1: 每个 tile 结束后清理临时显存
            del tile_in, tile_mask, av_latent, av_noise, sampled_tile
            _soft_empty_cache()

    return output


# =============================================================================
# 主入口
# =============================================================================

def resample_video(
    model,
    positive,
    negative,
    video_latent: torch.Tensor,
    audio_latent: torch.Tensor | None,
    sampler_name: str,
    scheduler: str,
    sigmas,
    cfg: float,
    seed: int,
    scale: float,
    denoise_strategy: str,
    base_denoise: float,
    video_vae,
    segment_bundle,
    enable_temporal_chunking: bool,
    chunk_frames: int,
    chunk_overlap: int,
    enable_identity_injection: bool,
    face_denoise_reduction: float,
    color_match_mode: str,
    enable_quality_check: bool,
    enable_spatial_tiling: bool,
    tile_size_latent: int,
    tile_overlap_latent: int,
    previewer=None,
):
    """视频高分辨率重采样主逻辑。

    v2.7.1: 时间 chunk 循环末尾加入 soft_empty_cache。
    """
    B, C, T, H, W = video_latent.shape
    device = video_latent.device
    dtype = video_latent.dtype

    report_lines = [
        "=== Yimo H3 HighRes Resampler - Video v3.0.0 ===",
        f"input_shape={list(video_latent.shape)}",
        f"scale={scale}",
        f"denoise_strategy={denoise_strategy}",
        f"base_denoise={base_denoise}",
        f"spatial_tiling={'on' if enable_spatial_tiling else 'off'}",
    ]

    target_h_raw = H * scale
    target_w_raw = W * scale
    target_h = max(2, int(round(target_h_raw / 2)) * 2)
    target_w = max(2, int(round(target_w_raw / 2)) * 2)

    report_lines.append(
        f"target_latent_hw=({target_h}, {target_w}) "
        f"from_raw=({target_h_raw:.1f}, {target_w_raw:.1f})"
    )

    upsampled = _upsample_video_latent_5d(video_latent, target_h, target_w, dtype)
    report_lines.append(f"upsampled_shape={list(upsampled.shape)}")

    if enable_identity_injection and segment_bundle is not None:
        positive, n_inj = inject_identity_refs(positive, segment_bundle)
        report_lines.append(f"identity_injection: {n_inj} refs")

    if enable_temporal_chunking and T > tokens_for_frames(chunk_frames):
        bounds, _ = compute_temporal_bounds(T, chunk_frames, chunk_overlap)
        report_lines.append(f"temporal_chunks: {len(bounds)}")
    else:
        bounds = [(0, 0, T, frames_for_tokens(T))]
        report_lines.append("temporal_chunks: 1 (no chunking)")

    face_regions = []
    face_denoise = base_denoise
    if denoise_strategy == "face_aware":
        raw_face_regions = detect_global_face_regions(video_latent, video_vae)
        scale_h = target_h / max(1, H)
        scale_w = target_w / max(1, W)
        for (fy0, fy1, fx0, fx1) in raw_face_regions:
            ny0 = max(0, min(target_h, int(round(fy0 * scale_h))))
            ny1 = max(0, min(target_h, int(round(fy1 * scale_h))))
            nx0 = max(0, min(target_w, int(round(fx0 * scale_w))))
            nx1 = max(0, min(target_w, int(round(fx1 * scale_w))))
            if ny1 > ny0 and nx1 > nx0:
                face_regions.append((ny0, ny1, nx0, nx1))
        face_denoise = base_denoise * (1.0 - face_denoise_reduction)
        report_lines.append(
            f"face_regions={len(face_regions)} (scaled to upsampled latent "
            f"{target_h}x{target_w}), face_denoise={face_denoise:.3f}"
        )

    output_chunks = []
    quality_metrics = {}

    for idx, (k0, f0, k1, f1) in enumerate(bounds):
        chunk_v = upsampled[:, :, k0:k1].contiguous()
        chunk_audio = _extract_audio_slice(audio_latent, f0, f1, device, dtype)

        if denoise_strategy == "staged":
            chunk_sigmas = build_staged_sigmas(base_denoise, stages=3, device=device)
        else:
            chunk_sigmas = sigmas if sigmas is not None else build_staged_sigmas(base_denoise, stages=1, device=device)

        chunk_face_regions = face_regions
        if enable_spatial_tiling:
            chunk_out = _sample_chunk_tiled(
                chunk_video=chunk_v,
                chunk_audio=chunk_audio,
                model=model,
                positive=positive,
                negative=negative,
                sampler_name=sampler_name,
                scheduler=scheduler,
                cfg=cfg,
                seed=seed + idx * 999,
                chunk_sigmas=chunk_sigmas,
                base_denoise=base_denoise,
                face_denoise=face_denoise,
                face_regions=chunk_face_regions,
                tile_size=tile_size_latent,
                tile_overlap=tile_overlap_latent,
                chunk_id=idx,
            )
        else:
            C_v, T_v, H_v, W_v = chunk_v.shape[1:]
            video_mask = torch.full((1, 1, T_v, H_v, W_v), base_denoise, device=device, dtype=torch.float32)
            if denoise_strategy == "face_aware" and face_regions:
                for (fy0, fy1, fx0, fx1) in face_regions:
                    y0 = max(0, fy0)
                    y1 = min(H_v, fy1)
                    x0 = max(0, fx0)
                    x1 = min(W_v, fx1)
                    if y1 > y0 and x1 > x0:
                        video_mask[..., y0:y1, x0:x1] = face_denoise

            chunk_out = _sample_chunk_single(
                chunk_video=chunk_v,
                chunk_audio=chunk_audio,
                model=model,
                positive=positive,
                negative=negative,
                sampler_name=sampler_name,
                scheduler=scheduler,
                cfg=cfg,
                seed=seed + idx * 999,
                chunk_sigmas=chunk_sigmas,
                noise_mask_video=video_mask,
            )

        if color_match_mode != "off":
            ref_chunk = video_latent[:, :, k0:k1]
            ref_up = _upsample_video_latent_5d(ref_chunk, target_h, target_w, dtype)
            chunk_out = color_match_to_reference(chunk_out, ref_up, color_match_mode, 0.7)

        output_chunks.append(chunk_out)

        # v2.7.1: 每个时间 chunk 结束后清理
        del chunk_v, chunk_audio, chunk_sigmas
        _soft_empty_cache()

    if len(output_chunks) == 1:
        resampled = output_chunks[0]
    else:
        resampled = torch.cat(output_chunks, dim=2)

    if enable_quality_check:
        metrics = quality_check_video(video_latent, resampled, video_vae)
        quality_metrics = metrics
        report_lines.append(f"quality_verdict={metrics['verdict']}")
        if metrics.get("face_similarity") is not None:
            report_lines.append(f"face_similarity={metrics['face_similarity']:.4f}")
        if metrics.get("color_drift") is not None:
            report_lines.append(f"color_drift={metrics['color_drift']:.4f}")

    report_lines.append(f"output_shape={list(resampled.shape)}")

    return resampled, report_lines, quality_metrics