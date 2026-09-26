# -*- coding: utf-8 -*-
"""颜色匹配工具（共享模块）

从 sampler_core.py 和 highres_resampler_core.py 抽离，
避免同一逻辑维护两份。

v2.6.4:
- 统一入口：apply_color_match
- 5 档模式：off / low / medium / high / max
- 依赖 video_vae 用于像素级匹配
- 音频不再做无谓 clone

算法出处：
  - match_color_latent_mean / match_color_latent_mean_std:
      Reinhard et al., "Color Transfer between Images"
  - match_color_pixel_histogram: 经典直方图匹配
  - match_color_pixel_mkl: MKL 线性色彩迁移的鲁棒统计近似
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger("YimoH3")


# =============================================================================
# latent 层匹配
# =============================================================================

def match_color_latent_mean(
    current_video: torch.Tensor,
    reference_video: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    """latent 层逐通道均值对齐。"""
    ref_mean = reference_video.mean(dim=(0, 2, 3, 4), keepdim=True)
    cur_mean = current_video.mean(dim=(0, 2, 3, 4), keepdim=True)
    delta = (ref_mean - cur_mean) * float(strength)
    return current_video + delta


def match_color_latent_mean_std(
    current_video: torch.Tensor,
    reference_video: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    """latent 层均值+标准差对齐（Reinhard）。"""
    eps = 1e-6
    ref_mean = reference_video.mean(dim=(0, 2, 3, 4), keepdim=True)
    ref_std = reference_video.std(dim=(0, 2, 3, 4), keepdim=True) + eps
    cur_mean = current_video.mean(dim=(0, 2, 3, 4), keepdim=True)
    cur_std = current_video.std(dim=(0, 2, 3, 4), keepdim=True) + eps
    matched = (current_video - cur_mean) / cur_std * ref_std + ref_mean
    s = float(strength)
    return current_video * (1.0 - s) + matched * s


# =============================================================================
# 像素层匹配
# =============================================================================

def _compute_channel_histogram_cdf(values: torch.Tensor, n_bins: int = 256) -> torch.Tensor:
    """用 bincount 代替 histc，避免 CUDA 兼容性问题。"""
    scaled = (values.clamp(0.0, 1.0) * (n_bins - 1)).round().long()
    hist = torch.bincount(scaled, minlength=n_bins).to(values.dtype)
    cdf = torch.cumsum(hist, dim=0)
    total = cdf[-1]
    if total > 0:
        cdf = cdf / total
    else:
        cdf = torch.linspace(0.0, 1.0, n_bins, device=values.device, dtype=values.dtype)
    return cdf


def _monotonize_cdf(cdf: torch.Tensor) -> torch.Tensor:
    """保证 CDF 严格单调不减。"""
    return torch.cummax(cdf, dim=0).values


def match_color_pixel_histogram(
    current_frames: torch.Tensor,
    reference_frames: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    """逐通道直方图匹配。"""
    if current_frames.numel() == 0 or reference_frames.numel() == 0:
        return current_frames

    n_bins = 256
    cur_flat = current_frames.reshape(-1, current_frames.shape[-1])
    ref_flat = reference_frames.reshape(-1, reference_frames.shape[-1])

    matched_channels = []
    for c in range(cur_flat.shape[-1]):
        cur_c = cur_flat[:, c]
        ref_c = ref_flat[:, c]

        cur_cdf = _monotonize_cdf(_compute_channel_histogram_cdf(cur_c, n_bins))
        ref_cdf = _monotonize_cdf(_compute_channel_histogram_cdf(ref_c, n_bins))

        cur_cdf_cpu = cur_cdf.detach().cpu()
        ref_cdf_cpu = ref_cdf.detach().cpu()
        mapped_idx = torch.searchsorted(ref_cdf_cpu.contiguous(), cur_cdf_cpu.contiguous())
        mapped_idx = torch.clamp(mapped_idx, 0, n_bins - 1)

        bin_centers = torch.linspace(0.0, 1.0, n_bins, device=cur_c.device, dtype=cur_c.dtype)
        lut = bin_centers[mapped_idx.to(cur_c.device)]

        cur_scaled = (cur_c.clamp(0.0, 1.0) * (n_bins - 1)).round().long().clamp(0, n_bins - 1)
        mapped = lut[cur_scaled]

        matched_channels.append(mapped)

    matched_flat = torch.stack(matched_channels, dim=-1)
    matched = matched_flat.reshape(current_frames.shape)

    s = float(strength)
    return current_frames * (1.0 - s) + matched * s


def match_color_pixel_mkl(
    current_frames: torch.Tensor,
    reference_frames: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    """MKL 线性色彩迁移的鲁棒统计近似。"""
    if current_frames.numel() == 0 or reference_frames.numel() == 0:
        return current_frames

    cur_flat = current_frames.reshape(-1, current_frames.shape[-1])
    ref_flat = reference_frames.reshape(-1, reference_frames.shape[-1])

    matched_channels = []
    eps = 1e-6
    for c in range(cur_flat.shape[-1]):
        cur_c = cur_flat[:, c]
        ref_c = ref_flat[:, c]

        cur_c_cpu = cur_c.detach().float().cpu()
        ref_c_cpu = ref_c.detach().float().cpu()

        cur_med = cur_c_cpu.median()
        ref_med = ref_c_cpu.median()
        cur_q25 = cur_c_cpu.quantile(0.25)
        cur_q75 = cur_c_cpu.quantile(0.75)
        ref_q25 = ref_c_cpu.quantile(0.25)
        ref_q75 = ref_c_cpu.quantile(0.75)

        cur_iqr = (cur_q75 - cur_q25).item() + eps
        ref_iqr = (ref_q75 - ref_q25).item() + eps

        a = ref_iqr / cur_iqr
        b = ref_med.item() - a * cur_med.item()

        matched_channels.append(a * cur_c + b)

    matched_flat = torch.stack(matched_channels, dim=-1)
    matched = matched_flat.reshape(current_frames.shape)

    s = float(strength)
    return current_frames * (1.0 - s) + matched * s


# =============================================================================
# 辅助函数
# =============================================================================

def extract_tail_frames(video: torch.Tensor, n_frames: int) -> torch.Tensor:
    """从视频 latent [1, C, T, H, W] 中取尾部 N 个 latent 时间步。"""
    if video.ndim != 5:
        raise ValueError(f"Expected video latent [1,C,T,H,W], got {tuple(video.shape)}")
    t = video.shape[2]
    n = max(1, min(int(n_frames), t))
    return video[:, :, -n:, :, :].contiguous()


def decode_video_frames_for_match(
    video_latent: torch.Tensor,
    video_vae,
) -> torch.Tensor | None:
    """将视频 latent 解码为像素帧 [T, H, W, C]，值域 [0, 1]。失败返回 None。"""
    if video_vae is None:
        return None
    try:
        decoded = video_vae.decode(video_latent)
    except Exception as e:
        logger.warning("[ColorMatch] video_vae.decode failed: %s", e)
        return None

    if not isinstance(decoded, torch.Tensor):
        return None

    if decoded.ndim == 5:
        if decoded.shape[0] == 1:
            decoded = decoded.squeeze(0)
        elif decoded.shape[1] == 1:
            decoded = decoded.squeeze(1)
    if decoded.ndim == 4:
        if decoded.shape[-1] in (1, 3, 4):
            pass
        elif decoded.shape[0] in (1, 3, 4):
            decoded = decoded.permute(1, 2, 3, 0)
        else:
            return None
    else:
        return None

    if decoded.shape[-1] > 3:
        decoded = decoded[..., :3]
    return decoded.clamp(0.0, 1.0)


# =============================================================================
# 统一入口
# =============================================================================

def apply_color_match(
    sampled: dict,
    reference: dict | None,
    mode: str,
    strength: float,
    reference_frames: int,
    video_vae=None,
) -> tuple[dict, dict]:
    """
    颜色匹配统一入口。

    返回: (new_sampled, info)
    """
    import comfy.nested_tensor
    from .core import nested_av_parts

    info = {
        "mode": mode,
        "strength": float(strength),
        "reference_frames": int(reference_frames),
        "status": "unknown",
        "reference_source": "none",
        "matched_frames": 0,
        "match_level": "none",
    }

    if mode == "off" or strength <= 0.0:
        info["status"] = "disabled"
        return sampled, info

    if reference is None:
        info["status"] = "skipped (no reference)"
        return sampled, info

    try:
        cur_video, cur_audio = nested_av_parts(sampled)
        ref_video, _ = nested_av_parts(reference)
    except Exception as e:
        info["status"] = f"failed (parse: {e})"
        return sampled, info

    if cur_video.shape[-2:] != ref_video.shape[-2:]:
        info["status"] = (
            f"skipped (canvas mismatch: "
            f"cur={tuple(cur_video.shape[-2:])} ref={tuple(ref_video.shape[-2:])})"
        )
        return sampled, info

    ref_tail = extract_tail_frames(ref_video, reference_frames)
    info["reference_source"] = "reference tail"
    info["reference_shape"] = list(ref_tail.shape)

    if mode == "low":
        new_video = match_color_latent_mean(cur_video, ref_tail, strength)
        info["status"] = "applied"
        info["match_level"] = "latent (mean)"
        info["matched_frames"] = int(cur_video.shape[2])
    elif mode == "medium":
        new_video = match_color_latent_mean_std(cur_video, ref_tail, strength)
        info["status"] = "applied"
        info["match_level"] = "latent (mean+std)"
        info["matched_frames"] = int(cur_video.shape[2])
    elif mode in ("high", "max"):
        if video_vae is None:
            new_video = match_color_latent_mean_std(cur_video, ref_tail, strength)
            info["status"] = "applied (fallback to medium, no video_vae)"
            info["match_level"] = "latent (mean+std, fallback)"
            info["matched_frames"] = int(cur_video.shape[2])
        else:
            n_match = max(1, min(int(reference_frames), int(cur_video.shape[2])))
            cur_head = cur_video[:, :, :n_match, :, :]
            cur_head_pixels = decode_video_frames_for_match(cur_head, video_vae)
            ref_tail_pixels = decode_video_frames_for_match(ref_tail, video_vae)
            if cur_head_pixels is None or ref_tail_pixels is None:
                new_video = match_color_latent_mean_std(cur_video, ref_tail, strength)
                info["status"] = "applied (fallback to medium, decode failed)"
                info["match_level"] = "latent (mean+std, fallback)"
                info["matched_frames"] = int(cur_video.shape[2])
            else:
                if mode == "high":
                    matched_pixels = match_color_pixel_histogram(
                        cur_head_pixels, ref_tail_pixels, strength
                    )
                else:
                    matched_pixels = match_color_pixel_mkl(
                        cur_head_pixels, ref_tail_pixels, strength
                    )
                try:
                    matched_latent = video_vae.encode(matched_pixels)
                except Exception as e:
                    logger.warning("[ColorMatch] video_vae.encode failed: %s", e)
                    new_video = match_color_latent_mean_std(cur_video, ref_tail, strength)
                    info["status"] = "applied (fallback to medium, encode failed)"
                    info["match_level"] = "latent (mean+std, fallback)"
                    info["matched_frames"] = int(cur_video.shape[2])
                else:
                    new_video = cur_video.clone()
                    n_lat = min(matched_latent.shape[2], new_video.shape[2])
                    new_video[:, :, :n_lat, :, :] = matched_latent[:, :, :n_lat, :, :].to(
                        device=new_video.device, dtype=new_video.dtype
                    )
                    info["status"] = "applied"
                    info["match_level"] = f"pixel ({mode})"
                    info["matched_frames"] = int(n_lat)
    else:
        info["status"] = f"skipped (unknown mode: {mode})"
        return sampled, info

    out = sampled.copy()
    # v2.6.3: 音频未被修改，直接复用而非 clone
    out["samples"] = comfy.nested_tensor.NestedTensor((new_video, cur_audio))
    return out, info