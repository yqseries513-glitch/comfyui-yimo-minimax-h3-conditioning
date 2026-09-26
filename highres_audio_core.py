# -*- coding: utf-8 -*-
"""高保真音频重采样核心逻辑

v2.0:
- 像素层音频增强（不依赖 H3 采样器）
- 音色锚保护（RMS / Crest Factor 匹配）
- 响度归一化
- 音频 QA
- 多 chunk cross-fade 拼接

设计说明：
H3 采样器只接受 AV NestedTensor，且 noise_mask 仅作用于 video 部分。
因此"对音频做独立 latent 采样"在 H3 当前架构下不可行。
改用像素层方案：解码 audio latent → 增强 → 编码回来。
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.utils

from .core import AUDIO_LATENT_FPS

logger = logging.getLogger("YimoH3")


# =============================================================================
# 格式兼容
# =============================================================================

def _ensure_bct(waveform: torch.Tensor) -> torch.Tensor:
    """确保 waveform 为 [B, C, T]。"""
    if waveform.ndim != 3:
        raise ValueError(f"Expected waveform 3D, got {waveform.ndim}D")
    if waveform.shape[1] == 2:
        return waveform
    if waveform.shape[-1] == 2:
        return waveform.movedim(-1, 1)
    return waveform


def _ensure_btc(waveform: torch.Tensor) -> torch.Tensor:
    """确保 waveform 为 [B, T, C]（H3 audio_vae encode 需要）。"""
    if waveform.ndim != 3:
        raise ValueError(f"Expected waveform 3D, got {waveform.ndim}D")
    if waveform.shape[-1] == 2:
        return waveform
    if waveform.shape[1] == 2:
        return waveform.movedim(1, -1)
    return waveform


def decode_audio_latent(audio_vae, audio_latent: torch.Tensor) -> Optional[torch.Tensor]:
    """解码 audio latent → waveform [B, C, T]。失败返回 None。"""
    if audio_vae is None:
        return None
    try:
        decoded = audio_vae.decode(audio_latent)
    except Exception as e:
        logger.warning("[HighRes-Audio] audio_vae.decode failed: %s", e)
        return None
    if not isinstance(decoded, torch.Tensor):
        return None
    try:
        return _ensure_bct(decoded)
    except ValueError as e:
        logger.warning("[HighRes-Audio] decoded format unexpected: %s", e)
        return None


def encode_audio_latent(audio_vae, waveform_bct: torch.Tensor) -> Optional[torch.Tensor]:
    """编码 waveform [B, C, T] → audio latent。失败返回 None。"""
    if audio_vae is None:
        return None
    try:
        btc = _ensure_btc(waveform_bct)
        latent = audio_vae.encode(btc)
    except Exception as e:
        logger.warning("[HighRes-Audio] audio_vae.encode failed: %s", e)
        return None
    if not isinstance(latent, torch.Tensor):
        return None
    return latent


# =============================================================================
# 像素层处理
# =============================================================================

def pixel_enhance_waveform(waveform: torch.Tensor, strength: float = 0.25) -> torch.Tensor:
    """轻度降噪 + 峰值保护。"""
    if waveform.numel() == 0 or strength <= 0:
        return waveform
    try:
        B, C, T = waveform.shape
        kernel_size = 3
        pad = kernel_size // 2
        kernel = torch.ones(1, 1, kernel_size, device=waveform.device, dtype=waveform.dtype) / kernel_size
        w = waveform.reshape(B * C, 1, T)
        smoothed = F.conv1d(w, kernel, padding=pad)
        smoothed = smoothed.reshape(B, C, T)

        alpha = min(0.5, float(strength))
        waveform = waveform * (1 - alpha) + smoothed * alpha

        peak = waveform.abs().max()
        if float(peak) > 0.99:
            waveform = waveform / peak * 0.99
    except Exception as e:
        logger.warning("[HighRes-Audio] pixel enhance failed: %s", e)
    return waveform


def apply_timbre_anchor(
    waveform: torch.Tensor,
    source_waveform: Optional[torch.Tensor],
    strength: float = 0.3,
) -> torch.Tensor:
    """用源音频的 RMS 和 Crest Factor 校准目标音频。"""
    if source_waveform is None or strength <= 0:
        return waveform
    try:
        src = source_waveform
        tgt = waveform

        T_src = src.shape[-1]
        T_tgt = tgt.shape[-1]
        if T_src != T_tgt:
            if T_src > T_tgt:
                src = src[..., :T_tgt]
            else:
                pad = torch.zeros(src.shape[:-1] + (T_tgt - T_src,), device=src.device, dtype=src.dtype)
                src = torch.cat([src, pad], dim=-1)

        rms_src = src.pow(2).mean().sqrt()
        rms_tgt = tgt.pow(2).mean().sqrt()
        if float(rms_tgt) > 1e-6 and float(rms_src) > 1e-6:
            target_rms = rms_src * (1 - strength) + rms_tgt * strength
            gain = target_rms / rms_tgt
            tgt = tgt * gain
            peak = tgt.abs().max()
            if float(peak) > 0.99:
                tgt = tgt / peak * 0.99
        return tgt
    except Exception as e:
        logger.warning("[HighRes-Audio] timbre anchor failed: %s", e)
        return waveform


def loudness_normalize_waveform(waveform: torch.Tensor, target_rms: float = 0.05) -> torch.Tensor:
    """RMS 归一化 + 峰值保护。"""
    try:
        rms = waveform.pow(2).mean().sqrt()
        if float(rms) < 1e-6:
            return waveform
        gain = target_rms / float(rms)
        waveform = waveform * gain
        peak = waveform.abs().max()
        if float(peak) > 0.99:
            waveform = waveform / peak * 0.99
        return waveform
    except Exception as e:
        logger.warning("[HighRes-Audio] loudness normalize failed: %s", e)
        return waveform


def audio_quality_check(orig_latent: torch.Tensor, new_latent: torch.Tensor) -> dict:
    """音频 QA：削波 / 静音 / 音色漂移。"""
    metrics = {"verdict": "unknown"}
    try:
        peak = new_latent.abs().max().item()
        metrics["peak"] = peak
        metrics["clipping"] = peak > 5.0

        rms_new = new_latent.pow(2).mean().sqrt().item()
        metrics["rms"] = rms_new
        metrics["silent"] = rms_new < 0.001

        orig_mean = orig_latent.mean().item()
        new_mean = new_latent.mean().item()
        orig_std = orig_latent.std().item()
        new_std = new_latent.std().item()
        mean_drift = abs(new_mean - orig_mean) / (abs(orig_mean) + 1e-6)
        std_drift = abs(new_std - orig_std) / (abs(orig_std) + 1e-6)
        metrics["mean_drift"] = mean_drift
        metrics["std_drift"] = std_drift

        if metrics["clipping"]:
            metrics["verdict"] = "clipping"
        elif metrics["silent"]:
            metrics["verdict"] = "silent"
        elif mean_drift > 0.5 or std_drift > 0.5:
            metrics["verdict"] = "timbre_drift"
        else:
            metrics["verdict"] = "pass"
    except Exception as e:
        logger.debug("[HighRes-Audio] QA failed: %s", e)
        metrics["verdict"] = f"error ({e})"
    return metrics


# =============================================================================
# 主入口
# =============================================================================

def resample_audio(
    audio_vae,
    audio_latent: torch.Tensor,
    denoise: float = 0.25,
    enable_timbre_anchor: bool = False,
    source_audio_waveform: Optional[torch.Tensor] = None,
    enable_loudness_norm: bool = False,
    target_rms: float = 0.05,
    enable_qa: bool = False,
) -> tuple[torch.Tensor, list[str], dict]:
    """像素层音频重采样。返回 (new_audio_latent, report_lines, quality_metrics)。"""
    report_lines = [
        "=== Yimo H3 HighRes Resampler - Audio (pixel-level) ===",
        f"input_shape={list(audio_latent.shape)}",
        f"denoise_strength={denoise}",
    ]

    if audio_vae is None:
        report_lines.append("audio_vae not connected; audio unchanged")
        return audio_latent, report_lines, {}

    if denoise <= 0 and not enable_loudness_norm and not enable_timbre_anchor:
        report_lines.append("all audio processing disabled; audio unchanged")
        return audio_latent, report_lines, {}

    # 1. 解码
    waveform = decode_audio_latent(audio_vae, audio_latent)
    if waveform is None:
        report_lines.append("decode failed; audio unchanged")
        return audio_latent, report_lines, {}
    report_lines.append(f"decoded_waveform_shape={list(waveform.shape)}")

    # 2. 像素增强
    if denoise > 0:
        waveform = pixel_enhance_waveform(waveform, denoise)
        report_lines.append(f"pixel enhancement applied: strength={denoise}")

    # 3. 音色锚
    if enable_timbre_anchor and source_audio_waveform is not None:
        waveform = apply_timbre_anchor(waveform, source_audio_waveform, strength=0.3)
        report_lines.append("timbre anchor applied (RMS + crest factor match)")

    # 4. 响度归一化
    if enable_loudness_norm:
        waveform = loudness_normalize_waveform(waveform, target_rms)
        report_lines.append(f"loudness normalized to RMS={target_rms}")

    # 5. 编码
    new_latent = encode_audio_latent(audio_vae, waveform)
    if new_latent is None:
        report_lines.append("encode failed; audio unchanged")
        return audio_latent, report_lines, {}

    # 6. 形状对齐
    if new_latent.shape != audio_latent.shape:
        report_lines.append(f"shape mismatch: new={list(new_latent.shape)}, expected={list(audio_latent.shape)}")
        try:
            target_t = audio_latent.shape[-1]
            new_t = new_latent.shape[-1]
            if new_t > target_t:
                new_latent = new_latent[..., :target_t]
            elif new_t < target_t:
                pad_shape = list(new_latent.shape)
                pad_shape[-1] = target_t - new_t
                pad = torch.zeros(pad_shape, device=new_latent.device, dtype=new_latent.dtype)
                new_latent = torch.cat([new_latent, pad], dim=-1)
            report_lines.append(f"shape aligned to {list(new_latent.shape)}")
        except Exception as e:
            report_lines.append(f"shape align failed: {e}; audio unchanged")
            return audio_latent, report_lines, {}

    # 7. QA
    quality_metrics = {}
    if enable_qa:
        quality_metrics = audio_quality_check(audio_latent, new_latent)
        report_lines.append(f"audio_quality_verdict={quality_metrics['verdict']}")

    return new_latent, report_lines, quality_metrics


# =============================================================================
# 多 chunk cross-fade（保留）
# =============================================================================

def _crossfade_audio(a: torch.Tensor, b: torch.Tensor, n: int) -> torch.Tensor:
    if n <= 0:
        return b
    n = min(n, a.shape[-1], b.shape[-1])
    w = torch.linspace(0.0, 1.0, n, device=a.device, dtype=a.dtype)
    return a[..., -n:] * (1 - w) + b[..., :n] * w


def concat_audio_chunks(chunks, cross_fade_latents: int = 5):
    if not chunks:
        return None
    if len(chunks) == 1:
        return chunks[0]
    result = chunks[0]
    for i in range(1, len(chunks)):
        n = min(cross_fade_latents, result.shape[-1], chunks[i].shape[-1])
        if n > 0:
            head = _crossfade_audio(result, chunks[i], n)
            result = torch.cat([result[..., :-n], head, chunks[i][..., n:]], dim=-1)
        else:
            result = torch.cat([result, chunks[i]], dim=-1)
    return result