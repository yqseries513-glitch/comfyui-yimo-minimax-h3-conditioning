# 核心工具：画布 / 时间网格 / 音频 / 曲线 / 文本增强 / 输入归一化。
from __future__ import annotations

import math
import os
import re
import time
import logging
from collections.abc import Mapping
from typing import Any

import torch

import comfy.model_management
import comfy.nested_tensor
import comfy.utils


logger = logging.getLogger("YimoH3")

CANVAS_MULTIPLE = 32
REF_IMAGE_SHORT_EDGE = 2048
FPS = 24
AUDIO_LATENT_FPS = 40
MIN_TRAINED_FRAMES = 124
MAX_TRAINED_FRAMES = 362


# =============================================================================
# 统一路径管理函数
# =============================================================================

def get_yimo_output_dir(subdir: str = "") -> str:
    base_dir = os.path.join(os.getcwd(), "output", "yimo_h3")
    if subdir:
        base_dir = os.path.join(base_dir, subdir)
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def get_single_sampler_dir(project_name: str = "") -> str:
    subdir = "single_sampler"
    if project_name:
        subdir = os.path.join(subdir, sanitize_filename(project_name))
    return get_yimo_output_dir(subdir)


def get_sequence_sampler_cache_dir(chain_id: str = "") -> str:
    safe_id = sanitize_chain_id(chain_id) if chain_id else "default"
    return os.path.join(
        get_yimo_output_dir("sequence_sampler"),
        "_segment_cache",
        safe_id
    )


def get_sequence_sampler_export_dir(project_name: str = "") -> str:
    subdir = "sequence_sampler/exports"
    if project_name:
        subdir = os.path.join(subdir, sanitize_filename(project_name))
    return get_yimo_output_dir(subdir)


def sanitize_filename(name: str) -> str:
    if not name:
        return "default"
    return re.sub(r'[^a-zA-Z0-9_\-. ]', '_', str(name))[:64]


# =============================================================================
# 核心功能函数
# =============================================================================

def align_frame_count(frame_count: int) -> int:
    frame_count = max(5, int(frame_count))
    return frame_count + ((5 - frame_count) % 17)


def align_frame_count_down(frame_count: int) -> int:
    frame_count = int(frame_count)
    if frame_count < 5:
        return 5
    return frame_count - ((frame_count - 5) % 17)


def video_latent_t(frame_count: int) -> int:
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def temporal_shape(length: int) -> tuple[int, int, int]:
    frame_count = align_frame_count(length)
    duration = frame_count / FPS
    return frame_count, video_latent_t(frame_count), round(duration * AUDIO_LATENT_FPS)


def adapt_canvas(width: int, height: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive")
    return (
        max(CANVAS_MULTIPLE, round(width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
        max(CANVAS_MULTIPLE, round(height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
    )


def resize_image(image: torch.Tensor, width: int, height: int, crop: str = "disabled") -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError(f"Expected IMAGE [B,H,W,C], got {tuple(image.shape)}")
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def empty_av_latent(width: int, height: int, length: int) -> tuple[dict, int]:
    if width % 32 or height % 32:
        raise ValueError("MiniMax H3 width and height must be divisible by 32")
    frame_count, latent_t, audio_t = temporal_shape(length)
    device = comfy.model_management.intermediate_device()
    video = torch.zeros((1, 24, latent_t, height // 16, width // 16), device=device)
    audio = torch.zeros((1, 32, 2, audio_t), device=device)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count


def nested_av_parts(av_latent: dict) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(av_latent, dict) or "samples" not in av_latent:
        raise ValueError("Expected a MiniMax H3 joint AV LATENT")
    samples = av_latent["samples"]
    if not getattr(samples, "is_nested", False):
        raise ValueError("Expected a nested MiniMax H3 joint video/audio latent")
    parts = tuple(samples.unbind())
    if len(parts) != 2:
        raise ValueError(f"Expected exactly two AV latent parts, got {len(parts)}")
    video, audio = parts
    if video.ndim != 5 or audio.ndim != 4:
        raise ValueError(
            "Unexpected MiniMax H3 AV latent layout: "
            f"video={tuple(video.shape)}, audio={tuple(audio.shape)}"
        )
    if video.shape[0] != 1 or audio.shape[0] != 1:
        raise ValueError("MiniMax H3 currently supports batch size 1 only")
    return video, audio


def validate_audio(audio: dict, name: str = "audio") -> tuple[torch.Tensor, int]:
    if not isinstance(audio, dict):
        raise ValueError(f"{name} must be a connected AUDIO value")
    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if not isinstance(waveform, torch.Tensor) or sample_rate is None:
        raise ValueError(f"{name} is missing waveform or sample_rate")
    if waveform.ndim != 3:
        raise ValueError(f"{name} must use [batch,channels,samples], got {tuple(waveform.shape)}")
    if waveform.shape[0] < 1 or waveform.shape[1] < 1 or waveform.shape[2] < 1:
        raise ValueError(f"{name} is empty")
    if waveform.shape[0] != 1:
        raise ValueError(f"{name} must have batch size 1 for MiniMax H3")
    return waveform, int(sample_rate)


def encode_audio_once(audio_vae, audio: dict) -> torch.Tensor:
    try:
        import torchaudio
    except ImportError:
        raise ImportError(
            "torchaudio is required for audio encoding. "
            "Please install: pip install torchaudio"
        )

    waveform, sample_rate = validate_audio(audio)
    vae_sample_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if sample_rate != vae_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, vae_sample_rate)
    latent = audio_vae.encode(waveform[:1].movedim(1, -1))
    if not isinstance(latent, torch.Tensor) or latent.ndim != 4:
        raise ValueError("The audio VAE did not return [B,C,stereo,T] latent data")
    return latent


def fit_audio_latent(encoded_audio: torch.Tensor, template_audio: torch.Tensor) -> torch.Tensor:
    if encoded_audio.ndim != 4 or template_audio.ndim != 4:
        raise ValueError("MiniMax H3 audio latents must use [B,C,stereo,T]")
    if encoded_audio.shape[1:-1] != template_audio.shape[1:-1]:
        raise ValueError(
            "Audio VAE latent layout mismatch: "
            f"got {tuple(encoded_audio.shape)}, target {tuple(template_audio.shape)}"
        )
    if encoded_audio.shape[0] != template_audio.shape[0]:
        if encoded_audio.shape[0] == 1:
            encoded_audio = encoded_audio.expand(template_audio.shape[0], -1, -1, -1)
        else:
            raise ValueError("Audio latent batch cannot be matched to the AV latent")
    target_t = template_audio.shape[-1]
    if encoded_audio.shape[-1] > target_t:
        encoded_audio = encoded_audio[..., :target_t]
    elif encoded_audio.shape[-1] < target_t:
        padding = encoded_audio.new_zeros((*encoded_audio.shape[:-1], target_t - encoded_audio.shape[-1]))
        encoded_audio = torch.cat((encoded_audio, padding), dim=-1)
    return encoded_audio.to(device=template_audio.device, dtype=template_audio.dtype)


_SENTINEL_TYPE_NAMES = frozenset({
    "UnconnectedValue", "Undefined", "Null", "NoneType",
    "EmptyDict", "EmptyList", "EmptyTuple",
    "BypassedValue", "MutedValue", "IgnoredValue", "Placeholder",
    "DisconnectedValue", "SkippedValue", "Void", "NullTensor",
})


def is_connected_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (dict, list, tuple, str)) and len(value) == 0:
        return False
    type_name = type(value).__name__
    if type_name in _SENTINEL_TYPE_NAMES or any(s in type_name for s in ("Unconnected", "Undefined", "Bypass", "Mute", "Ignore", "Placeholder", "Disconnect", "Skip", "Void")):
        return False
    if isinstance(value, torch.Tensor):
        if value.numel() == 0 or any(dim == 0 for dim in value.shape):
            return False
        if value.numel() == 1 and float(value.abs().max()) == 0.0:
            logger.debug(
                "[is_connected_value] detected single-element zero tensor (shape=%s); "
                "treating as disconnected placeholder.", tuple(value.shape)
            )
            return False
    if isinstance(value, dict) and ("waveform" in value or "sample_rate" in value):
        waveform = value.get("waveform")
        sample_rate = value.get("sample_rate")
        if not isinstance(waveform, torch.Tensor) or sample_rate is None:
            return False
    return True


def sorted_autogrow_items(values: Mapping | None) -> list[tuple[int, object]]:
    if not values:
        return []

    def sort_key(item: tuple[str, Any]) -> int:
        key = str(item[0])
        try:
            return int(key.rsplit("_", 1)[-1])
        except ValueError:
            return 10_000

    output: list[tuple[int, object]] = []
    fallback_counter = 1
    for key, value in sorted(values.items(), key=sort_key):
        if not is_connected_value(value):
            continue
        try:
            ordinal = int(str(key).rsplit("_", 1)[-1])
        except ValueError:
            ordinal = fallback_counter
            fallback_counter += 1
        output.append((ordinal, value))
    return output


def sorted_autogrow_values(values: Mapping | None) -> list:
    return [value for _, value in sorted_autogrow_items(values)]


def split_noise_masks(av_latent: dict, video: torch.Tensor, audio: torch.Tensor):
    masks = av_latent.get("noise_mask")
    if masks is None:
        return None, None
    if getattr(masks, "is_nested", False):
        parts = tuple(masks.unbind())
        if len(parts) == 2:
            return parts
    if isinstance(masks, torch.Tensor):
        return masks, None
    raise ValueError("Unsupported AV noise_mask layout")


def replace_audio_latent(av_latent: dict, encoded_audio: torch.Tensor, denoise_strength: float) -> dict:
    video, template_audio = nested_av_parts(av_latent)
    fitted = fit_audio_latent(encoded_audio, template_audio)
    video_mask, _ = split_noise_masks(av_latent, video, template_audio)
    if video_mask is None:
        video_mask = torch.ones_like(video)
    audio_mask = torch.full_like(fitted, float(denoise_strength))
    output = av_latent.copy()
    output["samples"] = comfy.nested_tensor.NestedTensor((video, fitted))
    output["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
    return output


def align_keyframe_position(pos: int, frame_count: int) -> int:
    pos = int(pos)
    frame_count = int(frame_count)

    if pos <= 0:
        return 0
    if pos >= frame_count - 1:
        return frame_count - 1

    n = round((pos - 5) / 17)
    aligned = 17 * max(0, n) + 5

    max_n = max(0, (frame_count - 7) // 17)
    if aligned > frame_count - 2:
        aligned = 17 * max_n + 5

    return aligned


def max_intermediate_keyframes_for_frames(frame_count: int) -> int:
    frame_count = int(frame_count)
    if frame_count < 22:
        return 0
    return max(0, (frame_count - 7) // 17)


def sanitize_chain_id(chain_id: str) -> str:
    if not chain_id:
        return "default"
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", str(chain_id))
    return safe[:64]


def cleanup_expired_contexts(base_dir: str, max_age_hours: float = 24.0) -> int:
    if not os.path.isdir(base_dir):
        return 0
    now = time.time()
    max_age_seconds = max_age_hours * 3600.0
    removed = 0
    for root, _dirs, files in os.walk(base_dir):
        for fname in files:
            if not fname.endswith(".pt"):
                continue
            fpath = os.path.join(root, fname)
            try:
                mtime = os.path.getmtime(fpath)
                if now - mtime > max_age_seconds:
                    os.remove(fpath)
                    removed += 1
            except OSError:
                pass
    if removed:
        logger.info("Cleaned up %d expired context files from %s", removed, base_dir)
    return removed


# =============================================================================
# Image input normalization
# =============================================================================

def normalize_image_input(value: Any, name: str = "image") -> list[torch.Tensor]:
    if not is_connected_value(value):
        return []

    result: list[torch.Tensor] = []

    if isinstance(value, torch.Tensor):
        if value.ndim == 3:
            result.append(value.unsqueeze(0))
        elif value.ndim == 4:
            for i in range(value.shape[0]):
                result.append(value[i : i + 1])
        else:
            raise ValueError(
                f"{name} tensor must be 3D [H,W,C] or 4D [B,H,W,C], "
                f"got {value.ndim}D shape {tuple(value.shape)}"
            )

    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            if not isinstance(item, torch.Tensor):
                raise ValueError(
                    f"{name}[{idx}] must be a torch.Tensor, got {type(item).__name__}"
                )
            if item.ndim == 3:
                result.append(item.unsqueeze(0))
            elif item.ndim == 4:
                if item.shape[0] == 1:
                    result.append(item)
                else:
                    for j in range(item.shape[0]):
                        result.append(item[j : j + 1])
            else:
                raise ValueError(
                    f"{name}[{idx}] tensor must be 3D or 4D, "
                    f"got {item.ndim}D shape {tuple(item.shape)}"
                )
    else:
        raise ValueError(
            f"{name} must be an IMAGE tensor or a list/tuple of IMAGE tensors, "
            f"got {type(value).__name__}"
        )

    return result


def get_video_frame_count_from_latent(video: torch.Tensor) -> int:
    if video.shape[2] <= 2:
        return 5
    n = (video.shape[2] - 2) // 5
    return 17 * n + 5


# =============================================================================
# 参考强度时间曲线
# =============================================================================

REF_CURVE_DIRECTIONS = [
    "constant",
    "concept_at_start",
    "concept_at_end",
    "concept_at_middle",
    "concept_at_ends",
]

REF_CURVE_SHAPES = [
    "linear",
    "ease",
    "sigmoid",
    "exponential",
    "quadratic",
    "cubic",
]

REF_CURVE_DIRECTION_DESCRIPTIONS = {
    "constant": "全程恒定强度",
    "concept_at_start": "概念在前段出现（强度从高到低衰减）",
    "concept_at_end": "概念在后段出现（强度从低到高增强）",
    "concept_at_middle": "概念在中段出现（中间强，两端弱）",
    "concept_at_ends": "概念在两端出现（两端强，中间弱）",
}

REF_CURVE_SHAPE_DESCRIPTIONS = {
    "linear": "线性",
    "ease": "平滑（S曲线）",
    "sigmoid": "S型曲线",
    "exponential": "指数曲线",
    "quadratic": "二次曲线",
    "cubic": "三次曲线",
}

REF_CURVE_PRESETS: dict[str, dict[str, str] | None] = {
    "off":         {"direction": "constant",          "shape": "linear"},
    "fade_out":    {"direction": "concept_at_start",  "shape": "ease"},
    "fade_in":     {"direction": "concept_at_end",    "shape": "ease"},
    "peak_middle": {"direction": "concept_at_middle", "shape": "sigmoid"},
    "peak_ends":   {"direction": "concept_at_ends",   "shape": "sigmoid"},
    "custom":      None,
}

REF_CURVE_PRESET_OPTIONS = list(REF_CURVE_PRESETS.keys())


def resolve_ref_curve(
    preset: str,
    custom_direction: str = "constant",
    custom_shape: str = "linear",
) -> tuple[str, str]:
    if preset == "custom":
        d = custom_direction if custom_direction in REF_CURVE_DIRECTIONS else "constant"
        s = custom_shape if custom_shape in REF_CURVE_SHAPES else "linear"
        return d, s
    cfg = REF_CURVE_PRESETS.get(preset)
    if cfg is None:
        return "constant", "linear"
    return cfg["direction"], cfg["shape"]


def compute_reference_curve(
    frame_index: int,
    total_frames: int,
    direction: str = "constant",
    shape: str = "linear",
    base_strength: float = 1.0,
) -> float:
    if total_frames <= 1:
        return max(0.0, min(1.0, base_strength))

    t = frame_index / (total_frames - 1)

    if direction == "constant":
        envelope = 1.0
    elif direction == "concept_at_start":
        envelope = 1.0 - t
    elif direction == "concept_at_end":
        envelope = t
    elif direction == "concept_at_middle":
        envelope = 1.0 - abs(2.0 * t - 1.0)
    elif direction == "concept_at_ends":
        envelope = abs(2.0 * t - 1.0)
    else:
        envelope = 1.0

    if shape == "linear":
        pass
    elif shape == "ease":
        envelope = 3 * envelope ** 2 - 2 * envelope ** 3
    elif shape == "sigmoid":
        envelope = 1.0 / (1.0 + math.exp(-10.0 * (envelope - 0.5)))
    elif shape == "exponential":
        envelope = (math.exp(envelope) - 1.0) / (math.e - 1.0)
    elif shape == "quadratic":
        envelope = envelope ** 2
    elif shape == "cubic":
        envelope = envelope ** 3

    return max(0.0, min(1.0, base_strength * envelope))


# =============================================================================
# 参考强度语义化分级
# =============================================================================

REFERENCE_RETENTION_LEVELS = {
    "fully_preserved": {"value": 1.0, "description": "完全保留参考素材的所有特征"},
    "partially_preserved": {"value": 0.7, "description": "保留大部分特征，允许少量变化"},
    "attribute_transfer": {"value": 0.4, "description": "仅传递风格/属性，不保留身份"},
    "weak_reference": {"value": 0.15, "description": "弱参考，仅提供轻微引导"},
    "no_reference": {"value": 0.0, "description": "不使用参考"},
}


def get_retention_value(level: str) -> float:
    return REFERENCE_RETENTION_LEVELS.get(level, {}).get("value", 1.0)


# =============================================================================
# 文本信号增强
# =============================================================================

def amplify_text_conditioning(
    conditioning: list,
    strength: float = 2.0,
    mode: str = "deviation",
    renorm: bool = True,
) -> list:
    if strength == 1.0:
        return conditioning

    try:
        result = []
        for item in conditioning:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                result.append(item)
                continue
            cond_tensor, cond_dict = item[0], item[1]
            if not isinstance(cond_tensor, torch.Tensor):
                result.append(item)
                continue

            new_dict = dict(cond_dict) if isinstance(cond_dict, dict) else cond_dict

            if mode == "deviation":
                seq_mean = cond_tensor.mean(dim=1, keepdim=True)
                deviation = cond_tensor - seq_mean
                amplified = seq_mean + deviation * strength
            else:
                amplified = cond_tensor * strength

            if renorm:
                original_rms = cond_tensor.pow(2).mean().sqrt()
                amplified_rms = amplified.pow(2).mean().sqrt()
                if amplified_rms > 0:
                    amplified = amplified * (original_rms / amplified_rms)

            if isinstance(new_dict, dict):
                new_dict["_yimo_text_boost"] = {
                    "strength": strength,
                    "mode": mode,
                    "renorm": renorm,
                }

            result.append([amplified, new_dict])
        return result if result else conditioning
    except Exception as e:
        logger.warning("[YimoH3] Text conditioning amplification failed: %s", e)
        return conditioning