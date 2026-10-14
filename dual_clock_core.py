# -*- coding: utf-8 -*-
"""Yimo H3 AV 双时钟采样核心

来源与授权
----------
双时钟数学与 KSAMPLER 注册思路参考：
  T8star-Aix / comfyui-minimax-h3-audio-T8 (GPL-3.0-or-later)
  https://github.com/T8mars/comfyui-minimax-h3-audio-T8
  文件：h3_t8/sampling.py

本文件包含基于该项目的衍生实现。
原版权归 T8star-Aix 及贡献者所有；修改部分版权归 yimo 所有。
本插件整体按 GPL-3.0-or-later 分发。
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch

import comfy.model_management
import comfy.model_sampling
import comfy.nested_tensor
import comfy.samplers
import comfy.utils
from comfy.k_diffusion.sampling import to_d

from .core import nested_av_parts

logger = logging.getLogger("YimoH3")

logger.info("[DualClock] dual_clock_core.py v5.0.0 loaded")


DEFAULT_SAMPLER_NAME = "dual_clock_euler"
DEFAULT_SCHEDULER_NAME = "native_flow"
BETA57_SCHEDULER_NAME = "beta57"
BETA57_ALPHA = 0.5
BETA57_BETA = 0.7

DEFAULT_SHIFT_VIDEO = 12.0
DEFAULT_SHIFT_AUDIO = 3.0

SAMPLER_OPTIONS: list[str] = [DEFAULT_SAMPLER_NAME]
if hasattr(comfy.model_sampling, "ModelSamplingAV"):
    SAMPLER_OPTIONS.extend(
        name
        for name in getattr(comfy.samplers, "SAMPLER_NAMES", [])
        if name != DEFAULT_SAMPLER_NAME
    )

SCHEDULER_OPTIONS: list[str] = [
    DEFAULT_SCHEDULER_NAME,
    BETA57_SCHEDULER_NAME,
    *(
        name
        for name in getattr(comfy.samplers, "SCHEDULER_NAMES", [])
        if name not in {DEFAULT_SCHEDULER_NAME, BETA57_SCHEDULER_NAME}
    ),
]


def shift_sigma(base_sigma, shift: float):
    return shift * base_sigma / (1.0 + (shift - 1.0) * base_sigma)


def time_shift_sigma(sigma, from_shift: float, to_shift: float):
    base_sigma = sigma / (from_shift + sigma * (1.0 - from_shift))
    return shift_sigma(base_sigma, to_shift)


def time_shift_slope(sigma, from_shift: float, to_shift: float):
    base_sigma = sigma / (from_shift + sigma * (1.0 - from_shift))
    numerator = to_shift * (1.0 + (from_shift - 1.0) * base_sigma) ** 2
    denominator = from_shift * (1.0 + (to_shift - 1.0) * base_sigma) ** 2
    return numerator / denominator


def native_flow_sigmas(steps: int, shift_video: float) -> torch.Tensor:
    base_sigmas = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)
    return shift_sigma(base_sigmas, shift_video)


def build_dual_clock_sigmas(
    steps: int,
    shift_video: float = DEFAULT_SHIFT_VIDEO,
    shift_audio: float = DEFAULT_SHIFT_AUDIO,
    scheduler: str = DEFAULT_SCHEDULER_NAME,
    model_sampling=None,
) -> torch.Tensor:
    if scheduler == DEFAULT_SCHEDULER_NAME:
        return native_flow_sigmas(steps, shift_video)

    if scheduler == BETA57_SCHEDULER_NAME:
        beta_scheduler = getattr(comfy.samplers, "beta_scheduler", None)
        if beta_scheduler is None:
            raise RuntimeError(
                "beta57 需要较新版 ComfyUI 自带的 beta_scheduler；"
                "请更新 ComfyUI 后重试。"
            )
        if model_sampling is None:
            raise ValueError("beta57 scheduler 需要 model_sampling 参数")
        return beta_scheduler(
            model_sampling,
            steps,
            alpha=BETA57_ALPHA,
            beta=BETA57_BETA,
        ).cpu()

    if scheduler not in getattr(comfy.samplers, "SCHEDULER_NAMES", []):
        raise ValueError(
            f"未知的 scheduler: {scheduler!r}；可用选项: "
            f"{', '.join(SCHEDULER_OPTIONS)}"
        )
    if model_sampling is None:
        raise ValueError(f"scheduler={scheduler} 需要 model_sampling 参数")
    return comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps).cpu()


class MiniMaxH3FlowSampling(
    comfy.model_sampling.ModelSamplingDiscreteFlow,
    comfy.model_sampling.CONST,
):
    @property
    def audio_scale(self):
        return 1.0

    @property
    def audio_shift(self):
        return DEFAULT_SHIFT_AUDIO


def model_uses_raw_audio_velocity(model) -> bool:
    base_model = getattr(model, "model", None)
    return callable(getattr(base_model, "audio_scale", None))


def _diagnose_audio_velocity_protocol(model) -> dict:
    base_model = getattr(model, "model", None)
    has_attr = hasattr(base_model, "audio_scale")
    is_callable = callable(getattr(base_model, "audio_scale", None))
    return {
        "base_model_type": type(base_model).__name__ if base_model is not None else "None",
        "has_audio_scale_attr": has_attr,
        "audio_scale_is_callable": is_callable,
        "audio_velocity_is_raw": is_callable,
        "protocol_hint": (
            "new (post-2026-08-06, raw audio velocity)"
            if is_callable
            else "legacy (audio velocity is scaled by slope)"
        ),
    }


def _make_sampling(
    model,
    original_sampling,
    shift_video: float,
    shift_audio: float,
    use_native_av: bool,
):
    if use_native_av:
        native_av_cls = getattr(comfy.model_sampling, "ModelSamplingAV", None)
        if native_av_cls is None or not model_uses_raw_audio_velocity(model):
            raise RuntimeError(
                "所选 ComfyUI 采样器需要原生 MiniMax H3 FLOW_AV 支持。"
                "请更新 ComfyUI，或将 sampler_name 改为 dual_clock_euler。"
            )

        class MiniMaxH3NativeAVSampling(
            native_av_cls, comfy.model_sampling.CONST
        ):
            pass

        model_sampling = MiniMaxH3NativeAVSampling(model.model.model_config)
        model_sampling.set_parameters(shift=shift_video, audio_shift=shift_audio)
    else:
        model_sampling = MiniMaxH3FlowSampling(model.model.model_config)
        model_sampling.set_parameters(shift=shift_video)

    if hasattr(original_sampling, "noise_scale"):
        try:
            model_sampling.set_noise_scale(original_sampling.noise_scale)
        except Exception as e:
            logger.debug("[DualClock] set_noise_scale failed: %s", e)
    return model_sampling


def _audio_step_scale(
    sigma_video,
    sigma_audio,
    slope_audio,
    denoise_mask,
    audio_velocity_is_raw: bool,
):
    flat_scale = -sigma_video
    dual_scale = (
        -sigma_audio if audio_velocity_is_raw else -sigma_audio / slope_audio
    )
    if denoise_mask is None:
        return dual_scale
    return flat_scale + denoise_mask * (dual_scale - flat_scale)


def _rebase_partial_audio_start(
    model,
    x: torch.Tensor,
    sigma_video,
    *,
    video_values: int,
    shift_video: float,
    shift_audio: float,
) -> torch.Tensor:
    noise = getattr(model, "noise", None)
    latent_image = getattr(model, "latent_image", None)
    if not isinstance(noise, torch.Tensor) or not isinstance(latent_image, torch.Tensor):
        return x
    if noise.shape != x.shape or latent_image.shape != x.shape:
        return x

    sigma_video_t = torch.as_tensor(sigma_video, device=x.device, dtype=x.dtype)
    sigma_audio_t = torch.as_tensor(
        time_shift_sigma(sigma_video_t, shift_video, shift_audio),
        device=x.device,
        dtype=x.dtype,
    )
    if bool(
        torch.isclose(
            sigma_video_t, sigma_audio_t, rtol=0.0, atol=1.0e-7
        ).all()
    ):
        return x
    if bool((sigma_video_t <= 0.0).any()):
        raise ValueError(
            "MiniMax H3 partial audio restart 需要正的初始 sigma"
        )

    audio_x = x[..., video_values:]
    audio_latent = latent_image[..., video_values:].to(
        device=x.device, dtype=x.dtype
    )
    sigma_shape = sigma_video_t.shape[:1] + (1,) * (audio_x.ndim - 1)
    sigma_video_r = sigma_video_t.reshape(sigma_shape)
    sigma_audio_r = sigma_audio_t.reshape(sigma_shape)

    scaled_noise = (
        audio_x - (1.0 - sigma_video_r) * audio_latent
    ) / sigma_video_r
    rebased_audio = (
        sigma_audio_r * scaled_noise + (1.0 - sigma_audio_r) * audio_latent
    )
    return torch.cat((x[..., :video_values], rebased_audio), dim=-1)


def sample_minimax_h3_dual_clock_euler(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    *,
    video_values: int,
    packed_values: int,
    shift_video: float,
    shift_audio: float,
    audio_velocity_is_raw: bool = False,
):
    extra_args = {} if extra_args is None else extra_args
    if x.shape[-1] != packed_values:
        raise ValueError(
            "MiniMax H3 packed latent 在 sampler setup 后发生变化："
            f"expected {packed_values}, got {x.shape[-1]}"
        )

    denoise_mask = extra_args.get("denoise_mask")
    audio_mask = None
    if denoise_mask is not None:
        if denoise_mask.shape[-1] != packed_values:
            raise ValueError(
                "MiniMax H3 denoise mask 与 packed AV latent 长度不匹配"
            )
        audio_mask = denoise_mask[..., video_values:]

    x = _rebase_partial_audio_start(
        model,
        x,
        sigmas[0],
        video_values=video_values,
        shift_video=shift_video,
        shift_audio=shift_audio,
    )

    s_in = x.new_ones([x.shape[0]])
    for step in comfy.utils.model_trange(len(sigmas) - 1, disable=disable):
        sigma_video = sigmas[step]
        sigma_video_next = sigmas[step + 1]
        denoised = model(x, sigma_video * s_in, **extra_args)
        derivative = to_d(x, sigma_video, denoised)

        sigma_audio = time_shift_sigma(sigma_video, shift_video, shift_audio)
        sigma_audio_next = time_shift_sigma(
            sigma_video_next, shift_video, shift_audio
        )
        slope_audio = time_shift_slope(sigma_video, shift_video, shift_audio)
        video_delta = sigma_video_next - sigma_video
        audio_delta = sigma_audio_next - sigma_audio
        if not audio_velocity_is_raw:
            audio_delta = audio_delta / slope_audio
        if audio_mask is not None:
            audio_delta = video_delta + audio_mask * (audio_delta - video_delta)

        if callback is not None:
            endpoint_scale = _audio_step_scale(
                sigma_video,
                sigma_audio,
                slope_audio,
                audio_mask,
                audio_velocity_is_raw,
            )
            denoised[..., video_values:] = (
                x[..., video_values:]
                + derivative[..., video_values:] * endpoint_scale
            )
            callback({
                "x": x,
                "i": step,
                "sigma": sigma_video,
                "sigma_hat": sigma_video,
                "denoised": denoised,
            })

        x = torch.cat((
            x[..., :video_values] + derivative[..., :video_values] * video_delta,
            x[..., video_values:] + derivative[..., video_values:] * audio_delta,
        ), dim=-1)
    return x


def _build_dual_clock_sampler(
    *,
    video_values: int,
    packed_values: int,
    shift_video: float,
    shift_audio: float,
    audio_velocity_is_raw: bool,
    extra_options=None,
    inpaint_options=None,
):
    def sampler_function(
        model_wrap, x, sigmas, extra_args=None, callback=None, disable=None
    ):
        return sample_minimax_h3_dual_clock_euler(
            model_wrap,
            x,
            sigmas,
            extra_args=extra_args,
            callback=callback,
            disable=disable,
            video_values=video_values,
            packed_values=packed_values,
            shift_video=shift_video,
            shift_audio=shift_audio,
            audio_velocity_is_raw=audio_velocity_is_raw,
        )

    sampler_function.__name__ = "sample_minimax_h3_dual_clock_euler"
    sampler_function._yimo_minimax_h3_video_values = int(video_values)
    sampler_function._yimo_minimax_h3_packed_values = int(packed_values)
    sampler_function._yimo_minimax_h3_shift_video = float(shift_video)
    sampler_function._yimo_minimax_h3_shift_audio = float(shift_audio)
    sampler_function._yimo_minimax_h3_audio_velocity_is_raw = bool(
        audio_velocity_is_raw
    )
    return comfy.samplers.KSAMPLER(
        sampler_function,
        extra_options={} if extra_options is None else dict(extra_options),
        inpaint_options={} if inpaint_options is None else dict(inpaint_options),
    )


def rebind_dual_clock_sampler(model, av_latent: dict, sampler):
    sampler_function = getattr(sampler, "sampler_function", None)
    if getattr(sampler_function, "__name__", None) != (
        "sample_minimax_h3_dual_clock_euler"
    ):
        return sampler

    video, audio = nested_av_parts(av_latent)
    if video.shape[1] != 24 or audio.shape[1] != 32 or audio.shape[2] != 2:
        raise ValueError(
            "重绑第二遍 sampler 时 AV 通道异常："
            f"video={tuple(video.shape)}, audio={tuple(audio.shape)}"
        )

    options = getattr(model, "model_options", {}).get("transformer_options", {})
    shift_video = getattr(
        sampler_function,
        "_yimo_minimax_h3_shift_video",
        options.get("yimo_minimax_h3_sigma_shift_video"),
    )
    shift_audio = getattr(
        sampler_function,
        "_yimo_minimax_h3_shift_audio",
        options.get("yimo_minimax_h3_sigma_shift_audio"),
    )
    if shift_video is None or shift_audio is None:
        raise ValueError(
            "双时钟 sampler 缺少 video/audio shift 元数据；"
            "请重新创建 Dual-Clock Sampler 后再做第二遍。"
        )

    return _build_dual_clock_sampler(
        video_values=math.prod(video.shape[1:]),
        packed_values=math.prod(video.shape[1:])
        + math.prod(audio.shape[1:]),
        shift_video=float(shift_video),
        shift_audio=float(shift_audio),
        audio_velocity_is_raw=bool(
            getattr(
                sampler_function,
                "_yimo_minimax_h3_audio_velocity_is_raw",
                model_uses_raw_audio_velocity(model),
            )
        ),
        extra_options=getattr(sampler, "extra_options", None),
        inpaint_options=getattr(sampler, "inpaint_options", None),
    )


def _extract_av_from_segment(segment_latent: dict) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(segment_latent, dict) or "samples" not in segment_latent:
        raise ValueError(
            "dual clock setup 需要一个 AV LATENT dict（含 'samples' 键）"
        )
    return nested_av_parts(segment_latent)


def setup_dual_clock_sampling(
    model,
    segment_latent: dict,
    steps: int,
    shift_video: float = DEFAULT_SHIFT_VIDEO,
    shift_audio: float = DEFAULT_SHIFT_AUDIO,
    sampler_name: str = DEFAULT_SAMPLER_NAME,
    scheduler: str = DEFAULT_SCHEDULER_NAME,
) -> tuple[Any, Any, torch.Tensor, dict, Any]:
    video, audio = _extract_av_from_segment(segment_latent)
    if video.shape[1] != 24 or audio.shape[1] != 32 or audio.shape[2] != 2:
        raise ValueError(
            "MiniMax H3 AV 通道异常："
            f"video={tuple(video.shape)}, audio={tuple(audio.shape)}"
        )

    if sampler_name not in SAMPLER_OPTIONS:
        raise ValueError(
            f"未知的 sampler: {sampler_name!r}；可用选项: "
            f"{', '.join(SAMPLER_OPTIONS)}"
        )

    use_native_av = sampler_name != DEFAULT_SAMPLER_NAME

    audio_velocity_is_raw = model_uses_raw_audio_velocity(model)

    original_sampling = model.get_model_object("model_sampling")
    model_sampling = _make_sampling(
        model,
        original_sampling,
        shift_video,
        shift_audio,
        use_native_av,
    )
    model.add_object_patch("model_sampling", model_sampling)

    transformer_options = model.model_options.get(
        "transformer_options", {}
    ).copy()
    transformer_options["yimo_minimax_h3_sigma_shift_video"] = shift_video
    transformer_options["yimo_minimax_h3_sigma_shift_audio"] = shift_audio
    model.model_options["transformer_options"] = transformer_options

    if use_native_av:
        sampler = comfy.samplers.sampler_object(sampler_name)
    else:
        video_values = math.prod(video.shape[1:])
        packed_values = video_values + math.prod(audio.shape[1:])
        sampler = _build_dual_clock_sampler(
            video_values=video_values,
            packed_values=packed_values,
            shift_video=shift_video,
            shift_audio=shift_audio,
            audio_velocity_is_raw=audio_velocity_is_raw,
        )

    sigmas = build_dual_clock_sigmas(
        steps=steps,
        shift_video=shift_video,
        shift_audio=shift_audio,
        scheduler=scheduler,
        model_sampling=model_sampling,
    )

    diagnostics = {
        "use_native_av": bool(use_native_av),
        "audio_velocity_is_raw": bool(audio_velocity_is_raw),
        "protocol": _diagnose_audio_velocity_protocol(model),
        "video_values": int(math.prod(video.shape[1:])),
        "packed_values": int(
            math.prod(video.shape[1:]) + math.prod(audio.shape[1:])
        ),
        "shift_video": float(shift_video),
        "shift_audio": float(shift_audio),
        "steps": int(steps),
        "sampler_name": str(sampler_name),
        "scheduler": str(scheduler),
        "sigmas_length": int(len(sigmas)),
        "sigma_first": float(sigmas[0].item()) if len(sigmas) > 0 else 0.0,
        "sigma_last": float(sigmas[-1].item()) if len(sigmas) > 0 else 0.0,
        "video_shape": list(video.shape),
        "audio_shape": list(audio.shape),
    }
    return model, sampler, sigmas, diagnostics, original_sampling


def run_dual_clock_sample(
    model,
    sampler,
    sigmas: torch.Tensor,
    positive,
    negative,
    latent: dict,
    cfg: float,
    seed: int,
    denoise: float = 1.0,
    noise_mask=None,
    callback=None,
    disable_pbar: bool = False,
    device=None,
    original_sampling=None,
):
    if device is None:
        device = comfy.model_management.get_torch_device()

    samples = latent.get("samples")
    if samples is None:
        raise ValueError("latent 缺少 'samples' 字段")

    noise = _make_noise_for_latent(latent, seed, device)

    try:
        sampled = comfy.samplers.sample(
            model=model,
            noise=noise,
            positive=positive,
            negative=negative,
            cfg=cfg,
            device=device,
            sampler=sampler,
            sigmas=sigmas,
            model_options=model.model_options,
            latent_image=samples,
            denoise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )
    finally:
        if original_sampling is not None:
            try:
                model.add_object_patch("model_sampling", original_sampling)
            except Exception as e:
                logger.warning(
                    "[DualClock] restore original_sampling failed: %s", e
                )
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            comfy.model_management.soft_empty_cache()
        except Exception:
            pass

    return sampled


def _make_noise_for_latent(latent: dict, seed: int, device):
    samples = latent.get("samples")
    if samples is None:
        raise ValueError("latent 缺少 'samples' 字段")

    generator = torch.Generator(device=device).manual_seed(int(seed))

    if getattr(samples, "is_nested", False):
        parts = tuple(samples.unbind())
        noise_parts = [
            torch.randn(p.shape, generator=generator, device=device, dtype=p.dtype)
            for p in parts
        ]
        return comfy.nested_tensor.NestedTensor(tuple(noise_parts))
    else:
        return torch.randn(
            samples.shape,
            generator=generator,
            device=device,
            dtype=samples.dtype,
        )