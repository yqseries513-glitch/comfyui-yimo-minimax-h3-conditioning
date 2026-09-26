# -*- coding: utf-8 -*-
"""Yimo H3 后处理分流/合流节点

YimoH3PostProcessSplit：
  输入 AV Latent，输出三种格式供用户按需连接：
  - 视频 Latent（潜空间超分方案的输入）
  - 音频 Latent（旁路，最终在合流器里与超分后的视频合并）
  - 图像帧序列（像素级超分方案的输入，需要 video_vae）

YimoH3PostProcessMerge：
  支持两种视频输入（latent / image frames，二选一）+ 原始音频 Latent，
  重新组装为 AV Latent，供 VAE Decode 或后续处理。

v2.6.1: Merge 输入端顺序调整为 video_latent 优先，audio_latent 加 optional，
        避免前端把 audio_latent 自动顶到第一行导致连线交叉。
"""

from __future__ import annotations

import json
import logging

import torch

import comfy.nested_tensor
from comfy_api.latest import io

from .core import nested_av_parts, is_connected_value, CANVAS_MULTIPLE, FPS
from .i18n import t

logger = logging.getLogger("YimoH3")
CATEGORY = t("category")


def _estimate_frame_count(latent_t: int) -> int:
    if latent_t <= 2:
        return 5
    n = (latent_t - 2) // 5
    return 17 * n + 5


def _decode_to_frames(video_vae, video_latent: torch.Tensor) -> torch.Tensor | None:
    """把 video latent 解码为 [T, H, W, C] 图像帧序列。失败返回 None。"""
    if video_vae is None:
        return None
    try:
        decoded = video_vae.decode(video_latent)
    except Exception as e:
        logger.warning("[YimoH3PostProcessSplit] video_vae.decode failed: %s", e)
        return None

    if not isinstance(decoded, torch.Tensor):
        return None

    if decoded.ndim == 5:
        if decoded.shape[0] == 1:
            decoded = decoded.squeeze(0)
        elif decoded.shape[1] == 1:
            decoded = decoded.squeeze(1)

    if decoded.ndim != 4:
        return None

    if decoded.shape[-1] in (1, 3, 4):
        pass
    elif decoded.shape[0] in (1, 3, 4):
        decoded = decoded.permute(1, 2, 3, 0)
    else:
        return None

    if decoded.shape[-1] > 3:
        decoded = decoded[..., :3]
    return decoded.clamp(0.0, 1.0)


def _extract_samples_tensor(value, expected_ndim: int, name: str) -> torch.Tensor:
    """从 LATENT dict / raw tensor 中提取 tensor。"""
    if isinstance(value, dict) and "samples" in value:
        tensor = value["samples"]
    elif isinstance(value, torch.Tensor):
        tensor = value
    else:
        raise ValueError(f"{name} 格式异常：{type(value).__name__}")

    if not isinstance(tensor, torch.Tensor) or tensor.ndim != expected_ndim:
        raise ValueError(
            f"{name} 必须是 {expected_ndim}D tensor，got "
            f"{type(tensor).__name__} ndim={getattr(tensor, 'ndim', '?')}"
        )
    return tensor


class YimoH3PostProcessSplit(io.ComfyNode):
    """后处理分流器：把 AV Latent 拆成三种输出格式。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YimoH3PostProcessSplit",
            display_name=t("post_process_split_display_name"),
            description=t("post_process_split_description"),
            category=CATEGORY,
            inputs=[
                io.Latent.Input("av_latent",
                    tooltip=t("post_process_split_av_latent_tooltip")),
                io.Vae.Input("video_vae", optional=True,
                    tooltip=t("post_process_split_video_vae_tooltip")),
            ],
            outputs=[
                io.Latent.Output(display_name=t("post_process_split_output_video_latent")),
                io.Latent.Output(display_name=t("post_process_split_output_audio_latent")),
                io.Image.Output(display_name=t("post_process_split_output_image_frames")),
                io.String.Output(display_name=t("post_process_split_output_metadata")),
            ],
        )

    @classmethod
    def execute(cls, av_latent, video_vae=None, **kwargs):
        if av_latent is None:
            raise ValueError(t("err_post_process_no_av_latent"))

        try:
            video, audio = nested_av_parts(av_latent)
        except Exception as e:
            raise ValueError(t("err_post_process_av_parse", error=e))

        # 输出纯 video / audio latent
        video_latent_out = {"samples": video.detach().clone()}
        audio_latent_out = {"samples": audio.detach().clone()}

        # 可选：解码图像帧序列
        image_frames = None
        decoded_ok = False
        if is_connected_value(video_vae):
            decoded_frames = _decode_to_frames(video_vae, video)
            if decoded_frames is not None:
                image_frames = decoded_frames
                decoded_ok = True

        # 元数据
        latent_t = int(video.shape[2])
        audio_t = int(audio.shape[-1])
        latent_h = int(video.shape[-2])
        latent_w = int(video.shape[-1])
        pixel_w = latent_w * 16
        pixel_h = latent_h * 16
        frame_count = _estimate_frame_count(latent_t)

        metadata = {
            "video_shape": list(video.shape),
            "audio_shape": list(audio.shape),
            "latent_t": latent_t,
            "audio_t": audio_t,
            "latent_h": latent_h,
            "latent_w": latent_w,
            "pixel_width": pixel_w,
            "pixel_height": pixel_h,
            "estimated_frame_count": frame_count,
            "canvas_multiple": CANVAS_MULTIPLE,
            "fps": FPS,
            "decoded_frames_available": decoded_ok,
        }
        if image_frames is not None:
            metadata["image_frames_shape"] = list(image_frames.shape)

        metadata_json = json.dumps(metadata, ensure_ascii=False, indent=2)

        logger.info(
            "[YimoH3PostProcessSplit] video=%s audio=%s frames=%s",
            tuple(video.shape), tuple(audio.shape),
            tuple(image_frames.shape) if image_frames is not None else "None",
        )

        return io.NodeOutput(
            video_latent_out,
            audio_latent_out,
            image_frames,
            metadata_json,
        )


class YimoH3PostProcessMerge(io.ComfyNode):
    """后处理合流器：把超分后的视频与原始音频合并为 AV Latent。

    v2.6.1: 输入端顺序调整
    - video_latent 置于第一行（连放大器输出时走直线，减少线缆交叉）
    - audio_latent 加 optional=True，避免前端把它当必填项顶到第一行
    - 逻辑仍会校验 audio_latent 必须连接，未连时抛错
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YimoH3PostProcessMerge",
            display_name=t("post_process_merge_display_name"),
            description=t("post_process_merge_description"),
            category=CATEGORY,
            inputs=[
                # v2.6.1: video_latent 放最上，与放大器输出端对齐
                io.Latent.Input("video_latent", optional=True,
                    tooltip=t("post_process_merge_video_latent_tooltip")),
                # v2.6.1: 加 optional=True，避免被前端自动提到最前
                io.Latent.Input("audio_latent", optional=True,
                    tooltip=t("post_process_merge_audio_latent_tooltip")),
                io.Image.Input("video_frames", optional=True,
                    tooltip=t("post_process_merge_video_frames_tooltip")),
                io.Vae.Input("video_vae", optional=True,
                    tooltip=t("post_process_merge_video_vae_tooltip")),
            ],
            outputs=[
                io.Latent.Output(display_name=t("post_process_merge_output_av_latent")),
                io.String.Output(display_name=t("post_process_merge_output_report")),
            ],
        )

    @classmethod
    def execute(cls, video_latent=None, audio_latent=None,
                video_frames=None, video_vae=None, **kwargs):

        report_lines = ["=== Yimo H3 PostProcess Merge ==="]

        has_latent = is_connected_value(video_latent)
        has_frames = video_frames is not None and isinstance(video_frames, torch.Tensor) and video_frames.numel() > 0

        if has_latent and has_frames:
            raise ValueError(t("err_post_process_both_video_inputs"))
        if not has_latent and not has_frames:
            raise ValueError(t("err_post_process_no_video_input"))
        if audio_latent is None:
            raise ValueError(t("err_post_process_no_audio_latent"))

        audio_tensor = _extract_samples_tensor(audio_latent, 4, "audio_latent")

        if has_latent:
            video_tensor = _extract_samples_tensor(video_latent, 5, "video_latent")
            report_lines.append("video_source=latent")
            report_lines.append(f"video_latent_shape={list(video_tensor.shape)}")
        else:
            # 从 image frames 重新编码
            if not is_connected_value(video_vae):
                raise ValueError(t("err_post_process_frames_need_vae"))
            if video_frames.ndim != 4:
                raise ValueError(
                    f"video_frames 必须是 [T,H,W,C] 的 IMAGE，got ndim={video_frames.ndim}"
                )

            h = int(video_frames.shape[1])
            w = int(video_frames.shape[2])
            if h % CANVAS_MULTIPLE != 0 or w % CANVAS_MULTIPLE != 0:
                raise ValueError(
                    t("err_post_process_frames_size_multiple",
                      w=w, h=h, multiple=CANVAS_MULTIPLE)
                )
            if h % 2 != 0 or w % 2 != 0:
                raise ValueError(t("err_post_process_frames_size_even", w=w, h=h))

            try:
                video_tensor = video_vae.encode(video_frames[..., :3])
            except Exception as e:
                raise RuntimeError(t("err_post_process_encode", error=e))

            if not isinstance(video_tensor, torch.Tensor) or video_tensor.ndim != 5:
                raise ValueError(t("err_post_process_encode_shape"))

            report_lines.append("video_source=frames (re-encoded via video_vae)")
            report_lines.append(f"frames_shape={list(video_frames.shape)}")
            report_lines.append(f"encoded_latent_shape={list(video_tensor.shape)}")

        # 尺寸一致性校验（仅当 video_latent 直连时可能失效）
        if video_tensor.shape[0] != audio_tensor.shape[0]:
            if video_tensor.shape[0] == 1:
                pass  # batch=1 允许
            else:
                raise ValueError(
                    f"video_latent batch={video_tensor.shape[0]} 与 "
                    f"audio_latent batch={audio_tensor.shape[0]} 不匹配"
                )

        try:
            av_samples = comfy.nested_tensor.NestedTensor(
                (video_tensor, audio_tensor)
            )
        except Exception as e:
            raise RuntimeError(t("err_post_process_nested", error=e))

        out = {"samples": av_samples}

        report_lines.append(f"audio_latent_shape={list(audio_tensor.shape)}")
        report_lines.append("final_av_latent=assembled")

        logger.info(
            "[YimoH3PostProcessMerge] video=%s audio=%s",
            tuple(video_tensor.shape), tuple(audio_tensor.shape),
        )

        return io.NodeOutput(out, "\n".join(report_lines))