# -*- coding: utf-8 -*-
"""Yimo H3 Face Restore Node (v2.5.2) - 采样后修复

ComfyUI 节点：对已采样的 AV Latent 进行面部修复。
底层引擎在 face_restore.py。
"""

from __future__ import annotations

import logging

import comfy.nested_tensor
from comfy_api.latest import io

from .core import nested_av_parts, is_connected_value
from .i18n import t
from .face_restore import (
    build_identity_bank,
    decode_video_latent,
    restore_frames,
    stats_report,
    _get_gfpgan_restorer,
    _get_insightface_analyzer,
)

logger = logging.getLogger("YimoH3")
CATEGORY = t("category")


class YimoH3FaceRestore(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YimoH3FaceRestore",
            display_name="Yimo H3 面部修复 (采样后)",
            description="对已采样的 AV Latent 进行面部修复。自动选择最佳方案: GFPGAN > OpenCV增强。所有依赖均可选。",
            category=CATEGORY,
            inputs=[
                io.Latent.Input("av_latent", tooltip="待修复的 AV 潜变量 (来自采样器输出)"),
                io.Vae.Input("video_vae", tooltip="MiniMax H3 video VAE (用于解码/编码帧)"),

                io.Image.Input("face_images", optional=True,
                    tooltip="面部参考图 (可选)。支持单图/多图/批次，用于身份匹配和多人修复。"),

                io.Combo.Input("repair_backend",
                    options=["auto", "gfpgan", "opencv", "insightface"],
                    default="auto",
                    tooltip="选择面部修复后端: auto=自动选择, gfpgan=最佳效果, opencv=轻量快速"),

                io.Float.Input("restore_strength", default=0.7, min=0.0, max=1.0, step=0.05,
                    tooltip="修复强度：0=不修复，1=完全替换。建议0.5-0.8之间"),
                io.Float.Input("det_threshold", default=0.5, min=0.0, max=1.0, step=0.05,
                    tooltip="人脸检测置信度阈值"),
                io.Int.Input("max_faces_per_frame", default=10, min=1, max=30,
                    tooltip="每帧最大检测人脸数。多人场景可调高"),
                io.Boolean.Input("restore_unmatched", default=True,
                    tooltip="是否修复所有检测到的人脸。True=修复所有人脸，False=仅修复匹配到参考图的人脸"),
                io.Float.Input("feather", default=0.25, min=0.0, max=0.5,
                    tooltip="边缘羽化宽度。值越大，修复区域边缘过渡越平滑"),
                io.Int.Input("min_face_size", default=64, min=16, max=256,
                    tooltip="最小人脸尺寸 (像素)。小于此值的人脸将被跳过"),
                io.Int.Input("detection_stride", default=1, min=1, max=8,
                    tooltip="每 N 帧检测修复一次。>1 可加速处理，但中间帧不修复"),
                io.Boolean.Input("encode_back", default=True,
                    tooltip="True=修复后重新编码回AV潜变量；False=仅输出修复后的帧序列(用于预览)"),
            ],
            outputs=[
                io.Latent.Output(display_name="修复后 AV 潜变量"),
                io.Image.Output(display_name="修复后帧序列"),
                io.String.Output(display_name="修复报告"),
            ],
        )

    @classmethod
    def execute(cls, av_latent, video_vae, face_images, restore_strength,
                det_threshold, max_faces_per_frame, restore_unmatched,
                feather, min_face_size, detection_stride, encode_back,
                repair_backend="auto", **kwargs):

        bank = None
        if is_connected_value(face_images):
            bank = build_identity_bank(face_images, det_threshold=det_threshold)
            if bank:
                logger.info("身份参考库构建完成: %d 人", len(bank))
            else:
                logger.warning("face_images 未检测到有效人脸，将进行通用修复")

        try:
            video, audio = nested_av_parts(av_latent)
        except Exception as e:
            raise ValueError(f"AV Latent 解析失败: {e}。请确保输入来自 YimoH3 采样器输出。")

        frames = decode_video_latent(video, video_vae)
        logger.info("解码帧数: %d", frames.shape[0] if frames is not None else 0)

        try:
            restored, stats = restore_frames(
                frames, bank,
                strength=restore_strength,
                det_threshold=det_threshold,
                max_faces_per_frame=max_faces_per_frame,
                restore_unmatched=restore_unmatched,
                feather=feather,
                min_face_size=min_face_size,
                stride=detection_stride,
                repair_backend=repair_backend,
            )
        except Exception as e:
            logger.error("面部修复失败: %s", e)
            restored = frames
            stats = {"error": str(e), "frames": frames.shape[0] if frames is not None else 0}

        if encode_back:
            try:
                new_video = video_vae.encode(restored)
                out_latent = {"samples": comfy.nested_tensor.NestedTensor((new_video, audio))}
            except Exception as e:
                logger.error("视频重新编码失败: %s", e)
                out_latent = av_latent
        else:
            out_latent = av_latent

        report = cls._build_report(stats, encode_back)

        return io.NodeOutput(out_latent, restored, report)

    @classmethod
    def _build_report(cls, stats: dict, encode_back: bool) -> str:
        lines = [
            "=== Yimo H3 Face Restore 报告 ===",
            "",
            "【修复统计】",
        ]

        if "error" in stats:
            lines.append(f"  ❌ 修复失败: {stats['error']}")
        else:
            lines.extend([
                f"  总帧数: {stats.get('frames', 0)}",
                f"  检测人脸: {stats.get('faces_total', 0)}",
                f"  修复人脸: {stats.get('faces_restored', 0)}",
                f"  未匹配跳过: {stats.get('unmatched', 0)}",
                f"  身份命中: {stats.get('person_hits', {}) or '无'}",
                f"  检测后端: {stats.get('detection', 'unknown')}",
                f"  修复后端: {stats.get('backend', 'unknown')}",
            ])

        lines.extend([
            "",
            "【设置】",
            f"  encode_back: {encode_back}",
            "",
            "【安装建议(可选)】",
            "  最佳效果: pip install insightface onnxruntime-gpu",
            "  GFPGAN升级: pip install gfpgan --no-deps facexlib",
            "  注: 即使不安装任何额外依赖，仍可使用 OpenCV 增强降级方案",
        ])

        return "\n".join(lines)