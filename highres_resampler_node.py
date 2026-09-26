# -*- coding: utf-8 -*-
"""Yimo H3 高分辨率重采样器节点 (实验性)

v2.7.0:
- 删除未使用的 face_images 输入；身份参考统一走 segment_bundle。

v2.0:
- 视频：高分辨率重采样 + 身份感知 + 时间分块 + 空间分块
- 音频：像素层增强 + 音色锚 + 响度归一化 + QA
"""

from __future__ import annotations

import json
import logging

import torch

from comfy_api.latest import io

from .core import nested_av_parts, CANVAS_MULTIPLE
from .i18n import t
from .highres_resampler_core import resample_video
from .highres_audio_core import resample_audio

logger = logging.getLogger("YimoH3")
CATEGORY = t("category")

_PIXEL_TO_LATENT = 16


def _pixel_to_latent(pixel_size: int) -> int:
    return max(1, pixel_size // _PIXEL_TO_LATENT)


class YimoH3HighResResampler(io.ComfyNode):
    """高分辨率重采样器：把低分辨率 AV Latent 提升到 2~3 倍分辨率。

    [实验性] 建议优先使用 YimoH3PDDTwoPass（PDD 双采）或像素级超分（SeedVR2）。
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YimoH3HighResResampler",
            display_name="Yimo H3 高分辨率重采样器 (实验性)",
            description="[实验性] 把低分辨率 latent 通过双线性上采样 + 重采样去噪提升到 1.2~4 倍。"
                        "注意：本节点与 Turbo / Lightning 等加速 LoRA 存在机制冲突，"
                        "且不是社区主流方案。建议优先使用 YimoH3PDDTwoPass (PDD 双采) "
                        "或像素级超分 (SeedVR2)。",
            category=CATEGORY,
            inputs=[
                # ===== 1. 模型类 =====
                io.Model.Input("model", tooltip="MiniMax H3 DiT 模型。"),
                io.Vae.Input("video_vae", optional=True,
                    tooltip="H3 Video VAE。用于面部检测与质量控制；不连则跳过相关功能。"),
                io.Vae.Input("audio_vae", optional=True,
                    tooltip="H3 Audio VAE。仅在 enable_audio_resample=true 时必需。"),

                # ===== 2. 辅助类 =====
                io.Sampler.Input("sampler", tooltip="ComfyUI 采样器。本节点目前固定使用 euler。"),
                io.Sigmas.Input("sigmas", optional=True,
                    tooltip="可选 SIGMAS。不连则内部按 base_denoise 生成。"),

                # ===== 3. 数据类 =====
                io.Latent.Input("latent", tooltip="待重采样的 AV Latent。"),
                io.Conditioning.Input("positive", tooltip="正向条件。"),
                io.Conditioning.Input("negative", optional=True, tooltip="负向条件。"),
                io.Conditioning.Input("segment_bundle", optional=True,
                    tooltip="可选：连接 YimoH3Conditioning 的 segment 输出，"
                            "自动读取身份参考图和源音频"),

                # ===== 4. 视频重采样参数 =====
                io.Float.Input("video_scale", default=2.0, min=1.2, max=4.0, step=0.1,
                    tooltip="视频重采样倍数。2.0 表示 480×864 → 960×1728"),
                io.Combo.Input("denoise_strategy",
                    options=["uniform", "staged", "face_aware"],
                    default="face_aware",
                    tooltip="重采样策略：\n"
                            "uniform=全图统一 denoise（最快）\n"
                            "staged=分阶段 sigma 曲线（平衡）\n"
                            "face_aware=面部感知 + 自适应 denoise（最保身份，推荐）"),
                io.Float.Input("base_denoise", default=0.5, min=0.1, max=0.9, step=0.05,
                    tooltip="基准 denoise。face_aware 时面部区域会在此基础上再降低"),
                io.Float.Input("face_denoise_reduction", default=0.5, min=0.0, max=1.0, step=0.05,
                    tooltip="面部区域的 denoise 降低比例"),
                io.Boolean.Input("enable_identity_injection", default=True,
                    tooltip="开启后从 segment_bundle 读取身份图，在第二遍重采样时再次注入"),

                # ===== 5. 时间分块 =====
                io.Boolean.Input("enable_temporal_chunking", default=True,
                    tooltip="是否按时间分块采样。长视频建议开启。"),
                io.Int.Input("chunk_frames", default=73, min=22, max=300, step=1,
                    tooltip="每个时间块的帧数。会自动对齐到 H3 网格"),
                io.Int.Input("chunk_overlap", default=22, min=0, max=100, step=1,
                    tooltip="时间分块之间的重叠帧数。越大过渡越平滑，但计算量越大。"),

                # ===== 6. 空间分块 =====
                io.Boolean.Input("enable_spatial_tiling", default=False,
                    tooltip="开启空间分块：把 latent 按空间切成小块分别采样，"
                            "突破显存限制。低分辨率下可不开。"),
                io.Int.Input("tile_size_pixel", default=512, min=256, max=2048, step=64,
                    tooltip="空间分块的 tile 大小（像素）。会转换为 latent 尺寸（÷16）"),
                io.Int.Input("tile_overlap_pixel", default=128, min=32, max=512, step=32,
                    tooltip="空间分块的重叠带大小（像素）。越大过渡越平滑，但重复计算越多"),

                # ===== 7. 颜色匹配 =====
                io.Combo.Input("color_match_mode",
                    options=["off", "low", "medium", "high"],
                    default="medium",
                    tooltip="重采样后与原图的颜色匹配模式。off=不匹配，medium=Reinhard 均值+标准差，high=max 仅对第一帧应用同 medium。"),

                # ===== 8. 音频参数 =====
                io.Boolean.Input("enable_audio_resample", default=False,
                    tooltip="是否对音频做像素层增强。需要连接 audio_vae。默认关闭以节省时间。"),
                io.Float.Input("audio_denoise", default=0.25, min=0.0, max=0.6, step=0.05,
                    tooltip="音频增强强度。0.2~0.3 推荐"),
                io.Boolean.Input("enable_timbre_anchor", default=False,
                    tooltip="音色锚：用源音频的 RMS / Crest Factor 校准重采样音频的音色"),
                io.Boolean.Input("enable_loudness_norm", default=False,
                    tooltip="响度归一化，多段拼接时必备"),
                io.Float.Input("target_rms", default=0.05, min=0.001, max=0.2, step=0.001,
                    tooltip="响度归一化目标 RMS。0.05 是常见语音/音乐折中值。"),

                # ===== 9. 质量检查 =====
                io.Boolean.Input("enable_quality_check", default=False,
                    tooltip="重采样完成后自动检测面部相似度、色调偏差、音频质量"),

                # ===== 10. 采样参数 =====
                io.Float.Input("cfg", default=1.0, min=0.0, max=2.0, step=0.1,
                    tooltip="CFG 引导强度。建议 1.0（重采样不依赖 CFG）。"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, step=1,
                    tooltip="随机种子。"),
            ],
            outputs=[
                io.Latent.Output(display_name="高分辨率 AV Latent"),
                io.String.Output(display_name="重采样报告"),
                io.String.Output(display_name="质量报告 JSON"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model, latent, positive, sampler,
        negative=None, sigmas=None,
        video_vae=None, audio_vae=None,
        segment_bundle=None,
        video_scale=2.0, denoise_strategy="face_aware",
        base_denoise=0.5, face_denoise_reduction=0.5,
        enable_identity_injection=True,
        enable_temporal_chunking=True,
        chunk_frames=73, chunk_overlap=22,
        enable_spatial_tiling=False,
        tile_size_pixel=512, tile_overlap_pixel=128,
        color_match_mode="medium",
        enable_audio_resample=False,
        audio_denoise=0.25,
        enable_timbre_anchor=False,
        enable_loudness_norm=False,
        target_rms=0.05,
        enable_quality_check=False,
        cfg=1.0, seed=0,
        **kwargs
    ):
        # 解析 AV latent
        try:
            video, audio = nested_av_parts(latent)
        except Exception as e:
            raise ValueError(f"输入 latent 不是 AV NestedTensor: {e}")

        report_lines = []
        quality_json = {}

        sampler_name = cls._resolve_sampler_name(sampler)

        tile_size_latent = _pixel_to_latent(tile_size_pixel)
        tile_overlap_latent = _pixel_to_latent(tile_overlap_pixel)

        # ========== 视频重采样 ==========
        resampled_video, video_report, video_qa = resample_video(
            model=model,
            positive=positive,
            negative=negative,
            video_latent=video,
            audio_latent=audio,
            sampler_name=sampler_name,
            scheduler="simple",
            sigmas=sigmas,
            cfg=cfg,
            seed=seed,
            scale=video_scale,
            denoise_strategy=denoise_strategy,
            base_denoise=base_denoise,
            video_vae=video_vae,
            segment_bundle=segment_bundle,
            enable_temporal_chunking=enable_temporal_chunking,
            chunk_frames=chunk_frames,
            chunk_overlap=chunk_overlap,
            enable_identity_injection=enable_identity_injection,
            face_denoise_reduction=face_denoise_reduction,
            color_match_mode=color_match_mode,
            enable_quality_check=enable_quality_check,
            enable_spatial_tiling=enable_spatial_tiling,
            tile_size_latent=tile_size_latent,
            tile_overlap_latent=tile_overlap_latent,
        )
        report_lines.extend(video_report)
        if video_qa:
            quality_json["video"] = video_qa

        # ========== 音频重采样 ==========
        source_audio_waveform = None
        if enable_timbre_anchor and segment_bundle is not None:
            try:
                cd = segment_bundle[0][1] if isinstance(segment_bundle, list) else None
                yimo_data = cd.get("_yimo_data", {}) if isinstance(cd, dict) else {}
                audio_dict = yimo_data.get("audio")
                if isinstance(audio_dict, dict) and "waveform" in audio_dict:
                    source_audio_waveform = audio_dict["waveform"]
            except Exception as e:
                logger.debug("[HighRes] source audio extraction failed: %s", e)

        if enable_audio_resample:
            resampled_audio, audio_report, audio_qa = resample_audio(
                audio_vae=audio_vae,
                audio_latent=audio,
                denoise=audio_denoise,
                enable_timbre_anchor=enable_timbre_anchor,
                source_audio_waveform=source_audio_waveform,
                enable_loudness_norm=enable_loudness_norm,
                target_rms=target_rms,
                enable_qa=enable_quality_check,
            )
            report_lines.extend(audio_report)
            if audio_qa:
                quality_json["audio"] = audio_qa
        else:
            resampled_audio = audio
            report_lines.append("=== Audio: resample disabled, unchanged ===")

        # ========== 组装输出 ==========
        import comfy.nested_tensor
        out_latent = {"samples": comfy.nested_tensor.NestedTensor((resampled_video, resampled_audio))}

        report_lines.append("=== final ===")
        report_lines.append(f"video_out_shape={list(resampled_video.shape)}")
        report_lines.append(f"audio_out_shape={list(resampled_audio.shape)}")

        quality_json_str = json.dumps(quality_json, ensure_ascii=False, indent=2)

        return io.NodeOutput(
            out_latent,
            "\n".join(report_lines),
            quality_json_str,
        )

    @staticmethod
    def _resolve_sampler_name(sampler):
        try:
            if hasattr(sampler, "sampler_function"):
                fn_name = sampler.sampler_function.__name__
                mapping = {
                    "sample_euler": "euler",
                    "sample_euler_ancestral": "euler_ancestral",
                    "sample_dpmpp_2m": "dpmpp_2m",
                    "sample_heun": "heun",
                }
                return mapping.get(fn_name, "euler")
        except Exception:
            pass
        return "euler"