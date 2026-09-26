from __future__ import annotations

import logging

from comfy_api.latest import io

from .conditioning import build_conditioning
from .core import REF_CURVE_DIRECTIONS, REF_CURVE_SHAPES
from .i18n import t

logger = logging.getLogger("YimoH3")

CATEGORY = t("category")
MAX_RESOLUTION = 16384


class YimoH3Conditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YimoH3Conditioning",
            display_name=t("node_display_name"),
            description=t("node_description"),
            category=CATEGORY,
            inputs=[
                io.Clip.Input("clip", tooltip="Native MiniMax H3 Qwen3-VL CLIP."),
                io.Vae.Input("video_vae", tooltip="MiniMax H3 video VAE."),
                io.Vae.Input("audio_vae", tooltip="MiniMax H3 audio VAE."),

                io.Combo.Input("workflow_mode",
                    display_name=t("workflow_mode_input"),
                    options=["text", "first_frame", "last_frame", "first_last_frame", "references", "hybrid"],
                    default="text",
                    tooltip=t("workflow_mode_tooltip"),
                ),
                io.Combo.Input("audio_policy",
                    display_name=t("audio_policy_input"),
                    options=["keep_source", "remix_source", "reference_only", "generate_new"],
                    default="generate_new",
                    tooltip=t("audio_policy_tooltip"),
                ),

                io.Boolean.Input("show_all_params", default=False, tooltip=t("show_all_params_tooltip")),

                # v2.8.1: 上一段段数据包（可选），用于继承素材，避免重复连线
                io.Conditioning.Input("prev_segment_bundle", optional=True,
                    tooltip="可选。连接上一段 YimoH3Conditioning 的「段数据包」输出，"
                            "本段会自动继承上一段的素材（image / video / audio refs），"
                            "不需要在本段重复连接相同的素材。\n"
                            "触发条件：本段未连接任何本地素材，且 workflow_mode 为 references / hybrid，"
                            "且本段画布与上一段一致。\n"
                            "注意：继承只对 DiT 端生效，CLIP 端看不到这些素材，"
                            "因此本段提示词里不要写 <Picture N> 之类引用，改用自然语言描述。"),

                io.String.Input("prompt", multiline=True, dynamic_prompts=True,
                    tooltip="正向提示词。支持 <Picture N> / <Video N> / <Audio N> / <Subject N> 等媒体标签。"),
                io.String.Input("negative_prompt", multiline=True, dynamic_prompts=True, default="", advanced=True,
                    tooltip="负向提示词。留空则使用空字符串。"),

                io.Int.Input("width", default=1344, min=32, max=MAX_RESOLUTION, step=32,
                    tooltip="视频宽度（像素）。自动向上对齐到 32 的倍数。"),
                io.Int.Input("height", default=768, min=32, max=MAX_RESOLUTION, step=32,
                    tooltip="视频高度（像素）。自动向上对齐到 32 的倍数。"),
                io.Boolean.Input("auto_resolution", default=False, advanced=True,
                    tooltip="根据输入媒体原生分辨率给出建议，但不会覆盖用户手填的 width/height，仅在报告中提示。"),
                io.Int.Input("length", default=124, min=5, max=3600, step=17,
                    tooltip="24fps; snapped up to the 17n+5 H3 grid."),

                io.Image.Input("first_frame", optional=True,
                    tooltip="首帧图。first_frame / first_last_frame / hybrid 模式生效。"),
                io.Image.Input("last_frame", optional=True,
                    tooltip="尾帧图。last_frame / first_last_frame / hybrid 模式生效。"),

                io.Image.Input("keyframes", optional=True,
                    tooltip="中间关键帧图片，支持图像批次[B,H,W,C]或图像列表，最多9张，可配合 keyframe_positions 指定插帧位置"),
                io.String.Input("keyframe_positions", default="", tooltip=t("keyframe_positions_tooltip")),

                io.Image.Input("ref_images", optional=True,
                    tooltip="参考图像，支持图像批次或图像列表，最多9张"),
                io.Autogrow.Input("ref_videos", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Image.Input("ref_video", optional=True,
                        tooltip="IMAGE frame batch at 24fps."),
                    prefix="ref_video_", min=1, max=3)),
                io.Audio.Input("source_audio", optional=True,
                    tooltip="源音频。keep_source / remix_source / reference_only 策略需要。"),
                io.Audio.Input("override_audio", optional=True,
                    tooltip="覆盖输出的混音音频。连接后 Mux Audio 端口输出此音频，忽略 source_audio。"),
                io.Autogrow.Input("ref_video_audios", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Audio.Input("ref_video_audio", optional=True,
                        tooltip="与同编号 ref_video 配对的音频。仅当 ref_video_N 已连接时生效。"),
                    prefix="ref_video_audio_", min=1, max=3)),
                io.Autogrow.Input("style_audios", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Audio.Input("style_audio", optional=True),
                    prefix="style_audio_", min=1, max=3)),

                io.Combo.Input("ref_image_size", options=["match", "max"], default="match", advanced=True,
                    tooltip="match=缩放到与画布一致；max=缩放到短边 2048 但不放大。"),
                io.Combo.Input("reference_video_policy", options=["official_2_to_15s", "model_minimum"],
                    default="official_2_to_15s", advanced=True,
                    tooltip="official_2_to_15s=严格按官方 48~360 帧要求；model_minimum=宽松模式，低于 48 帧也允许。"),
                io.Int.Input("ref_video_start_frame", default=0, min=0, max=3600, step=1, advanced=True,
                    tooltip="从参考视频的第几帧开始截取。0 表示从头开始。"),
                io.Float.Input("reference_strength", default=1.0, min=0.0, max=1.0, step=0.001, advanced=True,
                    tooltip="参考强度原始数值（0-1）。与 reference_retention 共同决定最终强度。"),
                io.String.Input("identity_image_indices", default="", advanced=True,
                    tooltip=t("identity_image_indices_tooltip")),

                io.Combo.Input("reference_retention",
                    display_name=t("reference_retention_input"),
                    options=["fully_preserved", "partially_preserved", "attribute_transfer", "weak_reference", "no_reference"],
                    default="fully_preserved",
                    advanced=False,
                    tooltip=t("reference_retention_tooltip"),
                ),

                io.Combo.Input("ref_curve_direction",
                    display_name=t("ref_curve_direction_input"),
                    options=REF_CURVE_DIRECTIONS,
                    default="constant",
                    advanced=False,
                    tooltip=t("ref_curve_direction_tooltip"),
                ),
                io.Combo.Input("ref_curve_shape",
                    display_name=t("ref_curve_shape_input"),
                    options=REF_CURVE_SHAPES,
                    default="linear",
                    advanced=True,
                    tooltip=t("ref_curve_shape_tooltip"),
                ),

                io.Float.Input("text_boost_strength", default=1.0, min=0.25, max=8.0, step=0.25,
                    advanced=True,
                    display_name=t("text_boost_strength_input"),
                    tooltip=t("text_boost_strength_tooltip")),
                io.Combo.Input("text_boost_mode",
                    options=["deviation", "naive"],
                    default="deviation",
                    advanced=True,
                    tooltip=t("text_boost_mode_tooltip")),
                io.Boolean.Input("text_boost_renorm", default=True, advanced=True,
                    tooltip=t("text_boost_renorm_tooltip")),

                io.Boolean.Input("strict_prompt_tags", default=False, advanced=True,
                    tooltip="开启后，若提示词中的 <Picture N> 等标签数量与实际连接媒体不匹配，直接报错；关闭则仅警告。"),

                io.Int.Input("audio_offset_frames", default=0, min=-3600, max=3600, step=1, advanced=True,
                    tooltip="音频整体偏移帧数（正=延后，负=提前）。超过目标长度的部分会被硬截断。"),
                io.Float.Input("audio_denoise_strength", default=0.35, min=0.0, max=1.0, step=0.01, advanced=True,
                    tooltip="remix_source 策略下的音频重绘强度。0=完全保留源音频，1=完全重绘。"),

                io.Boolean.Input("clip_video_grayout", default=False, advanced=True,
                    tooltip=t("clip_video_grayout_tooltip")),

                io.Combo.Input("ref_video_preprocessing",
                    options=["none", "blur_desaturate"],
                    default="none",
                    advanced=True,
                    tooltip=t("ref_video_preprocessing_tooltip"),
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name=t("output_segment")),
                io.Conditioning.Output(display_name=t("output_positive")),
                io.Conditioning.Output(display_name=t("output_negative")),
                io.Latent.Output(display_name=t("output_av_latent")),
                io.Audio.Output(display_name=t("output_mux_audio")),
                io.String.Output(display_name=t("output_conditioned_prompt")),
                io.String.Output(display_name=t("output_report")),
            ],
        )

    @classmethod
    def execute(cls, workflow_mode, audio_policy, show_all_params, clip, video_vae, audio_vae, prompt, negative_prompt,
                width, height, auto_resolution, length, first_frame=None, last_frame=None,
                keyframes=None, keyframe_positions="",
                ref_images=None, ref_videos=None, source_audio=None, override_audio=None,
                ref_video_audios=None, style_audios=None,
                ref_image_size="match", reference_video_policy="official_2_to_15s", ref_video_start_frame=0,
                reference_strength=1.0, identity_image_indices="",
                reference_retention="fully_preserved",
                ref_curve_direction="constant",
                ref_curve_shape="linear",
                text_boost_strength=1.0, text_boost_mode="deviation", text_boost_renorm=True,
                strict_prompt_tags=False,
                audio_offset_frames=0, audio_denoise_strength=0.35,
                clip_video_grayout=False, ref_video_preprocessing="none",
                prev_segment_bundle=None, **kwargs):

        result_tuple = build_conditioning(
            workflow_mode=workflow_mode,
            audio_policy=audio_policy,
            clip=clip,
            video_vae=video_vae,
            audio_vae=audio_vae,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            auto_resolution=auto_resolution,
            length=length,
            first_frame=first_frame,
            last_frame=last_frame,
            keyframes=keyframes,
            keyframe_positions=keyframe_positions,
            ref_images=ref_images,
            ref_videos=ref_videos,
            ref_video_audios=ref_video_audios,
            style_audios=style_audios,
            source_audio=source_audio,
            override_audio=override_audio,
            ref_image_size=ref_image_size,
            reference_video_policy=reference_video_policy,
            ref_video_start_frame=ref_video_start_frame,
            reference_strength=reference_strength,
            identity_image_indices=identity_image_indices,
            reference_retention=reference_retention,
            ref_curve_direction=ref_curve_direction,
            ref_curve_shape=ref_curve_shape,
            text_boost_strength=text_boost_strength,
            text_boost_mode=text_boost_mode,
            text_boost_renorm=text_boost_renorm,
            strict_prompt_tags=strict_prompt_tags,
            audio_offset_frames=audio_offset_frames,
            audio_denoise_strength=audio_denoise_strength,
            clip_video_grayout=clip_video_grayout,
            ref_video_preprocessing=ref_video_preprocessing,
            prev_segment_bundle=prev_segment_bundle,
        )

        # result_tuple[0] 是 segment_bundle（第 1 个输出「段数据包」）。
        if result_tuple[0] is None:
            raise RuntimeError(
                "段数据包构建失败（segment_bundle=None）。常见原因：\n"
                "1) clip 未连接或 encode_from_tokens_scheduled 返回空；\n"
                "2) strict_prompt_tags=true 但提示词标签与实际媒体数量不匹配；\n"
                "3) conditioning 内部失败。\n"
                "请查看终端日志中 [YimoH3] 的 warning / compat 信息以定位具体原因。"
            )

        return io.NodeOutput(*result_tuple)