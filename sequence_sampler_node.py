from __future__ import annotations

import sys
import traceback
import logging
import os
import torch
from datetime import datetime

import comfy.samplers
import comfy.utils
from comfy_api.latest import io

from .core import (
    sorted_autogrow_items,
    nested_av_parts,
    is_connected_value,
    get_sequence_sampler_export_dir,
    get_yimo_output_dir,
    sanitize_filename,
)
from .face_restore import build_identity_bank
from .i18n import t
from .sampler_core import (
    sequence_sample,
    _isolate_payload_cond,
    build_pdd_sigmas,
)
from .color_match_utils import apply_color_match

logger = logging.getLogger("YimoH3")

CATEGORY = t("category")

# v2.7.2: overlap_list 负值回退修复（与 ConcatenateSegments 对齐）。
# v2.7.1: _smart_chunked_decode 内存阈值改为参数化，默认值如下。
_CHUNK_MEMORY_THRESHOLD_DEFAULT_MB = 4096

try:
    class YimoH3SequenceSampler(io.ComfyNode):

        @classmethod
        def define_schema(cls):
            try:
                try:
                    sampler_names = comfy.samplers.KSampler.SAMPLERS
                except AttributeError:
                    try:
                        sampler_names = comfy.samplers.SAMPLER_NAMES
                    except AttributeError:
                        sampler_names = ["euler", "euler_ancestral", "heun", "dpmpp_2m",
                                         "dpmpp_2m_sde", "dpmpp_sde", "dpmpp_2s_ancestral",
                                         "dpm_fast", "dpm_adaptive", "ddim", "uni_pc", "uni_pc_bh2"]
                try:
                    scheduler_names = comfy.samplers.KSampler.SCHEDULERS
                except AttributeError:
                    try:
                        scheduler_names = comfy.samplers.SCHEDULER_NAMES
                    except AttributeError:
                        scheduler_names = ["normal", "karras", "exponential", "sgm_uniform",
                                           "simple", "ddim_uniform"]

                return io.Schema(
                    node_id="YimoH3SequenceSampler",
                    display_name=t("seqsampler_display_name"),
                    description=t("seqsampler_description"),
                    category=CATEGORY,
                    inputs=[
                        io.Model.Input("model", tooltip="MiniMax H3 DiT model."),
                        io.Vae.Input("video_vae", optional=True, tooltip=t("video_vae_tooltip")),
                        io.Vae.Input("audio_vae", optional=True, tooltip=t("audio_vae_tooltip")),

                        io.Autogrow.Input("segments", optional=False, template=io.Autogrow.TemplatePrefix(
                            input=io.Conditioning.Input("segment", optional=True,
                                tooltip="Segment bundle from YimoH3Conditioning.segment"),
                            prefix="segment_", min=1, max=8),
                            tooltip="1~8 个段数据包。每个来自一个 YimoH3Conditioning 的「段数据包」输出。"),

                        io.Combo.Input("sampling_mode",
                            options=["standard", "pdd", "pdd_two_stage"],
                            default="standard",
                            tooltip="standard=标准采样；pdd=PDD 单阶段加速；"
                                    "pdd_two_stage=PDD 双采（LOW → 插值放大 → HIGH）。"),
                        io.Custom("SIGMAS").Input("external_sigmas", optional=True,
                            tooltip="PDD 模式的 sigma 轨迹。如果不连接，插件会自动生成。standard 模式忽略此输入。"),

                        io.Int.Input("pdd_split_step", default=4, min=1, max=7, step=1,
                            advanced=True,
                            tooltip="PDD 双采切分索引。sigmas 共 9 点（8 步），split_step=4 表示"
                                    "LOW 阶段跑前 4 步、HIGH 阶段跑后 4 步。\n"
                                    "仅 sampling_mode=pdd_two_stage 时生效。"),
                        io.Float.Input("pdd_scale_by", default=1.5, min=1.0, max=4.0, step=0.05,
                            advanced=True,
                            tooltip="PDD 双采中间态放大倍数。LOW 阶段结束后，video latent 会按此倍数"
                                    "做双线性空间放大，再进入 HIGH 阶段。\n"
                                    "1.5 是推荐起点；2.0 及以上对显存要求显著提高。\n"
                                    "仅 sampling_mode=pdd_two_stage 时生效。"),

                        io.Int.Input("overlap_frames", default=22, min=0, max=200, step=1,
                            tooltip="默认段间重叠帧数。会在 17n+5 网格上按最近邻映射：\n"
                                    "  0        → 无注入（视频/音频都不会有段间连贯）\n"
                                    "  1~5      → effective 5（极短衔接，仅适合纯动作衔接，"
                                    "换装/换风格场景可能失效，会被 DiT 当作全局参考模板）\n"
                                    "  6~22     → effective 22（标准衔接，推荐值）\n"
                                    "  23 以上  → 按最近邻映射到 39 / 56 / ...\n"
                                    "换装、换风格场景请使用 >= 22。\n"
                                    "仅在 overlap_list 留空或全部为负数时生效。"),
                        io.String.Input("overlap_list", default="", advanced=True,
                            tooltip=t("overlap_list_tooltip")
                                    + "\n注意：填负数（如 -1）会被视为「未设置」，"
                                    "回退到 overlap_frames；0 仍然是有效的硬切标记。"),

                        io.Combo.Input("transition_mode",
                            options=["context_inject", "hard_concat"],
                            default="context_inject",
                            tooltip="视频连续性：\n"
                                    "context_inject=注入前段视频 tail latent，拼接时删除重叠帧（视频连贯）\n"
                                    "hard_concat=不注入，各段视频独立（视频不连贯）"),

                        io.Combo.Input("audio_continuity",
                            options=["context_inject", "break"],
                            default="context_inject",
                            advanced=False,
                            tooltip="音频连续性：\n"
                                    "context_inject=注入前段音频 tail latent（音频连贯）\n"
                                    "break=不注入，各段音频独立（音频不连贯）\n"
                                    "注意：音频注入依赖 overlap_frames>0。"),

                        io.Int.Input("resample_segment", default=-1, min=-1, max=7, step=1, advanced=True,
                            tooltip="指定从第几段开始重新采样（0-based）。-1=全部重新采样。"
                                    "需要 chain_id 非空才能加载缓存；chain_id 为空时会退化为全量重采样并记录警告。"),
                        io.String.Input("segment_seeds", default="", advanced=True,
                            tooltip="逗号分隔的每段独立种子，如：0,123,456,none,789。"
                                    "留空则所有段使用基础 seed 派生。"
                                    "非法项会被视为 none 并记录警告，不会中断采样。"),
                        io.String.Input("chain_id", default="default", advanced=True,
                            tooltip="缓存标识符。设置后每段采样结果会缓存到磁盘。"),

                        io.Int.Input("stop_after_segment", default=-1, min=-1, max=7, step=1, advanced=True,
                            tooltip="在采样完指定段后提前停止。-1=不提前停止，采样所有段。"),

                        io.Boolean.Input("auto_inherit_tail", default=True, advanced=True,
                            tooltip=t("auto_inherit_tail_tooltip")
                                    + "\n注意：仅在 transition_mode=hard_concat 或 effective_overlap=0 时生效。"),
                        io.Boolean.Input("inherit_identity", default=True, advanced=True,
                            tooltip=t("inherit_identity_tooltip")
                                    + "\n注意：仅对 references / hybrid 模式生效。"),

                        io.Boolean.Input("enable_ref_curve", default=False, advanced=True,
                            tooltip="启用参考强度时间曲线。对长视频，会根据每段在全局时间轴上的位置"
                                    "动态调整参考强度。单段时使用段中点强度。"),

                        io.Boolean.Input("override_ref_curve", default=False, advanced=True,
                            tooltip="覆盖所有分段的参考曲线配置。开启后，所有分段将统一使用下面指定的"
                                    "曲线方向与形状，忽略各自 segment 中携带的 _yimo_ref_curve。"
                                    "建议与 enable_ref_curve 同时开启。"
                                    "若方向或形状设为 inherit，则该项保持各分段自身配置。"),
                        io.Combo.Input("ref_curve_direction_override",
                            options=["inherit", "constant", "concept_at_start", "concept_at_end", "concept_at_middle", "concept_at_ends"],
                            default="inherit",
                            advanced=True,
                            tooltip="统一覆盖所有分段的曲线方向。inherit=使用各分段自身携带的配置。"
                                    "仅在 override_ref_curve=true 且非 inherit 时覆盖。"),
                        io.Combo.Input("ref_curve_shape_override",
                            options=["inherit", "linear", "ease", "sigmoid", "exponential", "quadratic", "cubic"],
                            default="inherit",
                            advanced=True,
                            tooltip="统一覆盖所有分段的曲线形状。inherit=使用各分段自身携带的配置。"
                                    "仅在 override_ref_curve=true 且非 inherit 时覆盖。"),

                        io.Combo.Input("sampler_name", options=sampler_names, default="euler",
                            tooltip="采样器。PDD 模式会自动切换为 euler。"),
                        io.Combo.Input("scheduler", options=scheduler_names, default="normal",
                            tooltip="调度器。PDD 模式下此参数无效（使用传入的 sigmas）。"),
                        io.Int.Input("steps", default=20, min=1, max=100, step=1,
                            tooltip="采样步数。PDD 模式下由 sigmas 长度决定，此参数被忽略。"),
                        io.Float.Input("cfg", default=7.0, min=0.0, max=100.0, step=0.1,
                            tooltip="CFG 引导强度。PDD LoRA 训练时 CFG=1，建议保持 1.0。"),
                        io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01,
                            tooltip="降噪强度。1.0=完全重绘，0=不变。PDD 双采时 LOW 和 HIGH 阶段共用此值。"),
                        io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, step=1,
                            tooltip="基础种子。每段派生 seed + i * 12345；如已设 segment_seeds 则按逐段覆盖。"),

                        io.Image.Input("face_images", optional=True,
                            tooltip="面部参考图。支持单图/多图/批次，用于身份匹配和多人修复。"
                                    "最佳效果: pip install insightface onnxruntime-gpu"),
                        io.Boolean.Input("face_restore", default=False, advanced=True,
                            tooltip="启用采样中面部修复。修复发生在 tail 提取之前，段间继承的是修复后的帧。"
                                    "自动选择最佳后端: GFPGAN > OpenCV增强"),
                        io.Combo.Input("face_restore_backend",
                            options=["auto", "gfpgan", "insightface", "opencv"],
                            default="auto",
                            advanced=True,
                            tooltip="选择面部修复后端: auto=自动选择, gfpgan=最佳效果(GFPGAN), "
                                    "insightface=InsightFace修复, opencv=轻量快速(OpenCV增强)"),
                        io.Float.Input("face_restore_strength", default=0.7, min=0.0, max=1.0, step=0.05,
                            advanced=True, tooltip="面部修复强度。建议0.5-0.8之间"),
                        io.Float.Input("face_similarity_threshold", default=0.45, min=0.0, max=1.0, step=0.01,
                            advanced=True, tooltip="ArcFace 余弦匹配阈值：同一人通常>0.5"),

                        io.Combo.Input("color_match_mode",
                            options=["off", "low", "medium", "high", "max"],
                            default="off",
                            advanced=True,
                            tooltip=t("color_match_mode_tooltip"),
                        ),
                        io.Float.Input("color_match_strength",
                            default=0.7, min=0.0, max=1.0, step=0.05,
                            advanced=True,
                            tooltip=t("color_match_strength_tooltip"),
                        ),
                        io.Int.Input("color_match_reference_frames",
                            default=1, min=1, max=17, step=1,
                            advanced=True,
                            tooltip=t("color_match_reference_frames_tooltip"),
                        ),

                        io.Combo.Input("output_format",
                            options=["latent", "images"],
                            default="latent",
                            advanced=True,
                            tooltip="最终输出格式。\n"
                                    "latent：输出 AV Latent（原始行为）。\n"
                                    "images：输出 IMAGE 序列，并在内部按 overlap=0 分块独立解码，"
                                    "避免硬切段整段闪烁。需要连接 video_vae。"),

                        io.Int.Input("chunk_memory_threshold_mb",
                            default=_CHUNK_MEMORY_THRESHOLD_DEFAULT_MB,
                            min=512, max=32768, step=256,
                            advanced=True,
                            tooltip="images 模式解码 chunk 的内存阈值（MB）。\n"
                                    "累积解码帧字节数低于此值时保留在内存，超过后溢出到磁盘。\n"
                                    "默认 4096 MB (4 GB)，显存/内存紧张时可调小。\n"
                                    "仅在 output_format=images 时生效。"),

                        io.Boolean.Input("auto_export_final", default=False, advanced=True,
                            tooltip="是否自动导出最终拼接结果到 output/yimo_h3/sequence_sampler/exports/"),
                        io.String.Input("export_name", default="", advanced=True,
                            tooltip="导出文件名（不含扩展名）。留空则使用 chain_id + 时间戳。"),
                    ],
                    outputs=[
                        io.Latent.Output(display_name=t("seqsampler_output_latent")),
                        io.String.Output(display_name=t("seqsampler_output_report")),
                        io.Latent.Output(display_name="分段0AV潜变量"),
                        io.Latent.Output(display_name="分段1AV潜变量"),
                        io.Latent.Output(display_name="分段2AV潜变量"),
                        io.Latent.Output(display_name="分段3AV潜变量"),
                        io.Latent.Output(display_name="分段4AV潜变量"),
                        io.Latent.Output(display_name="分段5AV潜变量"),
                        io.Latent.Output(display_name="分段6AV潜变量"),
                        io.Latent.Output(display_name="分段7AV潜变量"),
                        io.Image.Output(display_name="长视频图像序列 (images 模式)"),
                    ],
                )
            except Exception as e:
                print("=" * 70, file=sys.stderr)
                print("YimoH3SequenceSampler.define_schema FAILED:", e, file=sys.stderr)
                traceback.print_exc()
                print("=" * 70, file=sys.stderr)
                raise

        @classmethod
        def execute(cls, model, video_vae=None, audio_vae=None,
                    segments=None,
                    sampling_mode="standard", external_sigmas=None,
                    pdd_split_step=4, pdd_scale_by=1.5,
                    overlap_frames=22, overlap_list="", transition_mode="context_inject",
                    audio_continuity="context_inject",
                    resample_segment=-1, segment_seeds="", chain_id="default", stop_after_segment=-1,
                    auto_inherit_tail=True, inherit_identity=True,
                    enable_ref_curve=False,
                    override_ref_curve=False, ref_curve_direction_override="inherit", ref_curve_shape_override="inherit",
                    sampler_name="euler", scheduler="normal", steps=20, cfg=7.0, denoise=1.0, seed=0,
                    face_images=None, face_restore=False,
                    face_restore_backend="auto",
                    face_restore_strength=0.7, face_similarity_threshold=0.45,
                    color_match_mode="off", color_match_strength=0.7, color_match_reference_frames=1,
                    output_format="latent",
                    chunk_memory_threshold_mb=_CHUNK_MEMORY_THRESHOLD_DEFAULT_MB,
                    auto_export_final=False, export_name="",
                    **kwargs):

            if model is None:
                raise ValueError("model 未连接")

            segment_items = sorted_autogrow_items(segments)
            if not segment_items:
                raise ValueError(t("err_no_segments"))

            parsed_segments = []
            for ordinal, seg_cond in segment_items:
                if seg_cond is None:
                    raise ValueError(
                        f"段_{ordinal} 为 None。这表示上游 YimoH3Conditioning 节点内部执行失败，"
                        f"导致段数据包未生成。常见原因：\n"
                        f"1) ref_video 帧数不足（official_2_to_15s 要求 48~360 帧）；\n"
                        f"2) strict_prompt_tags=true 但 prompt 标签与实际连接媒体数量不匹配；\n"
                        f"3) 参考图/视频尺寸异常。\n"
                        f"【解决】把 reference_video_policy 改为 model_minimum，"
                        f"strict_prompt_tags 改为 false 后重试。"
                    )
                if not isinstance(seg_cond, list) or len(seg_cond) == 0:
                    raise ValueError(t("err_invalid_segment", ordinal=ordinal))

                if not isinstance(seg_cond[0], (list, tuple)):
                    raise ValueError(
                        f"段_{ordinal} 格式异常：不是标准 conditioning 列表（got {type(seg_cond[0]).__name__}）。"
                        f"请确保连接的是 YimoH3Conditioning 的第 8 个输出「段数据包」(segment)。"
                    )
                if len(seg_cond[0]) < 2:
                    raise ValueError(
                        f"段_{ordinal} 格式异常：缺少 conditioning dict（只有 tensor，无 dict）。"
                        f"这通常是因为误接了「正向条件」(positive) 或「负向条件」(negative) 输出。"
                        f"请确保连接的是 YimoH3Conditioning 的第 8 个输出「段数据包」(segment)。"
                    )

                cond_dict = seg_cond[0][1]
                if not isinstance(cond_dict, dict):
                    raise ValueError(
                        f"段_{ordinal} 格式异常：conditioning 的第二项不是 dict（got {type(cond_dict).__name__}）。"
                        f"请确保连接的是 YimoH3Conditioning 的第 8 个输出「段数据包」(segment)。"
                    )

                if not cond_dict.get("_yimo_segment"):
                    if "pooled_output" in cond_dict or "minimax_payload" in cond_dict:
                        raise ValueError(
                            f"段_{ordinal} 检测到普通 conditioning 数据（缺少 _yimo_segment 标记）。"
                            f"请确保连接的是 YimoH3Conditioning 的第 8 个输出「段数据包」(segment)，"
                            f"而非「正向条件」(positive) 或「负向条件」(negative)。"
                        )
                    raise ValueError(t("err_not_segment_bundle", ordinal=ordinal))

                data = cond_dict.get("_yimo_data", {})
                parsed_segments.append({
                    "positive": data.get("positive"),
                    "negative": data.get("negative"),
                    "latent": data.get("latent"),
                    "audio": data.get("audio"),
                    "frame_count": data.get("frame_count", 124),
                    "render_frames": data.get("render_frames", 124),
                    "prompt": data.get("prompt", ""),
                    "mode": data.get("mode", "text"),
                    "keyframe_specs": data.get("keyframe_specs", []),
                    "ref_blocks": data.get("ref_blocks", []),
                    "identity_refs": data.get("identity_refs", []),
                    "style_refs": data.get("style_refs", []),
                    "identity_ordinals": data.get("identity_ordinals", []),
                    "audio_policy": data.get("audio_policy", "generate_new"),
                    "width": data.get("width", 1344),
                    "height": data.get("height", 768),
                    "ref_curve": data.get("ref_curve", {}),
                })

            if len(parsed_segments) == 0:
                raise ValueError(t("err_no_valid_segments"))

            # override_ref_curve 支持 inherit
            override_report = ""
            if override_ref_curve:
                overridden_count = 0
                skipped_count = 0
                for i, seg in enumerate(parsed_segments):
                    positive = seg.get("positive")
                    if not isinstance(positive, list) or len(positive) == 0:
                        continue
                    first_item = positive[0]
                    if not isinstance(first_item, (list, tuple)) or len(first_item) < 2:
                        continue
                    cond_tensor, cond_dict = first_item[0], first_item[1]
                    if not isinstance(cond_dict, dict):
                        continue

                    new_cond_dict = dict(cond_dict)
                    new_cond_dict = _isolate_payload_cond(new_cond_dict)

                    curve_cfg = dict(new_cond_dict.get("_yimo_ref_curve", {}))
                    changed = False
                    if ref_curve_direction_override != "inherit":
                        curve_cfg["direction"] = ref_curve_direction_override
                        changed = True
                    if ref_curve_shape_override != "inherit":
                        curve_cfg["shape"] = ref_curve_shape_override
                        changed = True
                    new_cond_dict["_yimo_ref_curve"] = curve_cfg

                    seg["positive"] = [[cond_tensor, new_cond_dict]] + list(positive[1:])
                    if changed:
                        overridden_count += 1
                    else:
                        skipped_count += 1

                override_report = (
                    f"override_ref_curve=True: overrode {overridden_count} segment(s) "
                    f"(direction={ref_curve_direction_override}, shape={ref_curve_shape_override}); "
                    f"skipped {skipped_count} (both inherit)"
                )

            base_latent = parsed_segments[0]["latent"]
            base_video, base_audio = None, None
            try:
                base_video, base_audio = nested_av_parts(base_latent)
            except Exception:
                pass

            for i, seg in enumerate(parsed_segments):
                if seg["positive"] is None or seg["latent"] is None:
                    raise ValueError(t("err_segment_incomplete", index=i))
                if base_video is not None:
                    try:
                        v, a = nested_av_parts(seg["latent"])
                        base_spatial = tuple(base_video.shape[-2:])
                        curr_spatial = tuple(v.shape[-2:])
                        if curr_spatial != base_spatial:
                            seg_w = seg.get("width", "unknown")
                            seg_h = seg.get("height", "unknown")
                            base_w = parsed_segments[0].get("width", "unknown")
                            base_h = parsed_segments[0].get("height", "unknown")

                            diag = (
                                f"Spatial mismatch detected: "
                                f"segment_0 latent_spatial={base_spatial} (wh={base_w}x{base_h}) vs "
                                f"segment_{i} latent_spatial={curr_spatial} (wh={seg_w}x{seg_h}). "
                            )

                            if curr_spatial == (base_spatial[1], base_spatial[0]):
                                diag += (
                                    f"HINT: segment_{i} width/height appears swapped "
                                    f"(latent HxW = {curr_spatial[0]}x{curr_spatial[1]} vs base {base_spatial[0]}x{base_spatial[1]}). "
                                    f"Check if width/height are reversed in conditioning node."
                                )
                            else:
                                diag += (
                                    f"Possible causes: 1) auto_resolution=True with different input image sizes; "
                                    f"2) manual width/height values differ between segments; "
                                    f"3) one segment entered error_result path (defaults to 1344x768)."
                                )
                            raise ValueError(diag)
                    except Exception as e:
                        if "Spatial mismatch detected" in str(e):
                            raise
                        raise ValueError(t("err_segment_invalid", index=i, error=str(e)))

            # v2.7.2: overlap_list 负值回退修复
            # 全负 → effective_overlaps=None（下游自动用 overlap_frames）
            # 混用 → 负值单点回退，正值保留
            # 0 仍是有效的硬切标记
            effective_overlaps = None
            if overlap_list and str(overlap_list).strip():
                try:
                    parts = [p.strip() for p in str(overlap_list).split(",") if p.strip()]
                    parsed = [int(p) for p in parts]
                    if all(p < 0 for p in parsed):
                        effective_overlaps = None
                        logger.info(
                            "[YimoH3SequenceSampler] overlap_list=%r 全为负数，"
                            "视为未设置，回退到 overlap_frames=%d。",
                            overlap_list, int(overlap_frames),
                        )
                    else:
                        default_olap = int(overlap_frames)
                        effective_overlaps = [default_olap if p < 0 else p for p in parsed]
                        if any(p < 0 for p in parsed):
                            logger.info(
                                "[YimoH3SequenceSampler] overlap_list=%r 含负值，"
                                "负值单点回退到 overlap_frames=%d，结果=%r。",
                                overlap_list, default_olap, effective_overlaps,
                            )
                except ValueError as e:
                    raise ValueError(t("err_overlap_list_parse", error=e))

            parsed_segment_seeds = None
            if segment_seeds and str(segment_seeds).strip():
                parts = [p.strip() for p in str(segment_seeds).split(",") if p.strip()]
                parsed = []
                invalid_tokens = []
                for p in parts:
                    if p.lower() == "none":
                        parsed.append(None)
                        continue
                    try:
                        parsed.append(int(p))
                    except ValueError:
                        parsed.append(None)
                        invalid_tokens.append(p)

                if invalid_tokens:
                    logger.warning(
                        "[YimoH3SequenceSampler] segment_seeds 含无法解析的项 %s，"
                        "已按 'none' 处理（该段使用基础 seed 派生）。",
                        invalid_tokens,
                    )

                if any(s is not None for s in parsed):
                    parsed_segment_seeds = parsed
                else:
                    logger.warning(
                        "[YimoH3SequenceSampler] segment_seeds=%r 中没有有效整数项，忽略该输入。",
                        segment_seeds,
                    )

            face_bank = None
            if face_restore and is_connected_value(face_images):
                face_bank = build_identity_bank(face_images)
                if face_bank is None:
                    logger.warning("face_restore enabled but no valid face detected in face_images")

            effective_sampler = sampler_name
            effective_sigmas = None
            effective_steps = steps

            if sampling_mode == "pdd":
                if external_sigmas is not None:
                    effective_sigmas = external_sigmas
                else:
                    effective_sigmas = build_pdd_sigmas(model, num_steps=32, block_size=4)
                effective_sampler = "euler"
                effective_steps = len(effective_sigmas) - 1 if effective_sigmas is not None else 8
                logger.info(
                    "[YimoH3SequenceSampler] PDD mode: sigmas length=%d, sampler=euler",
                    len(effective_sigmas) if effective_sigmas is not None else 0,
                )
            elif sampling_mode == "pdd_two_stage":
                if external_sigmas is not None:
                    effective_sigmas = external_sigmas
                else:
                    effective_sigmas = build_pdd_sigmas(model, num_steps=32, block_size=4)
                effective_sampler = "euler"
                effective_steps = len(effective_sigmas) - 1 if effective_sigmas is not None else 8
                logger.info(
                    "[YimoH3SequenceSampler] PDD two-stage mode: split_step=%d, scale_by=%.2f, "
                    "sigmas length=%d",
                    pdd_split_step, pdd_scale_by,
                    len(effective_sigmas) if effective_sigmas is not None else 0,
                )

            actual_segments = len(parsed_segments)
            if stop_after_segment >= 0:
                actual_segments = min(stop_after_segment + 1, actual_segments)
            if resample_segment >= 0 and resample_segment < len(parsed_segments):
                actual_segments = actual_segments - resample_segment
            total_steps = max(1, actual_segments * effective_steps)
            pbar = comfy.utils.ProgressBar(total_steps)

            device = comfy.model_management.get_torch_device()
            previewer = None
            try:
                latent_format = getattr(getattr(model, "model", None), "latent_format", None)
                if latent_format is None:
                    latent_format = getattr(model, "latent_format", None)
                if latent_format is not None:
                    previewer = comfy.utils.get_previewer(device, latent_format)
            except Exception:
                pass

            def _progress_callback(step, x0, x, total):
                preview_bytes = None
                if previewer is not None:
                    try:
                        if hasattr(x0, "is_nested") and x0.is_nested:
                            parts = x0.unbind()
                            video_latent = parts[0]
                            preview_bytes = previewer.decode_latent_to_preview_image(video_latent)
                        else:
                            preview_bytes = previewer.decode_latent_to_preview_image(x0)
                    except Exception:
                        pass
                pbar.update_absolute(step, total, preview_bytes)
                return preview_bytes

            face_params = {
                "strength": face_restore_strength,
                "similarity_threshold": face_similarity_threshold,
                "repair_backend": face_restore_backend,
            }

            final_latent, report_lines, segment_latents, segment_summaries = sequence_sample(
                model=model,
                segments=parsed_segments,
                overlap_frames=effective_overlaps if effective_overlaps is not None else overlap_frames,
                transition_mode=transition_mode,
                audio_continuity=audio_continuity,
                sampler_name=effective_sampler,
                scheduler=scheduler,
                steps=effective_steps,
                cfg=cfg,
                seed=seed,
                denoise=denoise,
                callback=_progress_callback,
                video_vae=video_vae,
                audio_vae=audio_vae,
                auto_inherit_tail=auto_inherit_tail,
                inherit_identity=inherit_identity,
                resample_segment=resample_segment,
                segment_seeds=parsed_segment_seeds,
                chain_id=chain_id,
                stop_after_segment=stop_after_segment,
                face_restore=face_restore and face_bank is not None,
                face_bank=face_bank,
                face_params=face_params,
                previewer=previewer,
                enable_ref_curve=enable_ref_curve,
                sigmas=effective_sigmas,
                color_match_mode=color_match_mode,
                color_match_strength=color_match_strength,
                color_match_reference_frames=color_match_reference_frames,
                sampling_mode=sampling_mode,
                pdd_split_step=pdd_split_step,
                pdd_scale_by=pdd_scale_by,
            )

            if override_report:
                report_lines.append(override_report)

            # images 输出格式，按 overlap=0 分块独立解码
            final_images = None
            if output_format == "images":
                if video_vae is None:
                    report_lines.append(
                        "output_format=images 需要 video_vae，已自动回退到 latent 输出。"
                    )
                else:
                    try:
                        final_images = cls._smart_chunked_decode(
                            final_latent, parsed_segments,
                            effective_overlaps if effective_overlaps is not None else overlap_frames,
                            overlap_frames, transition_mode, video_vae,
                            report_lines,
                            chunk_memory_threshold_mb=chunk_memory_threshold_mb,
                        )
                    except Exception as e:
                        logger.warning("smart chunked decode failed: %s", e)
                        report_lines.append(f"smart chunked decode failed: {e}")
                        final_images = None

            if auto_export_final and final_latent is not None:
                try:
                    export_dir = get_sequence_sampler_export_dir()

                    if export_name.strip():
                        filename = f"{sanitize_filename(export_name)}.pt"
                    else:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"sequence_{sanitize_filename(chain_id)}_{timestamp}.pt"

                    filepath = os.path.join(export_dir, filename)

                    export_data = {
                        "final_latent": final_latent,
                        "metadata": {
                            "chain_id": chain_id,
                            "total_segments": len(parsed_segments),
                            "segments_sampled": len(segment_latents),
                            "overlap_frames": overlap_frames,
                            "overlap_list": overlap_list,
                            "resolved_overlaps": (
                                list(effective_overlaps) if effective_overlaps is not None
                                else int(overlap_frames)
                            ),
                            "transition_mode": transition_mode,
                            "audio_continuity": audio_continuity,
                            "sampler": effective_sampler,
                            "scheduler": scheduler,
                            "steps": effective_steps,
                            "cfg": cfg,
                            "seed": seed,
                            "denoise": denoise,
                            "auto_inherit_tail": auto_inherit_tail,
                            "inherit_identity": inherit_identity,
                            "resample_segment": resample_segment,
                            "stop_after_segment": stop_after_segment,
                            "face_restore": face_restore,
                            "face_restore_backend": face_restore_backend,
                            "enable_ref_curve": enable_ref_curve,
                            "override_ref_curve": override_ref_curve,
                            "ref_curve_direction_override": ref_curve_direction_override,
                            "ref_curve_shape_override": ref_curve_shape_override,
                            "sampling_mode": sampling_mode,
                            "used_external_sigmas": external_sigmas is not None,
                            "pdd_split_step": pdd_split_step if sampling_mode == "pdd_two_stage" else None,
                            "pdd_scale_by": pdd_scale_by if sampling_mode == "pdd_two_stage" else None,
                            "color_match_mode": color_match_mode,
                            "color_match_strength": color_match_strength,
                            "color_match_reference_frames": color_match_reference_frames,
                            "output_format": output_format,
                            "chunk_memory_threshold_mb": chunk_memory_threshold_mb,
                            "timestamp": datetime.now().isoformat(),
                            "version": "3.0.0",
                            "export_name": export_name or "(auto)",
                        }
                    }

                    export_data["segments_summary"] = []
                    for idx, summary in enumerate(segment_summaries):
                        export_data["segments_summary"].append({
                            "index": idx,
                            "mode": summary.mode,
                            "frame_count": summary.frame_count,
                            "render_frames": summary.render_frames,
                            "tail_overlap_frames": summary.tail_overlap_frames,
                            "audio_policy": summary.audio_policy,
                        })

                    torch.save(export_data, filepath)
                    report_lines.append(f"💾 最终结果已导出: {filepath}")
                except Exception as e:
                    logger.warning("自动导出最终结果失败: %s", e)
                    report_lines.append(f"⚠️ 自动导出失败: {e}")

            report = "\n".join(report_lines)

            if output_format == "images" and final_images is not None:
                padded = [None] * 8
                return io.NodeOutput(None, report, *padded, final_images)

            padded = segment_latents + [None] * (8 - len(segment_latents))
            return io.NodeOutput(final_latent, report, *padded, None)

        @staticmethod
        def _smart_chunked_decode(final_latent, parsed_segments, pair_overlaps_source,
                                   overlap_frames, transition_mode, video_vae,
                                   report_lines,
                                   chunk_memory_threshold_mb=_CHUNK_MEMORY_THRESHOLD_DEFAULT_MB):
            """把已拼接的 AV Latent 按 overlap=0 边界分组，组内独立解码，最终像素拼接。

            v2.7.1: 内存阈值参数化（默认 4096 MB）。
            """
            import os as _os
            import shutil as _shutil
            import tempfile as _tempfile
            import comfy.model_management as _mm

            from .sampler_core import _overlap_to_latent

            threshold_bytes = int(chunk_memory_threshold_mb) * 1024 * 1024

            n_pairs = max(0, len(parsed_segments) - 1)
            if isinstance(pair_overlaps_source, list):
                pair_overlaps = list(pair_overlaps_source)[:n_pairs]
                while len(pair_overlaps) < n_pairs:
                    pair_overlaps.append(0)
            else:
                pair_overlaps = [int(pair_overlaps_source)] * n_pairs

            if transition_mode != "context_inject":
                pair_overlaps = [0] * n_pairs

            chunks = []
            start = 0
            for i, ov in enumerate(pair_overlaps):
                if ov == 0:
                    chunks.append((start, i + 1))
                    start = i + 1
            chunks.append((start, n_pairs + 1))

            report_lines.append("=== v2.7.2 smart_chunked_decode ===")
            report_lines.append(f"n_chunks={len(chunks)} boundaries={chunks}")
            report_lines.append(f"memory_threshold_mb={chunk_memory_threshold_mb}")

            seg_latent_ts = []
            for seg in parsed_segments:
                fc = seg.get("render_frames", seg.get("frame_count", 124))
                if fc <= 5:
                    seg_latent_ts.append(2)
                else:
                    seg_latent_ts.append(((fc - 5) // 17) * 5 + 2)

            trimmed_latent_ts = []
            for i, lt in enumerate(seg_latent_ts):
                if i == 0:
                    trimmed_latent_ts.append(lt)
                else:
                    ov = pair_overlaps[i - 1]
                    _, v_olap_t, _ = _overlap_to_latent(ov) if ov > 0 else (0, 0, 0)
                    v_olap_t = min(v_olap_t, lt)
                    trimmed_latent_ts.append(lt - v_olap_t)

            cum = [0]
            for lt in trimmed_latent_ts:
                cum.append(cum[-1] + lt)

            from .core import nested_av_parts
            final_video, _ = nested_av_parts(final_latent)

            tmp_root = _os.path.join(get_yimo_output_dir(), "_seq_chunk_cache")
            _os.makedirs(tmp_root, exist_ok=True)
            tmp_dir = None

            chunk_storage: list[tuple[str, object]] = []
            in_memory_bytes = 0
            spilled = False

            def _ensure_tmp_dir():
                nonlocal tmp_dir
                if tmp_dir is None:
                    tmp_dir = _tempfile.mkdtemp(prefix="yimo_seq_", dir=tmp_root)
                return tmp_dir

            try:
                for ci, (cs, ce) in enumerate(chunks):
                    c_t0 = cum[cs]
                    c_t1 = cum[ce]
                    if c_t1 <= c_t0:
                        continue
                    chunk_video = final_video[:, :, c_t0:c_t1].contiguous()
                    decoded = video_vae.decode(chunk_video)
                    if decoded.ndim == 5:
                        if decoded.shape[0] == 1:
                            decoded = decoded.squeeze(0)
                        elif decoded.shape[1] == 1:
                            decoded = decoded.squeeze(1)
                    if decoded.ndim != 4:
                        raise ValueError(f"chunk_{ci} decode shape 异常: {tuple(decoded.shape)}")
                    if decoded.shape[-1] > 3:
                        decoded = decoded[..., :3]
                    decoded = decoded.clamp(0.0, 1.0)

                    decoded_cpu = decoded.detach().to("cpu").contiguous()
                    chunk_bytes = decoded_cpu.numel() * decoded_cpu.element_size()

                    if in_memory_bytes + chunk_bytes <= threshold_bytes:
                        chunk_storage.append(("mem", decoded_cpu))
                        in_memory_bytes += chunk_bytes
                        report_lines.append(
                            f"chunk_{ci}: latent_t=[{c_t0},{c_t1}) frames={decoded_cpu.shape[0]} "
                            f"kept_in_memory ({chunk_bytes / 1024**2:.1f} MB)"
                        )
                    else:
                        _ensure_tmp_dir()
                        path = _os.path.join(tmp_dir, f"chunk_{ci:03d}.pt")
                        torch.save(decoded_cpu, path)
                        chunk_storage.append(("disk", path))
                        spilled = True
                        report_lines.append(
                            f"chunk_{ci}: latent_t=[{c_t0},{c_t1}) frames={decoded_cpu.shape[0]} "
                            f"spilled_to_disk ({chunk_bytes / 1024**2:.1f} MB)"
                        )

                    del decoded, decoded_cpu, chunk_video
                    try:
                        _mm.soft_empty_cache()
                    except Exception:
                        pass

                pixel_parts = []
                for kind, payload in chunk_storage:
                    if kind == "mem":
                        pixel_parts.append(payload)
                    else:
                        pixel_parts.append(
                            torch.load(payload, map_location="cpu", weights_only=True)
                        )
                final_images = torch.cat(pixel_parts, dim=0)
                report_lines.append(
                    f"final_images_frames={final_images.shape[0]} "
                    f"(in_memory_chunks={sum(1 for k, _ in chunk_storage if k == 'mem')}, "
                    f"disk_chunks={sum(1 for k, _ in chunk_storage if k == 'disk')})"
                )
                return final_images

            finally:
                if spilled and tmp_dir is not None:
                    try:
                        _shutil.rmtree(tmp_dir, ignore_errors=True)
                    except Exception:
                        pass

except Exception as _e:
    print("=" * 70, file=sys.stderr)
    print("YimoH3SequenceSampler MODULE LOAD FAILED:", _e, file=sys.stderr)
    traceback.print_exc()
    print("=" * 70, file=sys.stderr)
    raise