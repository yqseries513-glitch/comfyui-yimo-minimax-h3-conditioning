from __future__ import annotations

import logging
import time
import os
import torch
from datetime import datetime

import comfy.samplers
from comfy_api.latest import io

from .sampler_core import (
    _extract_tail_context,
    _inject_context,
    _sample_single_segment,
    _sample_single_segment_pdd_two_stage,
    _resolve_keyframes_absolute,
    _isolate_payload_cond,
    _apply_transition_to_conditioning,
    _build_prev_summary_from_latent,
    _overlap_to_latent,
    build_pdd_sigmas,
)
from .color_match_utils import apply_color_match
from .core import (
    nested_av_parts,
    is_connected_value,
    get_single_sampler_dir,
    sanitize_filename,
)
from .face_restore import build_identity_bank, restore_sampled_av
from .i18n import t

logger = logging.getLogger("YimoH3")

CATEGORY = t("category")


class YimoH3SingleSampler(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        try:
            sampler_names = comfy.samplers.KSampler.SAMPLERS
        except AttributeError:
            sampler_names = ["euler", "euler_ancestral", "heun", "dpmpp_2m", "dpmpp_2m_sde",
                             "dpmpp_sde", "dpmpp_2s_ancestral", "dpm_fast", "dpm_adaptive",
                             "ddim", "uni_pc", "uni_pc_bh2"]
        try:
            scheduler_names = comfy.samplers.KSampler.SCHEDULERS
        except AttributeError:
            scheduler_names = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]

        return io.Schema(
            node_id="YimoH3SingleSampler",
            display_name=t("singlesampler_display_name"),
            description=t("singlesampler_description"),
            category=CATEGORY,
            inputs=[
                io.Model.Input("model", tooltip="MiniMax H3 DiT model."),
                io.Vae.Input("video_vae", optional=True, tooltip=t("video_vae_tooltip")),
                io.Vae.Input("audio_vae", optional=True, tooltip=t("audio_vae_tooltip")),

                io.Conditioning.Input("segment", optional=False,
                    tooltip="来自 YimoH3Conditioning 的第 8 个输出「段数据包」。切勿误接 Positive/Negative。"),
                io.Latent.Input("prev_sampled_latent", optional=True,
                    tooltip="前段完整采样结果。连接后启用上下文注入与尾帧继承。"),

                io.Combo.Input("sampling_mode",
                    options=["standard", "pdd", "pdd_two_stage"],
                    default="standard",
                    tooltip="standard=标准采样；pdd=PDD 单阶段加速；"
                            "pdd_two_stage=PDD 双采（LOW → 插值放大 → HIGH，内部自行放大）。"),
                io.Custom("SIGMAS").Input("external_sigmas", optional=True,
                    tooltip="PDD 模式的 sigma 轨迹。如果不连接，插件会自动生成。"),

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
                    tooltip="与前段的重叠帧数。会在 17n+5 网格上按最近邻映射：\n"
                            "  0        → 无注入（视频不衔接）\n"
                            "  1~5      → effective 5（极短衔接，仅适合纯动作衔接，"
                            "换装/换风格场景可能失效，会被 DiT 当作全局参考模板）\n"
                            "  6~22     → effective 22（标准衔接，推荐值）\n"
                            "  23 以上  → 按最近邻映射到 39 / 56 / ...\n"
                            "换装、换风格场景请使用 >= 22。"),

                io.Combo.Input("transition_mode",
                    options=["context_inject", "hard_concat"],
                    default="context_inject",
                    tooltip="视频连续性：\n"
                            "context_inject=从 prev_sampled_latent 注入视频 tail（视频连贯）\n"
                            "hard_concat=不注入，各段视频独立（视频不连贯）"),

                io.Combo.Input("audio_continuity",
                    options=["context_inject", "break"],
                    default="context_inject",
                    advanced=False,
                    tooltip="音频连续性：\n"
                            "context_inject=从 prev_sampled_latent 注入音频 tail（音频连贯）\n"
                            "break=不注入，各段音频独立（音频不连贯）\n"
                            "注意：需要 overlap_frames>0 且 prev_sampled_latent 已连接。"),

                io.Boolean.Input("auto_inherit_tail", default=True, advanced=True,
                    tooltip="自动将前段尾帧解码后作为本段 first_frame 插入 conditioning。"
                            "本段用户显式提供 first_frame 时用户首帧优先。"
                            "注意：仅在 transition_mode=hard_concat 或 effective_overlap=0 时生效。"),

                io.Boolean.Input("inherit_identity", default=True, advanced=True,
                    tooltip="跨段继承身份参考图。\n"
                            "从 prev_sampled_latent 的 _yimo_prev_meta 读取上游 identity_refs / style_refs，"
                            "注入到当前段的 minimax_refs。"
                            "仅当当前段模式为 references / hybrid 时生效。"),

                io.Boolean.Input("enable_ref_curve", default=False, advanced=True,
                    tooltip="启用参考强度时间曲线。单段时使用段中点强度。"),

                io.Boolean.Input("override_ref_curve", default=False, advanced=True,
                    tooltip="启用后，本节点将覆盖 segment 中携带的曲线配置。"),
                io.Combo.Input("ref_curve_direction_override",
                    options=["inherit", "constant", "concept_at_start", "concept_at_end", "concept_at_middle", "concept_at_ends"],
                    default="inherit", advanced=True,
                    tooltip="覆盖 segment 中的曲线方向。inherit=使用上游配置。"),
                io.Combo.Input("ref_curve_shape_override",
                    options=["inherit", "linear", "ease", "sigmoid", "exponential", "quadratic", "cubic"],
                    default="inherit", advanced=True,
                    tooltip="覆盖 segment 中的曲线形状。inherit=使用上游配置。"),

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
                    tooltip="随机种子。"),

                io.Image.Input("face_images", optional=True,
                    tooltip="面部参考图。支持单图/多图/批次，用于身份匹配和多人修复。"),
                io.Boolean.Input("face_restore", default=False, advanced=True,
                    tooltip="启用采样中面部修复。"),
                io.Combo.Input("face_restore_backend",
                    options=["auto", "gfpgan", "insightface", "opencv"],
                    default="auto", advanced=True,
                    tooltip="选择面部修复后端。"),
                io.Float.Input("face_restore_strength", default=0.7, min=0.0, max=1.0, step=0.05,
                    advanced=True, tooltip="面部修复强度。"),
                io.Float.Input("face_similarity_threshold", default=0.45, min=0.0, max=1.0, step=0.01,
                    advanced=True, tooltip="ArcFace 余弦匹配阈值。"),

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

                io.Boolean.Input("auto_save", default=False, advanced=True,
                    tooltip="是否自动保存采样结果。"),
                io.String.Input("save_name", default="", advanced=True,
                    tooltip="保存文件名。"),
                io.String.Input("project_name", default="", advanced=True,
                    tooltip="项目名称。"),
            ],
            outputs=[
                io.Latent.Output(display_name=t("singlesampler_output_latent")),
                io.String.Output(display_name=t("singlesampler_output_report")),
            ],
        )

    @classmethod
    def execute(cls, model, video_vae=None, audio_vae=None,
                segment=None, prev_sampled_latent=None,
                sampling_mode="standard", external_sigmas=None,
                pdd_split_step=4, pdd_scale_by=1.5,
                overlap_frames=22, transition_mode="context_inject",
                audio_continuity="context_inject",
                auto_inherit_tail=True,
                inherit_identity=True,
                enable_ref_curve=False,
                override_ref_curve=False, ref_curve_direction_override="inherit", ref_curve_shape_override="inherit",
                sampler_name="euler", scheduler="normal", steps=20, cfg=7.0, denoise=1.0, seed=0,
                face_images=None, face_restore=False,
                face_restore_backend="auto",
                face_restore_strength=0.7, face_similarity_threshold=0.45,
                color_match_mode="off", color_match_strength=0.7, color_match_reference_frames=1,
                auto_save=False, save_name="", project_name="",
                **kwargs):

        if model is None:
            raise ValueError("model 未连接")

        if not isinstance(segment, list) or len(segment) == 0:
            raise ValueError("segment 不是有效的段数据包；请连接 YimoH3Conditioning 的 segment 输出")

        if not isinstance(segment[0], (list, tuple)) or len(segment[0]) < 2:
            raise ValueError("segment 格式异常：不是标准 conditioning 列表。")

        cond_dict = segment[0][1]
        if not isinstance(cond_dict, dict):
            raise ValueError("segment 格式异常：conditioning 的第二项不是 dict。")

        if not cond_dict.get("_yimo_segment"):
            raise ValueError("segment 缺少 _yimo_segment 标记；请确保连接的是 YimoH3Conditioning 的 segment 输出")

        data = cond_dict.get("_yimo_data", {})
        positive = data.get("positive")
        negative = data.get("negative")
        latent = data.get("latent")
        frame_count = data.get("frame_count", 124)
        render_frames = data.get("render_frames", frame_count)
        mode = data.get("mode", "text")
        keyframe_specs = data.get("keyframe_specs", [])
        identity_refs = data.get("identity_refs", [])
        style_refs = data.get("style_refs", [])
        identity_ordinals = data.get("identity_ordinals", [])
        audio_policy = data.get("audio_policy", "generate_new")

        if positive is None or latent is None:
            raise ValueError("segment 数据包缺少正向条件或 AV 潜变量")

        if isinstance(positive, list) and len(positive) > 0:
            isolated_positive = []
            for item in positive:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    cond_tensor, cd = item[0], item[1]
                    if isinstance(cd, dict):
                        new_cd = dict(cd)
                        new_cd = _isolate_payload_cond(new_cd)
                        isolated_positive.append([cond_tensor, new_cd])
                    else:
                        isolated_positive.append(item)
                else:
                    isolated_positive.append(item)
            positive = isolated_positive

        # 计算 effective overlap
        if overlap_frames > 0:
            effective_overlap_frames, _, _ = _overlap_to_latent(overlap_frames)
        else:
            effective_overlap_frames = 0

        report_lines = [
            "=== Yimo H3 Single Sampler v3.0.0 ===",
            f"mode={mode}, frame_count={frame_count}, render_frames={render_frames}",
            f"sampling_mode={sampling_mode}",
            f"overlap_frames(requested)={overlap_frames}, "
            f"effective_overlap={effective_overlap_frames}",
            f"transition_mode(video)={transition_mode}",
            f"audio_continuity={audio_continuity}",
            f"auto_inherit_tail={auto_inherit_tail}",
            f"inherit_identity={inherit_identity}",
            f"enable_ref_curve={enable_ref_curve}",
            f"override_ref_curve={override_ref_curve}",
            f"sampler={sampler_name}/{scheduler}, steps={steps}, cfg={cfg}, denoise={denoise}, seed={seed}",
            f"face_restore={face_restore}, backend={face_restore_backend}",
            f"color_match_mode={color_match_mode}, strength={color_match_strength}, reference_frames={color_match_reference_frames}",
            f"auto_save={auto_save}, save_name={save_name or '(auto)'}, project={project_name or 'default'}",
        ]

        effective_sampler = sampler_name
        effective_sigmas = None
        effective_steps = steps

        if sampling_mode == "pdd":
            if external_sigmas is not None:
                effective_sigmas = external_sigmas
                report_lines.append(f"PDD mode: using external_sigmas (length={len(external_sigmas)})")
            else:
                effective_sigmas = build_pdd_sigmas(model, num_steps=32, block_size=4)
                report_lines.append(f"PDD mode: using internal sigmas (length={len(effective_sigmas)})")
            effective_sampler = "euler"
            effective_steps = len(effective_sigmas) - 1 if effective_sigmas is not None else 8

        elif sampling_mode == "pdd_two_stage":
            if external_sigmas is not None:
                effective_sigmas = external_sigmas
                report_lines.append(f"PDD two-stage mode: using external_sigmas (length={len(effective_sigmas)})")
            else:
                effective_sigmas = build_pdd_sigmas(model, num_steps=32, block_size=4)
                report_lines.append(f"PDD two-stage mode: using internal sigmas (length={len(effective_sigmas)})")
            effective_sampler = "euler"
            effective_steps = len(effective_sigmas) - 1 if effective_sigmas is not None else 8
            report_lines.append(
                f"PDD two-stage: split_step={pdd_split_step}, scale_by={pdd_scale_by} "
                f"(LOW → bilinear upscale → HIGH)"
            )

        else:
            report_lines.append("standard mode: using internal scheduler")

        # 提取 prev_context
        prev_context = None
        if (
            prev_sampled_latent is not None
            and overlap_frames > 0
            and (transition_mode == "context_inject" or audio_continuity == "context_inject")
        ):
            prev_context = _extract_tail_context(prev_sampled_latent, overlap_frames)
            if prev_context is not None:
                report_lines.append(
                    f"extracted tail context: requested={prev_context['requested_overlap_frames']}, "
                    f"effective={prev_context['effective_overlap_frames']}, "
                    f"actual={prev_context['overlap_frames']} "
                    f"(video_latent_t={prev_context['tail_latent_t']}, "
                    f"audio_latent_t={prev_context['tail_audio_t']})"
                )
            else:
                report_lines.append("prev_sampled_latent provided but tail extraction failed")

        # 独立控制视频/音频注入
        latent, ctx_report = _inject_context(
            latent,
            prev_context,
            video_continuity=transition_mode,
            audio_continuity=audio_continuity,
        )
        report_lines.append(ctx_report)

        # 构造 prev_summary（从 prev_latent 提取 refs 元数据）
        prev_summary = None
        if prev_sampled_latent is not None:
            prev_summary = _build_prev_summary_from_latent(
                prev_sampled_latent, video_vae, audio_vae,
                mode=mode, frame_count=frame_count,
            )
            if prev_summary is not None and (prev_summary.identity_refs or prev_summary.style_refs):
                report_lines.append(
                    f"prev_summary loaded: identity_refs={len(prev_summary.identity_refs)}, "
                    f"style_refs={len(prev_summary.style_refs)}"
                )

        # 关键帧移位
        if transition_mode == "context_inject" and effective_overlap_frames > 0:
            keyframe_shift = int(effective_overlap_frames)
        else:
            keyframe_shift = 0

        # auto_inherit_tail 生效条件
        effective_auto_inherit = (
            auto_inherit_tail
            and (transition_mode == "hard_concat" or effective_overlap_frames <= 0)
        )

        inherit_refs = bool(inherit_identity)

        positive, trans_report = _apply_transition_to_conditioning(
            positive,
            keyframe_specs,
            frame_count,
            keyframe_shift,
            effective_auto_inherit,
            prev_summary,
            video_vae,
            inherit_refs,
            identity_refs=prev_summary.identity_refs if prev_summary else [],
            style_refs=prev_summary.style_refs if prev_summary else [],
            curr_mode=mode,
        )
        report_lines.append(f"transition: {trans_report}")

        if auto_inherit_tail and prev_summary is not None and not effective_auto_inherit:
            report_lines.append(
                "auto_inherit_tail 已跳过：context_inject + effective_overlap>0 时，"
                "尾帧通过 latent 注入实现连续，无需再插入 first_frame@0。"
            )

        if override_ref_curve:
            curve_dir = ref_curve_direction_override if ref_curve_direction_override != "inherit" else "constant"
            curve_shape = ref_curve_shape_override if ref_curve_shape_override != "inherit" else "linear"
            if isinstance(positive, list) and len(positive) > 0:
                cond_dict_local = positive[0][1] if isinstance(positive[0], (list, tuple)) and len(positive[0]) > 1 else {}
                if isinstance(cond_dict_local, dict):
                    curve_cfg = dict(cond_dict_local.get("_yimo_ref_curve", {}))
                    curve_cfg["direction"] = curve_dir
                    curve_cfg["shape"] = curve_shape
                    cond_dict_local["_yimo_ref_curve"] = curve_cfg
                    report_lines.append(f"override_ref_curve: direction={curve_dir}, shape={curve_shape}")

        import comfy.utils
        import comfy.model_management

        pbar = comfy.utils.ProgressBar(effective_steps)
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

        _step_start = [time.time()]
        def _callback(step, x0, x, total_steps):
            now = time.time()
            elapsed = now - _step_start[0]
            _step_start[0] = now
            if total_steps <= 10 or step == 0 or step == total_steps - 1 or (step + 1) % 5 == 0:
                logger.info("[YimoH3SingleSampler] sampling step %d / %d (%.2fs/step)", step + 1, total_steps, elapsed)
            preview_bytes = None
            if previewer is not None:
                try:
                    if hasattr(x0, "is_nested") and x0.is_nested:
                        parts = x0.unbind()
                        video_latent = parts[0]
                        preview_bytes = previewer.decode_latent_to_preview_image(video_latent)
                    else:
                        preview_bytes = previewer.decode_latent_to_preview_image(x0)
                except Exception as e:
                    logger.debug("Preview decode failed: %s", e)
            pbar.update_absolute(step + 1, total_steps, preview_bytes)
            return preview_bytes

        # ---- 采样：双阶段 or 单阶段 ----
        if sampling_mode == "pdd_two_stage":
            if effective_sigmas is None:
                effective_sigmas = build_pdd_sigmas(model, num_steps=32, block_size=4)
            sampled = _sample_single_segment_pdd_two_stage(
                model=model,
                seed=seed,
                cfg=cfg,
                sampler_name=effective_sampler,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent=latent,
                full_sigmas=effective_sigmas,
                split_step=pdd_split_step,
                scale_by=pdd_scale_by,
                denoise=denoise,
                callback=_callback,
                disable_pbar=False,
                segment_offset=0,
                total_video_frames=frame_count,
            )
        else:
            sampled = _sample_single_segment(
                model, seed, effective_steps, cfg,
                effective_sampler, scheduler,
                positive, negative, latent,
                denoise=denoise,
                callback=_callback,
                disable_pbar=False,
                segment_offset=0,
                total_video_frames=frame_count,
                sigmas=effective_sigmas,
            )

        if face_restore and is_connected_value(face_images) and video_vae is not None:
            bank = build_identity_bank(face_images)
            if bank:
                try:
                    sampled, frep = restore_sampled_av(
                        sampled, video_vae, bank,
                        strength=face_restore_strength,
                        similarity_threshold=face_similarity_threshold,
                        repair_backend=face_restore_backend,
                    )
                    report_lines.append(f"face_restore: {frep}")
                except Exception as e:
                    logger.warning("面部修复失败: %s", e)
                    report_lines.append(f"face_restore 失败: {e}")
            else:
                report_lines.append("face_restore: face_images 未检测到人脸，跳过")
        elif face_restore and not is_connected_value(face_images):
            report_lines.append("face_restore: 需要连接 face_images，本次跳过")
        elif face_restore and video_vae is None:
            report_lines.append("face_restore: 需要连接 video_vae，本次跳过")

        color_match_info = {
            "mode": color_match_mode,
            "strength": float(color_match_strength),
            "reference_frames": int(color_match_reference_frames),
            "status": "skipped (no prev_sampled_latent)",
            "reference_source": "none",
            "matched_frames": 0,
            "match_level": "none",
        }
        if prev_sampled_latent is not None and color_match_mode != "off":
            sampled, color_match_info = apply_color_match(
                sampled=sampled,
                reference=prev_sampled_latent,
                mode=color_match_mode,
                strength=color_match_strength,
                reference_frames=color_match_reference_frames,
                video_vae=video_vae,
            )
            color_match_info["reference_source"] = "prev_sampled_latent"
        elif color_match_mode == "off":
            color_match_info["status"] = "disabled"

        report_lines.append("--- color match ---")
        report_lines.append(f"mode={color_match_info['mode']}")
        report_lines.append(f"strength={color_match_info['strength']:.2f}")
        report_lines.append(f"reference_frames={color_match_info['reference_frames']}")
        report_lines.append(f"status={color_match_info['status']}")
        report_lines.append(f"match_level={color_match_info['match_level']}")
        report_lines.append(f"matched_frames={color_match_info['matched_frames']}")

        # 附加 _yimo_prev_meta，供下游 SingleSampler 继承 refs
        sampled_out = dict(sampled)
        prev_meta = {
            "mode": mode,
            "frame_count": frame_count,
            "render_frames": render_frames,
            "audio_policy": audio_policy,
            "prompt": data.get("prompt", ""),
            "identity_refs": list(identity_refs),
            "style_refs": list(style_refs),
            "identity_ordinals": list(identity_ordinals),
        }
        sampled_out["_yimo_prev_meta"] = prev_meta

        if auto_save:
            try:
                save_dir = get_single_sampler_dir(project_name)

                if save_name.strip():
                    filename = f"{sanitize_filename(save_name)}.pt"
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"single_{mode}_{seed}_{timestamp}.pt"

                filepath = os.path.join(save_dir, filename)

                save_data = {
                    "sampled": sampled_out,
                    "metadata": {
                        "mode": mode,
                        "frame_count": frame_count,
                        "render_frames": render_frames,
                        "seed": seed,
                        "steps": effective_steps,
                        "cfg": cfg,
                        "sampler": effective_sampler,
                        "scheduler": scheduler,
                        "denoise": denoise,
                        "sampling_mode": sampling_mode,
                        "pdd_split_step": pdd_split_step if sampling_mode == "pdd_two_stage" else None,
                        "pdd_scale_by": pdd_scale_by if sampling_mode == "pdd_two_stage" else None,
                        "overlap_frames": overlap_frames,
                        "effective_overlap_frames": effective_overlap_frames,
                        "transition_mode": transition_mode,
                        "audio_continuity": audio_continuity,
                        "auto_inherit_tail": auto_inherit_tail,
                        "inherit_identity": inherit_identity,
                        "enable_ref_curve": enable_ref_curve,
                        "override_ref_curve": override_ref_curve,
                        "color_match_mode": color_match_mode,
                        "color_match_strength": color_match_strength,
                        "color_match_reference_frames": color_match_reference_frames,
                        "color_match_status": color_match_info["status"],
                        "timestamp": datetime.now().isoformat(),
                        "version": "3.0.0",
                        "project": project_name or "default",
                        "prompt": data.get("prompt", ""),
                        "save_name": save_name or "(auto)",
                    }
                }

                torch.save(save_data, filepath)
                report_lines.append(f"💾 自动保存: {filepath}")
            except Exception as e:
                logger.warning("自动保存失败: %s", e)
                report_lines.append(f"⚠️ 自动保存失败: {e}")

        report_lines.append("segment sampling complete")

        return io.NodeOutput(sampled_out, "\n".join(report_lines))