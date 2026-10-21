# -*- coding: utf-8 -*-
"""Yimo H3 AV双时钟单段采样器

节点 ID: YimoH3SingleSampler
显示名: Yimo H3 AV双时钟单段采样器
分类:   Yimo/MiniMax H3/音频
"""

from __future__ import annotations

import logging
import time
import os
from datetime import datetime

import torch

import comfy.model_management
import comfy.nested_tensor
import comfy.samplers
import comfy.utils
from comfy_api.latest import io

from .core import (
    nested_av_parts,
    is_connected_value,
    get_single_sampler_dir,
    sanitize_filename,
)
from .i18n import t
from .sampler_core import (
    _extract_tail_context,
    _inject_context,
    _sample_single_segment,
    _isolate_payload_cond,
    _apply_transition_to_conditioning,
    _build_prev_summary_from_latent,
    _overlap_to_latent,
)
from .color_match_utils import apply_color_match
from .face_restore import build_identity_bank, restore_sampled_av
from .dual_clock_core import (
    DEFAULT_SAMPLER_NAME as DUAL_CLOCK_SAMPLER_NAME,
    DEFAULT_SCHEDULER_NAME as DUAL_CLOCK_SCHEDULER_NAME,
    DEFAULT_SHIFT_AUDIO,
    DEFAULT_SHIFT_VIDEO,
    SCHEDULER_OPTIONS as DUAL_CLOCK_SCHEDULER_OPTIONS,
    setup_dual_clock_sampling,
    run_dual_clock_sample,
)

logger = logging.getLogger("YimoH3")

CATEGORY = t("category")


def _build_sampler_options() -> list[str]:
    options = [DUAL_CLOCK_SAMPLER_NAME]
    native = list(getattr(comfy.samplers, "SAMPLER_NAMES", []) or [])
    for name in native:
        if name not in options:
            options.append(name)
    return options


def _build_scheduler_options() -> list[str]:
    options = list(DUAL_CLOCK_SCHEDULER_OPTIONS)
    native = list(getattr(comfy.samplers, "SCHEDULER_NAMES", []) or [])
    for name in native:
        if name not in options:
            options.append(name)
    return options


class YimoH3SingleSampler(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        sampler_options = _build_sampler_options()
        scheduler_options = _build_scheduler_options()

        return io.Schema(
            node_id="YimoH3SingleSampler",
            display_name=t("singlesampler_display_name"),
            description=(
                "MiniMax H3 音视频单段采样器（含双时钟）。\n"
                "sampler_name=dual_clock_euler → 双时钟（video/audio 独立 sigma 轨迹）。\n"
                "sampler_name=其它 → 单时钟（ComfyUI 原生 AV 调度）。\n"
                "段间逻辑（上下文注入 / 尾帧继承 / 身份继承 / 面部修复 / 颜色匹配 / "
                "断点续传 / 参考曲线）全部保留。"
            ),
            category=CATEGORY,
            inputs=[
                io.Model.Input("model", tooltip="MiniMax H3 DiT model."),
                io.Vae.Input("video_vae", optional=True, tooltip=t("video_vae_tooltip")),

                io.Conditioning.Input(
                    "segment", optional=False,
                    tooltip="来自 YimoH3Conditioning 的「段数据包」。切勿误接 Positive/Negative。",
                ),
                io.Latent.Input(
                    "prev_sampled_latent", optional=True,
                    tooltip="前段完整采样结果。连接后启用上下文注入与尾帧继承。",
                ),

                io.Combo.Input(
                    "sampler_name",
                    options=sampler_options,
                    default=DUAL_CLOCK_SAMPLER_NAME,
                    tooltip=(
                        "dual_clock_euler = 双时钟 Euler（推荐）。\n"
                        "其它 ComfyUI 原生采样器 = 单时钟，自动降级。\n"
                        "单时钟下 shift_video / shift_audio 无效。"
                    ),
                ),
                io.Combo.Input(
                    "scheduler",
                    options=scheduler_options,
                    default=DUAL_CLOCK_SCHEDULER_NAME,
                    tooltip=(
                        "native_flow = 双时钟专用 shift 网格（推荐）。\n"
                        "beta57 = 双时钟 beta 调度器。\n"
                        "其它 = ComfyUI 内置。"
                    ),
                ),
                io.Int.Input(
                    "steps", default=20, min=1, max=100, step=1,
                    tooltip="采样步数。无加速 LoRA 时推荐 20~30。",
                ),
                io.Float.Input(
                    "shift_video", default=DEFAULT_SHIFT_VIDEO,
                    min=0.01, max=100.0, step=0.01,
                    tooltip="视频 sigma shift。默认 12.0。\n仅 dual_clock_euler 时生效。",
                ),
                io.Float.Input(
                    "shift_audio", default=DEFAULT_SHIFT_AUDIO,
                    min=0.01, max=100.0, step=0.01,
                    tooltip="音频 sigma shift。默认 3.0。\n仅 dual_clock_euler 时生效。",
                ),
                io.Float.Input(
                    "cfg", default=1.0, min=0.0, max=100.0, step=0.1,
                    tooltip="CFG 引导强度。加速 LoRA 下 1.0；无加速 LoRA 时 4~7。",
                ),
                io.Float.Input(
                    "denoise", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="降噪强度。1.0=完全重绘。",
                ),
                io.Int.Input(
                    "seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, step=1,
                    tooltip="随机种子。",
                ),

                io.Int.Input(
                    "overlap_frames", default=22, min=0, max=200, step=1,
                    tooltip="与前段的重叠帧数。17n+5 网格最近邻映射。",
                ),
                io.Combo.Input(
                    "transition_mode",
                    options=["context_inject", "hard_concat"],
                    default="context_inject",
                    tooltip="视频连续性。",
                ),
                io.Combo.Input(
                    "audio_continuity",
                    options=["context_inject", "break"],
                    default="context_inject",
                    advanced=False,
                    tooltip="音频连续性。",
                ),
                io.Boolean.Input(
                    "auto_inherit_tail", default=True, advanced=True,
                    tooltip="自动尾帧继承。",
                ),
                io.Boolean.Input(
                    "inherit_identity", default=True, advanced=True,
                    tooltip="跨段身份继承。",
                ),

                io.Boolean.Input(
                    "enable_ref_curve", default=False, advanced=True,
                    tooltip="启用参考强度时间曲线。",
                ),
                io.Boolean.Input(
                    "override_ref_curve", default=False, advanced=True,
                    tooltip="覆盖 segment 曲线配置。",
                ),
                io.Combo.Input(
                    "ref_curve_direction_override",
                    options=["inherit", "constant", "concept_at_start",
                             "concept_at_end", "concept_at_middle", "concept_at_ends"],
                    default="inherit", advanced=True,
                    tooltip="覆盖曲线方向。",
                ),
                io.Combo.Input(
                    "ref_curve_shape_override",
                    options=["inherit", "linear", "ease", "sigmoid",
                             "exponential", "quadratic", "cubic"],
                    default="inherit", advanced=True,
                    tooltip="覆盖曲线形状。",
                ),

                io.Image.Input("face_images", optional=True,
                    tooltip="面部参考图。"),
                io.Boolean.Input("face_restore", default=False, advanced=True,
                    tooltip="启用采样中面部修复。"),
                io.Combo.Input("face_restore_backend",
                    options=["auto", "gfpgan", "insightface", "opencv"],
                    default="auto", advanced=True,
                    tooltip="面部修复后端。"),
                io.Float.Input("face_restore_strength", default=0.7, min=0.0, max=1.0, step=0.05,
                    advanced=True, tooltip="面部修复强度。"),
                io.Float.Input("face_similarity_threshold", default=0.45, min=0.0, max=1.0, step=0.01,
                    advanced=True, tooltip="ArcFace 余弦匹配阈值。"),

                io.Combo.Input("color_match_mode",
                    options=["off", "low", "medium", "high", "max"],
                    default="off", advanced=True,
                    tooltip=t("color_match_mode_tooltip")),
                io.Float.Input("color_match_strength",
                    default=0.7, min=0.0, max=1.0, step=0.05, advanced=True,
                    tooltip=t("color_match_strength_tooltip")),
                io.Int.Input("color_match_reference_frames",
                    default=1, min=1, max=17, step=1, advanced=True,
                    tooltip=t("color_match_reference_frames_tooltip")),

                io.Boolean.Input("auto_save", default=False, advanced=True,
                    tooltip="是否自动保存。"),
                io.String.Input("save_name", default="", advanced=True,
                    tooltip="保存文件名。"),
                io.String.Input("project_name", default="", advanced=True,
                    tooltip="项目子文件夹名。"),
            ],
            outputs=[
                io.Latent.Output(display_name=t("singlesampler_output_latent")),
                io.String.Output(display_name=t("singlesampler_output_report")),
            ],
        )

    @classmethod
    def execute(
        cls, model, video_vae=None,
        segment=None, prev_sampled_latent=None,
        sampler_name=DUAL_CLOCK_SAMPLER_NAME,
        scheduler=DUAL_CLOCK_SCHEDULER_NAME,
        steps=20,
        shift_video=DEFAULT_SHIFT_VIDEO,
        shift_audio=DEFAULT_SHIFT_AUDIO,
        cfg=1.0,
        denoise=1.0,
        seed=0,
        overlap_frames=22,
        transition_mode="context_inject",
        audio_continuity="context_inject",
        auto_inherit_tail=True,
        inherit_identity=True,
        enable_ref_curve=False,
        override_ref_curve=False,
        ref_curve_direction_override="inherit",
        ref_curve_shape_override="inherit",
        face_images=None,
        face_restore=False,
        face_restore_backend="auto",
        face_restore_strength=0.7,
        face_similarity_threshold=0.45,
        color_match_mode="off",
        color_match_strength=0.7,
        color_match_reference_frames=1,
        auto_save=False,
        save_name="",
        project_name="",
        **kwargs,
    ):
        # ------------------------------------------------------------------
        # 1. 基础校验
        # ------------------------------------------------------------------
        if model is None:
            raise ValueError("model 未连接")
        if not isinstance(segment, list) or len(segment) == 0:
            raise ValueError("segment 不是有效的段数据包。")
        if not isinstance(segment[0], (list, tuple)) or len(segment[0]) < 2:
            raise ValueError("segment 格式异常：不是标准 conditioning 列表。")
        cond_dict = segment[0][1]
        if not isinstance(cond_dict, dict):
            raise ValueError("segment 格式异常：conditioning 第二项不是 dict。")
        if not cond_dict.get("_yimo_segment"):
            raise ValueError("segment 缺少 _yimo_segment 标记。")

        # ------------------------------------------------------------------
        # 2. 采样器分发
        # ------------------------------------------------------------------
        use_dual_clock = (sampler_name == DUAL_CLOCK_SAMPLER_NAME)

        native_samplers = set(getattr(comfy.samplers, "SAMPLER_NAMES", []) or [])
        if use_dual_clock:
            native_sampler_name = None
        elif sampler_name in native_samplers:
            native_sampler_name = sampler_name
        else:
            raise ValueError(
                f"未知的 sampler_name: {sampler_name!r}。\n"
                f"合法值：{DUAL_CLOCK_SAMPLER_NAME} 或 ComfyUI 原生采样器。"
            )

        # ------------------------------------------------------------------
        # 3. segment 数据解析
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # 4. effective overlap / 段间 / prev_summary
        # ------------------------------------------------------------------
        if overlap_frames > 0:
            effective_overlap_frames, _, _ = _overlap_to_latent(overlap_frames)
        else:
            effective_overlap_frames = 0

        report_lines = [
            "=== Yimo H3 AV双时钟单段采样器 v5.0.0 ===",
            f"mode={mode}, frame_count={frame_count}, render_frames={render_frames}",
            f"sampler_name={sampler_name} (use_dual_clock={use_dual_clock})",
            f"scheduler={scheduler}",
            f"steps={steps}, cfg={cfg}, denoise={denoise}, seed={seed}",
            f"shift_video={shift_video}, shift_audio={shift_audio} "
            f"({'生效' if use_dual_clock else '单时钟下无效'})",
            f"overlap_frames(requested)={overlap_frames}, effective={effective_overlap_frames}",
            f"transition_mode(video)={transition_mode}, audio_continuity={audio_continuity}",
            f"auto_inherit_tail={auto_inherit_tail}, inherit_identity={inherit_identity}",
            f"enable_ref_curve={enable_ref_curve}, override_ref_curve={override_ref_curve}",
            f"face_restore={face_restore}, backend={face_restore_backend}",
            f"color_match_mode={color_match_mode}, strength={color_match_strength}",
            f"auto_save={auto_save}, project={project_name or 'default'}",
        ]

        if not use_dual_clock:
            report_lines.append(
                f"note: sampler_name={sampler_name} → 单时钟（ComfyUI 原生）。"
                f"shift_video / shift_audio 无效。"
            )

        prev_context = None
        if (
            prev_sampled_latent is not None
            and overlap_frames > 0
            and (transition_mode == "context_inject" or audio_continuity == "context_inject")
        ):
            prev_context = _extract_tail_context(prev_sampled_latent, overlap_frames)
            if prev_context is not None:
                report_lines.append(
                    f"tail context: requested={prev_context['requested_overlap_frames']}, "
                    f"effective={prev_context['effective_overlap_frames']}, "
                    f"actual={prev_context['overlap_frames']}"
                )

        latent, ctx_report = _inject_context(
            latent, prev_context,
            video_continuity=transition_mode,
            audio_continuity=audio_continuity,
        )
        report_lines.append(ctx_report)

        prev_summary = None
        if prev_sampled_latent is not None:
            prev_summary = _build_prev_summary_from_latent(
                prev_sampled_latent, video_vae,
                mode=mode, frame_count=frame_count,
            )
            if prev_summary is not None and (
                prev_summary.identity_refs or prev_summary.style_refs
            ):
                report_lines.append(
                    f"prev_summary: identity_refs={len(prev_summary.identity_refs)}, "
                    f"style_refs={len(prev_summary.style_refs)}"
                )

        # ------------------------------------------------------------------
        # 5. 关键帧移位 / 尾帧继承 / refs 继承
        # ------------------------------------------------------------------
        if transition_mode == "context_inject" and effective_overlap_frames > 0:
            keyframe_shift = int(effective_overlap_frames)
        else:
            keyframe_shift = 0

        effective_auto_inherit = (
            auto_inherit_tail
            and (transition_mode == "hard_concat" or effective_overlap_frames <= 0)
        )

        positive, trans_report = _apply_transition_to_conditioning(
            positive, keyframe_specs, frame_count,
            keyframe_shift, effective_auto_inherit,
            prev_summary, video_vae, bool(inherit_identity),
            identity_refs=prev_summary.identity_refs if prev_summary else [],
            style_refs=prev_summary.style_refs if prev_summary else [],
            curr_mode=mode,
        )
        report_lines.append(f"transition: {trans_report}")

        if override_ref_curve:
            if isinstance(positive, list) and len(positive) > 0:
                first = positive[0]
                if isinstance(first, (list, tuple)) and len(first) >= 2:
                    cd_local = first[1]
                    if isinstance(cd_local, dict):
                        new_cd = dict(cd_local)
                        new_cd = _isolate_payload_cond(new_cd)
                        curve_cfg = dict(new_cd.get("_yimo_ref_curve", {}))
                        changed = False
                        if ref_curve_direction_override != "inherit":
                            curve_cfg["direction"] = ref_curve_direction_override
                            changed = True
                        if ref_curve_shape_override != "inherit":
                            curve_cfg["shape"] = ref_curve_shape_override
                            changed = True
                        if changed:
                            new_cd["_yimo_ref_curve"] = curve_cfg
                            positive = [[first[0], new_cd]] + list(positive[1:])
                            report_lines.append(
                                f"override_ref_curve: direction={curve_cfg.get('direction')}, "
                                f"shape={curve_cfg.get('shape')}"
                            )
                        else:
                            report_lines.append(
                                "override_ref_curve=True but both direction/shape=inherit."
                            )

        # ------------------------------------------------------------------
        # 6. 进度条 / previewer / callback
        # ------------------------------------------------------------------
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

        pbar = comfy.utils.ProgressBar(steps)
        step_timer = [time.time()]

        def _progress_callback(step, x0, x, total_steps):
            now = time.time()
            elapsed = now - step_timer[0]
            step_timer[0] = now
            if (total_steps <= 10 or step == 0 or step == total_steps - 1 or (step + 1) % 5 == 0):
                logger.info(
                    "[YimoH3SingleSampler] step %d/%d (%.2fs/step)",
                    step + 1, total_steps, elapsed,
                )
            preview_bytes = None
            if previewer is not None:
                try:
                    if hasattr(x0, "is_nested") and x0.is_nested:
                        parts = x0.unbind()
                        preview_bytes = previewer.decode_latent_to_preview_image(parts[0])
                    else:
                        preview_bytes = previewer.decode_latent_to_preview_image(x0)
                except Exception as e:
                    logger.debug("preview decode failed: %s", e)
            pbar.update_absolute(step + 1, total_steps, preview_bytes)
            return preview_bytes

        # ------------------------------------------------------------------
        # 7. 采样核心
        # ------------------------------------------------------------------
        sampled = None

        if use_dual_clock:
            # ============ 双时钟路径 ============
            try:
                patched_model, sampler_obj, dc_sigmas, diag, original_sampling = (
                    setup_dual_clock_sampling(
                        model=model,
                        segment_latent=latent,
                        steps=steps,
                        shift_video=shift_video,
                        shift_audio=shift_audio,
                        sampler_name=sampler_name,
                        scheduler=scheduler,
                    )
                )
            except Exception as e:
                logger.error("[DualClock] setup failed: %s", e)
                report_lines.append(f"dual clock setup failed: {type(e).__name__}: {e}")
                raise

            report_lines.append("--- dual clock diagnostics ---")
            report_lines.append(
                f"use_native_av={diag['use_native_av']}, "
                f"audio_velocity_is_raw={diag['audio_velocity_is_raw']}"
            )
            report_lines.append(f"protocol={diag['protocol']['protocol_hint']}")
            report_lines.append(
                f"sigmas_length={diag['sigmas_length']}, "
                f"sigma_first={diag['sigma_first']:.4f}, "
                f"sigma_last={diag['sigma_last']:.4f}"
            )

            noise_mask = latent.get("noise_mask")

            sampled_samples = run_dual_clock_sample(
                model=patched_model,
                sampler=sampler_obj,
                sigmas=dc_sigmas,
                positive=positive,
                negative=negative,
                latent=latent,
                cfg=cfg,
                seed=seed,
                denoise=denoise,
                noise_mask=noise_mask,
                callback=_progress_callback,
                disable_pbar=False,
                device=device,
                original_sampling=original_sampling,
            )
            sampled = {"samples": sampled_samples}
            if noise_mask is not None:
                sampled["noise_mask"] = noise_mask
            report_lines.append("dual clock standard sampling complete")

        else:
            # ============ 单时钟路径 ============
            sampled = _sample_single_segment(
                model=model, seed=seed, steps=steps, cfg=cfg,
                sampler_name=native_sampler_name, scheduler=scheduler,
                positive=positive, negative=negative, latent=latent,
                denoise=denoise, callback=_progress_callback,
                disable_pbar=False, segment_offset=0,
                total_video_frames=frame_count,
            )
            report_lines.append(
                f"single clock (native {native_sampler_name}) standard sampling complete"
            )

        # ------------------------------------------------------------------
        # 8. 面部修复（允许 bank=None，走通用修复路径）
        # ------------------------------------------------------------------
        if face_restore and video_vae is not None:
            bank = None
            if is_connected_value(face_images):
                bank = build_identity_bank(face_images)
                if bank:
                    report_lines.append(
                        f"face_restore: identity bank built ({len(bank)} faces)"
                    )
                else:
                    report_lines.append(
                        "face_restore: no identity bank "
                        "(InsightFace 不可用或参考图无人脸); "
                        "will restore ALL detected faces via restore_unmatched=True"
                    )
            else:
                report_lines.append(
                    "face_restore: face_images 未连接; "
                    "will restore ALL detected faces via restore_unmatched=True"
                )

            try:
                sampled, frep = restore_sampled_av(
                    sampled, video_vae, bank,
                    strength=face_restore_strength,
                    similarity_threshold=face_similarity_threshold,
                    repair_backend=face_restore_backend,
                )
                report_lines.append(f"face_restore: {frep}")
            except Exception as e:
                logger.warning("face_restore 失败: %s", e)
                report_lines.append(f"face_restore 失败: {e}")
        elif face_restore and video_vae is None:
            report_lines.append("face_restore: 未连接 video_vae，跳过")

        # ------------------------------------------------------------------
        # 9. 颜色匹配
        # ------------------------------------------------------------------
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
                sampled=sampled, reference=prev_sampled_latent,
                mode=color_match_mode, strength=color_match_strength,
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

        # ------------------------------------------------------------------
        # 10. _yimo_prev_meta
        # ------------------------------------------------------------------
        sampled_out = dict(sampled)
        sampled_out["_yimo_prev_meta"] = {
            "mode": mode,
            "frame_count": frame_count,
            "render_frames": render_frames,
            "audio_policy": audio_policy,
            "prompt": data.get("prompt", ""),
            "identity_refs": list(identity_refs),
            "style_refs": list(style_refs),
            "identity_ordinals": list(identity_ordinals),
            "use_dual_clock": bool(use_dual_clock),
            "shift_video": float(shift_video),
            "shift_audio": float(shift_audio),
        }

        # ------------------------------------------------------------------
        # 11. 自动保存
        # ------------------------------------------------------------------
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
                        "steps": steps,
                        "cfg": cfg,
                        "sampler": sampler_name,
                        "scheduler": scheduler,
                        "denoise": denoise,
                        "use_dual_clock": bool(use_dual_clock),
                        "shift_video": float(shift_video),
                        "shift_audio": float(shift_audio),
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
                        "version": "5.0.0",
                        "project": project_name or "default",
                        "prompt": data.get("prompt", ""),
                        "save_name": save_name or "(auto)",
                    },
                }
                torch.save(save_data, filepath)
                report_lines.append(f"💾 自动保存: {filepath}")
            except Exception as e:
                logger.warning("自动保存失败: %s", e)
                report_lines.append(f"⚠️ 自动保存失败: {e}")

        # 统一 video/audio device，避免下游 pack_latents 因 device 分叉报错。
        try:
            _v, _a = nested_av_parts(sampled_out)
            if _v.device != _a.device:
                sampled_out = dict(sampled_out)
                sampled_out["samples"] = comfy.nested_tensor.NestedTensor(
                    (_v, _a.to(device=_v.device, dtype=_v.dtype))
                )
        except Exception:
            pass

        report_lines.append("segment sampling complete")
        return io.NodeOutput(sampled_out, "\n".join(report_lines))