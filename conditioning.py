# 本模块参考 ComfyUI 官方 MiniMax H3 节点实现：
#   - comfy.model_base.MiniMaxH3.extra_conds
#   - comfy.ldm.minimax.model.PackedLayout
# 字段语义（minimax_payload / minimax_keyframes / minimax_refs / cond_video_latents 等）
# 与官方保持一致。感谢 ComfyUI 官方实现。

from __future__ import annotations

import dataclasses
import logging
import math
from typing import Any

import torch

import node_helpers
from comfy.ldm.minimax.model import PackedLayout
from comfy.model_base import MiniMaxH3 as MiniMaxH3BaseModel

from .core import (
    AUDIO_LATENT_FPS,
    CANVAS_MULTIPLE,
    FPS,
    REF_IMAGE_SHORT_EDGE,
    adapt_canvas,
    align_frame_count_down,
    align_keyframe_position,
    amplify_text_conditioning,
    empty_av_latent,
    encode_audio_once,
    fit_audio_latent,
    get_retention_value,
    is_connected_value,
    nested_av_parts,
    normalize_image_input,
    replace_audio_latent,
    resize_image,
    sorted_autogrow_items,
    sorted_autogrow_values,
    temporal_shape,
    validate_audio,
    max_intermediate_keyframes_for_frames,
)
from .i18n import t
from .prompt_tags import (
    media_map_json,
    prepare_prompt,
    detect_subject_tags,
    canonicalize_media_tags,
)


logger = logging.getLogger("YimoH3")

_WORKFLOW_MODES: dict[str, dict] = {
    "text": {"task_type": "T2VA", "model_hint": "fl2va", "needs_first": False, "needs_last": False, "needs_refs": False, "description": "纯文本生成视频"},
    "first_frame": {"task_type": "I2VA", "model_hint": "fl2va", "needs_first": True, "needs_last": False, "needs_refs": False, "description": "首帧图引导生成"},
    "last_frame": {"task_type": "L2VA", "model_hint": "fl2va", "needs_first": False, "needs_last": True, "needs_refs": False, "description": "尾帧图引导生成"},
    "first_last_frame": {"task_type": "FL2VA", "model_hint": "fl2va", "needs_first": True, "needs_last": True, "needs_refs": False, "description": "首尾帧约束生成"},
    "references": {"task_type": "Ref2VA", "model_hint": "ref2va", "needs_first": False, "needs_last": False, "needs_refs": True, "description": "纯参考媒体驱动"},
    "hybrid": {"task_type": "Hybrid", "model_hint": "ref2va", "needs_first": False, "needs_last": False, "needs_refs": True, "description": "混合模式（关键帧+参考媒体）"},
}

_AUDIO_POLICIES: dict[str, dict] = {
    "keep_source": {"audio_mode": "lock_source", "add_source_as_reference": False, "needs_source": True, "default_denoise": 0.0, "description": "锁定原音频"},
    "remix_source": {"audio_mode": "remix_source", "add_source_as_reference": False, "needs_source": True, "default_denoise": None, "description": "重塑音频"},
    "reference_only": {"audio_mode": "reference_only", "add_source_as_reference": True, "needs_source": True, "default_denoise": None, "description": "音频语义参考"},
    "generate_new": {"audio_mode": "native", "add_source_as_reference": False, "needs_source": False, "default_denoise": None, "description": "完全生成新音频"},
}


@dataclasses.dataclass
class ConditioningResult:
    """v2.4.6: 统一返回值结构，避免元组解包崩溃。

    v2.6.5: to_tuple() 顺序修复 —— 与 conditioning_node.py 的 outputs 顺序对齐。
    outputs 顺序为：
      1. segment_bundle (段数据包)
      2. positive (正向条件)
      3. negative (负向条件)
      4. latent (AV 潜变量)
      5. audio (混音音频)
      6. prompt (条件化提示词)
      7. report (报告，内含 media_map JSON 块)
    """
    positive: Any
    negative: Any
    latent: dict
    audio: dict
    prompt: str
    report: str
    segment_bundle: Any | None = None
    warnings: list[str] = dataclasses.field(default_factory=list)

    def to_tuple(self) -> tuple:
        return (
            self.segment_bundle,
            self.positive,
            self.negative,
            self.latent,
            self.audio,
            self.prompt,
            self.report,
        )


def _resolve_workflow_mode(mode, first_frame, last_frame, has_refs):
    warnings = []
    mode_cfg = _WORKFLOW_MODES.get(mode)
    if mode_cfg is None:
        raise ValueError(t("err_unknown_workflow_mode", mode=mode))
    task_type = mode_cfg["task_type"]
    model_hint = mode_cfg["model_hint"]
    has_first = first_frame is not None
    has_last = last_frame is not None
    if mode_cfg["needs_first"] and not has_first:
        raise ValueError(t("err_mode_needs_first", mode=mode))
    if mode_cfg["needs_last"] and not has_last:
        raise ValueError(t("err_mode_needs_last", mode=mode))

    # v2.8.1: references 模式允许无本地 refs（可能是从 prev_segment_bundle 继承）。
    # 调用方（resolve_modes）会把 prev_segment_bundle 存在性也算入 has_refs，
    # 因此这里保留原有报错逻辑；只有真正的"既无本地素材又无继承"才会触发。
    if mode_cfg["needs_refs"] and not has_refs:
        raise ValueError(t("err_mode_needs_refs", mode=mode))

    if mode == "hybrid" and not (has_first or has_last):
        raise ValueError(t("err_hybrid_requires_frame"))
    if mode not in {"references", "hybrid"} and has_refs:
        warnings.append(t("warn_refs_ignored", mode=mode))
    return task_type, model_hint, warnings


def _resolve_audio_policy(policy, has_source_audio, audio_denoise_strength):
    warnings = []
    cfg = _AUDIO_POLICIES.get(policy)
    if cfg is None:
        raise ValueError(t("err_unknown_audio_policy", policy=policy))
    if cfg["needs_source"] and not has_source_audio:
        warnings.append(t("warn_audio_policy_downgrade", policy=policy))
        cfg = _AUDIO_POLICIES["generate_new"]
    audio_mode = cfg["audio_mode"]
    add_ref = cfg["add_source_as_reference"]
    denoise = float(cfg["default_denoise"]) if cfg["default_denoise"] is not None else float(audio_denoise_strength)
    return audio_mode, add_ref, denoise, warnings


def _resize_reference_image(image, width, height, ref_image_size):
    h, w = int(image.shape[1]), int(image.shape[2])
    scale = min(1.0, math.sqrt((width * height) / (w * h))) if ref_image_size == "match" else min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
    target_width = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    target_height = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return resize_image(image[:1], target_width, target_height), target_width, target_height


def _encode_reference_audio(audio_vae, audio):
    latent = encode_audio_once(audio_vae, audio)
    return latent, int(latent.shape[-1])


def _apply_audio_offset(encoded_audio, offset_frames):
    if offset_frames == 0:
        return encoded_audio
    offset_t = round(offset_frames * AUDIO_LATENT_FPS / FPS)
    target_t = encoded_audio.shape[-1]
    if offset_t > 0:
        padding = encoded_audio.new_zeros((*encoded_audio.shape[:-1], min(offset_t, target_t)))
        shifted = torch.cat((padding, encoded_audio[..., :-offset_t] if offset_t < target_t else encoded_audio[..., :0]), dim=-1)
    else:
        offset_t = -offset_t
        padding = encoded_audio.new_zeros((*encoded_audio.shape[:-1], min(offset_t, target_t)))
        shifted = torch.cat((encoded_audio[..., offset_t:] if offset_t < target_t else encoded_audio[..., :0], padding), dim=-1)
    if shifted.shape[-1] != target_t:
        if shifted.shape[-1] > target_t:
            shifted = shifted[..., :target_t]
        else:
            pad = encoded_audio.new_zeros((*encoded_audio.shape[:-1], target_t - shifted.shape[-1]))
            shifted = torch.cat((shifted, pad), dim=-1)
    return shifted


def _make_silent_audio(length, sample_rate=32000):
    frame_count, _, _ = temporal_shape(length)
    duration = frame_count / FPS
    samples = max(1, round(duration * sample_rate))
    waveform = torch.zeros((1, 2, samples))
    return {"waveform": waveform, "sample_rate": sample_rate}


@torch.no_grad()
def _preprocess_reference_video_gpu(frames, mode):
    if mode == "none" or frames is None or frames.numel() == 0:
        return frames
    if mode == "blur_desaturate":
        try:
            import torchvision.transforms.functional as TF
            x = frames.permute(0, 3, 1, 2)
            h, w = x.shape[-2], x.shape[-1]
            short_edge = min(h, w)
            sigma = max(16.0, short_edge / 48.0)
            kernel_size = int(6 * sigma) // 2 * 2 + 1
            x = TF.gaussian_blur(x, kernel_size=kernel_size, sigma=sigma)
            x = TF.adjust_saturation(x, saturation_factor=0.05)
            noise = torch.randn_like(x) * 0.08
            x = (x + noise).clamp(0.0, 1.0)
            x = TF.gaussian_blur(x, kernel_size=15, sigma=4.0)
            return x.permute(0, 2, 3, 1).clamp(0.0, 1.0)
        except Exception as e:
            raise RuntimeError(t("err_ref_video_gpu_preprocess", error=e))
    return frames


class _BuildContext:
    """v2.6.3: 报告版本号统一到 v2.6.3。

    v2.8.1: 新增 prev_segment_bundle 支持，可从上一段继承素材。
    """

    def __init__(self, **kwargs):
        self.workflow_mode = kwargs.get("workflow_mode", "text")
        self.audio_policy = kwargs.get("audio_policy", "generate_new")
        self.clip = kwargs.get("clip")
        self.video_vae = kwargs.get("video_vae")
        self.audio_vae = kwargs.get("audio_vae")
        self.prompt = kwargs.get("prompt", "")
        self.negative_prompt = kwargs.get("negative_prompt", "")
        self.width = kwargs.get("width", 1344)
        self.height = kwargs.get("height", 768)
        self.auto_resolution = kwargs.get("auto_resolution", False)
        self.length = kwargs.get("length", 124)
        self.first_frame = kwargs.get("first_frame")
        self.last_frame = kwargs.get("last_frame")
        self.ref_images = kwargs.get("ref_images")
        self.ref_videos = kwargs.get("ref_videos")
        self.style_audios = kwargs.get("style_audios")
        self.ref_video_audios = kwargs.get("ref_video_audios")
        self.source_audio = kwargs.get("source_audio")
        self.override_audio = kwargs.get("override_audio")
        self.ref_image_size = kwargs.get("ref_image_size", "match")
        self.reference_video_policy = kwargs.get("reference_video_policy", "official_2_to_15s")
        self.ref_video_start_frame = kwargs.get("ref_video_start_frame", 0)
        self.reference_strength = kwargs.get("reference_strength", 1.0)
        self.identity_image_indices = kwargs.get("identity_image_indices", "")
        self.strict_prompt_tags = kwargs.get("strict_prompt_tags", False)
        self.audio_offset_frames = kwargs.get("audio_offset_frames", 0)
        self.audio_denoise_strength = kwargs.get("audio_denoise_strength", 0.35)
        self.clip_video_grayout = kwargs.get("clip_video_grayout", False)
        self.ref_video_preprocessing = kwargs.get("ref_video_preprocessing", "none")
        self.keyframes = kwargs.get("keyframes")
        self.keyframe_positions = kwargs.get("keyframe_positions", "")

        self.reference_retention = kwargs.get("reference_retention", "fully_preserved")
        self.ref_curve_direction = kwargs.get("ref_curve_direction", "constant")
        self.ref_curve_shape = kwargs.get("ref_curve_shape", "linear")
        self.text_boost_strength = kwargs.get("text_boost_strength", 1.0)
        self.text_boost_mode = kwargs.get("text_boost_mode", "deviation")
        self.text_boost_renorm = kwargs.get("text_boost_renorm", True)

        # v2.8.1: 上一段的段数据包（可选）
        self.prev_segment_bundle = kwargs.get("prev_segment_bundle")

        self.compat_warnings: list[str] = []
        self.prompt_warnings: list[str] = []
        self.subject_tag_infos: list[str] = []
        self.resolved_task = ""
        self.model_hint = ""
        self.resolved_audio_mode = ""
        self.resolved_add_ref = False
        self.resolved_denoise = 0.0
        self.frame_count = 0
        self.render_frames = 0
        self.latent: dict | None = None
        self.template_audio = None
        self.keyframe_entries: list[tuple[int, Any, Any, str]] = []
        self.real_ref_items: list[dict] = []
        self.real_ref_blocks: list[dict] = []
        self.picture_labels: list[str] = []
        self.video_labels: list[str] = []
        self.audio_labels: list[str] = []
        self.keyframe_labels: list[str] = []
        self.encoded_source: Any = None
        self.source_audio_ordinal = 0
        self.identity_ordinals: set[int] = set()
        self.counts: dict[str, int] = {}
        self.conditioned_prompt = ""
        self.conditioning = None
        self.negative_conditioning = None
        self.output_audio: dict | None = None
        self.media_map = "{}"
        self.report_lines: list[str] = []
        self.keyframe_specs: list[dict] = []
        self.identity_ref_blocks: list[dict] = []
        self.style_ref_blocks: list[dict] = []
        self.is_official_fl2v_mode: bool = False
        self.port_mapping: list[dict] = []

    def filter_inputs(self):
        self.first_frame = self.first_frame if is_connected_value(self.first_frame) else None
        self.last_frame = self.last_frame if is_connected_value(self.last_frame) else None
        self.source_audio = self.source_audio if is_connected_value(self.source_audio) else None
        self.override_audio = self.override_audio if is_connected_value(self.override_audio) else None
        self.keyframes = self.keyframes if is_connected_value(self.keyframes) else None
        self.ref_images = self.ref_images if is_connected_value(self.ref_images) else None

    def filter_inputs_by_mode(self):
        mode_needs_refs = self.workflow_mode in {"references", "hybrid"}

        if not mode_needs_refs:
            if self.ref_images or self.ref_videos or self.style_audios or self.ref_video_audios:
                self.compat_warnings.append(
                    t("warn_refs_cleared_by_mode", mode=self.workflow_mode)
                )
            self.ref_images = None
            self.ref_videos = None
            self.style_audios = None
            self.ref_video_audios = None
            self.identity_image_indices = ""

        if self.workflow_mode == "references":
            if self.first_frame is not None or self.last_frame is not None or self.keyframes is not None:
                self.compat_warnings.append(
                    t("warn_keyframes_cleared_by_mode", mode=self.workflow_mode)
                )
            self.first_frame = None
            self.last_frame = None
            self.keyframes = None
            self.keyframe_positions = ""

        if self.workflow_mode == "text":
            self.first_frame = None
            self.last_frame = None
            self.keyframes = None
            self.keyframe_positions = ""

    def resolve_canvas(self):
        orig_w, orig_h = self.width, self.height
        self.width = max(CANVAS_MULTIPLE, round(self.width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        self.height = max(CANVAS_MULTIPLE, round(self.height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        if self.width != orig_w or self.height != orig_h:
            self.compat_warnings.append(
                t("warn_size_aligned", orig_w=orig_w, orig_h=orig_h, new_w=self.width, new_h=self.height, multiple=CANVAS_MULTIPLE)
            )

        if self.auto_resolution:
            suggested_w, suggested_h = None, None
            if self.first_frame is not None:
                suggested_w, suggested_h = adapt_canvas(int(self.first_frame.shape[2]), int(self.first_frame.shape[1]))
            elif self.ref_images:
                ref_vals = normalize_image_input(self.ref_images, "ref_images")
                if ref_vals:
                    suggested_w, suggested_h = adapt_canvas(int(ref_vals[0].shape[2]), int(ref_vals[0].shape[1]))
            elif self.keyframes:
                kf_vals = normalize_image_input(self.keyframes, "keyframes")
                if kf_vals:
                    suggested_w, suggested_h = adapt_canvas(int(kf_vals[0].shape[2]), int(kf_vals[0].shape[1]))
            elif self.last_frame is not None:
                suggested_w, suggested_h = adapt_canvas(int(self.last_frame.shape[2]), int(self.last_frame.shape[1]))

            if suggested_w is not None and suggested_h is not None:
                if suggested_w != self.width or suggested_h != self.height:
                    self.compat_warnings.append(
                        f"auto_resolution suggestion: input media native resolution suggests "
                        f"{suggested_w}x{suggested_h}, but honoring user-specified "
                        f"{self.width}x{self.height} as requested"
                    )

        if not 0.0 <= self.audio_denoise_strength <= 1.0:
            raise ValueError(t("err_audio_denoise_range"))

    def prepare_latent(self):
        self.render_frames = self.length
        self.latent, self.frame_count = empty_av_latent(self.width, self.height, self.length)
        _, self.template_audio = nested_av_parts(self.latent)

    def collect_references(self):
        _raw_keyframes = normalize_image_input(self.keyframes, "keyframes")
        _raw_keyframes = [(i + 1, img) for i, img in enumerate(_raw_keyframes)]

        _raw_ref_images = normalize_image_input(self.ref_images, "ref_images")
        _raw_ref_images = [(i + 1, img) for i, img in enumerate(_raw_ref_images)]

        _raw_ref_videos = sorted_autogrow_items(self.ref_videos)
        _raw_style_audios = sorted_autogrow_items(self.style_audios)

        max_kf = max_intermediate_keyframes_for_frames(self.frame_count)
        if len(_raw_keyframes) > max_kf:
            raise ValueError(
                t("err_keyframe_limits")
                + f" 当前 {self.frame_count} 帧视频最多允许 {max_kf} 个中间关键帧（已排除首/尾帧）。"
            )

        self.ref_image_list = list(_raw_ref_images)
        self.ref_image_values = [v for _, v in self.ref_image_list]
        self.ref_video_entries = [(o, v) for o, v in _raw_ref_videos]
        self.ref_video_values = [v for _, v in self.ref_video_entries]
        self.style_audio_values = [v for _, v in _raw_style_audios]
        self._raw_keyframes = _raw_keyframes

        _raw_ref_video_audios = sorted_autogrow_items(self.ref_video_audios)
        self.ref_video_audio_by_ordinal = dict(_raw_ref_video_audios)
        video_ordinals = {ordinal for ordinal, _ in self.ref_video_entries}
        orphan_soundtracks = sorted(set(self.ref_video_audio_by_ordinal) - video_ordinals)
        if orphan_soundtracks:
            raise ValueError(
                "Reference-video soundtrack(s) have no same-numbered video: "
                + ", ".join(map(str, orphan_soundtracks))
            )

        if len(self.ref_image_values) > 9 or len(self.ref_video_values) > 3 or len(self.style_audio_values) > 3:
            raise ValueError(t("err_ref_limits"))

        mode_needs_refs = self.workflow_mode in {"references", "hybrid"}
        if not mode_needs_refs and (self.ref_image_values or self.ref_video_values or self.style_audio_values):
            self.compat_warnings.append(t("warn_refs_cleared_by_mode", mode=self.workflow_mode))

        self.has_refs = bool(self.ref_image_values or self.ref_video_values or self.style_audio_values)

    def resolve_modes(self):
        # v2.8.1: 如果有 prev_segment_bundle，则 references / hybrid 模式允许本地无 refs
        has_any_refs = self.has_refs or (self.prev_segment_bundle is not None)
        self.resolved_task, self.model_hint, mode_warnings = _resolve_workflow_mode(
            self.workflow_mode, self.first_frame, self.last_frame, has_any_refs
        )
        self.compat_warnings.extend(mode_warnings)

        self.resolved_audio_mode, self.resolved_add_ref, self.resolved_denoise, audio_warnings = _resolve_audio_policy(
            self.audio_policy, self.source_audio is not None, self.audio_denoise_strength
        )
        self.compat_warnings.extend(audio_warnings)

        self.is_official_fl2v_mode = (
            self.workflow_mode in {"first_frame", "last_frame", "first_last_frame"}
            and not self._raw_keyframes
        )

    def resolve_identity_ordinals(self):
        if self.identity_image_indices.strip():
            connected_ordinals = {o for o, _ in self.ref_image_list}
            for part in self.identity_image_indices.split(","):
                try:
                    ordinal = int(part.strip())
                    if ordinal in connected_ordinals:
                        self.identity_ordinals.add(ordinal)
                    else:
                        self.compat_warnings.append(
                            t("warn_identity_not_connected", ordinal=ordinal)
                        )
                except ValueError:
                    self.compat_warnings.append(t("warn_invalid_identity_index", part=part.strip()))

    def build_keyframes(self):
        if self.is_official_fl2v_mode:
            self.keyframe_images = []
            self.keyframes_out = []
            self.keyframe_labels = []
            self.keyframe_specs = []

            if self.first_frame is not None:
                image = resize_image(self.first_frame[:1], self.width, self.height, "disabled")
                encoded = self.video_vae.encode(image)
                self.keyframe_images.append(image)
                self.keyframes_out.append({
                    "resolved_frame_index": 0,
                    "latent": encoded,
                })
                self.keyframe_labels.append("first_frame (exact frame 0)")

            if self.last_frame is not None:
                image = resize_image(self.last_frame[:1], self.width, self.height, "center")
                encoded = self.video_vae.encode(image)
                self.keyframe_images.append(image)
                self.keyframes_out.append({
                    "resolved_frame_index": self.frame_count - 1,
                    "latent": encoded,
                })
                self.keyframe_labels.append(f"last_frame (exact frame {self.frame_count - 1})")
            return

        entries = []

        if self.first_frame is not None:
            image = resize_image(self.first_frame[:1], self.width, self.height, "disabled")
            encoded = self.video_vae.encode(image)
            entries.append((0, image, encoded, "first_frame"))

        if self.last_frame is not None:
            image = resize_image(self.last_frame[:1], self.width, self.height, "center")
            encoded = self.video_vae.encode(image)
            entries.append((self.frame_count - 1, image, encoded, "last_frame"))

        if self._raw_keyframes:
            n_kf = len(self._raw_keyframes)
            kf_positions = []

            if self.keyframe_positions.strip():
                for part in self.keyframe_positions.split(","):
                    part_stripped = part.strip()
                    if not part_stripped:
                        continue
                    try:
                        kf_positions.append(int(part_stripped))
                    except ValueError:
                        raise ValueError(t("err_invalid_keyframe_position", part=part_stripped))
                if len(kf_positions) != n_kf:
                    raise ValueError(
                        t("err_keyframe_count_mismatch", n_images=n_kf, n_positions=len(kf_positions))
                    )
            else:
                occupied = {idx for idx, _, _, _ in entries}
                for i in range(1, n_kf + 1):
                    raw_pos = round((self.frame_count - 1) * i / (n_kf + 1))
                    aligned_pos = align_keyframe_position(raw_pos, self.frame_count)

                    if aligned_pos in occupied:
                        candidates = [
                            align_keyframe_position(raw_pos + sign * steps * 17, self.frame_count)
                            for steps in range(1, 20)
                            for sign in (1, -1)
                        ]
                        for c in candidates:
                            if c not in occupied and 0 < c < self.frame_count - 1:
                                aligned_pos = c
                                break
                    kf_positions.append(aligned_pos)
                    occupied.add(aligned_pos)

            kf_positions = kf_positions[:n_kf]
            occupied = {idx for idx, _, _, _ in entries}
            for (ordinal, image), pos in zip(self._raw_keyframes, kf_positions):
                aligned_pos = align_keyframe_position(pos, self.frame_count)
                if aligned_pos in occupied:
                    self.compat_warnings.append(t("warn_keyframe_conflict", ordinal=ordinal, pos=aligned_pos))
                    continue
                occupied.add(aligned_pos)
                image_resized = resize_image(image[:1], self.width, self.height, "disabled")
                encoded = self.video_vae.encode(image_resized)
                entries.append((aligned_pos, image_resized, encoded, f"keyframe_{ordinal}"))

        entries.sort(key=lambda x: x[0])
        self.keyframe_entries = entries

        self.keyframes_out = []
        self.keyframe_images = []
        self.keyframe_labels = []
        self.keyframe_specs = []
        for pos, img, enc, label in entries:
            self.keyframe_images.append(img)
            self.keyframe_labels.append(f"{label} (exact frame {pos})")
            self.keyframes_out.append({
                "resolved_frame_index": pos,
                "latent": enc,
                "latent_h": enc.shape[-2],
                "latent_w": enc.shape[-1],
                "latent_t": int(enc.shape[2]),
            })
            role = "first_frame" if label == "first_frame" else "last_frame" if label == "last_frame" else "intermediate"
            rel = 0 if role == "first_frame" else -1 if role == "last_frame" else (pos / (self.frame_count - 1) if self.frame_count > 1 else 0.5)
            self.keyframe_specs.append({
                "role": role,
                "relative_index": rel,
                "latent": enc,
                "latent_h": enc.shape[-2],
                "latent_w": enc.shape[-1],
                "latent_t": int(enc.shape[2]),
                "label": label,
            })

    def encode_ref_images(self):
        for ordinal, image in self.ref_image_list:
            resized, ref_width, ref_height = _resize_reference_image(image, self.width, self.height, self.ref_image_size)
            encoded = self.video_vae.encode(resized)
            block = {
                "kind": "image",
                "latent_h": ref_height // 16,
                "latent_w": ref_width // 16,
                "latent": encoded,
            }
            self.real_ref_items.append({"type": "image", "data": resized})
            self.real_ref_blocks.append(block)
            if ordinal in self.identity_ordinals:
                self.picture_labels.append(f"ref_image_{ordinal} (identity)")
                self.identity_ref_blocks.append(block)
            else:
                self.picture_labels.append(f"ref_image_{ordinal} (style)")
                self.style_ref_blocks.append(block)

    def encode_ref_videos(self):
        for index, (video_ordinal, frames) in enumerate(self.ref_video_entries, 1):
            if frames.ndim != 4 or frames.shape[0] < 5:
                raise ValueError(f"ref_video_{video_ordinal} must contain at least 5 IMAGE frames")
            input_frame_count = int(frames.shape[0])
            if self.reference_video_policy == "official_2_to_15s" and not (2 * FPS <= input_frame_count <= 15 * FPS):
                raise ValueError(f"ref_video_{video_ordinal} has {input_frame_count} frames; official guidance is 48-360 frames at 24fps")
            if self.ref_video_start_frame > 0:
                if self.ref_video_start_frame >= frames.shape[0]:
                    raise ValueError(f"ref_video_{video_ordinal} start_frame {self.ref_video_start_frame} exceeds total {frames.shape[0]} frames")
                frames = frames[self.ref_video_start_frame:]
            source_height, source_width = int(frames.shape[1]), int(frames.shape[2])
            canvas_width, canvas_height = adapt_canvas(source_width, source_height)
            if source_width * source_height < canvas_width * canvas_height:
                canvas_width = max(CANVAS_MULTIPLE, round(source_width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                canvas_height = max(CANVAS_MULTIPLE, round(source_height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            frames = resize_image(frames, canvas_width, canvas_height)

            if self.ref_video_preprocessing != "none":
                try:
                    frames = _preprocess_reference_video_gpu(frames, self.ref_video_preprocessing)
                except RuntimeError as e:
                    self.compat_warnings.append(t("err_gpu_preprocess_runtime", error=e))

            frames = frames[:self.frame_count]
            aligned_count = align_frame_count_down(int(frames.shape[0]))
            if aligned_count < 5:
                raise ValueError(t("err_ref_video_too_short"))
            frames = frames[:aligned_count]
            encoded_video = self.video_vae.encode(frames)

            soundtrack = self.ref_video_audio_by_ordinal.get(video_ordinal)
            encoded_soundtrack, soundtrack_t = None, 0
            if soundtrack is not None:
                encoded_soundtrack, soundtrack_t = _encode_reference_audio(self.audio_vae, soundtrack)
                self.real_ref_items.append({"type": "audio"})
                self.audio_labels.append(f"ref_video_audio_{video_ordinal}")

            sample_indices = list(range(0, frames.shape[0], FPS // 2))
            clip_frames = frames[sample_indices]
            if self.clip_video_grayout:
                gray_clip_frames = torch.full_like(clip_frames, 0.5)
                self.real_ref_items.append({
                    "type": "video",
                    "data": gray_clip_frames,
                    "timestamps": [sample_index / FPS for sample_index in sample_indices]
                })
            else:
                self.real_ref_items.append({
                    "type": "video",
                    "data": clip_frames,
                    "timestamps": [sample_index / FPS for sample_index in sample_indices]
                })
            self.real_ref_blocks.append({
                "kind": "video_audio" if soundtrack_t else "video",
                "latent_t": int(encoded_video.shape[2]),
                "latent_h": canvas_height // 16,
                "latent_w": canvas_width // 16,
                "ref_audio_t": soundtrack_t,
                "latent": encoded_video,
                "audio_latent": encoded_soundtrack,
            })
            self.video_labels.append(f"ref_video_{video_ordinal}")

    def encode_source_audio(self):
        if self.source_audio is None:
            return
        if self.audio_policy == "generate_new":
            self.compat_warnings.append(t("warn_source_audio_skipped", policy=self.audio_policy))
            self.source_audio = None
            return

        try:
            self.encoded_source = fit_audio_latent(encode_audio_once(self.audio_vae, self.source_audio), self.template_audio)
        except Exception as e:
            self.compat_warnings.append(t("err_source_audio_encode", error=e))
            self.source_audio = None
            return

        if self.audio_offset_frames != 0:
            self.encoded_source = _apply_audio_offset(self.encoded_source, self.audio_offset_frames)
            self.compat_warnings.append(
                t("warn_audio_offset_hard_trunc", offset=self.audio_offset_frames)
            )

        if self.encoded_source is not None and self.resolved_add_ref:
            self.real_ref_items.append({"type": "audio"})
            self.real_ref_blocks.append({
                "kind": "audio",
                "ref_audio_t": int(self.encoded_source.shape[-1]),
                "audio_latent": self.encoded_source,
            })
            self.source_audio_ordinal = len(self.audio_labels) + 1
            self.audio_labels.append(f"source_audio (primary source)")

    def encode_style_audios(self):
        if self.workflow_mode not in {"references", "hybrid"}:
            return
        for ordinal, audio in sorted_autogrow_items(self.style_audios):
            if self.resolved_audio_mode == "native":
                self.compat_warnings.append(t("warn_style_audio_skipped", ordinal=ordinal))
                continue
            encoded_audio, audio_t = _encode_reference_audio(self.audio_vae, audio)
            self.real_ref_items.append({"type": "audio"})
            self.real_ref_blocks.append({
                "kind": "audio",
                "ref_audio_t": audio_t,
                "audio_latent": encoded_audio,
            })
            self.audio_labels.append(f"style_audio_{ordinal}")

    # -------------------------------------------------------------------------
    # v2.8.1: 从上一段的段数据包继承素材
    # -------------------------------------------------------------------------

    def inherit_from_prev_segment(self):
        """从上一段的段数据包继承素材（image / video / audio refs）。

        触发条件（全部满足才继承）：
        1. 连了 prev_segment_bundle
        2. workflow_mode 是 references / hybrid
        3. 本段未连接任何本地素材
        4. 画布与上一段一致

        继承：real_ref_items / real_ref_blocks / identity_refs / style_refs / labels
        不继承：prompt / workflow_mode / audio_policy / 参考强度曲线 / 任何参数设定
        """
        bundle = self.prev_segment_bundle
        if bundle is None:
            return

        # 只在 references / hybrid 模式下生效
        if self.workflow_mode not in {"references", "hybrid"}:
            self.compat_warnings.append(
                f"prev_segment_bundle 已连接，但 workflow_mode={self.workflow_mode} 不支持 refs，忽略继承。"
            )
            return

        # 只在本段未连接任何素材时继承
        has_local_refs = bool(
            self.ref_image_values
            or self.ref_video_values
            or self.style_audio_values
            or self.ref_video_audio_by_ordinal
        )
        if has_local_refs:
            self.compat_warnings.append(
                "prev_segment_bundle 已连接，但本段也连接了本地素材；跳过继承（本段素材优先）。"
            )
            return

        # 解析 bundle
        if not isinstance(bundle, list) or len(bundle) == 0:
            self.compat_warnings.append("prev_segment_bundle 不是有效的 conditioning，忽略继承。")
            return
        first = bundle[0]
        if not isinstance(first, (list, tuple)) or len(first) < 2:
            self.compat_warnings.append("prev_segment_bundle[0] 格式异常，忽略继承。")
            return
        cond_dict = first[1]
        if not isinstance(cond_dict, dict) or not cond_dict.get("_yimo_segment"):
            self.compat_warnings.append("prev_segment_bundle 缺少 _yimo_segment 标记，忽略继承。")
            return

        data = cond_dict.get("_yimo_data", {})
        if not isinstance(data, dict):
            self.compat_warnings.append("prev_segment_bundle._yimo_data 格式异常，忽略继承。")
            return

        # 画布校验
        prev_w = data.get("width")
        prev_h = data.get("height")
        if prev_w is not None and prev_h is not None:
            if int(prev_w) != self.width or int(prev_h) != self.height:
                self.compat_warnings.append(
                    f"prev_segment_bundle 画布 {prev_w}x{prev_h} 与本段 {self.width}x{self.height} "
                    f"不一致，忽略继承（避免空间不匹配）。"
                )
                return

        prev_ref_blocks = data.get("ref_blocks", [])
        prev_real_ref_items = data.get("real_ref_items", [])
        prev_picture_labels = data.get("picture_labels", [])
        prev_video_labels = data.get("video_labels", [])
        prev_audio_labels = data.get("audio_labels", [])
        prev_identity_refs = data.get("identity_refs", [])
        prev_style_refs = data.get("style_refs", [])

        if not prev_ref_blocks:
            self.compat_warnings.append("prev_segment_bundle 里没有可继承的素材，忽略继承。")
            return

        # 追加到本段
        self.real_ref_blocks.extend(prev_ref_blocks)
        self.real_ref_items.extend(prev_real_ref_items)
        self.identity_ref_blocks.extend(prev_identity_refs)
        self.style_ref_blocks.extend(prev_style_refs)
        self.picture_labels.extend(f"inherited:{lb}" for lb in prev_picture_labels)
        self.video_labels.extend(f"inherited:{lb}" for lb in prev_video_labels)
        self.audio_labels.extend(f"inherited:{lb}" for lb in prev_audio_labels)

        self.compat_warnings.append(
            f"prev_segment_bundle 素材继承成功："
            f"pictures={len(prev_picture_labels)}, videos={len(prev_video_labels)}, "
            f"audios={len(prev_audio_labels)}, blocks={len(prev_ref_blocks)}。"
            f"注意：本段提示词里不要使用 <Picture N> 等引用，继承只对 DiT 端生效。"
        )

    def build_prompt(self):
        self.counts = {
            "pictures": len(self.picture_labels),
            "videos": len(self.video_labels),
            "audios": len(self.audio_labels),
        }

        self.conditioned_prompt, prompt_warnings = prepare_prompt(
            self.prompt, self.counts,
            source_audio_ordinal=self.source_audio_ordinal,
            strict=self.strict_prompt_tags
        )
        self.prompt_warnings = prompt_warnings

        all_subject_tags, non_standard_tags = detect_subject_tags(self.conditioned_prompt)
        if all_subject_tags:
            self.subject_tag_infos.append(
                f"detected {len(all_subject_tags)} semantic subject tag(s): "
                + ", ".join(all_subject_tags[:8])
                + (" ..." if len(all_subject_tags) > 8 else "")
            )
        if non_standard_tags:
            self.subject_tag_infos.append(
                f"non-standard subject tag(s) detected: {', '.join(non_standard_tags[:5])}. "
                f"Consider using <Subject N> (model-native syntax) for better stability."
            )

    def encode_clip(self):
        if self.is_official_fl2v_mode:
            tokens = self.clip.tokenize(self.conditioned_prompt, images=self.keyframe_images)
            self.conditioning = self.clip.encode_from_tokens_scheduled(tokens)
            if self.keyframes_out:
                self.conditioning = node_helpers.conditioning_set_values(
                    self.conditioning, {"minimax_keyframes": self.keyframes_out}
                )
            if self.text_boost_strength != 1.0:
                self.conditioning = amplify_text_conditioning(
                    self.conditioning,
                    strength=self.text_boost_strength,
                    mode=self.text_boost_mode,
                    renorm=self.text_boost_renorm,
                )
                logger.info(
                    "[YimoH3] FL2V text conditioning boosted: strength=%.2f, mode=%s",
                    self.text_boost_strength, self.text_boost_mode
                )
            return

        if self.keyframes_out and self.real_ref_blocks:
            ref_items = self.real_ref_items + [{"type": "image", "data": image} for image in self.keyframe_images]
            tokens = self.clip.tokenize(self.conditioned_prompt, minimax_ref_items=ref_items)
        elif self.real_ref_blocks:
            tokens = self.clip.tokenize(self.conditioned_prompt, minimax_ref_items=self.real_ref_items)
        else:
            tokens = self.clip.tokenize(self.conditioned_prompt, images=self.keyframe_images)

        self.conditioning = self.clip.encode_from_tokens_scheduled(tokens)

        if self.text_boost_strength != 1.0:
            self.conditioning = amplify_text_conditioning(
                self.conditioning,
                strength=self.text_boost_strength,
                mode=self.text_boost_mode,
                renorm=self.text_boost_renorm,
            )
            logger.info(
                "[YimoH3] Text conditioning boosted: strength=%.2f, mode=%s, renorm=%s",
                self.text_boost_strength, self.text_boost_mode, self.text_boost_renorm
            )

        values = {}
        if self.keyframes_out:
            values.update({"minimax_keyframes": self.keyframes_out, "minimax_frame_count": self.frame_count})
        if self.real_ref_blocks:
            values["minimax_refs"] = self.real_ref_blocks

        retention_value = get_retention_value(self.reference_retention)
        effective_strength = min(retention_value, self.reference_strength) if self.reference_strength < 1.0 else retention_value
        values["minimax_visual_cond_noise_aug"] = float(effective_strength)

        values["_yimo_ref_curve"] = {
            "direction": self.ref_curve_direction,
            "shape": self.ref_curve_shape,
            "base_strength": float(effective_strength),
            "frame_count": self.frame_count,
            "retention": self.reference_retention,
        }

        if values:
            self.conditioning = node_helpers.conditioning_set_values(self.conditioning, values)

    def build_negative(self):
        if self.negative_prompt.strip():
            neg_tokens = self.clip.tokenize(self.negative_prompt)
            self.negative_conditioning = self.clip.encode_from_tokens_scheduled(neg_tokens)
        else:
            neg_tokens = self.clip.tokenize("")
            self.negative_conditioning = self.clip.encode_from_tokens_scheduled(neg_tokens)

    def apply_audio_policy(self):
        if self.resolved_audio_mode == "lock_source" and self.encoded_source is not None:
            self.latent = replace_audio_latent(self.latent, self.encoded_source, 0.0)
        elif self.resolved_audio_mode == "remix_source" and self.encoded_source is not None:
            self.latent = replace_audio_latent(self.latent, self.encoded_source, self.resolved_denoise)

    def resolve_output_audio(self):
        self.output_audio = self.override_audio if is_connected_value(self.override_audio) else self.source_audio
        if self.output_audio is not None:
            try:
                _ = validate_audio(self.output_audio, "output_audio")
            except Exception as e:
                self.compat_warnings.append(t("err_override_audio_validation", error=e))
                self.output_audio = None
        if self.output_audio is None:
            self.output_audio = _make_silent_audio(self.length)
            self.report_lines.append("note: no source_audio input; mux_audio is silent placeholder")

    def _build_port_mapping(self) -> list[dict]:
        mapping: list[dict] = []

        valid_ref_ordinals = {o for o, _ in self.ref_image_list}
        for ordinal in range(1, 10):
            if ordinal in valid_ref_ordinals:
                effective_index = next(
                    idx + 1 for idx, (o, _) in enumerate(self.ref_image_list) if o == ordinal
                )
                role = "identity" if ordinal in self.identity_ordinals else "style"
                mapping.append({
                    "port": f"ref_image_{ordinal}",
                    "tag": f"<Picture {effective_index}>",
                    "status": f"valid ({role})",
                })

        valid_video_ordinals = {o for o, _ in self.ref_video_entries}
        for ordinal in range(1, 4):
            if ordinal in valid_video_ordinals:
                effective_index = next(
                    idx + 1 for idx, (o, _) in enumerate(self.ref_video_entries) if o == ordinal
                )
                mapping.append({
                    "port": f"ref_video_{ordinal}",
                    "tag": f"<Video {effective_index}>",
                    "status": "valid",
                })

        style_audio_ordinals = [o for o, _ in sorted_autogrow_items(self.style_audios)]
        for ordinal in style_audio_ordinals:
            effective_index = style_audio_ordinals.index(ordinal) + 1
            mapping.append({
                "port": f"style_audio_{ordinal}",
                "tag": f"<Audio {effective_index}>",
                "status": "valid",
            })

        return mapping

    def assemble_report(self):
        lines = [
            "=== Yimo H3 Conditioning v3.0.0 ===",
            f"task={self.resolved_task}",
            f"model_hint={self.model_hint}",
            f"audio_mode={self.resolved_audio_mode}",
            f"audio_policy={self.audio_policy}",
            f"workflow_mode={self.workflow_mode}",
            f"raw_refs_received: images={len(self.ref_image_values)}, videos={len(self.ref_video_values)}, style_audios={len(self.style_audio_values)}",
            f"keyframes_received: images={len(self._raw_keyframes)}, positions={self.keyframe_positions if self.keyframe_positions else 'auto'}",
            f"frames={self.frame_count} ({self.frame_count / FPS:.3f}s at 24fps)",
            f"render_frames={self.render_frames} ({self.render_frames / FPS:.3f}s at 24fps)",
            f"canvas={self.width}x{self.height}",
            f"ref_pictures={len(self.picture_labels)}, keyframes={len(self.keyframe_labels)}, videos={len(self.video_labels)}, ref_audios={len(self.audio_labels)}",
            f"source_audio_tag={'<Audio 1>' if self.source_audio_ordinal else 'none'}",
            f"reference_strength={self.reference_strength:.3f}",
            f"reference_retention={self.reference_retention} (value={get_retention_value(self.reference_retention):.2f})",
            f"ref_curve_direction={self.ref_curve_direction}",
            f"ref_curve_shape={self.ref_curve_shape}",
            f"text_boost_strength={self.text_boost_strength:.2f}, mode={self.text_boost_mode}, renorm={self.text_boost_renorm}",
            f"audio_offset={self.audio_offset_frames}frames",
            f"strict_prompt_tags={self.strict_prompt_tags}",
            f"clip_video_grayout={'on' if self.clip_video_grayout else 'off'}",
            f"ref_video_preprocessing={self.ref_video_preprocessing}",
            f"prev_segment_bundle={'connected' if self.prev_segment_bundle is not None else 'none'}",
        ]

        if self.keyframes_out:
            kf_summary = " | ".join(self.keyframe_labels)
            lines.append(f"keyframe_timeline={kf_summary}")
            if self._raw_keyframes:
                lines.append("image_hierarchy=keyframes (first/last/intermediate, not in <Picture> tags) > identity_refs > style_refs")
            else:
                lines.append("image_hierarchy=keyframes (first/last, not in <Picture> tags) > identity_refs > style_refs")
        else:
            lines.append("image_hierarchy=identity_refs > style_refs (no keyframes)")

        if self.identity_ordinals:
            lines.append(f"identity_refs={sorted(self.identity_ordinals)} (by port ordinal)")
            style_ordinals = sorted({o for o, _ in self.ref_image_list} - self.identity_ordinals)
            lines.append(f"style_refs={style_ordinals} (by port ordinal)")
        else:
            style_ordinals = sorted({o for o, _ in self.ref_image_list})
            lines.append("identity_refs=none (all images treated as style refs)")
            lines.append(f"style_refs={style_ordinals} (by port ordinal)")

        if self.media_map and self.media_map != "{}":
            lines.append("")
            lines.append("--- 媒体映射 JSON (原 media_map_json 输出) ---")
            lines.append(self.media_map)

        if self.auto_resolution:
            lines.append("auto_resolution=applied")
        if self.ref_video_start_frame > 0:
            lines.append(f"ref_video_start_frame={self.ref_video_start_frame}")

        lines.extend(f"info: {s}" for s in self.subject_tag_infos)
        lines.extend(f"warning: {w}" for w in self.prompt_warnings)
        lines.extend(f"compat: {w}" for w in self.compat_warnings)
        self.report_lines = lines

    def build_media_map(self):
        self.port_mapping = self._build_port_mapping()
        self.media_map = media_map_json(
            self.picture_labels,
            self.video_labels,
            self.audio_labels,
            self.source_audio_ordinal,
            keyframe_labels=self.keyframe_labels,
            port_mapping=self.port_mapping,
        )

    def build_segment_bundle(self):
        if self.conditioning is None:
            logger.warning("build_segment_bundle: conditioning is None, returning None")
            return None
        if not isinstance(self.conditioning, list) or len(self.conditioning) == 0:
            logger.warning("build_segment_bundle: conditioning is not a non-empty list, returning None")
            return None
        first_item = self.conditioning[0]
        if not isinstance(first_item, (list, tuple)) or len(first_item) < 2:
            logger.warning("build_segment_bundle: conditioning[0] does not contain [tensor, dict], returning None")
            return None
        if not isinstance(first_item[1], dict):
            logger.warning("build_segment_bundle: conditioning[0][1] is not a dict, returning None")
            return None

        cond_tensor = first_item[0]
        cond_dict = first_item[1].copy()
        cond_dict["_yimo_segment"] = True
        _yimo_data = {
            "positive": self.conditioning,
            "negative": self.negative_conditioning,
            "latent": self.latent,
            "audio": self.output_audio,
            "frame_count": self.frame_count,
            "render_frames": self.render_frames,
            "prompt": self.conditioned_prompt,
            "media_map": self.media_map,
            "mode": self.workflow_mode,
            "keyframe_specs": self.keyframe_specs,
            "ref_blocks": self.real_ref_blocks,
            "identity_refs": self.identity_ref_blocks,
            "style_refs": self.style_ref_blocks,
            "identity_ordinals": sorted(self.identity_ordinals),
            "audio_policy": self.audio_policy,
            "width": self.width,
            "height": self.height,
            "port_mapping": self.port_mapping,
            "ref_curve": {
                "direction": self.ref_curve_direction,
                "shape": self.ref_curve_shape,
                "base_strength": float(min(get_retention_value(self.reference_retention), self.reference_strength) if self.reference_strength < 1.0 else get_retention_value(self.reference_retention)),
                "retention": self.reference_retention,
            },
            "text_boost": {
                "strength": self.text_boost_strength,
                "mode": self.text_boost_mode,
                "renorm": self.text_boost_renorm,
            },
            # v2.8.1: 供下游段继承素材使用
            "real_ref_items": self.real_ref_items,
            "picture_labels": self.picture_labels,
            "video_labels": self.video_labels,
            "audio_labels": self.audio_labels,
        }
        if self.is_official_fl2v_mode:
            _yimo_data["is_official_fl2v_mode"] = True
        cond_dict["_yimo_data"] = _yimo_data
        return [[cond_tensor, cond_dict]]

    def to_result(self) -> ConditioningResult:
        return ConditioningResult(
            positive=self.conditioning,
            negative=self.negative_conditioning,
            latent=self.latent,
            audio=self.output_audio,
            prompt=self.conditioned_prompt,
            report="\n".join(self.report_lines),
            segment_bundle=self.build_segment_bundle(),
            warnings=self.compat_warnings + self.prompt_warnings,
        )

    @staticmethod
    def error_result(clip, prompt, negative_prompt, width, height, length, error_report, warnings=None):
        neg_tokens = clip.tokenize(negative_prompt or "")
        negative_conditioning = clip.encode_from_tokens_scheduled(neg_tokens)
        tokens = clip.tokenize(prompt)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        aligned_w = max(CANVAS_MULTIPLE, round(width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        aligned_h = max(CANVAS_MULTIPLE, round(height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        latent, _ = empty_av_latent(aligned_w, aligned_h, length)
        silent = _make_silent_audio(length)
        return ConditioningResult(
            positive=conditioning,
            negative=negative_conditioning,
            latent=latent,
            audio=silent,
            prompt=prompt,
            report=error_report,
            segment_bundle=None,
            warnings=warnings or [],
        )


def build_conditioning(
    workflow_mode="text",
    audio_policy="generate_new",
    clip=None,
    video_vae=None,
    audio_vae=None,
    prompt="",
    negative_prompt="",
    width=1344,
    height=768,
    auto_resolution=False,
    length=124,
    first_frame=None,
    last_frame=None,
    ref_images=None,
    ref_videos=None,
    style_audios=None,
    ref_video_audios=None,
    source_audio=None,
    override_audio=None,
    ref_image_size="match",
    reference_video_policy="official_2_to_15s",
    ref_video_start_frame=0,
    reference_strength=1.0,
    identity_image_indices="",
    strict_prompt_tags=False,
    audio_offset_frames=0, audio_denoise_strength=0.35,
    clip_video_grayout=False, ref_video_preprocessing="none",
    keyframes=None,
    keyframe_positions="",
    reference_retention="fully_preserved",
    ref_curve_direction="constant",
    ref_curve_shape="linear",
    text_boost_strength=1.0,
    text_boost_mode="deviation",
    text_boost_renorm=True,
    prev_segment_bundle=None,
):
    """v2.8.1: 新增 prev_segment_bundle，可从上一段继承素材。"""
    ctx = _BuildContext(
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
        strict_prompt_tags=strict_prompt_tags,
        audio_offset_frames=audio_offset_frames,
        audio_denoise_strength=audio_denoise_strength,
        clip_video_grayout=clip_video_grayout,
        ref_video_preprocessing=ref_video_preprocessing,
        reference_retention=reference_retention,
        ref_curve_direction=ref_curve_direction,
        ref_curve_shape=ref_curve_shape,
        text_boost_strength=text_boost_strength,
        text_boost_mode=text_boost_mode,
        text_boost_renorm=text_boost_renorm,
        prev_segment_bundle=prev_segment_bundle,
    )

    ctx.filter_inputs()
    ctx.filter_inputs_by_mode()
    ctx.resolve_canvas()
    ctx.prepare_latent()
    ctx.collect_references()
    ctx.resolve_modes()
    ctx.resolve_identity_ordinals()
    ctx.build_keyframes()
    ctx.encode_ref_images()
    ctx.encode_ref_videos()
    ctx.encode_source_audio()
    ctx.encode_style_audios()
    ctx.inherit_from_prev_segment()   # v2.8.1: 在 build_prompt / encode_clip 前继承
    ctx.build_prompt()
    ctx.encode_clip()
    ctx.build_negative()
    ctx.apply_audio_policy()
    ctx.resolve_output_audio()
    ctx.build_media_map()
    ctx.assemble_report()

    return ctx.to_result().to_tuple()