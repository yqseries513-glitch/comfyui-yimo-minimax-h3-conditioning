from __future__ import annotations

import logging
import os
import shutil
import tempfile

import torch

import comfy.model_management
from comfy_api.latest import io

from .sampler_core import _concatenate_av_latents, _resolve_pair_overlaps, _overlap_to_latent
from .core import nested_av_parts, sorted_autogrow_items, get_yimo_output_dir
from .i18n import t

logger = logging.getLogger("YimoH3")
CATEGORY = t("category")

# v2.7.1: 拼接解码 chunk 的内存阈值。
_CHUNK_MEMORY_THRESHOLD_BYTES = 4 * 1024 ** 3

# v2.8.2: output_mode 合法值
_VALID_OUTPUT_MODES = ("both", "av_only", "images_only")


def _soft_empty_cache():
    try:
        comfy.model_management.soft_empty_cache()
    except Exception:
        pass


# =============================================================================
# 分块 / overlap 辅助
# =============================================================================

def _chunk_indices(video_overlaps: list[int]) -> list[tuple[int, int]]:
    """按 overlap=0 分块。返回 [(start_idx, end_idx), ...]。"""
    chunks: list[tuple[int, int]] = []
    start = 0
    for i, ov in enumerate(video_overlaps):
        if ov == 0:
            chunks.append((start, i + 1))
            start = i + 1
    chunks.append((start, len(video_overlaps) + 1))
    return chunks


def _resolve_shared_source(overlap_frames, overlap_list, n_pairs):
    """解析共享 overlap 来源。

    v2.8.1 修复：若 overlap_list 解析后所有值均为负数（如用户误填 -1），
    视为"未设置"，回退到 overlap_frames。
    混用情况下（如 "22,-1,30"），负值单点回退到 overlap_frames，正值保留。
    0 仍然是有效的硬切标记，不会被回退。
    """
    if overlap_list and str(overlap_list).strip():
        try:
            parts = [p.strip() for p in str(overlap_list).split(",") if p.strip()]
            parsed = [int(p) for p in parts]
            # 全负 → 视为"未设置"
            if all(p < 0 for p in parsed):
                return int(overlap_frames)
            # 混用 → 负值单点回退
            default_olap = int(overlap_frames)
            return [default_olap if p < 0 else p for p in parsed]
        except ValueError as e:
            raise ValueError(f"overlap_list 解析失败 ({e})，应为逗号分隔的整数")
    return int(overlap_frames)


def _overlap_effective(overlap_frames: int) -> int:
    if overlap_frames <= 0:
        return 0
    eff, _, _ = _overlap_to_latent(overlap_frames)
    return eff


# =============================================================================
# 节点
# =============================================================================

class YimoH3ConcatenateSegments(io.ComfyNode):
    """将多段 SingleSampler 输出的 AV Latent 拼接成长视频。

    v2.8.2:
    - 新增 output_mode 三档开关：both / av_only / images_only。
      av_only 跳过 VAE 解码；images_only 不累积完整 AV Latent。
    - 修复 AV Latent 输出不完整的 bug：多 chunk / 多段时正确拼接为完整长视频。
      旧版只返回第一个 chunk 的 latent（bug），现改为按 overlap=0 全量拼接。
    - 报告里明确标注各输出的构建状态。

    v2.8.1:
    - 修复 overlap_list 填负数时被 max(0, x) 静默变成 0 的 bug。
    - 负数视为"未设置"，回退到 overlap_frames。

    v2.8.0:
    - 输出 AV Latent + IMAGE 序列 + AUDIO
    - 双模式：chunked / independent
    - audio_vae 可选，连接后输出音频波形

    v2.7.1:
    - 解码 chunk 内存优先 + 超阈值落盘，避免长视频 OOM。
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YimoH3ConcatenateSegments",
            display_name="Yimo H3 段拼接器",
            description=(
                "将多段采样输出拼接成长视频，输出 AV Latent + IMAGE 序列 + AUDIO。\n"
                "chunked=按 overlap=0 分块，块内潜空间拼接（一镜到底零损失），块间像素层拼接；\n"
                "independent=每段独立解码，像素层拼接（最保守）。\n"
                "output_mode 控制 AV Latent 与 IMAGE 序列的构建：\n"
                "  both=同时构建（默认，旧行为）\n"
                "  av_only=只构建完整 AV Latent，跳过 VAE 解码（更快、更省显存）\n"
                "  images_only=只构建 IMAGE 序列，不累积完整 AV Latent\n"
                "未选中的输出端口返回 None。"
            ),
            category=CATEGORY,
            inputs=[
                io.Combo.Input(
                    "concat_mode",
                    options=["chunked", "independent"],
                    default="chunked",
                    tooltip=(
                        "chunked：按 overlap=0 自动分块，块内潜空间拼接（一镜到底部分零代际损失），"
                        "块间像素层拼接（硬切不闪）。\n"
                        "independent：每段独立 VAE 解码，像素层拼接。最保守，"
                        "适合不信任潜空间拼接的场景。"
                    ),
                ),
                io.Combo.Input(
                    "output_mode",
                    options=list(_VALID_OUTPUT_MODES),
                    default="both",
                    tooltip=(
                        "输出模式：\n"
                        "  both=同时构建 AV Latent 和 IMAGE 序列（旧行为）\n"
                        "  av_only=只构建完整 AV Latent，跳过 VAE 解码（更快、更省显存）\n"
                        "  images_only=只构建 IMAGE 序列，不累积完整 AV Latent\n"
                        "未选中的输出端口返回 None。"
                    ),
                ),
                io.Autogrow.Input(
                    "parts", optional=False,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Latent.Input(
                            "part", optional=True,
                            tooltip="单段采样器输出的 AV Latent。",
                        ),
                        prefix="part_", min=1, max=8,
                    ),
                    tooltip="1~8 段 AV Latent。通常连接多个 SingleSampler 的输出。",
                ),
                io.Int.Input(
                    "overlap_frames", default=22, min=0, max=200, step=1,
                    tooltip="共用 overlap 帧数。会按 17n+5 网格最近邻映射。\n"
                            "仅在 overlap_list 留空或全部为负数时生效。",
                ),
                io.String.Input(
                    "overlap_list", default="", advanced=True,
                    tooltip="逗号分隔的逐对 overlap 帧数，如：22,0,22。\n"
                            "0 表示硬切（会触发分块）。\n"
                            "留空 或 全部填负数（如 -1）时回退到 overlap_frames。\n"
                            "混用（如 22,-1,30）时负值单点回退，正值保留。\n"
                            "优先级高于 overlap_frames。",
                ),
                io.Int.Input(
                    "video_overlap_frames", default=-1, min=-1, max=200, step=1,
                    advanced=True,
                    tooltip="视频单独使用的 overlap 帧数。-1 表示跟随 overlap_frames / overlap_list。",
                ),
                io.Int.Input(
                    "audio_overlap_frames", default=-1, min=-1, max=200, step=1,
                    advanced=True,
                    tooltip="音频单独使用的 overlap 帧数。-1 表示跟随 overlap_frames / overlap_list。",
                ),
                io.Vae.Input(
                    "video_vae",
                    tooltip="H3 video VAE。output_mode 含 images 时必需（用于把 video latent 解码为 IMAGE 序列）。\n"
                            "output_mode=av_only 时不会实际使用。",
                ),
                io.Vae.Input(
                    "audio_vae", optional=True,
                    tooltip="H3 audio VAE，可选。连接后输出音频波形；不连则 audio 输出为 None。",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="拼接后 AV Latent"),
                io.Image.Output(display_name="拼接后图像序列"),
                io.Audio.Output(display_name="拼接后音频"),
                io.String.Output(display_name="拼接报告"),
            ],
        )

    @classmethod
    def execute(cls, concat_mode, parts, video_vae, audio_vae=None,
                overlap_frames=22, overlap_list="",
                video_overlap_frames=-1, audio_overlap_frames=-1,
                output_mode="both", **kwargs):

        part_items = sorted_autogrow_items(parts)
        if not part_items:
            raise ValueError("未提供任何输入")

        data_list = [data for _, data in part_items]
        n = len(data_list)
        n_pairs = max(0, n - 1)

        for i, item in enumerate(data_list):
            if not isinstance(item, dict) or "samples" not in item:
                raise ValueError(
                    f"part_{i + 1} 必须是 AV Latent dict，"
                    f"got {type(item).__name__}"
                )

        # v2.8.2: output_mode 解析
        if output_mode not in _VALID_OUTPUT_MODES:
            raise ValueError(
                f"未知的 output_mode: {output_mode}，"
                f"合法值: {', '.join(_VALID_OUTPUT_MODES)}"
            )
        need_av = output_mode in ("both", "av_only")
        need_images = output_mode in ("both", "images_only")

        if need_images and video_vae is None:
            raise ValueError(
                "output_mode 含 images 输出（both / images_only）时，video_vae 是必需的"
            )
        if not need_images and video_vae is None:
            # av_only 模式下 video_vae 不会被实际使用
            logger.info(
                "[YimoH3ConcatenateSegments] output_mode=%s: video_vae not connected, "
                "IMAGE decode skipped.", output_mode,
            )

        shared_source = _resolve_shared_source(overlap_frames, overlap_list, n_pairs)
        video_source = int(video_overlap_frames) if video_overlap_frames >= 0 else shared_source
        audio_source = int(audio_overlap_frames) if audio_overlap_frames >= 0 else shared_source

        video_overlaps = _resolve_pair_overlaps(video_source, n_pairs)
        audio_overlaps = _resolve_pair_overlaps(audio_source, n_pairs)

        if concat_mode == "chunked":
            chunks = _chunk_indices(list(video_overlaps))
            final_images, chunk_info, final_av = cls._run_chunked(
                data_list, video_overlaps, audio_overlaps, video_vae, chunks,
                need_av=need_av, need_images=need_images,
            )
        elif concat_mode == "independent":
            final_images, chunk_info, final_av = cls._run_independent(
                data_list, video_overlaps, audio_overlaps, video_vae,
                need_av=need_av, need_images=need_images,
            )
        else:
            raise ValueError(f"未知的 concat_mode: {concat_mode}")

        final_audio, audio_note = cls._run_audio(
            data_list, video_overlaps, audio_overlaps, audio_vae,
        )

        report_lines = [
            "=== Yimo H3 ConcatenateSegments v3.0.0 ===",
            f"concat_mode={concat_mode}",
            f"output_mode={output_mode} "
            f"(av={'built' if need_av else 'skipped'}, "
            f"images={'built' if need_images else 'skipped'})",
            f"input_parts={n}",
            f"overlap_frames={overlap_frames}",
            f"overlap_list={overlap_list!r}",
            f"resolved_shared_source={shared_source if isinstance(shared_source, int) else list(shared_source)}",
            f"video_overlaps(requested)={list(video_overlaps)}",
            f"video_overlaps(effective)={[_overlap_effective(o) for o in video_overlaps]}",
            f"audio_overlaps(requested)={list(audio_overlaps)}",
            f"audio_overlaps(effective)={[_overlap_effective(o) for o in audio_overlaps]}",
            f"chunk_memory_threshold_mb={_CHUNK_MEMORY_THRESHOLD_BYTES / 1024**2:.0f}",
            "--- 视频 ---",
            *chunk_info,
        ]

        if final_images is not None:
            report_lines.append(f"final_images_frames={final_images.shape[0]}")
            report_lines.append(f"final_images_hw=({final_images.shape[1]}, {final_images.shape[2]})")
        else:
            report_lines.append("final_images=skipped (output_mode)")

        if final_av is not None:
            # 打印完整 AV latent 的形状，便于用户确认
            try:
                v_shape, a_shape = None, None
                if isinstance(final_av, dict):
                    samples = final_av.get("samples")
                    if getattr(samples, "is_nested", False):
                        parts = tuple(samples.unbind())
                        if len(parts) == 2:
                            v_shape = tuple(parts[0].shape)
                            a_shape = tuple(parts[1].shape)
                if v_shape is not None:
                    report_lines.append(f"final_av_video_shape={list(v_shape)}")
                    report_lines.append(f"final_av_audio_shape={list(a_shape)}")
                else:
                    report_lines.append("final_av_latent=present")
            except Exception:
                report_lines.append("final_av_latent=present")
        else:
            report_lines.append("final_av_latent=skipped (output_mode)")

        report_lines.append("--- 音频 ---")
        report_lines.append(audio_note)
        report_lines.append("拼接完成")
        report = "\n".join(report_lines)

        logger.info(
            "[YimoH3ConcatenateSegments] mode=%s output_mode=%s parts=%d frames=%s audio=%s av=%s",
            concat_mode, output_mode, n,
            final_images.shape[0] if final_images is not None else "skipped",
            "yes" if final_audio is not None else "no",
            "yes" if final_av is not None else "no",
        )

        return io.NodeOutput(final_av, final_images, final_audio, report)

    # -------------------------------------------------------------------------
    # chunked 模式
    # -------------------------------------------------------------------------

    @classmethod
    def _run_chunked(cls, data_list, video_overlaps, audio_overlaps, video_vae, chunks,
                     need_av=True, need_images=True):
        """按 overlap=0 分块，块内潜空间拼接，块间像素层拼接。

        v2.8.2:
        - need_av / need_images 控制输出构建，避免不必要的解码或 AV 累积。
        - 修复 AV Latent 输出不完整的 bug：多 chunk 时按 overlap=0 全量拼接。
        """
        chunk_info: list[str] = []
        chunk_latents = [] if need_av else None
        chunk_storage: list[tuple[str, object]] = []
        in_memory_bytes = 0
        spilled = False
        tmp_dir = None
        tmp_root = os.path.join(get_yimo_output_dir(), "_concat_chunk_cache")
        os.makedirs(tmp_root, exist_ok=True)

        def _ensure_tmp_dir():
            nonlocal tmp_dir
            if tmp_dir is None:
                tmp_dir = tempfile.mkdtemp(prefix="yimo_concat_", dir=tmp_root)
            return tmp_dir

        try:
            for ci, (cs, ce) in enumerate(chunks):
                chunk_segs = data_list[cs:ce]
                if len(chunk_segs) == 1:
                    chunk_latent = chunk_segs[0]
                else:
                    chunk_v_olap = list(video_overlaps[cs:ce - 1])
                    chunk_a_olap = list(audio_overlaps[cs:ce - 1])
                    chunk_latent = _concatenate_av_latents(
                        chunk_segs, chunk_v_olap, chunk_a_olap,
                    )

                if need_av:
                    chunk_latents.append(chunk_latent)

                if need_images:
                    cv, _ = nested_av_parts(chunk_latent)
                    decoded = cls._decode_video_latent(video_vae, cv)
                    decoded_cpu = decoded.detach().to("cpu").contiguous()
                    chunk_bytes = decoded_cpu.numel() * decoded_cpu.element_size()

                    if in_memory_bytes + chunk_bytes <= _CHUNK_MEMORY_THRESHOLD_BYTES:
                        chunk_storage.append(("mem", decoded_cpu))
                        in_memory_bytes += chunk_bytes
                        storage_note = f"kept_in_memory ({chunk_bytes / 1024**2:.1f} MB)"
                    else:
                        _ensure_tmp_dir()
                        path = os.path.join(tmp_dir, f"chunk_{ci:03d}.pt")
                        torch.save(decoded_cpu, path)
                        chunk_storage.append(("disk", path))
                        spilled = True
                        storage_note = f"spilled_to_disk ({chunk_bytes / 1024**2:.1f} MB)"

                    chunk_info.append(
                        f"chunk_{ci}: segs=[{cs},{ce}) latent_t={cv.shape[2]} "
                        f"frames={decoded_cpu.shape[0]} {storage_note}"
                    )

                    del decoded, decoded_cpu, cv
                else:
                    # 只打印形状信息，不解码
                    cv, _ = nested_av_parts(chunk_latent)
                    chunk_info.append(
                        f"chunk_{ci}: segs=[{cs},{ce}) latent_t={cv.shape[2]} "
                        f"(image decode skipped)"
                    )
                    del cv

                del chunk_latent
                _soft_empty_cache()

            # ---- final_images ----
            if need_images:
                pixel_parts = []
                for kind, payload in chunk_storage:
                    if kind == "mem":
                        pixel_parts.append(payload)
                    else:
                        pixel_parts.append(
                            torch.load(payload, map_location="cpu", weights_only=True)
                        )
                final_images = torch.cat(pixel_parts, dim=0)
                chunk_info.append(
                    f"chunk_storage: in_memory={sum(1 for k, _ in chunk_storage if k == 'mem')}, "
                    f"disk={sum(1 for k, _ in chunk_storage if k == 'disk')}"
                )
            else:
                final_images = None
                chunk_info.append("chunk_storage: disabled (image decode skipped)")

            # ---- final_av ----
            if need_av:
                if not chunk_latents:
                    final_av = None
                elif len(chunk_latents) == 1:
                    final_av = chunk_latents[0]
                else:
                    # 块间是硬切（overlap=0），直接全量拼接
                    final_av = _concatenate_av_latents(
                        chunk_latents,
                        [0] * (len(chunk_latents) - 1),
                        [0] * (len(chunk_latents) - 1),
                    )
                    chunk_info.append(
                        f"final_av: concatenated {len(chunk_latents)} chunks (overlap=0)"
                    )
            else:
                final_av = None

            return final_images, chunk_info, final_av

        finally:
            if spilled and tmp_dir is not None:
                try:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception:
                    pass

    # -------------------------------------------------------------------------
    # independent 模式
    # -------------------------------------------------------------------------

    @classmethod
    def _run_independent(cls, data_list, video_overlaps, audio_overlaps, video_vae,
                         need_av=True, need_images=True):
        """每段独立解码，像素层按 overlap 裁帧后拼接。

        v2.8.2:
        - need_av / need_images 控制输出构建。
        - AV Latent 输出改为按 video_overlaps / audio_overlaps 全量拼接，
          与像素层拼接逻辑保持一致。
        """
        chunk_info: list[str] = []
        chunk_storage: list[tuple[str, object]] = []
        in_memory_bytes = 0
        spilled = False
        tmp_dir = None
        tmp_root = os.path.join(get_yimo_output_dir(), "_concat_chunk_cache")
        os.makedirs(tmp_root, exist_ok=True)

        def _ensure_tmp_dir():
            nonlocal tmp_dir
            if tmp_dir is None:
                tmp_dir = tempfile.mkdtemp(prefix="yimo_concat_", dir=tmp_root)
            return tmp_dir

        try:
            for i, seg in enumerate(data_list):
                v, _ = nested_av_parts(seg)

                if need_images:
                    decoded = cls._decode_video_latent(video_vae, v)

                    if i > 0:
                        eff = _overlap_effective(video_overlaps[i - 1])
                        if eff > 0 and eff < decoded.shape[0]:
                            decoded = decoded[eff:]
                            trimmed = eff
                        else:
                            trimmed = 0
                    else:
                        trimmed = 0

                    decoded_cpu = decoded.detach().to("cpu").contiguous()
                    chunk_bytes = decoded_cpu.numel() * decoded_cpu.element_size()

                    if in_memory_bytes + chunk_bytes <= _CHUNK_MEMORY_THRESHOLD_BYTES:
                        chunk_storage.append(("mem", decoded_cpu))
                        in_memory_bytes += chunk_bytes
                        storage_note = f"kept_in_memory ({chunk_bytes / 1024**2:.1f} MB)"
                    else:
                        _ensure_tmp_dir()
                        path = os.path.join(tmp_dir, f"seg_{i:03d}.pt")
                        torch.save(decoded_cpu, path)
                        chunk_storage.append(("disk", path))
                        spilled = True
                        storage_note = f"spilled_to_disk ({chunk_bytes / 1024**2:.1f} MB)"

                    chunk_info.append(
                        f"seg_{i}: latent_t={v.shape[2]} frames={decoded_cpu.shape[0]} "
                        f"trimmed={trimmed} {storage_note}"
                    )

                    del decoded, decoded_cpu
                else:
                    chunk_info.append(
                        f"seg_{i}: latent_t={v.shape[2]} (image decode skipped)"
                    )

                del v
                _soft_empty_cache()

            # ---- final_images ----
            if need_images:
                pixel_parts = []
                for kind, payload in chunk_storage:
                    if kind == "mem":
                        pixel_parts.append(payload)
                    else:
                        pixel_parts.append(
                            torch.load(payload, map_location="cpu", weights_only=True)
                        )
                final_images = torch.cat(pixel_parts, dim=0)
                chunk_info.append(
                    f"chunk_storage: in_memory={sum(1 for k, _ in chunk_storage if k == 'mem')}, "
                    f"disk={sum(1 for k, _ in chunk_storage if k == 'disk')}"
                )
            else:
                final_images = None
                chunk_info.append("chunk_storage: disabled (image decode skipped)")

            # ---- final_av ----
            if need_av:
                if not data_list:
                    final_av = None
                elif len(data_list) == 1:
                    final_av = data_list[0]
                else:
                    # 与像素层拼接一致：按 video_overlaps / audio_overlaps 裁剪后拼接
                    final_av = _concatenate_av_latents(
                        data_list,
                        list(video_overlaps),
                        list(audio_overlaps),
                    )
                    chunk_info.append(
                        f"final_av: concatenated {len(data_list)} segments "
                        f"(video_overlaps={list(video_overlaps)}, "
                        f"audio_overlaps={list(audio_overlaps)})"
                    )
            else:
                final_av = None

            return final_images, chunk_info, final_av

        finally:
            if spilled and tmp_dir is not None:
                try:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception:
                    pass

    # -------------------------------------------------------------------------
    # 音频
    # -------------------------------------------------------------------------

    @classmethod
    def _run_audio(cls, data_list, video_overlaps, audio_overlaps, audio_vae):
        try:
            final_audio_latent_dict = _concatenate_av_latents(
                data_list, video_overlaps, audio_overlaps,
            )
            _, final_audio_latent = nested_av_parts(final_audio_latent_dict)
        except Exception as e:
            logger.warning("音频 latent 拼接失败: %s", e)
            return None, f"音频 latent 拼接失败: {e}"

        if audio_vae is None:
            return None, "audio_vae 未连接，audio 输出为 None（音频 latent 已拼好但未解码）"

        try:
            decoded = audio_vae.decode(final_audio_latent)
        except Exception as e:
            logger.warning("audio_vae.decode failed: %s", e)
            return None, f"audio_vae.decode 失败: {e}"

        if not isinstance(decoded, torch.Tensor):
            return None, f"audio_vae.decode 返回非 Tensor: {type(decoded).__name__}"

        waveform = cls._ensure_bct(decoded)
        sample_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))

        audio_out = {
            "waveform": waveform,
            "sample_rate": sample_rate,
        }
        note = (
            f"audio decoded: waveform_shape={tuple(waveform.shape)}, sr={sample_rate} "
            f"(audio_latent_shape={tuple(final_audio_latent.shape)})"
        )
        return audio_out, note

    @staticmethod
    def _ensure_bct(waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3:
            raise ValueError(f"Expected 3D waveform, got {waveform.ndim}D")
        if waveform.shape[1] == 2:
            return waveform
        if waveform.shape[-1] == 2:
            return waveform.movedim(-1, 1)
        return waveform

    @staticmethod
    def _decode_video_latent(video_vae, video_latent: torch.Tensor) -> torch.Tensor:
        decoded = video_vae.decode(video_latent)
        if not isinstance(decoded, torch.Tensor):
            raise ValueError(
                f"video_vae.decode 返回非 Tensor：{type(decoded).__name__}"
            )
        if decoded.ndim == 5:
            if decoded.shape[0] == 1:
                decoded = decoded.squeeze(0)
            elif decoded.shape[1] == 1:
                decoded = decoded.squeeze(1)
        if decoded.ndim != 4:
            raise ValueError(
                f"video_vae.decode 返回形状异常：{tuple(decoded.shape)}"
            )
        if decoded.shape[-1] > 3:
            decoded = decoded[..., :3]
        return decoded.clamp(0.0, 1.0)