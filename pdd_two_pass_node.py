# -*- coding: utf-8 -*-
"""Yimo H3 PDD 双采封装节点

本节点封装 MiniMax H3 PDD 4+4 双采工作流：
  LOW 阶段（前 N 步）→ 学习型 latent 放大 → HIGH 阶段（剩余步数）

参考/借鉴来源：
  - T8star-Aix/comfyui-minimax-h3-audio-T8 (GPL-3.0-or-later)
    PDD 4+4 双采工作流结构（SplitSigmas + LOW/HIGH 分阶段采样）
  - Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc (Apache-2.0)
    MiniMaxH3PDDAccApply 节点的 MODEL + SIGMAS 输出格式
  - LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler
    学习型 latent 放大节点（通过 NODE_CLASS_MAPPINGS 动态调用，不内嵌）

许可证：GPL-3.0-or-later（与 T8 工作流兼容）

注意：本节点仅适用于 PDD LoRA，不兼容 Turbo / Lightning / DMD 等其他加速 LoRA。
"""

from __future__ import annotations

import logging

import torch

import comfy.model_management
import comfy.nested_tensor
import comfy.sample

from comfy_api.latest import io

from .core import nested_av_parts
from .i18n import t
from .sampler_core import _sample_single_segment

logger = logging.getLogger("YimoH3")
CATEGORY = t("category")


# =============================================================================
# LBH 学习型 latent 放大节点动态调用
# =============================================================================

# LBH 插件常见的节点 ID 候选（按优先级）
_LBH_NODE_CANDIDATES = (
    "MinimaxH3LatentUpscaler3D",
    "MinimaxH3LatentUpscaler2D",
    "MinimaxH3LatentUpscaler",
)


def _find_lbh_upscaler_node():
    """从 ComfyUI 全局节点表中查找 LBH 学习型放大节点。"""
    try:
        import nodes
    except ImportError:
        return None, None

    node_map = getattr(nodes, "NODE_CLASS_MAPPINGS", {})
    if not node_map:
        return None, None

    for name in _LBH_NODE_CANDIDATES:
        cls = node_map.get(name)
        if cls is not None:
            return cls, name

    # 模糊匹配兜底
    for name, cls in node_map.items():
        if "MinimaxH3LatentUpscaler" in name:
            return cls, name

    return None, None


def _call_comfy_node(cls, **overrides):
    """动态调用 ComfyUI 节点类的 execute，未指定的参数用 INPUT_TYPES 默认值填充。"""
    try:
        input_types = cls.INPUT_TYPES()
    except Exception as e:
        raise RuntimeError(f"无法获取节点 {cls.__name__} 的输入定义: {e}") from e

    all_inputs: dict[str, tuple[str, object]] = {}
    for section in ("required", "optional"):
        section_dict = input_types.get(section, {})
        for name, spec in section_dict.items():
            all_inputs[name] = (section, spec)

    kwargs: dict = {}
    for name, (_section, spec) in all_inputs.items():
        if name in overrides:
            kwargs[name] = overrides[name]
            continue

        if not isinstance(spec, (list, tuple)) or len(spec) == 0:
            continue

        type_spec = spec[0]
        options = spec[1] if len(spec) > 1 else {}

        if isinstance(type_spec, list):
            if type_spec:
                kwargs[name] = type_spec[0]
            continue

        if not isinstance(options, dict):
            continue

        if "default" in options:
            kwargs[name] = options["default"]
            continue

        if type_spec == "INT":
            kwargs[name] = options.get("min", 0)
        elif type_spec == "FLOAT":
            kwargs[name] = options.get("min", 0.0)
        elif type_spec == "BOOLEAN":
            kwargs[name] = False
        elif type_spec == "STRING":
            kwargs[name] = ""

    instance = cls()
    if not hasattr(instance, "execute"):
        raise RuntimeError(f"节点 {cls.__name__} 没有 execute 方法")

    result = instance.execute(**kwargs)

    # 兼容 io.NodeOutput / tuple / 单值 三种返回形式
    if hasattr(result, "result") and result.result is not None:
        return result.result
    if isinstance(result, (list, tuple)):
        return result
    return (result,)


def _run_lbh_upscale(latent: dict, scale_by: float) -> dict:
    """调用 LBH 学习型 latent 放大节点，把 latent 放大 scale_by 倍。"""
    cls, node_id = _find_lbh_upscaler_node()
    if cls is None:
        raise RuntimeError(
            "PDD 双采需要学习型 Latent 放大节点（LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler）。"
            "请确认该插件已安装并被 ComfyUI 正确加载。"
        )

    logger.info("[PDD TwoPass] using LBH upscaler: %s", node_id)

    mode_value: object
    try:
        input_types = cls.INPUT_TYPES()
        mode_spec = None
        for section in ("required", "optional"):
            section_dict = input_types.get(section, {})
            if "mode" in section_dict:
                mode_spec = section_dict["mode"]
                break

        if isinstance(mode_spec, (list, tuple)) and len(mode_spec) > 0:
            mode_options = mode_spec[0]
            if isinstance(mode_options, list):
                # COMBO 模式：优先选带 "scale by" 的选项
                for opt in mode_options:
                    if isinstance(opt, str) and "scale by" in opt.lower():
                        mode_value = opt
                        break
                else:
                    mode_value = mode_options[0] if mode_options else "scale by multiplier"
            else:
                mode_value = "scale by multiplier"
        else:
            mode_value = "scale by multiplier"
    except Exception:
        mode_value = "scale by multiplier"

    overrides = {
        "latent": latent,
        "mode": mode_value,
    }

    # 如果 LBH 节点有 scale 参数，尝试覆盖
    try:
        input_types = cls.INPUT_TYPES()
        for section in ("required", "optional"):
            for name in input_types.get(section, {}).keys():
                lname = name.lower()
                if lname in ("scale", "scale_by", "multiplier", "upscale_by"):
                    overrides[name] = float(scale_by)
    except Exception:
        pass

    result = _call_comfy_node(cls, **overrides)

    out = result[0] if isinstance(result, (list, tuple)) and result else None
    if not isinstance(out, dict) or "samples" not in out:
        raise RuntimeError(f"LBH upscaler 返回格式异常: {type(out).__name__}")

    return out


# =============================================================================
# SIGMAS 切分
# =============================================================================

def _split_sigmas(sigmas, split_step: int):
    """把一条完整 SIGMAS 轨迹切分为噪声端（先跑）和干净端（后跑）两段。

    ComfyUI 原生 SplitSigmas(step=k) 的语义：
      噪声端 sigmas = sigmas[:k+1]   （长度 k+1，跑 k 步）→ 用于 LOW 阶段
      干净端 sigmas = sigmas[k:]     （长度 n-k，跑 n-k-1 步）→ 用于 HIGH 阶段
    两段共享边界值 sigmas[k]，保证两阶段之间无缝衔接。
    """
    if sigmas is None:
        raise ValueError("PDD 双采需要提供 sigmas（来自 PDD Apply 节点）")

    n = len(sigmas)
    if split_step <= 0 or split_step >= n - 1:
        raise ValueError(
            f"split_step={split_step} 超出有效范围。"
            f"sigmas 共 {n} 个值（{n - 1} 步），split_step 应在 1~{n - 2} 之间。"
        )

    noise_end_sigmas = sigmas[: split_step + 1]
    clean_end_sigmas = sigmas[split_step:]
    return noise_end_sigmas, clean_end_sigmas


# =============================================================================
# PDD 双采节点
# =============================================================================

class YimoH3PDDTwoPass(io.ComfyNode):
    """PDD 双采封装节点。

    LOW 阶段（前 N 步）→ 学习型 latent 放大 → HIGH 阶段（剩余步数）。
    总 Transformer 调用保持为 PDD 训练的 NFE（如 8），与 LoRA 轨迹一致。

    仅适用 PDD LoRA。不兼容 Turbo / Lightning / DMD。
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YimoH3PDDTwoPass",
            display_name="Yimo H3 PDD 双采封装",
            description="PDD 双采：LOW 阶段 → 学习型 latent 放大 → HIGH 阶段。"
                        "需要连接 PDD Apply 节点（如 MiniMaxH3PDDAccApply）"
                        "输出的 MODEL 和 SIGMAS。仅支持 PDD LoRA，"
                        "不兼容 Turbo / Lightning / DMD 等其他加速 LoRA。",
            category=CATEGORY,
            inputs=[
                # ===== 模型与 sigma =====
                io.Model.Input("model",
                    tooltip="来自 PDD Apply 节点的 MODEL。"),
                io.Custom("SIGMAS").Input("sigmas",
                    tooltip="来自 PDD Apply 节点的完整 sigma 轨迹（9 点对应 8 步）。"),
                io.Int.Input("split_step", default=4, min=1, max=7, step=1,
                    tooltip="切分索引。PDD 8 步轨迹共 9 个 sigma 值，split_step=4 表示"
                            "LOW 阶段跑前 4 步、HIGH 阶段跑后 4 步。"
                            "可调为 3 / 5 等，只要满足 1 <= split_step <= NFE-1。"),

                # ===== 可选覆盖端口（高级用法） =====
                io.Custom("SIGMAS").Input("low_stage_sigmas_override", optional=True,
                    tooltip="可选。覆盖 LOW 阶段（先跑，噪声端）的 sigmas。"
                            "若连接，则忽略内部切分，使用此 sigmas 跑 LOW 阶段。"),
                io.Custom("SIGMAS").Input("high_stage_sigmas_override", optional=True,
                    tooltip="可选。覆盖 HIGH 阶段（后跑，干净端）的 sigmas。"
                            "若连接，则忽略内部切分，使用此 sigmas 跑 HIGH 阶段。"),

                # ===== 条件与 latent =====
                io.Conditioning.Input("positive",
                    tooltip="正向条件。来自 YimoH3Conditioning 的 Positive 输出。"),
                io.Conditioning.Input("negative", optional=True,
                    tooltip="负向条件。来自 YimoH3Conditioning 的 Negative 输出。"),
                io.Latent.Input("latent",
                    tooltip="低分辨率模板 latent。LOW 阶段以此为起点。"),

                # ===== 放大参数 =====
                io.Float.Input("scale_by", default=1.5, min=1.0, max=4.0, step=0.05,
                    tooltip="学习型 latent 放大倍数。默认 1.5×。"),

                # ===== 采样参数 =====
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, step=1,
                    tooltip="随机种子。LOW 和 HIGH 阶段共用。"),
                io.Float.Input("cfg", default=1.0, min=0.0, max=2.0, step=0.1,
                    tooltip="PDD 不支持 CFG，保持 1.0。"),
            ],
            outputs=[
                io.Latent.Output(display_name="高分辨率 AV Latent"),
                io.String.Output(display_name="双采报告"),
            ],
        )

    @classmethod
    def execute(cls, model, sigmas, split_step=4,
                low_stage_sigmas_override=None, high_stage_sigmas_override=None,
                positive=None, negative=None, latent=None,
                scale_by=1.5, seed=0, cfg=1.0, **kwargs):

        report_lines = [
            "=== Yimo H3 PDD TwoPass v3.0.0 ===",
            f"split_step={split_step}",
            f"scale_by={scale_by}",
            f"cfg={cfg}, seed={seed}",
        ]

        # ---- 1. 解析 SIGMAS ----
        if low_stage_sigmas_override is not None and high_stage_sigmas_override is not None:
            noise_end_sigmas = low_stage_sigmas_override
            clean_end_sigmas = high_stage_sigmas_override
            report_lines.append("sigmas_source=manual_override")
        else:
            noise_end_sigmas, clean_end_sigmas = _split_sigmas(sigmas, split_step)
            report_lines.append(f"sigmas_source=internal_split(step={split_step})")

        n_low_steps = len(noise_end_sigmas) - 1
        n_high_steps = len(clean_end_sigmas) - 1
        report_lines.append(f"LOW_stage_steps={n_low_steps} (noise end)")
        report_lines.append(f"HIGH_stage_steps={n_high_steps} (clean end)")

        if n_low_steps <= 0 or n_high_steps <= 0:
            raise ValueError(
                f"切分后 LOW/HIGH 阶段步数必须都 >= 1，"
                f"当前 LOW={n_low_steps}, HIGH={n_high_steps}"
            )

        device = comfy.model_management.get_torch_device()

        # ---- 2. LOW 阶段：噪声端 sigmas，低分辨率 ----
        logger.info("[PDD TwoPass] LOW stage start: %d steps", n_low_steps)
        low_latent = _sample_single_segment(
            model=model,
            seed=seed,
            steps=n_low_steps,
            cfg=cfg,
            sampler_name="euler",
            scheduler="simple",
            positive=positive,
            negative=negative,
            latent=latent,
            denoise=1.0,
            callback=None,
            disable_pbar=False,
            sigmas=noise_end_sigmas.to(device) if isinstance(noise_end_sigmas, torch.Tensor) else noise_end_sigmas,
        )

        low_v, _ = nested_av_parts(low_latent)
        report_lines.append(f"LOW_output_video_shape={list(low_v.shape)}")

        # ---- 3. 学习型 latent 放大 ----
        logger.info("[PDD TwoPass] LBH latent upscale, scale=%.2f", scale_by)
        upscaled_latent = _run_lbh_upscale(low_latent, scale_by)

        up_v, up_a = nested_av_parts(upscaled_latent)
        report_lines.append(f"upscaled_video_shape={list(up_v.shape)}")
        report_lines.append(f"upscaled_audio_shape={list(up_a.shape)}")

        # ---- 4. HIGH 阶段：干净端 sigmas，放大后的 latent ----
        logger.info("[PDD TwoPass] HIGH stage start: %d steps", n_high_steps)
        final_latent = _sample_single_segment(
            model=model,
            seed=seed,
            steps=n_high_steps,
            cfg=cfg,
            sampler_name="euler",
            scheduler="simple",
            positive=positive,
            negative=negative,
            latent=upscaled_latent,
            denoise=1.0,
            callback=None,
            disable_pbar=False,
            sigmas=clean_end_sigmas.to(device) if isinstance(clean_end_sigmas, torch.Tensor) else clean_end_sigmas,
        )

        final_v, final_a = nested_av_parts(final_latent)
        report_lines.append(f"HIGH_output_video_shape={list(final_v.shape)}")
        report_lines.append(f"HIGH_output_audio_shape={list(final_a.shape)}")
        report_lines.append("two_pass_complete")

        return io.NodeOutput(final_latent, "\n".join(report_lines))