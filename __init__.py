if __package__:
    from .nodes import comfy_entrypoint
else:
    import importlib.util
    from pathlib import Path
    import sys
    import types

    _package_name = "_yimo_minimax_h3_direct"
    _package_root = Path(__file__).resolve().parent
    _package = types.ModuleType(_package_name)
    _package.__path__ = [str(_package_root)]
    sys.modules.setdefault(_package_name, _package)
    _spec = importlib.util.spec_from_file_location(f"{_package_name}.nodes", _package_root / "nodes.py")
    _nodes = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _nodes
    assert _spec.loader is not None
    _spec.loader.exec_module(_nodes)
    comfy_entrypoint = _nodes.comfy_entrypoint

__all__ = ["comfy_entrypoint"]

import os
import logging

# v2.7.0: 统一使用标准日志
logger = logging.getLogger("YimoH3")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

# v2.7.0: JS 文件与 Python 文件同级
WEB_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# =============================================================================
# v2.7.0: Monkey-patch ComfyUI official MiniMaxH3.extra_conds
# ---------------------------------------------------------------------------
# Patch 目标是 ComfyUI 官方 comfy.model_base.MiniMaxH3.extra_conds。
# 若官方实现已自带合并逻辑，本 patch 会被跳过（幂等）。感谢 ComfyUI 官方实现。
#
# Root cause: When both minimax_keyframes and minimax_refs are present,
# ComfyUI's official MiniMaxH3.extra_conds overwrites payload["cond_video_latents"]
# with refs only, dropping keyframes. PackedLayout still reserves token slots for
# keyframes (1008 spatial patches for 1344x768 canvas), causing:
#   RuntimeError: shape mismatch: value tensor [N-1008, 96] -> indexing [N, 96]
# Fix: Merge keyframes and refs latents into cond_video_latents when both exist.
#
# 守卫策略（v2.7.0）：
#   - 优先检查官方实现是否已带合并逻辑（源码/字节码）——但这只是启发式。
#   - 若启发式判定"已合并"，仍然记录日志，但允许在明确 marker 不存在时继续 patch。
#     因为本 patch 本身是幂等的：refs-first 合并 keyframes 的 latents 即使重复执行
#     也不会破坏语义。如果未来官方重构字段名，本 patch 仍能兜底。
#   - 使用 _YIMO_PATCH_MARKER 防止同一会话内重复 patch。
# =============================================================================


_YIMO_PATCH_MARKER = "_yimo_minimax_h3_extra_conds_patch_v3_0_0"


def _official_already_merges(original) -> bool:
    """启发式探测：官方 extra_conds 是否已经包含 keyframes+refs 合并逻辑。"""
    import inspect as _inspect

    # 1) 源码字符串探测
    try:
        src = _inspect.getsource(original)
        if (
            "cond_video_latents" in src
            and "keyframes" in src
            and "refs" in src
            and "merged_video_latents" in src
        ):
            return True
    except (OSError, TypeError):
        pass

    # 2) 字节码 co_names 探测
    code = getattr(original, "__code__", None)
    if code is not None:
        co_names = set(code.co_names)
        if (
            "keyframes" in co_names
            and "refs" in co_names
            and "merged_video_latents" in co_names
        ):
            return True

    return False


def _try_patch_minimax_h3():
    """尝试 patch MiniMaxH3.extra_conds，带有版本守卫和防重复加载保护。"""
    try:
        import comfy.model_base
        import comfy.conds  # noqa: F401

        target_class = comfy.model_base.MiniMaxH3
        method_name = "extra_conds"
        original = getattr(target_class, method_name, None)

        if original is None:
            logger.info("[YimoH3 patch] SKIP: MiniMaxH3.extra_conds not found.")
            return

        # 防重复 patch：优先看官方方法上是否已有本插件标记
        if getattr(original, _YIMO_PATCH_MARKER, False):
            logger.info("[YimoH3 patch] SKIP: already patched by YimoH3 in this session.")
            return

        # 启发式能力探测：官方可能已包含合并逻辑
        # 注意：字符串/字节码探测只是启发式。即使命中，我们仍继续执行 patch（幂等）。
        # 若官方改了字段名但语义仍是"覆盖 refs"，启发式可能失败，此时 patch 仍然执行。
        if _official_already_merges(original):
            logger.info(
                "[YimoH3 patch] NOTE: official implementation appears to already "
                "contain keyframes+refs merge logic (heuristic). YimoH3 patch will "
                "still be applied for safety; it is idempotent."
            )

        def _patched_minimax_h3_extra_conds(self, **kwargs):
            out = original(self, **kwargs)

            minimax_payload_cond = out.get("minimax_payload")
            if minimax_payload_cond is None or not hasattr(minimax_payload_cond, "cond"):
                return out

            payload = minimax_payload_cond.cond
            if not isinstance(payload, dict):
                return out

            keyframes = payload.get("keyframes")
            refs = payload.get("refs")

            if keyframes is not None and refs is not None:
                merged_video_latents = []
                merged_audio_latents = []

                # 顺序必须与 CLIP 端一致：refs 在前，keyframes 在后
                for ref in refs:
                    if "latent" in ref and ref["latent"] is not None:
                        merged_video_latents.append(ref["latent"])
                    if "audio_latent" in ref and ref["audio_latent"] is not None:
                        merged_audio_latents.append(ref["audio_latent"])

                for kf in keyframes:
                    merged_video_latents.append(kf["latent"])
                    if "audio_latent" in kf and kf["audio_latent"] is not None:
                        merged_audio_latents.append(kf["audio_latent"])

                payload["cond_video_latents"] = merged_video_latents
                if merged_audio_latents:
                    payload["cond_audio_latents"] = merged_audio_latents

            return out

        setattr(_patched_minimax_h3_extra_conds, _YIMO_PATCH_MARKER, True)
        _patched_minimax_h3_extra_conds._yimo_original = original

        setattr(target_class, method_name, _patched_minimax_h3_extra_conds)
        logger.info(
            "[YimoH3 patch] Patched MiniMaxH3.extra_conds to merge keyframes + refs "
            "cond_video_latents (refs-first order). If official implementation changes, "
            "please report at the plugin repository."
        )
    except Exception as e:
        logger.warning("[YimoH3 patch] Failed to patch MiniMaxH3.extra_conds: %s", e)


_try_patch_minimax_h3()