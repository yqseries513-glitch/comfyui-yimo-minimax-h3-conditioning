# -*- coding: utf-8 -*-
"""Yimo H3 Face Restore Engine

人脸修复引擎：检测 → 身份匹配 → 修复（GFPGAN / OpenCV）→ 贴回。

依赖（全部可选，未安装时自动降级）：
  - insightface（人脸检测 + ArcFace embedding）
  - onnxruntime-gpu / onnxruntime（insightface 推理后端）
  - gfpgan + basicsr + facexlib（高质量人脸重建）
  - opencv-python（基础图像处理，必需）

第三方项目：
  - GFPGAN: https://github.com/TencentARC/GFPGAN (Apache-2.0)
  - InsightFace: https://github.com/deepinsight/insightface (MIT)
  - OpenCV: https://github.com/opencv/opencv (Apache-2.0)
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from typing import Any

import numpy as np
import torch

import comfy.model_management
import comfy.nested_tensor

from .core import nested_av_parts, normalize_image_input, is_connected_value

logger = logging.getLogger("YimoH3")


def _soft_empty_cache():
    try:
        comfy.model_management.soft_empty_cache()
    except Exception:
        pass


# =============================================================================
# 全局状态
# =============================================================================

_GFPGAN_RESTORER = None
_INSIGHTFACE_AVAILABLE = False
_INSIGHTFACE_ANALYZER = None
_OPENCV_AVAILABLE = False
_OPENCV_DETECTOR_READY = False

_OPENCV_MISSING_WARNED = False
_INSIGHTFACE_FAIL_COUNT = 0
_INSIGHTFACE_FAIL_THRESHOLD = 10

_MODEL_DIRS_CACHE = None
_AUTOREGISTER_DONE = False

_FACE_RESTORE_FAIL_LOG_COUNT = 0
_FACE_RESTORE_FAIL_LOG_LIMIT = 3

warnings.filterwarnings("ignore", category=UserWarning, module="gfpgan")
warnings.filterwarnings("ignore", category=FutureWarning, module="basicsr")


# =============================================================================
# 模型目录管理
# =============================================================================

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


_YIMO_KINDS = {
    "insightface": os.path.join("models", "buffalo_l"),
    "facerestore_models": "GFPGANv1.4.pth",
    "opencv": "deploy.prototxt",
}


def _collect_all_registered_paths() -> list[str]:
    paths: list[str] = []
    try:
        import folder_paths
    except Exception:
        return paths

    md = getattr(folder_paths, "models_dir", None)
    if isinstance(md, str) and md:
        paths.append(md)

    fmap = getattr(folder_paths, "folder_names_and_paths", None)
    if isinstance(fmap, dict):
        for _name, entry in fmap.items():
            if not entry:
                continue
            if isinstance(entry, (tuple, list)) and len(entry) >= 1:
                sub = entry[0]
            else:
                sub = entry
            if isinstance(sub, str):
                sub = [sub]
            if not isinstance(sub, (list, tuple)):
                continue
            for p in sub:
                if isinstance(p, str) and p:
                    paths.append(p)
    return paths


def _autoregister_yimo_kinds():
    global _AUTOREGISTER_DONE
    if _AUTOREGISTER_DONE:
        return
    _AUTOREGISTER_DONE = True

    try:
        import folder_paths
    except Exception:
        return

    add_fn = getattr(folder_paths, "add_model_folder_path", None)
    if not callable(add_fn):
        logger.debug("[YimoH3] folder_paths.add_model_folder_path 不可用，跳过自动注册")
        return

    all_paths = _collect_all_registered_paths()
    if not all_paths:
        return

    def _already_registered(kind: str, marker: str) -> bool:
        try:
            existing = folder_paths.get_folder_paths(kind) or []
        except Exception:
            return False
        for p in existing:
            if isinstance(p, str) and os.path.exists(os.path.join(p, marker)):
                return True
        return False

    found: dict[str, str] = {}
    for kind, marker in _YIMO_KINDS.items():
        if _already_registered(kind, marker):
            found[kind] = ""

    for base_path in all_paths:
        cur = base_path
        for _ in range(6):
            for kind, marker in _YIMO_KINDS.items():
                if kind in found:
                    continue
                cand = os.path.join(cur, kind)
                if os.path.exists(os.path.join(cand, marker)):
                    found[kind] = cand
            if len(found) >= len(_YIMO_KINDS):
                break
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
        if len(found) >= len(_YIMO_KINDS):
            break

    for kind, path in found.items():
        if not path:
            continue
        try:
            add_fn(kind, path)
            logger.info("[YimoH3] auto-registered %s -> %s", kind, path)
        except Exception as e:
            logger.debug("[YimoH3] register %s=%s failed: %s", kind, path, e)


def _find_extra_model_paths_yaml() -> list[str]:
    candidates: list[str] = []
    try:
        argv = list(sys.argv)
        for i, arg in enumerate(argv):
            if arg == "--extra-model-paths-config" and i + 1 < len(argv):
                candidates.append(argv[i + 1])
            elif isinstance(arg, str) and arg.startswith("--extra-model-paths-config="):
                candidates.append(arg.split("=", 1)[1])
    except Exception:
        pass

    try:
        import folder_paths
        for attr in ("base_path", "base_path_"):
            base = getattr(folder_paths, attr, None)
            if isinstance(base, str) and base:
                for name in ("extra_model_paths.yaml", "extra_model_paths.yml"):
                    candidates.append(os.path.join(base, name))
    except Exception:
        pass

    for name in ("extra_model_paths.yaml", "extra_model_paths.yml"):
        candidates.append(os.path.join(os.getcwd(), name))

    result: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        if not c:
            continue
        try:
            key = os.path.normcase(os.path.abspath(c))
        except Exception:
            key = c
        if key in seen:
            continue
        seen.add(key)
        if os.path.isfile(c):
            result.append(c)
    return result


def _extract_yaml_base_paths(yaml_path: str) -> list[str]:
    result: list[str] = []
    try:
        import yaml
    except ImportError:
        return result
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return result
    if not isinstance(cfg, dict):
        return result
    for _name, section in cfg.items():
        if not isinstance(section, dict):
            continue
        bp = section.get("base_path")
        if isinstance(bp, str) and bp.strip():
            result.append(bp.strip())
    return result


def _collect_candidate_bases() -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(path: str):
        if not path or not isinstance(path, str):
            return
        if not os.path.isdir(path):
            return
        try:
            key = os.path.normcase(os.path.normpath(os.path.abspath(path)))
        except Exception:
            key = path
        if key in seen:
            return
        seen.add(key)
        candidates.append(path)

    for p in _collect_all_registered_paths():
        cur = p
        for _ in range(6):
            _add(cur)
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent

    for yp in _find_extra_model_paths_yaml():
        for base in _extract_yaml_base_paths(yp):
            _add(base)
            _add(os.path.join(base, "models"))

    _add(os.path.join(os.getcwd(), "models"))
    return candidates


def _resolve_model_dir(kind: str, marker_relpath: str, model_bases: list[str]) -> str:
    candidates = [os.path.join(base, kind) for base in model_bases]
    for cand in candidates:
        if os.path.exists(os.path.join(cand, marker_relpath)):
            logger.info("[YimoH3] %s root resolved to: %s", kind, cand)
            return cand
    return candidates[0] if candidates else os.path.join(os.getcwd(), "models", kind)


def _get_model_dirs():
    global _MODEL_DIRS_CACHE
    if _MODEL_DIRS_CACHE is not None:
        return _MODEL_DIRS_CACHE

    _autoregister_yimo_kinds()

    result: dict[str, str | None] = {}
    for kind, marker in _YIMO_KINDS.items():
        result[kind] = None
        try:
            import folder_paths
            paths = list(folder_paths.get_folder_paths(kind) or [])
            for p in reversed(paths):
                if isinstance(p, str) and os.path.exists(os.path.join(p, marker)):
                    logger.info("[YimoH3] %s resolved via folder_paths: %s", kind, p)
                    result[kind] = p
                    break
        except Exception:
            pass

    missing = [k for k, v in result.items() if v is None]
    if missing:
        bases = _collect_candidate_bases()
        logger.debug("[YimoH3] fallback search paths: %s", bases)
        for k in missing:
            result[k] = _resolve_model_dir(k, _YIMO_KINDS[k], bases)

    _MODEL_DIRS_CACHE = {k: v for k, v in result.items()}  # type: ignore
    for path in _MODEL_DIRS_CACHE.values():
        _ensure_dir(path)
    return _MODEL_DIRS_CACHE


def _get_gfpgan_model_paths():
    dirs = _get_model_dirs()
    facerestore_dir = dirs["facerestore_models"]
    for name in ("GFPGANv1.4.pth", "GFPGANv1.3.pth", "GFPGAN.pth"):
        p = os.path.join(facerestore_dir, name)
        if os.path.exists(p):
            return p
    return os.path.join(facerestore_dir, "GFPGANv1.4.pth")


def _get_insightface_root():
    return _get_model_dirs()["insightface"]


def _get_opencv_model_paths():
    opencv_dir = _get_model_dirs()["opencv"]
    return {
        "prototxt": os.path.join(opencv_dir, "deploy.prototxt"),
        "caffemodel": os.path.join(opencv_dir, "res10_300x300_ssd_iter_140000.caffemodel"),
    }


# =============================================================================
# 依赖可用性检查
# =============================================================================

def _ensure_opencv_available():
    global _OPENCV_AVAILABLE
    if _OPENCV_AVAILABLE:
        return True
    try:
        import cv2
        _OPENCV_AVAILABLE = True
        return True
    except ImportError:
        logger.debug("OpenCV not installed; face restore will use fallback")
        return False


def _get_insightface_analyzer():
    global _INSIGHTFACE_AVAILABLE, _INSIGHTFACE_ANALYZER, _INSIGHTFACE_FAIL_COUNT

    if _INSIGHTFACE_ANALYZER is not None:
        return _INSIGHTFACE_ANALYZER
    if _INSIGHTFACE_FAIL_COUNT >= _INSIGHTFACE_FAIL_THRESHOLD:
        return None

    try:
        from insightface.app import FaceAnalysis
    except ImportError as e:
        logger.debug("insightface 未安装或 import 失败: %s", e)
        _INSIGHTFACE_FAIL_COUNT = _INSIGHTFACE_FAIL_THRESHOLD
        return None

    root = _get_insightface_root()
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = 0
    else:
        providers = ["CPUExecutionProvider"]
        ctx_id = -1
    det_size = (640, 640) if use_gpu else (320, 320)

    original_level = logger.level
    try:
        logger.setLevel(logging.WARNING)
        app = FaceAnalysis(name="buffalo_l", root=root, providers=providers)
        app.prepare(ctx_id=ctx_id, det_size=det_size)
        _INSIGHTFACE_AVAILABLE = True
        _INSIGHTFACE_ANALYZER = app
        _INSIGHTFACE_FAIL_COUNT = 0
        logger.info(
            "insightface 加载成功 (buffalo_l, providers=%s, det_size=%s)",
            providers, det_size,
        )
        return app
    except Exception as e:
        _INSIGHTFACE_FAIL_COUNT += 1
        logger.error(
            "insightface 加载失败 (第 %d 次): %s: %s\n"
            "  root=%s\n"
            "  providers=%s\n"
            "  请检查：\n"
            "  1) <models>/insightface/models/buffalo_l/ 下是否有 5 个 .onnx；\n"
            "  2) onnxruntime-gpu 是否与 CUDA 版本匹配；\n"
            "  3) 是否缺少 onnxruntime 或 onnx 包。",
            _INSIGHTFACE_FAIL_COUNT, type(e).__name__, e,
            root, providers,
            exc_info=True,
        )
        return None
    finally:
        logger.setLevel(original_level)


def _get_opencv_detector():
    global _OPENCV_DETECTOR_READY, _OPENCV_MISSING_WARNED

    if _OPENCV_DETECTOR_READY:
        return True
    if not _ensure_opencv_available():
        return False

    paths = _get_opencv_model_paths()
    prototxt = paths["prototxt"]
    caffemodel = paths["caffemodel"]

    if not os.path.exists(prototxt) or not os.path.exists(caffemodel):
        if not _OPENCV_MISSING_WARNED:
            logger.warning(
                "OpenCV 人脸检测模型缺失，OpenCV fallback 将被禁用。"
                "如需启用，请下载以下文件到 %s：\n"
                "  - deploy.prototxt: %s\n"
                "  - res10_300x300_ssd_iter_140000.caffemodel: %s",
                os.path.dirname(prototxt),
                "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/deploy.prototxt",
                "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/res10_300x300_ssd_iter_140000.caffemodel",
            )
            _OPENCV_MISSING_WARNED = True
        return False

    _OPENCV_DETECTOR_READY = True
    return True


# =============================================================================
# 检测层
# =============================================================================

def detect_faces_insightface(img_bgr: np.ndarray, det_threshold: float = 0.5):
    analyzer = _get_insightface_analyzer()
    if analyzer is None:
        return []
    try:
        faces = analyzer.get(img_bgr)
        results = []
        for f in faces:
            if float(f.det_score) < det_threshold:
                continue
            results.append({
                "bbox": f.bbox.astype(np.float32),
                "kps": f.kps.astype(np.float32) if f.kps is not None else None,
                "embedding": f.normed_embedding.astype(np.float32) if f.normed_embedding is not None else None,
                "score": float(f.det_score),
            })
        results.sort(key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]), reverse=True)
        return results
    except Exception as e:
        logger.debug("insightface 检测失败: %s", e)
        return []


def detect_faces_opencv(img_bgr: np.ndarray, det_threshold: float = 0.5):
    if not _get_opencv_detector():
        return []
    try:
        import cv2
        paths = _get_opencv_model_paths()
        net = cv2.dnn.readNetFromCaffe(paths["prototxt"], paths["caffemodel"])
        (h, w) = img_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img_bgr, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )
        net.setInput(blob)
        detections = net.forward()
        results = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > det_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    results.append({
                        "bbox": np.array([x1, y1, x2, y2], dtype=np.float32),
                        "kps": None, "embedding": None,
                        "score": float(confidence),
                    })
        return results
    except Exception as e:
        logger.debug("OpenCV 检测失败: %s", e)
        return []


def detect_faces(
    img_bgr: np.ndarray,
    det_threshold: float = 0.5,
    fallback_to_opencv: bool = True,
    prefer_insightface: bool = True,
):
    if prefer_insightface:
        faces = detect_faces_insightface(img_bgr, det_threshold)
        if faces:
            return faces
        if not fallback_to_opencv:
            return []
        if _INSIGHTFACE_FAIL_COUNT >= _INSIGHTFACE_FAIL_THRESHOLD:
            return detect_faces_opencv(img_bgr, det_threshold)

    if _get_opencv_detector():
        return detect_faces_opencv(img_bgr, det_threshold)
    if prefer_insightface and _INSIGHTFACE_FAIL_COUNT < _INSIGHTFACE_FAIL_THRESHOLD:
        return detect_faces_insightface(img_bgr, det_threshold)
    return []


# =============================================================================
# GFPGAN 修复器加载
# =============================================================================

def _get_gfpgan_restorer():
    global _GFPGAN_RESTORER

    if _GFPGAN_RESTORER is not None:
        return _GFPGAN_RESTORER

    try:
        from gfpgan import GFPGANer
    except ImportError as e:
        logger.warning("[YimoH3] GFPGAN 未安装或 import 失败: %s", e)
        return None

    model_path = _get_gfpgan_model_paths()
    if not os.path.exists(model_path):
        logger.warning(
            "[YimoH3] GFPGAN 模型不存在: %s\n"
            "  请将 GFPGANv1.4.pth 放在 <ComfyUI 模型根>/facerestore_models/ 下。",
            model_path,
        )
        return None

    use_cuda = torch.cuda.is_available()
    logger.info(
        "[YimoH3] 尝试初始化 GFPGANer: model_path=%s, device=%s",
        model_path, "cuda" if use_cuda else "cpu",
    )

    try:
        _GFPGAN_RESTORER = GFPGANer(
            model_path=model_path,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device="cuda" if use_cuda else "cpu",
        )
        logger.info("[YimoH3] GFPGAN 加载成功 (device=%s)", "cuda" if use_cuda else "cpu")
        return _GFPGAN_RESTORER
    except Exception as e:
        logger.warning(
            "[YimoH3] GFPGAN 加载失败 (CUDA 路径): %s: %s",
            type(e).__name__, e, exc_info=True,
        )

    if use_cuda:
        logger.info("[YimoH3] 尝试回退到 CPU 重新初始化 GFPGANer ...")
        try:
            _GFPGAN_RESTORER = GFPGANer(
                model_path=model_path,
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
                device="cpu",
            )
            logger.info("[YimoH3] GFPGAN 加载成功 (device=cpu, fallback)")
            return _GFPGAN_RESTORER
        except Exception as e:
            logger.warning(
                "[YimoH3] GFPGAN 加载失败 (CPU 回退): %s: %s",
                type(e).__name__, e, exc_info=True,
            )

    return None


# =============================================================================
# OpenCV 增强
# =============================================================================

def enhance_face_opencv(face_patch: np.ndarray, strength: float = 0.5) -> np.ndarray:
    if face_patch.size == 0 or not _ensure_opencv_available():
        return face_patch
    try:
        import cv2
        denoised = cv2.bilateralFilter(face_patch, 9, 75, 75)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) / 1.0
        sharpened = cv2.filter2D(denoised, -1, kernel)
        return cv2.addWeighted(denoised, 1 - strength, sharpened, strength, 0)
    except Exception:
        return face_patch


# =============================================================================
# 对齐 / 贴回核心
# =============================================================================

_ARCFACE_DST_112 = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041],
], dtype=np.float32)


def align_face(img_bgr: np.ndarray, kps: np.ndarray | None, size: int = 512):
    try:
        import cv2
    except ImportError:
        return img_bgr, None
    h, w = img_bgr.shape[:2]

    if kps is None:
        center = (w // 2, h // 2)
        half = min(w, h) // 2
        x1 = max(0, center[0] - half); y1 = max(0, center[1] - half)
        x2 = min(w, center[0] + half); y2 = min(h, center[1] + half)
        cropped = img_bgr[y1:y2, x1:x2]
        if cropped.size == 0:
            return cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR), None
        return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LINEAR), None

    dst = _ARCFACE_DST_112 * (size / 112.0)
    try:
        if len(kps) == 5:
            M, _ = cv2.estimateAffinePartial2D(kps, dst, method=cv2.LMEDS)
            if M is None:
                M = cv2.getAffineTransform(kps[:3].astype(np.float32), dst[:3].astype(np.float32))
            aligned = cv2.warpAffine(
                img_bgr, M, (size, size),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
            )
            return aligned, M
    except Exception as e:
        logger.debug("align_face warpAffine failed: %s", e)
    return cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR), None


def _compute_valid_mask(restored_aligned, inv_M, w, h, fx1, fy1, fx2, fy2):
    try:
        import cv2
        ones = np.ones(restored_aligned.shape[:2], dtype=np.uint8) * 255
        warped = cv2.warpAffine(
            ones, inv_M, (w, h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        return warped[fy1:fy2, fx1:fx2].astype(np.float32) / 255.0
    except Exception as e:
        logger.debug("valid_mask computation failed: %s", e)
        return np.ones((fy2 - fy1, fx2 - fx1), dtype=np.float32)


def _gfpgan_enhance_with_fallback(restorer, aligned, strength):
    """GFPGAN enhance 封装。

    GFPGAN 的 enhance() 返回三元组：
        (cropped_faces_list, restored_faces_list, restored_img)
    has_aligned=True 时 restored_img 为 None，restored_faces 是 list。
    CUDA OOM 时自动尝试移到 CPU 重试。
    """
    def _do_enhance():
        cropped_faces, restored_faces, restored_img = restorer.enhance(
            aligned, has_aligned=True, paste_back=False
        )
        # restored_faces 是 list[np.ndarray]，取第一个
        if isinstance(restored_faces, list):
            if len(restored_faces) == 0:
                raise ValueError("GFPGAN enhance returned empty restored_faces list")
            out = restored_faces[0]
        elif isinstance(restored_faces, np.ndarray):
            out = restored_faces
        else:
            raise ValueError(f"GFPGAN enhance returned unexpected type: {type(restored_faces)}")
        if not isinstance(out, np.ndarray) or out.size == 0:
            raise ValueError("GFPGAN enhance returned empty ndarray")
        return out

    # 第一次尝试（当前 device）
    try:
        restored_aligned = _do_enhance()
        return restored_aligned, True
    except Exception as e:
        err_msg = str(e).lower()
        is_oom = "out of memory" in err_msg or ("cuda" in err_msg and "memory" in err_msg)
        logger.warning(
            "[YimoH3] GFPGAN enhance 失败: %s: %s (is_oom=%s)",
            type(e).__name__, e, is_oom,
            exc_info=not is_oom,
        )

    # CUDA OOM → 尝试把 GFPGAN 移到 CPU 重试
    if is_oom:
        try:
            _soft_empty_cache()
            logger.info("[YimoH3] 尝试将 GFPGAN 移到 CPU 重试 ...")
            try:
                restorer.device = torch.device("cpu")
                if hasattr(restorer, "gfpgan") and restorer.gfpgan is not None:
                    restorer.gfpgan.to("cpu")
                if hasattr(restorer, "face_helper") and restorer.face_helper is not None:
                    if hasattr(restorer.face_helper, "device"):
                        restorer.face_helper.device = torch.device("cpu")
            except Exception as me:
                logger.debug("move GFPGAN to cpu failed: %s", me)

            restored_aligned = _do_enhance()
            logger.info("[YimoH3] GFPGAN enhance 在 CPU 上成功")
            return restored_aligned, True
        except Exception as e2:
            logger.warning("[YimoH3] GFPGAN enhance CPU 回退也失败: %s: %s", type(e2).__name__, e2)

    # 最终降级到 OpenCV（返回 None 让调用方决定）
    return None, False


def restore_single_face(img_bgr, face, strength=0.7, feather=0.25, repair_backend="auto"):
    """返回 ndarray（成功）或 None（失败）。"""
    try:
        return _restore_single_face_impl(img_bgr, face, strength, feather, repair_backend)
    except Exception as e:
        logger.warning(
            "[YimoH3] restore_single_face 未预期错误: %s: %s",
            type(e).__name__, e, exc_info=True,
        )
        return None


def _restore_single_face_impl(img_bgr, face, strength, feather, repair_backend):
    try:
        import cv2
    except ImportError:
        return None

    h, w = img_bgr.shape[:2]
    bbox = face["bbox"]; kps = face.get("kps")
    x1, y1, x2, y2 = bbox.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return img_bgr

    margin = max(5, int((x2 - x1) * 0.1))
    fx1, fy1 = max(0, x1 - margin), max(0, y1 - margin)
    fx2, fy2 = min(w, x2 + margin), min(h, y2 + margin)
    if fx2 <= fx1 or fy2 <= fy1:
        return img_bgr

    face_patch = img_bgr[fy1:fy2, fx1:fx2].copy()
    if kps is not None:
        local_kps = kps - np.array([fx1, fy1], dtype=np.float32)
        aligned, M = align_face(face_patch, local_kps)
    else:
        aligned, M = align_face(face_patch, None)
    if aligned is None or aligned.size == 0:
        return img_bgr

    if repair_backend in ["auto", "gfpgan"]:
        restorer = _get_gfpgan_restorer()
        if restorer is not None:
            restored_aligned, used_gfpgan = _gfpgan_enhance_with_fallback(
                restorer, aligned, strength
            )
            if restored_aligned is None:
                # GFPGAN 失败 → 降级 OpenCV
                restored_aligned = enhance_face_opencv(aligned, strength * 0.5)
        else:
            restored_aligned = enhance_face_opencv(aligned, strength * 0.5)
    elif repair_backend == "insightface":
        restored_aligned = enhance_face_opencv(aligned, strength * 0.7)
    else:
        restored_aligned = enhance_face_opencv(aligned, strength)

    if restored_aligned is None or restored_aligned.size == 0:
        return None
    if restored_aligned.shape[:2] != aligned.shape[:2]:
        try:
            restored_aligned = cv2.resize(restored_aligned, (aligned.shape[1], aligned.shape[0]), interpolation=cv2.INTER_LINEAR)
        except Exception:
            return None

    if M is not None:
        try:
            inv_M = cv2.invertAffineTransform(M)
        except Exception as e:
            logger.debug("invertAffineTransform failed: %s", e)
            inv_M = None
    else:
        inv_M = None

    if inv_M is not None:
        paste = cv2.warpAffine(restored_aligned, inv_M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        patch_restored = paste[fy1:fy2, fx1:fx2].copy()
        valid_mask = _compute_valid_mask(restored_aligned, inv_M, w, h, fx1, fy1, fx2, fy2)
    else:
        try:
            patch_restored = cv2.resize(restored_aligned, (fx2 - fx1, fy2 - fy1), interpolation=cv2.INTER_LINEAR)
        except Exception:
            return None
        valid_mask = np.ones((fy2 - fy1, fx2 - fx1), dtype=np.float32)

    if patch_restored.shape[:2] != face_patch.shape[:2]:
        try:
            patch_restored = cv2.resize(patch_restored, (face_patch.shape[1], face_patch.shape[0]), interpolation=cv2.INTER_LINEAR)
        except Exception:
            return None

    if strength < 1.0:
        blended = cv2.addWeighted(face_patch.astype(np.float32), 1.0 - strength, patch_restored.astype(np.float32), strength, 0)
    else:
        blended = patch_restored.astype(np.float32)

    vm3 = valid_mask[..., None]
    patch_final = face_patch.astype(np.float32) * (1.0 - vm3) + blended * vm3

    mask = np.ones((fy2 - fy1, fx2 - fx1), dtype=np.float32)
    k = max(3, int(min(fy2 - fy1, fx2 - fx1) * feather) | 1)
    if k > 2:
        try:
            mask = cv2.GaussianBlur(mask, (k, k), 0)
        except Exception:
            pass
    final_mask = mask * valid_mask
    fm3 = final_mask[..., None]

    img_bgr[fy1:fy2, fx1:fx2] = (
        img_bgr[fy1:fy2, fx1:fx2].astype(np.float32) * (1.0 - fm3)
        + patch_final.astype(np.float32) * fm3
    ).astype(np.uint8)
    return img_bgr


# =============================================================================
# 身份库
# =============================================================================

def build_identity_bank(face_images, det_threshold: float = 0.5):
    if not is_connected_value(face_images):
        return None
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not installed, cannot build identity bank")
        return None

    imgs = normalize_image_input(face_images, "face_images")
    if not imgs:
        return None

    bank = []
    for idx, t in enumerate(imgs, 1):
        arr = (t[0].detach().cpu().numpy() * 255.0).astype(np.uint8)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        faces = detect_faces_insightface(bgr, det_threshold)
        if not faces:
            if _INSIGHTFACE_FAIL_COUNT >= _INSIGHTFACE_FAIL_THRESHOLD or _get_insightface_analyzer() is None:
                logger.warning(
                    "face_images[%d]: InsightFace 不可用，无法提取身份特征；跳过该参考图", idx
                )
            else:
                logger.warning("face_images[%d]: 未检测到人脸，已跳过", idx)
            continue
        bank.append({
            "id": idx,
            "bbox": faces[0]["bbox"],
            "kps": faces[0].get("kps"),
            "embedding": faces[0].get("embedding"),
            "score": faces[0]["score"],
        })
    if bank:
        logger.info("身份库构建完成: %d 人", len(bank))
    return bank or None


def cos_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# =============================================================================
# 帧序列修复
# =============================================================================

_FACE_RESTORE_CLEANUP_INTERVAL = 32


def restore_frames(
    frames_rgb01: torch.Tensor,
    bank=None,
    strength: float = 0.7,
    similarity_threshold: float = 0.45,
    det_threshold: float = 0.5,
    max_faces_per_frame: int = 10,
    restore_unmatched: bool = True,
    feather: float = 0.25,
    min_face_size: int = 64,
    stride: int = 1,
    repair_backend: str = "auto",
):
    global _FACE_RESTORE_FAIL_LOG_COUNT

    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not installed, cannot restore faces")
        return frames_rgb01, {"error": "OpenCV not installed", "frames": frames_rgb01.shape[0]}

    device, dtype = frames_rgb01.device, frames_rgb01.dtype
    arr = (frames_rgb01.detach().cpu().numpy().clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
    T = arr.shape[0]

    actual_backend = repair_backend
    if repair_backend == "auto":
        if _get_gfpgan_restorer() is not None:
            actual_backend = "gfpgan"
        elif _get_insightface_analyzer() is not None:
            actual_backend = "insightface"
        else:
            actual_backend = "opencv"
    elif repair_backend == "gfpgan" and _get_gfpgan_restorer() is None:
        logger.warning("GFPGAN 不可用，降级到 OpenCV 增强")
        actual_backend = "opencv"
    elif repair_backend == "insightface" and _get_insightface_analyzer() is None:
        logger.warning("InsightFace 不可用，降级到 OpenCV 增强")
        actual_backend = "opencv"

    stats = {
        "frames": T, "faces_total": 0, "faces_restored": 0, "unmatched": 0,
        "person_hits": {}, "backend": actual_backend, "detection": "opencv",
    }

    if actual_backend == "insightface":
        prefer_insightface = True; fallback_to_opencv = False
        stats["detection"] = "insightface (no fallback)"
        logger.info("使用 InsightFace 进行人脸检测（不回退 OpenCV）")
    elif actual_backend == "opencv":
        prefer_insightface = False; fallback_to_opencv = True
        stats["detection"] = "opencv"
        logger.info("使用 OpenCV DNN 进行人脸检测")
    else:
        prefer_insightface = True
        fallback_to_opencv = _get_opencv_detector()
        stats["detection"] = "insightface+opencv" if fallback_to_opencv else "insightface"
        logger.info("使用 InsightFace 进行人脸检测（OpenCV 作为 fallback）" if fallback_to_opencv
                    else "使用 InsightFace 进行人脸检测（OpenCV 不可用，无 fallback）")

    if actual_backend == "gfpgan":
        logger.info("使用 GFPGAN 进行面部修复 (高质量)")
    elif actual_backend == "insightface":
        logger.info("使用 InsightFace 进行面部修复 (中等质量)")
    else:
        logger.info("使用 OpenCV 增强进行面部修复 (轻量)")

    if not prefer_insightface and not _get_opencv_detector():
        logger.warning("无可用的人脸检测器，跳过面部修复")
        stats["error"] = "no detection backend available"
        return frames_rgb01, stats

    for t in range(T):
        if stride > 1 and (t % stride) != 0 and t != T - 1:
            if t > 0 and t % _FACE_RESTORE_CLEANUP_INTERVAL == 0:
                _soft_empty_cache()
            continue

        bgr = cv2.cvtColor(arr[t], cv2.COLOR_RGB2BGR)
        dets = detect_faces(
            bgr, det_threshold,
            fallback_to_opencv=fallback_to_opencv,
            prefer_insightface=prefer_insightface,
        )[:max_faces_per_frame]

        if not dets:
            if t > 0 and t % _FACE_RESTORE_CLEANUP_INTERVAL == 0:
                _soft_empty_cache()
            continue

        changed = False
        for face in dets:
            bbox = face["bbox"]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area < min_face_size * min_face_size:
                continue
            stats["faces_total"] += 1

            matched = False
            if bank and face.get("embedding") is not None:
                best_sim = -1.0; best_id = None
                for person in bank:
                    if person.get("embedding") is None:
                        continue
                    sim = cos_similarity(face["embedding"], person["embedding"])
                    if sim > best_sim:
                        best_sim = sim; best_id = person["id"]
                matched = best_sim >= similarity_threshold
                if matched:
                    stats["person_hits"][f"person_{best_id}"] = stats["person_hits"].get(f"person_{best_id}", 0) + 1

            if not matched and not restore_unmatched:
                stats["unmatched"] += 1
                continue

            try:
                new_bgr = restore_single_face(bgr, face, strength, feather, actual_backend)
            except Exception as e:
                # 前 N 次用 warning + traceback，之后降级到 debug
                if _FACE_RESTORE_FAIL_LOG_COUNT < _FACE_RESTORE_FAIL_LOG_LIMIT:
                    _FACE_RESTORE_FAIL_LOG_COUNT += 1
                    logger.warning(
                        "[YimoH3] 人脸修复失败 (第 %d/%d 次): %s: %s",
                        _FACE_RESTORE_FAIL_LOG_COUNT, _FACE_RESTORE_FAIL_LOG_LIMIT,
                        type(e).__name__, e, exc_info=True,
                    )
                else:
                    logger.debug("人脸修复失败: %s", e)
                continue

            if new_bgr is not None:
                bgr = new_bgr
                stats["faces_restored"] += 1
                changed = True

        if changed:
            arr[t] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if t > 0 and t % _FACE_RESTORE_CLEANUP_INTERVAL == 0:
            _soft_empty_cache()

    out = torch.from_numpy(arr).to(device=device, dtype=dtype) / 255.0
    return out, stats


def stats_report(stats: dict) -> str:
    hits = ", ".join(f"{k}:{v}" for k, v in sorted(stats.get("person_hits", {}).items())) or "none"
    return (
        f"detection={stats.get('detection', 'unknown')}, "
        f"repair={stats.get('backend', 'unknown')}, "
        f"faces={stats.get('faces_total', 0)} detected, "
        f"{stats.get('faces_restored', 0)} restored ({hits}), "
        f"unmatched_skipped={stats.get('unmatched', 0)}"
    )


# =============================================================================
# 潜变量解码 / 修复入口
# =============================================================================

def decode_video_latent(video_latent: torch.Tensor, video_vae) -> torch.Tensor:
    decoded = video_vae.decode(video_latent)
    if not isinstance(decoded, torch.Tensor):
        raise ValueError(f"video_vae.decode returned {type(decoded).__name__}, expected Tensor")
    if decoded.ndim == 5:
        if decoded.shape[0] == 1:
            decoded = decoded.squeeze(0)
        elif decoded.shape[1] == 1:
            decoded = decoded.squeeze(1)
    if decoded.ndim == 4:
        if decoded.shape[-1] in (1, 3, 4):
            return decoded
        elif decoded.shape[0] in (1, 3, 4):
            return decoded.permute(1, 2, 3, 0)
    raise ValueError(f"Unexpected video_vae.decode output shape: {tuple(decoded.shape)}")


def restore_sampled_av(sampled: dict, video_vae, bank, **params):
    video, audio = nested_av_parts(sampled)
    frames = decode_video_latent(video, video_vae)
    restored, stats = restore_frames(frames, bank, **params)
    new_video = video_vae.encode(restored)
    out = sampled.copy()
    out["samples"] = comfy.nested_tensor.NestedTensor((new_video, audio))
    return out, stats_report(stats)