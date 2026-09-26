# -*- coding: utf-8 -*-
"""Yimo H3 Face Restore Engine (v2.7.1) - 使用 ComfyUI 标准模型目录

v2.7.1:
- restore_frames 每 32 帧执行一次 soft_empty_cache，避免长视频显存累积。
- 新增 _soft_empty_cache() 辅助函数。

v2.6.1:
- restore_frames: 加 .clip(0.0, 1.0)，防止 uint8 回绕黑点。

v2.5.2 重构：
- align_face 使用 BORDER_REPLICATE 避免 warp 黑边；
- restore_single_face 使用 valid_mask 只贴有效区域。

面部修复逻辑分层：
  1. 检测层：InsightFace 优先 → OpenCV 兜底
  2. 身份匹配层：InsightFace ArcFace embedding → 余弦相似度
  3. 修复层：
       - gfpgan  → 真正的人脸重建（如果可用）
       - insightface → OpenCV 增强（轻量锐化，无重建能力）
       - opencv  → OpenCV 增强
       - auto    → GFPGAN 优先，否则 OpenCV 增强
  4. 贴回层：align_face 反变换 + valid_mask + feather mask 三重保证
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any

import numpy as np
import torch

import comfy.model_management
import comfy.nested_tensor

from .core import nested_av_parts, normalize_image_input, is_connected_value

logger = logging.getLogger("YimoH3")


def _soft_empty_cache():
    """v2.7.1: 安全地请求 ComfyUI 清空缓存。

    不抛异常，失败时静默跳过。用于长视频逐帧修复循环，
    避免中间张量与 CUDA 缓存随帧数累积。
    """
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

warnings.filterwarnings("ignore", category=UserWarning, module="gfpgan")
warnings.filterwarnings("ignore", category=FutureWarning, module="basicsr")


# =============================================================================
# 模型目录管理
# =============================================================================

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _get_model_dirs():
    global _MODEL_DIRS_CACHE

    if _MODEL_DIRS_CACHE is not None:
        return _MODEL_DIRS_CACHE

    try:
        import folder_paths
        try:
            model_base = folder_paths.get_folder_paths("models")[0]
        except (KeyError, IndexError):
            model_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
            if not os.path.exists(model_base):
                model_base = os.path.join(os.getcwd(), "models")
    except Exception:
        model_base = os.path.join(os.getcwd(), "models")

    _MODEL_DIRS_CACHE = {
        "facerestore": os.path.join(model_base, "facerestore_models"),
        "insightface": os.path.join(model_base, "insightface"),
        "opencv": os.path.join(model_base, "opencv"),
    }

    for path in _MODEL_DIRS_CACHE.values():
        _ensure_dir(path)

    return _MODEL_DIRS_CACHE


def _get_gfpgan_model_paths():
    dirs = _get_model_dirs()
    facerestore_dir = dirs["facerestore"]
    possible_paths = [
        os.path.join(facerestore_dir, "GFPGANv1.4.pth"),
        os.path.join(facerestore_dir, "GFPGANv1.3.pth"),
        os.path.join(facerestore_dir, "GFPGAN.pth"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return possible_paths[0]


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
    except ImportError:
        logger.debug("insightface 未安装，将使用 OpenCV 备用方案")
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
        logger.setLevel(logging.ERROR)

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
        logger.warning(
            "insightface 加载失败 (第 %d 次): %s: %s",
            _INSIGHTFACE_FAIL_COUNT, type(e).__name__, e,
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

    try:
        _OPENCV_DETECTOR_READY = True
        return True
    except Exception as e:
        logger.debug("OpenCV 模型加载失败: %s", e)
        return False


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
        prototxt = paths["prototxt"]
        caffemodel = paths["caffemodel"]

        net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
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
                        "kps": None,
                        "embedding": None,
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
# 修复器加载
# =============================================================================

def _get_gfpgan_restorer():
    global _GFPGAN_RESTORER

    if _GFPGAN_RESTORER is not None:
        return _GFPGAN_RESTORER

    try:
        from gfpgan import GFPGANer

        model_path = _get_gfpgan_model_paths()
        if not os.path.exists(model_path):
            logger.info("GFPGAN 模型不存在，使用 OpenCV 增强降级方案")
            logger.info("请将 GFPGANv1.4.pth 放置到: %s", model_path)
            return None

        _GFPGAN_RESTORER = GFPGANer(
            model_path=model_path,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        logger.info("GFPGAN 加载成功")
        return _GFPGAN_RESTORER
    except ImportError:
        logger.debug("GFPGAN 未安装，使用 OpenCV 增强降级方案")
        return None
    except Exception as e:
        logger.debug("GFPGAN 加载失败: %s", e)
        return None


# =============================================================================
# OpenCV 增强（"修复"降级方案）
# =============================================================================

def enhance_face_opencv(face_patch: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """OpenCV 双边滤波 + 锐化。无重建能力，只能轻微增强。"""
    if face_patch.size == 0 or not _ensure_opencv_available():
        return face_patch

    try:
        import cv2
        denoised = cv2.bilateralFilter(face_patch, 9, 75, 75)
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]]) / 1.0
        sharpened = cv2.filter2D(denoised, -1, kernel)
        result = cv2.addWeighted(denoised, 1 - strength, sharpened, strength, 0)
        return result
    except Exception:
        return face_patch


# =============================================================================
# 对齐 / 贴回核心
# =============================================================================

# ArcFace 112×112 标准对齐坐标，来自 InsightFace：
#   https://github.com/deepinsight/insightface
# 5 点对应：左眼、右眼、鼻尖、左嘴角、右嘴角。
_ARCFACE_DST_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
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
        x1 = max(0, center[0] - half)
        y1 = max(0, center[1] - half)
        x2 = min(w, center[0] + half)
        y2 = min(h, center[1] + half)
        cropped = img_bgr[y1:y2, x1:x2]
        if cropped.size == 0:
            return cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR), None
        return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LINEAR), None

    dst = _ARCFACE_DST_112 * (size / 112.0)

    try:
        if len(kps) == 5:
            M, _ = cv2.estimateAffinePartial2D(kps, dst, method=cv2.LMEDS)
            if M is None:
                M = cv2.getAffineTransform(
                    kps[:3].astype(np.float32),
                    dst[:3].astype(np.float32)
                )
            aligned = cv2.warpAffine(
                img_bgr, M, (size, size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            return aligned, M
    except Exception as e:
        logger.debug("align_face warpAffine failed: %s", e)

    return cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR), None


def _compute_valid_mask(restored_aligned: np.ndarray, inv_M: np.ndarray,
                        w: int, h: int, fx1: int, fy1: int,
                        fx2: int, fy2: int) -> np.ndarray:
    try:
        import cv2
        ones = np.ones(restored_aligned.shape[:2], dtype=np.uint8) * 255
        warped = cv2.warpAffine(
            ones, inv_M, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return warped[fy1:fy2, fx1:fx2].astype(np.float32) / 255.0
    except Exception as e:
        logger.debug("valid_mask computation failed: %s", e)
        return np.ones((fy2 - fy1, fx2 - fx1), dtype=np.float32)


def restore_single_face(
    img_bgr: np.ndarray,
    face: dict,
    strength: float = 0.7,
    feather: float = 0.25,
    repair_backend: str = "auto",
) -> np.ndarray:
    try:
        import cv2
    except ImportError:
        return img_bgr

    h, w = img_bgr.shape[:2]
    bbox = face["bbox"]
    kps = face.get("kps")

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
            try:
                _, restored_aligned, _ = restorer.enhance(
                    aligned, has_aligned=True, paste_back=False
                )
            except Exception as e:
                logger.debug("GFPGAN enhance failed: %s", e)
                restored_aligned = enhance_face_opencv(aligned, strength * 0.5)
        else:
            restored_aligned = enhance_face_opencv(aligned, strength * 0.5)
    elif repair_backend == "insightface":
        restored_aligned = enhance_face_opencv(aligned, strength * 0.7)
    else:
        restored_aligned = enhance_face_opencv(aligned, strength)

    if restored_aligned is None or restored_aligned.size == 0:
        return img_bgr

    if restored_aligned.shape[:2] != aligned.shape[:2]:
        try:
            restored_aligned = cv2.resize(
                restored_aligned, (aligned.shape[1], aligned.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        except Exception:
            return img_bgr

    if M is not None:
        try:
            inv_M = cv2.invertAffineTransform(M)
        except Exception as e:
            logger.debug("invertAffineTransform failed: %s", e)
            inv_M = None
    else:
        inv_M = None

    if inv_M is not None:
        paste = cv2.warpAffine(
            restored_aligned, inv_M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        patch_restored = paste[fy1:fy2, fx1:fx2].copy()
        valid_mask = _compute_valid_mask(
            restored_aligned, inv_M, w, h, fx1, fy1, fx2, fy2
        )
    else:
        try:
            patch_restored = cv2.resize(
                restored_aligned, (fx2 - fx1, fy2 - fy1),
                interpolation=cv2.INTER_LINEAR,
            )
        except Exception:
            return img_bgr
        valid_mask = np.ones((fy2 - fy1, fx2 - fx1), dtype=np.float32)

    if patch_restored.shape[:2] != face_patch.shape[:2]:
        try:
            patch_restored = cv2.resize(
                patch_restored, (face_patch.shape[1], face_patch.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        except Exception:
            return img_bgr

    if strength < 1.0:
        blended = cv2.addWeighted(
            face_patch.astype(np.float32),
            1.0 - strength,
            patch_restored.astype(np.float32),
            strength,
            0,
        )
    else:
        blended = patch_restored.astype(np.float32)

    vm3 = valid_mask[..., None]
    patch_final = (
        face_patch.astype(np.float32) * (1.0 - vm3)
        + blended * vm3
    )

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
                    "face_images[%d]: InsightFace 不可用，无法提取身份特征；"
                    "跳过该参考图（OpenCV 无法提供 embedding，不能用于身份匹配）", idx
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
# 帧序列修复（v2.7.1 修改点）
# =============================================================================

# v2.7.1: 每处理多少帧执行一次显存清理
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
    """对帧序列执行人脸检测+修复。

    v2.7.1: 每处理 32 帧执行一次 soft_empty_cache，避免长视频显存累积。
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not installed, cannot restore faces")
        return frames_rgb01, {"error": "OpenCV not installed", "frames": frames_rgb01.shape[0]}

    device, dtype = frames_rgb01.device, frames_rgb01.dtype
    # v2.6.1: 加 clamp 防止 uint8 回绕
    arr = (
        frames_rgb01.detach().cpu().numpy().clip(0.0, 1.0) * 255.0
    ).round().astype(np.uint8)
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
        "frames": T,
        "faces_total": 0,
        "faces_restored": 0,
        "unmatched": 0,
        "person_hits": {},
        "backend": actual_backend,
        "detection": "opencv",
    }

    if actual_backend == "insightface":
        prefer_insightface = True
        fallback_to_opencv = False
        stats["detection"] = "insightface (no fallback)"
        logger.info("使用 InsightFace 进行人脸检测（不回退 OpenCV）")
    elif actual_backend == "opencv":
        prefer_insightface = False
        fallback_to_opencv = True
        stats["detection"] = "opencv"
        logger.info("使用 OpenCV DNN 进行人脸检测")
    else:
        prefer_insightface = True
        fallback_to_opencv = _get_opencv_detector()
        stats["detection"] = "insightface+opencv" if fallback_to_opencv else "insightface"
        if fallback_to_opencv:
            logger.info("使用 InsightFace 进行人脸检测（OpenCV 作为 fallback）")
        else:
            logger.info("使用 InsightFace 进行人脸检测（OpenCV 不可用，无 fallback）")

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
            # v2.7.1: 跳过的帧也要定期清理
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
            # v2.7.1: 无人脸帧也要定期清理
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
                best_sim = -1.0
                best_id = None
                for person in bank:
                    if person.get("embedding") is None:
                        continue
                    sim = cos_similarity(face["embedding"], person["embedding"])
                    if sim > best_sim:
                        best_sim = sim
                        best_id = person["id"]
                matched = best_sim >= similarity_threshold
                if matched:
                    stats["person_hits"][f"person_{best_id}"] = \
                        stats["person_hits"].get(f"person_{best_id}", 0) + 1

            if not matched and not restore_unmatched:
                stats["unmatched"] += 1
                continue

            try:
                bgr = restore_single_face(bgr, face, strength, feather, actual_backend)
                stats["faces_restored"] += 1
                changed = True
            except Exception as e:
                logger.debug("人脸修复失败: %s", e)

        if changed:
            arr[t] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # v2.7.1: 每 32 帧清理一次显存缓存
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
    """将 video latent 解码为帧序列。输出统一为 [T, H, W, C]。"""
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

    raise ValueError(
        f"Unexpected video_vae.decode output shape: {tuple(decoded.shape)}"
    )


def restore_sampled_av(sampled: dict, video_vae, bank, **params):
    """对采样后的 AV latent 执行面部修复。"""
    video, audio = nested_av_parts(sampled)
    frames = decode_video_latent(video, video_vae)
    restored, stats = restore_frames(frames, bank, **params)
    new_video = video_vae.encode(restored)
    out = sampled.copy()
    out["samples"] = comfy.nested_tensor.NestedTensor((new_video, audio))
    return out, stats_report(stats)