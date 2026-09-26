from __future__ import annotations

import json
import re
from typing import Any


# =============================================================================
# 媒体标签规范化
# ---------------------------------------------------------------------------
# v2.6.1: 仅匹配带尖括号的标签，避免误伤正文裸词。
#   <Picture 1> / <picture 1> / <Image 1> / <image 1> / <IMG 1> / <Pic 1>
#   <Video 1>   / <video 1>   / <Vid 1>
#   <Audio 1>   / <audio 1>   / <Sound 1>
# 归一化目标：官方标准格式 <Picture N> / <Video N> / <Audio N>
# =============================================================================

_TAG_TYPE_ALIASES = {
    "picture": "Picture",
    "image": "Picture",
    "img": "Picture",
    "pic": "Picture",
    "photo": "Picture",
    "video": "Video",
    "vid": "Video",
    "audio": "Audio",
    "sound": "Audio",
}

_ALIAS_PATTERN = "|".join(sorted(_TAG_TYPE_ALIASES.keys(), key=len, reverse=True))

# v2.6.1: 只保留尖括号分支
MEDIA_TAG_RE = re.compile(
    rf"<\s*({_ALIAS_PATTERN})\s*(\d+)\s*>",
    re.IGNORECASE,
)

OFFICIAL_TAG_RE = re.compile(r"<(Picture|Video|Audio)\s+(\d+)>", re.IGNORECASE)

_SUBJECT_TAG_RE = re.compile(
    r"<\s*("
    r"Subject|subject|SUBJECT|"
    r"人物|角色|服装|衣服|鞋子|物品|主体|对象|场景|背景|环境|风格|动作|姿态"
    r")\s*(\d+)\s*>",
    re.UNICODE,
)

_STANDARD_SUBJECT_RE = re.compile(r"<\s*Subject\s+(\d+)\s*>", re.IGNORECASE)

_NATIVE_LABEL_WHITELIST = frozenset({
    "人物", "角色", "服装", "衣服", "鞋子", "物品",
    "主体", "对象", "场景", "背景", "环境", "风格",
})


def canonicalize_media_tags(prompt: str) -> str:
    """把各种写法的媒体标签统一为官方格式 <Picture N> / <Video N> / <Audio N>。"""

    def replacement(match: re.Match) -> str:
        raw_type = (match.group(1) or "").lower()
        ordinal_str = match.group(2)
        if not raw_type or not ordinal_str:
            return match.group(0)
        official_type = _TAG_TYPE_ALIASES.get(raw_type)
        if official_type is None:
            return match.group(0)
        try:
            ordinal = int(ordinal_str)
        except ValueError:
            return match.group(0)
        return f"<{official_type} {ordinal}>"

    return MEDIA_TAG_RE.sub(replacement, prompt or "")


def detect_subject_tags(prompt: str) -> tuple[list[str], list[str]]:
    if not prompt:
        return [], []

    all_tags: list[str] = []
    non_standard: list[str] = []

    for match in _SUBJECT_TAG_RE.finditer(prompt):
        raw_label = match.group(1)
        full_tag = match.group(0)
        all_tags.append(full_tag)

        if _STANDARD_SUBJECT_RE.fullmatch(full_tag):
            continue
        if raw_label in _NATIVE_LABEL_WHITELIST:
            continue
        non_standard.append(full_tag)

    seen = set()
    deduped_all = []
    for t in all_tags:
        if t not in seen:
            seen.add(t)
            deduped_all.append(t)

    seen = set()
    deduped_non_standard = []
    for t in non_standard:
        if t not in seen:
            seen.add(t)
            deduped_non_standard.append(t)

    return deduped_all, deduped_non_standard


def prepare_prompt(
    prompt: str,
    counts: dict[str, int],
    source_audio_ordinal: int = 0,
    strict: bool = True,
) -> tuple[str, list[str]]:
    normalized = canonicalize_media_tags(prompt)

    warnings: list[str] = []
    limits = {
        "picture": int(counts.get("pictures", 0)),
        "video": int(counts.get("videos", 0)),
        "audio": int(counts.get("audios", 0)),
    }

    for match in OFFICIAL_TAG_RE.finditer(normalized):
        media_type = match.group(1).lower()
        ordinal = int(match.group(2))
        if ordinal < 1 or ordinal > limits[media_type]:
            warnings.append(
                f"{match.group(0)} is not connected; available {media_type} count is {limits[media_type]}"
            )

    if strict and any("is not connected" in w for w in warnings):
        raise ValueError("MiniMax H3 prompt media tag validation failed: " + "; ".join(warnings))

    return normalized, warnings


def media_map_json(
    pictures: list[str], videos: list[str], audios: list[str], source_audio_ordinal: int = 0,
    keyframe_labels: list[str] | None = None,
    port_mapping: list[dict] | None = None,
) -> str:
    data: dict[str, Any] = {
        "pictures": {str(index + 1): label for index, label in enumerate(pictures)},
        "videos": {str(index + 1): label for index, label in enumerate(videos)},
        "audios": {str(index + 1): label for index, label in enumerate(audios)},
        "source_audio_ordinal": source_audio_ordinal or None,
    }
    if keyframe_labels:
        data["keyframes"] = {str(index + 1): label for index, label in enumerate(keyframe_labels)}
    if port_mapping:
        data["port_mapping"] = port_mapping
    return json.dumps(data, ensure_ascii=False, indent=2)