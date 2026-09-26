# -*- coding: utf-8 -*-
"""Yimo H3 标准提示词格式化节点

将去重后的 8 个官方字段输入组合为 MiniMax H3 官方格式的三段式或六段式提示词。

每个 section 前面插入一个只读的 Combo 标题 widget（display_name 会由 ComfyUI
原生渲染在 widget 上方），用户直接看到字段名，不受输入内容影响。
"""

from __future__ import annotations

import logging

from comfy_api.latest import io

from .i18n import t

logger = logging.getLogger("YimoH3")

CATEGORY = t("category")


# =============================================================================
# 字段顺序定义
# =============================================================================

_THREE_SECTION_ORDER = [
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
]

_SIX_SECTION_ORDER = [
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
]


def _compose_prompt(
    structure: str,
    fields: dict[str, str],
    global_suffix: str,
) -> tuple[str, list[dict]]:
    """组合提示词。"""
    if structure == "six_section":
        order = _SIX_SECTION_ORDER
    else:
        order = _THREE_SECTION_ORDER

    parts: list[str] = []
    section_info: list[dict] = []

    for idx, field_name in enumerate(order, 1):
        content = (fields.get(field_name) or "").strip()
        if content:
            parts.append(f"{field_name}:\n{content}")
            section_info.append({
                "index": idx,
                "field": field_name,
                "chars": len(content),
                "status": "included",
            })
        else:
            section_info.append({
                "index": idx,
                "field": field_name,
                "chars": 0,
                "status": "skipped (empty)",
            })

    all_fields = set(_THREE_SECTION_ORDER) | set(_SIX_SECTION_ORDER)
    for field_name in sorted(all_fields - set(order)):
        content = (fields.get(field_name) or "").strip()
        section_info.append({
            "index": "-",
            "field": field_name,
            "chars": len(content) if content else 0,
            "status": "ignored (not in current structure)" if not content else "ignored (has content but not in current structure)",
        })

    suffix_str = (global_suffix or "").strip()
    if suffix_str:
        parts.append(suffix_str)
        section_info.append({
            "index": "suffix",
            "field": "global_suffix",
            "chars": len(suffix_str),
            "status": "included",
        })
    else:
        section_info.append({
            "index": "suffix",
            "field": "global_suffix",
            "chars": 0,
            "status": "skipped (empty)",
        })

    composed = "\n\n".join(parts)
    return composed, section_info


class YimoH3PromptComposer(io.ComfyNode):
    """将分段输入组合为 MiniMax H3 官方格式的标准提示词。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YimoH3PromptComposer",
            display_name=t("prompt_composer_display_name"),
            description=t("prompt_composer_description"),
            category=CATEGORY,
            inputs=[
                io.Combo.Input("structure",
                    display_name=t("prompt_composer_structure_input"),
                    options=["three_section", "six_section"],
                    default="three_section",
                    tooltip=t("prompt_composer_structure_tooltip"),
                ),

                # ----- 三段式独有 -----
                # 标题 widget（只读 Combo）
                io.Combo.Input("_header_integrated_multimodal_description",
                    display_name="integrated_multimodal_description",
                    options=[""],
                    default="",
                    tooltip=t("prompt_composer_field_integrated_multimodal_description_tooltip"),
                ),
                io.String.Input("integrated_multimodal_description",
                    display_name="integrated_multimodal_description",
                    multiline=True, dynamic_prompts=True, default="",
                    tooltip=t("prompt_composer_field_integrated_multimodal_description_tooltip"),
                ),

                # ----- 六段式独有 -----
                io.Combo.Input("_header_subject_definitions",
                    display_name="subject_definitions",
                    options=[""],
                    default="",
                    tooltip=t("prompt_composer_field_subject_definitions_tooltip"),
                ),
                io.String.Input("subject_definitions",
                    display_name="subject_definitions",
                    multiline=True, dynamic_prompts=True, default="",
                    tooltip=t("prompt_composer_field_subject_definitions_tooltip"),
                ),

                io.Combo.Input("_header_summary",
                    display_name="summary",
                    options=[""],
                    default="",
                    tooltip=t("prompt_composer_field_summary_tooltip"),
                ),
                io.String.Input("summary",
                    display_name="summary",
                    multiline=True, dynamic_prompts=True, default="",
                    tooltip=t("prompt_composer_field_summary_tooltip"),
                ),

                io.Combo.Input("_header_retention_analysis",
                    display_name="retention_analysis",
                    options=[""],
                    default="",
                    tooltip=t("prompt_composer_field_retention_analysis_tooltip"),
                ),
                io.String.Input("retention_analysis",
                    display_name="retention_analysis",
                    multiline=True, dynamic_prompts=True, default="",
                    tooltip=t("prompt_composer_field_retention_analysis_tooltip"),
                ),

                io.Combo.Input("_header_detailed_description",
                    display_name="detailed_description",
                    options=[""],
                    default="",
                    tooltip=t("prompt_composer_field_detailed_description_tooltip"),
                ),
                io.String.Input("detailed_description",
                    display_name="detailed_description",
                    multiline=True, dynamic_prompts=True, default="",
                    tooltip=t("prompt_composer_field_detailed_description_tooltip"),
                ),

                # ----- 三段式与六段式共用 -----
                io.Combo.Input("_header_overall_soundscape",
                    display_name="overall_soundscape",
                    options=[""],
                    default="",
                    tooltip=t("prompt_composer_field_overall_soundscape_tooltip"),
                ),
                io.String.Input("overall_soundscape",
                    display_name="overall_soundscape",
                    multiline=True, dynamic_prompts=True, default="",
                    tooltip=t("prompt_composer_field_overall_soundscape_tooltip"),
                ),

                io.Combo.Input("_header_non_diegetic_music",
                    display_name="non_diegetic_music",
                    options=[""],
                    default="",
                    tooltip=t("prompt_composer_field_non_diegetic_music_tooltip"),
                ),
                io.String.Input("non_diegetic_music",
                    display_name="non_diegetic_music",
                    multiline=True, dynamic_prompts=True, default="",
                    tooltip=t("prompt_composer_field_non_diegetic_music_tooltip"),
                ),

                # ----- 通用 -----
                io.Combo.Input("_header_global_suffix",
                    display_name=t("prompt_composer_global_suffix_input"),
                    options=[""],
                    default="",
                    tooltip=t("prompt_composer_global_suffix_tooltip"),
                ),
                io.String.Input("global_suffix",
                    display_name=t("prompt_composer_global_suffix_input"),
                    multiline=True, dynamic_prompts=True, default="",
                    tooltip=t("prompt_composer_global_suffix_tooltip"),
                ),
            ],
            outputs=[
                io.String.Output(display_name=t("prompt_composer_output_prompt")),
                io.String.Output(display_name=t("prompt_composer_output_report")),
            ],
        )

    @classmethod
    def execute(cls,
                structure,
                _header_integrated_multimodal_description="",
                integrated_multimodal_description="",
                _header_subject_definitions="",
                subject_definitions="",
                _header_summary="",
                summary="",
                _header_retention_analysis="",
                retention_analysis="",
                _header_detailed_description="",
                detailed_description="",
                _header_overall_soundscape="",
                overall_soundscape="",
                _header_non_diegetic_music="",
                non_diegetic_music="",
                _header_global_suffix="",
                global_suffix="",
                **kwargs):

        fields = {
            "integrated_multimodal_description": integrated_multimodal_description,
            "subject_definitions": subject_definitions,
            "summary": summary,
            "retention_analysis": retention_analysis,
            "detailed_description": detailed_description,
            "overall_soundscape": overall_soundscape,
            "non_diegetic_music": non_diegetic_music,
        }

        composed, section_info = _compose_prompt(structure, fields, global_suffix)

        report_lines = [
            "=== Yimo H3 标准提示词格式化 ===",
            f"structure={structure}",
        ]
        for info in section_info:
            report_lines.append(
                f"[{info['index']}] {info['field']}: "
                f"{info['chars']} chars [{info['status']}]"
            )
        report_lines.append("---")
        report_lines.append(f"output_chars={len(composed)}")
        report_lines.append(f"output_lines={composed.count(chr(10)) + 1 if composed else 0}")

        report = "\n".join(report_lines)

        return io.NodeOutput(composed, report)