# -*- coding: utf-8 -*-
"""Yimo H3 加载保存的潜变量节点 (v2.6.3)"""

from __future__ import annotations

import os
import json
import logging
import torch
from comfy_api.latest import io

from .i18n import t
from .core import is_connected_value

logger = logging.getLogger("YimoH3")

CATEGORY = t("category")


class YimoH3LoadSavedLatent(io.ComfyNode):
    """加载之前保存的 AV Latent 文件，主要用于 SingleSampler 的断点续跑。"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="YimoH3LoadSavedLatent",
            display_name="Yimo H3 加载保存的潜变量",
            description="加载之前保存的 AV Latent 文件，用于 SingleSampler 的断点续跑。",
            category=CATEGORY,
            inputs=[
                io.String.Input("file_path", default="",
                    tooltip="完整的文件路径，如: output/yimo_h3/single_sampler/project/segment_0.pt"),
                io.Combo.Input("load_type",
                    options=["single_sampler", "sequence_sampler_export", "sequence_sampler_cache"],
                    default="single_sampler",
                    tooltip="选择加载哪种类型的文件:\n"
                            "- single_sampler: 加载 SingleSampler 保存的完整 AV Latent\n"
                            "- sequence_sampler_export: 加载 SequenceSampler 导出的最终结果\n"
                            "- sequence_sampler_cache: 加载 SequenceSampler 的缓存段"),
                io.Int.Input("segment_index", default=-1, min=-1, max=20,
                    tooltip="从 sequence_sampler_cache 中加载指定段。"
                            "-1 表示加载整个缓存文件（包含所有段）"),
                io.Boolean.Input("verbose", default=True, advanced=True,
                    tooltip="是否在报告中显示详细元数据"),
            ],
            outputs=[
                io.Latent.Output(display_name="AV 潜变量"),
                io.String.Output(display_name="元数据（JSON）"),
                io.String.Output(display_name="加载报告"),
            ],
        )

    @classmethod
    def execute(cls, file_path, load_type, segment_index, verbose):
        if not file_path or not str(file_path).strip():
            raise ValueError("请指定文件路径")

        file_path = str(file_path).strip()
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        if not file_path.endswith(".pt"):
            logger.warning("文件扩展名不是 .pt，尝试加载: %s", file_path)

        try:
            # 该节点需要加载包含自定义类的 .pt（如 NestedTensor、SegmentSummary），
            # 无法使用 weights_only=True。请只加载你自己生成的可信文件。
            data = torch.load(file_path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise RuntimeError(f"加载文件失败: {e}")

        if not isinstance(data, dict):
            raise ValueError("文件内容不是预期的 dict 格式")

        logger.warning(
            "[YimoH3LoadSavedLatent] loaded %s with weights_only=False; "
            "only load files you trust.", file_path,
        )

        latent = None
        metadata = {}
        report_lines = [
            "=== Yimo H3 LoadSavedLatent 报告 ===",
            f"文件: {file_path}",
            f"加载类型: {load_type}",
            "【安全提示】本节点使用 weights_only=False 加载 .pt，请仅加载可信来源的文件。",
        ]

        if load_type == "single_sampler":
            latent = data.get("sampled")
            metadata = data.get("metadata", {})
            if latent is None:
                raise ValueError("single_sampler 格式错误：缺少 'sampled' 字段")
            report_lines.append("✅ 加载 SingleSampler 结果")
            report_lines.append(f"   mode: {metadata.get('mode', 'unknown')}")
            report_lines.append(f"   seed: {metadata.get('seed', 'unknown')}")
            report_lines.append(f"   frame_count: {metadata.get('frame_count', 'unknown')}")
            report_lines.append(f"   project: {metadata.get('project', 'default')}")
            report_lines.append(f"   timestamp: {metadata.get('timestamp', 'unknown')}")

        elif load_type == "sequence_sampler_export":
            latent = data.get("final_latent")
            metadata = data.get("metadata", {})
            if latent is None:
                raise ValueError("sequence_sampler_export 格式错误：缺少 'final_latent' 字段")
            report_lines.append("✅ 加载 SequenceSampler 导出结果")
            report_lines.append(f"   chain_id: {metadata.get('chain_id', 'unknown')}")
            report_lines.append(f"   total_segments: {metadata.get('total_segments', 'unknown')}")
            report_lines.append(f"   segments_sampled: {metadata.get('segments_sampled', 'unknown')}")
            report_lines.append(f"   timestamp: {metadata.get('timestamp', 'unknown')}")

        elif load_type == "sequence_sampler_cache":
            if segment_index >= 0:
                if "sampled" in data:
                    if segment_index == 0:
                        latent = data.get("sampled")
                        summary = data.get("summary", {})
                        report_lines.append("✅ 加载 SequenceSampler 缓存段 (单段)")
                        report_lines.append(f"   mode: {summary.get('mode', 'unknown')}")
                        report_lines.append(f"   frame_count: {summary.get('frame_count', 'unknown')}")
                    else:
                        all_segments = data.get("segments", [])
                        if segment_index < len(all_segments):
                            latent = all_segments[segment_index].get("sampled")
                            report_lines.append(f"✅ 加载 SequenceSampler 缓存段 #{segment_index}")
                        else:
                            raise ValueError(f"段索引 {segment_index} 超出范围（共 {len(all_segments)} 段）")
                else:
                    all_segments = data.get("segments", [])
                    if segment_index < len(all_segments):
                        latent = all_segments[segment_index].get("sampled")
                        report_lines.append(f"✅ 加载 SequenceSampler 缓存段 #{segment_index}")
                    else:
                        raise ValueError(f"段索引 {segment_index} 超出范围（共 {len(all_segments)} 段）")
            else:
                if "segments" in data:
                    segments_data = data.get("segments", [])
                    if segments_data:
                        latent = segments_data[0].get("sampled")
                        report_lines.append(f"✅ 加载 SequenceSampler 缓存（共 {len(segments_data)} 段）")
                        report_lines.append("   ⚠️ 注意：仅返回第一段，如需完整视频请使用 sequence_sampler_export")
                elif "sampled" in data:
                    latent = data.get("sampled")
                    report_lines.append("✅ 加载 SequenceSampler 缓存段")
                else:
                    raise ValueError("sequence_sampler_cache 格式错误：缺少 'sampled' 或 'segments' 字段")

            metadata = data.get("metadata", data.get("summary", {}))

        else:
            raise ValueError(f"未知的加载类型: {load_type}")

        if latent is None:
            raise ValueError("未能提取 AV 潜变量，请检查文件格式是否正确")

        if verbose:
            metadata_json = json.dumps(metadata, ensure_ascii=False, indent=2, default=str)
        else:
            slim_metadata = {}
            for k in ["mode", "seed", "frame_count", "chain_id", "total_segments", "timestamp", "project", "version"]:
                if k in metadata:
                    slim_metadata[k] = metadata[k]
            metadata_json = json.dumps(slim_metadata, ensure_ascii=False, indent=2, default=str)

        report_lines.append("")
        report_lines.append("【使用提示】")
        report_lines.append("  - SingleSampler: 将加载的 AV Latent 连接到 prev_sampled_latent")
        report_lines.append("  - SequenceSampler: 加载的 latent 可直接用于 VAE Decode")
        report_lines.append(f"  - Latent 类型: {type(latent).__name__}")

        report = "\n".join(report_lines)

        return io.NodeOutput(latent, metadata_json, report)