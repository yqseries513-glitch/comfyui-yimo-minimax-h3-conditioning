from __future__ import annotations

import json
import os

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "yimo_config.json")
_DEFAULT_LANG = "zh"

_CATALOGUE = {
    "node_display_name": {"zh": "Yimo H3 音画条件", "en": "Yimo H3 Audio Conditioning"},
    "node_description": {
        "zh": "H3 音画条件节点。支持标签规范化、端口映射表、参考保留等级、参考曲线预设、"
              "文本信号增强、后处理分流。",
        "en": "H3 AV conditioning node. Supports tag normalization, port-to-tag mapping, "
              "reference retention levels, reference curve presets, text signal boost, "
              "and post-process split.",
    },
    "category": {"zh": "Yimo/MiniMax H3/音频", "en": "Yimo/MiniMax H3/Audio"},

    "output_positive": {"zh": "正向条件", "en": "Positive"},
    "output_negative": {"zh": "负向条件", "en": "Negative"},
    "output_av_latent": {"zh": "AV 潜变量", "en": "AV Latent"},
    "output_mux_audio": {"zh": "混音音频", "en": "Mux Audio"},
    "output_conditioned_prompt": {"zh": "条件化提示词", "en": "Conditioned Prompt"},
    "output_report": {"zh": "报告", "en": "Report"},
    "output_segment": {"zh": "段数据包", "en": "Segment Bundle"},

    "seqsampler_display_name": {"zh": "Yimo H3 AV双时钟序列采样器", "en": "Yimo H3 AV Dual-Clock Sequence Sampler"},
    "seqsampler_description": {
        "zh": "MiniMax H3 音视频序列采样器（含双时钟）。sampler_name=dual_clock_euler → 双时钟（video/audio 独立 sigma 轨迹）；其它采样器 → 单时钟（ComfyUI 原生 AV 调度）。段间逻辑（上下文注入 / 尾帧继承 / 身份继承 / 面部修复 / 颜色匹配 / 断点续采 / 参考曲线 / images 模式 / 自动导出）全部保留。",
        "en": "MiniMax H3 AV sequence sampler (dual-clock capable). sampler_name=dual_clock_euler → dual-clock; other samplers → single-clock. Segment-level logic fully preserved.",
    },
    "seqsampler_output_latent": {"zh": "长视频 AV 潜变量", "en": "Long Video AV Latent"},
    "seqsampler_output_report": {"zh": "采样报告", "en": "Sampling Report"},

    "singlesampler_display_name": {"zh": "Yimo H3 AV双时钟单段采样器", "en": "Yimo H3 AV Dual-Clock Single Sampler"},
    "singlesampler_description": {
        "zh": "MiniMax H3 音视频双时钟单段采样器。sampler_name=dual_clock_euler → 双时钟（video/audio 独立 sigma 轨迹）；其它采样器 → 单时钟（ComfyUI 原生 AV 调度）。段间逻辑（上下文注入 / 尾帧继承 / 身份继承 / 面部修复 / 颜色匹配 / 断点续传 / 参考曲线）全部保留。",
        "en": "MiniMax H3 AV dual-clock single-segment sampler. Segment-level logic fully preserved.",
    },
    "singlesampler_output_latent": {"zh": "采样后 AV 潜变量", "en": "Sampled AV Latent"},
    "singlesampler_output_report": {"zh": "采样报告", "en": "Sampling Report"},

    "loadlatent_display_name": {"zh": "Yimo H3 加载保存的潜变量", "en": "Yimo H3 Load Saved Latent"},
    "loadlatent_description": {
        "zh": "加载之前保存的 AV Latent 文件，用于 SingleSampler 的断点续跑。",
        "en": "Load previously saved AV Latent files for SingleSampler resume."
    },
    "loadlatent_output_latent": {"zh": "AV 潜变量", "en": "AV Latent"},
    "loadlatent_output_metadata": {"zh": "元数据（JSON）", "en": "Metadata (JSON)"},
    "loadlatent_output_report": {"zh": "加载报告", "en": "Load Report"},
    "file_path_tooltip": {
        "zh": "完整的文件路径，如: output/yimo_h3/single_sampler/project/segment_0.pt",
        "en": "Full file path, e.g.: output/yimo_h3/single_sampler/project/segment_0.pt"
    },
    "load_type_tooltip": {
        "zh": "选择加载哪种类型的文件:\n- single_sampler: 加载 SingleSampler 保存的完整 AV Latent\n- sequence_sampler_export: 加载 SequenceSampler 导出的最终结果\n- sequence_sampler_cache: 加载 SequenceSampler 的缓存段",
        "en": "Select file type to load:\n- single_sampler: SingleSampler saved AV Latent\n- sequence_sampler_export: SequenceSampler exported final result\n- sequence_sampler_cache: SequenceSampler cache segment"
    },

    "workflow_mode_input": {"zh": "工作流模式", "en": "Workflow Mode"},
    "workflow_mode_tooltip": {
        "zh": "选择生成模式：text=纯文本，first_frame=首帧图，last_frame=尾帧图，first_last_frame=首尾帧，references=纯参考，hybrid=混合。",
        "en": "Select generation mode: text, first_frame, last_frame, first_last_frame, references, or hybrid.",
    },
    "audio_policy_input": {"zh": "音频策略", "en": "Audio Policy"},
    "audio_policy_tooltip": {
        "zh": "keep_source=锁定原音频，remix_source=重塑音频，reference_only=语义参考，generate_new=完全生成。",
        "en": "keep_source=lock original, remix_source=remix original, reference_only=semantic reference, generate_new=generate new.",
    },
    "show_all_params_tooltip": {
        "zh": "开启后显示所有参数，关闭时根据工作流模式自动精简界面。仅影响 UI 显隐，不影响生成结果。",
        "en": "When enabled, show all parameters. When disabled, auto-simplify UI based on workflow mode. UI-only.",
    },
    "keyframe_positions_tooltip": {
        "zh": "逗号分隔的插帧位置（帧数，0-based），如：40,80,105。留空则自动均匀分布。节点会自动对齐到 17n+5 网格。",
        "en": "Comma-separated frame positions (0-based), e.g. 40,80,105. Leave empty for auto-distribution. Aligned to 17n+5 grid.",
    },
    "overlap_list_tooltip": {
        "zh": "逗号分隔的逐段 overlap 帧数，如：22,40,30。第 i 个值控制 segment_i 与 segment_{i+1} 之间的 overlap。留空则统一使用 overlap_frames。",
        "en": "Comma-separated per-pair overlap frames. The i-th value controls overlap between segment_i and segment_{i+1}. Leave empty to use uniform overlap_frames.",
    },
    "video_vae_tooltip": {
        "zh": "用于解码前段采样结果的尾帧像素图，实现自动继承为下一段的 first_frame。不连接则关闭尾帧继承功能。",
        "en": "Used to decode tail frame pixels from previous segment for automatic inheritance as next segment's first_frame.",
    },
    "auto_inherit_tail_tooltip": {
        "zh": "自动将前段尾帧作为本段 first_frame（当本段含首帧约束且前段有可用尾帧时）。",
        "en": "Automatically use previous segment's tail frame as this segment's first_frame.",
    },
    "inherit_identity_tooltip": {
        "zh": "跨段继承身份参考图（identity_refs），保持长视频中人物外观一致性。",
        "en": "Cross-segment inheritance of identity reference images to maintain character consistency.",
    },
    "audio_continuity_tooltip": {
        "zh": "context_inject=注入前段音频 tail latent（音频连贯）；break=不注入，各段音频独立。\n注意：音频注入依赖 overlap_frames>0。",
        "en": "context_inject=inject previous segment audio tail latent; break=no injection. Note: audio injection requires overlap_frames>0.",
    },
    "err_audio_denoise_range": {"zh": "audio_denoise_strength 必须在 0 到 1 之间", "en": "audio_denoise_strength must be between 0 and 1"},
    "err_ref_limits": {"zh": "MiniMax H3 参考上限为 9 张图片、3 个视频和 3 个独立音频", "en": "MiniMax H3 reference limits are 9 pictures, 3 videos, and 3 standalone audios"},
    "err_ref_video_too_short": {"zh": "参考视频太短，17n+5 对齐后不足 5 帧", "en": "Reference video is too short after 17n+5 alignment"},
    "err_unknown_workflow_mode": {"zh": "未知的工作流模式：{mode}", "en": "Unknown workflow mode: {mode}"},
    "err_unknown_audio_policy": {"zh": "未知的音频策略：{policy}", "en": "Unknown audio policy: {policy}"},
    "err_mode_needs_first": {"zh": "模式 {mode} 需要 first_frame 输入", "en": "Mode {mode} requires first_frame input"},
    "err_mode_needs_last": {"zh": "模式 {mode} 需要 last_frame 输入", "en": "Mode {mode} requires last_frame input"},
    "err_mode_needs_refs": {"zh": "模式 {mode} 需要至少一个参考媒体输入", "en": "Mode {mode} requires at least one reference media input"},
    "err_hybrid_requires_frame": {"zh": "混合模式需要 first_frame 和/或 last_frame", "en": "HYBRID requires first_frame and/or last_frame"},
    "err_keyframe_limits": {"zh": "中间关键帧数量超过当前帧数允许的上限", "en": "Keyframe count exceeds the limit for current frame count"},
    "err_invalid_keyframe_position": {"zh": "关键帧位置 [{part}] 无效，只允许整数或留空", "en": "Keyframe position [{part}] is invalid; only integers or empty string allowed"},
    "err_keyframe_count_mismatch": {"zh": "关键帧位置数({n_positions})与关键帧图片数({n_images})不匹配", "en": "Keyframe position count({n_positions}) does not match image count({n_images})"},
    "err_no_segments": {"zh": "未提供任何段数据包，请连接 YimoH3Conditioning 的 segment 输出", "en": "No segment bundles provided; connect YimoH3Conditioning.segment outputs."},
    "err_no_valid_segments": {"zh": "没有有效的段数据包", "en": "No valid segment bundles found."},
    "err_invalid_segment": {"zh": "段_{ordinal} 不是有效的段数据包", "en": "Segment_{ordinal} is not a valid segment bundle."},
    "err_not_segment_bundle": {"zh": "段_{ordinal} 缺少 _yimo_segment 标记，请确保来自 YimoH3Conditioning.segment", "en": "Segment_{ordinal} missing _yimo_segment flag; ensure it comes from YimoH3Conditioning.segment."},
    "err_segment_incomplete": {"zh": "段_{index} 缺少正向条件或 AV 潜变量", "en": "Segment_{index} is missing positive conditioning or AV latent."},
    "err_segment_spatial_mismatch": {"zh": "段_{index} 的画布分辨率与首段不一致", "en": "Segment_{index} canvas resolution mismatch with first segment."},
    "err_segment_invalid": {"zh": "段_{index} 无效: {error}", "en": "Segment_{index} invalid: {error}"},
    "err_source_audio_encode": {"zh": "源音频编码失败 ({error})；视为未连接", "en": "source_audio encoding failed ({error}); treated as unconnected."},
    "err_override_audio_validation": {"zh": "override_audio 验证失败 ({error})；视为未连接", "en": "override_audio validation failed ({error}); treated as unconnected."},
    "err_overlap_list_parse": {"zh": "overlap_list 解析失败 ({error})，应为逗号分隔的整数", "en": "overlap_list parse failed ({error}); expected comma-separated integers"},

    "warn_invalid_identity_index": {"zh": "身份图序号 [{part}] 无法解析，已忽略", "en": "Identity image index [{part}] is invalid and ignored."},
    "warn_invalid_keyframe_position": {"zh": "关键帧位置 [{part}] 无法解析，已忽略", "en": "Keyframe position [{part}] is invalid and ignored."},
    "warn_keyframe_conflict": {"zh": "关键帧_{ordinal} 位置 {pos} 与现有关键帧冲突，已跳过", "en": "Keyframe_{ordinal} at position {pos} conflicts with existing keyframe; skipped."},
    "warn_audio_offset_hard_trunc": {"zh": "audio_offset={offset}: 使用硬截断+零填充，偏移边界可能出现静音间隙", "en": "audio_offset={offset}: hard truncation applied; silence gaps may appear at offset boundaries."},
    "warn_style_audio_skipped": {"zh": "style_audio_{ordinal} 已连接但 audio_policy=generate_new；为避免直接复制，跳过音频参考。如需使用风格音频，请将 audio_policy 设为 reference_only 或 remix_source。", "en": "style_audio_{ordinal} connected but audio_policy=generate_new; skipped to avoid direct copy."},
    "warn_identity_not_connected": {"zh": "identity_image_indices: ref_image_{ordinal} 未连接，已忽略", "en": "identity_image_indices: ref_image_{ordinal} is not connected; ignored."},
    "warn_tail_inherit_failed": {"zh": "段_{index} 尾帧继承失败: {error}", "en": "Segment {index} tail frame inheritance failed: {error}"},
    "warn_identity_inherit_failed": {"zh": "段_{index} 身份参考图继承失败: {error}", "en": "Segment {index} identity ref inheritance failed: {error}"},
    "warn_size_aligned": {
        "zh": "画布尺寸已从 {orig_w}x{orig_h} 自动对齐为 {new_w}x{new_h}（需为 {multiple} 的倍数）",
        "en": "Canvas size auto-aligned from {orig_w}x{orig_h} to {new_w}x{new_h} (must be multiple of {multiple})",
    },
    "warn_refs_cleared_by_mode": {
        "zh": "模式 {mode} 不支持参考媒体，已自动忽略 ref_images/ref_videos/style_audios/ref_video_audios",
        "en": "Mode {mode} does not support reference media; refs auto-cleared.",
    },
    "warn_keyframes_cleared_by_mode": {
        "zh": "模式 {mode} 不支持关键帧约束，已自动忽略 first_frame/last_frame/keyframes",
        "en": "Mode {mode} does not support keyframe constraints; keyframes auto-cleared.",
    },
    "warn_source_audio_skipped": {
        "zh": "audio_policy={policy} 不需要源音频，跳过 source_audio 编码以节省资源",
        "en": "audio_policy={policy} does not need source audio; skipped encoding to save resources.",
    },
    "warn_refs_ignored": {
        "zh": "模式 {mode} 不支持参考媒体，已自动忽略 ref_images/ref_videos/style_audios/ref_video_audios",
        "en": "Mode {mode} does not support reference media; refs auto-cleared.",
    },
    "warn_audio_policy_downgrade": {
        "zh": "音频策略 {policy} 需要 source_audio 但未连接，自动降级为 generate_new",
        "en": "Audio policy {policy} requires source_audio but not connected; downgrading to generate_new.",
    },
    "identity_image_indices": {"zh": "身份图序号", "en": "Identity Image Indices"},
    "identity_image_indices_tooltip": {
        "zh": "用逗号分隔指定哪些 ref_images 是人物身份图（从1开始，对应 ref_image_N 端口编号）。空=没有身份图（纯服装/物品替换）。",
        "en": "Comma-separated 1-based indices of ref_images that define person identity, matching ref_image_N port ordinals. Empty = no identity refs.",
    },

    "reference_retention_input": {"zh": "参考保留等级", "en": "Reference Retention"},
    "reference_retention_tooltip": {
        "zh": "语义化参考强度控制：\n"
              "fully_preserved=完全保留参考素材的所有特征（推荐用于服装/身份复刻）\n"
              "partially_preserved=保留大部分特征，允许少量变化\n"
              "attribute_transfer=仅传递风格/属性，不保留身份\n"
              "weak_reference=弱参考，仅提供轻微引导\n"
              "no_reference=不使用参考",
        "en": "Semantic reference strength control.",
    },
    "reference_retention_scale_input": {"zh": "保留强度微调", "en": "Retention Scale"},
    "reference_retention_scale_tooltip": {
        "zh": "在 reference_retention 基础上的连续微调系数（0~1）。\n"
              "最终强度 = retention 档位值 × scale。\n"
              "默认 1.0 表示完全使用 retention 档位值；0.5 表示减半。",
        "en": "Continuous fine-tune factor on top of reference_retention (0~1). Final strength = retention_value × scale.",
    },

    "ref_curve_input": {"zh": "参考曲线预设", "en": "Reference Curve Preset"},
    "ref_curve_tooltip": {
        "zh": "参考强度在时间轴上的变化预设：\n"
              "off=全程恒定（默认，等同无曲线）\n"
              "fade_out=前强后弱\n"
              "fade_in=前弱后强\n"
              "peak_middle=中段强两端弱\n"
              "peak_ends=两端强中段弱\n"
              "custom=自定义方向与形状（展开下方两个 advanced 参数）\n"
              "注意：需在采样器端 enable_ref_curve=True 才会被消费。",
        "en": "Reference strength timeline preset.",
    },
    "ref_curve_direction_input": {"zh": "曲线方向（自定义）", "en": "Curve Direction (Custom)"},
    "ref_curve_direction_tooltip": {
        "zh": "参考强度在视频时间轴上的变化方向。",
        "en": "Reference strength variation direction over video timeline.",
    },
    "ref_curve_shape_input": {"zh": "曲线形状（自定义）", "en": "Curve Shape (Custom)"},
    "ref_curve_shape_tooltip": {
        "zh": "曲线形状：linear=线性，ease=平滑S曲线，sigmoid=S型，"
              "exponential=指数，quadratic=二次，cubic=三次",
        "en": "Curve shape: linear, ease, sigmoid, exponential, quadratic, cubic",
    },

    "text_boost_strength_input": {"zh": "文本信号增强", "en": "Text Signal Boost"},
    "text_boost_strength_tooltip": {
        "zh": "放大文本条件信号，提高提示词对 DiT 的影响力。\n"
              "1.0=无操作，2.0=推荐起点，最大8.0。",
        "en": "Amplify text conditioning signal. 1.0=no-op, 2.0=recommended start, max 8.0.",
    },
    "text_boost_mode_tooltip": {
        "zh": "增强模式：deviation=仅放大偏离均值的部分（推荐，保持语义中心稳定）；"
              "naive=整体缩放（简单但可能过饱和）",
        "en": "Boost mode: deviation=amplify only deviations from mean (recommended); naive=scale entire tensor.",
    },
    "text_boost_renorm_tooltip": {
        "zh": "是否重新归一化以保持输入尺度（推荐开启）",
        "en": "Re-normalize to preserve input scale (recommended)",
    },

    "color_match_mode_tooltip": {
        "zh": "颜色匹配档位，用于多段拼接时减轻接缝处的色调/亮度跳变。\n"
              "off=不处理（默认）\n"
              "low=latent 层逐通道均值匹配\n"
              "medium=latent 层均值+标准差匹配（Reinhard）\n"
              "high=像素层直方图匹配（需 video_vae）\n"
              "max=像素层 MKL 线性色彩迁移（需 video_vae）",
        "en": "Color matching mode for reducing color/brightness jumps at seams.",
    },
    "color_match_strength_tooltip": {
        "zh": "颜色匹配强度。0=不匹配（等同 off），1=完全匹配。推荐 0.5~0.8。",
        "en": "Color match strength. 0=no match, 1=full match. Recommended 0.5~0.8.",
    },
    "color_match_reference_frames_tooltip": {
        "zh": "参考帧数。从参考段尾部取 N 个 latent 时间步做颜色统计。\n"
              "N 越大越稳定，但越可能引入内容差异。默认 1。",
        "en": "Number of reference frames. Default 1.",
    },

    "post_process_split_display_name": {"zh": "Yimo H3 后处理分流器", "en": "Yimo H3 Post-Process Split"},
    "post_process_split_description": {
        "zh": "把 AV Latent 拆分为视频 Latent / 音频 Latent / 图像帧序列三种输出。",
        "en": "Split AV Latent into video latent / audio latent / image frames.",
    },
    "post_process_split_av_latent_tooltip": {"zh": "来自采样器输出的 AV Latent。", "en": "AV Latent from sampler output."},
    "post_process_split_video_vae_tooltip": {"zh": "H3 Video VAE。连接后才能输出「图像帧序列」；不连接时该端口为空。", "en": "H3 Video VAE. Required to produce image frames."},
    "post_process_split_output_video_latent": {"zh": "视频 Latent", "en": "Video Latent"},
    "post_process_split_output_audio_latent": {"zh": "音频 Latent", "en": "Audio Latent"},
    "post_process_split_output_image_frames": {"zh": "图像帧序列", "en": "Image Frames"},
    "post_process_split_output_metadata": {"zh": "元数据 JSON", "en": "Metadata JSON"},

    "post_process_merge_display_name": {"zh": "Yimo H3 后处理合流器", "en": "Yimo H3 Post-Process Merge"},
    "post_process_merge_description": {
        "zh": "把超分后的视频（Latent 或 IMAGE，二选一）与原始音频 Latent 重新组装成 AV Latent。",
        "en": "Merge the upscaled video (latent or IMAGE, pick one) with the original audio latent back into AV Latent.",
    },
    "post_process_merge_video_latent_tooltip": {"zh": "潜空间超分后的视频 Latent。与 video_frames 二选一。", "en": "Video latent after latent-space upscale. Pick one of video_latent / video_frames."},
    "post_process_merge_video_frames_tooltip": {"zh": "像素级超分后的图像帧序列 [T,H,W,C]。与 video_latent 二选一，需要 video_vae 重新编码。", "en": "Image frames after pixel-space upscale [T,H,W,C]. Pick one of video_latent / video_frames; requires video_vae for re-encoding."},
    "post_process_merge_audio_latent_tooltip": {"zh": "原始音频 Latent（来自分流器的「音频 Latent」输出）。", "en": "Original audio latent (from Split's 'audio latent' output)."},
    "post_process_merge_video_vae_tooltip": {"zh": "H3 Video VAE。仅在输入为 video_frames 时必需。", "en": "H3 Video VAE. Only required when input is video_frames."},
    "post_process_merge_output_av_latent": {"zh": "合并后的 AV Latent", "en": "Merged AV Latent"},
    "post_process_merge_output_report": {"zh": "合并报告", "en": "Merge Report"},

    "err_post_process_no_av_latent": {"zh": "av_latent 未连接。请连接到采样器输出的 AV Latent。", "en": "av_latent not connected. Connect an AV latent from a sampler."},
    "err_post_process_av_parse": {"zh": "AV Latent 解析失败: {error}", "en": "Failed to parse AV Latent: {error}"},
    "err_post_process_both_video_inputs": {"zh": "video_latent 和 video_frames 同时连接。请只连其中一个。", "en": "Both video_latent and video_frames are connected. Only connect one."},
    "err_post_process_no_video_input": {"zh": "既没有连接 video_latent 也没有连接 video_frames。请至少连接一个视频输入。", "en": "Neither video_latent nor video_frames is connected. Connect at least one video input."},
    "err_post_process_no_audio_latent": {"zh": "audio_latent 未连接。请从分流器的「音频 Latent」输出获取。", "en": "audio_latent not connected. Use Split's 'audio latent' output."},
    "err_post_process_frames_need_vae": {"zh": "video_frames 已连接但 video_vae 未连接。像素级超分需要 video_vae 重新编码。", "en": "video_frames is connected but video_vae is not. Pixel-space upscale requires video_vae for re-encoding."},
    "err_post_process_frames_size_multiple": {"zh": "video_frames 尺寸 {w}x{h} 必须是 {multiple} 的倍数。请在超分后裁剪/填充到合法尺寸。", "en": "video_frames size {w}x{h} must be a multiple of {multiple}. Crop/pad after upscale."},
    "err_post_process_frames_size_even": {"zh": "video_frames 的宽高必须是偶数（DiT 2x2 patch），got {w}x{h}。", "en": "video_frames width/height must be even (DiT 2x2 patch), got {w}x{h}."},
    "err_post_process_encode": {"zh": "video_frames 编码失败: {error}", "en": "video_frames encoding failed: {error}"},
    "err_post_process_encode_shape": {"zh": "video_vae.encode 返回异常格式（非 5D tensor）。", "en": "video_vae.encode returned unexpected format (non-5D tensor)."},
    "err_post_process_nested": {"zh": "NestedTensor 组装失败: {error}", "en": "Failed to assemble NestedTensor: {error}"},
}


def get_language() -> str:
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            lang = cfg.get("language", _DEFAULT_LANG)
            if lang in ("zh", "en"):
                return lang
    except Exception:
        pass
    return _DEFAULT_LANG


def t(key: str, **kwargs) -> str:
    lang = get_language()
    text = _CATALOGUE.get(key, {}).get(lang)
    if text is None:
        text = _CATALOGUE.get(key, {}).get(_DEFAULT_LANG, key)
    return text.format(**kwargs) if kwargs else text