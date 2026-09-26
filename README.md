# Yimo MiniMax H3 Conditioning v3.0.0 完整使用手册

面向 ComfyUI 的 MiniMax H3 音画联合生成增强节点套件。

---

## 一、插件简介

**Yimo MiniMax H3 Conditioning** 是面向 ComfyUI 的 MiniMax H3 音画联合生成模型的自定义节点套件，专注于：

- **长视频生成**：通过多段采样 + 段间上下文注入，突破单次采样时长限制
- **多段一致性**：跨段继承关键帧、参考素材、身份图、尾帧
- **参考强度精确控制**：语义分级、时间曲线、文本信号增强
- **采样模式参数化**：支持 `standard` / `pdd` / `pdd_two_stage` 三种采样模式
- **面部修复**：采样后自动检测并修复人脸，支持身份匹配
- **后处理分流/合流**：AV Latent ↔ 视频 Latent / 音频 Latent / 图像帧序列
- **高分辨率重采样**：实验性节点，支持时间/空间分块、颜色匹配、质量检查
- **标签规范化**：多种媒体标签写法自动归一化

> **版本**：v3.0.0  
> **作者**：yimo  
> **许可证**：GPL-3.0-or-later  
> **兼容 ComfyUI**：最新版（含 `comfy_api.latest`）  
> **Python**：3.10+

---

## 二、环境依赖

### 2.1 必需依赖（ComfyUI 已提供）

| 依赖 | 说明 |
|------|------|
| `torch` | PyTorch，由 ComfyUI 运行时提供 |
| `torchvision` | 图像变换，要求 ≥ 0.15.0（`ref_video_preprocessing=blur_desaturate` 需要） |
| `torchaudio` | 音频重采样，延迟导入，仅在需要编码音频时加载 |
| `numpy` | 数值计算，要求 ≥ 1.24.0 |
| `comfy` / `comfy_api` | ComfyUI 内核 API |
| `node_helpers` | ComfyUI 内置条件操作辅助 |

### 2.2 必需依赖（需手动安装）

```bash
pip install opencv-python>=4.8.0
pip install numpy>=1.24.0
```

### 2.3 可选依赖（面部修复增强）

面部修复功能默认使用 OpenCV 降级方案，无需任何额外依赖。如需更高质量，可选装：

```bash
# 最佳效果：InsightFace 人脸检测 + 特征提取
pip install insightface onnxruntime-gpu
# 或（无 GPU）
pip install insightface onnxruntime

# GFPGAN 高质量人脸修复（注意使用 --no-deps 避免版本冲突）
pip install gfpgan --no-deps facexlib
```

| 后端 | 依赖 | 质量 | 说明 |
|------|------|------|------|
| `opencv` | opencv-python | 中 | 默认降级方案，无需额外安装 |
| `insightface` | insightface + onnxruntime | 中高 | 人脸检测 + ArcFace 身份匹配 |
| `gfpgan` | gfpgan + facexlib | 高 | 最佳修复质量 |
| `auto` | 自动选择 | 最佳 | 按 GFPGAN > InsightFace > OpenCV 顺序选择 |

### 2.4 模型文件放置位置

使用 ComfyUI 标准模型目录（通过 `folder_paths` 自动定位）：

```
ComfyUI/models/
├── facerestore_models/
│   └── GFPGANv1.4.pth                          # GFPGAN 模型（可选）
├── insightface/
│   └── models/buffalo_l/                       # InsightFace 模型（可选，首次运行自动下载）
└── opencv/
    ├── deploy.prototxt                          # OpenCV DNN 配置（可选）
    └── res10_300x300_ssd_iter_140000.caffemodel # OpenCV DNN 权重（可选）
```

**OpenCV DNN 模型手动下载**：

```bash
mkdir -p ComfyUI/models/opencv
cd ComfyUI/models/opencv
wget https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/deploy.prototxt
wget https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/res10_300x300_ssd_iter_140000.caffemodel
```

### 2.5 配置文件

插件根目录下的 `yimo_config.json`：

```json
{
  "language": "zh",
  "reference_defaults": {
    "retention": "fully_preserved",
    "curve_direction": "constant",
    "curve_shape": "linear"
  },
  "text_boost_defaults": {
    "strength": 1.0,
    "mode": "deviation",
    "renorm": true
  }
}
```

| 字段 | 可选值 | 说明 |
|------|--------|------|
| `language` | `"zh"` / `"en"` | 节点 UI 语言 |

---

## 三、节点清单

| 节点 ID | 显示名 | 功能 |
|---|---|---|
| `YimoH3Conditioning` | Yimo H3 音画条件 | 构建正向/负向条件、AV Latent、混音音频、段数据包 |
| `YimoH3SingleSampler` | Yimo H3 单段采样器 | 单段采样 + 上下文注入 + 尾帧/身份继承 |
| `YimoH3SequenceSampler` | Yimo H3 序列采样器 | 多段循环采样 + 拼接成长视频 + 断点续采 |
| `YimoH3ConcatenateSegments` | Yimo H3 段拼接器 | 将多段 AV Latent 拼接成长视频 |
| `YimoH3FaceRestore` | Yimo H3 面部修复（采样后） | 独立的人脸检测与修复 |
| `YimoH3LoadSavedLatent` | Yimo H3 加载保存的潜变量 | 加载 .pt 文件，用于断点续跑 |
| `YimoH3PromptComposer` | Yimo H3 标准提示词格式化 | 组合官方三段式 / 六段式提示词 |
| `YimoH3PostProcessSplit` | Yimo H3 后处理分流器 | AV Latent → 视频 / 音频 / 图像帧 |
| `YimoH3PostProcessMerge` | Yimo H3 后处理合流器 | 超分后视频 + 音频 → AV Latent |
| `YimoH3HighResResampler` | Yimo H3 高分辨率重采样器（实验性） | 双线性上采样 + 重采样去噪 |
| `YimoH3PDDTwoPass` | Yimo H3 PDD 双采封装 | LOW → LBH 放大 → HIGH |

---

## 四、节点参数详解

### 4.1 YimoH3Conditioning — 音画条件构建

**功能**：构建 MiniMax H3 模型所需的正向/负向条件、AV Latent 模板、混音音频，并输出**段数据包**供采样器使用。

#### 4.1.1 输入端口（连接类）

| 端口 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `clip` | CLIP | ✅ | MiniMax H3 原生 Qwen3-VL CLIP |
| `video_vae` | VAE | ✅ | H3 视频 VAE |
| `audio_vae` | VAE | ✅ | H3 音频 VAE |
| `prev_segment_bundle` | CONDITIONING | ❌ | 上一段的「段数据包」，用于继承素材（references / hybrid 模式） |
| `first_frame` | IMAGE | ❌ | 首帧图 |
| `last_frame` | IMAGE | ❌ | 尾帧图 |
| `keyframes` | IMAGE | ❌ | 中间关键帧（支持批次或列表，最多 9 张） |
| `ref_images` | IMAGE | ❌ | 参考图片（支持批次或列表，最多 9 张） |
| `ref_videos` | AUTOGROW | ❌ | 参考视频（最多 3 个，每个 48~360 帧） |
| `ref_video_audios` | AUTOGROW | ❌ | 与 ref_video 同编号的音频（最多 3 个） |
| `style_audios` | AUTOGROW | ❌ | 独立风格音频（最多 3 个） |
| `source_audio` | AUDIO | ❌ | 源音频（用于 keep_source / remix_source / reference_only） |
| `override_audio` | AUDIO | ❌ | 覆盖输出的混音音频 |

#### 4.1.2 工作流参数

| 参数 | 类型 | 默认 | 可选值 | 说明 |
|------|------|------|--------|------|
| `workflow_mode` | COMBO | `text` | `text` / `first_frame` / `last_frame` / `first_last_frame` / `references` / `hybrid` | 生成模式 |
| `audio_policy` | COMBO | `generate_new` | `keep_source` / `remix_source` / `reference_only` / `generate_new` | 音频策略 |
| `show_all_params` | BOOLEAN | `false` | — | 显示所有参数（JS 控制界面精简） |
| `prompt` | STRING | — | — | 正向提示词 |
| `negative_prompt` | STRING | `""` | — | 负向提示词 |
| `width` | INT | `1344` | 32~16384 | 视频宽度（对齐到 32 倍数） |
| `height` | INT | `768` | 32~16384 | 视频高度（对齐到 32 倍数） |
| `auto_resolution` | BOOLEAN | `false` | — | 根据输入媒体建议分辨率（仅报告，不覆盖用户值） |
| `length` | INT | `124` | 5~3600 | 帧数（对齐到 17n+5） |

#### 4.1.3 提示词标签系统

**媒体标签（素材引用）**——以下写法全部自动归一化：

| 用户写法 | 归一化结果 |
|----------|-----------|
| `<Picture 1>` / `<picture 1>` / `<Image 1>` / `<image 1>` / `<IMG 1>` / `<Pic 1>` / `<Photo 1>` | `<Picture 1>` |
| `<Video 1>` / `<video 1>` / `<Vid 1>` | `<Video 1>` |
| `<Audio 1>` / `<audio 1>` / `<Sound 1>` | `<Audio 1>` |

**语义主体标签（推荐 `<Subject N>`）**：

`<Subject N>` 是 MiniMax H3 模型原生支持的语义标签，用于定义「由多张 `<Picture N>` 共同描述的同一主体」。

推荐写法：

```text
subject_definitions:
<Subject 1> is the woman whose facial identity, hairstyle, and body appearance come from <Picture 1>, <Picture 2>, and <Picture 3>.
<Subject 2> is the outfit whose design, color, and fabric details come from <Picture 4>, <Picture 5>, and <Picture 6>.
<Subject 3> is the shoes whose design, color, and style come from <Picture 7> and <Picture 8>.

summary:
The target video shows <Subject 1> wearing <Subject 2> and <Subject 3>, performing the specified action.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - preserve facial identity, hairstyle, and body appearance; none of the original clothing, shoes, background, or lighting from these pictures should appear.
<Subject 2> ([Shot 1] clothing): fully_preserved - preserve the outfit design, color, and fabric details; none of the original person, background, or lighting from these pictures should appear.
<Subject 3> ([Shot 1] footwear): fully_preserved - preserve the shoes' design, color, and style; none of the original person, background, or lighting from these pictures should appear.

detailed_description:
[Shot 1] A medium shot shows <Subject 1> wearing the outfit from <Subject 2> and the shoes from <Subject 3>, performing [具体动作].
```

**关于非标准主体标签**：

中文语义标签（如 `<人物1>`、`<服装1>`、`<鞋子1>`）也能用，模型（Qwen3-VL）能理解其语义。但插件不做代码级解析，只检测并在报告中给出 info 提示。建议优先使用 `<Subject N>`。

**`retention_analysis` 块的重要性**：

模型对参考图的应用规则在 `retention_analysis` 中定义。如果提示词里没有这个块，或没为每组参考图指定明确的"保留范围"，模型会默认把每张图的所有特征都往输出里塞——这是多参考图混杂的根本原因。

正确的 `retention_analysis` 应为每组 `<Subject>` 明确：
- **保留什么**（facial identity / design / color 等）
- **不保留什么**（用 "none of the original X should appear" 明确排除边界）

#### 4.1.4 参考强度语义分级

| 参数 | 类型 | 默认 | 可选值 |
|------|------|------|--------|
| `reference_retention` | COMBO | `fully_preserved` | `fully_preserved` / `partially_preserved` / `attribute_transfer` / `weak_reference` / `no_reference` |

**数值映射**：

| 等级 | 数值 | 适用场景 |
|------|------|----------|
| `fully_preserved` | 1.0 | 完全保留参考特征（服装/身份复刻） |
| `partially_preserved` | 0.7 | 保留大部分，允许少量变化 |
| `attribute_transfer` | 0.4 | 仅传递风格/属性，不保留身份 |
| `weak_reference` | 0.15 | 弱参考，仅轻微引导 |
| `no_reference` | 0.0 | 不使用参考 |

#### 4.1.5 参考强度时间曲线

| 参数 | 类型 | 默认 | 可选值 |
|------|------|------|--------|
| `ref_curve_direction` | COMBO | `constant` | `constant` / `concept_at_start` / `concept_at_end` / `concept_at_middle` / `concept_at_ends` |
| `ref_curve_shape` | COMBO | `linear` | `linear` / `ease` / `sigmoid` / `exponential` / `quadratic` / `cubic` |

**方向含义**：

| 方向 | 效果 |
|------|------|
| `constant` | 全程恒定强度（推荐用于单一服装/身份） |
| `concept_at_start` | 前段强，后段弱（1-t 衰减） |
| `concept_at_end` | 前段弱，后段强（t 增长） |
| `concept_at_middle` | 中间强，两端弱（三角形包络） |
| `concept_at_ends` | 两端强，中间弱（V 形包络） |

**形状含义**：

| 形状 | 数学形式 |
|------|----------|
| `linear` | `x` |
| `ease` | `3x² - 2x³`（S 曲线） |
| `sigmoid` | `1/(1+e^(-10(x-0.5)))` |
| `exponential` | `(e^x - 1)/(e - 1)` |
| `quadratic` | `x²` |
| `cubic` | `x³` |

#### 4.1.6 文本信号增强

| 参数 | 类型 | 默认 | 范围 | 说明 |
|------|------|------|------|------|
| `text_boost_strength` | FLOAT | `1.0` | 0.25~8.0 | 文本增强倍数，1.0=无操作，2.0=推荐起点 |
| `text_boost_mode` | COMBO | `deviation` | `deviation` / `naive` | 增强模式 |
| `text_boost_renorm` | BOOLEAN | `true` | — | 是否重新归一化保持输入尺度 |

**背景**：MiniMax H3 的 Qwen3-VL 编码器区分度较低（语义不同的提示词嵌入仅差 3-5% RMS），适度增强可提高提示词遵循度。

#### 4.1.7 参考素材高级参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `ref_image_size` | COMBO | `match` | `match`=匹配画布大小 / `max`=最大短边 2048 |
| `reference_video_policy` | COMBO | `official_2_to_15s` | `official_2_to_15s`=官方建议 48~360 帧 / `model_minimum`=宽松模式 |
| `ref_video_start_frame` | INT | `0` | 参考视频起始帧 |
| `reference_strength` | FLOAT | `1.0` | 原始参考强度（与 retention 共同决定最终值） |
| `identity_image_indices` | STRING | `""` | 逗号分隔的身份图序号（如 `1,3`），对应 `ref_image_N` 端口 |
| `clip_video_grayout` | BOOLEAN | `false` | CLIP 端视频涂灰（实验性） |
| `ref_video_preprocessing` | COMBO | `none` | `none` / `blur_desaturate`（GPU 强模糊+降饱和） |

#### 4.1.8 音频与提示词参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `strict_prompt_tags` | BOOLEAN | `false` | 严格校验媒体标签数量是否匹配 |
| `audio_offset_frames` | INT | `0` | 音频偏移帧数 |
| `audio_denoise_strength` | FLOAT | `0.35` | 音频重绘强度（remix_source 时生效） |

#### 4.1.9 输出端口

| 序号 | 端口 | 类型 | 说明 |
|------|------|------|------|
| 1 | `Segment Bundle` | CONDITIONING | **段数据包（连接采样器）** |
| 2 | `Positive` | CONDITIONING | 正向条件 |
| 3 | `Negative` | CONDITIONING | 负向条件 |
| 4 | `AV Latent` | LATENT | 音画潜变量模板 |
| 5 | `Mux Audio` | AUDIO | 混音音频 |
| 6 | `Conditioned Prompt` | STRING | 处理后的提示词 |
| 7 | `Report` | STRING | 详细构建报告（含端口映射表与 media_map JSON） |

⚠️ **重要**：连接采样器时，请使用**第 1 个输出「段数据包」**，切勿误接 Positive/Negative。

#### 4.1.10 端口映射表（报告）

`Report` 输出包含一段 `port-to-tag mapping`，明确列出**原始端口名 → 有效 `<Picture N>` 编号**：

```
--- port-to-tag mapping ---
  ref_image_1 -> <Picture 1> (valid (identity))
  ref_image_2 -> <Picture 2> (valid (identity))
  ref_image_3 -> <Picture 3> (valid (identity))
  ref_image_4 -> <Picture 4> (valid (style))
  ...
```

**关于端口忽略**：使用 ComfyUI 原生的节点右键忽略（Bypass / Mute）即可让某个素材不参与编号，不影响其他端口的排列。

#### 4.1.11 从上一段继承素材（`prev_segment_bundle`）

将上一段 `YimoH3Conditioning` 的「段数据包」连到本段的 `prev_segment_bundle`，本段会自动继承上一段的素材。

**触发条件（全部满足才继承）**：
1. 连接了 `prev_segment_bundle`
2. `workflow_mode` 为 `references` / `hybrid`
3. 本段未连接任何本地素材
4. 画布与上一段一致

**继承**：`real_ref_items` / `real_ref_blocks` / `identity_refs` / `style_refs` / labels  
**不继承**：`prompt` / `workflow_mode` / `audio_policy` / 曲线配置 / 任何参数

⚠️ 继承只对 DiT 端生效，CLIP 端看不到这些素材，因此**本段提示词不要写 `<Picture N>` 引用**，改用自然语言描述。

---

### 4.2 YimoH3SingleSampler — 单段采样器

**功能**：对单段条件执行采样，支持前段上下文注入、尾帧继承、面部修复、自动保存、采样模式切换。

#### 4.2.1 输入端口

| 端口 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `model` | MODEL | ✅ | MiniMax H3 DiT 模型 |
| `segment` | CONDITIONING | ✅ | 来自 `YimoH3Conditioning.Segment Bundle` |
| `video_vae` | VAE | ❌ | 用于尾帧继承与面部修复 |
| `audio_vae` | VAE | ❌ | 音频处理 |
| `prev_sampled_latent` | LATENT | ❌ | 前段完整采样结果（上下文注入 / 尾帧继承 / 身份继承） |
| `face_images` | IMAGE | ❌ | 面部参考图（批次/列表） |
| `external_sigmas` | SIGMAS | ❌ | PDD 模式的 sigma 轨迹 |

#### 4.2.2 采样模式

| 参数 | 类型 | 默认 | 可选值 |
|------|------|------|--------|
| `sampling_mode` | COMBO | `standard` | `standard` / `pdd` / `pdd_two_stage` |
| `pdd_split_step` | INT | `4` | 1~7，仅 `pdd_two_stage` 生效 |
| `pdd_scale_by` | FLOAT | `1.5` | 1.0~4.0，仅 `pdd_two_stage` 生效 |

- `standard`：使用 ComfyUI 标准调度器（兼容普通 LoRA、Turbo、Lightx2v 等）
- `pdd`：使用 PDD 训练网格的 sigmas（需配合 PDD LoRA），`sampler_name` 自动切换为 `euler`
- `pdd_two_stage`：LOW → 双线性空间放大 → HIGH；内部自处理放大

#### 4.2.3 段间过渡参数

| 参数 | 类型 | 默认 | 可选值 | 说明 |
|------|------|------|--------|------|
| `overlap_frames` | INT | `22` | 0~200 | 与前段重叠帧数 |
| `transition_mode` | COMBO | `context_inject` | `context_inject` / `hard_concat` | 视频连续性 |
| `audio_continuity` | COMBO | `context_inject` | `context_inject` / `break` | 音频连续性（依赖 overlap_frames>0） |
| `auto_inherit_tail` | BOOLEAN | `true` | — | 前段尾帧作为本段 first_frame |
| `inherit_identity` | BOOLEAN | `true` | — | 跨段继承身份参考图 |

**overlap_frames 的 17n+5 映射规则**：

| 输入 | effective |
|------|-----------|
| 0 | 0（无注入） |
| 1~5 | 5 |
| 6~22 | 22（推荐） |
| 23+ | 39 / 56 / ...（最近邻） |

⚠️ 换装、换风格场景请使用 `>= 22`；过短的 tail 会被 DiT 当作全局参考模板。

#### 4.2.4 参考曲线覆盖

| 参数 | 类型 | 默认 | 可选值 |
|------|------|------|--------|
| `enable_ref_curve` | BOOLEAN | `false` | — |
| `override_ref_curve` | BOOLEAN | `false` | — |
| `ref_curve_direction_override` | COMBO | `inherit` | `inherit` / `constant` / `concept_at_start` / `concept_at_end` / `concept_at_middle` / `concept_at_ends` |
| `ref_curve_shape_override` | COMBO | `inherit` | `inherit` / `linear` / `ease` / `sigmoid` / `exponential` / `quadratic` / `cubic` |

**单段内曲线行为**：MiniMax H3 的 DiT 注意力作用在整段 latent 上，无法真正逐帧变化。SingleSampler 在单段内使用段中点强度近似。希望单段全程强参考，推荐 `override_ref_curve=true` + `ref_curve_direction_override=constant`。

#### 4.2.5 采样参数

| 参数 | 类型 | 默认 | 范围 |
|------|------|------|------|
| `sampler_name` | COMBO | `euler` | ComfyUI 全部采样器 |
| `scheduler` | COMBO | `normal` | ComfyUI 全部调度器 |
| `steps` | INT | `20` | 1~100 |
| `cfg` | FLOAT | `7.0` | 0~100 |
| `denoise` | FLOAT | `1.0` | 0~1 |
| `seed` | INT | `0` | 0~2^64-1 |

#### 4.2.6 面部修复参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `face_restore` | BOOLEAN | `false` | 启用采样中面部修复 |
| `face_restore_backend` | COMBO | `auto` | `auto` / `gfpgan` / `insightface` / `opencv` |
| `face_restore_strength` | FLOAT | `0.7` | 修复强度（0.5~0.8 推荐） |
| `face_similarity_threshold` | FLOAT | `0.45` | ArcFace 余弦匹配阈值（同一人通常 > 0.5） |

#### 4.2.7 颜色匹配参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `color_match_mode` | COMBO | `off` | `off` / `low` / `medium` / `high` / `max` |
| `color_match_strength` | FLOAT | `0.7` | 0~1 |
| `color_match_reference_frames` | INT | `1` | 1~17 |

#### 4.2.8 自动保存参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `auto_save` | BOOLEAN | `false` | 是否自动保存采样结果 |
| `save_name` | STRING | `""` | 文件名（不含扩展名） |
| `project_name` | STRING | `""` | 项目子文件夹名 |

**保存路径**：`ComfyUI/output/yimo_h3/single_sampler/{project_name}/{save_name}.pt`

#### 4.2.9 输出端口

| 端口 | 类型 | 说明 |
|------|------|------|
| `采样后 AV 潜变量` | LATENT | 采样结果（含 `_yimo_prev_meta`，供下游继承） |
| `采样报告` | STRING | 详细采样日志 |

---

### 4.3 YimoH3SequenceSampler — 序列采样器

**功能**：接收多段条件，循环采样并自动拼接成长视频，支持段间过渡、断点续采、每段独立种子、面部修复、采样模式切换。

#### 4.3.1 输入端口

| 端口 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `model` | MODEL | ✅ | MiniMax H3 DiT 模型 |
| `segments` | AUTOGROW | ✅ | 1~8 个 segment 数据包 |
| `video_vae` | VAE | ❌ | 用于尾帧提取（自动继承） |
| `audio_vae` | VAE | ❌ | 用于音频连续性 |
| `face_images` | IMAGE | ❌ | 面部参考图 |
| `external_sigmas` | SIGMAS | ❌ | PDD 模式的 sigma 轨迹 |

#### 4.3.2 采样模式

同 `SingleSampler`：`sampling_mode` / `pdd_split_step` / `pdd_scale_by`。

#### 4.3.3 段间控制参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `overlap_frames` | INT | `22` | 段间重叠帧数（overlap_list 留空/全负时生效） |
| `overlap_list` | STRING | `""` | 逐对 overlap（逗号分隔，如 `22,40,30`），优先级更高 |
| `transition_mode` | COMBO | `context_inject` | `context_inject` / `hard_concat` |
| `audio_continuity` | COMBO | `context_inject` | `context_inject` / `break` |
| `auto_inherit_tail` | BOOLEAN | `true` | 自动继承前段尾帧 |
| `inherit_identity` | BOOLEAN | `true` | 跨段继承身份参考图 |

**overlap_list 负值处理**：
- 全为负数（如 `-1`）→ 视为未设置，回退到 `overlap_frames`
- 混用（如 `22,-1,30`）→ 负值单点回退到 `overlap_frames`，正值保留
- `0` 仍是有效的硬切标记

#### 4.3.4 断点续采参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `chain_id` | STRING | `default` | 缓存标识符 |
| `resample_segment` | INT | `-1` | 从第几段重新采样（-1=全部） |
| `segment_seeds` | STRING | `""` | 每段独立种子（逗号分隔，`none` 跳过） |
| `stop_after_segment` | INT | `-1` | 采样到第几段后提前停止（-1=全部） |

#### 4.3.5 参考曲线开关

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `enable_ref_curve` | BOOLEAN | `false` | 启用参考强度时间曲线 |
| `override_ref_curve` | BOOLEAN | `false` | 覆盖所有分段的曲线配置 |
| `ref_curve_direction_override` | COMBO | `inherit` | 统一覆盖曲线方向 |
| `ref_curve_shape_override` | COMBO | `inherit` | 统一覆盖曲线形状 |

#### 4.3.6 采样参数

同 `SingleSampler` 的 `sampler_name` / `scheduler` / `steps` / `cfg` / `denoise` / `seed`。

#### 4.3.7 面部修复参数

同 `SingleSampler` 的 `face_restore` / `face_restore_backend` / `face_restore_strength` / `face_similarity_threshold`。

#### 4.3.8 颜色匹配参数

同 `SingleSampler` 的 `color_match_mode` / `color_match_strength` / `color_match_reference_frames`。

#### 4.3.9 输出格式

| 参数 | 类型 | 默认 | 可选值 |
|------|------|------|--------|
| `output_format` | COMBO | `latent` | `latent` / `images` |
| `chunk_memory_threshold_mb` | INT | `4096` | 512~32768 |

- `latent`：输出 AV Latent（原始行为）
- `images`：按 overlap=0 边界分块独立解码，避免硬切段整帧闪烁（需 `video_vae`）
- `chunk_memory_threshold_mb`：images 模式下累积解码帧字节数阈值，超过后溢出到磁盘

#### 4.3.10 自动导出参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `auto_export_final` | BOOLEAN | `false` | 是否自动导出最终拼接结果 |
| `export_name` | STRING | `""` | 导出文件名（不含扩展名） |

**导出路径**：`ComfyUI/output/yimo_h3/sequence_sampler/exports/{export_name}.pt`

#### 4.3.11 输出端口

| 端口 | 类型 | 说明 |
|------|------|------|
| `长视频 AV 潜变量` | LATENT | 拼接后的完整长视频（output_format=images 时为 None） |
| `采样报告` | STRING | 详细日志 |
| `分段 0~7 AV 潜变量` | LATENT × 8 | 各分段独立输出（未使用的为 None） |
| `长视频图像序列 (images 模式)` | IMAGE | output_format=images 时的图像序列 |

---

### 4.4 YimoH3ConcatenateSegments — 段拼接器

**功能**：将多段 `SingleSampler` 输出的 AV Latent 按 overlap 拼接成长视频。

#### 4.4.1 输入参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `concat_mode` | COMBO | `chunked` | `chunked` / `independent` |
| `output_mode` | COMBO | `both` | `both` / `av_only` / `images_only` |
| `parts` | AUTOGROW | — | 1~8 个 AV Latent |
| `overlap_frames` | INT | `22` | 共用 overlap 帧数 |
| `overlap_list` | STRING | `""` | 逐对 overlap（逗号分隔） |
| `video_overlap_frames` | INT | `-1` | 视频单独使用的 overlap（-1=跟随共用） |
| `audio_overlap_frames` | INT | `-1` | 音频单独使用的 overlap（-1=跟随共用） |
| `video_vae` | VAE | ✅ | output_mode 含 images 时必需 |
| `audio_vae` | VAE | ❌ | 连接后输出音频波形 |

**concat_mode**：
- `chunked`：按 overlap=0 分块，块内潜空间拼接（一镜到底零代际损失），块间像素层拼接
- `independent`：每段独立 VAE 解码，像素层拼接（最保守）

**output_mode**：
- `both`：同时构建 AV Latent 和 IMAGE 序列（默认）
- `av_only`：只构建完整 AV Latent，跳过 VAE 解码（更快、更省显存）
- `images_only`：只构建 IMAGE 序列，不累积完整 AV Latent

#### 4.4.2 输出端口

| 端口 | 类型 | 说明 |
|------|------|------|
| `拼接后 AV Latent` | LATENT | 拼接结果（av_only / both） |
| `拼接后图像序列` | IMAGE | 图像序列（images_only / both） |
| `拼接后音频` | AUDIO | 音频波形（需连接 audio_vae） |
| `拼接报告` | STRING | 帧数统计与详情 |

**解码内存策略**：累积解码帧字节数低于 4 GB 时保留在内存，超过后溢出到磁盘（`output/yimo_h3/_concat_chunk_cache/`）。

---

### 4.5 YimoH3FaceRestore — 面部修复（采样后）

**功能**：对已采样的 AV Latent 独立进行面部修复，不占用采样流程。

#### 4.5.1 输入参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `av_latent` | LATENT | — | 待修复的 AV Latent |
| `video_vae` | VAE | — | 用于解码/编码帧 |
| `face_images` | IMAGE | — | 身份参考图（可选） |
| `repair_backend` | COMBO | `auto` | `auto` / `gfpgan` / `opencv` / `insightface` |
| `restore_strength` | FLOAT | `0.7` | 修复强度 |
| `det_threshold` | FLOAT | `0.5` | 人脸检测阈值 |
| `max_faces_per_frame` | INT | `10` | 每帧最大检测数 |
| `restore_unmatched` | BOOLEAN | `true` | 是否修复未匹配参考图的人脸 |
| `feather` | FLOAT | `0.25` | 边缘羽化 |
| `min_face_size` | INT | `64` | 最小人脸尺寸（像素） |
| `detection_stride` | INT | `1` | 检测步长（>1 加速但中间帧不修复） |
| `encode_back` | BOOLEAN | `true` | 是否重新编码回 AV Latent |

#### 4.5.2 输出端口

| 端口 | 类型 | 说明 |
|------|------|------|
| `修复后 AV 潜变量` | LATENT | 修复结果 |
| `修复后帧序列` | IMAGE | 修复后的帧（预览用） |
| `修复报告` | STRING | 检测/修复统计 |

---

### 4.6 YimoH3LoadSavedLatent — 加载保存的潜变量

**功能**：加载之前保存的 AV Latent 文件，用于断点续跑。

#### 4.6.1 输入参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `file_path` | STRING | `""` | 完整文件路径 |
| `load_type` | COMBO | `single_sampler` | `single_sampler` / `sequence_sampler_export` / `sequence_sampler_cache` |
| `segment_index` | INT | `-1` | 从缓存加载指定段（-1=整个文件） |
| `verbose` | BOOLEAN | `true` | 是否显示详细元数据 |

#### 4.6.2 输出端口

| 端口 | 类型 | 说明 |
|------|------|------|
| `AV 潜变量` | LATENT | 加载的潜变量 |
| `元数据（JSON）` | STRING | 文件元数据 |
| `加载报告` | STRING | 加载详情 |

⚠️ **安全提示**：本节点使用 `weights_only=False` 加载 .pt，请仅加载可信来源的文件。

---

### 4.7 YimoH3PromptComposer — 标准提示词格式化

**功能**：将分段输入组合为 MiniMax H3 官方格式的三段式或六段式提示词。

#### 4.7.1 输入参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `structure` | COMBO | `three_section` | `three_section` / `six_section` |

**three_section 字段**：
- `integrated_multimodal_description`
- `overall_soundscape`
- `non_diegetic_music`

**six_section 字段**：
- `subject_definitions`
- `summary`
- `retention_analysis`
- `detailed_description`
- `overall_soundscape`
- `non_diegetic_music`

**通用**：
- `global_suffix`（末尾追加，留空时不输出）

字段为空时自动跳过。

#### 4.7.2 输出端口

| 端口 | 类型 | 说明 |
|------|------|------|
| `组合后的提示词` | STRING | 最终提示词 |
| `组合报告` | STRING | 每个字段的字符数与包含状态 |

---

### 4.8 YimoH3PostProcessSplit — 后处理分流器

**功能**：把 AV Latent 拆分为视频 Latent / 音频 Latent / 图像帧序列。

#### 4.8.1 输入参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `av_latent` | LATENT | ✅ | 来自采样器输出 |
| `video_vae` | VAE | ❌ | 连接后才能输出「图像帧序列」 |

#### 4.8.2 输出端口

| 端口 | 类型 | 说明 |
|------|------|------|
| `视频 Latent` | LATENT | 潜空间超分方案的输入 |
| `音频 Latent` | LATENT | 旁路，最终在合流器中与超分后视频合并 |
| `图像帧序列` | IMAGE | 像素级超分方案的输入 |
| `元数据 JSON` | STRING | 形状、帧数、像素尺寸等 |

---

### 4.9 YimoH3PostProcessMerge — 后处理合流器

**功能**：把超分后的视频（Latent 或 IMAGE，二选一）与原始音频 Latent 重新组装成 AV Latent。

#### 4.9.1 输入参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `video_latent` | LATENT | ⚠️ | 与 `video_frames` 二选一 |
| `audio_latent` | LATENT | ✅ | 来自分流器的「音频 Latent」输出 |
| `video_frames` | IMAGE | ⚠️ | 与 `video_latent` 二选一，需 `video_vae` |
| `video_vae` | VAE | ❌ | 输入为 `video_frames` 时必需 |

**尺寸约束**：`video_frames` 的宽高必须是 `CANVAS_MULTIPLE`（32）的倍数，且为偶数。

#### 4.9.2 输出端口

| 端口 | 类型 | 说明 |
|------|------|------|
| `合并后的 AV Latent` | LATENT | 组装后的 AV Latent |
| `合并报告` | STRING | 组装详情 |

---

### 4.10 YimoH3HighResResampler — 高分辨率重采样器（实验性）

**功能**：把低分辨率 AV Latent 通过双线性上采样 + 重采样去噪提升到 1.2~4 倍。

⚠️ **实验性节点**：与 Turbo / Lightning 等加速 LoRA 存在机制冲突，不是社区主流方案。建议优先使用 `YimoH3PDDTwoPass` 或像素级超分（SeedVR2）。

#### 4.10.1 输入端口

| 端口 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `model` | MODEL | ✅ | MiniMax H3 DiT 模型 |
| `latent` | LATENT | ✅ | 待重采样的 AV Latent |
| `positive` | CONDITIONING | ✅ | 正向条件 |
| `sampler` | SAMPLER | ✅ | ComfyUI 采样器（目前固定使用 euler） |
| `negative` | CONDITIONING | ❌ | 负向条件 |
| `sigmas` | SIGMAS | ❌ | 可选 SIGMAS |
| `video_vae` | VAE | ❌ | 面部检测与质量控制 |
| `audio_vae` | VAE | ❌ | `enable_audio_resample=true` 时必需 |
| `segment_bundle` | CONDITIONING | ❌ | 自动读取身份参考图和源音频 |

#### 4.10.2 视频重采样参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `video_scale` | FLOAT | `2.0` | 1.2~4.0 |
| `denoise_strategy` | COMBO | `face_aware` | `uniform` / `staged` / `face_aware` |
| `base_denoise` | FLOAT | `0.5` | 0.1~0.9 |
| `face_denoise_reduction` | FLOAT | `0.5` | 面部 denoise 降低比例 |
| `enable_identity_injection` | BOOLEAN | `true` | 从 segment_bundle 读取身份图，第二遍注入 |

**denoise_strategy**：
- `uniform`：全图统一 denoise（最快）
- `staged`：分阶段 sigma 曲线（平衡）
- `face_aware`：面部感知 + 自适应 denoise（最保身份，推荐）

#### 4.10.3 时间/空间分块

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `enable_temporal_chunking` | BOOLEAN | `true` | 是否按时间分块 |
| `chunk_frames` | INT | `73` | 每个时间块的帧数 |
| `chunk_overlap` | INT | `22` | 时间块重叠帧数 |
| `enable_spatial_tiling` | BOOLEAN | `false` | 空间分块 |
| `tile_size_pixel` | INT | `512` | tile 大小（像素） |
| `tile_overlap_pixel` | INT | `128` | tile 重叠带（像素） |

#### 4.10.4 颜色匹配 / 音频 / QA

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `color_match_mode` | COMBO | `medium` | `off` / `low` / `medium` / `high` |
| `enable_audio_resample` | BOOLEAN | `false` | 是否对音频做像素层增强 |
| `audio_denoise` | FLOAT | `0.25` | 0.0~0.6 |
| `enable_timbre_anchor` | BOOLEAN | `false` | 用源音频的 RMS / Crest Factor 校准音色 |
| `enable_loudness_norm` | BOOLEAN | `false` | 响度归一化 |
| `target_rms` | FLOAT | `0.05` | 响度归一化目标 RMS |
| `enable_quality_check` | BOOLEAN | `false` | 面部相似度、色调偏差、音频质量检测 |
| `cfg` | FLOAT | `1.0` | 建议保持 1.0 |
| `seed` | INT | `0` | 随机种子 |

#### 4.10.5 输出端口

| 端口 | 类型 | 说明 |
|------|------|------|
| `高分辨率 AV Latent` | LATENT | 重采样结果 |
| `重采样报告` | STRING | 详细日志 |
| `质量报告 JSON` | STRING | QA 指标 |

---

### 4.11 YimoH3PDDTwoPass — PDD 双采封装

**功能**：封装 MiniMax H3 PDD 4+4 双采工作流。

- LOW 阶段（前 N 步）→ 学习型 latent 放大 → HIGH 阶段（剩余步数）
- 需要连接 PDD Apply 节点（如 `MiniMaxH3PDDAccApply`）输出的 MODEL 和 SIGMAS
- 仅适用 PDD LoRA，不兼容 Turbo / Lightning / DMD

#### 4.11.1 输入参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `model` | MODEL | ✅ | 来自 PDD Apply 节点的 MODEL |
| `sigmas` | SIGMAS | ✅ | 来自 PDD Apply 节点的完整 sigma 轨迹（9 点对应 8 步） |
| `split_step` | INT | `4` | 切分索引（1~7） |
| `low_stage_sigmas_override` | SIGMAS | ❌ | 覆盖 LOW 阶段 sigmas |
| `high_stage_sigmas_override` | SIGMAS | ❌ | 覆盖 HIGH 阶段 sigmas |
| `positive` | CONDITIONING | ✅ | 正向条件 |
| `negative` | CONDITIONING | ❌ | 负向条件 |
| `latent` | LATENT | ✅ | 低分辨率模板 latent |
| `scale_by` | FLOAT | `1.5` | 1.0~4.0，学习型 latent 放大倍数 |
| `seed` | INT | `0` | 随机种子 |
| `cfg` | FLOAT | `1.0` | PDD 不支持 CFG，保持 1.0 |

**依赖**：学习型 Latent 放大节点（LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler）需已安装。

#### 4.11.2 输出端口

| 端口 | 类型 | 说明 |
|------|------|------|
| `高分辨率 AV Latent` | LATENT | 双采结果 |
| `双采报告` | STRING | 详细日志 |

---

## 五、使用方法

### 5.1 快速开始：text 模式生成短视频

```
1. 拖入 YimoH3Conditioning 节点
2. 连接 clip / video_vae / audio_vae
3. workflow_mode = text
4. prompt = "a woman walking in the park"
5. length = 124（约 5 秒）
6. width = 1344, height = 768
7. 拖入 YimoH3SingleSampler
8. 将 Conditioning 的「段数据包」连接到 SingleSampler 的 segment
9. 连接 model 到 SingleSampler
10. SingleSampler 输出 → VAE Decode → Save Video
```

### 5.2 参考素材驱动：服装复刻（全程强控制）

```
YimoH3Conditioning:
├── workflow_mode: references
├── ref_images: 连接服装参考图
├── reference_retention: fully_preserved    ← 100%强度
├── ref_curve_direction: constant           ← 全程恒定
├── ref_curve_shape: linear
├── text_boost_strength: 1.0
├── prompt: |
│     subject_definitions:
│     <Subject 1> is the woman from <Picture 1>.
│     <Subject 2> is the outfit from <Picture 2>.
│     summary: <Subject 1> wearing <Subject 2>.
│     retention_analysis:
│     <Subject 1>: fully_preserved — facial identity only; none of the original clothing should appear.
│     <Subject 2>: fully_preserved — outfit design/color only; none of the original person should appear.
└── length: 124
```

### 5.3 提示词与参考冲突时优先提示词

```
YimoH3Conditioning:
├── workflow_mode: references
├── ref_images: 连接参考图
├── reference_retention: partially_preserved  ← 允许变化
├── text_boost_strength: 2.5                   ← 放大文本信号
├── text_boost_mode: deviation
└── text_boost_renorm: true
```

### 5.4 首尾帧约束生成

```
YimoH3Conditioning:
├── workflow_mode: first_last_frame
├── first_frame: 连接首帧
├── last_frame: 连接尾帧
└── length: 124
```

### 5.5 中间关键帧插值

```
YimoH3Conditioning:
├── workflow_mode: first_frame
├── first_frame: 连接首帧
├── keyframes: 连接 3 张中间帧（批次）
├── keyframe_positions: "40,80,105"   ← 手动指定位置
└── length: 124
```

### 5.6 长视频分段生成

```
YimoH3Conditioning × 3:
├── Segment 0: length=124, prompt="场景1"
├── Segment 1: length=124, prompt="场景2"
└── Segment 2: length=124, prompt="场景3"

YimoH3SequenceSampler:
├── segments: 连接3个段数据包
├── overlap_frames: 22
├── transition_mode: context_inject
├── audio_continuity: context_inject
├── auto_inherit_tail: true
├── inherit_identity: true
├── chain_id: "my_video"
└── enable_ref_curve: false
```

### 5.7 用 prev_segment_bundle 继承素材

```
Segment 0（连接全部 ref_images）:
YimoH3Conditioning:
├── workflow_mode: references
├── ref_images: 连接人物 + 服装 + 鞋子参考图
└── 段数据包 → SequenceSampler.segment_0

Segment 1（不连接任何本地素材）:
YimoH3Conditioning:
├── workflow_mode: references
├── prev_segment_bundle: 连接 Segment 0 的段数据包   ← 关键
├── prompt: 用自然语言描述（不要写 <Picture N>）
└── 段数据包 → SequenceSampler.segment_1
```

### 5.8 断点续采

**第一次运行**：

```
YimoH3SequenceSampler:
├── chain_id: "my_project"
└── 运行完成后自动缓存每段到磁盘
```

**中断后恢复**：

```
YimoH3SequenceSampler:
├── chain_id: "my_project"        ← 相同 chain_id
├── resample_segment: 2            ← 从第2段继续
└── 自动加载已缓存的段0、段1
```

### 5.9 SingleSampler 流水线式逐段生成

```
第1次运行：
YimoH3Conditioning → YimoH3SingleSampler:
├── auto_save: true
├── save_name: "segment_0"
└── project_name: "character_A"
→ 保存到 output/yimo_h3/single_sampler/character_A/segment_0.pt

第2次运行：
YimoH3LoadSavedLatent:
├── file_path: "output/yimo_h3/single_sampler/character_A/segment_0.pt"
└── load_type: "single_sampler"
→ 输出 AV Latent → 连接到 SingleSampler.prev_sampled_latent

YimoH3SingleSampler:
├── overlap_frames: 22
├── auto_inherit_tail: true
├── auto_save: true
└── save_name: "segment_1"
```

### 5.10 多参考图正确用法（人物 + 服装 + 鞋子）

```
YimoH3Conditioning:
├── workflow_mode: references
├── ref_images: [人物1, 人物2, 人物3, 服装1, 服装2, 服装3, 鞋子1, 鞋子2]  ← 共8张
├── identity_image_indices: "1,2,3"    ← 前3张标记为身份图
├── reference_retention: fully_preserved
├── ref_curve_direction: constant
├── text_boost_strength: 2.0
├── prompt: |
│     subject_definitions:
│     <Subject 1> is the woman whose facial identity, hairstyle, and body appearance come from <Picture 1>, <Picture 2>, and <Picture 3>.
│     <Subject 2> is the outfit whose design, color, and fabric details come from <Picture 4>, <Picture 5>, and <Picture 6>.
│     <Subject 3> is the shoes whose design, color, and style come from <Picture 7> and <Picture 8>.
│
│     summary:
│     The target video shows <Subject 1> wearing <Subject 2> and <Subject 3>, performing the specified action.
│
│     retention_analysis:
│     <Subject 1> (appears in [Shot 1]): fully_preserved - preserve facial identity, hairstyle, and body appearance; none of the original clothing, shoes, background, or lighting from these pictures should appear.
│     <Subject 2> ([Shot 1] clothing): fully_preserved - preserve the outfit design, color, and fabric details; none of the original person, background, or lighting from these pictures should appear.
│     <Subject 3> ([Shot 1] footwear): fully_preserved - preserve the shoes' design, color, and style; none of the original person, background, or lighting from these pictures should appear.
│
│     detailed_description:
│     [Shot 1] A medium shot shows <Subject 1> wearing the outfit from <Subject 2> and the shoes from <Subject 3>, performing [具体动作].
└── length: 124
```

**排查步骤**：
1. 检查 Report 输出的 port-to-tag mapping，确认 8 张图编号正确
2. 如果结果仍混杂：先简化为 1人物 + 1服装 + 1鞋子测试；提高 `text_boost_strength` 到 3.0；确认 `retention_analysis` 中明确了"不保留什么"

### 5.11 使用 PDD 加速采样

```
YimoH3Conditioning → 正常构建条件
         ↓
YimoH3SingleSampler / YimoH3SequenceSampler:
├── sampling_mode: pdd
├── external_sigmas: 连接 T8 PDD 节点输出的 sigmas（推荐）
│                     或留空，插件会自动生成近似的 PDD sigmas
├── sampler_name: 自动切换为 euler
└── steps: 自动根据 sigmas 长度计算
```

⚠️ 使用 PDD LoRA 时必须使用 `sampling_mode=pdd`；不能与其他 distill LoRA（Turbo 等）叠加。

### 5.12 使用 PDD 双采

```
YimoH3SingleSampler / YimoH3SequenceSampler:
├── sampling_mode: pdd_two_stage
├── pdd_split_step: 4
├── pdd_scale_by: 1.5
└── external_sigmas: 连接 PDD Apply 节点的 sigmas（可选）
```

### 5.13 后处理分流/合流（超分）

```
采样器 → YimoH3PostProcessSplit:
├── video_latent → 潜空间超分节点 → YimoH3PostProcessMerge.video_latent
├── audio_latent → YimoH3PostProcessMerge.audio_latent
└── image_frames → 像素级超分（SeedVR2 等）→ YimoH3PostProcessMerge.video_frames
                                            └── video_vae: 连接
```

---

## 六、输出文件结构

```
ComfyUI/output/yimo_h3/
├── single_sampler/
│   └── {project_name}/
│       └── {save_name}.pt                   # SingleSampler 保存的段
├── sequence_sampler/
│   ├── _segment_cache/
│   │   └── {chain_id}/
│   │       └── seg_0000.pt, seg_0001.pt...  # 断点续采缓存
│   └── exports/
│       └── {export_name}.pt                  # 自动导出的最终结果
├── _concat_chunk_cache/                      # ConcatenateSegments 溢出到磁盘的临时帧
└── _seq_chunk_cache/                         # SequenceSampler images 模式溢出临时帧
```

**`.pt` 文件格式**（`SingleSampler`）：

```python
{
    "sampled": { "samples": NestedTensor((video, audio)) },
    "metadata": {
        "mode": "references",
        "frame_count": 124,
        "seed": 12345,
        "steps": 20,
        "cfg": 7.0,
        "sampler": "euler",
        "scheduler": "normal",
        "sampling_mode": "standard",
        "timestamp": "2026-09-26T12:34:56",
        "version": "3.0.0",
        "project": "character_A",
    }
}
```

**`.pt` 文件格式**（`SequenceSampler` 段缓存 `seg_XXXX.pt`，v2 格式）：

```python
{
    "cache_format": "v2",
    "sampled_payload": {
        "_nested_parts": [video_tensor_cpu, audio_tensor_cpu],
        # 或 "_tensor": single_tensor_cpu
        # 可选 "_noise_mask_nested": [...]
    },
    "summary": { "mode": "...", "frame_count": 124, ... },
    "segment_index": 0,
}
```

> v2 格式为纯 tensor dict，`torch.load(..., weights_only=True)` 可安全加载。旧的 v1 格式（含 `sampled` 原始 NestedTensor）会被识别为 cache miss 并触发重采样，不再 fallback 到不安全加载。

---

## 七、常见问题

### Q1：`segment` 输出为 None？

**原因**：
1. `strict_prompt_tags=true` 但提示词标签与实际媒体数量不匹配
2. `ref_video` 帧数不足（`official_2_to_15s` 要求 48~360 帧）
3. 参考图/视频尺寸异常
4. `source_audio` 编码失败

**解决**：
- 将 `reference_video_policy` 改为 `model_minimum`
- 将 `strict_prompt_tags` 改为 `false`
- 查看终端日志中的 `[YimoH3]` 错误信息

### Q2：面部修复报错 "OpenCV prototxt not found"？

按 §2.4 下载 `deploy.prototxt` 和 `res10_300x300_ssd_iter_140000.caffemodel` 到 `ComfyUI/models/opencv/`。

### Q3：长视频拼接后帧数不对？

所有帧数自动对齐到 MiniMax H3 的 17n+5 网格。检查每段的 `length` 是否合理、`overlap_frames` 是否被正确对齐，并查看采样报告的 `estimated_total_frames`。

### Q4：参考强度在后半段衰减？

DiT 内部交叉注意力在采样后期自然向自注意力倾斜。这是模型固有特性。

**解决**：
- 使用 `reference_retention: fully_preserved`（数值 1.0）
- 使用 `ref_curve_direction: constant`
- 提高 `text_boost_strength` 至 2.0~3.0
- 如需更精细控制，使用 `SequenceSampler` + `enable_ref_curve=true`

### Q5：提示词不生效？

提高 `text_boost_strength`（推荐 2.0~3.0），保持 `text_boost_mode: deviation` 和 `text_boost_renorm: true`，并在提示词中明确引用媒体标签（如 `<Picture 1> 的衣服`）。

### Q6：多参考图混杂（人物/服装/鞋子互相污染）？

**原因**：
1. 提示词缺少 `retention_analysis` 块
2. `<Subject>` 定义没有明确"不保留什么"
3. `text_boost_strength` 太低

**解决**：
- 添加完整的 `retention_analysis` 块
- 使用 `identity_image_indices` 明确标记身份图
- 提高 `text_boost_strength` 到 2.0~3.0
- 先简化为 1人物 + 1服装 + 1鞋子测试
- 查看 Report 的 port-to-tag mapping

### Q7：非标准主体标签（如 `<人物1>`）能用吗？

能。Qwen3-VL 编码器能理解中文语义标签，但插件不做代码解析，只检测并给出 info 提示。建议为稳定性考虑使用 `<Subject N>` 英文标准标签。

### Q8：PDD 模式效果不理想？

- 优先连接 T8 PDD 节点输出的 `external_sigmas`
- 确认加载了 PDD 专用 LoRA
- 确认没有与其他 distill LoRA 叠加

### Q9：升级后旧工作流报错？

新增参数导致 `widgets_values` 位置错配。删除旧 YimoH3 节点，重新拖入新节点，重新设置参数。

### Q10：断点续采时提示 "legacy format ... treat as cache miss"？

v2.6.3 起段缓存改用纯 tensor 序列化（`cache_format=v2`），旧版本缓存无法在 `weights_only=True` 下安全加载。属于预期行为，系统会自动重采样缺失段。

清理旧缓存：删除 `ComfyUI/output/yimo_h3/sequence_sampler/_segment_cache/{chain_id}/` 下的旧 `.pt` 文件。

### Q11：`keep_source` / `remix_source` 怎么工作？

- **`keep_source`**（`audio_mask=0`）：采样后完全恢复源音频
- **`remix_source`**（`audio_mask=0~1`，由 `audio_denoise_strength` 控制）：采样后按 `audio_mask` 与源音频线性插值。`0.35` 表示 35% 采样结果 + 65% 源音频

> 注：MiniMax H3 采样器当前只接受 video noise_mask，audio mask 的语义由插件在采样后处理实现。

### Q12：`prev_segment_bundle` 继承后提示词里的 `<Picture N>` 不生效？

继承只对 DiT 端生效，CLIP 端看不到被继承的素材。因此**本段提示词不要写 `<Picture N>` 引用**，改用自然语言描述。

### Q13：`output_mode=images_only` 时 AV Latent 输出为 None？

这是预期行为。`images_only` 模式不累积完整 AV Latent，节省显存。如需 AV Latent，使用 `both` 或 `av_only`。

---

## 八、致谢

本插件在设计与实现过程中参考了以下项目与思路：

- **MiniMax H3 模型团队** — 音画联合生成模型本体
- **ComfyUI 官方与社区** — 节点框架、`comfy_api.latest` 接口
- **ComfyUI 官方 MiniMax H3 节点实现** — `comfy.model_base.MiniMaxH3`、`comfy.ldm.minimax.model.PackedLayout`、`minimax_payload` / `minimax_keyframes` / `minimax_refs` / `cond_video_latents` 等字段语义
- **T8star-Aix** — PDD 加速与双时钟采样思路（项目：`comfyui-minimax-h3-audio-T8`，https://github.com/T8mars/comfyui-minimax-h3-audio-T8）
- **RES4LYF** — `res_multistep` 采样器
- **GFPGAN（Tencent ARC）** — 高质量人脸修复模型
- **InsightFace** — 人脸检测、ArcFace embedding、112×112 标准对齐坐标
- **OpenCV** — 人脸检测 DNN、双边滤波、仿射变换等基础图像处理
- **Reinhard et al.** — 颜色迁移（latent mean+std 匹配）
- **经典直方图匹配 / MKL 线性色彩迁移** — 像素级颜色匹配算法
- **Jalen-Brunson** — PDD LoRA 加载与头库安装逻辑（项目：ComfyUI-MiniMax-H3-PDD-Acc，许可证：Apache-2.0）
- **LBH-123-AI** — 学习型 Latent 放大节点（项目：Comfyui_Minimax_h3_latent_Upscaler，调用不内嵌）

---

**文档版本**：v3.0.0  
**最后更新**：2026-09-26

