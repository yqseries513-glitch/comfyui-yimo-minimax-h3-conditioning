# Yimo MiniMax H3 Conditioning v5.0.0 完整使用手册

面向 ComfyUI 的 MiniMax H3 音画联合生成增强节点套件。

---

## 一、插件简介

**Yimo MiniMax H3 Conditioning** 是面向 ComfyUI 的 MiniMax H3 音画联合生成模型的自定义节点套件，专注于：

- **长视频生成**：多段采样 + 段间上下文注入，突破单次采样时长限制
- **多段一致性**：跨段继承关键帧、参考素材、身份图、尾帧
- **双时钟采样**：video / audio 使用独立 shift sigma 轨迹，恢复运镜并保住音频质量
- **参考强度控制**：语义分级、时间曲线、文本信号增强
- **面部修复**：采样后自动检测并修复人脸，支持身份匹配、多人参考、GFPGAN / InsightFace / OpenCV 多级降级
- **后处理分流/合流**：AV Latent ↔ 视频 Latent / 音频 Latent / 图像帧序列
- **标签规范化**：多种媒体标签写法自动归一化

> **版本**：v5.0.0  
> **面部修复引擎**：v5.0.0  
> **作者**：yimo  
> **许可证**：GPL-3.0-or-later  
> **兼容 ComfyUI**：最新版（含 `comfy_api.latest`）  
> **Python**：3.10+  
> **最后更新**：2026-10-15

---

## 二、环境依赖

### 2.1 必需依赖

```text
opencv-python>=4.8.0
numpy>=1.24.0
```

`torch` / `torchaudio` / `torchvision` 由 ComfyUI 运行时提供。

### 2.2 可选依赖：面部修复

面部修复所有依赖均为可选。未安装时自动降级到 OpenCV 增强，不会阻塞主流程。

推荐安装：

```bash
pip install insightface onnxruntime-gpu
pip install gfpgan --no-deps facexlib
pip install basicsr
```

| 包 | 用途 | 备注 |
|---|---|---|
| insightface | 人脸检测 + ArcFace embedding | 依赖 onnxruntime |
| onnxruntime-gpu | ONNX 推理后端 | CUDA 不兼容时自动降级 CPU |
| gfpgan | 高质量人脸重建 | 必须加 `--no-deps` |
| facexlib | GFPGAN 辅助库 | 提供检测器 + 解析器 |
| basicsr | GFPGAN 底层依赖 | 需手动安装 |
| opencv-python | 人脸检测 DNN、双边滤波、仿射变换、贴回 | 必需依赖 |

### 2.3 常见版本冲突处理

#### 问题 A：albucore 版本冲突

```text
ImportError: cannot import name 'median_blur' from 'albucore'
```

修复：

```bash
pip install --no-deps "albucore==0.2.18"
```

#### 问题 B：basicsr 与 PyTorch 2.x 不兼容

```text
ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'
```

修复：

```bash
python -c "p=r'<python>\Lib\site-packages\basicsr\data\degradations.py'; s=open(p,encoding='utf-8').read(); s=s.replace('from torchvision.transforms.functional_tensor import rgb_to_grayscale','from torchvision.transforms.functional import rgb_to_grayscale'); open(p,'w',encoding='utf-8').write(s); print('patched')"
```

或手动打开：

```text
<python>\Lib\site-packages\basicsr\data\degradations.py
```

第 8 行：

```text
改前: from torchvision.transforms.functional_tensor import rgb_to_grayscale
改后: from torchvision.transforms.functional import rgb_to_grayscale
```

---

## 三、节点清单

| 节点 ID | 显示名 | 功能 |
|---|---|---|
| `YimoH3Conditioning` | Yimo H3 音画条件 | 构建正向/负向条件、AV Latent、混音音频、段数据包 |
| `YimoH3SingleSampler` | Yimo H3 AV双时钟单段采样器 | 单段采样 + 双时钟 + 段间逻辑 + 面部修复 |
| `YimoH3SequenceSampler` | Yimo H3 AV双时钟序列采样器 | 多段循环采样 + 双时钟 + 拼接成长视频 + 断点续采 + 面部修复 |
| `YimoH3ConcatenateSegments` | Yimo H3 段拼接器 | 将多段 AV Latent 拼接成长视频 |
| `YimoH3FaceRestore` | Yimo H3 面部修复（采样后） | 独立的人脸检测与修复 |
| `YimoH3LoadSavedLatent` | Yimo H3 加载保存的潜变量 | 加载 `.pt` 文件，用于断点续跑 |
| `YimoH3PostProcessSplit` | Yimo H3 后处理分流器 | AV Latent → 视频 / 音频 / 图像帧 |
| `YimoH3PostProcessMerge` | Yimo H3 后处理合流器 | 超分后视频 + 音频 → AV Latent |

---

## 四、核心概念：双时钟与单时钟

采样器只有 `sampler_name` 决定时钟模式：

| sampler_name | 时钟 | 说明 |
|---|---|---|
| `dual_clock_euler` | 双时钟 | video / audio 独立 sigma 轨迹（`shift_video=12`, `shift_audio=3`）。推荐。 |
| `euler` / `dpmpp_2m` / ... | 单时钟 | ComfyUI 原生 AV 调度。`shift_video` / `shift_audio` 无效。 |

**双时钟优势**：恢复运镜（video 用高 shift），保住音频（audio 用低 shift）。  
**单时钟下 audio 时钟跟随 video**，音频质量可能下降。

推荐路线：

```text
sampler_name: dual_clock_euler
steps: 20~30
cfg: 4~7
shift_video: 12.0
shift_audio: 3.0
不加载任何加速 LoRA
```

---

## 五、节点参数详解

### 5.1 YimoH3Conditioning — 音画条件构建

节点 ID：`YimoH3Conditioning`  
显示名：Yimo H3 音画条件

#### 5.1.1 输入端口

| 端口 | 类型 | 必需 | 说明 |
|---|---|---|---|
| clip | CLIP | ✅ | MiniMax H3 原生 Qwen3-VL CLIP |
| video_vae | VAE | ✅ | H3 Video VAE |
| audio_vae | VAE | ✅ | H3 Audio VAE |
| prev_segment_bundle | CONDITIONING | ❌ | 上一段的「段数据包」，用于跨段素材继承 |
| first_frame / last_frame | IMAGE | ❌ | 首/尾帧图 |
| keyframes | IMAGE | ❌ | 中间关键帧图（批次或列表，最多 9 张） |
| ref_images | IMAGE | ❌ | 参考图（批次或列表，最多 9 张） |
| ref_videos | AUTOGROW | ❌ | 参考视频（最多 3 个） |
| source_audio | AUDIO | ❌ | 源音频 |
| override_audio | AUDIO | ❌ | 覆盖输出的混音音频 |
| ref_video_audios | AUTOGROW | ❌ | 与 `ref_video_N` 配对的音频（最多 3 个） |
| style_audios | AUTOGROW | ❌ | 风格参考音频（最多 3 个） |

#### 5.1.2 关键参数

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| workflow_mode | COMBO | text | `text` / `first_frame` / `last_frame` / `first_last_frame` / `references` / `hybrid` |
| audio_policy | COMBO | generate_new | `keep_source` / `remix_source` / `reference_only` / `generate_new` |
| width / height | INT | 1344 / 768 | 自动对齐到 32 的倍数 |
| length | INT | 124 | 24fps；自动对齐到 `17n+5` 网格 |
| reference_retention | COMBO | fully_preserved | `fully_preserved` / `partially_preserved` / `attribute_transfer` / `weak_reference` / `no_reference` |
| reference_retention_scale | FLOAT | 1.0 | 0~1；最终强度 = 档位值 × scale |
| ref_curve | COMBO | off | `off` / `fade_out` / `fade_in` / `peak_middle` / `peak_ends` / `custom` |
| text_boost_strength | FLOAT | 1.0 | 0.25~8.0；文本信号增强强度 |
| text_boost_mode | COMBO | deviation | `deviation` / `naive` |
| strict_prompt_tags | BOOLEAN | false | 严格校验提示词媒体标签 |
| audio_offset_frames | INT | 0 | 音频整体偏移帧数。正数=延后，负数=提前。仅对 `keep_source` / `remix_source` / `reference_only` 生效。 |

#### 5.1.3 输出端口

| 端口 | 类型 | 说明 |
|---|---|---|
| 段数据包 | CONDITIONING | 直接接采样器 `segment` |
| 正向条件 | CONDITIONING | 通用正向条件 |
| 负向条件 | CONDITIONING | 通用负向条件 |
| AV 潜变量 | LATENT | 空 AV Latent 模板 |
| 混音音频 | AUDIO | 用于最终混音 |
| 条件化提示词 | STRING | 规范化后的提示词 |
| 报告 | STRING | 详细日志 |

### 5.2 YimoH3SingleSampler — Yimo H3 AV双时钟单段采样器

节点 ID：`YimoH3SingleSampler`  
显示名：Yimo H3 AV双时钟单段采样器

#### 5.2.1 输入端口

| 端口 | 类型 | 必需 | 说明 |
|---|---|---|---|
| model | MODEL | ✅ | MiniMax H3 DiT 模型 |
| segment | CONDITIONING | ✅ | 来自 `YimoH3Conditioning.Segment Bundle` |
| video_vae | VAE | ❌ | 用于尾帧继承与面部修复 |
| prev_sampled_latent | LATENT | ❌ | 前段完整采样结果 |
| face_images | IMAGE | ❌ | 面部参考图 |

#### 5.2.2 采样参数

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| sampler_name | COMBO | `dual_clock_euler` | `dual_clock_euler`=双时钟；其它=单时钟 |
| scheduler | COMBO | `native_flow` | 双时钟专用 |
| steps | INT | 20 | 无加速 LoRA 推荐 20~30 |
| shift_video | FLOAT | 12.0 | 仅 `dual_clock_euler` 下生效 |
| shift_audio | FLOAT | 3.0 | 仅 `dual_clock_euler` 下生效 |
| cfg | FLOAT | 1.0 | 加速 LoRA 下 1.0；无加速 LoRA 时 4~7 |
| denoise | FLOAT | 1.0 | 降噪强度 |
| seed | INT | 0 | 随机种子 |

#### 5.2.3 段间控制参数

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| overlap_frames | INT | 22 | 与前段重叠帧数。`17n+5` 网格映射 |
| transition_mode | COMBO | `context_inject` | `context_inject` / `hard_concat` |
| audio_continuity | COMBO | `context_inject` | `context_inject` / `break` |
| auto_inherit_tail | BOOLEAN | true | 前段尾帧作为本段 first_frame |
| inherit_identity | BOOLEAN | true | 跨段继承身份参考图 |

#### 5.2.4 参考曲线

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| enable_ref_curve | BOOLEAN | false | 启用参考强度时间曲线 |
| override_ref_curve | BOOLEAN | false | 覆盖 segment 曲线配置 |
| ref_curve_direction_override | COMBO | inherit | 曲线方向覆盖 |
| ref_curve_shape_override | COMBO | inherit | 曲线形状覆盖 |

#### 5.2.5 面部修复 / 颜色匹配 / 自动保存

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| face_images | IMAGE | — | 面部参考图 |
| face_restore | BOOLEAN | false | 启用采样中面部修复 |
| face_restore_backend | COMBO | auto | `auto` / `gfpgan` / `insightface` / `opencv` |
| face_restore_strength | FLOAT | 0.7 | 对口型建议 0.5~0.6 |
| face_similarity_threshold | FLOAT | 0.45 | ArcFace 余弦匹配阈值 |
| color_match_mode | COMBO | off | `off` / `low` / `medium` / `high` / `max` |
| color_match_strength | FLOAT | 0.7 | 颜色匹配强度 |
| color_match_reference_frames | INT | 1 | 参考帧数 |
| auto_save | BOOLEAN | false | 是否自动保存 |
| save_name | STRING | "" | 保存文件名 |
| project_name | STRING | "" | 项目子文件夹名 |

#### 5.2.6 输出端口

| 端口 | 类型 | 说明 |
|---|---|---|
| 采样后 AV 潜变量 | LATENT | 采样结果 |
| 采样报告 | STRING | 详细采样日志 |

### 5.3 YimoH3SequenceSampler — Yimo H3 AV双时钟序列采样器

节点 ID：`YimoH3SequenceSampler`  
显示名：Yimo H3 AV双时钟序列采样器

参数基本同 SingleSampler，新增：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| overlap_list | STRING | "" | 逐对 overlap（逗号分隔，如 `22,40,30`）。负数视为未设置，0 是硬切 |
| chain_id | STRING | default | 缓存标识符。双时钟缓存目录独立（`_segment_cache_dc`） |
| resample_segment | INT | -1 | 从第几段重新采样（-1=全部） |
| segment_seeds | STRING | "" | 每段独立种子（逗号分隔，`none` 表示回退到派生种子） |
| stop_after_segment | INT | -1 | 采样到第几段后停止（-1=全部） |
| output_format | COMBO | latent | `latent` / `images` |
| chunk_memory_threshold_mb | INT | 4096 | images 模式的内存阈值 |
| auto_export_final | BOOLEAN | false | 是否自动导出 |
| export_name | STRING | "" | 导出文件名（不含扩展名） |

### 5.4 YimoH3ConcatenateSegments — 段拼接器

节点 ID：`YimoH3ConcatenateSegments`  
显示名：Yimo H3 段拼接器

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| concat_mode | COMBO | chunked | `chunked` / `independent` |
| output_mode | COMBO | both | `both` / `av_only` / `images_only` |
| overlap_frames | INT | 22 | 共用 overlap |
| overlap_list | STRING | "" | 逐对 overlap |
| video_overlap_frames | INT | -1 | -1=跟随共享源 |
| audio_overlap_frames | INT | -1 | -1=跟随共享源 |

### 5.5 YimoH3FaceRestore — 面部修复（采样后）

节点 ID：`YimoH3FaceRestore`  
显示名：Yimo H3 面部修复 (采样后)

**核心定位：把视频里的人脸变清晰，不是换脸。**

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| av_latent | LATENT | ✅ | 待修复的 AV Latent |
| video_vae | VAE | ✅ | 用于解码/编码帧 |
| face_images | IMAGE | ❌ | 面部参考图（可选） |
| repair_backend | COMBO | auto | `auto` / `gfpgan` / `opencv` / `insightface` |
| restore_strength | FLOAT | 0.7 | 对口型建议 0.5~0.6 |
| det_threshold | FLOAT | 0.5 | 人脸检测置信度阈值 |
| max_faces_per_frame | INT | 10 | 每帧最大检测人脸数 |
| restore_unmatched | BOOLEAN | true | 是否修复未匹配到参考图的人脸 |
| feather | FLOAT | 0.25 | 边缘羽化宽度 |
| min_face_size | INT | 64 | 最小人脸尺寸 |
| detection_stride | INT | 1 | 每 N 帧检测修复一次。对口型必须保持 1 |
| encode_back | BOOLEAN | true | 是否编码回 AV 潜变量 |

### 5.6 YimoH3LoadSavedLatent — 加载保存的潜变量

节点 ID：`YimoH3LoadSavedLatent`  
显示名：Yimo H3 加载保存的潜变量

输出：AV 潜变量 / 元数据 JSON / 加载报告。

> **安全提示**：本节点使用 `weights_only=False` 加载 `.pt`，请仅加载可信来源的文件。

### 5.7 YimoH3PostProcessSplit — 后处理分流器

节点 ID：`YimoH3PostProcessSplit`  
显示名：Yimo H3 后处理分流器

输出：视频 Latent / 音频 Latent / 图像帧序列 / 元数据 JSON。

### 5.8 YimoH3PostProcessMerge — 后处理合流器

节点 ID：`YimoH3PostProcessMerge`  
显示名：Yimo H3 后处理合流器

输出：合并后的 AV Latent / 合并报告。

---

## 六、使用方法

### 6.1 质量优先：双时钟 + 无加速 LoRA

```text
YimoH3Conditioning × 3:
  workflow_mode: text
  prompt: ...
  length: 124

YimoH3SequenceSampler:
  sampler_name: dual_clock_euler
  steps: 20~30
  cfg: 4~7
  shift_video: 12.0
  shift_audio: 3.0
  overlap_frames: 22
  transition_mode: context_inject
  audio_continuity: context_inject
  chain_id: "my_project"
  不加载任何加速 LoRA
```

### 6.2 单段 + 手动续采

```text
YimoH3Conditioning (segment_0) → YimoH3SingleSampler → 保存
YimoH3Conditioning (segment_1) → YimoH3SingleSampler (prev_sampled_latent ← 上一段)
...
```

### 6.3 断点续采

在 `YimoH3SequenceSampler` 中：

```text
chain_id: "my_project"        # 所有段自动缓存到该 chain
resample_segment: 2           # 从第 3 段开始重采（前 2 段读缓存）
stop_after_segment: 3         # 采样完第 4 段后停止
segment_seeds: "0,123,none"   # 前两段指定种子，第三段用派生种子
```

### 6.4 images 模式

```text
YimoH3SequenceSampler:
  output_format: images
  video_vae: ← H3 Video VAE
  chunk_memory_threshold_mb: 4096
```

### 6.5 面部修复

三处节点都能用：

| 节点 | 用途 |
|---|---|
| YimoH3SingleSampler | 单段采样时顺带修复 |
| YimoH3SequenceSampler | 多段序列采样时每段修复 |
| YimoH3FaceRestore | 独立节点，采样后单独修复 |

单段采样器典型配置（对口型黄金配置）：

```text
YimoH3SingleSampler:
  face_restore: true
  face_restore_backend: auto 或 gfpgan
  face_restore_strength: 0.5 ~ 0.6
  face_similarity_threshold: 0.35 ~ 0.45
  face_images: ← 可选，只连想“重点修”的人的清晰正脸
  video_vae: ← 必连（用于解码/编码）
```

> 采样器里的 `face_restore=True` 不要求身份库必须非空。  
> 即使 `face_images` 未连接、InsightFace 不可用、参考图检测不到人脸，也会调用通用修复路径，对所有检测到的人脸生效。

### 6.6 后处理超分

**潜空间超分**：

```text
采样器 → YimoH3PostProcessSplit → video_latent → 超分节点 → video_latent → YimoH3PostProcessMerge → AV Latent
```

**像素级超分**：

```text
采样器 → YimoH3PostProcessSplit → image_frames → 超分节点 → video_frames → YimoH3PostProcessMerge → AV Latent
```

### 6.7 两段一镜到底 + 音画同步实战

**场景**：一段参考动作视频切成两段，一首歌切成两段。第一段用第一段素材，第二段用第二段素材。要求最终拼接后动作连贯、歌曲连贯。

**条件节点（第二段）设置**：

```text
YimoH3Conditioning:
  source_audio: 连接第二段音频
  audio_policy: keep_source 或 remix_source
  audio_offset_frames: 22   (正偏移，整体延后)
```

**采样器设置**：

```text
YimoH3SingleSampler / SequenceSampler:
  overlap_frames: 22
  transition_mode: context_inject
  audio_continuity: context_inject
  auto_inherit_tail: false
  inherit_identity: true
  enable_ref_curve: false
```

**最终效果**：

- 视频动作：第一段结束画面被注入到第二段前 22 帧，拼接裁掉后，动作在时间轴上无缝衔接。
- 歌曲连贯：第二段音频被延后 22 帧，注入覆盖前 22 帧并在拼接时裁掉，最终输出正好从原始第二段音频的第 0 帧开始。
- 注意：第二段参考视频建议也从第 22 帧开始使用，以免模型在锁定帧上对应到错误的参考时间轴。

---

## 七、面部修复完整指南

### 7.1 功能分层

| 层 | 做什么 | 依赖包 | 依赖模型 |
|---|---|---|---|
| 1. 检测层 | 找脸、取 5 个关键点、取身份 embedding | insightface, onnxruntime(-gpu) | buffalo_l（5 个 onnx） |
| 2. 身份匹配层 | 与参考图比对，判断“是不是同一人” | 同第 1 层 | 同上 |
| 3. 修复层 | 人脸重建（变清晰） | gfpgan, basicsr, facexlib | GFPGANv1.4.pth + facexlib 两个 pth |
| 4. 贴回层 | 把修好的脸贴回原图 | opencv-python | 无 |

四层都成功，face_restore 才真正生效。

### 7.2 模型文件准备

共 4 组模型，合计约 800 MB。

#### 7.2.1 InsightFace buffalo_l（约 340 MB）

5 个 `.onnx` 文件：

- `det_10g.onnx`
- `w600k_r50.onnx`
- `1k3d68.onnx`
- `2d106det.onnx`
- `genderage.onnx`

下载：  
`https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip`

放置：  
`<ComfyUI 模型根>/insightface/models/buffalo_l/`

注意路径要带 `models` 这一层。

#### 7.2.2 GFPGANv1.4.pth（约 348 MB）

下载：  
`https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth`

放置：  
`<ComfyUI 模型根>/facerestore_models/GFPGANv1.4.pth`

#### 7.2.3 facexlib 两个权重

- `detection_Resnet50_Final.pth`  
  `https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth`
- `parsing_parsenet.pth`  
  `https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth`

放置：

```text
<python>/Lib/site-packages/gfpgan/weights/
  detection_Resnet50_Final.pth
  parsing_parsenet.pth
```

如果日志显示下载到 `facexlib/weights/`，则改放 `facexlib/weights/`。

#### 7.2.4 OpenCV 检测模型（可选）

- `deploy.prototxt`  
  `https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/deploy.prototxt`
- `res10_300x300_ssd_iter_140000.caffemodel`  
  `https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/res10_300x300_ssd_iter_140000.caffemodel`

放置：  
`<ComfyUI 模型根>/opencv/`

### 7.3 模型目录总览

```text
<ComfyUI 模型根>/
├── vae/
├── unet/
├── checkpoints/
├── clip/
├── insightface/
│   └── models/
│       └── buffalo_l/
│           ├── det_10g.onnx
│           ├── w600k_r50.onnx
│           ├── 1k3d68.onnx
│           ├── 2d106det.onnx
│           └── genderage.onnx
├── facerestore_models/
│   └── GFPGANv1.4.pth
└── opencv/
    ├── deploy.prototxt
    └── res10_300x300_ssd_iter_140000.caffemodel
```

Python 环境：

```text
<python>/Lib/site-packages/gfpgan/weights/
  detection_Resnet50_Final.pth
  parsing_parsenet.pth
```

### 7.4 路径解析机制

插件自动查找模型路径：

1. 查 ComfyUI 已注册类别：`folder_paths.get_folder_paths("insightface" / "facerestore_models" / "opencv")`。
2. 自动注册：遍历 `folder_paths.folder_names_and_paths`，向上追溯最多 6 层，找到 `insightface/models/buffalo_l` 等标记后自动注册。
3. 通用兜底：解析 `extra_model_paths.yaml` 的 `base_path` + `cwd/models`。

通常无需修改 `extra_model_paths.yaml`。

### 7.5 使用方式

#### 7.5.1 三处节点都能用

| 节点 | 用途 |
|---|---|
| YimoH3SingleSampler | 单段采样时顺带修复 |
| YimoH3SequenceSampler | 多段序列采样时每段修复 |
| YimoH3FaceRestore | 独立节点，采样后单独修复 |

#### 7.5.2 关键参数

| 参数 | 默认 | 说明 |
|---|---|---|
| face_restore | false | 是否启用 |
| face_images | — | 参考人脸（可选） |
| face_restore_backend | auto | auto / gfpgan / insightface / opencv |
| face_restore_strength | 0.7 | 对口型建议 0.5~0.6 |
| face_similarity_threshold | 0.45 | 身份匹配阈值 |

#### 7.5.3 采样报告解读

```text
face_restore: identity bank built (1 faces)
face_restore: detection=insightface, repair=gfpgan, faces=124 detected, 124 restored (person_1:35), unmatched_skipped=0
```

| 字段 | 含义 |
|---|---|
| identity bank built (N faces) | 从 face_images 提取了 N 个人的 embedding |
| detection | 实际用的检测器 |
| repair | 实际用的修复后端 |
| faces=N detected | 检测到的人脸总数 |
| M restored | 修复成功的人脸数 |
| (person_1:X, person_2:Y) | 匹配到各参考身份的人脸数 |
| (none) | 没有任何脸匹配到 bank |
| unmatched_skipped | 因为 restore_unmatched=False 而跳过的未匹配人脸数 |

### 7.6 身份匹配机制

`face_images` 输入里的第 1 张图 = `person_1`，第 2 张 = `person_2`，以此类推。  
每张参考图只取 1 张脸（最大那张，按 bbox 面积降序）。

匹配逻辑：

1. 提取 ArcFace embedding（512 维）
2. 跟 bank 里每个人算余弦相似度
3. 取最相似的那个人
4. 若最高分 >= `face_similarity_threshold`（默认 0.45）→ 命中
5. 否则 unmatched

`restore_unmatched`：

| 值 | 行为 |
|---|---|
| True（默认） | 所有检测到的脸都修（包括路人），未匹配的归 unmatched |
| False | 只修匹配到 bank 的脸，未匹配的跳过 |

### 7.7 常见问题排查

#### insightface 加载失败

| 错误 | 原因 | 处理 |
|---|---|---|
| FileNotFoundError: ... buffalo_l ... | 路径不对 | 确认 `<模型根>/insightface/models/buffalo_l/` 下有 5 个 onnx |
| onnxruntime CUDA 相关错误 | onnxruntime-gpu 与 CUDA 不兼容 | 自动降级 CPU |
| No module named 'insightface' | 未安装 | `pip install insightface onnxruntime-gpu` |

#### GFPGAN 加载失败

| 错误 | 原因 | 处理 |
|---|---|---|
| No module named 'basicsr' | 未安装 | `pip install basicsr` |
| No module named 'torchvision.transforms.functional_tensor' | basicsr 与 torch 2.x 不兼容 | 打补丁 |
| GFPGAN 模型不存在 | GFPGANv1.4.pth 没放对 | 放到 `<模型根>/facerestore_models/` |
| 卡在 Downloading ... detection_Resnet50_Final.pth | facexlib 权重没准备好 | 手动下载放到 `<python>/Lib/site-packages/gfpgan/weights/` |
| GFPGAN 加载失败 (CUDA) + 成功 (CPU fallback) | CUDA 初始化失败 | 自动回退 CPU，功能可用 |

#### faces=0 detected

- 视频里确实没有人脸
- 检测后端是 opencv 且没装 opencv 检测模型
- `det_threshold=0.5` 太高

#### M restored < N detected

终端搜 `[YimoH3] 人脸修复失败` 查看 traceback。

#### 完整诊断脚本

```bash
cd <python>

python -c "from insightface.app import FaceAnalysis; app=FaceAnalysis(name='buffalo_l', root=r'<模型根>\insightface', providers=['CUDAExecutionProvider','CPUExecutionProvider']); app.prepare(ctx_id=0, det_size=(640,640)); print('INSIGHTFACE OK')"

python -c "from gfpgan import GFPGANer; import torch; r=GFPGANer(model_path=r'<模型根>\facerestore_models\GFPGANv1.4.pth', upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None, device='cuda' if torch.cuda.is_available() else 'cpu'); print('GFPGANer OK')"
```

两条都打印 OK，face_restore 就能跑。

---

## 八、常见问题

### Q1：sampler_name 应该选哪个？

- 质量优先：`dual_clock_euler`
- 对照 / 特殊需求：ComfyUI 原生采样器（`euler` / `dpmpp_2m` / ...），此时走单时钟

### Q2：shift_audio 设了但没生效？

仅在 `sampler_name=dual_clock_euler` 下生效。单时钟下由 ComfyUI 原生调度接管。

### Q3：hard_concat 下总帧数不对？

H3 的 latent 网格在段边界处不连续，所以 hard_concat 下从 latent 反推的帧数是近似值。实际帧数 = 段数 × 每段帧数。如需精确帧数，用 `context_inject`。

### Q4：双时钟和单时钟的缓存能混用吗？

不能。两条路径物理隔离：

- 双时钟 → `_segment_cache_dc/{chain_id}/`
- 单时钟 → `_segment_cache/{chain_id}/`

即便 `chain_id` 相同，缓存也不会互串。

### Q5：序列采样器如何只重采某一段？

设置 `resample_segment=N`（0-based），前 N 段从缓存读，从第 N 段开始重采。前提是 `chain_id` 已缓存前 N 段，且缓存目录一致。

### Q6：segment_seeds 里怎么让某段用派生种子？

填 `none`。例如 `segment_seeds: "0,123,none,456"`。

### Q7：reference_retention 和 reference_retention_scale 的关系？

最终强度 = retention 档位值 × scale，clamp 到 `[0, 1]`。

- `fully_preserved = 1.0`
- `partially_preserved = 0.7`
- `attribute_transfer = 0.4`
- `weak_reference = 0.15`
- `no_reference = 0.0`

`scale` 是连续微调系数，默认 1.0。

### Q8：text_boost_strength 推荐值？

- 1.0 = 无操作
- 2.0 = 推荐起点
- 最大 8.0

MiniMax H3 的 Qwen3-VL 编码器区分度较低，适度增强可显著提高提示词遵循度。

### Q9：面部修复到底会不会换脸？

不会。它只做检测、身份匹配、清晰化/重建、贴回。

### Q10：face_restore=True 但没连 face_images，还会修吗？

会。会走 `restore_unmatched=True` 通用修复路径，对所有检测到的人脸生效。

### Q11：为什么报告里是 (none)？

表示所有检测到的人脸跟 `face_images` 里的身份 embedding 相似度都低于 `face_similarity_threshold`。  
常见原因：不是同一人、角度差、画风差异、参考图质量差。  
如果 `restore_unmatched=True`，所有脸仍然会被修复，只是没有身份区分。

### Q12：GFPGAN 和 OpenCV 修复有什么区别？

- `gfpgan`：真正的人脸重建，质量最好，需要 GFPGAN + basicsr + facexlib 模型。
- `insightface`：主要用 InsightFace 做检测/匹配，修复层仍可能降级到 OpenCV 增强。
- `opencv`：轻量锐化 + 双边滤波，不重建，速度快但效果有限。
- `auto`：自动选择最佳可用后端。

### Q13：CUDA OOM 怎么办？

GFPGAN 推理时若 CUDA OOM，会自动尝试把 GFPGAN 移到 CPU 重试。  
若仍失败，会降级到 OpenCV 增强。也可以手动把 `face_restore_backend` 设为 `opencv`。

---

## 九、致谢与第三方许可

本插件整体以 **GPL-3.0-or-later** 分发。参考/借鉴以下项目，详见 `THIRD_PARTY_NOTICES.md`：

- **ComfyUI 官方 MiniMax H3 实现**  
  项目：`https://github.com/comfyanonymous/ComfyUI`  
  许可证：GPL-3.0  
  使用方式：字段语义与 `minimax_payload` / `minimax_keyframes` / `minimax_refs` / `cond_video_latents` 结构参考；`__init__.py` 中的 monkey-patch 针对官方 `comfy.model_base.MiniMaxH3.extra_conds` 的兼容性补充。

- **T8star-Aix / comfyui-minimax-h3-audio-T8**  
  项目：`https://github.com/T8mars/comfyui-minimax-h3-audio-T8`  
  许可证：GPL-3.0-or-later  
  使用方式：双时钟采样数学与 KSAMPLER 注册思路参考，`dual_clock_core.py` 包含基于该项目的衍生实现。原文件：`h3_t8/sampling.py`。

- **GFPGAN**  
  项目：`https://github.com/TencentARC/GFPGAN`  
  许可证：Apache-2.0  
  使用方式：作为可选依赖用于高质量人脸修复，未内嵌源代码。

- **InsightFace**  
  项目：`https://github.com/deepinsight/insightface`  
  许可证：MIT  
  使用方式：作为可选依赖用于人脸检测与 ArcFace 身份匹配，112×112 标准对齐坐标来源于该项目。

- **OpenCV**  
  项目：`https://github.com/opencv/opencv`  
  许可证：Apache-2.0  
  使用方式：作为必需依赖用于基础图像处理。

- **算法思想引用**  
  - Reinhard et al., “Color Transfer between Images” — latent 层均值+标准差颜色迁移  
  - 经典直方图匹配 — 像素级逐通道直方图匹配  
  - MKL 线性色彩迁移 — 鲁棒统计近似实现

---

**文档版本**：v5.0.0  
**最后更新**：2026-10-15