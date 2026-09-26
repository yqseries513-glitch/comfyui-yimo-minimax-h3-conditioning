# Third-Party Notices

本插件（yimo-minimax-h3-conditioning）使用了以下第三方项目的思路、接口或代码片段。
各项目的版权归原作者所有，具体许可证以各项目官方仓库为准。

---

## ComfyUI 官方 MiniMax H3 实现

- 项目：https://github.com/comfyanonymous/ComfyUI
- 许可证：GPL-3.0
- 使用方式：字段语义与 `minimax_payload` / `minimax_keyframes` / `minimax_refs` /
  `cond_video_latents` 结构参考；`__init__.py` 中的 monkey-patch 针对官方
  `comfy.model_base.MiniMaxH3.extra_conds` 的补充。

## T8star-Aix / comfyui-minimax-h3-audio-T8

- 项目：https://github.com/T8mars/comfyui-minimax-h3-audio-T8
- 许可证：GPL-3.0-or-later
- 使用方式：PDD 双时钟采样与 4+4 双采工作流结构参考
  （`sampler_core.py`、`pdd_two_pass_node.py`）。

## Jalen-Brunson / ComfyUI-MiniMax-H3-PDD-Acc

- 项目：https://github.com/Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc
- 许可证：Apache-2.0
- Copyright (c) Jalen-Brunson
- 使用方式：`MiniMaxH3PDDAccApply` 节点的 MODEL + SIGMAS 接口格式参考
  （`pdd_two_pass_node.py`）。

## LBH-123-AI / Comfyui_Minimax_h3_latent_Upscaler

- 项目：https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler
- 使用方式：通过 ComfyUI 的 `NODE_CLASS_MAPPINGS` 动态调用该节点的放大功能，
  **未内嵌其代码**，用户需自行安装该插件。

## RES4LYF

- 项目：https://github.com/ClownsharkBatwing/RES4LYF
- 许可证：GPL-3.0
- 使用方式：本插件仅提供 `sampler_name` 参数入口，**未内嵌其采样器代码**，
  用户需在 ComfyUI 中自行安装该插件才能使用 `res_multistep` 等采样器。

## GFPGAN

- 项目：https://github.com/TencentARC/GFPGAN
- 许可证：Apache-2.0
- 使用方式：作为**可选依赖**用于高质量人脸修复，通过 `pip install gfpgan` 引入，
  未内嵌任何源代码。

## InsightFace

- 项目：https://github.com/deepinsight/insightface
- 许可证：MIT
- 使用方式：作为**可选依赖**用于人脸检测与 ArcFace 身份匹配，
  112×112 标准对齐坐标来源于该项目。

## OpenCV

- 项目：https://github.com/opencv/opencv
- 许可证：Apache-2.0
- 使用方式：作为**必需依赖**用于基础图像处理（人脸检测 DNN、双边滤波、仿射变换等）。

---

## 算法思想引用

以下算法为公开学术成果，本插件基于其思想独立实现：

- **Reinhard et al., "Color Transfer between Images"** — latent 层均值+标准差颜色迁移
- **经典直方图匹配** — 像素级逐通道直方图匹配
- **MKL 线性色彩迁移** — 鲁棒统计近似实现

---

如需转载、分发或二次开发本插件，请同时遵守上述各项目的许可证要求。
