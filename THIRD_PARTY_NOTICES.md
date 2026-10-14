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
- 本插件不复制 ComfyUI 官方源代码，仅做兼容性 monkey-patch。

## T8star-Aix / comfyui-minimax-h3-audio-T8

- 项目：https://github.com/T8mars/comfyui-minimax-h3-audio-T8
- 许可证：GPL-3.0-or-later
- 使用方式：双时钟采样数学与 KSAMPLER 注册思路参考。
- 本仓库 `dual_clock_core.py` 包含基于该项目的衍生实现，原文件为 `h3_t8/sampling.py`。
- 原版权归 T8star-Aix 及贡献者所有；修改部分版权归 yimo 所有。
- 本插件整体按 GPL-3.0-or-later 分发，符合 GPL 兼容要求。

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