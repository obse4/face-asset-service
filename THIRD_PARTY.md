# 第三方组件与许可留档

本服务**只**使用许可明确允许商用的组件。任何新增依赖都必须在此登记代码许可与**权重许可**两项——
两者可以不同，且**代码许可不能覆盖权重许可**（InsightFace 就是这种情形）。

## 一、实际使用的组件

| 组件 | 用途 | 代码许可 | **权重许可** | 商用 | 留档文件 |
|---|---|---|---|---|---|
| YuNet `face_detection_yunet_2023mar.onnx` | 人脸检测 + 5 点对齐 | Apache-2.0（opencv_zoo 仓库） | **MIT**（模型目录自带 LICENSE） | ✅ | `licenses/LICENSE-YuNet-MIT.txt` |
| SFace `face_recognition_sface_2021dec.onnx` | 人脸 embedding（**备用轻量**，128 维；实测身份裕度为负，不用于身份归并） | Apache-2.0 | **Apache-2.0**（模型目录自带 LICENSE） | ✅ | `licenses/LICENSE-SFace-Apache-2.0.txt` |
| AuraFace `glintr100.onnx` | 人脸 embedding（**默认**，512 维；实测裕度 +0.378） | Apache-2.0 | **Apache-2.0** | ✅ | `licenses/LICENSE-AuraFace-Apache-2.0.txt` |
| OpenCV（`opencv-contrib-python-headless`） | 提供 `FaceDetectorYN` / `FaceRecognizerSF` | Apache-2.0 | — | ✅ | 随依赖分发 |
| FastAPI / uvicorn / pydantic / numpy | HTTP 服务外壳 | MIT / BSD / MIT / BSD | — | ✅ | 随依赖分发 |

每项的 sha256 与字节数见 `models.lock`；下载器必须校验，详见该文件的 `_caution`。

## 二、明确排除的组件（含理由）

### 1. InsightFace `buffalo_l` / `antelopev2` —— 排除

代码 MIT，但**权重仅限非商业研究**。官方 README 原文（逐字引用，用于合规留档）：

> The code of InsightFace is released under the MIT License. There is no limitation for both academic
> and commercial usage.
>
> **The training data containing the annotation (and the models trained with these data) are available
> for non-commercial research purposes only.**
>
> Both manual-downloading models from our github repo and auto-downloading models with our
> python-library follow the above license policy (which is for non-commercial research purposes only).

其模型库 README 另写明 “ALL models are available for non-commercial research purposes only”，
官方亦设有付费商用授权渠道。**因此本服务不得安装 `insightface`，也不得调用会触发其权重自动下载的
任何上层封装**（例如 `deepface` 的默认配置、`CompreFace` 的 insightface 配置）。

### 2. `Idiap/EdgeFace-Base` —— 排除

代码 BSD-3-Clause（看似理想、体积小、方法新），但 Hugging Face 权重为 **CC-BY-NC-SA-4.0**，
属非商用。典型"只看代码许可就会踩中"的陷阱。

### 3. CompreFace —— 排除

代码 Apache-2.0，但其 README 自述基于 FaceNet/InsightFace，最佳配置会引入 InsightFace 系权重；
且发布停滞（last release 2023-08）。

## 三、已知的共同残留风险（必须记录，不得假装不存在）

几乎所有高精度人脸识别权重最终都源自 **MS-Celeb-1M / CASIA-WebFace / VGGFace2 / WebFace** 等
**研究用途数据集**（MS1M 已被微软撤回）。即便模型作者给出了宽松许可，我们依赖的是**作者的授权声明**，
而非一条可独立核验的干净权属链。

因此本项目的选型原则是：**只选用权属声明明确的项目**，其中 AuraFace 是本次调研里唯一由作者
**明确声明**训练数据已清商用的候选。

若后续要用于高风险场景，建议就"训练数据权属链"单独做一次法务确认；本条属于**风险提示**，
不是法律意见。

## 四、变更流程

1. 新增模型 → 先查 **权重许可**（不是只看代码许可），写入 `models.lock` 的 `models` 或 `excluded`。
2. 下载后**必须**校验 size + sha256（并支持断点续传）。
3. 在本文件登记许可与来源链接，并把许可原文落到 `licenses/`。
4. 若某项许可不干净，写进"排除"一节并说明理由，避免后续有人重新引入。
