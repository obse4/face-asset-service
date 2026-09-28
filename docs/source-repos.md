# 源仓库与 fork 清单

本文件列出整条流水线**实际依赖的外部仓库**、钉住的版本、许可，以及建议的 fork 动作。
用途：把"可复刻的痕迹"补全到**上游来源**这一层——换机器/换人接手时，能据此重建全部依赖。

> 记录时间 2026-09-28。所有 commit 与 hash 均为当时实测取到，非记忆值。

## 一、建议你 fork / 新建的仓库

| # | 仓库 | 动作 | 为什么 |
|---|---|---|---|
| 1 | **`obse4/face-asset-service`** | **新建**（本地已有 3 次提交，只差 push） | 我们自己的 L2 服务：人脸检测/对齐/embedding + 许可留档（`models.lock`、`THIRD_PARTY.md`、`licenses/`） |
| 2 | **`opencv/opencv_zoo`** | **fork 到 obse4** | 我们两个权重（YuNet、SFace）与其**各自的 LICENSE 文件**都在此仓库。fork 它 = 把"权重 + 许可原文"一起锚定，是许可合规最直接的证据 |
| 3 | `fal/AuraFace-v1`（HuggingFace） | 可选：在 HF 上 duplicate | 默认 embedding 模型 glintr100 的来源。它是 HF 仓库不是 GitHub 仓库，用 HF 的 duplicate 功能复制 |
| 4 | `dsh-video-timeline`、`dsh-face-assets` | 可选：各建一个仓库 | 两个 DSH 插件是我们自有代码，目前在 DSH 工作区（`~/Documents/deepseek-harness/默认工作区/`）**没有版本控制** |

push 第 1 项的现成命令：

```bash
cd "/Users/wanghan/Documents/ChatGPT/192.168.9.21大模型服务器/face-asset-service"
git remote add origin git@github.com:obse4/face-asset-service.git
git push -u origin main
```

## 二、全部依赖仓库（含已 fork 与"不要 fork"）

| 组件 | 上游仓库 | 钉住的版本 | 许可（代码 / 权重） | 用途 | 状态 |
|---|---|---|---|---|---|
| OmniShotCut（分镜/转场） | `UVA-Computer-Vision-Lab/OmniShotCut` | 你的 fork `obse4/OmniShotCut` 分支 `codex/llm-server-deployment`，HEAD `b55db5bf6eeeca213e0f0cd2c83441f389a08f76` | MIT | L1 分镜与转场标签 | **已 fork** ✓ |
| OmniShotCut 权重 | HF `uva-cv-lab/OmniShotCut` | revision `7b59cc525168ea13d36dcf08db6885fd7edc2215`；`OmniShotCut_ckpt.pth` sha256 `5948ea78e00626c0e6c5e742e64873ef872cf4a5071d2a0841aed51c3e686cfa` | 见上游 | 分镜推理 | 钉 hash，无需 fork |
| MOSS Transcribe（ASR+说话人） | `OpenMOSS/MOSS-Transcribe-Diarize` | 你的 fork `obse4/MOSS-Transcribe-Diarize` 分支 `codex/llm-server-deployment`，HEAD `76a84abd125104421ea4bef6cfa131dcec03333f`；已配 `upstream` 远端 | Apache-2.0 | L1 语音转写与说话人 | **已 fork** ✓ |
| MOSS 权重 | HF `OpenMOSS-Team/MOSS-Transcribe-Diarize` | revision `704aa4a9c304e8520be88901e0d1960158ef5b15` | Apache-2.0 | 转写推理 | 钉 revision，无需 fork |
| **YuNet**（人脸检测 + 5 点） | `opencv/opencv_zoo`（模型目录；训练源 `ShiqiYu/libfacedetection.train`） | `opencv_zoo` main `47534e27c9851bb1128ccc0102f1145e27f23f98`；**文件按 sha256 钉死**：`8f2383e4…552fa4` | Apache-2.0 / **MIT** | L2 检测与对齐 | **建议 fork `opencv_zoo`** |
| **SFace**（备用 embedding） | 同上 `opencv_zoo` | 文件 sha256 `0ba9fbfa…c34e79` | Apache-2.0 / **Apache-2.0** | L2 备用（身份不可分，见下） | 同上 |
| YuNet 训练源 | `ShiqiYu/libfacedetection.train` | main `dca340aa082c71081a68d17db8e58b33a58a914b` | BSD-3 | 溯源 | 可选 |
| **AuraFace glintr100**（默认 embedding） | HF `fal/AuraFace-v1` | 文件 sha256 `a7933ea5…e25c60`，size 260694151 | **Apache-2.0** | L2 默认嵌入（512 维） | 可选 duplicate |
| 通用图像嵌入（后期场景聚类） | HF `facebook/dinov2-with-registers-base` | 尚未下载 | Apache-2.0 | L3 场景聚类 | 未来用 |

### 我们自己的代码

| 组件 | 位置 | 版本控制 | 说明 |
|---|---|---|---|
| `face-asset-service` | `~/Documents/ChatGPT/192.168.9.21大模型服务器/face-asset-service` | ✅ 本地 git，3 次提交 | 待 push 到 `obse4/face-asset-service` |
| `dsh-video-timeline`（P0 插件） | `~/Documents/deepseek-harness/默认工作区/dsh-video-timeline` | ❌ 无 | 建议也建仓库 |
| `dsh-face-assets`（L2 插件） | `~/Documents/deepseek-harness/默认工作区/dsh-face-assets` | ❌ 无 | 建议也建仓库 |
| 部署项目（PLAN/STATUS/verification/rehearsal） | `~/Documents/ChatGPT/192.168.9.21大模型服务器` | ⚠️ 有 `.git` 但**零提交**（全部 untracked） | 痕迹只在磁盘上，建议至少提交一次 |

## 三、明确**不要** fork / 引入的仓库

| 仓库 | 许可 | 为什么排除 |
|---|---|---|
| `deepinsight/insightface`（含 `buffalo_l` / `antelopev2`） | 代码 MIT，**权重仅限非商业研究** | 官方 README 原文：*"The training data … (and the models trained with these data) are available for non-commercial research purposes only."* 且官方另售商用授权。**`pip install insightface` + `FaceAnalysis()` 会自动下载这些权重**，是当前最大的合规陷阱 |
| `Idiap/EdgeFace-Base` | 代码 BSD-3，**权重 CC-BY-NC-SA-4.0** | 看似理想（小、新、许可宽松的代码），实为陷阱：权重非商用 |
| `serengil/deepface` | MIT（代码） | 代码可商用，但其默认配置会引导到 InsightFace 系权重；且它的核心价值是"支持大量模型"，与本项目刻意收窄的依赖面相反 |
| CompreFace | Apache-2.0（代码） | 最佳配置依赖 InsightFace 系权重，且发布停滞 |
| `hypit-ai/hypit` | **source-available**（非 OSI 开源）：Apache-2.0 **加**两条限制——禁止多租户/SaaS 使用其代码或衍生作品；禁止商业再分发 | 我们只借鉴**设计思想**（锚点身份、语义锚点、provenance 副作用文件），**未抄任何代码**。若要引其代码必须先取得商用授权 |

## 四、许可合规的落点（便于审计）

| 证据 | 位置 |
|---|---|
| 三个权重的 URL + sha256 + 许可 + 商用判定 | `face-asset-service/models.lock` |
| 第三方组件与许可留档（含被排除项的理由与原文引用） | `face-asset-service/THIRD_PARTY.md` |
| 许可原文 | `face-asset-service/licenses/`（YuNet MIT、SFace Apache-2.0、AuraFace Apache-2.0） |
| 部署验收与实测数据 | `face-asset-service/deploy/STATUS.md` |
| 设计与全部实测依据 | `face-asset-service/docs/deployment-plan.md` |
