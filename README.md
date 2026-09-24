# face-asset-service

短剧投放素材流水线 **L2（资产层）**的基础服务：从关键帧里做人脸**检测 + 5 点对齐 + 质量打分 + embedding**，
为"哪些帧有清晰人脸""同一角色跨镜头/跨集归并"提供候选证据。

- 设计基线与全部实测依据：`../deploy/face-assets/PLAN.md`（本仓库同级目录）
- 部署与验收结论：`deploy/STATUS.md`
- 已部署：**http://192.168.9.21:8783**（Swagger 在 `/docs`）

## 一、先说许可（本项目最容易被忽略、后果最重的一点）

**只使用许可明确允许商用的权重。** 详见 [THIRD_PARTY.md](THIRD_PARTY.md) 与 [models.lock](models.lock)。

| 组件 | 代码许可 | **权重许可** | 商用 |
|---|---|---|---|
| **YuNet**（检测 + 5 点对齐） | Apache-2.0 | **MIT** | ✅ |
| **AuraFace `glintr100`**（embedding，**默认**） | Apache-2.0 | **Apache-2.0** | ✅ |
| SFace（备用轻量，128 维） | Apache-2.0 | Apache-2.0 | ✅（但身份不可分，见下） |
| ~~InsightFace `buffalo_l`/`antelopev2`~~ | MIT | **仅非商业研究** | ❌ **禁用** |
| ~~`Idiap/EdgeFace-Base`~~ | BSD-3 | **CC-BY-NC-SA-4.0** | ❌ |

> InsightFace 官方原文：*"The training data … (and the models trained with these data) are available for
> non-commercial research purposes only."* —— **MIT 的代码许可不能洗白权重**。
> **因此本仓库不得安装 `insightface`，也不得使用任何会触发其权重自动下载的上层封装**
> （`deepface` 默认配置、`CompreFace` 的 insightface 配置都属于此类）。

## 二、为什么默认 AuraFace 而不是更小的 SFace

用**肉眼确认的真值配对**在同一批真实素材上实测余弦：

| 模型 | 同一人最低 | 不同人最高 | 裕度 | 判定 |
|---|---|---|---|---|
| SFace (128d) | +0.374 | **+0.399** | **−0.025** | **不可分**：不同人竟高于同一人 |
| AuraFace (512d) | **+0.587** | +0.209 | **+0.378** | 干净可分 |

SFace 的裕度为负，意味着**不存在任何可用阈值**；这也解释了为什么单链接/平均链接聚类都会失败——
不是算法问题，是模型给不出可分特征。代价是 AuraFace 慢 4.2 倍，但服务端实测仍有 **15.61 帧/秒**
（整季 800 帧约 51 秒），完全够用。

**并且：embedding 只作候选证据。** 同一人的相似度可以低到 0.374 这一事实决定了
**人物身份必须锚定在"人工/LLM 确认的锚点图 + 角色记录"上**，不能依赖无监督聚类。

## 三、目录结构

```
face_service/
  config.py     # 全部由环境变量驱动
  engine.py     # 模型持有（懒加载/复用/释放）、检测、对齐、质量打分、embedding
  api.py        # FastAPI：/health/live /health/ready /detect /embed /analyze /api/runtime
deploy/
  Dockerfile.server          # CPU 镜像；基础镜像可用 ARG 覆盖
  compose.yaml               # 端口 8783，CPU/内存硬上限，权重只读挂载
  requirements-api.txt       # 冻结的运行依赖
  requirements-sface-only.txt# 精简路径（放弃身份可分性）
  download_model.py          # 断点续传 + size/sha256 强校验
  check_pins.py              # 核对 pin 在目标平台是否真有可安装 wheel
  build_server.sh            # 构建并记录 image sha256
  benchmark_inference.py     # 容器内吞吐基准（用于调 CPU/线程配额）
  verify_service.py          # 验收：走 HTTP 复核真值配对
  STATUS.md                  # 部署验收记录
  .env.example
models.lock                  # 权重 URL + sha256 + 许可（含"明确排除"清单）
THIRD_PARTY.md               # 第三方组件与许可留档
licenses/                    # 三份许可原文
```

## 四、接口

```
GET  /health/live   -> 200 {"status":"ok"}                 # 不触碰模型
GET  /health/ready  -> 200|503 {"device":"cpu","embedding_model":"auraface","embedding_dim":512,"loaded":true}
POST /detect        {"frames":[{"id","path"|"b64"}]}       # 检测 + 质量打分
POST /analyze       {"frames":[...]}                       # 检测 + embedding（推荐入口）
POST /embed         {"frames":[...]}                       # 同上（语义化入口）
GET  /api/runtime                                          # 引擎与限额信息
```

`faces[].quality.composite` 的构成刻意让**检测置信度与人脸像素宽主导、清晰度只作辅助**：
实测过一个反例——80px 的小脸清晰度高达 2180（小尺寸高通噪声），而清晰的正面照只有 225。
若让清晰度主导，会优先选出垃圾帧。

`face_key` = `sha1(模型名 + 量化 embedding)`，**不是身份**，只是让同一张脸跨集可被认出。
质量门控（默认 `det_score ≥ 0.85` 且人脸宽 `≥ 100px`）会以**精确率换召回率**：
`episode_008` 的 26 张脸被压到 16 张，并恰好排除了两个已知坏样本（VFX 叠影帧、80px 低置信误检）。
若某角色只出现在低置信帧里会被整体漏掉，此时应**对该角色单独降阈值重跑并标记为低置信候选**，
而不是降低全局阈值污染所有角色。

## 五、最快的上手方式

```bash
# 1) 取权重（校验 size + sha256；支持断点续传）
python3 deploy/check_pins.py                      # 先确认 pin 在目标平台可装
python3 deploy/download_model.py --lock models.lock --dest ./models

# 2) 本机起服务
FACE_ASSETS_MODELS_DIR=./models FACE_ASSETS_PORT=8783 python3 -m face_service.api

# 3) 验收（复核真值配对，而不只是健康检查）
python3 deploy/verify_service.py --base-url http://127.0.0.1:8783 \
    --frames-dir /path/to/shot-start-frames
```

## 六、两个"看起来像模型问题、实则不是"的坑（已内置防护）

1. **ONNX 文件被截断** → OpenCV/ORT 报 `Failed to parse ONNX model / Protobuf parsing failed`，
   看起来像模型不兼容，实际是文件只下到一半（SFace 10.7/38.7 MB、glintr100 51/260 MB 都遇到过）。
   故下载器**强制校验 size + sha256** 并支持断点续传。
2. **pin 写错** → `opencv-contrib-python-headless==4.13.0.88` 在 pypi 上**不存在**
   （conda 的 `cv2.__version__` 是四段式的 `4.13.0`，而 pip 是 `4.13.0.92`）。
   故新增 `check_pins.py`，在目标平台核对每个 pin 是否真有 wheel（并正确识别 `abi3` wheel）。

## 七、容器里必须显式钉住线程数

容器内 `nproc` 仍报告宿主核数（144），而 ONNX Runtime 默认按可见核数起线程池，
随后被 cgroup 配额限流 → 严重超订。**实测代价是 16 倍**（0.95 → 15.61 帧/秒）。
因此 `FACE_ASSETS_THREADS` 必须设置且与容器 `cpus` 相称；
调整配额时两者要一起调，并用 `deploy/benchmark_inference.py` 复测。
