# face-assets 部署验收（2026-09-24）

已部署于 **http://192.168.9.21:8783**（Swagger 在 `/docs`）。单容器、CPU-only、无数据库。
部署期间既有服务（MOSS 8781、OmniShotCut 8782、flashhead 8790、MiniMax-H3 8188/8288/8388/8488、
MellowDesign 8080/8780/3000、Mihomo）**全程保持运行且健康检查为 200**，容器清单只新增 `face-assets` 一个。

设计基线与全部实测依据见 `../../../deploy/face-assets/PLAN.md`。

## 1. 来源与镜像

| 项 | 值 |
|---|---|
| 服务端代码路径 | `/opt/face-assets` |
| 权重（只读挂载） | `/srv/face-assets/models` |
| 数据路径 | `/data/face-assets/{jobs,tmp,backups}` |
| 基础镜像 | `python:3.11-slim`（复用机上已有镜像，免拉取） |
| 最终镜像 | `face-assets:local` sha256 `ef8ed0935fae02b2331647b7e1a1296a0c672e1826abd039835dd836b944d973` |
| 首个镜像（已被取代） | sha256 `4b11bf2dccdd652f9c8b846983672cc7450bfd73d81d886e73b87605702107e2` |
| 密钥 | `/opt/face-assets/deploy/secrets/api-keys.json`（mode 600），compose 经 `deploy/.env`（mode 600）读取 |

镜像内依赖自证（实测输出）：

```
python 3.11.16
cv2 4.13.0  numpy 2.4.6  onnxruntime 1.30.0  fastapi 0.135.1
FaceDetectorYN_create: True
FaceRecognizerSF_create: True
onnxruntime providers: ['AzureExecutionProvider', 'CPUExecutionProvider']
```

**`providers` 里没有 CUDA** ⇔ 结构上不可能占用显存；`docker inspect` 亦确认 `DeviceRequests=null`。

## 2. 权重完整性

在服务端对 `models.lock` 逐项核对 sha256 与字节数，**三项全部一致**：

| 文件 | size | sha256 | 许可 |
|---|---|---|---|
| `face_detection_yunet_2023mar.onnx` | 232,589 | `8f2383e4…552fa4` | MIT |
| `face_recognition_sface_2021dec.onnx` | 38,696,353 | `0ba9fbfa…c34e79` | Apache-2.0 |
| `glintr100.onnx`（默认模型） | 260,694,151 | `a7933ea5…e25c60` | Apache-2.0 |

> 权重由本机（已完成哈希核验的副本）经 SSH 传输，因为**目标机无法访问 huggingface.co 与
> media.githubusercontent.com**（直连与 mihomo 代理均失败，代理还返回 502）。
> 记录此点是因为它影响复刻：换机器时若外网可达，直接跑 `deploy/download_model.py` 即可。

## 3. 功能验收

`deploy/verify_service.py` 从**本机走 LAN** 调用（不只是容器内自测），结果：

```
/health/live  -> 200 {'status': 'ok'}
/health/ready -> 200 device=cpu model=auraface dim=512 loaded=True
鉴权生效：无令牌被拒
/analyze 4 帧 -> 0.333s（12.01 帧/秒）
  shot_start_003: 质量=0.5937 宽=131px dim=512
  shot_start_006: 质量=0.6011 宽=259px dim=512
  shot_start_022: 质量=0.7056 宽=380px dim=512
  shot_start_025: 质量=0.8466 宽=299px dim=512
  同一人·青年: 006 ↔ 003 = +0.587 期望高 ✓
  同一人·老人: 022 ↔ 025 = +0.650 期望高 ✓（本地基准 0.650，偏差 0.000）
  不同人:      006 ↔ 022 = +0.183 期望低 ✓（本地基准 0.178，偏差 0.005）
验收全部通过 ✓
```

**跨机可复现性**：质量分与本地（Apple M4）逐项一致，余弦偏差 ≤0.005（即浮点噪声量级）。
这与 PLAN §12 步骤 6 的"可复刻"要求一致。

## 4. 吞吐：一次由"自我限流"造成的 16 倍损失

首次部署实测只有 **0.95 帧/秒**（26 帧 27.42s），远低于本机 8.60 帧/秒。
受控实验（同一批 26 帧，逐项改 `--cpus` 与线程数）显示**几乎全部损失来自我自己设的上限**：

| cpus | 线程数 | 吞吐 | 整季 800 帧 |
|---|---|---|---|
| 6 | 4（首次部署值） | 1.15 帧/秒 | 696s |
| 12 | 12 | 3.95 帧/秒 | 203s |
| 16 | 12 | 4.47 帧/秒 | 179s |
| 24 | 16 | 6.66 帧/秒 | 120s |
| 32 | 24（**最终采用**） | 12.68 帧/秒 | 63s |
| 48 | 32 | 18.64 帧/秒 | 43s |
| 不限 | 默认 | 19.31 帧/秒 | 41s |

**根因（一个真实的代码缺陷）**：容器内 `nproc` 仍报告宿主核数 **144**，而 ONNX Runtime 默认按可见
核数起线程池，却被 cgroup 配额限流 → 严重超订。原先 `FACE_ASSETS_THREADS` 配置项**被读取却从未
应用**（`InferenceSession` 未传 `SessionOptions`），线程数实际由环境变量偶然决定。
本次修复：显式 `options.intra_op_num_threads`，并令其与容器 CPU 配额相称。

修复后 HTTP 实测（对外真实路径）：

```
26 帧 / 1.67s = 15.61 帧/秒（16 张脸）
对比修复前 0.95 帧/秒 → 提升 16.4 倍
整季 800 帧 ≈ 51s
```

## 5. 资源与邻居性

| 项 | 值 |
|---|---|
| CPU 硬上限 | `NanoCpus=32e9`（32 核 ≈ 整机 22%，绝不吃满 144 核） |
| 内存硬上限 | 6 GiB |
| 空闲占用 | CPU **1.22%**、内存 **484.9 MiB** |
| GPU | **零占用**（无 DeviceRequests；`onnxruntime` 仅 CPU provider） |
| 重启策略 | `unless-stopped` |
| 日志 | json-file 轮转，20MB × 3 |
| 权重挂载 | `/srv/face-assets/models` **只读** |

部署与压测期间既有服务 8781/8782 健康检查均为 200，其余容器仍为 Up 9 days（未重启）。

> 诚实记录：压测与本服务的 CPU 密集运行会让 1 分钟负载均值从基线 0.25 升到 ~12–19。
> 这是**受 32 核硬上限约束**的突发占用，不是失控；15 分钟均值仍为 2.85。
> 若同机将来出现对延迟敏感的 CPU 工作负载，应下调 `cpus`/`FACE_ASSETS_THREADS`（两者要一起调）。

## 6. 本次遭遇的两个"看起来像模型问题、实则不是"的坑（均已修复并留档）

1. **ONNX 文件被截断** → OpenCV/ORT 报 `Failed to parse ONNX model / Protobuf parsing failed`，
   看起来像模型不兼容，实际是 SFace 只下到 10.7/38.7 MB、glintr100 只下到 51/260 MB。
   对策：`download_model.py` 强制校验 size+sha256 并支持断点续传（三条路径均已验证）。
2. **pin 写错** → 首个镜像构建失败：`opencv-contrib-python-headless==4.13.0.88` **在 pypi 上不存在**
   （我是从 conda 的 `cv2.__version__` 4.13.0 推的，而 pip 用四段式 `4.13.0.92`）。
   对策：新增 `deploy/check_pins.py`，在目标平台（cp311/manylinux-x86_64，并识别 abi3 wheel）
   逐项核对 pin 是否真有可安装的 wheel。

## 7. 未验证 / 已知限制

- **未做长时间稳定性运行**：本次为功能与资源验收，未做 24 小时连续跑或内存泄漏观测。
- **未做规模上限压测**：整季 800 帧的 51s 是线性外推（基于 26 帧实测），未真跑 800 帧批次。
- **未做并发压测**：多个客户端同时调用时的排队行为未测；当前实现是同步处理、无微批处理
  （PLAN §5.4 的微批属后续优化）。
- **未验证视频解码路径**：本服务只接受图像帧或 base64，抽帧由上游 ffmpeg 负责。
- **未测跨机重建**：`models.lock` 的离线重建流程尚未在第二台机器上实跑（因目标机无法访问 HF，
  本次是传输而非下载；换机复刻时需先确认外网可达性）。
- **动画素材未验证**：若剧集为动画，ArcFace 系模型（含 AuraFace）在风格化人脸上会退化，
  需先用 200 帧样本检查聚类纯度（PLAN §4.6 已记录该风险）。

## 8. 运维

```bash
# 查看状态 / 日志
cd /opt/face-assets && docker compose -p face-assets -f deploy/compose.yaml ps
docker logs --tail 50 face-assets

# 重启
cd /opt/face-assets && docker compose -p face-assets -f deploy/compose.yaml restart

# 更新代码后重建并滚动替换
tar czf - face_service deploy | ssh llm-server 'tar xzf - -C /opt/face-assets'
ssh llm-server 'cd /opt/face-assets && bash deploy/build_server.sh'
ssh llm-server 'cd /opt/face-assets && docker compose -p face-assets -f deploy/compose.yaml up -d --force-recreate'

# 验收
python3 deploy/verify_service.py --base-url http://192.168.9.21:8783 \
    --api-key "$(ssh llm-server 'python3 -c "import json;print(json.load(open(\"/opt/face-assets/deploy/secrets/api-keys.json\"))[\"face-assets\"][\"api_key\"])"')" \
    --frames-dir /data/face-assets/tmp/acceptance-frames
```

**回滚**：只作用于 `face-assets` 这一个 compose 项目（`docker compose -p face-assets ... down`，
**绝不加 `-v`**）。上一版镜像 sha256 见 §1，可用 `docker tag` 回退。
**绝不对 MOSS / H3 / Mihomo / MellowDesign 的数据或容器执行任何操作。**

## 9. 证据清单

| 文件 | 内容 |
|---|---|
| `/opt/face-assets/deploy/verification/build.log` | 最终构建日志 |
| `/opt/face-assets/deploy/verification/image-id.txt` | 最终镜像 sha256 |
| `/data/face-assets/tmp/acceptance-frames/` | 26 个真实镜头起点帧（验收输入） |
| `../../../deploy/face-assets/verification/` | 本机侧证据：检测质量、聚类对比、模型裕度对比、对照帧 |
