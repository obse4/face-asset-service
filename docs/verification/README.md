# face-assets 本机实测证据（2026-09-24）

全部由本机实跑产生，用于支撑 `../deployment-plan.md` §4 与 §7 的结论。

| 文件 | 内容 |
|---|---|
| `detect_quality.json` | `episode_008` 26 个镜头起点帧的 YuNet 检测结果：人脸框、置信度、清晰度、偏航代理、质量分 |
| `clusters.json` | 阈值聚类结果（用于暴露"单链接链式传染"失败） |
| `frames-sample/` | 真值对照样本：`006`=青年男主特写、`020`=老人（带金色钟表 VFX 叠影）、`022`=老人 3/4 侧脸、`025`=老人正面清晰照 |

## 关键实测数字

- 抽帧：每镜头起点 1 帧 = 26 帧（对比 naive 1fps 的 75 帧，省 65%）
- 26 帧中 20 帧检出人脸；质量门控（置信≥0.85 且宽≥100px）后剩 16 张
- 余弦真值配对：同一青年 6↔3 = **+0.735**；同一老人 22↔25 = **+0.374**；不同人 6↔22 = **+0.165**
- 单链接聚类把老人的 25 错并进青年类；平均链接把同一老人的 22/25 拆开 → **无监督聚类不可靠**
- CPU（Apple M4）：仅检测 56.9 帧/秒；检测+对齐+SFace embedding 33.5 帧/秒；20 集 520 帧 ≈ 15.5s
- SFace embedding 维度实测 = **128**

## 权重哈希（已实测）

| 文件 | sha256 | size |
|---|---|---|
| `face_detection_yunet_2023mar.onnx` | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` | 232589 |
| `face_recognition_sface_2021dec.onnx` | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` | 38696353 |
| `glintr100.onnx` | `a7933ea5330113b01c9b60351d8f4c33003f145d8470ac5f0e52ee2effe25c60` | 260694151 |

> 注意：SFace 与 glintr100 首次下载均被截断。截断的 ONNX 会让 OpenCV 报
> “Failed to parse ONNX model / Protobuf parsing failed”，**看起来像模型不兼容，实为文件不完整**。
> 因此下载器必须校验 size + sha256 并支持断点续传。
