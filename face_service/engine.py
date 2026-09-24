"""人脸嵌入引擎：YuNet 检测/对齐 + SFace 或 AuraFace 嵌入 + 质量打分。

许可依据见 ../THIRD_PARTY.md：YuNet 权重 MIT、SFace 权重 Apache-2.0、AuraFace 权重 Apache-2.0。
**本模块不得引入任何非商用权重**（尤其不得 import insightface 或任何会触发其权重下载的封装）。

设计要点（均来自真实素材实测，见 deploy/face-assets/PLAN.md §4）：

- **质量分以检测置信度与人脸像素宽为主，清晰度只作辅助**。实测反例：镜头 19 的 80px 小脸
  清晰度高达 2180（小尺寸高通噪声），而镜头 25 的清晰正面照只有 225。若让清晰度主导，
  会优先选出垃圾帧。
- **质量门控必须在 embedding 之前**。低置信 + 过小的人脸会污染后续聚类。
- **同一人的余弦相似度可以低到 0.374**（老人正面 vs 侧脸），而不同人是 0.165；
  因此 embedding 只作**候选证据**，身份判定必须靠锚点图 + 人工确认。
"""
from __future__ import annotations

import hashlib
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# ArcFace 标准 112x112 五点模板（用于 AuraFace 路径的对齐）
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# 质量分公式里"给满学分"的参照值
REFERENCE_FACE_PX = 200.0
REFERENCE_BLUR = 300.0
REFERENCE_YAW = 0.35


@dataclass
class QualityWeights:
    """质量分权重，便于按素材调参（默认值来自 episode_008 的实测排序）。"""

    size_exponent: float = 1.0
    blur_weight: float = 0.5
    frontal_weight: float = 0.4


@dataclass
class DetectedFace:
    """单张人脸的检测与质量结果。"""

    bbox: list[int]
    landmarks: list[list[float]]
    det_score: float
    face_px: int
    blur: float
    frontal: float
    composite: float
    embedding: list[float] | None = None
    embedding_model: str | None = None
    face_key: str | None = None
    extra: dict = field(default_factory=dict)


def composite_quality(det_score: float, face_px: int, blur: float, yaw_proxy: float, weights: QualityWeights) -> float:
    """计算合成质量分。

    结构：`det_score × 尺寸项 × 清晰度项 × 正面项`，各项都归一到约 `[0,1]`。
    尺寸项用幂次控制主导程度；清晰度与正面度只做温和修正（以 `weight` 决定其影响上限）。

    @param det_score: 检测置信度。
    @param face_px: 人脸框宽度（像素）。
    @param blur: 人脸裁切的 Laplacian 方差。
    @param yaw_proxy: 由 5 点估计的偏航代理值（0 为正脸）。
    @param weights: 权重配置。
    @returns 质量分，越大越好。
    """
    size_term = min(1.0, max(0.0, face_px) / REFERENCE_FACE_PX) ** weights.size_exponent
    blur_term = 1.0 - weights.blur_weight + weights.blur_weight * min(1.0, math.sqrt(max(blur, 0.0) / REFERENCE_BLUR))
    frontal_term = 1.0 - weights.frontal_weight + weights.frontal_weight * (
        1.0 - min(1.0, abs(yaw_proxy) / REFERENCE_YAW)
    )
    return float(det_score * size_term * blur_term * frontal_term)


def yaw_proxy_of(landmarks: np.ndarray) -> float:
    """由 5 点关键点估计偏航（正脸约 0，越大越侧）。

    用鼻尖相对双眼中心的水平偏移、以双眼间距归一化。它只是**代理量**，不是真实头部姿态。

    @param landmarks: 形状 `(5, 2)` 的关键点。
    @returns 偏航代理值。
    """
    eye_center_x = (landmarks[0][0] + landmarks[1][0]) / 2.0
    eye_width = abs(landmarks[1][0] - landmarks[0][0]) + 1e-6
    return float((landmarks[2][0] - eye_center_x) / eye_width)


def blur_of(crop: np.ndarray) -> float:
    """人脸裁切的 Laplacian 方差，作为清晰度代理。"""
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def face_key_of(model: str, embedding: np.ndarray, quantize_decimals: int = 4) -> str:
    """由模型名 + 量化后的 embedding 生成稳定的候选键。

    它**不是身份**，只是让同一张脸在不同集/不同批次里能被认出来。量化是为了容忍浮点抖动。

    @param model: 模型标识。
    @param embedding: 已归一化的特征向量。
    @param quantize_decimals: 量化小数位。
    @returns 形如 `sface:abcd…` 的键。
    """
    quantized = np.round(np.asarray(embedding, dtype=np.float64), quantize_decimals)
    payload = f"{model}|" + ",".join(f"{value:.4f}" for value in quantized)
    return f"{model}:{hashlib.sha1(payload.encode('utf-8')).hexdigest()}"


class FaceEngine:
    """线程安全的模型持有者：懒加载 + 复用 + 可释放。

    模型很小（YuNet 232KB、SFace 39MB），且实测 CPU 上检测 56.9 帧/秒、含嵌入 33.5 帧/秒，
    因此默认走 CPU：**占用 0 显存**，对只剩约 10GB/卡的邻居最友好。
    """

    def __init__(self, models_dir: str | Path, embedding_model: str = "sface", min_score: float = 0.85, min_face_px: int = 100, weights: QualityWeights | None = None, intra_op_threads: int | None = None) -> None:
        """初始化（不加载模型）。

        @param models_dir: 权重目录，须含 `face_detection_yunet_2023mar.onnx` 与所选嵌入模型。
        @param embedding_model: `sface` 或 `auraface`。
        @param min_score: 质量门控的检测置信度下限。
        @param min_face_px: 质量门控的人脸像素宽下限。
        @param weights: 质量分权重。
        @param intra_op_threads: ONNX Runtime 的 intra-op 线程数上限。
            必须显式设置：容器里 `nproc` 仍报告宿主核数（144），若交给 ORT 自行决定，
            它会按 144 线程起池、却被 cgroup 配额限流，造成严重超订与吞吐塌陷。
            实测（Xeon 8352V，26 帧）：cpus=6/OMP=4 → 1.15 帧/秒；cpus=32/线程 24 → 12.68 帧/秒。
        """
        self._models_dir = Path(models_dir)
        self._embedding_model = embedding_model
        self.min_score = min_score
        self.min_face_px = min_face_px
        self.weights = weights or QualityWeights()
        self.intra_op_threads = intra_op_threads
        self._lock = threading.Lock()
        self._detector = None
        self._recognizer = None
        self._ort_session = None
        self._ort_input = None
        self._ort_output = None
        self._input_size: tuple[int, int] | None = None

    @property
    def embedding_model(self) -> str:
        """当前嵌入模型标识。"""
        return self._embedding_model

    @property
    def embedding_dim(self) -> int:
        """当前嵌入模型维度（SFace=128，AuraFace=512）。"""
        return 128 if self._embedding_model == "sface" else 512

    def _ensure_loaded(self) -> None:
        """懒加载模型；调用方须自行持锁或接受一次性的重复加载。"""
        if self._detector is not None:
            return
        yunet = self._models_dir / "face_detection_yunet_2023mar.onnx"
        if not yunet.exists():
            raise FileNotFoundError(f"缺少 YuNet 权重：{yunet}（先运行 deploy/download_model.py）")
        self._detector = cv2.FaceDetectorYN.create(
            str(yunet), "", (320, 320), score_threshold=0.5, nms_threshold=0.3, top_k=5000
        )
        if self._embedding_model == "sface":
            sface = self._models_dir / "face_recognition_sface_2021dec.onnx"
            if not sface.exists():
                raise FileNotFoundError(f"缺少 SFace 权重：{sface}（先运行 deploy/download_model.py）")
            self._recognizer = cv2.FaceRecognizerSF.create(str(sface), "")
        elif self._embedding_model == "auraface":
            # 延迟导入：CPU + SFace 路径不需要 onnxruntime。
            import onnxruntime as ort

            glint = self._models_dir / "glintr100.onnx"
            if not glint.exists():
                raise FileNotFoundError(f"缺少 AuraFace 权重：{glint}（先运行 deploy/download_model.py）")
            providers = [name for name in ("CUDAExecutionProvider", "CPUExecutionProvider") if name in ort.get_available_providers()]
            options = ort.SessionOptions()
            if self.intra_op_threads:
                # 显式钉住线程数，与容器的 CPU 配额相称；不设就会按宿主核数起池而被限流。
                options.intra_op_num_threads = int(self.intra_op_threads)
                options.inter_op_num_threads = 1
            self._ort_session = ort.InferenceSession(str(glint), sess_options=options, providers=providers)
            self._ort_input = self._ort_session.get_inputs()[0]
            self._ort_output = self._ort_session.get_outputs()[0]
        else:
            raise ValueError(f"未知嵌入模型：{self._embedding_model}（可选 sface / auraface）")

    def unload(self) -> None:
        """释放模型引用（CPU 路径下的价值主要在可测性；GPU 路径下会真正释放显存）。"""
        with self._lock:
            self._detector = None
            self._recognizer = None
            self._ort_session = None
            self._ort_input = None
            self._ort_output = None
            self._input_size = None

    def _detect_raw(self, image: np.ndarray) -> np.ndarray | None:
        """执行一次检测，必要时重设输入尺寸（OpenCV 要求与图像同尺寸）。"""
        height, width = image.shape[:2]
        if self._input_size != (width, height):
            self._detector.setInputSize((width, height))
            self._input_size = (width, height)
        _, faces = self._detector.detect(image)
        return faces

    def _embed_sface(self, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        """SFace 路径：用 OpenCV 自带的 alignCrop + feature。"""
        aligned = self._recognizer.alignCrop(image, face)
        return np.asarray(self._recognizer.feature(aligned), dtype=np.float64).flatten()

    def _embed_auraface(self, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        """AuraFace 路径：5 点相似变换到 112x112，按 ArcFace 预处理后跑 ONNX。"""
        landmarks = face[4:14].reshape(5, 2).astype(np.float32)
        matrix, _ = cv2.estimateAffinePartial2D(landmarks, ARCFACE_TEMPLATE, method=cv2.LMEDS)
        aligned = cv2.warpAffine(image, matrix, (112, 112), borderValue=0.0)
        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = ((rgb - 127.5) / 128.0).transpose(2, 0, 1)[None, ...]
        raw = self._ort_session.run([self._ort_output.name], {self._ort_input.name: blob})[0]
        return np.asarray(raw, dtype=np.float64).flatten()

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        """L2 归一化，使余弦相似度可直接用点积。"""
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector

    def analyze(self, image: np.ndarray, with_embedding: bool = True, max_faces: int = 32) -> list[DetectedFace]:
        """检测 + 质量打分（+ 嵌入）。

        @param image: BGR 图像。
        @param with_embedding: 是否计算嵌入。
        @param max_faces: 单帧最多返回的人脸数（按质量分降序）。
        @returns 按质量分降序的人脸列表；已通过质量门控。
        """
        with self._lock:
            self._ensure_loaded()
            faces = self._detect_raw(image)
            if faces is None or len(faces) == 0:
                return []
            height, width = image.shape[:2]
            results: list[DetectedFace] = []
            for face in faces:
                x, y, bw, bh = face[:4]
                det_score = float(face[-1])
                if det_score < self.min_score or bw < self.min_face_px:
                    continue
                x0, y0 = max(0, int(x)), max(0, int(y))
                x1, y1 = min(width, int(x + bw)), min(height, int(y + bh))
                crop = image[y0:y1, x0:x1]
                blur = blur_of(crop)
                yaw = yaw_proxy_of(face[4:14].reshape(5, 2))
                embedding = None
                model_name = None
                key = None
                if with_embedding:
                    raw = self._embed_sface(image, face) if self._embedding_model == "sface" else self._embed_auraface(image, face)
                    embedding = self._normalize(raw)
                    model_name = self._embedding_model
                    key = face_key_of(model_name, embedding)
                results.append(
                    DetectedFace(
                        bbox=[int(x), int(y), int(bw), int(bh)],
                        landmarks=[[float(p[0]), float(p[1])] for p in face[4:14].reshape(5, 2)],
                        det_score=det_score,
                        face_px=int(bw),
                        blur=round(blur, 2),
                        frontal=round(1.0 - min(1.0, abs(yaw) / REFERENCE_YAW), 4),
                        composite=round(composite_quality(det_score, int(bw), blur, yaw, self.weights), 4),
                        embedding=None if embedding is None else [round(float(v), 6) for v in embedding],
                        embedding_model=model_name,
                        face_key=key,
                        extra={"yaw_proxy": round(yaw, 4)},
                    )
                )
            results.sort(key=lambda item: item.composite, reverse=True)
            if len(results) > max_faces:
                results = results[:max_faces]
            return results

    def warmup(self) -> None:
        """预热：加载模型并用一张合成图跑一次，避免首个真实请求承担加载时延。"""
        with self._lock:
            self._ensure_loaded()
            self._detect_raw(np.zeros((320, 320, 3), dtype=np.uint8))

    def health(self) -> dict:
        """返回健康信息（不触发模型加载）。"""
        return {
            "embedding_model": self._embedding_model,
            "embedding_dim": self.embedding_dim,
            "loaded": self._detector is not None,
            "min_score": self.min_score,
            "min_face_px": self.min_face_px,
            "models_dir": str(self._models_dir),
        }
