"""HTTP API：把 FaceEngine 暴露成服务。

接口设计见 `deploy/face-assets/PLAN.md` §6。要点：

- `/detect`：只检测 + 质量打分（回答"哪些帧有清晰人脸"）。
- `/embed`：对给定人脸（或自动检测）输出 embedding 与 `face_key`。
- `/analyze`：检测 + embedding 一次完成，**客户端应当优先用这个**。
- `face_key` **不是身份**，只是让同一张脸跨集可被认出的候选键；身份判定必须靠锚点图 + 人工确认。
"""
from __future__ import annotations

import base64
import binascii
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import Settings, load_settings
from .engine import FaceEngine

logger = logging.getLogger("face_service")


class FrameRef(BaseModel):
    """一帧的引用：给 `path` 或 `b64` 之一。"""

    id: str = Field(description="调用方自定的帧标识，例如 ep008_f00345")
    path: str | None = Field(default=None, description="服务端可读的图像路径")
    b64: str | None = Field(default=None, description="base64 编码的图像字节")


class DetectRequest(BaseModel):
    """`/detect` 与 `/analyze` 的请求体。"""

    frames: list[FrameRef]
    with_embedding: bool = Field(default=False, description="/analyze 会强制为真")


class EmbedRequest(BaseModel):
    """`/embed` 的请求体。"""

    frames: list[FrameRef]
    max_faces: int | None = None


def decode_frame(ref: FrameRef) -> np.ndarray:
    """把一帧引用解码成 BGR 图像。

    @param ref: 帧引用。
    @returns BGR 图像数组。
    @throws HTTPException 当既未提供 path/b64、路径不可读或图像无法解码时。
    """
    if ref.b64:
        try:
            payload = base64.b64decode(ref.b64, validate=True)
        except (binascii.Error, ValueError) as error:
            raise HTTPException(status_code=400, detail=f"帧 {ref.id} 的 b64 不是合法 base64：{error}") from error
        buffer = np.frombuffer(payload, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    elif ref.path:
        image = cv2.imread(ref.path, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail=f"帧 {ref.id} 的图像无法读取或解码：{ref.path}")
    else:
        raise HTTPException(status_code=400, detail=f"帧 {ref.id} 必须提供 path 或 b64 之一。")
    if image is None:
        raise HTTPException(status_code=400, detail=f"帧 {ref.id} 的图像无法解码。")
    return image


def face_to_dict(face, with_embedding: bool) -> dict:
    """把检测结果转成响应字典。

    @param face: `DetectedFace`。
    @param with_embedding: 是否包含 embedding 字段。
    @returns 响应字典。
    """
    payload = {
        "bbox": face.bbox,
        "landmarks": face.landmarks,
        "det_score": round(face.det_score, 4),
        "quality": {
            "composite": face.composite,
            "face_px": face.face_px,
            "blur": face.blur,
            "frontal": face.frontal,
            "yaw_proxy": face.extra.get("yaw_proxy"),
        },
    }
    if with_embedding:
        payload["embedding"] = face.embedding
        payload["embedding_model"] = face.embedding_model
        payload["face_key"] = face.face_key
    return payload


def create_app(settings: Settings | None = None) -> FastAPI:
    """构造 FastAPI 应用。

    @param settings: 配置；缺省从环境变量装载。
    @returns 应用实例。
    """
    resolved = settings or load_settings()
    engine = FaceEngine(
        models_dir=resolved.models_dir,
        embedding_model=resolved.embedding_model,
        min_score=resolved.min_score,
        min_face_px=resolved.min_face_px,
        intra_op_threads=resolved.intra_op_threads,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """启动时预热模型；关闭时释放。"""
        if not resolved.auth_enabled:
            logger.warning("未设置 FACE_ASSETS_API_KEY：服务不启用鉴权，只应部署在可信内网。")
        logger.info("模型目录=%s 嵌入模型=%s", resolved.models_dir, resolved.embedding_model)
        if resolved.warmup:
            try:
                engine.warmup()
                logger.info("模型预热完成")
            except FileNotFoundError as error:
                # 不因缺权重而拒绝启动：/health/live 仍可用，/health/ready 会报未就绪，
                # 这样运维能区分"进程没起来"与"权重没下好"。
                logger.error("模型预热失败（/health/ready 将报未就绪）：%s", error)
        yield
        engine.unload()

    app = FastAPI(title="face-assets", version="0.1.0", lifespan=lifespan)

    async def require_auth(request: Request) -> None:
        """可选的 Bearer 鉴权。"""
        if not resolved.auth_enabled:
            return
        header = request.headers.get("authorization", "")
        expected = f"Bearer {resolved.api_key}"
        if header != expected:
            raise HTTPException(status_code=401, detail="缺少或错误的 Bearer 令牌。")

    @app.get("/health/live")
    async def health_live() -> dict:
        """存活探针：不触碰模型。"""
        return {"status": "ok"}

    @app.get("/health/ready")
    async def health_ready() -> JSONResponse:
        """就绪探针：报告模型是否已加载。"""
        info = engine.health()
        payload = {"device": "cpu", "models": [info["embedding_model"]], **info}
        return JSONResponse(status_code=200 if info["loaded"] else 503, content=payload)

    @app.get("/api/runtime", dependencies=[Depends(require_auth)])
    async def runtime() -> dict:
        """运行时信息。"""
        return {
            "engine": engine.health(),
            "limits": {
                "max_faces": resolved.max_faces,
                "min_score": resolved.min_score,
                "min_face_px": resolved.min_face_px,
            },
            "auth": resolved.auth_enabled,
        }

    @app.post("/detect", dependencies=[Depends(require_auth)])
    async def detect(request: DetectRequest) -> dict:
        """检测 + 质量打分，不含 embedding。"""
        return _run(engine, request.frames, with_embedding=False, max_faces=resolved.max_faces)

    @app.post("/analyze", dependencies=[Depends(require_auth)])
    async def analyze(request: DetectRequest) -> dict:
        """检测 + 质量打分 + embedding（推荐入口）。"""
        return _run(engine, request.frames, with_embedding=True, max_faces=resolved.max_faces)

    @app.post("/embed", dependencies=[Depends(require_auth)])
    async def embed(request: EmbedRequest) -> dict:
        """embedding 专用入口（同样会做检测与质量门控）。"""
        return _run(engine, request.frames, with_embedding=True, max_faces=request.max_faces or resolved.max_faces)

    return app


def _run(engine: FaceEngine, frames: list[FrameRef], with_embedding: bool, max_faces: int) -> dict:
    """对一批帧执行检测/嵌入。

    @param engine: 引擎。
    @param frames: 帧列表。
    @param with_embedding: 是否计算 embedding。
    @param max_faces: 单帧最多返回人脸数。
    @returns 响应字典。
    """
    results = []
    for ref in frames:
        image = decode_frame(ref)
        height, width = image.shape[:2]
        faces = engine.analyze(image, with_embedding=with_embedding, max_faces=max_faces)
        results.append(
            {
                "id": ref.id,
                "w": width,
                "h": height,
                "faces": [face_to_dict(face, with_embedding) for face in faces],
            }
        )
    return {
        "embedding_model": engine.embedding_model if with_embedding else None,
        "embedding_dim": engine.embedding_dim if with_embedding else None,
        "frame_count": len(results),
        "frames": results,
    }


def main() -> None:
    """以 uvicorn 启动服务（容器与本地都走这里）。"""
    import uvicorn

    settings = load_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level=settings.log_level)


if __name__ == "__main__":
    main()
