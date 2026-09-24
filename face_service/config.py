"""配置：全部由环境变量驱动，便于容器化与在不同服务器复刻。"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env_flag(name: str, default: bool) -> bool:
    """读取布尔型环境变量（`1/true/yes/on` 视为真）。"""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """服务配置。"""

    models_dir: str
    embedding_model: str
    min_score: float
    min_face_px: int
    max_faces: int
    api_key: str | None
    warmup: bool
    host: str
    port: int
    log_level: str
    # CPU 路径下显式限制线程数，避免与同机其它服务抢满整机 CPU（当好邻居）。
    intra_op_threads: int | None

    @property
    def auth_enabled(self) -> bool:
        """是否启用 Bearer 鉴权。"""
        return bool(self.api_key)


def load_settings() -> Settings:
    """从环境变量装载配置。

    未设 `FACE_ASSETS_API_KEY` 时不启用鉴权（仅适用于可信内网），并在启动日志中明确告警——
    本服务没有其它服务那样的按客户端隔离需求。

    @returns 配置对象。
    """
    return Settings(
        models_dir=os.environ.get("FACE_ASSETS_MODELS_DIR", "/srv/face-assets/models"),
        # 默认 AuraFace：实测 SFace 在本项目素材上的"同一人/不同人"余弦裕度为 **负**（不可分），
        # 而 AuraFace 裕度 +0.378。详见 deploy/face-assets/PLAN.md §4.6。
        embedding_model=os.environ.get("FACE_ASSETS_EMBEDDING_MODEL", "auraface").strip().lower(),
        min_score=float(os.environ.get("FACE_ASSETS_MIN_SCORE", "0.85")),
        min_face_px=int(os.environ.get("FACE_ASSETS_MIN_FACE_PX", "100")),
        max_faces=int(os.environ.get("FACE_ASSETS_MAX_FACES", "32")),
        api_key=os.environ.get("FACE_ASSETS_API_KEY") or None,
        warmup=_env_flag("FACE_ASSETS_WARMUP", True),
        host=os.environ.get("FACE_ASSETS_HOST", "0.0.0.0"),
        port=int(os.environ.get("FACE_ASSETS_PORT", "8000")),
        log_level=os.environ.get("FACE_ASSETS_LOG_LEVEL", "info"),
        intra_op_threads=(
            int(os.environ["FACE_ASSETS_THREADS"]) if os.environ.get("FACE_ASSETS_THREADS") else None
        ),
    )
