"""face-assets：人物资产提取服务（人脸检测 + 对齐 + embedding + 质量打分）。

许可与选型依据见仓库根目录 `THIRD_PARTY.md` 与 `models.lock`；
部署设计与实测依据见 `deploy/face-assets/PLAN.md`。
"""

from .config import Settings, load_settings
from .engine import DetectedFace, FaceEngine, QualityWeights

__all__ = ["Settings", "load_settings", "FaceEngine", "DetectedFace", "QualityWeights"]
__version__ = "0.1.0"
