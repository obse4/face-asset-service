#!/usr/bin/env python3
"""容器内推理基准：量化"当好邻居"的 CPU 限制对吞吐的代价。

背景：本机（Apple M4）实测默认模型 AuraFace 为 8.60 帧/秒，但部署到服务器后
（Xeon Platinum 8352V @2.10GHz，容器限 cpus=6 / OMP_NUM_THREADS=4）实测只有 0.95 帧/秒。
除了两代 CPU 的单核性能差，**我们自己的线程/CPU 上限也可能贡献了很大一部分损失**。
这个脚本就是用来把"猜"变成"量"：在不同 --cpus / OMP_NUM_THREADS 组合下跑同一批帧。

用法（在容器内）：
    python3 deploy/benchmark_inference.py --frames-dir /data/face-assets/tmp/acceptance-frames
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from face_service.engine import FaceEngine  # noqa: E402


def main() -> int:
    """跑一次基准并打印结果。"""
    parser = argparse.ArgumentParser(description="容器内推理基准")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--models-dir", default=os.environ.get("FACE_ASSETS_MODELS_DIR", "/srv/face-assets/models"))
    parser.add_argument("--model", default=os.environ.get("FACE_ASSETS_EMBEDDING_MODEL", "auraface"))
    parser.add_argument("--repeat", type=int, default=1, help="重复轮数以摊薄首次开销")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.frames_dir, "*.png")) + glob.glob(os.path.join(args.frames_dir, "*.jpg")))
    if not paths:
        print(f"在 {args.frames_dir} 找不到测试帧", file=sys.stderr)
        return 1
    images = [cv2.imread(path) for path in paths]
    images = [image for image in images if image is not None]

    engine = FaceEngine(
        args.models_dir,
        args.model,
        intra_op_threads=int(os.environ["FACE_ASSETS_THREADS"]) if os.environ.get("FACE_ASSETS_THREADS") else None,
    )
    started = time.perf_counter()
    engine.warmup()
    warmup = time.perf_counter() - started

    best = None
    for round_index in range(args.repeat):
        started = time.perf_counter()
        faces = 0
        for image in images:
            faces += len(engine.analyze(image))
        elapsed = time.perf_counter() - started
        fps = len(images) / elapsed
        print(f"  第 {round_index + 1} 轮：{len(images)} 帧 / {elapsed:.2f}s = {fps:.2f} 帧/秒（{faces} 张脸）")
        best = fps if best is None else max(best, fps)

    print(f"  预热 {warmup:.2f}s | 线程数上限 env OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '未设')} "
          f"| 可见 CPU={os.cpu_count()}")
    print(f"  最佳 {best:.2f} 帧/秒 → 整季 800 帧约 {800 / best:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
