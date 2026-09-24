#!/usr/bin/env python3
"""face-assets 服务验收脚本。

验收思路：**同样的输入必须得到与本地已核验结果一致的输出**，而不是只看健康检查。
因此这里用 `episode_008` 的真实镜头起点帧，走完整 HTTP 路径，并复核已知的真值配对：

    同一人·青年：镜头 6 ↔ 3      应高分
    同一人·老人：镜头 22 ↔ 25    应高分
    不同人：      镜头 6 ↔ 22     应低分

本机（Apple M4）实测的基准值：25↔22 = 0.650、25↔6 = 0.178、6↔3 = 0.587。

用法：
    python3 verify_service.py --base-url http://192.168.9.21:8783 --api-key <key> \
        --frames-dir /data/face-assets/tmp/acceptance-frames
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

# 真值配对与期望方向（"high" = 同一人，"low" = 不同人）
GROUND_TRUTH = [
    ("shot_start_006", "shot_start_003", "high", "同一人·青年"),
    ("shot_start_022", "shot_start_025", "high", "同一人·老人"),
    ("shot_start_006", "shot_start_022", "low", "不同人"),
]
# 本地基准余弦（用于确认行为一致，允许一定容差）
BASELINE = {("shot_start_022", "shot_start_025"): 0.650, ("shot_start_006", "shot_start_022"): 0.178}


def post(base_url: str, path: str, payload: dict, api_key: str | None) -> dict:
    """发一个 JSON POST 请求并解析响应。"""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}{path}", data=data, method="POST", headers={"content-type": "application/json"}
    )
    if api_key:
        request.add_header("authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def get(base_url: str, path: str, api_key: str | None = None) -> tuple[int, dict]:
    """发一个 GET 请求，返回状态码与响应体。"""
    request = urllib.request.Request(f"{base_url}{path}")
    if api_key:
        request.add_header("authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:  # type: ignore[attr-defined]
        return error.code, json.loads(error.read().decode("utf-8") or "{}")


def cosine(a: list[float], b: list[float]) -> float:
    """两个向量的余弦相似度（服务端已 L2 归一化，这里仍显式归一化以求稳）。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def main() -> int:
    """执行验收。"""
    parser = argparse.ArgumentParser(description="face-assets 服务验收")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--frames-dir", required=True, help="容器内可见的测试帧目录")
    parser.add_argument("--tolerance", type=float, default=0.02, help="与本地基准的允许偏差")
    args = parser.parse_args()

    failures: list[str] = []
    print(f"验收目标：{args.base_url}")

    # ---- 1) 健康探针 ----
    code, live = get(args.base_url, "/health/live")
    print(f"  /health/live  -> {code} {live}")
    if code != 200:
        failures.append("health/live 非 200")

    code, ready = get(args.base_url, "/health/ready")
    print(f"  /health/ready -> {code} device={ready.get('device')} model={ready.get('embedding_model')} "
          f"dim={ready.get('embedding_dim')} loaded={ready.get('loaded')}")
    if code != 200 or not ready.get("loaded"):
        failures.append("health/ready 未就绪")
    if ready.get("embedding_model") != "auraface":
        failures.append(f"默认模型不是 auraface 而是 {ready.get('embedding_model')}")

    # ---- 2) 鉴权 ----
    try:
        post(args.base_url, "/detect", {"frames": []}, None)
        failures.append("未带令牌竟然通过了 /detect（鉴权未生效）")
        print("  ✗ 鉴权未生效：无令牌也通过了 /detect")
    except Exception as error:  # noqa: BLE001 - 只关心是否被拒
        print(f"  鉴权生效：无令牌被拒（{type(error).__name__}）")

    # ---- 3) 真值配对复核（走 /analyze 完整路径）----
    names = sorted({name for pair in GROUND_TRUTH for name in pair[:2]})
    frames = [{"id": f"{name}_{i}", "path": f"{args.frames_dir}/{name}.png"} for i, name in enumerate(names)]
    started = time.perf_counter()
    result = post(args.base_url, "/analyze", {"frames": frames}, args.api_key)
    elapsed = time.perf_counter() - started
    print(f"  /analyze {len(frames)} 帧 -> {elapsed:.3f}s（{len(frames) / elapsed:.2f} 帧/秒）")

    embeddings: dict[str, list[float]] = {}
    for frame in result["frames"]:
        key = frame["id"].rsplit("_", 1)[0]
        if frame["faces"]:
            best = frame["faces"][0]
            embeddings[key] = best["embedding"]
            print(f"    {key}: 质量={best['quality']['composite']} 宽={best['quality']['face_px']}px "
                  f"dim={len(best['embedding'])}")
        else:
            print(f"    {key}: 未通过质量门控（无脸）")

    for a, b, expect, label in GROUND_TRUTH:
        if a not in embeddings or b not in embeddings:
            failures.append(f"{label} 缺少特征：{a}/{b}")
            continue
        score = cosine(embeddings[a], embeddings[b])
        ok = score > 0.35 if expect == "high" else score < 0.35
        base = BASELINE.get((a, b))
        drift = "" if base is None else f"（本地基准 {base:.3f}，偏差 {abs(score - base):.3f}）"
        print(f"    {label}: {a} ↔ {b} = {score:+.3f} 期望{'高' if expect == 'high' else '低'} "
              f"{'✓' if ok else '✗'} {drift}")
        if not ok:
            failures.append(f"{label} 方向不符：{score:+.3f}")
        if base is not None and abs(score - base) > args.tolerance:
            failures.append(f"{label} 与本地基准偏差 {abs(score - base):.3f} 超过容差 {args.tolerance}")

    # ---- 4) 汇总 ----
    if failures:
        print("\n验收未通过：")
        for item in failures:
            print(f"  ✗ {item}")
        return 1
    print("\n验收全部通过 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
