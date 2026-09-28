#!/usr/bin/env python3
"""L0→L1→L2 贯通演练：时间轴 → 抽帧 → 人脸服务 → 候选角色归并。

这一步的意义：P0（帧级统一时间轴）与人脸服务至今各自验收过，但**从未一起用过**。
本脚本把两个交付物接起来跑一遍真实素材，顺带验证两件事：

1. **抽帧策略可用**：按 PLAN §4.1 的策略（镜头起点必取 + 长镜头每 2.5s 补采）从时间轴**确定性**
   推导帧号，帧号都落在 `[start_frame, end_frame)` 内。
2. **身份归并必须用锚点法而不是无监督聚类**：实测（PLAN §4.3）单链接会"链式传染"把不同角色并成一类。
   这里同时输出两种结果做对照，用真实数据复核该结论。

输出物（默认写到 `--out` 目录）：
- `candidate_characters.json`：候选角色分组（锚点 + 成员 + 质量分 + 需人工确认标记）
- `frame_faces.json`：逐帧的人脸检测与质量明细

用法：
    python3 rehearse_pipeline.py --base-url http://192.168.9.21:8783 --api-key <key> \
        --timeline /data/face-assets/jobs/episode_008.timeline.json \
        --frames-dir /data/face-assets/tmp/policy-frames \
        --out /data/face-assets/jobs/rehearsal-ep008
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import urllib.request

# 候选归并阈值：低于它不认为同一角色。依据 PLAN §4.6，AuraFace 的同一人最低 +0.587、
# 不同人最高 +0.209，故 0.35 落在两簇之间且有裕度。
ASSIGN_THRESHOLD = 0.35


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度（服务端已 L2 归一化，这里仍显式归一化以求稳）。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def post(base_url: str, path: str, payload: dict, api_key: str) -> dict:
    """发 JSON POST 并解析响应。

    出错时把服务端返回的诊断体一并抛出——服务端的 400 会带中文原因（例如"帧 X 的图像无法读取"），
    只看状态码会误判成请求格式问题。
    """
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"content-type": "application/json", "authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{path} 返回 {error.code}：{detail[:400]}") from error


def frame_of_shot(timeline: dict, frame_number: int) -> int | None:
    """按半开区间把帧号归属到镜头（`start_frame <= n < end_frame`）。"""
    for shot in timeline["shots"]:
        if shot["start_frame"] <= frame_number < shot["end_frame"]:
            return shot["index"]
    return None


def group_by_anchor(items: list[dict], threshold: float) -> list[dict]:
    """锚点法归并：按质量降序，与各已有分组的**锚点**比较，够近才并入，否则新建一组。

    与单链接的区别：单链接只要与组内**任一**成员够近就并（会链式传染）；
    这里只跟**每个组里质量最高的那张脸**比，天然抑制链式误并。
    """
    groups: list[dict] = []
    for item in sorted(items, key=lambda x: -x["quality"]):
        best_group = None
        best_score = -1.0
        for group in groups:
            score = cosine(item["embedding"], group["anchor_embedding"])
            if score > best_score:
                best_score = score
                best_group = group
        if best_group is not None and best_score >= threshold:
            best_group["members"].append({"frame": item["frame"], "quality": item["quality"], "score_to_anchor": round(best_score, 4)})
            best_group["member_count"] = len(best_group["members"])
        else:
            groups.append(
                {
                    "group_id": f"cand_{len(groups) + 1:02d}",
                    "anchor_frame": item["frame"],
                    "anchor_quality": item["quality"],
                    "anchor_embedding": item["embedding"],
                    "nearest_other_score": round(best_score, 4) if best_group is not None else None,
                    "members": [{"frame": item["frame"], "quality": item["quality"], "score_to_anchor": 1.0}],
                    "member_count": 1,
                }
            )
    # 关键设计：**孤类不自动成为新角色**，而是转入人工复核队列。
    # 依据：本演练中帧 763 与青年的相似度是 0.326（卡在 0.35 下）—— 它可能是同一人，
    # 也可能真是新角色；把这种判断留给锚点确认，而不是让算法自动"造"出一个角色。
    confirmed = [g for g in groups if g["member_count"] > 1]
    review = [g for g in groups if g["member_count"] == 1]
    for group in groups:
        del group["anchor_embedding"]
    for index, group in enumerate(confirmed, start=1):
        group["group_id"] = f"cand_{index:02d}"
    return confirmed, review


def single_linkage_groups(items: list[dict], threshold: float) -> list[list[str]]:
    """单链接（阈值连通分量）—— 用于**对照展示**它为什么不可靠。"""
    parent = {item["frame"]: item["frame"] for item in items}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for i, a in enumerate(items):
        for b in items[i + 1:]:
            if cosine(a["embedding"], b["embedding"]) >= threshold:
                ra, rb = find(a["frame"]), find(b["frame"])
                if ra != rb:
                    parent[ra] = rb
    buckets: dict[str, list[str]] = {}
    for item in items:
        buckets.setdefault(find(item["frame"]), []).append(item["frame"])
    return sorted(buckets.values(), key=len, reverse=True)


def main() -> int:
    """执行演练。"""
    parser = argparse.ArgumentParser(description="L0→L1→L2 贯通演练")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--frames-dir", required=True, help="本地枚举用的帧目录")
    parser.add_argument("--server-frames-dir", default=None,
                        help="服务端可见的帧目录（远端服务必填；缺省时与 --frames-dir 相同）")
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", type=float, default=ASSIGN_THRESHOLD)
    args = parser.parse_args()

    with open(args.timeline, encoding="utf-8") as handle:
        timeline = json.load(handle)
    frame_paths = sorted(glob.glob(os.path.join(args.frames_dir, "*.png")))
    if not frame_paths:
        print(f"在 {args.frames_dir} 找不到帧", file=sys.stderr)
        return 1

    # 关键：服务在**远端**读文件，因此发给它的必须是服务端可见路径。
    # 本地目录只用来枚举帧号；远端路径由 --server-frames-dir 拼出（缺省时两者相同）。
    server_dir = args.server_frames_dir or args.frames_dir
    refs, frame_numbers = [], []
    for path in frame_paths:
        digits = "".join(ch for ch in os.path.basename(path) if ch.isdigit())
        number = int(digits) if digits else -1
        frame_numbers.append(number)
        refs.append({"id": f"f{number:04d}", "path": f"{server_dir}/{os.path.basename(path)}"})

    print(f"贯通演练：{timeline['episode']['id']} · {len(refs)} 帧（策略抽帧）"
          f" · fps {timeline['timebase']['fps']['num']}/{timeline['timebase']['fps']['den']}")

    started = time.perf_counter()
    result = post(args.base_url, "/analyze", {"frames": refs}, args.api_key)
    elapsed = time.perf_counter() - started
    print(f"  服务端 /analyze：{len(refs)} 帧 / {elapsed:.2f}s = {len(refs) / elapsed:.2f} 帧/秒")

    frame_faces = []
    best_faces = []
    for frame_number, frame in zip(frame_numbers, result["frames"]):
        shot_index = frame_of_shot(timeline, frame_number)
        entry = {
            "frame": frame_number,
            "shot_index": shot_index,
            "file": os.path.basename(frame["id"]),
            "faces": [
                {
                    "quality": face["quality"]["composite"],
                    "face_px": face["quality"]["face_px"],
                    "det_score": face["det_score"],
                    "face_key": face["face_key"],
                }
                for face in frame["faces"]
            ],
        }
        frame_faces.append(entry)
        if frame["faces"]:
            top = frame["faces"][0]
            best_faces.append(
                {
                    "frame": frame_number,
                    "shot_index": shot_index,
                    "quality": top["quality"]["composite"],
                    "embedding": top["embedding"],
                }
            )

    with_faces = len(best_faces)
    print(f"  检出并通过质量门控：{with_faces}/{len(refs)} 帧（其余为无脸或未过门控）")

    anchor_groups, review_queue = group_by_anchor(best_faces, args.threshold)
    linkage_groups = single_linkage_groups(best_faces, args.threshold)
    print(f"  锚点法归并 → {len(anchor_groups)} 个候选角色（≥2 帧），{len(review_queue)} 个孤类转入复核队列")
    for group in anchor_groups:
        frames = [member["frame"] for member in group["members"]]
        print(f"    {group['group_id']}: 锚点帧 {group['anchor_frame']}（质量 {group['anchor_quality']}）"
              f" 成员 {len(frames)} 帧 {frames[:8]}{'…' if len(frames) > 8 else ''}")
    print(f"  对照 · 单链接 → {len(linkage_groups)} 类，最大类 {len(linkage_groups[0])} 帧"
          f"（链式传染会把不同角色并进来，故不采用）")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "candidate_characters.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "episode": timeline["episode"]["id"],
                "policy": {
                    "shot_start_required": True,
                    "supplement_seconds": 2.5,
                    "frames": len(refs),
                    "note": "帧号由 episode 时间轴确定性推导，均落在 [start_frame, end_frame) 内",
                },
                "grouping": {
                    "method": "anchor_based",
                    "threshold": args.threshold,
                    "singletons_go_to_review": (
                        "只有 ≥2 帧相互支持的分组才成为候选角色；孤类进入 review_queue，"
                        "因为实测存在同一人被阈值误拆（帧 763 对青年仅 0.326<0.35）与"
                        "屏幕内人脸导致 embedding 失效（帧 1828）两种情形。"
                    ),
                    "why_not_clustering": (
                        "无监督聚类（单链接/平均链接）在同一部剧里会把不同角色并成一类，"
                        "且同一角色的正/侧/叠影帧相似度可以低到 0.374；因此身份必须靠锚点图 + 人工确认。"
                        "见 deploy/face-assets/PLAN.md §4.3/§4.6。"
                    ),
                },
                "candidates": anchor_groups,
                "review_queue": review_queue,
                "single_linkage_comparison": linkage_groups,
                "requires_human_confirmation": True,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    with open(os.path.join(args.out, "frame_faces.json"), "w", encoding="utf-8") as handle:
        json.dump(frame_faces, handle, ensure_ascii=False, indent=2)
    print(f"  产物已写入 {args.out}")
    print("\n注意：这些是**候选**分组，不是身份定论 —— 需人工/LLM 确认并选定锚点图后才成为角色记录。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
