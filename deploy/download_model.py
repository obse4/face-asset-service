#!/usr/bin/env python3
"""按 models.lock 下载并校验权重。

设计要点（全部来自实际踩坑，不是理论）：

1. **必须校验 size 与 sha256**。实测中 SFace 与 glintr100 首次下载都被中途截断，截断的 ONNX 会让
   OpenCV / ONNX Runtime 报 "Failed to parse ONNX model / Protobuf parsing failed"，
   **看起来像模型不兼容，实际是文件不完整**。不校验哈希就会一路误判成版本问题。
2. **必须支持断点续传**。260MB 的模型在跨网络下载时中断是常态，重头再来很贵。
3. **幂等**：已存在且校验通过的文件直接跳过，便于重复执行与离线重建。
4. 不依赖第三方库（只用标准库），因为部署阶段可能还没有 pip 依赖。

用法：
    python3 download_model.py --lock ../models.lock --dest /srv/face-assets/models
    python3 download_model.py --lock ../models.lock --dest ./models --only glintr100.onnx
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# 下载分块大小
CHUNK_BYTES = 1 << 20


def sha256_of(path: Path) -> str:
    """流式计算文件 sha256，避免把大模型读进内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, entry: dict) -> tuple[bool, str]:
    """校验文件的大小与哈希。

    @param path: 本地文件路径。
    @param entry: models.lock 中的一条模型记录。
    @returns `(是否通过, 说明)`。
    """
    if not path.exists():
        return False, "文件不存在"
    size = path.stat().st_size
    if size != entry["size"]:
        return False, f"大小不符：{size} != {entry['size']}（疑似被截断）"
    digest = sha256_of(path)
    if digest != entry["sha256"]:
        return False, f"sha256 不符：{digest} != {entry['sha256']}"
    return True, "校验通过"


def download(url: str, dest: Path, expected_size: int) -> None:
    """带断点续传的下载；按已下载字节数发 Range 请求。

    只有当已有字节数**严格小于**期望值时才续传。若已有文件已达到（或超过）期望大小，
    续传请求会从文件末尾开始、什么也补不回来，此时必须**从头重下**——这一条是为
    "大小正确但内容损坏"的情形准备的（哈希不符但长度相符时，续传永远修不好）。

    @param url: 下载地址。
    @param dest: 目标文件。
    @param expected_size: 期望字节数；用于判断是否可续传。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    if existing >= expected_size:
        # 无法通过续传修复（含"长度对但内容坏"），只能重下。
        if existing > 0:
            print(f"    已有 {existing} 字节且不小于期望值，无法续传，从头重下")
        dest.unlink(missing_ok=True)
        existing = 0

    request = urllib.request.Request(url, headers={"User-Agent": "dsh-face-assets/0.1"})
    if existing > 0:
        request.add_header("Range", f"bytes={existing}-")
        print(f"    续传：已有 {existing} 字节")

    mode = "ab" if existing > 0 else "wb"
    with urllib.request.urlopen(request, timeout=120) as response, dest.open(mode) as out:
        total = existing
        while True:
            block = response.read(CHUNK_BYTES)
            if not block:
                break
            out.write(block)
            total += len(block)
            percent = 100.0 * total / expected_size if expected_size else 0.0
            print(f"\r    {total}/{expected_size} 字节（{percent:.1f}%）", end="", flush=True)
    print()


def main() -> int:
    """入口。"""
    parser = argparse.ArgumentParser(description="按 models.lock 下载并校验权重")
    parser.add_argument("--lock", required=True, help="models.lock 路径")
    parser.add_argument("--dest", required=True, help="权重落盘目录")
    parser.add_argument("--only", action="append", default=None, help="只处理指定文件名（可重复）")
    parser.add_argument("--attempts", type=int, default=4, help="单个文件的下载尝试次数（默认 4）")
    args = parser.parse_args()

    lock_path = Path(args.lock).resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    dest_dir = Path(args.dest).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for entry in lock["models"]:
        name = entry["name"]
        if args.only and name not in args.only:
            continue

        target = dest_dir / name
        print(f"[{name}]")
        ok, reason = verify(target, entry)
        if ok:
            print(f"    已存在且{reason}，跳过")
            continue

        urls = [entry["url"]] + ([entry["mirror_url"]] if entry.get("mirror_url") else [])
        for attempt in range(1, args.attempts + 1):
            url = urls[(attempt - 1) % len(urls)]
            print(f"    尝试 {attempt}/{args.attempts}：{url}")
            try:
                download(url, target, entry["size"])
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                print(f"    下载中断：{error}")
                continue
            ok, reason = verify(target, entry)
            print(f"    {reason}")
            if ok:
                break
        else:
            failures.append(name)
            print(f"    ✗ {name} 未能取得完整文件；请检查网络或更换镜像地址后重试（支持续传）")
            continue

        if not verify(target, entry)[0]:
            failures.append(name)

    if failures:
        print(f"\n失败 {len(failures)} 项：{', '.join(failures)}", file=sys.stderr)
        return 1

    print(f"\n全部权重已就位并校验通过：{dest_dir}")
    print("许可文件随仓库保存，见 ../THIRD_PARTY.md 与 ../licenses/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
