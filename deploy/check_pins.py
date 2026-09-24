#!/usr/bin/env python3
"""核对 requirements 里每个 pin 在目标平台上是否真有可安装的 wheel。

为什么需要它：本次部署第一次构建失败，原因是我把 `opencv-contrib-python-headless`
的版本号写成了 conda 的 `cv2.__version__`（`4.13.0`），而 pip 的 wheel 版本是
`4.13.0.92` 这种四段式——**pypi 上根本没有 `4.13.0.88`**。这个脚本把"猜 pin"变成"查 pin"。

用法：python3 check_pins.py            # 默认检查 deploy/requirements-api.txt
      python3 check_pins.py <文件>
"""
from __future__ import annotations

import json
import sys
import urllib.request

PY_TAG = "cp311"  # 目标基础镜像 python:3.11-slim
PLATFORM_TAG = "x86_64"
# 目标系统是 Linux：必须限定 manylinux，否则会把 macOS 的 x86_64 wheel 误判为可用。
LINUX_TAG = "manylinux"
# 稳定 ABI：`cp37-abi3` 这类 wheel 可在 3.7+ 上安装（opencv 正是这种），
# 因此不能只找 `cp311`；这是本脚本第一版把 opencv 误判为"装不了"的原因。
ABI3_TAG = "abi3"


def versions_for(package: str, version: str) -> tuple[bool, str]:
    """查询某个 pin 是否有可安装的 wheel。

    @param package: 包名。
    @param version: 版本号。
    @returns `(是否可用, 说明)`。
    """
    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/{version}/json", timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as error:  # noqa: BLE001 - 任何失败都视为不可用并报出原因
        return False, f"查询失败：{error}"

    wheels = [item["filename"] for item in payload["urls"] if item["filename"].endswith(".whl")]
    # 纯 Python wheel 处处可用
    pure = [name for name in wheels if name.endswith("py3-none-any.whl")]
    # Linux 原生 wheel：匹配版本标签（cp311）或稳定 ABI（abi3），且必须是 manylinux
    native = [
        name
        for name in wheels
        if LINUX_TAG in name
        and PLATFORM_TAG in name
        and (PY_TAG in name or ABI3_TAG in name)
    ]
    if native:
        return True, f"Linux 原生 wheel ×{len(native)}（例 {native[0]}）"
    if pure:
        return True, f"纯 Python wheel ×{len(pure)}（例 {pure[0]}）"
    return False, f"无可用 wheel；该版本 wheels：{wheels[:3]}"


def parse_requirements(path: str) -> list[tuple[str, str]]:
    """从 requirements 文件里解析 `pkg==ver` 形式的 pin。"""
    pins: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            text = line.split("#", 1)[0].strip()
            if not text or "==" not in text:
                continue
            name, version = text.split("==", 1)
            pins.append((name.strip(), version.strip()))
    return pins


def main() -> int:
    """入口。"""
    path = sys.argv[1] if len(sys.argv) > 1 else "deploy/requirements-api.txt"
    pins = parse_requirements(path)
    if not pins:
        print(f"{path} 里没有找到 pkg==ver 形式的 pin")
        return 1

    print(f"检查 {path}（目标平台 {PY_TAG}/{PLATFORM_TAG}）")
    failures = 0
    for name, version in pins:
        ok, detail = versions_for(name, version)
        mark = "OK " if ok else "NO "
        print(f"  {mark} {name}=={version}  {detail}")
        if not ok:
            failures += 1

    if failures:
        print(f"\n{failures} 个 pin 在目标平台上装不了，必须改正")
        return 1
    print("\n全部 pin 在目标平台上可用")
    return 0


if __name__ == "__main__":
    sys.exit(main())
