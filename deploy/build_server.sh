#!/usr/bin/env bash
# 构建 face-assets 服务镜像。
#
# 为什么不用 --build-arg HTTP_PROXY：实测该机 **pypi.org 可直连**（200），
# 而 mihomo 代理当前返回 502 不可用。因此不做代理假设，直接从 pypi 安装。
# 基础镜像用本机已存在的 python:3.11-slim，避免拉取（可用 BASE_IMAGE 覆盖）。
set -euo pipefail
cd /opt/face-assets
mkdir -p deploy/verification

docker buildx build --load \
  -f deploy/Dockerfile.server \
  --build-arg "BASE_IMAGE=${BASE_IMAGE:-python:3.11-slim}" \
  -t face-assets:local . > deploy/verification/build.log 2>&1

docker image inspect face-assets:local --format '{{.Id}}' > deploy/verification/image-id.txt
printf 'face-assets 镜像已构建：%s\n' "$(cat deploy/verification/image-id.txt)"
