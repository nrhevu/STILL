#!/usr/bin/env bash
set -euo pipefail

IMAGE="${ROCM_DOCKER_IMAGE:-rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RENDER_GID="${RENDER_GID:-$(getent group render | cut -d: -f3)}"
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
DOCKER_UID="${DOCKER_UID:-$(id -u)}"
DOCKER_GID="${DOCKER_GID:-$(id -g)}"

if [[ -z "${RENDER_GID}" ]]; then
  echo "Could not determine render group gid. Set RENDER_GID explicitly." >&2
  exit 1
fi

exec docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add "${RENDER_GID}" \
  --user "${DOCKER_UID}:${DOCKER_GID}" \
  --ipc=host \
  -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}" \
  -e HOME=/workspace \
  -e USER="${USER:-vunguyen13}" \
  -e LOGNAME="${LOGNAME:-vunguyen13}" \
  -e XDG_CACHE_HOME=/workspace/data/cache \
  -e AMD_COMGR_CACHE_DIR=/workspace/data/comgr_cache \
  -e HF_HOME=/workspace/data/hf_cache \
  -e HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
  -e HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}" \
  -e TORCHINDUCTOR_CACHE_DIR=/workspace/data/torchinductor_cache \
  -e UV_CACHE_DIR=/workspace/data/uv_cache \
  -v "${PROJECT_ROOT}:/workspace" \
  -w /workspace \
  "${IMAGE}" \
  .uv-bootstrap/bin/uv run "$@"
