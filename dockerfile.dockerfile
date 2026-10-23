# syntax=docker/dockerfile:1.7
# ONNX Runtime CUDA + CUPTI + NVTX, compiled for multiple NVIDIA GPU families.
# Default: CUDA 12.9 (Ada + Blackwell, also Turing/Ampere/Hopper), Python 3.12.
# Target: Linux amd64. Requires NVIDIA Container Toolkit at runtime.

ARG CUDA_VERSION=12.9.2
ARG CUDA_APT_VERSION=12-9

# -------------------------- Builder --------------------------
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS builder

ARG CUDA_APT_VERSION
ARG ORT_VERSION=1.31.0
ARG BUILD_JOBS=4
# Native cubins for common SM versions; PTX for forward compatibility.
ARG CUDA_ARCHS="75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual"

ENV DEBIAN_FRONTEND=noninteractive \
    CCACHE_DIR=/root/.cache/ccache \
    CCACHE_MAXSIZE=30G \
    LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates git cmake ninja-build ccache \
    python3.12-dev python3.12-venv python3-pip \
    curl pkg-config zlib1g-dev \
    cuda-cupti-${CUDA_APT_VERSION} cuda-cupti-dev-${CUDA_APT_VERSION} \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "v${ORT_VERSION}" --recurse-submodules \
    --shallow-submodules https://github.com/microsoft/onnxruntime.git /src/onnxruntime

RUN python3 -m venv /opt/build-venv && \
    /opt/build-venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel packaging psutil && \
    /opt/build-venv/bin/pip install --no-cache-dir \
    -r /src/onnxruntime/tools/ci_build/github/linux/python/requirements.txt

ENV PATH=/opt/build-venv/bin:${PATH}
WORKDIR /src/onnxruntime

# BuildKit cache mounts persist across interrupted builds on the same builder.
# Persist both ccache outputs and the entire Ninja/CMake build tree.
# An interrupted RUN can then resume Ninja's incremental build on the next invocation.
RUN --mount=type=cache,id=ort-ccache,target=/root/.cache/ccache,sharing=locked \
    --mount=type=cache,id=ort-build-1.31-cuda12.9,target=/src/onnxruntime/build/Linux,sharing=locked \
    python tools/ci_build/build.py \
    --build_dir /src/onnxruntime/build/Linux \
    --allow_running_as_root \
    --skip_submodule_sync \
    --config Release \
    --update --build --build_wheel \
    --use_cuda \
    --cuda_home /usr/local/cuda \
    --cudnn_home /usr \
    --enable_cuda_profiling \
    --cmake_generator Ninja \
    --skip_tests \
    --parallel "${BUILD_JOBS}" \
    --nvcc_threads 1 \
    --flash_nvcc_threads 1 \
    --use_cache \
    --cmake_extra_defines \
      "CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}" \
      "onnxruntime_ENABLE_NVTX_PROFILE=ON" \
      "onnxruntime_BUILD_UNIT_TESTS=OFF" && \
    mkdir -p /opt/wheels && \
    cp /src/onnxruntime/build/Linux/Release/dist/*.whl /opt/wheels/ && \
    ccache --show-stats

# -------------------------- Runtime --------------------------
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu24.04 AS runtime

ARG CUDA_APT_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH} \
    PATH=/opt/ort-env/bin:${PATH} \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12-venv python3-pip ca-certificates zlib1g \
    cuda-cupti-${CUDA_APT_VERSION} \
    && rm -rf /var/lib/apt/lists/*

# Keep the wheel in the image for reuse, e.g. in other project environments.
COPY --from=builder /opt/wheels/*.whl /opt/wheels/

RUN python3 -m venv /opt/ort-env && \
    /opt/ort-env/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/ort-env/bin/pip install --no-cache-dir /opt/wheels/*.whl onnx && \
    python -c "import onnxruntime as ort; print('ORT:', ort.__version__, 'Providers:', ort.get_available_providers()); assert 'CUDAExecutionProvider' in ort.get_available_providers()"

WORKDIR /workspace
CMD ["bash"]
