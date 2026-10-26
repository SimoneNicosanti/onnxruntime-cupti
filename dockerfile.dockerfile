# syntax=docker/dockerfile:1.7
# ONNX Runtime 1.31.0: CUDA 12.9 + CUPTI + NVTX; CPython 3.12, 3.13 and 3.14.
# Linux x86_64 only. BuildKit is required for persistent Ninja/ccache mounts.
# Default target: runtime (three compatible wheels + three installed Python environments).

ARG CUDA_VERSION=12.9.2
ARG CUDA_APT_VERSION=12-9

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS builder
ARG CUDA_APT_VERSION
ARG ORT_VERSION=1.31.0
ARG BUILD_JOBS=12
ARG CUDA_ARCHS="75-real;80-real;86-real;89-real;90-real;100-real;120-real"

ENV DEBIAN_FRONTEND=noninteractive \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    CCACHE_DIR=/root/.cache/ccache \
    CCACHE_MAXSIZE=30G \
    LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /usr/local/bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates git cmake ninja-build ccache python3-minimal \
    curl pkg-config zlib1g-dev \
    cuda-cupti-${CUDA_APT_VERSION} cuda-cupti-dev-${CUDA_APT_VERSION} \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "v${ORT_VERSION}" --recurse-submodules \
    --shallow-submodules https://github.com/microsoft/onnxruntime.git /src/onnxruntime

# Pin to standard, GIL-enabled CPython ABI (cp312/cp313/cp314), not cp313t/cp314t.
# Separate build environments ensure build.py, setup.py, and CMake see the same interpreter.
RUN uv python install 3.12 3.13 3.14 && \
    for version in 3.12 3.13 3.14; do \
      abi="$(printf '%s' "$version" | tr -d '.')"; \
      env_dir="/opt/build-venv-${abi}"; \
      uv venv --managed-python --python "$version" "$env_dir" || exit 1; \
      uv pip install --python "$env_dir/bin/python" \
        pip setuptools wheel packaging psutil 'numpy>=2' \
        -r /src/onnxruntime/tools/ci_build/github/linux/python/requirements.txt || exit 1; \
      "$env_dir/bin/python" -c "import sys; assert sys.version_info[:2] == tuple(map(int, '$version'.split('.')))" || exit 1; \
    done

WORKDIR /src/onnxruntime

# Python 3.12: reuse the cache identifier from the initial single-Python Dockerfile.
# The build tree is mounted, so failed/interrupted Ninja builds are resumable.
RUN --mount=type=cache,id=ort-ccache,target=/root/.cache/ccache,sharing=locked \
    --mount=type=cache,id=ort-build-1.31-cuda12.9,target=/src/onnxruntime/build/Linux,sharing=locked \
    /opt/build-venv-312/bin/python tools/ci_build/build.py \
      --build_dir /src/onnxruntime/build/Linux \
      --allow_running_as_root --skip_submodule_sync \
      --config Release --update --build --build_wheel \
      --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr \
      --enable_cuda_profiling --cmake_generator Ninja \
      --skip_tests --parallel "${BUILD_JOBS}" \
      --nvcc_threads 1 --flash_nvcc_threads 1 --use_cache \
      --cmake_extra_defines \
        "CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}" \
        "onnxruntime_ENABLE_NVTX_PROFILE=ON" \
        "onnxruntime_BUILD_UNIT_TESTS=OFF" \
        "Python_EXECUTABLE=/opt/build-venv-312/bin/python" \
    && mkdir -p /opt/wheels \
    && cp /src/onnxruntime/build/Linux/Release/dist/*-cp312-cp312-*.whl /opt/wheels/ \
    && ccache --show-stats

# Python 3.13: its own Ninja/CMake cache, same compiler cache.
RUN --mount=type=cache,id=ort-ccache,target=/root/.cache/ccache,sharing=locked \
    --mount=type=cache,id=ort-build-1.31-cuda12.9-py313,target=/src/onnxruntime/build/Linux,sharing=locked \
    /opt/build-venv-313/bin/python tools/ci_build/build.py \
      --build_dir /src/onnxruntime/build/Linux \
      --allow_running_as_root --skip_submodule_sync \
      --config Release --update --build --build_wheel \
      --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr \
      --enable_cuda_profiling --cmake_generator Ninja \
      --skip_tests --parallel "${BUILD_JOBS}" \
      --nvcc_threads 1 --flash_nvcc_threads 1 --use_cache \
      --cmake_extra_defines \
        "CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}" \
        "onnxruntime_ENABLE_NVTX_PROFILE=ON" \
        "onnxruntime_BUILD_UNIT_TESTS=OFF" \
        "Python_EXECUTABLE=/opt/build-venv-313/bin/python" \
    && cp /src/onnxruntime/build/Linux/Release/dist/*-cp313-cp313-*.whl /opt/wheels/ \
    && ccache --show-stats

# Python 3.14: preserves its own Ninja/CMake cache and reuses ccache.
RUN --mount=type=cache,id=ort-ccache,target=/root/.cache/ccache,sharing=locked \
    --mount=type=cache,id=ort-build-1.31-cuda12.9-py314,target=/src/onnxruntime/build/Linux,sharing=locked \
    /opt/build-venv-314/bin/python tools/ci_build/build.py \
      --build_dir /src/onnxruntime/build/Linux \
      --allow_running_as_root --skip_submodule_sync \
      --config Release --update --build --build_wheel \
      --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr \
      --enable_cuda_profiling --cmake_generator Ninja \
      --skip_tests --parallel "${BUILD_JOBS}" \
      --nvcc_threads 1 --flash_nvcc_threads 1 --use_cache \
      --cmake_extra_defines \
        "CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}" \
        "onnxruntime_ENABLE_NVTX_PROFILE=ON" \
        "onnxruntime_BUILD_UNIT_TESTS=OFF" \
        "Python_EXECUTABLE=/opt/build-venv-314/bin/python" \
    && cp /src/onnxruntime/build/Linux/Release/dist/*-cp314-cp314-*.whl /opt/wheels/ \
    && ccache --show-stats

# Export-only stage: select with `docker build --target wheels` to avoid runtime installations.
FROM scratch AS wheels
COPY --from=builder /opt/wheels/ /opt/wheels/

# Fully usable image: all three interpreters, three independent ORT environments,
# and the original wheel artifacts. Default python on PATH is 3.14.
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu24.04 AS runtime
ARG CUDA_APT_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH} \
    PATH=/opt/ort-env-3.14/bin:${PATH} \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /usr/local/bin/
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates zlib1g python3-minimal cuda-cupti-${CUDA_APT_VERSION} \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/python/ /opt/python/
COPY --from=wheels /opt/wheels/ /opt/wheels/

# Fail the Docker build if any ABI-specific wheel is absent, incompatible, or fails to import.
RUN set -eu; \
    for version in 3.12 3.13 3.14; do \
      abi="$(printf '%s' "$version" | tr -d '.')"; \
      set -- /opt/wheels/*-cp${abi}-cp${abi}-*.whl; \
      test "$#" -eq 1 && test -f "$1"; \
      uv venv --managed-python --python "$version" "/opt/ort-env-${version}"; \
      uv pip install --python "/opt/ort-env-${version}/bin/python" "$1" onnx; \
      "/opt/ort-env-${version}/bin/python" -c \
        "import sys, onnxruntime as ort; assert sys.version_info[:2] == tuple(map(int, '$version'.split('.'))); assert ort.__version__ == '1.31.0'; assert 'CUDAExecutionProvider' in ort.get_available_providers(); print(sys.version, ort.__version__, ort.get_available_providers())"; \
    done

WORKDIR /workspace
CMD ["bash"]
