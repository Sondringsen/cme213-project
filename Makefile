# ---------------------------------------------------------------------------
# Convenience wrapper around CMake.
#
# Usage:
#   make          -- configure (if needed) + build everything
#   make test     -- build + run all CTest tests
#   make clean    -- delete the build directory
#   make refdata  -- generate PyTorch reference data (needs Python + torch)
#
# Override the build directory:
#   make BUILD_DIR=my_build
#
# Override the target GPU architecture (e.g. A100 = 80, RTX 3090 = 86):
#   make CUDA_ARCH=80
# ---------------------------------------------------------------------------

BUILD_DIR  ?= build
CUDA_ARCH  ?=

CMAKE_ARGS := -DCMAKE_BUILD_TYPE=Release
ifneq ($(CUDA_ARCH),)
    CMAKE_ARGS += -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH)
endif

.PHONY: all build test clean refdata

all: build

build:
	cmake -B $(BUILD_DIR) -S . $(CMAKE_ARGS)
	cmake --build $(BUILD_DIR) --parallel

test: build
	cd $(BUILD_DIR) && ctest --output-on-failure

refdata:
	python3 scripts/generate_ref_data.py

clean:
	rm -rf $(BUILD_DIR)
