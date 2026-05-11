#pragma once

// ---------------------------------------------------------------------------
// Tensor<T>
//
// A minimal owning wrapper around a contiguous device buffer. Holds the raw
// pointer, the shape (row-major), and not much else. Anything fancier
// (views, strides, broadcasting, autograd) is intentionally out of scope --
// the goal is to stop us juggling raw cudaMalloc/cudaFree pairs and to get
// RAII for free.
//
// Conventions:
//   - Layout is row-major. The last dimension is contiguous in memory.
//   - The tensor *owns* its buffer; copy is disabled, move transfers
//     ownership.
//   - `data()` returns a device pointer. Don't dereference it from host code.
// ---------------------------------------------------------------------------

#include "utils/cuda_check.hpp"

#include <cuda_runtime.h>
#include <cstddef>
#include <functional>
#include <numeric>
#include <utility>
#include <vector>

template <typename T>
class Tensor {
public:
    // Construct an uninitialized tensor of the given shape on the device.
    // The buffer is *not* zeroed; call zero() afterwards if you need that.
    explicit Tensor(std::vector<int> shape)
        : shape_(std::move(shape)),
          numel_(std::accumulate(shape_.begin(), shape_.end(),
                                 static_cast<size_t>(1),
                                 std::multiplies<size_t>())) {
        CUDA_CHECK(cudaMalloc(&d_data_, numel_ * sizeof(T)));
    }

    ~Tensor() {
        // We swallow the return value here on purpose: throwing from a
        // destructor is worse than leaking on shutdown.
        if (d_data_) cudaFree(d_data_);
    }

    // Non-copyable: copying a Tensor would silently allocate and duplicate
    // possibly gigabytes of GPU memory. Force the caller to be explicit.
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Movable: transfers ownership of the device buffer.
    Tensor(Tensor&& other) noexcept
        : d_data_(other.d_data_),
          shape_(std::move(other.shape_)),
          numel_(other.numel_) {
        other.d_data_ = nullptr;
        other.numel_ = 0;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            if (d_data_) cudaFree(d_data_);
            d_data_ = other.d_data_;
            shape_ = std::move(other.shape_);
            numel_ = other.numel_;
            other.d_data_ = nullptr;
            other.numel_ = 0;
        }
        return *this;
    }

    // Raw device pointer access. Kernels take these.
    T*       data()       { return d_data_; }
    const T* data() const { return d_data_; }

    size_t numel()      const { return numel_; }
    size_t size_bytes() const { return numel_ * sizeof(T); }
    const std::vector<int>& shape() const { return shape_; }
    int dim(int i) const { return shape_.at(i); }

    // Bulk host -> device copy. `host_ptr` must point to numel() elements.
    void copy_from_host(const T* host_ptr) {
        CUDA_CHECK(cudaMemcpy(d_data_, host_ptr, size_bytes(),
                              cudaMemcpyHostToDevice));
    }

    // Bulk device -> host copy. `host_ptr` must point to numel() elements.
    void copy_to_host(T* host_ptr) const {
        CUDA_CHECK(cudaMemcpy(host_ptr, d_data_, size_bytes(),
                              cudaMemcpyDeviceToHost));
    }

    // Zero the device buffer in place.
    void zero() {
        CUDA_CHECK(cudaMemset(d_data_, 0, size_bytes()));
    }

private:
    T*               d_data_ = nullptr;
    std::vector<int> shape_;
    size_t           numel_ = 0;
};
