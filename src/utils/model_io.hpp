#pragma once
// Save and load GPT-2 checkpoints to/from a flat binary file.
//
// File format:
//   [4 bytes]  magic = 0x47505432 ("GPT2")
//   [24 bytes] int32 header: n_layers, n_heads, C, V, S, B_train
//   [params]   all parameter tensors in param_grads() order, row-major FP32
//
// Loading into a model with a different B is fine — weights are B-independent.

#include "model/gpt2.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr uint32_t CKPT_MAGIC = 0x47505432u;

inline void save_model(GPT2& model, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("save_model: cannot write " + path);

    f.write(reinterpret_cast<const char*>(&CKPT_MAGIC), 4);
    int32_t hdr[6] = {
        model.cfg.n_layers, model.cfg.n_heads, model.cfg.C,
        model.cfg.V, model.cfg.S, model.cfg.B
    };
    f.write(reinterpret_cast<char*>(hdr), sizeof(hdr));

    for (auto& p : model.param_grads()) {
        std::vector<float> buf(static_cast<size_t>(p.n));
        cudaMemcpy(buf.data(), p.param,
                   static_cast<size_t>(p.n) * sizeof(float),
                   cudaMemcpyDeviceToHost);
        f.write(reinterpret_cast<char*>(buf.data()),
                static_cast<std::streamsize>(p.n) * sizeof(float));
    }
}

// Read the config header without loading weights — useful to know the model
// dimensions before constructing a GPT2 object.
inline GPT2Config peek_model_config(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("peek_model_config: cannot open " + path);
    uint32_t magic;
    f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != CKPT_MAGIC)
        throw std::runtime_error("peek_model_config: bad magic in " + path);
    int32_t hdr[6];
    f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    GPT2Config cfg;
    cfg.n_layers = hdr[0]; cfg.n_heads = hdr[1]; cfg.C = hdr[2];
    cfg.V = hdr[3]; cfg.S = hdr[4]; cfg.B = hdr[5];
    return cfg;
}

// Load weights from a checkpoint into an already-constructed model.
// The model must have matching n_layers, n_heads, C, V, S.
inline void load_model(GPT2& model, const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("load_model: cannot open " + path);
    uint32_t magic;
    f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != CKPT_MAGIC)
        throw std::runtime_error("load_model: bad magic in " + path);
    int32_t hdr[6];
    f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));

    for (auto& p : model.param_grads()) {
        std::vector<float> buf(static_cast<size_t>(p.n));
        f.read(reinterpret_cast<char*>(buf.data()),
               static_cast<std::streamsize>(p.n) * sizeof(float));
        cudaMemcpy(p.param, buf.data(),
                   static_cast<size_t>(p.n) * sizeof(float),
                   cudaMemcpyHostToDevice);
    }
}
