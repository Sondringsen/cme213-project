# CME 213 Final Project — Small LLM from Scratch

A small GPT-2-style transformer language model built from the ground up in C++/CUDA, distributed across multiple GPUs with MPI. The project implements the core parallel computing primitives behind modern LLM training: hand-written tiled GEMM kernels, Flash Attention (forward and backward), fused LayerNorm/GELU operations, mixed-precision (BF16/FP32) arithmetic, and Ring All-Reduce gradient synchronization. The goal is not a state-of-the-art model, but a thorough, hands-on understanding of what makes large-scale neural network training fast.

**Course:** CME 213, Spring 2026  
**Authors:** Nils Astrup Toft and Sondre Rogde
