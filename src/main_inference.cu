// main_inference.cu — Load a saved GPT-2 checkpoint and generate text.
//
// Usage:
//   ./build/run_inference <checkpoint> <vocab.txt> "<prompt>" [n_tokens=50]
//
// The prompt is tokenized the same way as prepare_data.py:
//   lowercase → strip non-[a-z0-9 ' < >] → split on whitespace
// Unknown words are mapped to <unk> (index 0).

#include "model/gpt2.hpp"
#include "utils/cuda_check.hpp"
#include "utils/model_io.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

static std::vector<std::string> load_vocab(const std::string& path) {
    std::vector<std::string> vocab;
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "Cannot open vocab: %s\n", path.c_str());
        std::exit(1);
    }
    std::string line;
    while (std::getline(f, line)) vocab.push_back(line);
    return vocab;
}

// Mirrors the tokenizer in prepare_data.py so the same vocab applies.
static std::vector<int> tokenize_prompt(
    const std::string& text,
    const std::unordered_map<std::string, int>& word2id)
{
    std::string cleaned;
    cleaned.reserve(text.size());
    for (unsigned char c : text) {
        char lc = static_cast<char>(std::tolower(c));
        if (std::isalnum(static_cast<unsigned char>(lc)) ||
            lc == ' ' || lc == '\'' || lc == '<' || lc == '>' ||
            std::isspace(static_cast<unsigned char>(lc)))
            cleaned += lc;
        else
            cleaned += ' ';
    }

    std::istringstream ss(cleaned);
    std::string tok;
    std::vector<int> ids;
    while (ss >> tok) {
        auto it = word2id.find(tok);
        ids.push_back(it != word2id.end() ? it->second : 0); // 0 = <unk>
    }
    return ids;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "Usage: %s <checkpoint> <vocab.txt> \"<prompt>\" [n_tokens=50]\n",
            argv[0]);
        return 1;
    }
    const std::string ckpt_path  = argv[1];
    const std::string vocab_path = argv[2];
    const std::string prompt_str = argv[3];
    int n_gen = (argc > 4) ? std::atoi(argv[4]) : 50;

    // ── Load vocab ──────────────────────────────────────────────────────────
    auto vocab = load_vocab(vocab_path);
    std::unordered_map<std::string, int> word2id;
    word2id.reserve(vocab.size());
    for (int i = 0; i < static_cast<int>(vocab.size()); ++i)
        word2id[vocab[i]] = i;

    // ── Read checkpoint header to get model dimensions ──────────────────────
    GPT2Config cfg = peek_model_config(ckpt_path);
    cfg.B = 1; // inference always runs with batch size 1

    std::printf("Model: layers=%d heads=%d C=%d V=%d S=%d\n",
                cfg.n_layers, cfg.n_heads, cfg.C, cfg.V, cfg.S);

    // ── Build model and load weights ────────────────────────────────────────
    CUDA_CHECK(cudaSetDevice(0));
    GPT2 model(cfg);
    load_model(model, ckpt_path);
    std::printf("Loaded checkpoint: %s\n", ckpt_path.c_str());

    // ── Tokenize prompt ──────────────────────────────────────────────────────
    std::vector<int> context = tokenize_prompt(prompt_str, word2id);
    if (context.empty()) context.push_back(1); // <eos> as fallback

    std::printf("\nPrompt (%zu tokens): ", context.size());
    for (int id : context)
        std::printf("%s ", vocab[static_cast<size_t>(id)].c_str());
    std::printf("\n\nGenerated:\n");
    for (int id : context)
        std::printf("%s ", vocab[static_cast<size_t>(id)].c_str());

    // ── Device buffer: one sequence of length S ──────────────────────────────
    const int S = cfg.S;
    const int V = cfg.V;
    std::vector<int>   h_ids(S, 0);
    std::vector<float> h_logits(static_cast<size_t>(S) * V);
    int* d_ids = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ids, S * sizeof(int)));

    // ── Autoregressive generation loop ───────────────────────────────────────
    for (int step = 0; step < n_gen; ++step) {
        int ctx_len = static_cast<int>(context.size());

        // Fill context window with the last S tokens
        int start = std::max(0, ctx_len - S);
        std::fill(h_ids.begin(), h_ids.end(), 0);
        for (int i = 0; i < S && (start + i) < ctx_len; ++i)
            h_ids[i] = context[static_cast<size_t>(start + i)];

        CUDA_CHECK(cudaMemcpy(d_ids, h_ids.data(), S * sizeof(int),
                              cudaMemcpyHostToDevice));

        model.forward(d_ids);
        CUDA_CHECK(cudaDeviceSynchronize());

        model.logits.copy_to_host(h_logits.data());

        // Logit row for the last non-padding position
        int last_pos = std::min(ctx_len, S) - 1;
        const float* row = h_logits.data() + static_cast<size_t>(last_pos) * V;

        // Greedy argmax
        int next_tok = static_cast<int>(std::max_element(row, row + V) - row);

        context.push_back(next_tok);
        std::printf("%s ", vocab[static_cast<size_t>(next_tok)].c_str());
        std::fflush(stdout);

        if (next_tok == 1) break; // <eos>
    }
    std::printf("\n");

    CUDA_CHECK(cudaFree(d_ids));
    return 0;
}
