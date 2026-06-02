# Step 6 — WikiText-2 Training (Optional)

**Priority:** Low. Only attempt if Steps 1-5 are complete before June 5.  
**Estimated time:** 3-4 hours  
**Fallback:** If not implemented, describe as future work in the report (1 paragraph).

---

## Why It Matters (and Why It's Optional)

Training on a real dataset validates that the system is a complete language model training pipeline, not just a benchmark harness. It also lets us show a perplexity curve (loss vs. time), which is a natural deliverable for an LLM project.

However, the graders care most about the performance analysis. A sentence or two explaining why we used synthetic data (to isolate performance measurement from data loading overhead, and because the profiling results are hardware-bound regardless of the data) is a perfectly acceptable substitute.

**Decision rule:** If the data pipeline takes longer than 3 hours to implement correctly, skip it and add the future-work paragraph instead.

---

## Data Pipeline Plan

### Step 1: Download and tokenize (Python)

Script: `scripts/prepare_data.py`

```python
#!/usr/bin/env python3
"""
Download WikiText-2 and produce a binary file of int32 token IDs.
Uses simple word-level tokenization (lowercase, split on whitespace/punctuation).
Output: data/wikitext2_{train,val,test}.bin -- arrays of int32
        data/vocab.txt -- one token per line
"""
import os, re, struct, urllib.request

DATA_URL = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/"
FILES = ["train.txt", "valid.txt", "test.txt"]
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s'<>]", " ", text)
    return text.split()

# Download
for f in FILES:
    path = os.path.join(OUT_DIR, f)
    if not os.path.exists(path):
        print(f"Downloading {f}...")
        urllib.request.urlretrieve(DATA_URL + f, path)

# Build vocabulary from training set only
print("Building vocabulary...")
with open(os.path.join(OUT_DIR, "train.txt")) as f:
    train_tokens = tokenize(f.read())

# Keep top V-2 tokens, add <unk> and <eos>
from collections import Counter
counts = Counter(train_tokens)
VOCAB_SIZE = 10000  # smaller vocab for speed
vocab = ["<unk>", "<eos>"] + [w for w, _ in counts.most_common(VOCAB_SIZE - 2)]
word2id = {w: i for i, w in enumerate(vocab)}

with open(os.path.join(OUT_DIR, "vocab.txt"), "w") as f:
    f.write("\n".join(vocab))
print(f"Vocabulary size: {len(vocab)}")

# Tokenize and write binary
def encode_file(src, dst):
    with open(src) as f:
        tokens = tokenize(f.read())
    ids = [word2id.get(t, 0) for t in tokens] + [word2id["<eos>"]]
    with open(dst, "wb") as f:
        f.write(struct.pack(f"{len(ids)}i", *ids))
    print(f"Wrote {len(ids)} tokens to {dst}")

for src_name, dst_name in [("train.txt", "train"), ("valid.txt", "val"), ("test.txt", "test")]:
    encode_file(os.path.join(OUT_DIR, src_name),
                os.path.join(OUT_DIR, f"wikitext2_{dst_name}.bin"))
```

Output:
- `data/wikitext2_train.bin` — ~2M int32 tokens
- `data/wikitext2_val.bin` — ~200K tokens
- `data/vocab.txt` — 10,000 words

### Step 2: C++ data loader

File: `src/data/data_loader.hpp`

```cpp
#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>

// Simple sequential sampler: reads the token file and produces
// (input_ids, target_ids) pairs from non-overlapping windows of length S.
struct DataLoader {
    std::vector<int> tokens;
    int S;           // sequence length
    int B;           // batch size
    int pos;         // current position in tokens

    DataLoader(const std::string& path, int seq_len, int batch_size)
        : S(seq_len), B(batch_size), pos(0)
    {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot open data file: " + path);
        f.seekg(0, std::ios::end);
        int n = f.tellg() / sizeof(int);
        f.seekg(0);
        tokens.resize(n);
        f.read(reinterpret_cast<char*>(tokens.data()), n * sizeof(int));
    }

    // Fill pre-allocated host buffers with the next B×S batch.
    // Returns false and wraps around if we hit the end.
    bool next_batch(std::vector<int>& ids, std::vector<int>& targets) {
        ids.resize(B * S);
        targets.resize(B * S);
        for (int b = 0; b < B; b++) {
            for (int s = 0; s < S; s++) {
                int idx = (pos + b * S + s) % (tokens.size() - 1);
                ids[b * S + s]     = tokens[idx];
                targets[b * S + s] = tokens[idx + 1];
            }
        }
        pos = (pos + B * S) % (tokens.size() - 1);
        return true;
    }
};
```

### Step 3: Integrate into main_distributed.cu

Modify `main_distributed.cu` to optionally read from a data file:

```cpp
// After CLI args:
std::string data_path = (argc > 6) ? argv[6] : "";
int V = data_path.empty() ? 512 : 10000;  // synthetic vs real vocab

// In training loop, replace gen_batch with:
if (!data_path.empty()) {
    loader.next_batch(h_local_ids, h_local_tgts);
} else {
    gen_batch(h_full_ids.data(), ...);
    scatter_batch(...);
}
```

### Step 4: Run training and plot loss curve

```bash
# Run on cluster
mpirun -np 1 ./build/train_distributed 200 6 512 256 16 data/wikitext2_train.bin
```

Expected: loss should start at ~log(10000) ≈ 9.2 (random) and decrease. After 200 steps at small model size it won't converge far, but a decreasing trend demonstrates correctness.

Plot: `scripts/plot_loss.py`

```python
import matplotlib.pyplot as plt

steps, losses = [], []
with open("logs/wikitext_training.log") as f:
    for line in f:
        if "step" in line and "loss" in line:
            # Parse: "step  N | loss X.XXXX | ..."
            parts = line.split("|")
            step = int(parts[0].split()[1])
            loss = float(parts[1].split()[1])
            steps.append(step)
            losses.append(loss)

plt.figure(figsize=(6, 3))
plt.plot(steps, losses)
plt.xlabel("Training step")
plt.ylabel("Cross-entropy loss")
plt.title("WikiText-2 training loss (6L, C=512)")
plt.axhline(y=9.21, color='gray', linestyle='--', label='Random baseline (log V)')
plt.legend()
plt.tight_layout()
plt.savefig("plots/loss_curve.png", dpi=150)
```

---

## Fallback: Report Paragraph (If Not Implemented)

If time runs out, add this paragraph to the Discussion section:

> "Our training loop is validated on synthetic random token sequences (V=512) which are sufficient for measuring throughput and scaling. We designed the data pipeline interface (DataLoader) to accept any binary file of int32 token IDs, allowing a drop-in replacement with WikiText-2. We omit real-data training from this report: the performance analysis (roofline, scaling, α+βn) is hardware-bound and independent of the actual tokens processed. Extending to WikiText-2 would require implementing weight tying (embedding and lm_head share parameters) and a learning rate schedule; we leave this as future work."

---

## BF16 Discussion (No Code Needed)

Whether or not WikiText-2 training is implemented, add this analysis to the report:

### Why BF16 was not implemented

Our hardware is the Quadro RTX 6000, which is Turing architecture (sm_75). Turing has:
- **FP16 tensor cores**: 2× throughput vs FP32 on 4×4×4 matrix ops
- **No BF16 tensor cores**: BF16 on Turing runs on regular CUDA cores at FP32 throughput

BF16 would give 2× memory bandwidth savings (weights stored as 16-bit) but no compute speedup on Turing. The correct strategy on Turing for mixed precision is FP16 compute (via cuBLAS tensor cores) with FP32 accumulation — but this requires the weight matrices to be multiples of 8 in all dimensions (tensor core alignment constraint).

Our model dimension C=256 is divisible by 8, so FP16 tensor cores would work. However, implementing correct FP16 GEMM with FP32 accumulation in a custom kernel requires careful handling of the `__half` type and is architecturally complex. We leave this for future work on Ampere+ hardware where BF16 tensor cores are available and BF16 becomes the preferred mixed-precision format.

**Key analysis point**: On A100 (Ampere), BF16 tensor cores give 312 TFLOPS vs 77 TFLOPS FP32. This is a 4× compute speedup *and* a 2× memory saving, making BF16 overwhelmingly preferred. On our RTX 6000, the speedup is 0× compute and 2× memory — marginal for a compute-bound kernel like GEMM.

---

## Checklist

- [ ] (Optional) Run `scripts/prepare_data.py` to download and tokenize WikiText-2
- [ ] (Optional) Implement `src/data/data_loader.hpp`
- [ ] (Optional) Modify `main_distributed.cu` to accept data path argument
- [ ] (Optional) Run 200-step training, plot loss curve
- [ ] Write BF16 analysis paragraph (always — no code needed)
- [ ] If not implemented: write fallback paragraph for Discussion section
