#!/usr/bin/env python3
"""
Download WikiText-2 and produce binary files of int32 token IDs.
Simple word-level tokenization: lowercase, strip non-alphanumeric (keep ' < >).

Output:
  data/wikitext2_train.bin  -- ~2M int32 tokens
  data/wikitext2_val.bin    -- ~200K tokens
  data/wikitext2_test.bin   -- ~200K tokens
  data/vocab.txt            -- one token per line (10 000 words)
"""

import os
import re
import struct
import urllib.request
from collections import Counter

DATA_URL = ("https://raw.githubusercontent.com/pytorch/examples/main/"
            "word_language_model/data/wikitext-2/")
FILES    = ["train.txt", "valid.txt", "test.txt"]
OUT_DIR  = "data"
os.makedirs(OUT_DIR, exist_ok=True)

VOCAB_SIZE = 10_000


def tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s'<>]", " ", text)
    return text.split()


# ── Download ────────────────────────────────────────────────────────────────
for fname in FILES:
    dst = os.path.join(OUT_DIR, fname)
    if not os.path.exists(dst):
        print(f"Downloading {fname}...")
        urllib.request.urlretrieve(DATA_URL + fname, dst)
    else:
        print(f"Already exists: {dst}")

# ── Build vocab from training set ───────────────────────────────────────────
print("Building vocabulary...")
with open(os.path.join(OUT_DIR, "train.txt")) as f:
    train_tokens = tokenize(f.read())

counts = Counter(train_tokens)
# Reserve index 0 for <unk>, 1 for <eos>
vocab   = ["<unk>", "<eos>"] + [w for w, _ in counts.most_common(VOCAB_SIZE - 2)]
word2id = {w: i for i, w in enumerate(vocab)}

with open(os.path.join(OUT_DIR, "vocab.txt"), "w") as f:
    f.write("\n".join(vocab))
print(f"Vocabulary size: {len(vocab)}")


# ── Tokenize and write binary ────────────────────────────────────────────────
def encode_file(src: str, dst: str):
    with open(src) as f:
        tokens = tokenize(f.read())
    # Replace every <eos> line marker and map unknowns to 0
    ids = [word2id.get(t, 0) for t in tokens] + [word2id["<eos>"]]
    with open(dst, "wb") as f:
        f.write(struct.pack(f"{len(ids)}i", *ids))
    print(f"Wrote {len(ids):,} tokens → {dst}")


encode_file(os.path.join(OUT_DIR, "train.txt"),
            os.path.join(OUT_DIR, "wikitext2_train.bin"))
encode_file(os.path.join(OUT_DIR, "valid.txt"),
            os.path.join(OUT_DIR, "wikitext2_val.bin"))
encode_file(os.path.join(OUT_DIR, "test.txt"),
            os.path.join(OUT_DIR, "wikitext2_test.bin"))
print("Done.")
