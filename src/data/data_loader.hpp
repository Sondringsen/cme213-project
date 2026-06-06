#pragma once
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// Sequential token loader for pre-tokenized binary files (int32 arrays).
// Each call to next_batch() returns the next total_tokens (ids, targets) pairs
// where target[i] = ids[i+1] (next-token prediction). Wraps on end-of-file.
struct DataLoader {
    std::vector<int> tokens;
    int pos = 0;

    explicit DataLoader(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("DataLoader: cannot open " + path);
        f.seekg(0, std::ios::end);
        int n = static_cast<int>(f.tellg()) / static_cast<int>(sizeof(int));
        f.seekg(0);
        tokens.resize(n);
        f.read(reinterpret_cast<char*>(tokens.data()), n * sizeof(int));
    }

    // Fill pre-allocated host buffers. ids[i] = tokens[pos+i],
    // targets[i] = tokens[pos+i+1]. Wraps around the file on overflow.
    void next_batch(int* ids, int* targets, int total_tokens) {
        int n = static_cast<int>(tokens.size());
        for (int i = 0; i < total_tokens; ++i) {
            ids[i]     = tokens[pos % n];
            targets[i] = tokens[(pos + 1) % n];
            ++pos;
        }
        // Keep pos in [0, n-1) so the wrap stays valid
        if (pos >= n - 1) pos = 0;
    }
};
