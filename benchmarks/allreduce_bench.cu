// Measures MPI_Allreduce latency vs message size.
// Outputs one line per (ranks, size) pair:
//   ALLREDUCE: ranks=4 n_bytes=4096 avg_us=12.34
// Run with mpirun -np {1,2,4}; collect output in one log file for plot_comm.py.

#include <mpi.h>
#include <cstdio>
#include <vector>
#include <chrono>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, n_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    // Sizes from 256 B up to 64 MB
    const std::vector<int> sizes_bytes = {
        256, 1024, 4096, 16384, 65536, 262144,
        1048576, 4194304, 16777216, 67108864
    };

    const int WARMUP = 20;
    const int TRIALS = 100;

    for (int n_bytes : sizes_bytes) {
        int n_floats = n_bytes / static_cast<int>(sizeof(float));
        std::vector<float> buf(n_floats, 1.0f);

        for (int i = 0; i < WARMUP; ++i)
            MPI_Allreduce(MPI_IN_PLACE, buf.data(), n_floats,
                          MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < TRIALS; ++i)
            MPI_Allreduce(MPI_IN_PLACE, buf.data(), n_floats,
                          MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = std::chrono::high_resolution_clock::now();

        double avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / TRIALS;

        if (rank == 0) {
            std::printf("ALLREDUCE: ranks=%d n_bytes=%d avg_us=%.2f\n",
                        n_ranks, n_bytes, avg_us);
            std::fflush(stdout);
        }
    }

    MPI_Finalize();
    return 0;
}
