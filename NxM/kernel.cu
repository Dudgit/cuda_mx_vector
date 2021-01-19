#include "run_cuda.cuh"
#include <fstream>

int main()
{
    std::vector<float> A(N);
    std::vector<float> B(M * N);
    std::vector<float> C0(M);
    std::vector<float> C1(M);
    std::vector<float> C2(M);
    fill_values(A, B, C0, C1, C2);
    float cuda_run_time = do_Cuda(A, B, C2);

    auto t0 = std::chrono::high_resolution_clock::now();
    vector_mx_naive(C0, A, B);
    auto t1 = std::chrono::high_resolution_clock::now();
    vector_mx_alg(C1, A, B);
    auto t2 = std::chrono::high_resolution_clock::now();
    checker(C0, C1, "C0 vs C1", "CPU Naive", "CPU improved");
    checker(C1, C2, "C0 vs C2", "CPU Improved", "GPU Naive");

    std::ofstream handler;
    std::ofstream cfg;

    handler.open("data/results.txt", std::ios_base::app);
    cfg.open("data/configs.txt", std::ios_base::app);
    handler << "CPU naive    Computation took: " << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0f << " ms\n";
    handler << "CPU Algorithm Computation took: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f << " ms\n";
    handler << "GPU improved Computation took: " << cuda_run_time << " ms.\n";
    handler << std::endl;
    cfg << "The N size were: " << N << ";  The M size were: " << M << std::endl;

    return 0;
}
