#include "run_cuda.cuh"
#include <fstream>

using namespace std;
int main()
{
    std::vector<float> A(N);
    std::vector<float> B(M * N);
    std::vector<float> C0(M);
    std::vector<float> C1(M);
    std::vector<float> C2(M);
    std::vector<float> C3(M);
    std::vector<float> C4(M);

    fill_values(A, B, C0, C1, C2, C3,C4);

    float gpu_1 = do_Cuda(A, B, C2);
    float gpu_2 = do_Cuda_2(A, B, C3);
    float gpu_3 = do_Cuda_3(A, B, C4);

    auto t0 = std::chrono::high_resolution_clock::now();
    vector_mx_naive(C0, A, B);
    auto t1 = std::chrono::high_resolution_clock::now();
    vector_mx_alg(C1, A, B);
    auto t2 = std::chrono::high_resolution_clock::now();
    int check_1 = checker(C0, C1, "C0 vs C1", "CPU Naive", "CPU improved");
    int check_2 = checker(C1, C2, "C0 vs C2", "CPU Improved", "GPU 1");
    int check_3 = checker(C0, C3, "C0 vs C3", "CPU Naive", "GPU 2 ");
    int check_4 = checker(C0, C4, "C0 vs C3", "CPU Naive", "GPU 2 ");


    std::ofstream handler;
    std::ofstream cfg;

    handler.open("data/results.txt", std::ios_base::app);
    cfg.open("data/configs.txt", std::ios_base::app);

    cout << "CPU naive    Computation took: " << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0f << " ms\n";
    cout << "CPU Algorithm Computation took: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f << " ms\n";
    cout << "GPU 1 Computation took: " << gpu_1  << " ms.\n";
    cout << "GPU 2 Computation took: " << gpu_2 << " ms.\n";
    cout << "GPU 3 Computation took: " << gpu_3  << " ms.\n";
    //if (check_1 == 0 && check_2 == 0 && check_3 == 0) { handler << "Solutions are matching" << std::endl; }
    //handler << std::endl;
    //cfg << "The N size were: " << N << ";  The M size were: " << M << std::endl;

    return 0;
}