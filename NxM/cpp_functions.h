#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


static const int MBS = 16;//matrix block size
const int block_sz = MBS;
const int n_blocks = M / MBS;
const float max_err = 1e-5f;


void fill_values(std::vector<float>& A, std::vector<float>& B, std::vector<float>& C0, std::vector<float>& C1, std::vector<float>& C2,std::vector<float>& C3)
{
    // Random number generator
    std::mt19937 mersenne_engine{ 42 };  // Generates random integers
    std::uniform_real_distribution<float> dist{ -0.1f, 0.1f };

    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
    // Fill A and B with random numbers
    generate(A.begin(), A.end(), gen);
    generate(B.begin(), B.end(), gen);

    //Fill C0,C1,C2 with 0
    std::fill(C0.begin(), C0.end(), 0.0f);
    std::fill(C1.begin(), C1.end(), 0.0f);
    std::fill(C2.begin(), C2.end(), 0.0f);
    std::fill(C3.begin(), C3.end(), 0.0f);
}


int checker(std::vector<float>  ref_1, std::vector<float> ref_2, std::string err_name, std::string T1, std::string T2)
{
    auto comparator = [](float l, float r) { return std::abs(l - r) < max_err; };

    for (int i = 0; i < M; ++i)
    {
        if (!comparator(ref_1[i], ref_2[i]))
        {
            std::cout << err_name << "[" << i << "] : " << ref_1[i] << "   " << ref_2[i] << " absolute error: " << std::abs(ref_1[i] - ref_2[i]) << "\n";
            return -1;
        }
    }

    if (std::equal(ref_1.begin(), ref_1.end(), ref_2.begin(), comparator))
    {
        std::cout << T1 << " matches " << T2 << "\n";
    }
    else
    {
        std::cout << "Mismatch in the " << T1 << " and " << T2 << "\n";
        return -1;
    }
    return 0;
}
