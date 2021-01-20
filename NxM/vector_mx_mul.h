#include <vector>
#include <numeric>


const int M = 256;
const int N = 2*256 * 1024;

void vector_mx_naive(std::vector<float>& C, std::vector<float> const& vec, std::vector<float> const& mx)
{
    for (int y = 0; y < M; ++y) // sor
    {
        for (int x = 0; x < N; ++x) // oszlop
        {
            C[y] += mx[y * N + x] * vec[x];
        }
    }
}

void vector_mx_alg(std::vector<float>& C, std::vector<float> const& vec, std::vector<float> const& mx)
{
    for (int i = 0; i < M; ++i)
    {
        C[i] = std::inner_product(mx.begin() + (i * N), mx.begin() + ((i + 1) * N), vec.begin(), .0f);
    }
}