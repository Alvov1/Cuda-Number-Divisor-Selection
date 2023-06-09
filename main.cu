#include <iostream>
#include <filesystem>
#include <thrust/device_vector.h>
#include "Camrary/Singles/src_gpu/multi_prec.h"

using Unsigned = multi_prec<2>;

std::vector<uint64_t> loadPrimeTable(const std::filesystem::path &fromLocation) {
    if (!std::filesystem::exists(fromLocation))
        throw std::invalid_argument("Prime table location is not found.");
    FILE *input = fopen(fromLocation.string().c_str(), "r");
    if (input == nullptr)
        throw std::runtime_error("Failed to open prime table file.");

    unsigned primeTableSize{};
    if (1 != fscanf(input, "%u\n", &primeTableSize))
        throw std::runtime_error("Failed to read prime table size.");

    std::vector <uint64_t> primes(primeTableSize);
    for (auto &prime: primes)
        if (1 != fscanf(input, "%llu\n", &prime))
            throw std::runtime_error("Failed to read prime number.");

    if (1 == fclose(input))
        throw std::runtime_error("Failed to close the input file.");

    return primes;
}

__global__ void findDivisor(const char* stringNumber, const uint64_t* primes, unsigned primesNumber, uint8_t* resultPlace) {
    const unsigned threadNumber = threadIdx.x + blockIdx.x * blockDim.x, blockNumber = blockIdx.x;
    if(threadNumber > 0) return;
    const unsigned threadsTotal = gridDim.x * blockDim.x, maxIt = 400000 / threadsTotal;
    Unsigned numberToFactorize = stringNumber, a = threadNumber * maxIt + 2;

    for(unsigned B = 2 + blockNumber; B < primesNumber; B += gridDim.x) {
        
    }
}

int main(int argc, const char* const* argv) {
    try {
        if(argc < 3)
            throw std::invalid_argument("Usage: <Number to factorize> <Prime table location>.");

        /* Prime table. */
        const std::filesystem::path primeTableLocation = argv[2];
        const thrust::device_vector<uint64_t> primes = loadPrimeTable(primeTableLocation);
        /* Factorizing number. */
        const thrust::device_vector<char> numberToFactorize = [&argv] {
            const std::string stringNumber = argv[1];

            thrust::device_vector<char> stringNumberGPU = std::vector<char>(stringNumber.begin(), stringNumber.end());
            stringNumberGPU.push_back('\0');

            return stringNumberGPU;
        } ();
        /* Place for result. */
        thrust::device_ptr<uint8_t> resultFlag {};

        findDivisor<<<16, 16>>>(
                thrust::raw_pointer_cast(numberToFactorize.data()),
                thrust::raw_pointer_cast(primes.data()),
                primes.size(),
                thrust::raw_pointer_cast(resultFlag));
        if(cudaSuccess != cudaDeviceSynchronize())
            throw std::runtime_error("Kernel launch failed.");

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}