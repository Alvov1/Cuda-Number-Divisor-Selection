#include <iostream>
#include <filesystem>
#include <thrust/device_vector.h>
#include <thrust/device_malloc.h>
#include "mpz/mpz.h"

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

__global__ void findDivisor(mpz_t n, const uint64_t* primes, unsigned* resultPlace) {
    const unsigned threadIndex = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned threads = gridDim.x * blockDim.x;
    const unsigned blockIndex = blockIdx.x;
    // unsigned i = blockIdx.x * blockDim.x;

    const unsigned max_it = 400000 / threads;

    const unsigned b_start = 2 + blockIndex;//blockIndex * blockDim.x / max_it;
    const unsigned b_inc = gridDim.x;//threads / max_it;

    unsigned B;
    const unsigned B_MAX = 2000000000;
    unsigned it;
    unsigned p_i;
    unsigned power;
    unsigned prime_ul;

    mpz_t a, d, e, b, tmp;

    mpz_init(&a);
    mpz_init(&d);
    mpz_init(&e);
    mpz_init(&b);
    mpz_init(&tmp);

    mpz_set_ui(&a, (unsigned long) threadIndex * max_it + 2);

    for (B = b_start; B < B_MAX; B += b_inc) {
        prime_ul = (unsigned long) primes[0];
        mpz_set_lui(&e, (unsigned long) 1);
        for (p_i = 0; prime_ul < B; p_i ++) {
            if (*resultPlace) return;

            power = (unsigned) (log((double) B) / log((double) prime_ul));
            mpz_mult_u(&tmp, &e, (unsigned) pow((double) prime_ul, (double) power));

            if (*resultPlace) return;

            mpz_set(&e, &tmp);
            prime_ul = primes[p_i + 1];
        }

        if (mpz_equal_one(&e)) continue;
        if (*resultPlace) return;

        for (it = 0; it < max_it; it ++) {

            if (*resultPlace) return;


            mpz_gcd(&d, &a, &n);                // gcd = gcd(a, n)

            if (mpz_gt_one(&d)) {               // if d > 1
                char buffer[1024] {};
                mpz_get_str(&d, buffer, 1024);
                printf("Found: %s\n", buffer);

                atomicAdd(resultPlace, it);
            }
            if (*resultPlace) return;

            mpz_powmod(&b, &a, &e, &n);         // b = (a ** e) % n
            mpz_addeq_i(&b, -1);                // b -= 1
            mpz_gcd(&d, &b, &n);                // d = gcd(tmp, n)

            if (*resultPlace) return;

            // success!
            if (mpz_gt_one(&d) && mpz_lt(&d, &n)) {
                char buffer[1024] {};
                mpz_get_str(&d, buffer, 1024);
                printf("Found: %s\n", buffer);

                atomicAdd(resultPlace, it);
            }

            mpz_addeq_i(&a, threads * max_it);              // a += 1
        }
    }
}

std::ostream& operator<<(std::ostream& stream, const mpz_t& number) {
    char buffer[1024] {};
    mpz_get_str(&number, buffer, 1024);
    return stream << buffer;
}

int main(int argc, const char* const* argv) {
    try {
        if(argc < 3)
            throw std::invalid_argument("Usage: <Number to factorize> <Prime table location>.");

        /* Factorizing number. */
        const mpz_t number = [&argv] {
            mpz_t number {}; mpz_init(&number); mpz_set_str(&number, argv[1]); return number;
        } ();
        std::cout << "Factorizing number 0x" << number << std::endl;

        /* Prime table. */
        const std::filesystem::path primeTableLocation = argv[2];
        const thrust::device_vector<uint64_t> primes = loadPrimeTable(primeTableLocation);
        std::cout << "Prime table prepared and loaded to device." << std::endl;

        /* Place for result. */
        thrust::device_ptr<unsigned> resultFlag = thrust::device_malloc<unsigned>(1);
        std::cout << "Data prepared and loaded." << std::endl;

        findDivisor<<<512, 512>>>(
                number,
                thrust::raw_pointer_cast(primes.data()),
                thrust::raw_pointer_cast(resultFlag));
        if(cudaSuccess != cudaDeviceSynchronize())
            throw std::runtime_error("Kernel launch failed.");
        std::cout << "Kernel launch completed." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}