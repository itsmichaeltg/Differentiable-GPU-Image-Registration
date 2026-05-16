#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#define REGISTRATION_CHECK_CUDA(ans) registration::gpu_assert((ans), __FILE__, __LINE__)

namespace registration {

inline void gpu_assert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(code) << " at "
                  << file << ":" << line << std::endl;
        std::exit(1);
    }
}

}  // namespace registration
