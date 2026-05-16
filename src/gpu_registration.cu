#include "registration/gpu_registration.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "registration/cuda_check.cuh"

namespace registration {

namespace {

struct DeviceAffine2D {
    float m00;
    float m01;
    float m02;
    float m10;
    float m11;
    float m12;
};

DeviceAffine2D to_device_affine(const Affine2D& matrix) {
    return {matrix.m00, matrix.m01, matrix.m02, matrix.m10, matrix.m11, matrix.m12};
}

cudaTextureAddressMode texture_address_mode(BoundaryMode boundary) {
    switch (boundary) {
        case BoundaryMode::Zero:
            return cudaAddressModeBorder;
        case BoundaryMode::Clamp:
            return cudaAddressModeClamp;
        case BoundaryMode::Wrap:
            return cudaAddressModeWrap;
    }
    return cudaAddressModeBorder;
}

__global__ void affine_texture_kernel(
    cudaTextureObject_t texture,
    float* output,
    int output_width,
    int output_height,
    DeviceAffine2D source_from_dest) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= output_width || y >= output_height) {
        return;
    }

    const float source_x =
        source_from_dest.m00 * static_cast<float>(x) +
        source_from_dest.m01 * static_cast<float>(y) + source_from_dest.m02;
    const float source_y =
        source_from_dest.m10 * static_cast<float>(x) +
        source_from_dest.m11 * static_cast<float>(y) + source_from_dest.m12;

    output[y * output_width + x] = tex2D<float>(texture, source_x + 0.5f, source_y + 0.5f);
}

__inline__ __device__ float warp_reduce_sum(float value) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__inline__ __device__ float block_reduce_sum(float value) {
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;

    value = warp_reduce_sum(value);
    if (lane == 0) {
        warp_sums[warp_id] = value;
    }
    __syncthreads();

    const int warp_count = (blockDim.x + warpSize - 1) / warpSize;
    value = threadIdx.x < warp_count ? warp_sums[lane] : 0.0f;
    if (warp_id == 0) {
        value = warp_reduce_sum(value);
    }
    return value;
}

__global__ void mse_reduce_kernel(
    const float* lhs,
    const float* rhs,
    int count,
    float* block_sums) {
    float thread_sum = 0.0f;
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) {
        const float diff = lhs[i] - rhs[i];
        thread_sum += diff * diff;
    }
    const float block_sum = block_reduce_sum(thread_sum);
    if (threadIdx.x == 0) {
        block_sums[blockIdx.x] = block_sum;
    }
}

__global__ void ncc_reduce_kernel(
    const float* lhs,
    const float* rhs,
    int count,
    float* block_sums) {
    float sum_lhs = 0.0f;
    float sum_rhs = 0.0f;
    float sum_lhs_sq = 0.0f;
    float sum_rhs_sq = 0.0f;
    float sum_cross = 0.0f;

    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) {
        const float a = lhs[i];
        const float b = rhs[i];
        sum_lhs += a;
        sum_rhs += b;
        sum_lhs_sq += a * a;
        sum_rhs_sq += b * b;
        sum_cross += a * b;
    }

    const float reduced_lhs = block_reduce_sum(sum_lhs);
    const float reduced_rhs = block_reduce_sum(sum_rhs);
    const float reduced_lhs_sq = block_reduce_sum(sum_lhs_sq);
    const float reduced_rhs_sq = block_reduce_sum(sum_rhs_sq);
    const float reduced_cross = block_reduce_sum(sum_cross);

    if (threadIdx.x == 0) {
        const int base = blockIdx.x * 5;
        block_sums[base + 0] = reduced_lhs;
        block_sums[base + 1] = reduced_rhs;
        block_sums[base + 2] = reduced_lhs_sq;
        block_sums[base + 3] = reduced_rhs_sq;
        block_sums[base + 4] = reduced_cross;
    }
}

void validate_cuda_warp_inputs(
    const Image& source,
    int output_width,
    int output_height,
    BoundaryMode boundary,
    float fill_value) {
    if (source.empty()) {
        throw std::invalid_argument("source image must not be empty");
    }
    if (source.channels != 1) {
        throw std::invalid_argument("CUDA texture warp currently supports grayscale images");
    }
    if (output_width <= 0 || output_height <= 0) {
        throw std::invalid_argument("output dimensions must be positive");
    }
    if (boundary == BoundaryMode::Zero && fill_value != 0.0f) {
        throw std::invalid_argument("CUDA texture border mode only supports zero fill");
    }
}

void require_cuda_device() {
    std::string reason;
    if (!cuda_device_available(&reason)) {
        throw std::runtime_error("CUDA device unavailable: " + reason);
    }
}

void upload_pair(const Image& lhs, const Image& rhs, float** device_lhs, float** device_rhs) {
    validate_same_shape(lhs, rhs);
    const std::size_t bytes = lhs.pixels.size() * sizeof(float);
    REGISTRATION_CHECK_CUDA(cudaMalloc(device_lhs, bytes));
    REGISTRATION_CHECK_CUDA(cudaMalloc(device_rhs, bytes));
    REGISTRATION_CHECK_CUDA(cudaMemcpy(*device_lhs, lhs.pixels.data(), bytes, cudaMemcpyHostToDevice));
    REGISTRATION_CHECK_CUDA(cudaMemcpy(*device_rhs, rhs.pixels.data(), bytes, cudaMemcpyHostToDevice));
}

int reduction_block_count(std::size_t count) {
    constexpr int block_size = 256;
    const std::size_t needed = (count + block_size - 1) / block_size;
    return static_cast<int>(std::max<std::size_t>(1, std::min<std::size_t>(needed, 1024)));
}

}  // namespace

bool cuda_device_available(std::string* reason) {
    int count = 0;
    const cudaError_t error = cudaGetDeviceCount(&count);
    if (error != cudaSuccess) {
        if (reason != nullptr) {
            *reason = cudaGetErrorString(error);
        }
        cudaGetLastError();
        return false;
    }
    if (count <= 0) {
        if (reason != nullptr) {
            *reason = "no CUDA devices found";
        }
        return false;
    }
    if (reason != nullptr) {
        *reason = "";
    }
    return true;
}

Image warp_affine_cuda(
    const Image& source,
    int output_width,
    int output_height,
    const Affine2D& source_from_dest,
    BoundaryMode boundary,
    float fill_value) {
    validate_cuda_warp_inputs(source, output_width, output_height, boundary, fill_value);

    require_cuda_device();

    cudaArray_t source_array = nullptr;
    cudaTextureObject_t texture = 0;
    float* device_output = nullptr;

    const cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
    REGISTRATION_CHECK_CUDA(cudaMallocArray(
        &source_array,
        &channel_desc,
        static_cast<std::size_t>(source.width),
        static_cast<std::size_t>(source.height)));

    REGISTRATION_CHECK_CUDA(cudaMemcpy2DToArray(
        source_array,
        0,
        0,
        source.pixels.data(),
        static_cast<std::size_t>(source.width) * sizeof(float),
        static_cast<std::size_t>(source.width) * sizeof(float),
        static_cast<std::size_t>(source.height),
        cudaMemcpyHostToDevice));

    cudaResourceDesc resource_desc{};
    resource_desc.resType = cudaResourceTypeArray;
    resource_desc.res.array.array = source_array;

    cudaTextureDesc texture_desc{};
    texture_desc.addressMode[0] = texture_address_mode(boundary);
    texture_desc.addressMode[1] = texture_address_mode(boundary);
    texture_desc.filterMode = cudaFilterModeLinear;
    texture_desc.readMode = cudaReadModeElementType;
    texture_desc.normalizedCoords = 0;

    REGISTRATION_CHECK_CUDA(cudaCreateTextureObject(&texture, &resource_desc, &texture_desc, nullptr));

    const std::size_t output_count = static_cast<std::size_t>(output_width) *
                                     static_cast<std::size_t>(output_height);
    REGISTRATION_CHECK_CUDA(cudaMalloc(&device_output, output_count * sizeof(float)));

    const dim3 block(16, 16);
    const dim3 grid(
        (output_width + block.x - 1) / block.x,
        (output_height + block.y - 1) / block.y);
    affine_texture_kernel<<<grid, block>>>(
        texture,
        device_output,
        output_width,
        output_height,
        to_device_affine(source_from_dest));
    REGISTRATION_CHECK_CUDA(cudaGetLastError());
    REGISTRATION_CHECK_CUDA(cudaDeviceSynchronize());

    Image output(output_width, output_height, 1);
    REGISTRATION_CHECK_CUDA(cudaMemcpy(
        output.pixels.data(),
        device_output,
        output_count * sizeof(float),
        cudaMemcpyDeviceToHost));

    REGISTRATION_CHECK_CUDA(cudaDestroyTextureObject(texture));
    REGISTRATION_CHECK_CUDA(cudaFreeArray(source_array));
    REGISTRATION_CHECK_CUDA(cudaFree(device_output));
    return output;
}

Image warp_affine_cuda(
    const Image& source,
    int output_width,
    int output_height,
    const TransformParams& params,
    BoundaryMode boundary,
    float fill_value) {
    return warp_affine_cuda(
        source,
        output_width,
        output_height,
        source_from_destination(params, source.width, source.height, output_width, output_height),
        boundary,
        fill_value);
}

float mse_cuda(const Image& lhs, const Image& rhs) {
    require_cuda_device();
    validate_same_shape(lhs, rhs);

    float* device_lhs = nullptr;
    float* device_rhs = nullptr;
    float* device_block_sums = nullptr;
    upload_pair(lhs, rhs, &device_lhs, &device_rhs);

    constexpr int block_size = 256;
    const int blocks = reduction_block_count(lhs.pixels.size());
    REGISTRATION_CHECK_CUDA(cudaMalloc(&device_block_sums, static_cast<std::size_t>(blocks) * sizeof(float)));

    mse_reduce_kernel<<<blocks, block_size>>>(
        device_lhs,
        device_rhs,
        static_cast<int>(lhs.pixels.size()),
        device_block_sums);
    REGISTRATION_CHECK_CUDA(cudaGetLastError());
    REGISTRATION_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> block_sums(static_cast<std::size_t>(blocks));
    REGISTRATION_CHECK_CUDA(cudaMemcpy(
        block_sums.data(),
        device_block_sums,
        block_sums.size() * sizeof(float),
        cudaMemcpyDeviceToHost));

    double total = 0.0;
    for (float value : block_sums) {
        total += static_cast<double>(value);
    }

    REGISTRATION_CHECK_CUDA(cudaFree(device_lhs));
    REGISTRATION_CHECK_CUDA(cudaFree(device_rhs));
    REGISTRATION_CHECK_CUDA(cudaFree(device_block_sums));

    return static_cast<float>(total / static_cast<double>(lhs.pixels.size()));
}

float normalized_cross_correlation_cuda(const Image& lhs, const Image& rhs) {
    require_cuda_device();
    validate_same_shape(lhs, rhs);

    float* device_lhs = nullptr;
    float* device_rhs = nullptr;
    float* device_block_sums = nullptr;
    upload_pair(lhs, rhs, &device_lhs, &device_rhs);

    constexpr int block_size = 256;
    const int blocks = reduction_block_count(lhs.pixels.size());
    REGISTRATION_CHECK_CUDA(cudaMalloc(&device_block_sums, static_cast<std::size_t>(blocks) * 5U * sizeof(float)));

    ncc_reduce_kernel<<<blocks, block_size>>>(
        device_lhs,
        device_rhs,
        static_cast<int>(lhs.pixels.size()),
        device_block_sums);
    REGISTRATION_CHECK_CUDA(cudaGetLastError());
    REGISTRATION_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> block_sums(static_cast<std::size_t>(blocks) * 5U);
    REGISTRATION_CHECK_CUDA(cudaMemcpy(
        block_sums.data(),
        device_block_sums,
        block_sums.size() * sizeof(float),
        cudaMemcpyDeviceToHost));

    double sum_lhs = 0.0;
    double sum_rhs = 0.0;
    double sum_lhs_sq = 0.0;
    double sum_rhs_sq = 0.0;
    double sum_cross = 0.0;
    for (int block = 0; block < blocks; ++block) {
        const std::size_t base = static_cast<std::size_t>(block) * 5U;
        sum_lhs += block_sums[base + 0];
        sum_rhs += block_sums[base + 1];
        sum_lhs_sq += block_sums[base + 2];
        sum_rhs_sq += block_sums[base + 3];
        sum_cross += block_sums[base + 4];
    }

    REGISTRATION_CHECK_CUDA(cudaFree(device_lhs));
    REGISTRATION_CHECK_CUDA(cudaFree(device_rhs));
    REGISTRATION_CHECK_CUDA(cudaFree(device_block_sums));

    const double n = static_cast<double>(lhs.pixels.size());
    const double numerator = sum_cross - (sum_lhs * sum_rhs) / n;
    const double lhs_var = sum_lhs_sq - (sum_lhs * sum_lhs) / n;
    const double rhs_var = sum_rhs_sq - (sum_rhs * sum_rhs) / n;
    if (lhs_var <= 0.0 || rhs_var <= 0.0) {
        throw std::invalid_argument("normalized cross-correlation requires non-constant images");
    }

    return static_cast<float>(numerator / std::sqrt(lhs_var * rhs_var));
}

}  // namespace registration
