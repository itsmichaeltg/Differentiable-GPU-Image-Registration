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

struct DeviceTransformParams {
    float theta;
    float scale;
    float tx;
    float ty;
};

DeviceAffine2D to_device_affine(const Affine2D& matrix) {
    return {matrix.m00, matrix.m01, matrix.m02, matrix.m10, matrix.m11, matrix.m12};
}

DeviceTransformParams to_device_params(const TransformParams& params) {
    return {params.theta, params.scale, params.tx, params.ty};
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

__device__ int clamp_index_device(int value, int upper_exclusive) {
    return min(max(value, 0), upper_exclusive - 1);
}

__device__ int wrap_index_device(int value, int upper_exclusive) {
    int wrapped = value % upper_exclusive;
    if (wrapped < 0) {
        wrapped += upper_exclusive;
    }
    return wrapped;
}

__device__ float global_pixel_or_boundary(
    const float* source,
    int width,
    int height,
    int x,
    int y,
    int boundary,
    float fill_value) {
    if (boundary == static_cast<int>(BoundaryMode::Zero)) {
        if (x < 0 || x >= width || y < 0 || y >= height) {
            return fill_value;
        }
        return source[y * width + x];
    }
    if (boundary == static_cast<int>(BoundaryMode::Clamp)) {
        const int cx = clamp_index_device(x, width);
        const int cy = clamp_index_device(y, height);
        return source[cy * width + cx];
    }
    const int wx = wrap_index_device(x, width);
    const int wy = wrap_index_device(y, height);
    return source[wy * width + wx];
}

__device__ float sample_global_bilinear(
    const float* source,
    int width,
    int height,
    float x,
    float y,
    int boundary,
    float fill_value) {
    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);

    const float p00 = global_pixel_or_boundary(source, width, height, x0, y0, boundary, fill_value);
    const float p10 = global_pixel_or_boundary(source, width, height, x1, y0, boundary, fill_value);
    const float p01 = global_pixel_or_boundary(source, width, height, x0, y1, boundary, fill_value);
    const float p11 = global_pixel_or_boundary(source, width, height, x1, y1, boundary, fill_value);
    const float top = p00 * (1.0f - tx) + p10 * tx;
    const float bottom = p01 * (1.0f - tx) + p11 * tx;
    return top * (1.0f - ty) + bottom * ty;
}

__global__ void affine_global_kernel(
    const float* source,
    float* output,
    int source_width,
    int source_height,
    int output_width,
    int output_height,
    DeviceAffine2D source_from_dest,
    int boundary,
    float fill_value) {
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

    output[y * output_width + x] = sample_global_bilinear(
        source,
        source_width,
        source_height,
        source_x,
        source_y,
        boundary,
        fill_value);
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

__global__ void loss_gradient_texture_kernel(
    cudaTextureObject_t source_texture,
    const float* target,
    int source_width,
    int source_height,
    int target_width,
    int target_height,
    DeviceTransformParams params,
    float* block_sums) {
    const float source_cx = 0.5f * static_cast<float>(source_width - 1);
    const float source_cy = 0.5f * static_cast<float>(source_height - 1);
    const float dest_cx = 0.5f * static_cast<float>(target_width - 1);
    const float dest_cy = 0.5f * static_cast<float>(target_height - 1);
    const float cos_theta = cosf(params.theta);
    const float sin_theta = sinf(params.theta);
    const float inv_count = 1.0f / static_cast<float>(target_width * target_height);

    float loss = 0.0f;
    float grad_theta = 0.0f;
    float grad_scale = 0.0f;
    float grad_tx = 0.0f;
    float grad_ty = 0.0f;

    const int count = target_width * target_height;
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) {
        const int x = i % target_width;
        const int y = i / target_width;
        const float u = static_cast<float>(x) - dest_cx;
        const float v = static_cast<float>(y) - dest_cy;
        const float rotated_x = cos_theta * u - sin_theta * v;
        const float rotated_y = sin_theta * u + cos_theta * v;
        const float source_x = source_cx + params.tx + params.scale * rotated_x;
        const float source_y = source_cy + params.ty + params.scale * rotated_y;

        const float warped = tex2D<float>(source_texture, source_x + 0.5f, source_y + 0.5f);
        const float error = warped - target[i];
        const float image_dx = 0.5f * (
            tex2D<float>(source_texture, source_x + 1.5f, source_y + 0.5f) -
            tex2D<float>(source_texture, source_x - 0.5f, source_y + 0.5f));
        const float image_dy = 0.5f * (
            tex2D<float>(source_texture, source_x + 0.5f, source_y + 1.5f) -
            tex2D<float>(source_texture, source_x + 0.5f, source_y - 0.5f));
        const float dtheta_x = params.scale * (-sin_theta * u - cos_theta * v);
        const float dtheta_y = params.scale * (cos_theta * u - sin_theta * v);
        const float factor = 2.0f * error * inv_count;

        loss += error * error;
        grad_theta += factor * (image_dx * dtheta_x + image_dy * dtheta_y);
        grad_scale += factor * (image_dx * rotated_x + image_dy * rotated_y);
        grad_tx += factor * image_dx;
        grad_ty += factor * image_dy;
    }

    const float reduced_loss = block_reduce_sum(loss);
    const float reduced_theta = block_reduce_sum(grad_theta);
    const float reduced_scale = block_reduce_sum(grad_scale);
    const float reduced_tx = block_reduce_sum(grad_tx);
    const float reduced_ty = block_reduce_sum(grad_ty);

    if (threadIdx.x == 0) {
        const int base = blockIdx.x * 5;
        block_sums[base + 0] = reduced_loss;
        block_sums[base + 1] = reduced_theta;
        block_sums[base + 2] = reduced_scale;
        block_sums[base + 3] = reduced_tx;
        block_sums[base + 4] = reduced_ty;
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

void validate_cuda_gradient_inputs(const Image& source, const Image& target) {
    validate_same_shape(source, target);
    if (source.channels != 1) {
        throw std::invalid_argument("CUDA loss/gradient currently supports grayscale images");
    }
}

void require_cuda_device() {
    std::string reason;
    if (!cuda_device_available(&reason)) {
        throw std::runtime_error("CUDA device unavailable: " + reason);
    }
}

cudaTextureObject_t create_float_texture(
    const Image& source,
    BoundaryMode boundary,
    cudaArray_t* source_array) {
    const cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
    REGISTRATION_CHECK_CUDA(cudaMallocArray(
        source_array,
        &channel_desc,
        static_cast<std::size_t>(source.width),
        static_cast<std::size_t>(source.height)));

    REGISTRATION_CHECK_CUDA(cudaMemcpy2DToArray(
        *source_array,
        0,
        0,
        source.pixels.data(),
        static_cast<std::size_t>(source.width) * sizeof(float),
        static_cast<std::size_t>(source.width) * sizeof(float),
        static_cast<std::size_t>(source.height),
        cudaMemcpyHostToDevice));

    cudaResourceDesc resource_desc{};
    resource_desc.resType = cudaResourceTypeArray;
    resource_desc.res.array.array = *source_array;

    cudaTextureDesc texture_desc{};
    texture_desc.addressMode[0] = texture_address_mode(boundary);
    texture_desc.addressMode[1] = texture_address_mode(boundary);
    texture_desc.filterMode = cudaFilterModeLinear;
    texture_desc.readMode = cudaReadModeElementType;
    texture_desc.normalizedCoords = 0;

    cudaTextureObject_t texture = 0;
    REGISTRATION_CHECK_CUDA(cudaCreateTextureObject(&texture, &resource_desc, &texture_desc, nullptr));
    return texture;
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
    float* device_output = nullptr;

    const cudaTextureObject_t texture = create_float_texture(source, boundary, &source_array);

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

Image warp_affine_cuda_global(
    const Image& source,
    int output_width,
    int output_height,
    const Affine2D& source_from_dest,
    BoundaryMode boundary,
    float fill_value) {
    validate_cuda_warp_inputs(source, output_width, output_height, boundary, fill_value);
    require_cuda_device();

    float* device_source = nullptr;
    float* device_output = nullptr;
    const std::size_t source_bytes = source.pixels.size() * sizeof(float);
    const std::size_t output_count = static_cast<std::size_t>(output_width) *
                                     static_cast<std::size_t>(output_height);
    REGISTRATION_CHECK_CUDA(cudaMalloc(&device_source, source_bytes));
    REGISTRATION_CHECK_CUDA(cudaMalloc(&device_output, output_count * sizeof(float)));
    REGISTRATION_CHECK_CUDA(cudaMemcpy(device_source, source.pixels.data(), source_bytes, cudaMemcpyHostToDevice));

    const dim3 block(16, 16);
    const dim3 grid(
        (output_width + block.x - 1) / block.x,
        (output_height + block.y - 1) / block.y);
    affine_global_kernel<<<grid, block>>>(
        device_source,
        device_output,
        source.width,
        source.height,
        output_width,
        output_height,
        to_device_affine(source_from_dest),
        static_cast<int>(boundary),
        fill_value);
    REGISTRATION_CHECK_CUDA(cudaGetLastError());
    REGISTRATION_CHECK_CUDA(cudaDeviceSynchronize());

    Image output(output_width, output_height, 1);
    REGISTRATION_CHECK_CUDA(cudaMemcpy(
        output.pixels.data(),
        device_output,
        output_count * sizeof(float),
        cudaMemcpyDeviceToHost));

    REGISTRATION_CHECK_CUDA(cudaFree(device_source));
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

LossGradient loss_gradient_cuda(
    const Image& source,
    const Image& target,
    const TransformParams& params,
    BoundaryMode boundary) {
    require_cuda_device();
    validate_cuda_gradient_inputs(source, target);
    validate_transform_params(params);

    cudaArray_t source_array = nullptr;
    const cudaTextureObject_t texture = create_float_texture(source, boundary, &source_array);

    float* device_target = nullptr;
    float* device_block_sums = nullptr;
    const std::size_t target_bytes = target.pixels.size() * sizeof(float);
    REGISTRATION_CHECK_CUDA(cudaMalloc(&device_target, target_bytes));
    REGISTRATION_CHECK_CUDA(cudaMemcpy(device_target, target.pixels.data(), target_bytes, cudaMemcpyHostToDevice));

    constexpr int block_size = 256;
    const int blocks = reduction_block_count(target.pixels.size());
    REGISTRATION_CHECK_CUDA(cudaMalloc(&device_block_sums, static_cast<std::size_t>(blocks) * 5U * sizeof(float)));

    loss_gradient_texture_kernel<<<blocks, block_size>>>(
        texture,
        device_target,
        source.width,
        source.height,
        target.width,
        target.height,
        to_device_params(params),
        device_block_sums);
    REGISTRATION_CHECK_CUDA(cudaGetLastError());
    REGISTRATION_CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> block_sums(static_cast<std::size_t>(blocks) * 5U);
    REGISTRATION_CHECK_CUDA(cudaMemcpy(
        block_sums.data(),
        device_block_sums,
        block_sums.size() * sizeof(float),
        cudaMemcpyDeviceToHost));

    double loss = 0.0;
    double grad_theta = 0.0;
    double grad_scale = 0.0;
    double grad_tx = 0.0;
    double grad_ty = 0.0;
    for (int block = 0; block < blocks; ++block) {
        const std::size_t base = static_cast<std::size_t>(block) * 5U;
        loss += block_sums[base + 0];
        grad_theta += block_sums[base + 1];
        grad_scale += block_sums[base + 2];
        grad_tx += block_sums[base + 3];
        grad_ty += block_sums[base + 4];
    }

    REGISTRATION_CHECK_CUDA(cudaDestroyTextureObject(texture));
    REGISTRATION_CHECK_CUDA(cudaFreeArray(source_array));
    REGISTRATION_CHECK_CUDA(cudaFree(device_target));
    REGISTRATION_CHECK_CUDA(cudaFree(device_block_sums));

    LossGradient output;
    output.loss = static_cast<float>(loss / static_cast<double>(target.pixels.size()));
    output.gradient.theta = static_cast<float>(grad_theta);
    output.gradient.scale = static_cast<float>(grad_scale);
    output.gradient.tx = static_cast<float>(grad_tx);
    output.gradient.ty = static_cast<float>(grad_ty);
    return output;
}

}  // namespace registration
