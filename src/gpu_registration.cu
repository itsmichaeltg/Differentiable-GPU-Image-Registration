#include "registration/gpu_registration.cuh"

#include <cuda_runtime.h>

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

    std::string reason;
    if (!cuda_device_available(&reason)) {
        throw std::runtime_error("CUDA device unavailable: " + reason);
    }

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

}  // namespace registration
