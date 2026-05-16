#pragma once

#include <string>

#include "registration/affine.hpp"
#include "registration/image.hpp"
#include "registration/optimizer.hpp"

namespace registration {

bool cuda_device_available(std::string* reason = nullptr);

Image warp_affine_cuda(
    const Image& source,
    int output_width,
    int output_height,
    const Affine2D& source_from_dest,
    BoundaryMode boundary = BoundaryMode::Zero,
    float fill_value = 0.0f);

Image warp_affine_cuda(
    const Image& source,
    int output_width,
    int output_height,
    const TransformParams& params,
    BoundaryMode boundary = BoundaryMode::Zero,
    float fill_value = 0.0f);

Image warp_affine_cuda_global(
    const Image& source,
    int output_width,
    int output_height,
    const Affine2D& source_from_dest,
    BoundaryMode boundary = BoundaryMode::Zero,
    float fill_value = 0.0f);

float mse_cuda(const Image& lhs, const Image& rhs);
float normalized_cross_correlation_cuda(const Image& lhs, const Image& rhs);
LossGradient loss_gradient_cuda(
    const Image& source,
    const Image& target,
    const TransformParams& params,
    BoundaryMode boundary = BoundaryMode::Clamp);

}  // namespace registration
