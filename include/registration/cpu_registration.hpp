#pragma once

#include "registration/affine.hpp"
#include "registration/image.hpp"

namespace registration {

Image warp_affine_cpu(
    const Image& source,
    int output_width,
    int output_height,
    const Affine2D& source_from_dest,
    BoundaryMode boundary = BoundaryMode::Zero,
    float fill_value = 0.0f);

Image warp_affine_cpu(
    const Image& source,
    int output_width,
    int output_height,
    const TransformParams& params,
    BoundaryMode boundary = BoundaryMode::Zero,
    float fill_value = 0.0f);

float mse_cpu(const Image& lhs, const Image& rhs);

}  // namespace registration
