#pragma once

#include "dgir/affine.h"
#include "dgir/image.h"

namespace dgir {

// Bilinear sampling at floating-point pixel coordinates (x, y), using clamp-to-edge boundary behavior.
float bilinear_sample_clamp(const Image& image, float x, float y);
// Warps source into an output image of size (out_w, out_h).
// source_to_target maps source pixel coordinates -> target/output pixel coordinates.
Image warp_affine_cpu(const Image& source, int out_w, int out_h, const Affine2x3& source_to_target);

}  // namespace dgir
