#pragma once

#include "dgir/affine.h"
#include "dgir/image.h"

namespace dgir {

float bilinear_sample_clamp(const Image& image, float x, float y);
Image warp_affine_cpu(const Image& source, int out_w, int out_h, const Affine2x3& source_to_target);

}  // namespace dgir
