#pragma once

#include <vector>

#include "dgir/affine.h"
#include "dgir/image.h"

namespace dgir {

struct AlignmentResult {
  TransformParams params;
  std::vector<float> losses;
};

AlignmentResult align_images_cpu(const Image& source,
                                 const Image& target,
                                 int iterations,
                                 float learning_rate,
                                 float gradient_epsilon = 1e-3f);

}  // namespace dgir
