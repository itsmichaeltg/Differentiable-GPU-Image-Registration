#pragma once

#include <vector>

#include "dgir/affine.h"
#include "dgir/image.h"

namespace dgir {

struct AlignmentResult {
  // Final transform parameters after the last optimization iteration.
  TransformParams params;
  // Per-iteration loss history (index i == loss after iteration i).
  std::vector<float> losses;
};

AlignmentResult align_images_cpu(const Image& source,
                                 const Image& target,
                                 int iterations,
                                 float learning_rate,
                                 float gradient_epsilon = 1e-3f);

}  // namespace dgir
