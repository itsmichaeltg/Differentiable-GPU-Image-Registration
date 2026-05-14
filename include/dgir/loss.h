#pragma once

#include <array>

#include "dgir/affine.h"
#include "dgir/image.h"

namespace dgir {

float mse_loss(const Image& a, const Image& b);

struct LossAndGradient {
  float loss = 0.0f;
  std::array<float, 4> grad = {0.0f, 0.0f, 0.0f, 0.0f};
};

// Starter CPU gradient (finite differences). Students should replace with GPU loss + backward pass.
LossAndGradient loss_and_numeric_gradient(const Image& source,
                                          const Image& target,
                                          const TransformParams& params,
                                          float epsilon = 1e-3f);

}  // namespace dgir
