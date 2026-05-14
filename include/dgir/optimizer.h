#pragma once

#include <array>

#include "dgir/affine.h"

namespace dgir {

TransformParams gradient_descent_step(const TransformParams& params,
                                      const std::array<float, 4>& grad,
                                      float learning_rate);

}  // namespace dgir
