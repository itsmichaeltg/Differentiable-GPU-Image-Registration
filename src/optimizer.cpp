#include "dgir/optimizer.h"

#include <algorithm>

namespace dgir {

TransformParams gradient_descent_step(const TransformParams& params,
                                      const std::array<float, 4>& grad,
                                      float learning_rate) {
  TransformParams next = params;
  next.tx -= learning_rate * grad[0];
  next.ty -= learning_rate * grad[1];
  next.theta_rad -= learning_rate * grad[2];
  next.scale -= learning_rate * grad[3];
  next.scale = std::clamp(next.scale, 0.1f, 10.0f);
  return next;
}

}  // namespace dgir
