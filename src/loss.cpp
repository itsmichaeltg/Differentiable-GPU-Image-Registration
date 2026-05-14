#include "dgir/loss.h"

#include <stdexcept>

namespace dgir {

float mse_loss(const Image&, const Image&) {
  throw std::logic_error("TODO(student): implement mse_loss");
}

LossAndGradient loss_and_numeric_gradient(const Image&,
                                          const Image&,
                                          const TransformParams&,
                                          float) {
  throw std::logic_error("TODO(student): implement loss_and_numeric_gradient");
}

}  // namespace dgir
