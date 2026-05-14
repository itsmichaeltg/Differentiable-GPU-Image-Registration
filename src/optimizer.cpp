#include "dgir/optimizer.h"

#include <stdexcept>

namespace dgir {

TransformParams gradient_descent_step(const TransformParams&,
                                      const std::array<float, 4>&,
                                      float) {
  throw std::logic_error("TODO(student): implement gradient_descent_step");
}

}  // namespace dgir
