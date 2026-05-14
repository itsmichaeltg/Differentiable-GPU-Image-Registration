#include "dgir/warp_cpu.h"

#include <stdexcept>

namespace dgir {

float bilinear_sample_clamp(const Image&, float, float) {
  throw std::logic_error("TODO(student): implement bilinear_sample_clamp");
}

Image warp_affine_cpu(const Image&, int, int, const Affine2x3&) {
  throw std::logic_error("TODO(student): implement warp_affine_cpu");
}

}  // namespace dgir
