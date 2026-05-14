#include "dgir/affine.h"

#include <cmath>
#include <stdexcept>

namespace dgir {

Affine2x3 make_affine(const TransformParams& params) {
  const float c = std::cos(params.theta_rad);
  const float s = std::sin(params.theta_rad);
  const float sc = params.scale;

  return {
      sc * c, -sc * s, params.tx,
      sc * s, sc * c,  params.ty,
  };
}

Affine2x3 invert_affine(const Affine2x3& a) {
  const float det = a.m00 * a.m11 - a.m01 * a.m10;
  if (std::fabs(det) < 1e-12f) {
    throw std::runtime_error("Affine matrix is singular");
  }

  const float inv_det = 1.0f / det;
  Affine2x3 inv;
  inv.m00 = a.m11 * inv_det;
  inv.m01 = -a.m01 * inv_det;
  inv.m10 = -a.m10 * inv_det;
  inv.m11 = a.m00 * inv_det;

  inv.m02 = -(inv.m00 * a.m02 + inv.m01 * a.m12);
  inv.m12 = -(inv.m10 * a.m02 + inv.m11 * a.m12);
  return inv;
}

std::array<float, 2> apply_affine(const Affine2x3& a, float x, float y) {
  return {
      a.m00 * x + a.m01 * y + a.m02,
      a.m10 * x + a.m11 * y + a.m12,
  };
}

}  // namespace dgir
