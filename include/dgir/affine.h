#pragma once

#include <array>

namespace dgir {

struct TransformParams {
  float tx = 0.0f;
  float ty = 0.0f;
  float theta_rad = 0.0f;
  float scale = 1.0f;
};

struct Affine2x3 {
  float m00 = 1.0f;
  float m01 = 0.0f;
  float m02 = 0.0f;
  float m10 = 0.0f;
  float m11 = 1.0f;
  float m12 = 0.0f;
};

Affine2x3 make_affine(const TransformParams& params);
Affine2x3 invert_affine(const Affine2x3& a);
std::array<float, 2> apply_affine(const Affine2x3& a, float x, float y);

}  // namespace dgir
