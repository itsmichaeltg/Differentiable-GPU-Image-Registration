#include "dgir/warp_cpu.h"

#include <algorithm>
#include <cmath>

namespace dgir {

namespace {

inline float clampf(float v, float lo, float hi) {
  return std::max(lo, std::min(v, hi));
}

}  // namespace

float bilinear_sample_clamp(const Image& image, float x, float y) {
  const float max_x = static_cast<float>(image.width - 1);
  const float max_y = static_cast<float>(image.height - 1);

  x = clampf(x, 0.0f, max_x);
  y = clampf(y, 0.0f, max_y);

  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const int x1 = std::min(x0 + 1, image.width - 1);
  const int y1 = std::min(y0 + 1, image.height - 1);

  const float wx = x - static_cast<float>(x0);
  const float wy = y - static_cast<float>(y0);

  const float v00 = image.at(x0, y0);
  const float v10 = image.at(x1, y0);
  const float v01 = image.at(x0, y1);
  const float v11 = image.at(x1, y1);

  const float top = v00 * (1.0f - wx) + v10 * wx;
  const float bot = v01 * (1.0f - wx) + v11 * wx;
  return top * (1.0f - wy) + bot * wy;
}

Image warp_affine_cpu(const Image& source, int out_w, int out_h, const Affine2x3& source_to_target) {
  Image out(out_w, out_h, 0.0f);
  const Affine2x3 target_to_source = invert_affine(source_to_target);

  for (int y = 0; y < out_h; ++y) {
    for (int x = 0; x < out_w; ++x) {
      const auto src_xy = apply_affine(target_to_source, static_cast<float>(x), static_cast<float>(y));
      out.at(x, y) = bilinear_sample_clamp(source, src_xy[0], src_xy[1]);
    }
  }

  return out;
}

}  // namespace dgir
