#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace dgir {

struct Image {
  int width = 0;
  int height = 0;
  std::vector<float> pixels;

  Image() = default;
  Image(int w, int h, float fill = 0.0f) : width(w), height(h), pixels(static_cast<std::size_t>(w * h), fill) {
    if (w <= 0 || h <= 0) {
      throw std::invalid_argument("Image dimensions must be positive");
    }
  }

  float& at(int x, int y) {
    return pixels.at(static_cast<std::size_t>(y * width + x));
  }

  float at(int x, int y) const {
    return pixels.at(static_cast<std::size_t>(y * width + x));
  }
};

Image make_checkerboard(int w, int h, int cell_size);

}  // namespace dgir
