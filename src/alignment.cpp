#include "dgir/alignment.h"

#include <stdexcept>

namespace dgir {

AlignmentResult align_images_cpu(const Image&,
                                 const Image&,
                                 int,
                                 float,
                                 float) {
  throw std::logic_error("TODO(student): implement align_images_cpu");
}

Image make_checkerboard(int w, int h, int cell_size) {
  if (cell_size <= 0) {
    throw std::invalid_argument("cell_size must be positive");
  }

  Image image(w, h, 0.0f);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const int cx = x / cell_size;
      const int cy = y / cell_size;
      image.at(x, y) = ((cx + cy) % 2 == 0) ? 0.2f : 0.8f;
    }
  }
  return image;
}

}  // namespace dgir
