#include "dgir/alignment.h"

#include "dgir/loss.h"
#include "dgir/optimizer.h"

namespace dgir {

AlignmentResult align_images_cpu(const Image& source,
                                 const Image& target,
                                 int iterations,
                                 float learning_rate,
                                 float gradient_epsilon) {
  AlignmentResult out;
  out.params = TransformParams{};

  out.losses.reserve(static_cast<std::size_t>(iterations));

  for (int i = 0; i < iterations; ++i) {
    const LossAndGradient lg = loss_and_numeric_gradient(source, target, out.params, gradient_epsilon);
    out.losses.push_back(lg.loss);
    out.params = gradient_descent_step(out.params, lg.grad, learning_rate);
  }

  return out;
}

Image make_checkerboard(int w, int h, int cell_size) {
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
