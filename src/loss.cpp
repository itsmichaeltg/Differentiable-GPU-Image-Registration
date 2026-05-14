#include "dgir/loss.h"

#include <stdexcept>

#include "dgir/warp_cpu.h"

namespace dgir {

float mse_loss(const Image& a, const Image& b) {
  if (a.width != b.width || a.height != b.height) {
    throw std::invalid_argument("MSE requires same image dimensions");
  }

  const std::size_t n = a.pixels.size();
  double accum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(a.pixels[i] - b.pixels[i]);
    accum += d * d;
  }
  return static_cast<float>(accum / static_cast<double>(n));
}

LossAndGradient loss_and_numeric_gradient(const Image& source,
                                          const Image& target,
                                          const TransformParams& params,
                                          float epsilon) {
  LossAndGradient result;

  auto evaluate = [&](const TransformParams& p) {
    const auto affine = make_affine(p);
    const Image warped = warp_affine_cpu(source, target.width, target.height, affine);
    return mse_loss(warped, target);
  };

  result.loss = evaluate(params);

  for (int i = 0; i < 4; ++i) {
    TransformParams plus = params;
    TransformParams minus = params;

    if (i == 0) { plus.tx += epsilon; minus.tx -= epsilon; }
    if (i == 1) { plus.ty += epsilon; minus.ty -= epsilon; }
    if (i == 2) { plus.theta_rad += epsilon; minus.theta_rad -= epsilon; }
    if (i == 3) { plus.scale += epsilon; minus.scale -= epsilon; }

    const float lp = evaluate(plus);
    const float lm = evaluate(minus);
    result.grad[static_cast<std::size_t>(i)] = (lp - lm) / (2.0f * epsilon);
  }

  return result;
}

}  // namespace dgir
