#include <iostream>

#include "dgir/alignment.h"
#include "dgir/affine.h"
#include "dgir/loss.h"
#include "dgir/warp_cpu.h"

int main() {
  using namespace dgir;

  const Image source = make_checkerboard(64, 64, 8);

  TransformParams gt;
  gt.tx = 5.0f;
  gt.ty = -4.0f;
  gt.theta_rad = 0.08f;
  gt.scale = 1.02f;

  const Image target = warp_affine_cpu(source, source.width, source.height, make_affine(gt));

  const AlignmentResult result = align_images_cpu(source, target, 30, 0.5f, 1e-2f);
  std::cout << "Initial loss: " << result.losses.front() << "\n";
  std::cout << "Final loss:   " << result.losses.back() << "\n";
  std::cout << "Recovered params -> tx=" << result.params.tx << ", ty=" << result.params.ty
            << ", theta=" << result.params.theta_rad << ", scale=" << result.params.scale << "\n";

  return 0;
}
