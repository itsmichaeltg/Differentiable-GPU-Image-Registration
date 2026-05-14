#include <cmath>
#include <cstdlib>
#include <iostream>

#include "dgir/alignment.h"
#include "dgir/affine.h"
#include "dgir/loss.h"
#include "dgir/warp_cpu.h"

namespace {

void expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    std::exit(1);
  }
}

bool approx(float a, float b, float eps = 1e-4f) {
  return std::fabs(a - b) <= eps;
}

}  // namespace

int main() {
  using namespace dgir;

  {
    Image a(4, 4, 0.0f);
    expect(approx(mse_loss(a, a), 0.0f), "MSE of identical images should be zero");
  }

  {
    TransformParams p;
    p.tx = 2.0f;
    p.ty = -3.0f;
    const auto a = make_affine(p);
    const auto xy = apply_affine(a, 1.0f, 2.0f);
    expect(approx(xy[0], 3.0f) && approx(xy[1], -1.0f), "Affine translation should map coordinates");
  }

  {
    const Image source = make_checkerboard(32, 32, 4);
    TransformParams gt;
    gt.tx = 3.0f;
    gt.ty = -2.0f;
    const Image target = warp_affine_cpu(source, source.width, source.height, make_affine(gt));

    const AlignmentResult result = align_images_cpu(source, target, 20, 0.3f, 1e-2f);
    expect(result.losses.size() == 20, "Alignment should record one loss per iteration");
    expect(result.losses.back() < result.losses.front(), "Alignment loop should reduce loss on synthetic translation");
  }

  std::cout << "All starter tests passed\n";
  return 0;
}
