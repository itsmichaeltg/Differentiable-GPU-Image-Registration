#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "dgir/alignment.h"
#include "dgir/affine.h"
#include "dgir/loss.h"
#include "dgir/optimizer.h"
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

template <typename Fn>
void expect_todo(Fn&& fn, const char* message) {
  bool threw = false;
  try {
    fn();
  } catch (const std::logic_error&) {
    threw = true;
  }
  expect(threw, message);
}

}  // namespace

int main() {
  using namespace dgir;

  {
    TransformParams p;
    p.tx = 2.0f;
    p.ty = -3.0f;
    const auto a = make_affine(p);
    const auto xy = apply_affine(a, 1.0f, 2.0f);
    expect(approx(xy[0], 3.0f) && approx(xy[1], -1.0f), "Affine translation should map coordinates");
  }

  {
    const Image board = make_checkerboard(8, 8, 2);
    expect(board.width == 8 && board.height == 8, "Checkerboard should have requested dimensions");
    expect(!approx(board.at(0, 0), board.at(2, 0)), "Checkerboard should alternate between neighboring cells");
  }

  {
    const Image source(4, 4, 0.25f);
    const Image target(4, 4, 0.75f);
    expect_todo([&] { (void)bilinear_sample_clamp(source, 1.5f, 1.5f); },
                "bilinear_sample_clamp should remain a student TODO");
    expect_todo([&] { (void)warp_affine_cpu(source, source.width, source.height, make_affine({})); },
                "warp_affine_cpu should remain a student TODO");
    expect_todo([&] { (void)mse_loss(source, target); },
                "mse_loss should remain a student TODO");
    expect_todo([&] { (void)loss_and_numeric_gradient(source, target, TransformParams{}); },
                "loss_and_numeric_gradient should remain a student TODO");
    expect_todo([&] { (void)gradient_descent_step(TransformParams{}, {0.0f, 0.0f, 0.0f, 0.0f}, 0.1f); },
                "gradient_descent_step should remain a student TODO");
    expect_todo([&] { (void)align_images_cpu(source, target, 10, 0.1f); },
                "align_images_cpu should remain a student TODO");
  }

  std::cout << "All starter tests passed\n";
  return 0;
}
