#include "dgir/warp_cuda.h"

namespace dgir {

CudaWarpStatus warp_affine_cuda(const Image&,
                                int,
                                int,
                                const Affine2x3&,
                                Image*) {
  // TODO(student):
  // 1) Upload image to device.
  // 2) Bind CUDA texture object with linear filtering.
  // 3) Launch affine warp kernel.
  // 4) Add boundary mode (clamp/pad) controls.
  return {false, "TODO: implement CUDA texture-based affine warp"};
}

}  // namespace dgir
