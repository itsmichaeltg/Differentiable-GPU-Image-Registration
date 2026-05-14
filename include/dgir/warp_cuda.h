#pragma once

#include "dgir/affine.h"
#include "dgir/image.h"

namespace dgir {

struct CudaWarpStatus {
  bool ok = false;
  const char* message = "CUDA module not implemented";
};

// TODO(student): Bind source image as CUDA texture object and implement bilinear affine warping.
CudaWarpStatus warp_affine_cuda(const Image& source,
                                int out_w,
                                int out_h,
                                const Affine2x3& source_to_target,
                                Image* output);

}  // namespace dgir
