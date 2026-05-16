#include "registration/cpu_registration.hpp"
#include "registration/gpu_registration.cuh"

#include <iostream>
#include <string>

#include "test_common.hpp"

using namespace registration;

int main() {
    std::string reason;
    if (!cuda_device_available(&reason)) {
        std::cout << "Skipping CUDA runtime warp test: " << reason << std::endl;
        return 0;
    }

    const Image image = make_gradient(16, 12);
    const Affine2D transform =
        Affine2D::translation(1.25f, -0.5f) *
        Affine2D::rotation_about(0.10f, 7.5f, 5.5f);

    const Image cpu = warp_affine_cpu(image, 16, 12, transform, BoundaryMode::Clamp);
    const Image gpu = warp_affine_cuda(image, 16, 12, transform, BoundaryMode::Clamp);
    CHECK_EQ(cpu.width, gpu.width);
    CHECK_EQ(cpu.height, gpu.height);
    CHECK_EQ(cpu.channels, gpu.channels);
    CHECK_NEAR(mse_cpu(cpu, gpu), 0.0f, 1.0e-5f);
    CHECK_NEAR(mse_cuda(cpu, gpu), 0.0f, 1.0e-5f);
    CHECK_NEAR(normalized_cross_correlation_cuda(cpu, gpu), 1.0f, 1.0e-4f);

    const Image gpu_zero =
        warp_affine_cuda(image, 16, 12, Affine2D::translation(20.0f, 0.0f), BoundaryMode::Zero);
    CHECK_NEAR(gpu_zero.at(0, 0), 0.0f, 1.0e-6f);

    const Image gpu_wrap =
        warp_affine_cuda(image, 16, 12, Affine2D::translation(16.0f, 0.0f), BoundaryMode::Wrap);
    CHECK_NEAR(gpu_wrap.at(0, 0), image.at(0, 0), 1.0e-5f);

    Image rgb(4, 4, 3);
    CHECK_THROWS(warp_affine_cuda(rgb, 4, 4, Affine2D::identity()));
    CHECK_THROWS(mse_cuda(Image(2, 2, 1), Image(2, 2, 2)));
    CHECK_THROWS(normalized_cross_correlation_cuda(Image(2, 2, 1, 1.0f), Image(2, 2, 1, 1.0f)));

    return 0;
}
