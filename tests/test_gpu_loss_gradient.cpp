#include "registration/cpu_registration.hpp"
#include "registration/gpu_registration.cuh"
#include "registration/optimizer.hpp"

#include <iostream>
#include <string>

#include "test_common.hpp"

using namespace registration;

int main() {
    std::string reason;
    if (!cuda_device_available(&reason)) {
        std::cout << "Skipping CUDA loss/gradient test: " << reason << std::endl;
        return 0;
    }

    const Image source = make_registration_pattern(24, 20);
    const TransformParams truth{0.03f, 1.0f, 1.25f, -0.75f};
    const Image target = warp_affine_cpu(source, 24, 20, truth, BoundaryMode::Clamp);

    const TransformParams estimate{0.01f, 1.0f, 0.5f, -0.25f};
    const LossGradient cpu = loss_gradient_cpu(source, target, estimate, BoundaryMode::Clamp);
    const LossGradient gpu = loss_gradient_cuda(source, target, estimate, BoundaryMode::Clamp);

    CHECK_NEAR(cpu.loss, gpu.loss, 1.0e-5f);
    CHECK_NEAR(cpu.gradient.tx, gpu.gradient.tx, 1.0e-4f);
    CHECK_NEAR(cpu.gradient.ty, gpu.gradient.ty, 1.0e-4f);
    CHECK_NEAR(cpu.gradient.theta, gpu.gradient.theta, 1.0e-3f);
    CHECK_NEAR(cpu.gradient.scale, gpu.gradient.scale, 1.0e-3f);

    OptimizerOptions options;
    options.backend = RegistrationBackend::CUDA;
    options.max_iterations = 30;
    options.pyramid_levels = 1;
    options.optimize_rotation = false;
    options.optimize_scale = false;
    options.learning_rate_translation = 0.25f;

    const RegistrationResult result = align_images(source, target, TransformParams{}, options);
    CHECK_TRUE(result.final_loss < result.initial_loss);

    CHECK_THROWS(loss_gradient_cuda(Image(4, 4, 3), Image(4, 4, 3), TransformParams{}));
    return 0;
}
