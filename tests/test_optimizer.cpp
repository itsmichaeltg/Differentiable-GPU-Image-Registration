#include "registration/cpu_registration.hpp"
#include "registration/optimizer.hpp"

#include "test_common.hpp"

using namespace registration;

int main() {
    const Image source = make_registration_pattern(48, 40);
    const TransformParams truth{0.0f, 1.0f, 2.25f, -1.50f};
    const Image target = warp_affine_cpu(source, 48, 40, truth, BoundaryMode::Clamp);

    OptimizerOptions options;
    options.max_iterations = 120;
    options.pyramid_levels = 2;
    options.optimize_rotation = false;
    options.optimize_scale = false;
    options.learning_rate_translation = 0.35f;
    options.tolerance = 1.0e-9f;

    const RegistrationResult result =
        align_images(source, target, TransformParams{}, options);

    CHECK_TRUE(result.final_loss < result.initial_loss * 0.10f);
    CHECK_NEAR(result.params.tx, truth.tx, 0.35f);
    CHECK_NEAR(result.params.ty, truth.ty, 0.35f);
    CHECK_TRUE(!result.history.empty());
    CHECK_EQ(result.aligned.width, target.width);
    CHECK_EQ(result.aligned.height, target.height);

    const LossGradient gradient = loss_gradient_cpu(source, target, result.params, BoundaryMode::Clamp);
    CHECK_TRUE(gradient.loss < result.initial_loss * 0.10f);

    CHECK_THROWS(align_images(source, Image(12, 12, 1), TransformParams{}, options));
    options.max_iterations = 0;
    CHECK_THROWS(align_images(source, target, TransformParams{}, options));

    const Image odd(5, 3, 1, 2.0f);
    const Image small = downsample_half(odd);
    CHECK_EQ(small.width, 3);
    CHECK_EQ(small.height, 2);
    CHECK_NEAR(small.at(2, 1), 2.0f, 1.0e-6f);

    return 0;
}
