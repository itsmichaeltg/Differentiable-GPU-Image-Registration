#include "registration/cpu_registration.hpp"

#include <cmath>

#include "test_common.hpp"

using namespace registration;

int main() {
    Image image(3, 2, 1);
    image.at(0, 0) = 1.0f;
    image.at(1, 0) = 2.0f;
    image.at(2, 0) = 3.0f;
    image.at(0, 1) = 4.0f;
    image.at(1, 1) = 5.0f;
    image.at(2, 1) = 6.0f;

    const Image identity = warp_affine_cpu(image, 3, 2, Affine2D::identity());
    CHECK_NEAR(mse_cpu(image, identity), 0.0f, 1.0e-7f);
    CHECK_NEAR(normalized_cross_correlation_cpu(image, identity), 1.0f, 1.0e-6f);

    const Image translated =
        warp_affine_cpu(image, 3, 2, Affine2D::translation(1.0f, 0.0f), BoundaryMode::Zero);
    CHECK_NEAR(translated.at(0, 0), 2.0f, 1.0e-6f);
    CHECK_NEAR(translated.at(1, 0), 3.0f, 1.0e-6f);
    CHECK_NEAR(translated.at(2, 0), 0.0f, 1.0e-6f);

    const Image clamped =
        warp_affine_cpu(image, 3, 2, Affine2D::translation(8.0f, 0.0f), BoundaryMode::Clamp);
    CHECK_NEAR(clamped.at(0, 0), 3.0f, 1.0e-6f);
    CHECK_NEAR(clamped.at(2, 1), 6.0f, 1.0e-6f);

    const Image wrapped =
        warp_affine_cpu(image, 3, 2, Affine2D::translation(3.0f, 0.0f), BoundaryMode::Wrap);
    CHECK_NEAR(wrapped.at(0, 0), 1.0f, 1.0e-6f);
    CHECK_NEAR(wrapped.at(2, 1), 6.0f, 1.0e-6f);

    const Image half = warp_affine_cpu(
        image,
        1,
        1,
        Affine2D::translation(0.5f, 0.5f),
        BoundaryMode::Clamp);
    CHECK_NEAR(half.at(0, 0), 3.0f, 1.0e-6f);

    Image three_channel(2, 1, 3);
    three_channel.at(0, 0, 0) = 1.0f;
    three_channel.at(0, 0, 1) = 2.0f;
    three_channel.at(0, 0, 2) = 3.0f;
    three_channel.at(1, 0, 0) = 4.0f;
    three_channel.at(1, 0, 1) = 5.0f;
    three_channel.at(1, 0, 2) = 6.0f;
    const Image shifted_rgb =
        warp_affine_cpu(three_channel, 2, 1, Affine2D::translation(1.0f, 0.0f));
    CHECK_NEAR(shifted_rgb.at(0, 0, 0), 4.0f, 1.0e-6f);
    CHECK_NEAR(shifted_rgb.at(0, 0, 1), 5.0f, 1.0e-6f);
    CHECK_NEAR(shifted_rgb.at(0, 0, 2), 6.0f, 1.0e-6f);

    CHECK_THROWS(warp_affine_cpu(image, 0, 2, Affine2D::identity()));
    CHECK_THROWS(mse_cpu(image, Image(3, 2, 2)));
    CHECK_THROWS(normalized_cross_correlation_cpu(Image(2, 2, 1, 5.0f), Image(2, 2, 1, 5.0f)));

    return 0;
}
