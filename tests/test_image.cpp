#include "registration/image.hpp"

#include <limits>

#include "test_common.hpp"

using namespace registration;

int main() {
    Image image(2, 2, 1);
    image.at(0, 0) = 1.0f;
    image.at(1, 0) = 3.0f;
    image.at(0, 1) = 5.0f;
    image.at(1, 1) = 7.0f;

    CHECK_EQ(image.size(), 4U);
    CHECK_NEAR(sample_bilinear(image, 0.0f, 0.0f), 1.0f, 1.0e-6f);
    CHECK_NEAR(sample_bilinear(image, 1.0f, 1.0f), 7.0f, 1.0e-6f);
    CHECK_NEAR(sample_bilinear(image, 0.5f, 0.5f), 4.0f, 1.0e-6f);

    CHECK_NEAR(sample_bilinear(image, -1.0f, 0.0f, 0, BoundaryMode::Zero, 9.0f), 9.0f, 1.0e-6f);
    CHECK_NEAR(sample_bilinear(image, -1.0f, 0.0f, 0, BoundaryMode::Clamp), 1.0f, 1.0e-6f);
    CHECK_NEAR(sample_bilinear(image, -1.0f, 0.0f, 0, BoundaryMode::Wrap), 3.0f, 1.0e-6f);

    Image rgb(1, 1, 3);
    rgb.at(0, 0, 0) = 0.25f;
    rgb.at(0, 0, 1) = 0.50f;
    rgb.at(0, 0, 2) = 0.75f;
    CHECK_NEAR(rgb.at(0, 0, 2), 0.75f, 1.0e-6f);

    const Image gradient = make_gradient(4, 3);
    CHECK_EQ(gradient.width, 4);
    CHECK_EQ(gradient.height, 3);
    CHECK_NEAR(gradient.at(0, 0), 0.0f, 1.0e-6f);
    CHECK_NEAR(gradient.at(3, 2), 1.0f, 1.0e-6f);

    const Image checker = make_checkerboard(4, 4, 2);
    CHECK_NEAR(checker.at(0, 0), 1.0f, 1.0e-6f);
    CHECK_NEAR(checker.at(2, 0), 0.0f, 1.0e-6f);
    CHECK_NEAR(checker.at(2, 2), 1.0f, 1.0e-6f);

    CHECK_THROWS(Image(0, 2, 1));
    CHECK_THROWS(Image(2, 0, 1));
    CHECK_THROWS(Image(2, 2, 0));
    CHECK_THROWS(Image(2, 2, 1, std::vector<float>{1.0f}));
    CHECK_THROWS(image.at(2, 0));
    CHECK_THROWS(image.at(0, 0, 1));
    CHECK_THROWS(sample_bilinear(image, std::numeric_limits<float>::quiet_NaN(), 0.0f));
    CHECK_THROWS(make_checkerboard(4, 4, 0));

    return 0;
}
