#include "registration/affine.hpp"

#include <cmath>
#include <limits>

#include "test_common.hpp"

using namespace registration;

int main() {
    const Affine2D identity = Affine2D::identity();
    const auto [ix, iy] = identity.transform(4.0f, -2.0f);
    CHECK_NEAR(ix, 4.0f, 1.0e-6f);
    CHECK_NEAR(iy, -2.0f, 1.0e-6f);

    const Affine2D translate = Affine2D::translation(3.0f, -5.0f);
    const auto [tx, ty] = translate.transform(2.0f, 8.0f);
    CHECK_NEAR(tx, 5.0f, 1.0e-6f);
    CHECK_NEAR(ty, 3.0f, 1.0e-6f);

    const Affine2D composed =
        Affine2D::translation(10.0f, 0.0f) * Affine2D::scale(2.0f, 3.0f);
    const auto [cx, cy] = composed.transform(4.0f, 2.0f);
    CHECK_NEAR(cx, 18.0f, 1.0e-6f);
    CHECK_NEAR(cy, 6.0f, 1.0e-6f);

    const Affine2D rotate = Affine2D::rotation(static_cast<float>(M_PI) * 0.5f);
    const auto [rx, ry] = rotate.transform(1.0f, 0.0f);
    CHECK_NEAR(rx, 0.0f, 1.0e-5f);
    CHECK_NEAR(ry, 1.0f, 1.0e-5f);

    const Affine2D round_trip = composed.inverse() * composed;
    CHECK_TRUE(nearly_equal(round_trip, Affine2D::identity(), 1.0e-5f));

    const Affine2D centered = source_from_destination({0.0f, 1.0f, 0.0f, 0.0f}, 5, 7, 5, 7);
    CHECK_TRUE(nearly_equal(centered, Affine2D::identity(), 1.0e-6f));

    const Affine2D centered_translation =
        source_from_destination({0.0f, 1.0f, 2.0f, -3.0f}, 5, 7, 5, 7);
    const auto [sx, sy] = centered_translation.transform(1.0f, 1.0f);
    CHECK_NEAR(sx, 3.0f, 1.0e-6f);
    CHECK_NEAR(sy, -2.0f, 1.0e-6f);

    CHECK_THROWS(Affine2D::scale(std::numeric_limits<float>::infinity(), 1.0f));
    const Affine2D singular{1.0f, 2.0f, 0.0f, 2.0f, 4.0f, 0.0f};
    CHECK_THROWS(singular.inverse());
    CHECK_THROWS(validate_transform_params({0.0f, 0.0f, 0.0f, 0.0f}));
    CHECK_THROWS(source_from_destination({0.0f, 1.0f, 0.0f, 0.0f}, 0, 4, 4, 4));

    return 0;
}
