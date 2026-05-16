#pragma once

#include <array>
#include <utility>

namespace registration {

struct Affine2D {
    float m00 = 1.0f;
    float m01 = 0.0f;
    float m02 = 0.0f;
    float m10 = 0.0f;
    float m11 = 1.0f;
    float m12 = 0.0f;

    static Affine2D identity();
    static Affine2D translation(float tx, float ty);
    static Affine2D scale(float sx, float sy);
    static Affine2D rotation(float radians);
    static Affine2D rotation_about(float radians, float cx, float cy);

    std::pair<float, float> transform(float x, float y) const;
    Affine2D inverse(float epsilon = 1.0e-8f) const;
    std::array<float, 6> to_array() const;
};

Affine2D operator*(const Affine2D& lhs, const Affine2D& rhs);
bool nearly_equal(const Affine2D& lhs, const Affine2D& rhs, float tolerance);

struct TransformParams {
    float theta = 0.0f;
    float scale = 1.0f;
    float tx = 0.0f;
    float ty = 0.0f;
};

void validate_transform_params(const TransformParams& params);
Affine2D source_from_destination(
    const TransformParams& params,
    int source_width,
    int source_height,
    int destination_width,
    int destination_height);

}  // namespace registration
