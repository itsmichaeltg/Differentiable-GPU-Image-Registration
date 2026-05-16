#include "registration/affine.hpp"

#include <cmath>
#include <stdexcept>

namespace registration {

namespace {

void validate_dimension(int value, const char* name) {
    if (value <= 0) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
}

}  // namespace

Affine2D Affine2D::identity() {
    return {};
}

Affine2D Affine2D::translation(float tx, float ty) {
    return {1.0f, 0.0f, tx, 0.0f, 1.0f, ty};
}

Affine2D Affine2D::scale(float sx, float sy) {
    if (!std::isfinite(sx) || !std::isfinite(sy)) {
        throw std::invalid_argument("scale values must be finite");
    }
    return {sx, 0.0f, 0.0f, 0.0f, sy, 0.0f};
}

Affine2D Affine2D::rotation(float radians) {
    if (!std::isfinite(radians)) {
        throw std::invalid_argument("rotation must be finite");
    }
    const float c = std::cos(radians);
    const float s = std::sin(radians);
    return {c, -s, 0.0f, s, c, 0.0f};
}

Affine2D Affine2D::rotation_about(float radians, float cx, float cy) {
    if (!std::isfinite(cx) || !std::isfinite(cy)) {
        throw std::invalid_argument("rotation center must be finite");
    }
    return Affine2D::translation(cx, cy) * Affine2D::rotation(radians) *
           Affine2D::translation(-cx, -cy);
}

std::pair<float, float> Affine2D::transform(float x, float y) const {
    if (!std::isfinite(x) || !std::isfinite(y)) {
        throw std::invalid_argument("coordinates must be finite");
    }
    return {m00 * x + m01 * y + m02, m10 * x + m11 * y + m12};
}

Affine2D Affine2D::inverse(float epsilon) const {
    const float det = m00 * m11 - m01 * m10;
    if (!std::isfinite(det) || std::fabs(det) <= epsilon) {
        throw std::invalid_argument("affine matrix is singular");
    }
    const float inv_det = 1.0f / det;
    Affine2D inv;
    inv.m00 = m11 * inv_det;
    inv.m01 = -m01 * inv_det;
    inv.m10 = -m10 * inv_det;
    inv.m11 = m00 * inv_det;
    inv.m02 = -(inv.m00 * m02 + inv.m01 * m12);
    inv.m12 = -(inv.m10 * m02 + inv.m11 * m12);
    return inv;
}

std::array<float, 6> Affine2D::to_array() const {
    return {m00, m01, m02, m10, m11, m12};
}

Affine2D operator*(const Affine2D& lhs, const Affine2D& rhs) {
    Affine2D out;
    out.m00 = lhs.m00 * rhs.m00 + lhs.m01 * rhs.m10;
    out.m01 = lhs.m00 * rhs.m01 + lhs.m01 * rhs.m11;
    out.m02 = lhs.m00 * rhs.m02 + lhs.m01 * rhs.m12 + lhs.m02;
    out.m10 = lhs.m10 * rhs.m00 + lhs.m11 * rhs.m10;
    out.m11 = lhs.m10 * rhs.m01 + lhs.m11 * rhs.m11;
    out.m12 = lhs.m10 * rhs.m02 + lhs.m11 * rhs.m12 + lhs.m12;
    return out;
}

bool nearly_equal(const Affine2D& lhs, const Affine2D& rhs, float tolerance) {
    const auto a = lhs.to_array();
    const auto b = rhs.to_array();
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

void validate_transform_params(const TransformParams& params) {
    if (!std::isfinite(params.theta) || !std::isfinite(params.scale) ||
        !std::isfinite(params.tx) || !std::isfinite(params.ty)) {
        throw std::invalid_argument("transform parameters must be finite");
    }
    if (params.scale <= 0.0f) {
        throw std::invalid_argument("transform scale must be positive");
    }
}

Affine2D source_from_destination(
    const TransformParams& params,
    int source_width,
    int source_height,
    int destination_width,
    int destination_height) {
    validate_transform_params(params);
    validate_dimension(source_width, "source_width");
    validate_dimension(source_height, "source_height");
    validate_dimension(destination_width, "destination_width");
    validate_dimension(destination_height, "destination_height");

    const float source_cx = 0.5f * static_cast<float>(source_width - 1);
    const float source_cy = 0.5f * static_cast<float>(source_height - 1);
    const float dest_cx = 0.5f * static_cast<float>(destination_width - 1);
    const float dest_cy = 0.5f * static_cast<float>(destination_height - 1);

    return Affine2D::translation(source_cx + params.tx, source_cy + params.ty) *
           Affine2D::rotation(params.theta) *
           Affine2D::scale(params.scale, params.scale) *
           Affine2D::translation(-dest_cx, -dest_cy);
}

}  // namespace registration
