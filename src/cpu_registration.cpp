#include "registration/cpu_registration.hpp"

#include <cmath>
#include <stdexcept>

namespace registration {

namespace {

void validate_output_shape(int width, int height) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("output dimensions must be positive");
    }
}

}  // namespace

Image warp_affine_cpu(
    const Image& source,
    int output_width,
    int output_height,
    const Affine2D& source_from_dest,
    BoundaryMode boundary,
    float fill_value) {
    validate_output_shape(output_width, output_height);
    if (source.empty()) {
        throw std::invalid_argument("source image must not be empty");
    }

    Image output(output_width, output_height, source.channels);
    for (int y = 0; y < output_height; ++y) {
        for (int x = 0; x < output_width; ++x) {
            const auto [source_x, source_y] =
                source_from_dest.transform(static_cast<float>(x), static_cast<float>(y));
            for (int c = 0; c < source.channels; ++c) {
                output.at(x, y, c) =
                    sample_bilinear(source, source_x, source_y, c, boundary, fill_value);
            }
        }
    }
    return output;
}

Image warp_affine_cpu(
    const Image& source,
    int output_width,
    int output_height,
    const TransformParams& params,
    BoundaryMode boundary,
    float fill_value) {
    return warp_affine_cpu(
        source,
        output_width,
        output_height,
        source_from_destination(params, source.width, source.height, output_width, output_height),
        boundary,
        fill_value);
}

float mse_cpu(const Image& lhs, const Image& rhs) {
    validate_same_shape(lhs, rhs);
    double sum = 0.0;
    for (std::size_t i = 0; i < lhs.pixels.size(); ++i) {
        const double diff = static_cast<double>(lhs.pixels[i]) - static_cast<double>(rhs.pixels[i]);
        sum += diff * diff;
    }
    return static_cast<float>(sum / static_cast<double>(lhs.pixels.size()));
}

}  // namespace registration
