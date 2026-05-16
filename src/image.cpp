#include "registration/image.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace registration {

namespace {

void validate_shape(int width, int height, int channels) {
    if (width <= 0) {
        throw std::invalid_argument("image width must be positive");
    }
    if (height <= 0) {
        throw std::invalid_argument("image height must be positive");
    }
    if (channels <= 0) {
        throw std::invalid_argument("image channel count must be positive");
    }
}

int clamp_index(int value, int upper_exclusive) {
    return std::clamp(value, 0, upper_exclusive - 1);
}

int wrap_index(int value, int upper_exclusive) {
    int wrapped = value % upper_exclusive;
    if (wrapped < 0) {
        wrapped += upper_exclusive;
    }
    return wrapped;
}

float pixel_or_boundary(
    const Image& image,
    int x,
    int y,
    int channel,
    BoundaryMode boundary,
    float fill_value) {
    if (channel < 0 || channel >= image.channels) {
        throw std::out_of_range("channel index out of range");
    }

    switch (boundary) {
        case BoundaryMode::Zero:
            if (x < 0 || x >= image.width || y < 0 || y >= image.height) {
                return fill_value;
            }
            return image.at(x, y, channel);
        case BoundaryMode::Clamp:
            return image.at(
                clamp_index(x, image.width),
                clamp_index(y, image.height),
                channel);
        case BoundaryMode::Wrap:
            return image.at(
                wrap_index(x, image.width),
                wrap_index(y, image.height),
                channel);
    }
    return fill_value;
}

}  // namespace

std::string boundary_mode_name(BoundaryMode mode) {
    switch (mode) {
        case BoundaryMode::Zero:
            return "zero";
        case BoundaryMode::Clamp:
            return "clamp";
        case BoundaryMode::Wrap:
            return "wrap";
    }
    return "unknown";
}

Image::Image(int image_width, int image_height, int image_channels, float fill)
    : width(image_width),
      height(image_height),
      channels(image_channels),
      pixels(static_cast<std::size_t>(image_width) *
             static_cast<std::size_t>(image_height) *
             static_cast<std::size_t>(image_channels),
             fill) {
    validate_shape(image_width, image_height, image_channels);
}

Image::Image(int image_width, int image_height, int image_channels, std::vector<float> values)
    : width(image_width),
      height(image_height),
      channels(image_channels),
      pixels(std::move(values)) {
    validate_shape(image_width, image_height, image_channels);
    const auto expected = static_cast<std::size_t>(image_width) *
                          static_cast<std::size_t>(image_height) *
                          static_cast<std::size_t>(image_channels);
    if (pixels.size() != expected) {
        throw std::invalid_argument("pixel vector size does not match image dimensions");
    }
}

std::size_t Image::size() const {
    return pixels.size();
}

bool Image::empty() const {
    return width <= 0 || height <= 0 || channels <= 0 || pixels.empty();
}

bool Image::same_shape(const Image& other) const {
    return width == other.width && height == other.height && channels == other.channels;
}

std::size_t Image::index(int x, int y, int channel) const {
    if (x < 0 || x >= width) {
        throw std::out_of_range("x index out of range");
    }
    if (y < 0 || y >= height) {
        throw std::out_of_range("y index out of range");
    }
    if (channel < 0 || channel >= channels) {
        throw std::out_of_range("channel index out of range");
    }
    return (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
            static_cast<std::size_t>(x)) *
               static_cast<std::size_t>(channels) +
           static_cast<std::size_t>(channel);
}

float& Image::at(int x, int y, int channel) {
    return pixels[index(x, y, channel)];
}

const float& Image::at(int x, int y, int channel) const {
    return pixels[index(x, y, channel)];
}

void validate_same_shape(const Image& lhs, const Image& rhs) {
    if (lhs.empty() || rhs.empty()) {
        throw std::invalid_argument("images must not be empty");
    }
    if (!lhs.same_shape(rhs)) {
        throw std::invalid_argument("images must have identical dimensions and channel counts");
    }
}

float sample_bilinear(
    const Image& image,
    float x,
    float y,
    int channel,
    BoundaryMode boundary,
    float fill_value) {
    if (image.empty()) {
        throw std::invalid_argument("image must not be empty");
    }
    if (!std::isfinite(x) || !std::isfinite(y)) {
        throw std::invalid_argument("sample coordinates must be finite");
    }

    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);

    const float p00 = pixel_or_boundary(image, x0, y0, channel, boundary, fill_value);
    const float p10 = pixel_or_boundary(image, x1, y0, channel, boundary, fill_value);
    const float p01 = pixel_or_boundary(image, x0, y1, channel, boundary, fill_value);
    const float p11 = pixel_or_boundary(image, x1, y1, channel, boundary, fill_value);

    const float top = p00 * (1.0f - tx) + p10 * tx;
    const float bottom = p01 * (1.0f - tx) + p11 * tx;
    return top * (1.0f - ty) + bottom * ty;
}

Image make_constant(int width, int height, int channels, float value) {
    return Image(width, height, channels, value);
}

Image make_gradient(int width, int height) {
    Image image(width, height, 1);
    const float denom_x = width > 1 ? static_cast<float>(width - 1) : 1.0f;
    const float denom_y = height > 1 ? static_cast<float>(height - 1) : 1.0f;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            image.at(x, y) =
                0.65f * static_cast<float>(x) / denom_x +
                0.35f * static_cast<float>(y) / denom_y;
        }
    }
    return image;
}

Image make_checkerboard(int width, int height, int tile_size) {
    if (tile_size <= 0) {
        throw std::invalid_argument("tile size must be positive");
    }
    Image image(width, height, 1);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int tile_x = x / tile_size;
            const int tile_y = y / tile_size;
            image.at(x, y) = ((tile_x + tile_y) % 2 == 0) ? 1.0f : 0.0f;
        }
    }
    return image;
}

Image make_registration_pattern(int width, int height) {
    Image image(width, height, 1);
    float min_value = 1.0e30f;
    float max_value = -1.0e30f;

    const float cx = 0.5f * static_cast<float>(width - 1);
    const float cy = 0.5f * static_cast<float>(height - 1);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float xf = static_cast<float>(x);
            const float yf = static_cast<float>(y);
            const float dx = (xf - cx) / std::max(1.0f, cx);
            const float dy = (yf - cy) / std::max(1.0f, cy);
            const float spot_a = std::exp(-8.0f * ((dx + 0.35f) * (dx + 0.35f) + (dy - 0.25f) * (dy - 0.25f)));
            const float spot_b = std::exp(-18.0f * ((dx - 0.30f) * (dx - 0.30f) + (dy + 0.20f) * (dy + 0.20f)));
            const float ridge = std::sin(0.23f * xf + 0.07f * yf) + std::cos(0.11f * xf - 0.19f * yf);
            const float value = 0.25f * ridge + 0.80f * spot_a + 0.55f * spot_b + 0.15f * dx;
            image.at(x, y) = value;
            min_value = std::min(min_value, value);
            max_value = std::max(max_value, value);
        }
    }

    const float range = max_value - min_value;
    if (range <= 0.0f) {
        return image;
    }
    for (float& value : image.pixels) {
        value = (value - min_value) / range;
    }
    return image;
}

Image downsample_half(const Image& image) {
    if (image.empty()) {
        throw std::invalid_argument("image must not be empty");
    }
    const int output_width = std::max(1, (image.width + 1) / 2);
    const int output_height = std::max(1, (image.height + 1) / 2);
    Image output(output_width, output_height, image.channels);

    for (int y = 0; y < output_height; ++y) {
        for (int x = 0; x < output_width; ++x) {
            for (int c = 0; c < image.channels; ++c) {
                float sum = 0.0f;
                int count = 0;
                for (int dy = 0; dy < 2; ++dy) {
                    for (int dx = 0; dx < 2; ++dx) {
                        const int sx = std::min(image.width - 1, 2 * x + dx);
                        const int sy = std::min(image.height - 1, 2 * y + dy);
                        sum += image.at(sx, sy, c);
                        ++count;
                    }
                }
                output.at(x, y, c) = sum / static_cast<float>(count);
            }
        }
    }

    return output;
}

}  // namespace registration
