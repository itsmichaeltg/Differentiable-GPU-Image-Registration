#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace registration {

enum class BoundaryMode {
    Zero,
    Clamp,
    Wrap
};

std::string boundary_mode_name(BoundaryMode mode);

class Image {
public:
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<float> pixels;

    Image() = default;
    Image(int image_width, int image_height, int image_channels, float fill = 0.0f);
    Image(int image_width, int image_height, int image_channels, std::vector<float> values);

    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] bool empty() const;
    [[nodiscard]] bool same_shape(const Image& other) const;
    [[nodiscard]] std::size_t index(int x, int y, int channel = 0) const;

    float& at(int x, int y, int channel = 0);
    const float& at(int x, int y, int channel = 0) const;
};

void validate_same_shape(const Image& lhs, const Image& rhs);

float sample_bilinear(
    const Image& image,
    float x,
    float y,
    int channel = 0,
    BoundaryMode boundary = BoundaryMode::Zero,
    float fill_value = 0.0f);

Image make_constant(int width, int height, int channels, float value);
Image make_gradient(int width, int height);
Image make_checkerboard(int width, int height, int tile_size);

}  // namespace registration
