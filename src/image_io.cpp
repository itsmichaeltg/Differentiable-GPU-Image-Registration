#include "registration/image_io.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace registration {

namespace {

std::string next_token(std::istream& input) {
    std::string token;
    while (input >> token) {
        if (!token.empty() && token[0] == '#') {
            std::string ignored;
            std::getline(input, ignored);
            continue;
        }
        return token;
    }
    throw std::runtime_error("unexpected end of PNM file");
}

int parse_positive_int(const std::string& token, const char* name) {
    std::size_t consumed = 0;
    const int value = std::stoi(token, &consumed);
    if (consumed != token.size() || value <= 0) {
        throw std::runtime_error(std::string(name) + " must be a positive integer");
    }
    return value;
}

float parse_pixel(const std::string& token, int max_value) {
    std::size_t consumed = 0;
    const int value = std::stoi(token, &consumed);
    if (consumed != token.size() || value < 0 || value > max_value) {
        throw std::runtime_error("PNM pixel value out of range");
    }
    return static_cast<float>(value) / static_cast<float>(max_value);
}

int quantize(float value) {
    const float clamped = std::clamp(value, 0.0f, 1.0f);
    return static_cast<int>(std::lround(clamped * 255.0f));
}

}  // namespace

Image read_pnm(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open input image: " + path.string());
    }

    const std::string magic = next_token(input);
    const int channels = magic == "P2" ? 1 : magic == "P3" ? 3 : 0;
    if (channels == 0) {
        throw std::runtime_error("only ASCII P2/P3 PNM images are supported");
    }

    const int width = parse_positive_int(next_token(input), "width");
    const int height = parse_positive_int(next_token(input), "height");
    const int max_value = parse_positive_int(next_token(input), "max value");
    if (max_value > 65535) {
        throw std::runtime_error("PNM max value is too large");
    }

    Image image(width, height, channels);
    for (float& value : image.pixels) {
        value = parse_pixel(next_token(input), max_value);
    }
    return image;
}

void write_pnm(const Image& image, const std::filesystem::path& path) {
    if (image.empty()) {
        throw std::invalid_argument("cannot write an empty image");
    }
    if (image.channels != 1 && image.channels != 3) {
        throw std::invalid_argument("PNM output supports one or three channels");
    }

    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open output image: " + path.string());
    }

    output << (image.channels == 1 ? "P2\n" : "P3\n");
    output << image.width << " " << image.height << "\n255\n";
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            for (int c = 0; c < image.channels; ++c) {
                output << quantize(image.at(x, y, c));
                if (!(x == image.width - 1 && c == image.channels - 1)) {
                    output << ' ';
                }
            }
        }
        output << '\n';
    }
}

}  // namespace registration
