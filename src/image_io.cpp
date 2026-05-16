#include "registration/image_io.hpp"

#include <png.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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

Image read_png(const std::filesystem::path& path) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        throw std::runtime_error("failed to open input image: " + path.string());
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        std::fclose(fp);
        throw std::runtime_error("png_create_read_struct failed");
    }
    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, nullptr, nullptr);
        std::fclose(fp);
        throw std::runtime_error("png_create_info_struct failed");
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, nullptr);
        std::fclose(fp);
        throw std::runtime_error("libpng error reading: " + path.string());
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    const int width = static_cast<int>(png_get_image_width(png, info));
    const int height = static_cast<int>(png_get_image_height(png, info));
    const png_byte color_type = png_get_color_type(png, info);
    const png_byte bit_depth = png_get_bit_depth(png, info);

    // Normalize to 8-bit RGB or grayscale.
    if (bit_depth == 16) {
        png_set_strip_16(png);
    }
    if (color_type == PNG_COLOR_TYPE_PALETTE) {
        png_set_palette_to_rgb(png);
    }
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
        png_set_expand_gray_1_2_4_to_8(png);
    }
    if (png_get_valid(png, info, PNG_INFO_tRNS)) {
        png_set_tRNS_to_alpha(png);
    }
    // Drop alpha channel.
    if (color_type & PNG_COLOR_MASK_ALPHA) {
        png_set_strip_alpha(png);
    }
    png_read_update_info(png, info);

    const int channels = static_cast<int>(png_get_channels(png, info));
    const std::size_t row_bytes = png_get_rowbytes(png, info);

    std::vector<png_byte> buffer(static_cast<std::size_t>(height) * row_bytes);
    std::vector<png_bytep> rows(static_cast<std::size_t>(height));
    for (int y = 0; y < height; ++y) {
        rows[static_cast<std::size_t>(y)] =
            buffer.data() + static_cast<std::size_t>(y) * row_bytes;
    }
    png_read_image(png, rows.data());
    png_destroy_read_struct(&png, &info, nullptr);
    std::fclose(fp);

    Image image(width, height, channels);
    for (int y = 0; y < height; ++y) {
        const png_byte* row = rows[static_cast<std::size_t>(y)];
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                image.at(x, y, c) =
                    static_cast<float>(row[x * channels + c]) / 255.0f;
            }
        }
    }
    return image;
}

void write_png(const Image& image, const std::filesystem::path& path) {
    if (image.empty()) {
        throw std::invalid_argument("cannot write an empty image");
    }
    if (image.channels != 1 && image.channels != 3) {
        throw std::invalid_argument("PNG output supports one or three channels");
    }

    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) {
        throw std::runtime_error("failed to open output image: " + path.string());
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        std::fclose(fp);
        throw std::runtime_error("png_create_write_struct failed");
    }
    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_write_struct(&png, nullptr);
        std::fclose(fp);
        throw std::runtime_error("png_create_info_struct failed");
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        std::fclose(fp);
        throw std::runtime_error("libpng error writing: " + path.string());
    }

    png_init_io(png, fp);

    const int color_type = image.channels == 1 ? PNG_COLOR_TYPE_GRAY : PNG_COLOR_TYPE_RGB;
    png_set_IHDR(png, info, static_cast<png_uint_32>(image.width),
                 static_cast<png_uint_32>(image.height), 8, color_type,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    const int stride = image.width * image.channels;
    std::vector<png_byte> row(static_cast<std::size_t>(stride));
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            for (int c = 0; c < image.channels; ++c) {
                row[static_cast<std::size_t>(x * image.channels + c)] =
                    static_cast<png_byte>(quantize(image.at(x, y, c)));
            }
        }
        png_write_row(png, row.data());
    }

    png_write_end(png, nullptr);
    png_destroy_write_struct(&png, &info);
    std::fclose(fp);
}

Image read_image(const std::filesystem::path& path) {
    if (path.extension() == ".png") {
        return read_png(path);
    }
    return read_pnm(path);
}

void write_image(const Image& image, const std::filesystem::path& path) {
    if (path.extension() == ".png") {
        write_png(image, path);
        return;
    }
    write_pnm(image, path);
}

}  // namespace registration
