#pragma once

#include <filesystem>

#include "registration/image.hpp"

namespace registration {

Image read_pnm(const std::filesystem::path& path);
void write_pnm(const Image& image, const std::filesystem::path& path);

Image read_png(const std::filesystem::path& path);
void write_png(const Image& image, const std::filesystem::path& path);

// Dispatch by file extension: .png → PNG, everything else → PNM.
Image read_image(const std::filesystem::path& path);
void write_image(const Image& image, const std::filesystem::path& path);

}  // namespace registration
