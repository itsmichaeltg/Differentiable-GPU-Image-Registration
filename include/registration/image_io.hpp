#pragma once

#include <filesystem>

#include "registration/image.hpp"

namespace registration {

Image read_pnm(const std::filesystem::path& path);
void write_pnm(const Image& image, const std::filesystem::path& path);

}  // namespace registration
