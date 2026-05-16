#include "registration/image_io.hpp"

#include <filesystem>
#include <fstream>

#include "test_common.hpp"

using namespace registration;

int main() {
    const std::filesystem::path temp = std::filesystem::temp_directory_path() / "registration_io_test.pgm";

    Image image(2, 2, 1);
    image.at(0, 0) = 0.0f;
    image.at(1, 0) = 0.25f;
    image.at(0, 1) = 0.5f;
    image.at(1, 1) = 1.0f;
    write_pnm(image, temp);

    const Image loaded = read_pnm(temp);
    CHECK_EQ(loaded.width, 2);
    CHECK_EQ(loaded.height, 2);
    CHECK_EQ(loaded.channels, 1);
    CHECK_NEAR(loaded.at(0, 0), 0.0f, 1.0f / 255.0f);
    CHECK_NEAR(loaded.at(1, 1), 1.0f, 1.0f / 255.0f);

    const std::filesystem::path comment_file =
        std::filesystem::temp_directory_path() / "registration_io_comments.ppm";
    {
        std::ofstream out(comment_file);
        out << "P3\n# comment\n2 1\n255\n255 0 0 0 255 0\n";
    }
    const Image rgb = read_pnm(comment_file);
    CHECK_EQ(rgb.channels, 3);
    CHECK_NEAR(rgb.at(0, 0, 0), 1.0f, 1.0e-6f);
    CHECK_NEAR(rgb.at(1, 0, 1), 1.0f, 1.0e-6f);

    CHECK_THROWS(read_pnm(std::filesystem::temp_directory_path() / "does_not_exist.pgm"));
    CHECK_THROWS(write_pnm(Image(1, 1, 2), temp));

    std::filesystem::remove(temp);
    std::filesystem::remove(comment_file);

    // PNG round-trip: grayscale
    const std::filesystem::path png_gray =
        std::filesystem::temp_directory_path() / "registration_io_test_gray.png";
    Image gray(2, 2, 1);
    gray.at(0, 0) = 0.0f;
    gray.at(1, 0) = 0.25f;
    gray.at(0, 1) = 0.5f;
    gray.at(1, 1) = 1.0f;
    write_png(gray, png_gray);
    const Image loaded_gray = read_png(png_gray);
    CHECK_EQ(loaded_gray.width, 2);
    CHECK_EQ(loaded_gray.height, 2);
    CHECK_EQ(loaded_gray.channels, 1);
    CHECK_NEAR(loaded_gray.at(0, 0), 0.0f, 1.0f / 255.0f);
    CHECK_NEAR(loaded_gray.at(1, 1), 1.0f, 1.0f / 255.0f);
    std::filesystem::remove(png_gray);

    // PNG round-trip: RGB
    const std::filesystem::path png_rgb =
        std::filesystem::temp_directory_path() / "registration_io_test_rgb.png";
    Image rgb_out(2, 1, 3);
    rgb_out.at(0, 0, 0) = 1.0f;
    rgb_out.at(1, 0, 1) = 1.0f;
    write_png(rgb_out, png_rgb);
    const Image loaded_rgb = read_png(png_rgb);
    CHECK_EQ(loaded_rgb.channels, 3);
    CHECK_NEAR(loaded_rgb.at(0, 0, 0), 1.0f, 1.0f / 255.0f);
    CHECK_NEAR(loaded_rgb.at(1, 0, 1), 1.0f, 1.0f / 255.0f);
    std::filesystem::remove(png_rgb);

    // read_image / write_image dispatch
    const std::filesystem::path dispatch_png =
        std::filesystem::temp_directory_path() / "registration_io_dispatch.png";
    write_image(gray, dispatch_png);
    const Image dispatched = read_image(dispatch_png);
    CHECK_EQ(dispatched.channels, 1);
    CHECK_NEAR(dispatched.at(1, 1), 1.0f, 1.0f / 255.0f);
    std::filesystem::remove(dispatch_png);

    return 0;
}
