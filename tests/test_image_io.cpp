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
    return 0;
}
