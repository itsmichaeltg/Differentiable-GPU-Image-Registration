#include <filesystem>
#include <fstream>
#include <iostream>

#include "registration/cpu_registration.hpp"
#include "registration/image_io.hpp"

using namespace registration;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " OUTPUT_DIR [width height]\n";
        return 2;
    }

    const std::filesystem::path output_dir = argv[1];
    const int width = argc >= 3 ? std::stoi(argv[2]) : 160;
    const int height = argc >= 4 ? std::stoi(argv[3]) : 128;

    try {
        std::filesystem::create_directories(output_dir);
        const Image source = make_registration_pattern(width, height);
        const TransformParams truth{0.08f, 1.0f, 8.0f, -5.0f};
        const Image target = warp_affine_cpu(source, width, height, truth, BoundaryMode::Clamp);

        write_pnm(source, output_dir / "source.pgm");
        write_pnm(target, output_dir / "target.pgm");

        std::ofstream truth_file(output_dir / "truth.txt");
        truth_file << "theta " << truth.theta << "\n";
        truth_file << "scale " << truth.scale << "\n";
        truth_file << "tx " << truth.tx << "\n";
        truth_file << "ty " << truth.ty << "\n";

        std::cout << "wrote " << (output_dir / "source.pgm") << "\n";
        std::cout << "wrote " << (output_dir / "target.pgm") << "\n";
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
