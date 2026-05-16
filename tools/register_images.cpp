#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "registration/cpu_registration.hpp"
#include "registration/image_io.hpp"
#include "registration/optimizer.hpp"

using namespace registration;

namespace {

void print_usage(const char* program) {
    std::cerr
        << "Usage: " << program << " SOURCE.pgm TARGET.pgm [options]\n"
        << "Options:\n"
        << "  --output PATH              Write aligned image (default: aligned.pgm)\n"
        << "  --iterations N             Iterations per pyramid level (default: 150)\n"
        << "  --pyramid N                Pyramid levels (default: 1)\n"
        << "  --mode translation|rigid|similarity\n"
        << "  --boundary zero|clamp|wrap\n"
        << "  --lr-translation VALUE     Translation learning rate\n"
        << "  --lr-theta VALUE           Rotation learning rate\n"
        << "  --lr-scale VALUE           Scale learning rate\n"
        << "  --frames DIR               Write alignment frames as PNM files\n"
        << "  --frame-stride N           Frame callback interval (default: 10)\n";
}

BoundaryMode parse_boundary(const std::string& value) {
    if (value == "zero") {
        return BoundaryMode::Zero;
    }
    if (value == "clamp") {
        return BoundaryMode::Clamp;
    }
    if (value == "wrap") {
        return BoundaryMode::Wrap;
    }
    throw std::invalid_argument("unknown boundary mode: " + value);
}

std::string frame_name(const std::filesystem::path& directory, int frame) {
    std::ostringstream name;
    name << "frame_" << std::setw(5) << std::setfill('0') << frame << ".pgm";
    return (directory / name.str()).string();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 2;
    }

    std::filesystem::path source_path = argv[1];
    std::filesystem::path target_path = argv[2];
    std::filesystem::path output_path = "aligned.pgm";
    std::filesystem::path frame_dir;
    bool write_frames = false;

    OptimizerOptions options;
    options.boundary = BoundaryMode::Clamp;
    options.callback_interval = 0;

    try {
        for (int i = 3; i < argc; ++i) {
            const std::string arg = argv[i];
            auto need_value = [&](const std::string& name) -> std::string {
                if (i + 1 >= argc) {
                    throw std::invalid_argument(name + " requires a value");
                }
                return argv[++i];
            };

            if (arg == "--output") {
                output_path = need_value(arg);
            } else if (arg == "--iterations") {
                options.max_iterations = std::stoi(need_value(arg));
            } else if (arg == "--pyramid") {
                options.pyramid_levels = std::stoi(need_value(arg));
            } else if (arg == "--boundary") {
                options.boundary = parse_boundary(need_value(arg));
            } else if (arg == "--lr-translation") {
                options.learning_rate_translation = std::stof(need_value(arg));
            } else if (arg == "--lr-theta") {
                options.learning_rate_theta = std::stof(need_value(arg));
            } else if (arg == "--lr-scale") {
                options.learning_rate_scale = std::stof(need_value(arg));
            } else if (arg == "--mode") {
                const std::string mode = need_value(arg);
                if (mode == "translation") {
                    options.optimize_rotation = false;
                    options.optimize_scale = false;
                } else if (mode == "rigid") {
                    options.optimize_rotation = true;
                    options.optimize_scale = false;
                } else if (mode == "similarity") {
                    options.optimize_rotation = true;
                    options.optimize_scale = true;
                } else {
                    throw std::invalid_argument("unknown mode: " + mode);
                }
            } else if (arg == "--frames") {
                frame_dir = need_value(arg);
                write_frames = true;
            } else if (arg == "--frame-stride") {
                options.callback_interval = std::stoi(need_value(arg));
            } else {
                throw std::invalid_argument("unknown option: " + arg);
            }
        }

        if (write_frames && options.callback_interval <= 0) {
            options.callback_interval = 10;
        }

        const Image source = read_pnm(source_path);
        const Image target = read_pnm(target_path);

        int frame = 0;
        IterationCallback callback = nullptr;
        if (write_frames) {
            std::filesystem::create_directories(frame_dir);
            callback = [&](const IterationRecord&, const Image& aligned) {
                write_pnm(aligned, frame_name(frame_dir, frame++));
            };
        }

        const RegistrationResult result =
            align_images(source, target, TransformParams{}, options, callback);
        write_pnm(result.aligned, output_path);

        std::cout << "initial_loss=" << result.initial_loss << "\n";
        std::cout << "final_loss=" << result.final_loss << "\n";
        std::cout << "iterations=" << result.iterations << "\n";
        std::cout << "theta=" << result.params.theta << "\n";
        std::cout << "scale=" << result.params.scale << "\n";
        std::cout << "tx=" << result.params.tx << "\n";
        std::cout << "ty=" << result.params.ty << "\n";
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
