#include <chrono>
#include <iostream>
#include <string>

#include "registration/cpu_registration.hpp"
#include "registration/gpu_registration.cuh"
#include "registration/optimizer.hpp"

using namespace registration;

namespace {

template <typename Function>
double time_ms(Function&& function) {
    const auto start = std::chrono::steady_clock::now();
    function();
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

void print_metric(const std::string& name, double milliseconds, float loss = -1.0f) {
    std::cout << name << "_ms=" << milliseconds;
    if (loss >= 0.0f) {
        std::cout << " loss=" << loss;
    }
    std::cout << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    const int size = argc >= 2 ? std::stoi(argv[1]) : 512;
    const int iterations = argc >= 3 ? std::stoi(argv[2]) : 10;
    const Image source = make_registration_pattern(size, size);
    const TransformParams truth{0.04f, 1.0f, 6.0f, -4.0f};
    const Image target = warp_affine_cpu(source, size, size, truth, BoundaryMode::Clamp);
    const Affine2D matrix = source_from_destination(truth, size, size, size, size);

    Image output;
    double cpu_warp_ms = time_ms([&] {
        for (int i = 0; i < iterations; ++i) {
            output = warp_affine_cpu(source, size, size, matrix, BoundaryMode::Clamp);
        }
    }) / static_cast<double>(iterations);
    print_metric("cpu_warp", cpu_warp_ms, mse_cpu(output, target));

    LossGradient gradient;
    double cpu_gradient_ms = time_ms([&] {
        for (int i = 0; i < iterations; ++i) {
            gradient = loss_gradient_cpu(source, target, truth, BoundaryMode::Clamp);
        }
    }) / static_cast<double>(iterations);
    print_metric("cpu_loss_gradient", cpu_gradient_ms, gradient.loss);

    std::string reason;
    if (!cuda_device_available(&reason)) {
        std::cout << "cuda_skipped=" << reason << "\n";
        return 0;
    }

    double cuda_texture_ms = time_ms([&] {
        for (int i = 0; i < iterations; ++i) {
            output = warp_affine_cuda(source, size, size, matrix, BoundaryMode::Clamp);
        }
    }) / static_cast<double>(iterations);
    print_metric("cuda_texture_warp", cuda_texture_ms, mse_cpu(output, target));

    double cuda_global_ms = time_ms([&] {
        for (int i = 0; i < iterations; ++i) {
            output = warp_affine_cuda_global(source, size, size, matrix, BoundaryMode::Clamp);
        }
    }) / static_cast<double>(iterations);
    print_metric("cuda_global_warp", cuda_global_ms, mse_cpu(output, target));

    double cuda_mse_ms = time_ms([&] {
        for (int i = 0; i < iterations; ++i) {
            volatile float loss = mse_cuda(output, target);
            (void)loss;
        }
    }) / static_cast<double>(iterations);
    print_metric("cuda_mse_reduce", cuda_mse_ms);

    double cuda_gradient_ms = time_ms([&] {
        for (int i = 0; i < iterations; ++i) {
            gradient = loss_gradient_cuda(source, target, truth, BoundaryMode::Clamp);
        }
    }) / static_cast<double>(iterations);
    print_metric("cuda_loss_gradient", cuda_gradient_ms, gradient.loss);

    return 0;
}
