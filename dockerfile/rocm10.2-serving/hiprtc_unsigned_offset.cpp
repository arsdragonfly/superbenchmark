// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <hip/hiprtc.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void check(hiprtcResult result) {
    if (result != HIPRTC_SUCCESS) {
        throw std::runtime_error(hiprtcGetErrorString(result));
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: hiprtc_unsigned_offset output.hsaco\n";
        return 2;
    }
    const char* source = R"(
extern "C" __global__ void read_u32(float* out, const float* src, unsigned offset) {
    unsigned lane = threadIdx.x;
    unsigned index = lane + offset;
    out[lane] = src[index];
}
)";
    hiprtcProgram program = nullptr;
    try {
        check(hiprtcCreateProgram(&program, source, "unsigned-offset.hip", 0, nullptr, nullptr));
        const char* options[] = {"--gpu-architecture=gfx1250", "-O3"};
        hiprtcResult compiled = hiprtcCompileProgram(program, 2, options);
        size_t log_size = 0;
        check(hiprtcGetProgramLogSize(program, &log_size));
        if (log_size > 1) {
            std::vector<char> log(log_size);
            check(hiprtcGetProgramLog(program, log.data()));
            std::cerr << log.data() << '\n';
        }
        check(compiled);
        size_t size = 0;
        check(hiprtcGetCodeSize(program, &size));
        if (size == 0) {
            throw std::runtime_error("HIPRTC produced an empty code object");
        }
        std::vector<char> code(size);
        check(hiprtcGetCode(program, code.data()));
        check(hiprtcDestroyProgram(&program));
        std::ofstream output(argv[1], std::ios::binary);
        output.write(code.data(), code.size());
        output.close();
        if (!output) {
            throw std::runtime_error("Cannot write output code object");
        }
        std::cout << "HIPRTC compiled " << size << " bytes\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        if (program != nullptr) {
            hiprtcResult result = hiprtcDestroyProgram(&program);
            if (result != HIPRTC_SUCCESS) {
                std::cerr << "Destroy program: " << hiprtcGetErrorString(result) << '\n';
            }
        }
        return 1;
    }
    return 0;
}
