// Simple test driver for debug_fp6_real_gemm
#include "cutlass_gemm_api.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void print_help(const char* prog) {
    printf("Usage: %s [M] [N] [K]\n", prog);
    printf("\n");
    printf("FP6/FP4 diagnostic test driver. Runs debug_fp6_real_gemm() from the PIMPL API\n");
    printf("to verify block-scaled GEMM correctness at a single problem size.\n");
    printf("\n");
    printf("Positional arguments:\n");
    printf("  M N K        Matrix dimensions (default: 128 128 128)\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s                    # default 128x128x128\n", prog);
    printf("  %s 256 256 256        # 256^3 problem\n", prog);
    printf("  %s 1024 1024 512      # rectangular\n", prog);
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            cutlass_gemm_api::CutlassComplexGemm::print_build_info();
            printf("\n");
            print_help(argv[0]);
            return 0;
        }
    }

    cutlass_gemm_api::CutlassComplexGemm::print_build_info();
    int M = 128, N = 128, K = 128;
    if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }
    cutlass_gemm_api::CutlassComplexGemm gemm;
    gemm.debug_fp6_real_gemm(M, N, K);
    return 0;
}
