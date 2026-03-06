/**
 * @file visibility_bf_kernels.cu
 * @brief CUDA kernels for the visibility beamformer pipeline.
 *
 * Kernels are defined as Arg/Functor pairs in include/kernels/visibility_bf.cuh
 * and launched through TunableKernel1D wrappers here for autotuning support.
 *
 * Note: PillboxGridScatter uses activeTuning() + preTune()/postTune() to
 * redirect atomicAdd writes to a scratch buffer during autotuning sweeps,
 * preventing corruption of the real UV grid from repeated kernel launches.
 *
 * Architecture note: TunableKernel1D and kernel_param<> live in namespace quda
 * (the local project's infrastructure namespace), so all Tunable classes are
 * defined there. The host-side wrapper functions are in namespace ggp (the
 * algorithm namespace) to match forward declarations in visibility_bf.cpp.
 */

#include <tunable_nd.h>
#include <instantiate.h>
#include <malloc_ggp.h>
#include <kernels/visibility_bf.cuh>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cmath>
#include <algorithm>

// ======================================================================
// TunableKernel1D classes (in namespace quda, where the base class lives)
// ======================================================================

namespace quda {

// Pillbox grid scatter (atomicAdd-based UV gridding)
//
// Because atomicAdd accumulates rather than overwrites, the autotuner's
// repeated apply() calls would corrupt the real output.  We redirect
// writes to a scratch buffer during tuning via activeTuning(), then
// use the real uv_grid for the final (non-tuning) launch.
class PillboxGridScatterTunable : TunableKernel1D {
protected:
  unsigned long long int N;
  const void *vis_tri;
  const void *baseline_uv_m;
  const void *freq_hz;
  void *uv_grid;
  float *tune_scratch = nullptr;
  size_t grid_bytes;
  int n_baselines;
  int Ng;
  int Nf_tile;
  int freq_offset;
  float cell_size_rad;
  unsigned int minThreads() const { return N; }
  int stream_idx;

public:
  PillboxGridScatterTunable(const void *vis_tri, const void *baseline_uv_m,
                            const void *freq_hz, void *uv_grid,
                            int n_baselines, int Ng, int Nf_tile,
                            int freq_offset, float cell_size_rad,
                            unsigned long long int N, int stream_idx) :
    TunableKernel1D(N, QUDA_CUDA_FIELD_LOCATION),
    N(N),
    vis_tri(vis_tri),
    baseline_uv_m(baseline_uv_m),
    freq_hz(freq_hz),
    uv_grid(uv_grid),
    grid_bytes(static_cast<size_t>(Nf_tile) * Ng * Ng * 2 * sizeof(float)),
    n_baselines(n_baselines),
    Ng(Ng),
    Nf_tile(Nf_tile),
    freq_offset(freq_offset),
    cell_size_rad(cell_size_rad),
    stream_idx(stream_idx)
  {
    apply(device::get_stream(stream_idx));
  }

  void preTune() override
  {
    tune_scratch = static_cast<float*>(device_malloc(grid_bytes));
    cudaMemset(tune_scratch, 0, grid_bytes);
  }

  void postTune() override
  {
    if (tune_scratch) { device_free(tune_scratch); tune_scratch = nullptr; }
  }

  void apply(const qudaStream_t &stream)
  {
    TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
    // During autotuning, scatter into scratch buffer to avoid corrupting
    // the real UV grid with accumulated junk from repeated launches.
    float *target = activeTuning() ? tune_scratch : static_cast<float*>(uv_grid);
    launch<PillboxGridScatter>(tp, stream,
                               PillboxGridScatterArg(
                                 static_cast<const float*>(vis_tri),
                                 static_cast<const float*>(baseline_uv_m),
                                 static_cast<const double*>(freq_hz),
                                 target,
                                 n_baselines, Ng, Nf_tile,
                                 freq_offset, cell_size_rad, N));
  }

  long long flops() const { return 0; }  // scatter, not compute-bound
  long long bytes() const {
    return N * (2 * sizeof(float) + 2 * sizeof(float) + sizeof(double) + 4 * sizeof(float));
  }
};

// Promote QC INT4 sign-magnitude to FP32 complex interleaved
class PromoteQcSmToFp32Tunable : TunableKernel1D {
protected:
  unsigned long long int N;
  const void *input;
  void *output;
  unsigned int minThreads() const { return N; }
  int stream_idx;

public:
  PromoteQcSmToFp32Tunable(void *output, const void *input,
                    unsigned long long int N, int stream_idx) :
    TunableKernel1D(N, QUDA_CUDA_FIELD_LOCATION),
    N(N),
    input(input),
    output(output),
    stream_idx(stream_idx)
  {
    apply(device::get_stream(stream_idx));
  }

  void apply(const qudaStream_t &stream)
  {
    TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
    launch<PromoteQcSm>(tp, stream,
                        PromoteQcSmArg(static_cast<float*>(output),
                                       static_cast<const uint8_t*>(input), N));
  }

  long long flops() const { return 0; }
  long long bytes() const { return N * (sizeof(uint8_t) + 2 * sizeof(float)); }
};

// Apply UV taper (in-place pointwise multiply)
class ApplyUvTaperTunable : TunableKernel1D {
protected:
  unsigned long long int N;
  void *uv_grid;
  const void *taper;
  int Ng;
  unsigned int minThreads() const { return N; }
  int stream_idx;

public:
  ApplyUvTaperTunable(void *uv_grid, const void *taper, int Ng,
                      unsigned long long int N, int stream_idx) :
    TunableKernel1D(N, QUDA_CUDA_FIELD_LOCATION),
    N(N),
    uv_grid(uv_grid),
    taper(taper),
    Ng(Ng),
    stream_idx(stream_idx)
  {
    apply(device::get_stream(stream_idx));
  }

  void apply(const qudaStream_t &stream)
  {
    TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
    launch<ApplyUvTaper>(tp, stream,
                         ApplyUvTaperArg(static_cast<float*>(uv_grid),
                                         static_cast<const float*>(taper),
                                         Ng, N));
  }

  long long flops() const { return 2 * N; }
  long long bytes() const { return N * (sizeof(float) + 2 * 2 * sizeof(float)); }
};

// Extract beam intensity from image plane
class ExtractBeamTunable : TunableKernel1D {
protected:
  unsigned long long int N;
  const void *image;
  const void *beam_pixels;
  void *beam_output;
  int Ng;
  int n_beam;
  float norm;
  unsigned int minThreads() const { return N; }
  int stream_idx;

public:
  ExtractBeamTunable(const void *image, const void *beam_pixels,
                     void *beam_output, int Ng, int n_beam,
                     float norm, unsigned long long int N, int stream_idx) :
    TunableKernel1D(N, QUDA_CUDA_FIELD_LOCATION),
    N(N),
    image(image),
    beam_pixels(beam_pixels),
    beam_output(beam_output),
    Ng(Ng),
    n_beam(n_beam),
    norm(norm),
    stream_idx(stream_idx)
  {
    apply(device::get_stream(stream_idx));
  }

  void apply(const qudaStream_t &stream)
  {
    TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
    launch<ExtractBeam>(tp, stream,
                        ExtractBeamArg(static_cast<const float*>(image),
                                       static_cast<const int*>(beam_pixels),
                                       static_cast<float*>(beam_output),
                                       Ng, n_beam, norm, N));
  }

  long long flops() const { return 4 * N; }
  long long bytes() const { return N * (2 * sizeof(float) + sizeof(float)); }
};

// Quantise beam intensities to 8-bit unsigned
class QuantiseBeamsTunable : TunableKernel1D {
protected:
  unsigned long long int N;
  const void *beam_intensity;
  void *output;
  float scale;
  unsigned int minThreads() const { return N; }
  int stream_idx;

public:
  QuantiseBeamsTunable(const void *beam_intensity, void *output,
                       float scale, unsigned long long int N, int stream_idx) :
    TunableKernel1D(N, QUDA_CUDA_FIELD_LOCATION),
    N(N),
    beam_intensity(beam_intensity),
    output(output),
    scale(scale),
    stream_idx(stream_idx)
  {
    apply(device::get_stream(stream_idx));
  }

  void apply(const qudaStream_t &stream)
  {
    TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
    launch<QuantiseBeams>(tp, stream,
                          QuantiseBeamsArg(static_cast<const float*>(beam_intensity),
                                           static_cast<unsigned char*>(output),
                                           scale, N));
  }

  long long flops() const { return 3 * N; }
  long long bytes() const { return N * (sizeof(float) + sizeof(unsigned char)); }
};

// Triangulate from full Hermitian matrix to packed lower triangle
class TriangulateFromHermTunable : TunableKernel1D {
protected:
  unsigned long long int N;
  const void *full_mat;
  void *tri_out;
  int mat_N;
  int batch_count;
  unsigned int minThreads() const { return N; }
  int stream_idx;

public:
  TriangulateFromHermTunable(const void *full_mat, void *tri_out,
                             int mat_N, int batch_count,
                             unsigned long long int N, int stream_idx) :
    TunableKernel1D(N, QUDA_CUDA_FIELD_LOCATION),
    N(N),
    full_mat(full_mat),
    tri_out(tri_out),
    mat_N(mat_N),
    batch_count(batch_count),
    stream_idx(stream_idx)
  {
    apply(device::get_stream(stream_idx));
  }

  void apply(const qudaStream_t &stream)
  {
    TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
    launch<TriangulateFromHerm>(tp, stream,
                                TriangulateFromHermArg(
                                  static_cast<const float*>(full_mat),
                                  static_cast<float*>(tri_out),
                                  mat_N, batch_count, N));
  }

  long long flops() const { return 0; }
  long long bytes() const {
    return N * 4 * sizeof(float);
  }
};

// Convert QC sign-magnitude (high=Re, low=Im) to TCC two's complement (low=Re, high=Im)
// In-place transform requires preTune/postTune to save/restore data during autotuning.
class ConvertQcSmToFtdTunable : TunableKernel1D {
protected:
  unsigned long long int N;
  void *data;
  void *tune_backup = nullptr;
  unsigned int minThreads() const { return N; }
  int stream_idx;

public:
  ConvertQcSmToFtdTunable(void *data, unsigned long long int N, int stream_idx) :
    TunableKernel1D(N, QUDA_CUDA_FIELD_LOCATION),
    N(N),
    data(data),
    stream_idx(stream_idx)
  {
    apply(device::get_stream(stream_idx));
  }

  void preTune() override
  {
    tune_backup = device_malloc(N);
    cudaMemcpy(tune_backup, data, N, cudaMemcpyDeviceToDevice);
  }

  void postTune() override
  {
    cudaMemcpy(data, tune_backup, N, cudaMemcpyDeviceToDevice);
    if (tune_backup) { device_free(tune_backup); tune_backup = nullptr; }
  }

  void apply(const qudaStream_t &stream)
  {
    TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
    launch<ConvertQcSmToFtd>(tp, stream,
                              ConvertQcSmToFtdArg(static_cast<uint8_t*>(data), N));
  }

  long long flops() const { return 0; }
  long long bytes() const { return 2 * N; }  // read + write each byte
};

} // namespace quda


// ======================================================================
// Host-side launch wrappers in namespace ggp
// (matching forward declarations in visibility_bf.cpp)
// ======================================================================

namespace ggp {

void pillboxGridScatter(const void *vis_tri, const void *baseline_uv_m,
                        const void *freq_hz, void *uv_grid,
                        int n_baselines, int Ng, int Nf_tile,
                        int freq_offset, float cell_size_rad, int stream_idx)
{
  unsigned long long int N = static_cast<unsigned long long int>(Nf_tile) * n_baselines;
  if (N == 0) return;
  quda::getProfile().TPSTART(quda::QUDA_PROFILE_COMPUTE);
  quda::PillboxGridScatterTunable(vis_tri, baseline_uv_m, freq_hz, uv_grid,
                            n_baselines, Ng, Nf_tile, freq_offset,
                            cell_size_rad, N, stream_idx);
  quda::getProfile().TPSTOP(quda::QUDA_PROFILE_COMPUTE);
}

void promoteQcSmToFp32(void *output, const void *input,
                       unsigned long long int N, int stream_idx)
{
  if (N == 0) return;
  quda::getProfile().TPSTART(quda::QUDA_PROFILE_COMPUTE);
  quda::PromoteQcSmToFp32Tunable(output, input, N, stream_idx);
  quda::getProfile().TPSTOP(quda::QUDA_PROFILE_COMPUTE);
}

void applyUvTaper(void *uv_grid, const void *taper,
                  int Ng, int Nf_tile, int stream_idx)
{
  unsigned long long int N = static_cast<unsigned long long int>(Nf_tile) * Ng * Ng;
  if (N == 0) return;
  quda::getProfile().TPSTART(quda::QUDA_PROFILE_COMPUTE);
  quda::ApplyUvTaperTunable(uv_grid, taper, Ng, N, stream_idx);
  quda::getProfile().TPSTOP(quda::QUDA_PROFILE_COMPUTE);
}

void extractBeamIntensity(const void *image, const void *beam_pixels,
                          void *beam_output, int Ng, int n_beam,
                          int Nf_tile, float norm, int stream_idx)
{
  unsigned long long int N = static_cast<unsigned long long int>(Nf_tile) * n_beam;
  if (N == 0) return;
  quda::getProfile().TPSTART(quda::QUDA_PROFILE_COMPUTE);
  quda::ExtractBeamTunable(image, beam_pixels, beam_output, Ng, n_beam, norm, N, stream_idx);
  quda::getProfile().TPSTOP(quda::QUDA_PROFILE_COMPUTE);
}

void quantiseBeams(const void *beam_intensity, void *output,
                   unsigned long long int N, float scale, int stream_idx)
{
  if (N == 0) return;
  quda::getProfile().TPSTART(quda::QUDA_PROFILE_COMPUTE);
  quda::QuantiseBeamsTunable(beam_intensity, output, scale, N, stream_idx);
  quda::getProfile().TPSTOP(quda::QUDA_PROFILE_COMPUTE);
}

void triangulateFromHermVis(const void *full_mat, void *tri_out,
                            int mat_N, int batch_count, int stream_idx)
{
  int n_baselines = mat_N * (mat_N + 1) / 2;
  unsigned long long int N = static_cast<unsigned long long int>(batch_count) * n_baselines;
  if (N == 0) return;
  quda::getProfile().TPSTART(quda::QUDA_PROFILE_COMPUTE);
  quda::TriangulateFromHermTunable(full_mat, tri_out, mat_N, batch_count, N, stream_idx);
  quda::getProfile().TPSTOP(quda::QUDA_PROFILE_COMPUTE);
}

void convertQcSmToFtd(void *data, unsigned long long int N, int stream_idx)
{
  if (N == 0) return;
  quda::getProfile().TPSTART(quda::QUDA_PROFILE_COMPUTE);
  quda::ConvertQcSmToFtdTunable(data, N, stream_idx);
  quda::getProfile().TPSTOP(quda::QUDA_PROFILE_COMPUTE);
}

int cublas_herk_batched_qc(
    const float* promoted_data,   // FP32 complex interleaved [batch x N x K x 2]
    float* result_data,           // FP32 complex full [batch x N x N x 2]
    float* tri_output,            // FP32 complex packed triangle [batch x n_baselines x 2]
    int N, int K, int batch,
    cudaStream_t stream)
{
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) return -1;
    cublasSetStream(handle, stream);

    // Promoted data is [N x K] row-major per batch = [K x N] column-major with lda=K.
    // C = A^H * A where A_stored is column-major [K x N] with lda=K.
    cuComplex alpha = {1.0f, 0.0f};
    cuComplex beta  = {0.0f, 0.0f};

    long long strideA = static_cast<long long>(N) * K;
    long long strideC = static_cast<long long>(N) * N;

    stat = cublasCgemmStridedBatched(
        handle,
        CUBLAS_OP_C,    // op(A) = A^H
        CUBLAS_OP_N,    // op(B) = B
        N, N, K,        // m, n, k
        &alpha,
        reinterpret_cast<const cuComplex*>(promoted_data), K, strideA,
        reinterpret_cast<const cuComplex*>(promoted_data), K, strideA,
        &beta,
        reinterpret_cast<cuComplex*>(result_data), N, strideC,
        batch);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(handle);
        return -2;
    }

    // Extract packed lower triangle (uses TunableKernel1D internally)
    triangulateFromHermVis(result_data, tri_output, N, batch, 0);

    cudaStreamSynchronize(stream);
    cublasDestroy(handle);
    return 0;
}

} // namespace ggp
