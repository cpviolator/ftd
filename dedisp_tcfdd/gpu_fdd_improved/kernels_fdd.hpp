// --- kernels_fdd.hpp ---
// FDD pipeline CUDA kernels: data generation, transpose, GEMM, extract.
// Textual include — no #pragma once.

// --- NEW DATA GENERATION KERNEL ---

/**
 * @brief Kernel to inject dispersed "burst" signals (width=0, scattering=0)
 *
 * This kernel is launched with a 1D grid covering all (batch, frequency) pairs.
 * Each thread computes the arrival time for its (b, f) pair and adds the
 * pulsar amplitude to the correct time bin.
 */
template <typename Real>
__global__ void kernel_inject_bursts(Real* intensity_matrix,
                                     const PulsarParams<Real>* all_params,
                                     int total_items, int Nf, int Nt,
                                     Real f_min, Real f_max,
                                     Real time_resolution, Real f_ref_GHz) {

  size_t b_f = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x; // MUST BE size_t
  if (b_f >= total_items * Nf) return;

  int b = b_f / Nf;
  int f_idx = b_f % Nf;

  const PulsarParams<Real>& pulsar = all_params[b];
  const Real freq_step = (f_max - f_min) / Nf;

  Real f_current_MHz = f_min + f_idx * freq_step;
  Real f_current_GHz = f_current_MHz / 1000.0;
  Real time_delay = DISPERSION_CONSTANT * pulsar.dm *
                    (1.0 / (f_current_GHz * f_current_GHz) -
                     1.0 / (f_ref_GHz * f_ref_GHz));

  // Note: Assumes width_s=0, so pulse_profile is just [1.0]
  // This logic is for "burst" mode.
  int center_time_bin =
      static_cast<int>((pulsar.pulse_start_time + time_delay) / time_resolution);

  if (center_time_bin >= 0 && center_time_bin < Nt) {
    size_t out_idx = (size_t)b * Nf * Nt + (size_t)f_idx * Nt + center_time_bin;

    // [PATCH] CRITICAL: Ensure we do not write past the allocated buffer
    // The total size is total_items * Nf * Nt.
    size_t total_elements = (size_t)total_items * Nf * Nt;

    if (out_idx < total_elements) {
        atomicAdd(&intensity_matrix[out_idx], pulsar.amplitude);
    }
  }
}

// Kernel to simulate 8-bit unsigned integer digitization [0, 255]
// Rounds floats to nearest integer and clamps.
template <typename Real>
__global__ void kernel_quantize_u8(Real* intensity_matrix, size_t total_elements) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx >= total_elements) return;

    Real val = intensity_matrix[idx];

    // 1. Round to nearest integer
    val = round(val);

    // 2. Clamp to [0, 255]
    if (val < 0.0f) val = 0.0f;
    if (val > 255.0f) val = 255.0f;

    intensity_matrix[idx] = val;
}

// --- FDD PIPELINE KERNELS (R2C Optimized) ---

/**
 * @brief Kernel 1: Copies Real input to Real padded buffer (Pre-R2C FFT)
 */
template <typename Real>
__global__ void kernel_copy_pad_real(const Real* input, Real* output,
                                     int count, int Nf, int Nt, int Nt_padded) {
  int t = blockIdx.y * blockDim.x + threadIdx.x;
  int b_f = blockIdx.x;

  if (t >= Nt_padded) return;

  int b = b_f / Nf;
  int f = b_f % Nf;

  size_t in_idx = (size_t)b * Nf * Nt + (size_t)f * Nt + t;
  size_t out_idx = (size_t)b_f * Nt_padded + t;

  if (t < Nt) {
    output[out_idx] = input[in_idx];
  } else {
    output[out_idx] = 0.0;
  }
}

/**
 * @brief Kernel 2: Batched transpose [b][f][k] -> [b][k][f]
 * Operates on the REDUCED complex dimension (Nt_complex)
 */
template <typename Real, typename ComplexType>
__global__ void kernel_transpose_f_k(const ComplexType* input,
                                     ComplexType* output, int count, int Nf,
                                     int Nt_complex) {
  __shared__ ComplexType tile[TILE_DIM][TILE_DIM + 1];

  int b = blockIdx.z;
  int f_base = blockIdx.x * TILE_DIM;
  int k_base = blockIdx.y * TILE_DIM;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int f_in = f_base + tx;
  int k_in = k_base + ty;

  if (f_in < Nf && k_in < Nt_complex) {
    size_t in_idx = (size_t)b * Nf * Nt_complex + (size_t)f_in * Nt_complex + k_in;
    tile[ty][tx] = input[in_idx];
  }
  __syncthreads();

  int f_out = f_base + ty;
  int k_out = k_base + tx;

  if (f_out < Nf && k_out < Nt_complex) {
    size_t out_idx = (size_t)b * Nt_complex * Nf + (size_t)k_out * Nf + f_out;
    output[out_idx] = tile[tx][ty];
  }
}

/**
 * @brief Kernel 3: Core FDD Gemm.
 * Loops over Nt_complex (approx Nt/2), saving 50% compute.
 */
template <typename Real, typename ComplexType>
__global__ void kernel_fdd_gemm_transpose(const ComplexType* d_phasors,
                                          const ComplexType* d_fft_buffer_T,
                                          ComplexType* d_ifft_buffer, int count,
                                          int Nf, int Nt_complex, int Ndm) {
  int b = blockIdx.x;
  int k = blockIdx.y;
  int dm = blockIdx.z;

  ComplexType sum = {0.0, 0.0};

  for (int f = threadIdx.x; f < Nf; f += blockDim.x) {
    size_t phasor_idx = (size_t)k * Ndm * Nf + (size_t)dm * Nf + f;
    const ComplexType A_val = d_phasors[phasor_idx];

    size_t fft_idx = (size_t)b * Nt_complex * Nf + (size_t)k * Nf + f;
    const ComplexType B_val = d_fft_buffer_T[fft_idx];

    Real ac = A_val.x * B_val.x;
    Real bd = A_val.y * B_val.y;
    Real ad = A_val.x * B_val.y;
    Real bc = A_val.y * B_val.x;
    sum.x += (ac - bd);
    sum.y += (ad + bc);
  }

  __shared__ ComplexType s_sum[256];
  s_sum[threadIdx.x] = sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      s_sum[threadIdx.x].x += s_sum[threadIdx.x + s].x;
      s_sum[threadIdx.x].y += s_sum[threadIdx.x + s].y;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    size_t out_idx = (size_t)b * Ndm * Nt_complex + (size_t)dm * Nt_complex + k;
    d_ifft_buffer[out_idx] = s_sum[0];
  }
}

/**
 * @brief Kernel 4: Extracts Real output from C2R transform (IFFT).
 * C2R output is strictly Real, so we just scale.
 */
template <typename Real>
__global__ void kernel_extract_scale_real(const Real* input, Real* output,
                                          int count, int Ndm, int Nt,
                                          int Nt_padded) {
  int t = blockIdx.y * blockDim.x + threadIdx.x;
  int b_dm = blockIdx.x;

  if (t >= Nt) return;

  int b = b_dm / Ndm;
  int dm = b_dm % Ndm;

  size_t in_idx = (size_t)b * Ndm * Nt_padded + (size_t)dm * Nt_padded + t;
  size_t out_idx = (size_t)b * Ndm * Nt + (size_t)dm * Nt + t;

  Real scale = (Real)1.0 / Nt_padded;
  output[out_idx] = input[in_idx] * scale;
}

/**
 * @brief Kernel 5: Generates phasors for Nt_complex bins only.
 */
template <typename Real, typename ComplexTypeGpu>
__global__ void kernel_generate_phasors(
    ComplexTypeGpu* d_phasors,          // Output buffer [Nt_complex][Ndm][Nf]
    const Real* d_f_k_values,           // Input buffer [Nt_complex]
    const Real* d_time_delays,          // Input buffer [Ndm][Nf]
    int Nf, int Ndm, int Nt_complex,
    bool conjugate_phasors)
{
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  int dm = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z;

  if (f >= Nf || dm >= Ndm || k >= Nt_complex) return;

  double f_k_val = (double)d_f_k_values[k];
  double time_delay = (double)d_time_delays[dm * Nf + f];
  double angle = 2.0 * PI_D * f_k_val * time_delay;

  double s, c;
  sincos(angle, &s, &c);

  size_t out_idx = (size_t)k * Ndm * Nf + (size_t)dm * Nf + f;
  d_phasors[out_idx].x = c;
  if (conjugate_phasors) d_phasors[out_idx].y = s;
  else d_phasors[out_idx].y = -s;
}

// --- CUBLAS KERNELS (UPDATED FOR R2C) ---

template <typename ComplexType>
__global__ void kernel_setup_gemm_pointers(
    void** d_A_pointers, void** d_B_pointers, void** d_C_pointers,
    const ComplexType* d_phasors,       // [k, dm, f]
    const ComplexType* d_fft_buffer_T,  // [k, f, b]
    ComplexType* d_gemm_out,            // [k, dm, b]
    int count, int Nf, int Ndm, int Nt_complex)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= Nt_complex) return;

    int BatchSize = count;
    d_A_pointers[k] = (void*)(d_phasors + (size_t)k * Ndm * Nf);
    d_B_pointers[k] = (void*)(d_fft_buffer_T + (size_t)k * Nf * BatchSize);
    d_C_pointers[k] = (void*)(d_gemm_out + (size_t)k * Ndm * BatchSize);
}

// [PATCH] Optimized FP32 Transpose
// Input:  [Batch, Nf, k] (Interleaved Complex)
// Output: [k, Nf, Batch] (Interleaved Complex)
// Optimization: Coalesced Reads on 'k', Shared Mem Transpose, Coalesced Write on 'b'
template <typename ComplexType>
__global__ void kernel_transpose_bfk_to_kfb(const ComplexType* input,
                                            ComplexType* output,
                                            int batch_size, int Nf, int Nt_complex) {
  // Shared memory tile (Complex is 2 floats, so we treat as struct)
  // Pad column to avoid bank conflicts
  __shared__ ComplexType tile[32][33];

  // Grid Layout:
  // x -> k (Time) - Fastest dimension in Input
  // y -> b (Batch) - Fastest dimension in Output
  // z -> f (Frequency) - Independent planes

  int k_origin = blockIdx.x * 32;
  int b_origin = blockIdx.y * 32;
  int f = blockIdx.z;

  // --- 1. COALESCED READ (Load into Shared Mem) ---
  // Map threadIdx.x to k (contiguous in Input)
  int k_in = k_origin + threadIdx.x;
  int b_in = b_origin + threadIdx.y;

  // Input Index: [b][f][k]
  // Stride is 1 (Complex element)
  if (k_in < Nt_complex && b_in < batch_size) {
    size_t in_idx = (size_t)b_in * Nf * Nt_complex + (size_t)f * Nt_complex + k_in;
    tile[threadIdx.y][threadIdx.x] = input[in_idx];
  }

  __syncthreads();

  // --- 2. COALESCED WRITE (Store from Shared Mem) ---
  // Swap roles: threadIdx.x now handles Batch (contiguous in Output)
  int b_out = b_origin + threadIdx.x;
  int k_out = k_origin + threadIdx.y;

  // Output Index: [k][f][b]
  // Stride is 1 (Complex element)
  if (b_out < batch_size && k_out < Nt_complex) {
    size_t out_idx = (size_t)k_out * Nf * batch_size + (size_t)f * batch_size + b_out;
    output[out_idx] = tile[threadIdx.x][threadIdx.y];
  }
}

template <typename ComplexType>
__global__ void kernel_transpose_kdb_to_bdk(const ComplexType* input, // [k,dm,b]
                                            ComplexType* output,      // [b,dm,k]
                                            int count, int Ndm, int Nt_complex) {
  __shared__ ComplexType tile[TILE_DIM][TILE_DIM + 1];

  int dm = blockIdx.z;
  int b_base = blockIdx.x * TILE_DIM;
  int k_base = blockIdx.y * TILE_DIM;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int b_in = b_base + tx;
  int k_in = k_base + ty;

  if (b_in < count && k_in < Nt_complex) {
    size_t in_idx = (size_t)k_in * Ndm * count + (size_t)dm * count + b_in;
    tile[ty][tx] = input[in_idx];
  }
  __syncthreads();

  int b_out = b_base + ty;
  int k_out = k_base + tx;

  if (b_out < count && k_out < Nt_complex) {
    size_t out_idx = (size_t)b_out * Ndm * Nt_complex + (size_t)dm * Nt_complex + k_out;
    output[out_idx] = tile[tx][ty];
  }
}
