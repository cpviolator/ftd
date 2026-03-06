// --- kernels_candidate.hpp ---
// Candidate search kernels: reduction, boxcar, precomputation.
// Textual include — no #pragma once.

// --- ADDED: Functor for Dynamic Scaling Reduction ---
struct AbsMaxFunctor {
    __host__ __device__ float operator()(const float& x) const {
        return fabsf(x);
    }
    // Overload for double if needed, though we reduce to float for scale calc
    __host__ __device__ float operator()(const double& x) const {
        return (float)fabs(x);
    }
};

  // --- PATCH: GPU Search Kernels ---
struct __align__(16) DeviceCandidate {
    float max_snr;
    float max_intensity;
    int dm_idx;
    int time_idx;
    int width;
    int pad[3]; // Explicit padding to reach 32 bytes (power of 2 is safer for bandwidth)
  };

__global__ void kernel_reset_candidates(DeviceCandidate* candidates, int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    // Initialize to negative infinity so any real data (even negative SNR) overwrites it
    candidates[idx].max_snr = -1.0e9f;
    candidates[idx].max_intensity = 0.0f;
    candidates[idx].dm_idx = -1;
    candidates[idx].time_idx = -1;
    candidates[idx].width = -1;
  }
}

// 2. Fused Boxcar and Global Max Finder
// Grid: (BatchSize, NumDmTrials)
// Block: (Threads covering TimeSamples)
template <typename Real>
__global__ void kernel_find_best_candidate(
    const Real* __restrict__ dedispersed_data, // [Batch, Ndm, Nt]
    DeviceCandidate* candidates,               // [Batch]
    int Ndm, int Nt,
    int width,
    float noise_mean, float noise_stddev)
{
    // Indexes
    int b = blockIdx.x;
    int dm = blockIdx.y;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Local registers for finding max in this thread's workload
    float local_max_val = -1e9f;
    int local_max_t = -1;

    // Offset to the specific DM trial row
    size_t row_offset = (size_t)b * Ndm * Nt + (size_t)dm * Nt;

    // 1. Boxcar Convolution Loop
    // Each thread processes multiple time bins if Nt > BlockDim
    for (int t = tid; t <= Nt - width; t += stride) {
        float sum = 0.0f;

        // Simple Boxcar Sum
        for (int w = 0; w < width; ++w) {
            sum += dedispersed_data[row_offset + t + w];
        }

        // Check Max
        if (sum > local_max_val) {
            local_max_val = sum;
            local_max_t = t;
        }
    }

    // 2. Warp/Block Reduction to find Max in this DM row
    // (Using basic shared memory reduction)
    __shared__ float s_vals[256];
    __shared__ int s_locs[256];

    // Initialize shared
    s_vals[tid] = local_max_val;
    s_locs[tid] = local_max_t;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_vals[tid + s] > s_vals[tid]) {
                s_vals[tid] = s_vals[tid + s];
                s_locs[tid] = s_locs[tid + s];
            }
        }
        __syncthreads();
    }

    // 3. Update Global Candidate (Check vs Previous Widths)
    if (tid == 0) {
        float mean_val = noise_mean * width;
        float std_val = noise_stddev * sqrtf((float)width);
        float snr = (s_vals[0] - mean_val) / std_val;

        int out_idx = b * Ndm + dm;

        // Read existing best SNR (Assume initialized to -1.0)
        float current_best_snr = candidates[out_idx].max_snr;

        if (snr > current_best_snr) {
            candidates[out_idx].max_snr = snr;
            candidates[out_idx].max_intensity = s_vals[0];
            candidates[out_idx].dm_idx = dm;
            candidates[out_idx].time_idx = s_locs[0];
            candidates[out_idx].width = width;
        }
    }
}

// Kernel to generate Time Delays and F_k values directly on GPU
template <typename Real>
__global__ void kernel_generate_precomp(
    Real* d_time_delays, // [Ndm, Nf]
    Real* d_f_k,         // [Nt_complex]
    int Ndm, int Nf, int Nt_complex,
    Real f_min, Real f_max, Real min_dm_search, Real max_dm_search,
    Real time_res, Real dispersion_constant)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_delays = Ndm * Nf;

    // 1. Generate Time Delays [Ndm, Nf]
    if (idx < total_delays) {
        int dm_idx = idx / Nf;
        int f_idx = idx % Nf;

        Real d_freq_MHz = (f_max - f_min) / Nf;
        Real f_current_GHz = (f_min + f_idx * d_freq_MHz) / 1000.0;
        Real f_ref_GHz = f_max / 1000.0;

        Real current_dm = (Ndm > 1)
            ? min_dm_search + (static_cast<Real>(dm_idx) / (Ndm - 1)) * (max_dm_search - min_dm_search)
            : min_dm_search;

        Real inv_f2 = 1.0 / (f_current_GHz * f_current_GHz);
        Real inv_ref2 = 1.0 / (f_ref_GHz * f_ref_GHz);

        d_time_delays[idx] = dispersion_constant * current_dm * (inv_f2 - inv_ref2);
    }

    // 2. Generate F_k [Nt_complex] (Use strided loop if grid is small)
    if (idx < Nt_complex) {
        d_f_k[idx] = static_cast<Real>(idx) / (2.0 * (Nt_complex - 1) * time_res);
        // Note: Exact formula depends on padding, using simple approximation for patch
        // Correct is: k / (Nt_padded * time_res)
    }
}
