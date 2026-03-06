#include <ggp.h>
#include <ggp_internal.h>
#include <device.h>
#include <tcc_interface.h>

#ifdef TCC_API

#include <cuda.h>
#include <cuda_runtime.h>
#include <libtcc/Correlator.h>

#include <mutex>
#include <array>
#include <stdexcept>
#include <cstdio>

// Forward declaration for CUDA target stream conversion
namespace quda { namespace target { namespace cuda {
  cudaStream_t get_stream(const qudaStream_t &stream);
} } }

// ---------------------------------------------------------------------------
// CUDA error checking
// ---------------------------------------------------------------------------
#define TCC_CUDA_CHECK(call)                                              \
  do {                                                                    \
    cudaError_t err = (call);                                             \
    if (err != cudaSuccess) {                                             \
      fprintf(stderr, "TCC CUDA error at %s:%d — %s\n",                  \
              __FILE__, __LINE__, cudaGetErrorString(err));               \
      throw std::runtime_error(cudaGetErrorString(err));                  \
    }                                                                     \
  } while (0)

namespace quda {

// ===========================================================================
// TCC Correlator LRU Cache (4 entries)
// ===========================================================================
// TCC construction triggers NVRTC compilation (~2-5 s), so caching is
// essential for repeated calls with the same parameters.

struct TCCCacheKey {
  unsigned nrReceivers;
  unsigned nrChannels;
  unsigned nrSamples;
  tcc::Format format;

  bool operator==(const TCCCacheKey &o) const {
    return nrReceivers == o.nrReceivers &&
           nrChannels  == o.nrChannels  &&
           nrSamples   == o.nrSamples   &&
           format      == o.format;
  }
};

struct TCCCacheEntry {
  TCCCacheKey key;
  std::unique_ptr<tcc::Correlator> correlator;
  unsigned age = 0; // lower = more recently used
};

static constexpr int TCC_CACHE_SIZE = 4;

struct TCCCache {
  std::array<TCCCacheEntry, TCC_CACHE_SIZE> entries;
  int count = 0;
  std::mutex mtx;

  tcc::Correlator *get_or_create(const TCCCacheKey &key) {
    std::lock_guard<std::mutex> lock(mtx);

    // Search for existing entry
    for (int i = 0; i < count; ++i) {
      if (entries[i].key == key) {
        // Mark as recently used
        for (int j = 0; j < count; ++j) entries[j].age++;
        entries[i].age = 0;
        return entries[i].correlator.get();
      }
    }

    // Not found — create new correlator
    cu::Device device(0);

    auto corr = std::make_unique<tcc::Correlator>(
        device, key.format, key.nrReceivers, key.nrChannels,
        key.nrSamples, 2 /* nrPolarizations */);

    tcc::Correlator *ptr = corr.get();

    if (count < TCC_CACHE_SIZE) {
      // Cache has room
      for (int j = 0; j < count; ++j) entries[j].age++;
      entries[count].key = key;
      entries[count].correlator = std::move(corr);
      entries[count].age = 0;
      count++;
    } else {
      // Evict oldest (highest age)
      int oldest = 0;
      for (int j = 1; j < TCC_CACHE_SIZE; ++j) {
        if (entries[j].age > entries[oldest].age) oldest = j;
      }
      for (int j = 0; j < TCC_CACHE_SIZE; ++j) entries[j].age++;
      entries[oldest].key = key;
      entries[oldest].correlator = std::move(corr);
      entries[oldest].age = 0;
      ptr = entries[oldest].correlator.get();
    }

    return ptr;
  }
};

static TCCCache &get_tcc_cache() {
  static TCCCache cache;
  return cache;
}

// ===========================================================================
// Reorder kernel: FTD layout → TCC layout
// ===========================================================================
// FTD DSA-2000 QC layout: [time][antenna][channel][pol]  (1 byte per complex)
//   Low nibble = Re (two's complement), High nibble = Im (two's complement)
//
// TCC i4 layout: [channel][time_outer][receiver][pol][time_inner]  (1 byte per complex)
//   where time_outer = n_samples / times_per_block, time_inner = times_per_block
//   Low nibble = Re (two's complement), High nibble = Im (two's complement)
//
// Both FTD and TCC use the same nibble convention (low=Re, high=Im) and the
// same two's complement encoding. This kernel only transposes dimensions and
// blocks the time axis — no nibble manipulation is needed.

__global__ void __launch_bounds__(256)
reorder_ftd_to_tcc_kernel(unsigned char *__restrict__ out,
                          const unsigned char *__restrict__ in,
                          unsigned n_time,
                          unsigned n_ant,
                          unsigned n_chan,
                          unsigned n_pol,
                          unsigned n_time_inner,
                          unsigned times_per_block)
{
  // Total elements = n_time * n_ant * n_chan * n_pol
  unsigned total = n_time * n_ant * n_chan * n_pol;
  for (unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += blockDim.x * gridDim.x)
  {
    // Decode FTD index: [time][ant][chan][pol]
    unsigned pol  = idx % n_pol;
    unsigned tmp  = idx / n_pol;
    unsigned chan = tmp % n_chan;
    tmp /= n_chan;
    unsigned ant  = tmp % n_ant;
    unsigned time = tmp / n_ant;

    // Decompose FTD time into sub-integration and sample
    unsigned ti     = time % n_time_inner;       // sub-integration index
    unsigned sample = time / n_time_inner;       // sample index (0..n_samples-1)

    // Map to TCC's expanded channel
    unsigned tcc_chan = chan * n_time_inner + ti;

    // Split sample into TCC time blocks
    unsigned time_outer = sample / times_per_block;
    unsigned time_inner = sample % times_per_block;

    // TCC index: [tcc_chan][time_outer][ant][pol][time_inner]
    unsigned n_samples    = n_time / n_time_inner;
    unsigned n_time_outer = n_samples / times_per_block;
    unsigned tcc_idx = ((((tcc_chan * n_time_outer + time_outer) * n_ant + ant) * n_pol + pol) * times_per_block + time_inner);

    // Straight copy — nibble convention matches between FTD and TCC
    out[tcc_idx] = in[idx];
  }
}

// ===========================================================================
// Output conversion kernel: TCC visibilities → FTD packed triangle
// ===========================================================================
// TCC i4 output layout: [channel][baseline][pol_y][pol_x] as int2 (integer complex)
//   where baseline index = recvX*(recvX+1)/2 + recvY  (recvY <= recvX, lower triangle)
//   and pol products are 2x2: [0][0]=XX, [0][1]=XY, [1][0]=YX, [1][1]=YY
//   Each visibility is int2: .x = Re (int32), .y = Im (int32)
//
// FTD packed triangle: [batch][baseline] as complex<float>
//   where batch = chan * n_pol * n_time_inner + pol * n_time_inner + ti
//   and baseline = r2*(r2+1)/2 + r1  (same indexing)
//
// For the TCC path we only extract the auto-pol products:
//   pol=0 → XX = tcc[0][0],  pol=1 → YY = tcc[1][1]

__global__ void __launch_bounds__(256)
tcc_vis_to_ftd_kernel(float *__restrict__ out,        // FTD triangle: complex float pairs
                      const int *__restrict__ vis,     // TCC i4 visibilities: int2 (re, im)
                      unsigned n_ant,
                      unsigned n_chan,
                      unsigned n_pol,
                      unsigned n_time_inner)
{
  unsigned n_baselines = n_ant * (n_ant + 1) / 2;

  // Total output elements = n_chan * n_pol * n_time_inner * n_baselines
  unsigned total = n_chan * n_pol * n_time_inner * n_baselines;

  for (unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += blockDim.x * gridDim.x)
  {
    // Decode FTD output index
    unsigned bl  = idx % n_baselines;
    unsigned tmp = idx / n_baselines;
    unsigned ti  = tmp % n_time_inner;
    tmp /= n_time_inner;
    unsigned pol = tmp % n_pol;
    unsigned chan_base = tmp / n_pol;

    // TCC channel index: for n_time_inner > 1, TCC channels = n_chan * n_time_inner
    unsigned tcc_chan = chan_base * n_time_inner + ti;

    // TCC visibility index: [tcc_chan][bl][pol_y][pol_x] with auto-pol: pol_y = pol_x = pol
    // Each visibility is int2 = 2 ints (re, im)
    unsigned tcc_vis_idx = ((tcc_chan * n_baselines + bl) * n_pol + pol) * n_pol + pol;

    // Read int2 from TCC and convert to float
    // TCC computes conj(A)*B (Im negated vs HERK convention A*conj(B)),
    // so negate Im to match HERK output sign.
    float re = (float)vis[tcc_vis_idx * 2 + 0];
    float im = -(float)vis[tcc_vis_idx * 2 + 1];

    // Write to FTD output (2 consecutive floats: re, im)
    out[idx * 2 + 0] = re;
    out[idx * 2 + 1] = im;
  }
}

// ===========================================================================
// Public API: correlateTCC
// ===========================================================================

void correlateTCC(void *tri_output,
                  const void *raw_input,
                  unsigned n_ant,
                  unsigned n_chan,
                  unsigned n_time,
                  unsigned n_time_inner,
                  unsigned n_pol,
                  int stream_idx)
{
  if (n_pol != 2) {
    throw std::runtime_error("TCC engine requires exactly 2 polarizations");
  }

  // Convert FTD stream index to actual cudaStream_t
  cudaStream_t stream = target::cuda::get_stream(device::get_stream(stream_idx));

  // TCC samples per channel
  unsigned n_samples = n_time / n_time_inner;

  // Check alignment: TCC requires samples to be a multiple of nrTimesPerBlock
  unsigned times_per_block = tcc::Correlator::nrTimesPerBlock(tcc::Format::i4);
  if (n_samples % times_per_block != 0) {
    char msg[256];
    snprintf(msg, sizeof(msg),
             "TCC: n_samples (%u) must be a multiple of nrTimesPerBlock (%u) for i4 format. "
             "Adjust n_time_per_payload / n_time_inner.",
             n_samples, times_per_block);
    throw std::runtime_error(msg);
  }

  // TCC treats n_time_inner as additional "channels" so all time bins are
  // processed in a single launch.
  unsigned tcc_channels = n_chan * n_time_inner;
  unsigned n_baselines = n_ant * (n_ant + 1) / 2;

  // ---- Allocate scratch buffers ----

  // Reordered input: same total bytes as raw input
  size_t input_bytes = (size_t)n_time * n_ant * n_chan * n_pol;
  void *reordered_input = nullptr;
  TCC_CUDA_CHECK(cudaMallocAsync(&reordered_input, input_bytes, stream));

  // TCC output visibilities: [tcc_channels][baselines][n_pol][n_pol] as int2 (i4 format)
  // int2 = 8 bytes per visibility (same size as complex<float>)
  size_t vis_bytes = (size_t)tcc_channels * n_baselines * n_pol * n_pol * 2 * sizeof(int);
  void *tcc_vis = nullptr;
  TCC_CUDA_CHECK(cudaMallocAsync(&tcc_vis, vis_bytes, stream));

  // Zero the visibility buffer (TCC may accumulate)
  TCC_CUDA_CHECK(cudaMemsetAsync(tcc_vis, 0, vis_bytes, stream));

  // ---- Launch reorder kernel ----
  {
    unsigned total_elems = n_time * n_ant * n_chan * n_pol;
    unsigned block = 256;
    unsigned grid = (total_elems + block - 1) / block;
    reorder_ftd_to_tcc_kernel<<<grid, block, 0, stream>>>(
        (unsigned char *)reordered_input,
        (const unsigned char *)raw_input,
        n_time, n_ant, n_chan, n_pol, n_time_inner, times_per_block);
  }

  // ---- Get or create cached TCC correlator ----
  TCCCacheKey key{n_ant, tcc_channels, n_samples, tcc::Format::i4};
  tcc::Correlator *correlator = get_tcc_cache().get_or_create(key);

  // ---- Launch TCC correlator ----
  // Convert CUDA runtime stream to driver API CUstream (binary compatible)
  correlator->launchAsync((CUstream)stream,
                          (CUdeviceptr)tcc_vis,
                          (CUdeviceptr)reordered_input,
                          false /* add */);

  // ---- Launch output conversion kernel ----
  {
    unsigned total_out = n_chan * n_pol * n_time_inner * n_baselines;
    unsigned block = 256;
    unsigned grid = (total_out + block - 1) / block;
    tcc_vis_to_ftd_kernel<<<grid, block, 0, stream>>>(
        (float *)tri_output,
        (const int *)tcc_vis,
        n_ant, n_chan, n_pol, n_time_inner);
  }

  // ---- Free scratch ----
  TCC_CUDA_CHECK(cudaFreeAsync(reordered_input, stream));
  TCC_CUDA_CHECK(cudaFreeAsync(tcc_vis, stream));
}

} // namespace quda

#else // !TCC_API

namespace quda {

void correlateTCC(void *tri_output,
                  const void *raw_input,
                  unsigned n_ant,
                  unsigned n_chan,
                  unsigned n_time,
                  unsigned n_time_inner,
                  unsigned n_pol,
                  int stream_idx)
{
  errorQuda("TCC not linked, please build with -DGGP_TCC=ON");
}

} // namespace quda

#endif // TCC_API
