#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace ggp {

  /**
   * @brief Transpose filterbank data from channel-major to beam-major layout.
   *
   * Input  layout: [n_channels, n_beams, n_time] (channel-major, row-major)
   * Output layout: [n_beams, n_channels, n_time] (beam-major, row-major)
   *
   * Uses 32x32 shared-memory tiles with +1 padding for bank-conflict avoidance.
   * Each CUDA block transposes one 32x32 tile of the (n_channels, n_beams) plane
   * for a single time step.
   *
   * @param output      Device pointer to output array [n_beams * n_channels * n_time]
   * @param input       Device pointer to input array  [n_channels * n_beams * n_time]
   * @param n_channels  Number of frequency channels (Nf)
   * @param n_beams     Number of beams (Nb)
   * @param n_time      Number of time samples (Nt)
   * @param stream      CUDA stream (nullptr = default stream)
   */
  void corner_turn_nf_nb(float *output, const float *input,
                          int n_channels, int n_beams, int n_time,
                          cudaStream_t stream = nullptr);

} // namespace ggp
