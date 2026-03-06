#pragma once

namespace quda {

/**
 * @brief Correlate raw antenna data using TCC (Tensor-Core Correlator).
 *
 * Accepts raw QC-format (INT4 sign-magnitude) input in FTD layout
 * [time][antenna][channel][pol], reorders to TCC layout, runs the
 * TCC correlator, and converts the output to FTD packed-triangle format
 * (complex float, lower-triangle, per-channel per-pol).
 *
 * @param[out] tri_output  Device pointer to packed triangle output (complex float).
 *                         Size: n_chan * n_pol * n_time_inner * n_ant*(n_ant+1)/2 * 2 * sizeof(float).
 * @param[in]  raw_input   Device pointer to raw QC data (1 byte per complex sample).
 *                         Size: n_time * n_ant * n_chan * n_pol bytes.
 * @param[in]  n_ant       Number of antennas (receivers).
 * @param[in]  n_chan      Number of frequency channels.
 * @param[in]  n_time      Total number of time samples per payload.
 * @param[in]  n_time_inner Number of fine time bins (output batches per channel*pol).
 * @param[in]  n_pol       Number of polarizations (must be 2).
 * @param[in]  stream_idx  FTD stream index (converted to cudaStream_t internally).
 */
void correlateTCC(void *tri_output,
                  const void *raw_input,
                  unsigned n_ant,
                  unsigned n_chan,
                  unsigned n_time,
                  unsigned n_time_inner,
                  unsigned n_pol,
                  int stream_idx);

} // namespace quda
