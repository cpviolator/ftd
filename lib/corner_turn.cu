#include <corner_turn.h>

namespace ggp {

  static constexpr int TILE_DIM = 32;
  static constexpr int BLOCK_ROWS = 8;

  /**
   * @brief 3D corner-turn kernel: transposes the first two dimensions of a
   *        [Nf, Nb, Nt] array to produce [Nb, Nf, Nt].
   *
   * Each block handles one 32x32 tile in the (Nf, Nb) plane for a single
   * time step (given by blockIdx.z). Uses shared memory with +1 column
   * padding to avoid bank conflicts on the transposed read.
   *
   * Grid:  ((Nb + 31) / 32, (Nf + 31) / 32, Nt)
   * Block: (32, 8, 1)
   *
   * Each thread loads 4 elements (strided by BLOCK_ROWS) from input into
   * shared memory, then writes 4 elements (strided) from the transposed
   * tile into output.
   */
  __global__ void __launch_bounds__(TILE_DIM * BLOCK_ROWS)
  corner_turn_kernel(float * __restrict__ output,
                     const float * __restrict__ input,
                     int n_channels, int n_beams, int n_time)
  {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    const int t = blockIdx.z;

    // Input coordinates: row = channel (f), col = beam (b)
    const int in_col = blockIdx.x * TILE_DIM + threadIdx.x; // beam index
    const int in_row_base = blockIdx.y * TILE_DIM + threadIdx.y; // channel base

    // Load tile from input[f, b, t] into shared memory
    // input index = f * (Nb * Nt) + b * Nt + t
    #pragma unroll
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
      int f = in_row_base + i;
      if (f < n_channels && in_col < n_beams) {
        tile[threadIdx.y + i][threadIdx.x] = input[f * n_beams * n_time + in_col * n_time + t];
      }
    }

    __syncthreads();

    // Output coordinates: row = beam (b), col = channel (f) (transposed)
    const int out_col = blockIdx.y * TILE_DIM + threadIdx.x; // channel index
    const int out_row_base = blockIdx.x * TILE_DIM + threadIdx.y; // beam base

    // Store transposed tile to output[b, f, t]
    // output index = b * (Nf * Nt) + f * Nt + t
    #pragma unroll
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
      int b = out_row_base + i;
      if (b < n_beams && out_col < n_channels) {
        output[b * n_channels * n_time + out_col * n_time + t] = tile[threadIdx.x][threadIdx.y + i];
      }
    }
  }

  void corner_turn_nf_nb(float *output, const float *input,
                          int n_channels, int n_beams, int n_time,
                          cudaStream_t stream)
  {
    dim3 block(TILE_DIM, BLOCK_ROWS, 1);
    dim3 grid((n_beams + TILE_DIM - 1) / TILE_DIM,
              (n_channels + TILE_DIM - 1) / TILE_DIM,
              n_time);

    corner_turn_kernel<<<grid, block, 0, stream>>>(output, input,
                                                    n_channels, n_beams, n_time);
  }

} // namespace ggp
