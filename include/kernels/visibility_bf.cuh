#pragma once

#include <comm_ggp.h>
#include <kernel.h>
#include <register_traits.h>

namespace quda {

  // ================================================================
  // Kernel 1: Promote QC INT4 sign-magnitude to FP32 complex interleaved
  //
  // DSA-2000 QC format: 1 byte per complex element
  //   high nibble = Re (sign-magnitude), low nibble = Im (sign-magnitude)
  //   Each nibble: bit 3 = sign, bits 2:0 = magnitude, range [-7, +7]
  //
  // Output: interleaved [Re, Im, Re, Im, ...] float pairs
  // Thread count N = number of complex elements (= number of input bytes)
  // ================================================================

  struct PromoteQcSmArg : kernel_param<> {
    float *output;
    const uint8_t *input;
    unsigned long long int N;

    PromoteQcSmArg(float *output, const uint8_t *input, unsigned long long int N) :
      kernel_param(dim3(N, 1, 1)),
      output(output),
      input(input),
      N(N)
    {
    }
  };

  template <typename Arg> struct PromoteQcSm {
    const Arg &arg;
    constexpr PromoteQcSm(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int idx)
    {
      uint8_t byte = arg.input[idx];
      // High nibble = Re (sign-magnitude)
      uint8_t re_raw = (byte >> 4) & 0x0F;
      int re_sign = (re_raw & 0x8) ? -1 : 1;
      int re_mag  = re_raw & 0x7;
      // Low nibble = Im (sign-magnitude)
      uint8_t im_raw = byte & 0x0F;
      int im_sign = (im_raw & 0x8) ? -1 : 1;
      int im_mag  = im_raw & 0x7;
      arg.output[2*idx]     = static_cast<float>(re_sign * re_mag);
      arg.output[2*idx + 1] = static_cast<float>(im_sign * im_mag);
    }
  };

  // ================================================================
  // Kernel 2: Apply UV taper (pointwise multiply)
  //
  // In-place multiply UV grid by real-valued taper weights.
  // The taper is [Ng x Ng] and is applied identically to each
  // frequency tile.  Each thread handles one complex grid point
  // (Re and Im both multiplied by the same weight).
  //
  // Thread count N = Nf_tile * Ng * Ng
  // ================================================================

  struct ApplyUvTaperArg : kernel_param<> {
    float *uv_grid;
    const float *taper;
    int Ng;
    unsigned long long int N;

    ApplyUvTaperArg(float *uv_grid, const float *taper, int Ng,
                    unsigned long long int N) :
      kernel_param(dim3(N, 1, 1)),
      uv_grid(uv_grid),
      taper(taper),
      Ng(Ng),
      N(N)
    {
    }
  };

  template <typename Arg> struct ApplyUvTaper {
    const Arg &arg;
    constexpr ApplyUvTaper(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int idx)
    {
      const int grid_plane = arg.Ng * arg.Ng;
      const int pixel = idx % grid_plane;
      const float w = arg.taper[pixel];

      const int base = idx * 2;
      arg.uv_grid[base]     *= w;
      arg.uv_grid[base + 1] *= w;
    }
  };

  // ================================================================
  // Kernel 3: Extract beam intensity from image plane
  //
  // For each (frequency tile, beam) pair, gather the pixel from the
  // image plane at the beam's (col, row) coordinate and compute
  // intensity = (re^2 + im^2) * norm, where norm = 1/(Ng^2) accounts
  // for cuFFT's unnormalised inverse transform.
  //
  // Thread count N = Nf_tile * n_beam
  // ================================================================

  struct ExtractBeamArg : kernel_param<> {
    const float *image;
    const int *beam_pixels;
    float *beam_output;
    int Ng;
    int n_beam;
    float norm;
    unsigned long long int N;

    ExtractBeamArg(const float *image, const int *beam_pixels,
                   float *beam_output, int Ng, int n_beam,
                   float norm, unsigned long long int N) :
      kernel_param(dim3(N, 1, 1)),
      image(image),
      beam_pixels(beam_pixels),
      beam_output(beam_output),
      Ng(Ng),
      n_beam(n_beam),
      norm(norm),
      N(N)
    {
    }
  };

  template <typename Arg> struct ExtractBeam {
    const Arg &arg;
    constexpr ExtractBeam(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int idx)
    {
      const int f_local = idx / arg.n_beam;
      const int b = idx % arg.n_beam;

      const int col_coord = arg.beam_pixels[b * 2];
      const int row_coord = arg.beam_pixels[b * 2 + 1];

      if (col_coord >= 0 && col_coord < arg.Ng &&
          row_coord >= 0 && row_coord < arg.Ng) {
        const long long grid_plane = static_cast<long long>(arg.Ng) * arg.Ng;
        const long long pixel_idx = (static_cast<long long>(f_local) * grid_plane +
                                     static_cast<long long>(row_coord) * arg.Ng + col_coord) * 2;
        const float re = arg.image[pixel_idx];
        const float im = arg.image[pixel_idx + 1];
        arg.beam_output[static_cast<long long>(f_local) * arg.n_beam + b] =
            (re * re + im * im) * arg.norm;
      } else {
        arg.beam_output[static_cast<long long>(f_local) * arg.n_beam + b] = 0.0f;
      }
    }
  };

  // ================================================================
  // Kernel 4: Quantise beam intensities to 8-bit unsigned
  //
  // output[i] = clamp(round(beam_intensity[i] * scale), 0, 255)
  //
  // Thread count N = total number of beam intensity values
  // ================================================================

  struct QuantiseBeamsArg : kernel_param<> {
    const float *beam_intensity;
    unsigned char *output;
    float scale;
    unsigned long long int N;

    QuantiseBeamsArg(const float *beam_intensity, unsigned char *output,
                     float scale, unsigned long long int N) :
      kernel_param(dim3(N, 1, 1)),
      beam_intensity(beam_intensity),
      output(output),
      scale(scale),
      N(N)
    {
    }
  };

  template <typename Arg> struct QuantiseBeams {
    const Arg &arg;
    constexpr QuantiseBeams(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int idx)
    {
      float val = arg.beam_intensity[idx] * arg.scale;
      // clamp to [0, 255]
      val = val < 0.0f ? 0.0f : (val > 255.0f ? 255.0f : val);
      // round to nearest integer
      val = val + 0.5f;
      arg.output[idx] = static_cast<unsigned char>(static_cast<int>(val));
    }
  };

  // ================================================================
  // Kernel 5: Extract packed lower triangle from full Hermitian matrix
  //
  // Input:  full [batch_count x N x N] complex interleaved floats
  // Output: packed lower triangle [batch_count x N*(N+1)/2] complex
  //
  // Uses grid-stride loop.  Thread count N_total = batch_count * n_baselines
  // ================================================================

  struct TriangulateFromHermArg : kernel_param<> {
    const float *full_mat;
    float *tri_out;
    int mat_N;           // matrix rank (number of antennas)
    int batch_count;
    int n_baselines;     // mat_N*(mat_N+1)/2
    unsigned long long int N_total;

    TriangulateFromHermArg(const float *full_mat, float *tri_out,
                           int mat_N, int batch_count,
                           unsigned long long int N_total) :
      kernel_param(dim3(N_total, 1, 1)),
      full_mat(full_mat),
      tri_out(tri_out),
      mat_N(mat_N),
      batch_count(batch_count),
      n_baselines(mat_N * (mat_N + 1) / 2),
      N_total(N_total)
    {
    }
  };

  template <typename Arg> struct TriangulateFromHerm {
    const Arg &arg;
    constexpr TriangulateFromHerm(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int idx)
    {
      const int b = idx / arg.n_baselines;
      const int k = idx % arg.n_baselines;

      // Decode packed index k -> (row, col) in lower triangle
      const int row = static_cast<int>((sqrtf(8.0f * k + 1.0f) - 1.0f) * 0.5f);
      const int col = k - row * (row + 1) / 2;

      // Read from full matrix [b, row, col]
      const long long src = (static_cast<long long>(b) * arg.mat_N * arg.mat_N +
                             static_cast<long long>(row) * arg.mat_N + col) * 2;
      const long long dst = (static_cast<long long>(b) * arg.n_baselines + k) * 2;
      arg.tri_out[dst]     = arg.full_mat[src];
      arg.tri_out[dst + 1] = arg.full_mat[src + 1];
    }
  };

  // ================================================================
  // Kernel 6: Pillbox grid scatter
  //
  // For each (frequency, baseline) pair, read the visibility from the
  // packed triangle, compute the UV cell from baseline offset + freq,
  // and atomicAdd to the UV grid.  Also adds conjugate at (-u,-v)
  // for cross-correlations.
  //
  // Thread count N = Nf_tile * n_baselines
  // ================================================================

  struct PillboxGridScatterArg : kernel_param<> {
    const float *vis_tri;
    const float *baseline_uv_m;
    const double *freq_hz;
    float *uv_grid;
    int n_baselines;
    int Ng;
    int Nf_tile;
    int freq_offset;
    float cell_size_rad;
    unsigned long long int N;

    PillboxGridScatterArg(const float *vis_tri, const float *baseline_uv_m,
                          const double *freq_hz, float *uv_grid,
                          int n_baselines, int Ng, int Nf_tile,
                          int freq_offset, float cell_size_rad,
                          unsigned long long int N) :
      kernel_param(dim3(N, 1, 1)),
      vis_tri(vis_tri),
      baseline_uv_m(baseline_uv_m),
      freq_hz(freq_hz),
      uv_grid(uv_grid),
      n_baselines(n_baselines),
      Ng(Ng),
      Nf_tile(Nf_tile),
      freq_offset(freq_offset),
      cell_size_rad(cell_size_rad),
      N(N)
    {
    }
  };

  template <typename Arg> struct PillboxGridScatter {
    const Arg &arg;
    constexpr PillboxGridScatter(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int idx)
    {
      static constexpr double C_LIGHT = 299792458.0;

      const int f_local = idx / arg.n_baselines;
      const int k = idx % arg.n_baselines;

      // Packed triangle: row = floor((sqrt(8k+1)-1)/2), col = k - row*(row+1)/2
      const int row = static_cast<int>((sqrtf(8.0f * k + 1.0f) - 1.0f) * 0.5f);
      const int col = k - row * (row + 1) / 2;
      const bool is_auto = (row == col);

      // Read visibility (interleaved Re, Im)
      const long long vis_idx = (static_cast<long long>(f_local) * arg.n_baselines + k) * 2;
      const float vis_re = arg.vis_tri[vis_idx];
      const float vis_im = arg.vis_tri[vis_idx + 1];

      // Baseline UV in metres
      const float u_m = arg.baseline_uv_m[k * 2];
      const float v_m = arg.baseline_uv_m[k * 2 + 1];

      // Frequency -> wavelength -> UV in wavelengths
      const double freq = arg.freq_hz[arg.freq_offset + f_local];
      const double lambda = C_LIGHT / freq;
      const float u_lam = static_cast<float>(u_m / lambda);
      const float v_lam = static_cast<float>(v_m / lambda);

      // UV cell indices (grid center at Ng/2)
      const float half_ng = 0.5f * arg.Ng;
      const int iu = static_cast<int>(roundf(u_lam / arg.cell_size_rad + half_ng));
      const int iv = static_cast<int>(roundf(v_lam / arg.cell_size_rad + half_ng));

      const long long grid_plane = static_cast<long long>(arg.Ng) * arg.Ng;

      // Scatter to (iu, iv)
      if (iu >= 0 && iu < arg.Ng && iv >= 0 && iv < arg.Ng) {
        const long long grid_idx = (static_cast<long long>(f_local) * grid_plane +
                                    static_cast<long long>(iv) * arg.Ng + iu) * 2;
#ifdef __CUDA_ARCH__
        atomicAdd(&arg.uv_grid[grid_idx],     vis_re);
        atomicAdd(&arg.uv_grid[grid_idx + 1], vis_im);
#else
        arg.uv_grid[grid_idx]     += vis_re;
        arg.uv_grid[grid_idx + 1] += vis_im;
#endif
      }

      // Conjugate at (-u, -v) -- skip for auto-correlations
      if (!is_auto) {
        const int iu_conj = static_cast<int>(roundf(-u_lam / arg.cell_size_rad + half_ng));
        const int iv_conj = static_cast<int>(roundf(-v_lam / arg.cell_size_rad + half_ng));
        if (iu_conj >= 0 && iu_conj < arg.Ng && iv_conj >= 0 && iv_conj < arg.Ng) {
          const long long grid_idx_conj = (static_cast<long long>(f_local) * grid_plane +
                                           static_cast<long long>(iv_conj) * arg.Ng + iu_conj) * 2;
#ifdef __CUDA_ARCH__
          atomicAdd(&arg.uv_grid[grid_idx_conj],      vis_re);
          atomicAdd(&arg.uv_grid[grid_idx_conj + 1], -vis_im);  // conjugate: negate Im
#else
          arg.uv_grid[grid_idx_conj]     += vis_re;
          arg.uv_grid[grid_idx_conj + 1] += -vis_im;
#endif
        }
      }
    }
  };

  // ================================================================
  // Kernel 7: Convert QC sign-magnitude to FTD/TCC two's complement
  //
  // DSA-2000 QC format (beamformer convention):
  //   high nibble = Re (sign-magnitude), low nibble = Im (sign-magnitude)
  //
  // TCC/FTD convention:
  //   low nibble = Re (two's complement), high nibble = Im (two's complement)
  //
  // Per byte:
  //   1) Extract Re from high nibble (SM), Im from low nibble (SM)
  //   2) Decode sign-magnitude: val = sign ? -mag : mag
  //   3) Encode as two's complement: tc = val & 0xF
  //   4) Pack: output = (im_tc << 4) | (re_tc & 0xF)
  //
  // In-place transformation. Thread count N = number of bytes.
  // ================================================================

  struct ConvertQcSmToFtdArg : kernel_param<> {
    uint8_t *data;
    unsigned long long int N;

    ConvertQcSmToFtdArg(uint8_t *data, unsigned long long int N) :
      kernel_param(dim3(N, 1, 1)),
      data(data),
      N(N)
    {
    }
  };

  template <typename Arg> struct ConvertQcSmToFtd {
    const Arg &arg;
    constexpr ConvertQcSmToFtd(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int idx)
    {
      uint8_t byte = arg.data[idx];

      // High nibble = Re (sign-magnitude)
      uint8_t re_sm = (byte >> 4) & 0x0F;
      int re_mag  = re_sm & 0x7;
      int re_val  = (re_sm & 0x8) ? -re_mag : re_mag;

      // Low nibble = Im (sign-magnitude)
      uint8_t im_sm = byte & 0x0F;
      int im_mag  = im_sm & 0x7;
      int im_val  = (im_sm & 0x8) ? -im_mag : im_mag;

      // Pack as two's complement: low nibble = Re, high nibble = Im
      uint8_t re_tc = static_cast<uint8_t>(re_val & 0xF);
      uint8_t im_tc = static_cast<uint8_t>(im_val & 0xF);
      arg.data[idx] = static_cast<uint8_t>((im_tc << 4) | (re_tc & 0x0F));
    }
  };

} // namespace quda
