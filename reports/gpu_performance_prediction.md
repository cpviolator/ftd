# GPU Performance Prediction: GB200 and VR200

*DSA-2000 FTD Pipeline — Predicted from GB10 (SM121) and GH200 (SM90) Measurements*

Generated: 2026-03-06

## 1. Executive Summary

This report predicts DSA-2000 FTD pipeline performance on two unmeasured GPUs:
- **GB200** (Blackwell datacenter, SM100, 148 SMs) — medium-high confidence
- **VR200** (Rubin, ~224 SMs) — low confidence (preliminary specs)

Predictions are derived from measured benchmarks on GB10 (Blackwell consumer, SM121, 48 SMs)
and GH200 (Hopper datacenter, SM90, 132 SMs) using a per-stage bottleneck-aware scaling model.

### Key Predictions

| Workload | GB10 (ms) | GH200 (ms) | GH200 ×GB10 | GB200 (ms) | GB200 ×GB10 | GB200 ×GH200 | VR200 (ms) | VR200 ×GB10 |
|----------|-----------|-------------|-------------|------------|-------------|--------------|------------|-------------|
| VoltBF 1ch (M=4000, batch=1) | 17.3 | 5.2 | 3.3x | **1.5** | 11.8x | 3.6x | **0.6** | 30.7x |
| VoltBF 8ch (M=4000, batch=8) | 91.7 | 11.0 | 8.3x | **5.3** | 17.5x | 2.1x | **2.4** | 38.3x |
| VoltBF 32ch (M=4000, batch=32) | 641.0 | 51.5 | 12.4x | **31.3** | 20.5x | 1.6x | **15.8** | 40.5x |
| VoltBF short cf8 (M=64, batch=200) | 71.5 | 6.3 | 11.3x | **3.2** | 22.6x | 2.0x | **1.1** | 66.6x |
| VisBF 1ch K128 | 10.2 | 1.5 | 6.7x | **0.5** | 19.6x | 2.9x | **0.3** | 32.1x |
| VisBF 8ch K128 | 80.7 | 10.3 | 7.9x | **3.6** | 22.4x | 2.9x | **2.2** | 36.7x |
| VisBF 32ch K128 | 322.8 | 43.2 | 7.5x | **14.8** | 21.9x | 2.9x | **9.0** | 35.8x |
| Dedisp b64 (CUTLASS FP8) | 63.2 | 5.2 | 12.3x | **2.4** | 26.4x | 2.2x | **0.9** | 71.0x |
| Dedisp b256 | 159.3 | 14.7 | 10.8x | **6.0** | 26.5x | 2.4x | **2.4** | 66.7x |

## 2. Hardware Comparison

| Parameter | GB10 (measured) | GH200 (measured) | GB200 (specs) | VR200 (est.) |
|-----------|----------------|-------------------|--------------|-------------|
| Architecture | Blackwell SM121 | Hopper SM90 | Blackwell SM100 | Rubin |
| SMs | 48 | 132 | 148 | ~224 |
| Clock (GHz) | 2.418 | 1.98 | ~2.1 | ~2.3 |
| FP8 Peak (TFLOPS) | 238 | 2,141 | 5,000 | ~16,000 |
| FP8 ops/SM/cycle | 2,048 | 8,192 | ~16,100 | ~31,000 |
| Mem BW (TB/s) | 0.546 | 4.0 | 8.0 | ~13.0 |
| L2 Cache (MB) | 24 | 60 | 126 | ~256 |
| Memory (GB) | 128 | 480 | 192 | 288 |

**Scaling ratios relative to GB10:**

| Metric | GH200/GB10 | GB200/GB10 | VR200/GB10 |
|--------|------------|------------|------------|
| FP8 TFLOPS | 9.0x | 21.0x | 67.2x |
| SM×Clock | 2.3x | 2.7x | 4.4x |
| Memory BW | 7.3x | 14.7x | 23.8x |
| FP32 TFLOPS | 2.2x | 2.7x | 4.4x |

## 3. Measured Data: GB10 vs GH200

### Observed Speedups (GH200 / GB10)

| Benchmark | GB10 (ms) | GH200 (ms) | Speedup | TFLOPS GB10 | TFLOPS GH200 |
|-----------|-----------|-------------|---------|-------------|--------------|
| dedisp_1600ch_1000dm_b64 | 39.0 | 4.5 | 8.6x | — | — |
| dedisp_1600ch_2000dm_b128 | 95.2 | 7.9 | 12.1x | — | — |
| dedisp_1600ch_2000dm_b16 | 36.4 | 5.2 | 6.9x | — | — |
| dedisp_1600ch_2000dm_b256 | 159.3 | 14.7 | 10.8x | — | — |
| dedisp_1600ch_2000dm_b32 | 45.2 | 5.7 | 7.9x | — | — |
| dedisp_1600ch_2000dm_b64 | 63.2 | 5.2 | 12.3x | — | — |
| dedisp_1600ch_2000dm_b64_pipe | 61.9 | 5.2 | 12.0x | — | — |
| dedisp_1600ch_512dm_b64 | 27.0 | 2.1 | 12.9x | — | — |
| visbf_16ch_K128 | 160.3 | 21.4 | 7.5x | — | — |
| visbf_1ch_K128 | 10.2 | 1.5 | 6.7x | — | — |
| visbf_1ch_K256 | 10.2 | 1.6 | 6.5x | — | — |
| visbf_1ch_K64 | 10.1 | 1.5 | 6.9x | — | — |
| visbf_2ch_K128 | 19.9 | 2.7 | 7.3x | — | — |
| visbf_32ch_K128 | 322.8 | 43.2 | 7.5x | — | — |
| visbf_4ch_K128 | 40.2 | 5.2 | 7.8x | — | — |
| visbf_8ch_K128 | 80.7 | 10.3 | 7.9x | — | — |
| voltbf_16ch | 207.5 | 22.8 | 9.1x | — | — |
| voltbf_1ch | 17.3 | 5.2 | 3.3x | — | — |
| voltbf_2ch | 34.4 | 9.6 | 3.6x | — | — |
| voltbf_32ch | 641.0 | 51.5 | 12.4x | — | — |
| voltbf_4ch | 44.6 | 5.5 | 8.1x | — | — |
| voltbf_8ch | 91.7 | 11.0 | 8.3x | — | — |
| voltbf_M10240 | 46.9 | 12.8 | 3.7x | — | — |
| voltbf_M12288 | 56.8 | 15.2 | 3.7x | — | — |
| voltbf_M14336 | 67.9 | 17.7 | 3.8x | — | — |
| voltbf_M16384 | 78.7 | 20.2 | 3.9x | — | — |
| voltbf_M4096 | 17.7 | 5.3 | 3.4x | — | — |
| voltbf_M6144 | 26.9 | 7.8 | 3.4x | — | — |
| voltbf_M8192 | 36.7 | 10.3 | 3.6x | — | — |
| voltbf_short_cf4 | 114.1 | 11.5 | 9.9x | — | — |
| voltbf_short_cf8 | 71.5 | 6.3 | 11.3x | — | — |
| voltbf_short_cf8_nb16 | 556.5 | 68.8 | 8.1x | — | — |
| voltbf_short_cf8_nb2 | 98.3 | 8.7 | 11.3x | — | — |
| voltbf_short_cf8_nb4 | 153.9 | 16.0 | 9.6x | — | — |
| voltbf_short_cf8_nb8 | 276.8 | 31.7 | 8.7x | — | — |
| voltbf_short_unfused | 50.3 | 4.3 | 11.7x | — | — |

## 4. Model Methodology

### Approach: Per-Stage Bottleneck-Aware Scaling

Each benchmark pipeline has 3–6 stages with independent scaling characteristics.
For each stage, we:

1. **Classify** the computational bottleneck (FP8 tensor core, memory BW, FP32 compute, etc.)
2. **Compute** the observed GB10→GH200 scaling factor
3. **Decompose** into hardware ratio × efficiency correction
4. **Project** to GB200/VR200 using the same efficiency correction

### Stage Classification

| Type | Bottleneck | Scaling Metric | Examples |
|------|-----------|----------------|----------|
| `cutlass_gemm` | FP8 tensor cores | FP8 TFLOPS | voltbf gemm_pol0/pol1, dedisp CUTLASS |
| `direct_ptx` | PTX mma.sync | SM count × clock | visbf herk |
| `memory` | HBM bandwidth | Memory BW | corner_turn, qc_transpose, transposes |
| `fft` | Mixed compute/BW | √(FP32 × BW) | cuFFT stages |
| `fp32_compute` | FP32 cores | FP32 TFLOPS | img_beam |
| `cublas_fp32` | cuBLAS FP32 | FP32 TFLOPS | dedisp cuBLAS GEMM |

### Efficiency Corrections

The efficiency correction captures everything beyond raw hardware ratios:
tile utilization, occupancy, pipeline depth, memory controller efficiency, etc.

```
predicted_speedup = raw_hardware_ratio × efficiency_correction
efficiency_correction = median(observed_speedup / raw_ratio)  across all benchmarks
```

**Fitted efficiency corrections (median across benchmarks):**

| Stage Type | Efficiency | Interpretation |
|-----------|------------|----------------|
| `cutlass_gemm` | 0.972 | Close to raw ratio — well-predicted by hardware specs |
| `direct_ptx` | 0.806 | Close to raw ratio — well-predicted by hardware specs |
| `fft` | 3.870 | GH200 gains more than raw ratio suggests (deeper pipeline, better occupancy) |
| `fp32_compute` | 5.463 | GH200 gains more than raw ratio suggests (deeper pipeline, better occupancy) |
| `memory` | 0.954 | Close to raw ratio — well-predicted by hardware specs |

## 5. Cross-Validation Results

Predict GH200 stage times from GB10 data alone, compare to measured GH200.

| Stage Type | MAPE | N samples | Quality |
|-----------|------|-----------|---------|
| `cutlass_gemm` | 35.3% | 47 | Acceptable (<40%) |
| `direct_ptx` | 2.5% | 8 | Good (<20%) |
| `fft` | 24.3% | 22 | Acceptable (<40%) |
| `fp32_compute` | 2.0% | 8 | Good (<20%) |
| `memory` | 63.0% | 85 | Poor (>40%) |

**Overall weighted MAPE: 25.4%**

Cross-validation MAPE > 30% indicates the stage type has significant
architecture-dependent behavior that our linear scaling model doesn't fully capture.
Predictions for these stages on GB200/VR200 carry wider uncertainty.

## 6. GB200 Predictions

**Confidence: Medium-High** — Same Blackwell architecture family as GB10.
SM100 (datacenter) has 8x more FP8 ops/SM/cycle than SM121 (consumer),
but the same ISA and tensor core instruction set.

### Voltage Beamformer

| Benchmark | GB10 (ms) | GH200 (ms) | GH200 ×GB10 | GB200 (ms) | GB200 range | GB200 ×GB10 | GB200 ×GH200 |
|-----------|-----------|-------------|-------------|------------|-------------|-------------|--------------|
| voltbf_16ch | 207.5 | 22.8 | 9.1x | **11.5** | [8.1–15.6] | 18.0x | 2.0x |
| voltbf_1ch | 17.3 | 5.2 | 3.3x | **1.5** | [0.7–3.0] | 11.8x | 3.6x |
| voltbf_2ch | 34.4 | 9.6 | 3.6x | **2.8** | [1.4–5.4] | 12.4x | 3.5x |
| voltbf_32ch | 641.0 | 51.5 | 12.4x | **31.3** | [18.7–50.6] | 20.5x | 1.6x |
| voltbf_4ch | 44.6 | 5.5 | 8.1x | **2.6** | [1.8–3.4] | 17.4x | 2.1x |
| voltbf_8ch | 91.7 | 11.0 | 8.3x | **5.3** | [3.7–7.0] | 17.5x | 2.1x |
| voltbf_M10240 | 46.9 | 12.8 | 3.7x | **3.7** | [1.8–7.5] | 12.5x | 3.4x |
| voltbf_M12288 | 56.8 | 15.2 | 3.7x | **4.5** | [2.1–9.0] | 12.6x | 3.4x |
| voltbf_M14336 | 67.9 | 17.7 | 3.8x | **5.3** | [2.5–10.7] | 12.9x | 3.4x |
| voltbf_M16384 | 78.7 | 20.2 | 3.9x | **6.1** | [2.9–12.2] | 12.9x | 3.3x |
| voltbf_M4096 | 17.7 | 5.3 | 3.4x | **1.5** | [0.7–3.0] | 11.9x | 3.5x |
| voltbf_M6144 | 26.9 | 7.8 | 3.4x | **2.2** | [1.0–4.5] | 12.0x | 3.5x |
| voltbf_M8192 | 36.7 | 10.3 | 3.6x | **3.0** | [1.4–6.0] | 12.3x | 3.5x |
| voltbf_short_cf4 | 114.1 | 11.5 | 9.9x | **5.4** | [3.8–7.2] | 21.3x | 2.1x |
| voltbf_short_cf8 | 71.5 | 6.3 | 11.3x | **3.2** | [2.0–4.6] | 22.6x | 2.0x |
| voltbf_short_cf8_nb16 | 556.5 | 68.8 | 8.1x | **30.5** | [21.1–41.9] | 18.2x | 2.3x |
| voltbf_short_cf8_nb2 | 98.3 | 8.7 | 11.3x | **4.4** | [2.8–6.5] | 22.4x | 2.0x |
| voltbf_short_cf8_nb4 | 153.9 | 16.0 | 9.6x | **7.5** | [5.1–10.5] | 20.5x | 2.1x |
| voltbf_short_cf8_nb8 | 276.8 | 31.7 | 8.7x | **14.3** | [10.2–19.2] | 19.3x | 2.2x |
| voltbf_short_unfused | 50.3 | 4.3 | 11.7x | **2.2** | [1.4–3.2] | 23.0x | 2.0x |

### Visibility Beamformer

| Benchmark | GB10 (ms) | GH200 (ms) | GH200 ×GB10 | GB200 (ms) | GB200 range | GB200 ×GB10 | GB200 ×GH200 |
|-----------|-----------|-------------|-------------|------------|-------------|-------------|--------------|
| visbf_16ch_K128 | 160.3 | 21.4 | 7.5x | **7.3** | [4.1–13.4] | 21.9x | 2.9x |
| visbf_1ch_K128 | 10.2 | 1.5 | 6.7x | **0.5** | [0.3–1.0] | 19.6x | 2.9x |
| visbf_1ch_K256 | 10.2 | 1.6 | 6.5x | **0.6** | [0.3–1.0] | 18.3x | 2.8x |
| visbf_1ch_K64 | 10.1 | 1.5 | 6.9x | **0.5** | [0.3–0.9] | 20.5x | 3.0x |
| visbf_2ch_K128 | 19.9 | 2.7 | 7.3x | **0.9** | [0.5–1.8] | 21.0x | 2.9x |
| visbf_32ch_K128 | 322.8 | 43.2 | 7.5x | **14.8** | [8.3–27.0] | 21.9x | 2.9x |
| visbf_4ch_K128 | 40.2 | 5.2 | 7.8x | **1.8** | [1.0–3.4] | 22.0x | 2.8x |
| visbf_8ch_K128 | 80.7 | 10.3 | 7.9x | **3.6** | [2.0–6.7] | 22.4x | 2.9x |

### Dedispersion

| Benchmark | GB10 (ms) | GH200 (ms) | GH200 ×GB10 | GB200 (ms) | GB200 range | GB200 ×GB10 | GB200 ×GH200 |
|-----------|-----------|-------------|-------------|------------|-------------|-------------|--------------|
| dedisp_1600ch_1000dm_b64 | 39.0 | 4.5 | 8.6x | **1.7** | [1.0–2.8] | 22.6x | 2.6x |
| dedisp_1600ch_2000dm_b128 | 95.2 | 7.9 | 12.1x | **3.5** | [2.1–5.4] | 27.4x | 2.3x |
| dedisp_1600ch_2000dm_b16 | 36.4 | 5.2 | 6.9x | **1.9** | [1.2–3.0] | 18.7x | 2.7x |
| dedisp_1600ch_2000dm_b256 | 159.3 | 14.7 | 10.8x | **6.0** | [4.0–9.1] | 26.5x | 2.4x |
| dedisp_1600ch_2000dm_b32 | 45.2 | 5.7 | 7.9x | **2.2** | [1.4–3.3] | 20.5x | 2.6x |
| dedisp_1600ch_2000dm_b64 | 63.2 | 5.2 | 12.3x | **2.4** | [1.5–3.8] | 26.4x | 2.2x |
| dedisp_1600ch_2000dm_b64_pipe | 61.9 | 5.2 | 12.0x | **2.6** | [1.7–3.8] | 23.6x | 2.0x |
| dedisp_1600ch_512dm_b64 | 27.0 | 2.1 | 12.9x | **0.9** | [0.5–1.6] | 28.4x | 2.2x |

### Per-Stage Breakdowns

**voltbf_1ch** (total: GB10=17.3, GH200=5.2, GB200=1.5 ms)

| Stage | Type | GB10 (ms) | GH200 (ms) | GB200 (ms) | Speedup |
|-------|------|-----------|-------------|------------|---------|
| qc_transpose | memory | 0.46 | 0.09 | 0.04 | 11.7x |
| gemm_pol0 | cutlass_gemm | 4.51 | 1.82 | 0.42 | 10.7x |
| gemm_pol1 | cutlass_gemm | 9.01 | 2.51 | 0.70 | 12.9x |
| time_integrate | memory | 1.25 | 0.13 | 0.08 | 16.0x |
| corner_turn | memory | 2.08 | 0.67 | 0.23 | 9.1x |

**voltbf_32ch** (total: GB10=641.0, GH200=51.5, GB200=31.3 ms)

| Stage | Type | GB10 (ms) | GH200 (ms) | GB200 (ms) | Speedup |
|-------|------|-----------|-------------|------------|---------|
| qc_transpose | memory | 67.53 | 4.78 | 3.48 | 19.4x |
| gemm_pol0 | cutlass_gemm | 113.97 | 11.98 | 5.43 | 21.0x |
| gemm_pol1 | cutlass_gemm | 123.27 | 12.47 | 5.76 | 21.4x |
| time_integrate | memory | 71.51 | 7.60 | 4.52 | 15.8x |
| corner_turn | memory | 264.75 | 14.68 | 12.07 | 21.9x |

**visbf_1ch_K128** (total: GB10=10.2, GH200=1.5, GB200=0.5 ms)

| Stage | Type | GB10 (ms) | GH200 (ms) | GB200 (ms) | Speedup |
|-------|------|-----------|-------------|------------|---------|
| herk | direct_ptx | 0.20 | 0.11 | 0.10 | 1.9x |
| pol_reduce | memory | 0.12 | 0.04 | 0.01 | 8.9x |
| img_scatter | memory | 1.08 | 0.11 | 0.07 | 16.2x |
| img_fft | fft | 4.52 | 0.73 | 0.15 | 29.8x |
| img_beam | fp32_compute | 3.66 | 0.30 | 0.11 | 33.7x |
| corner_turn | memory | 0.66 | 0.25 | 0.08 | 8.4x |

**visbf_32ch_K128** (total: GB10=322.8, GH200=43.2, GB200=14.8 ms)

| Stage | Type | GB10 (ms) | GH200 (ms) | GB200 (ms) | Speedup |
|-------|------|-----------|-------------|------------|---------|
| herk | direct_ptx | 5.33 | 2.94 | 2.75 | 1.9x |
| pol_reduce | memory | 5.35 | 0.97 | 0.44 | 12.1x |
| img_scatter | memory | 35.15 | 3.80 | 2.24 | 15.7x |
| img_fft | fft | 131.98 | 23.55 | 4.66 | 28.3x |
| img_beam | fp32_compute | 130.86 | 10.79 | 3.89 | 33.6x |
| corner_turn | memory | 14.18 | 1.18 | 0.79 | 17.9x |

**dedisp_1600ch_2000dm_b64** (total: GB10=63.2, GH200=5.2, GB200=2.4 ms)

| Stage | Type | GB10 (ms) | GH200 (ms) | GB200 (ms) | Speedup |
|-------|------|-----------|-------------|------------|---------|
| Forward FFT (R2C) | fft | 3.78 | 0.24 | 0.08 | 47.8x |
| Transpose 1 (Batch <-> Freq) | memory | 6.81 | 0.34 | 0.29 | 23.1x |
| CUTLASS FP8 GEMM | cutlass_gemm | 44.30 | 3.86 | 1.92 | 23.1x |
| Inverse FFT (C2R) | fft | 4.81 | 0.29 | 0.10 | 48.4x |

## 7. VR200 Predictions

**Confidence: Low** — Unreleased Rubin architecture (expected H2 2026).
FP8 TFLOPS estimate (~16,000) from SemiAnalysis; other specs are preliminary.
Confidence intervals are ±50%.

### Voltage Beamformer

| Benchmark | GB10 (ms) | GH200 (ms) | GB200 (ms) | VR200 (ms) | VR200 range | VR200 ×GB10 | VR200 ×GH200 |
|-----------|-----------|-------------|------------|------------|-------------|-------------|--------------|
| voltbf_16ch | 207.5 | 22.8 | 11.5 | **5.4** | [2.5–9.0] | 38.1x | 4.2x |
| voltbf_1ch | 17.3 | 5.2 | 1.5 | **0.6** | [0.2–1.3] | 30.7x | 9.3x |
| voltbf_2ch | 34.4 | 9.6 | 2.8 | **1.0** | [0.4–2.3] | 33.4x | 9.3x |
| voltbf_32ch | 641.0 | 51.5 | 31.3 | **15.8** | [6.0–32.1] | 40.5x | 3.3x |
| voltbf_4ch | 44.6 | 5.5 | 2.6 | **1.1** | [0.5–1.9] | 38.8x | 4.8x |
| voltbf_8ch | 91.7 | 11.0 | 5.3 | **2.4** | [1.1–3.9] | 38.3x | 4.6x |
| voltbf_M10240 | 46.9 | 12.8 | 3.7 | **1.5** | [0.5–3.5] | 32.1x | 8.7x |
| voltbf_M12288 | 56.8 | 15.2 | 4.5 | **1.8** | [0.6–4.2] | 32.2x | 8.6x |
| voltbf_M14336 | 67.9 | 17.7 | 5.3 | **2.1** | [0.7–5.0] | 32.7x | 8.5x |
| voltbf_M16384 | 78.7 | 20.2 | 6.1 | **2.4** | [0.8–5.7] | 32.7x | 8.4x |
| voltbf_M4096 | 17.7 | 5.3 | 1.5 | **0.6** | [0.2–1.4] | 31.1x | 9.2x |
| voltbf_M6144 | 26.9 | 7.8 | 2.2 | **0.9** | [0.3–2.0] | 31.2x | 9.1x |
| voltbf_M8192 | 36.7 | 10.3 | 3.0 | **1.2** | [0.4–2.8] | 31.7x | 8.9x |
| voltbf_short_cf4 | 114.1 | 11.5 | 5.4 | **1.8** | [0.8–2.9] | 64.7x | 6.5x |
| voltbf_short_cf8 | 71.5 | 6.3 | 3.2 | **1.1** | [0.5–1.9] | 66.6x | 5.9x |
| voltbf_short_cf8_nb16 | 556.5 | 68.8 | 30.5 | **12.1** | [5.4–20.4] | 46.2x | 5.7x |
| voltbf_short_cf8_nb2 | 98.3 | 8.7 | 4.4 | **1.5** | [0.6–2.8] | 63.7x | 5.6x |
| voltbf_short_cf8_nb4 | 153.9 | 16.0 | 7.5 | **2.7** | [1.2–4.7] | 56.9x | 5.9x |
| voltbf_short_cf8_nb8 | 276.8 | 31.7 | 14.3 | **5.3** | [2.5–8.9] | 52.0x | 6.0x |
| voltbf_short_unfused | 50.3 | 4.3 | 2.2 | **0.7** | [0.3–1.3] | 70.0x | 6.0x |

### Visibility Beamformer

| Benchmark | GB10 (ms) | GH200 (ms) | GB200 (ms) | VR200 (ms) | VR200 range | VR200 ×GB10 | VR200 ×GH200 |
|-----------|-----------|-------------|------------|------------|-------------|-------------|--------------|
| visbf_16ch_K128 | 160.3 | 21.4 | 7.3 | **4.5** | [1.7–9.8] | 36.0x | 4.8x |
| visbf_1ch_K128 | 10.2 | 1.5 | 0.5 | **0.3** | [0.1–0.7] | 32.1x | 4.8x |
| visbf_1ch_K256 | 10.2 | 1.6 | 0.6 | **0.3** | [0.1–0.8] | 30.0x | 4.6x |
| visbf_1ch_K64 | 10.1 | 1.5 | 0.5 | **0.3** | [0.1–0.7] | 33.6x | 4.9x |
| visbf_2ch_K128 | 19.9 | 2.7 | 0.9 | **0.6** | [0.2–1.3] | 34.4x | 4.7x |
| visbf_32ch_K128 | 322.8 | 43.2 | 14.8 | **9.0** | [3.4–19.8] | 35.8x | 4.8x |
| visbf_4ch_K128 | 40.2 | 5.2 | 1.8 | **1.1** | [0.4–2.5] | 36.0x | 4.6x |
| visbf_8ch_K128 | 80.7 | 10.3 | 3.6 | **2.2** | [0.8–4.9] | 36.7x | 4.7x |

### Dedispersion

| Benchmark | GB10 (ms) | GH200 (ms) | GB200 (ms) | VR200 (ms) | VR200 range | VR200 ×GB10 | VR200 ×GH200 |
|-----------|-----------|-------------|------------|------------|-------------|-------------|--------------|
| dedisp_1600ch_1000dm_b64 | 39.0 | 4.5 | 1.7 | **0.7** | [0.3–1.3] | 58.8x | 6.8x |
| dedisp_1600ch_2000dm_b128 | 95.2 | 7.9 | 3.5 | **1.3** | [0.5–2.7] | 71.1x | 5.9x |
| dedisp_1600ch_2000dm_b16 | 36.4 | 5.2 | 1.9 | **0.6** | [0.3–1.2] | 56.1x | 8.1x |
| dedisp_1600ch_2000dm_b256 | 159.3 | 14.7 | 6.0 | **2.4** | [1.0–4.7] | 66.7x | 6.2x |
| dedisp_1600ch_2000dm_b32 | 45.2 | 5.7 | 2.2 | **0.8** | [0.3–1.5] | 58.9x | 7.4x |
| dedisp_1600ch_2000dm_b64 | 63.2 | 5.2 | 2.4 | **0.9** | [0.3–1.8] | 71.0x | 5.8x |
| dedisp_1600ch_2000dm_b64_pipe | 61.9 | 5.2 | 2.6 | **0.8** | [0.4–1.4] | 75.5x | 6.3x |
| dedisp_1600ch_512dm_b64 | 27.0 | 2.1 | 0.9 | **0.4** | [0.1–0.9] | 65.0x | 5.0x |

## 8. Architectural Considerations

### Direct PTX Kernel vs CUTLASS on GB200

The voltage beamformer's `gemm_pol0`/`gemm_pol1` stages use different kernel
paths depending on the GPU:
- **GB10 (SM121)**: Uses CUTLASS 4M path (4 real FP8 sub-GEMMs via wgmma)
- **GH200 (SM90)**: Also uses CUTLASS (autotuner selects wgmma cooperative schedule)
- **GB200 (SM100)**: Will use CUTLASS wgmma with native SM100 tensor core throughput.
  SM100's FP8 ops/SM/cycle is ~16,100 (vs 2,048 on SM121, 8,192 on SM90),
  giving GB200 a massive FP8 compute advantage.

For the direct PTX HERK kernel (`mma.sync.aligned.m16n8k32`), the instruction
is portable across SM80+, but per-SM throughput may differ on SM100 vs SM90.
Our model uses SM×clock scaling for this regime.

### L2 Cache Effects

GB200's 126 MB L2 (vs 24 MB on GB10, 60 MB on GH200) benefits:
- Larger HERK batch tiles fit in L2 (batch_tile = L2 / (4×N²))
- Better FP8 operand reuse in CUTLASS sub-GEMMs (Strategy 4B)
- Reduced memory traffic for transpose operations

### Visibility Beamformer FFT Efficiency Analysis

The `img_fft` stage dominates the visibility beamformer pipeline (41–55% of total time).
It performs batched 2D complex-to-complex FFTs on the Ng×Ng imaging grid using `cufftPlanMany()`.

**Implementation**: Already efficiently batched — `cufftPlanMany()` with `batch = Nf_eff`
(number of effective frequency channels). All channels in a tile are processed in a single
cuFFT kernel call, not individual per-channel launches. Frequency tiling automatically
adapts to available GPU memory when Nf_eff exceeds capacity.

**FFT scaling with channel count** (Ng=4096, K=128, FP16):

| n_ch | Nf_eff | GB10 img_fft (ms) | ms/Nf_eff | GH200 img_fft (ms) | ms/Nf_eff | Speedup |
|-----:|-------:|------------------:|----------:|-------------------:|----------:|--------:|
| 1 | 2 | 4.52 | 2.26 | 0.73 | 0.365 | 6.2x |
| 2 | 4 | 8.55 | 2.14 | 1.41 | 0.352 | 6.1x |
| 4 | 8 | 16.96 | 2.12 | 2.80 | 0.350 | 6.1x |
| 8 | 16 | 33.61 | 2.10 | 5.63 | 0.352 | 6.0x |
| 16 | 32 | 66.40 | 2.08 | 11.80 | 0.369 | 5.6x |
| 32 | 64 | 131.98 | 2.06 | 23.55 | 0.368 | 5.6x |

**Key observations:**

1. **Linear scaling with Nf_eff**: Doubling channel count exactly doubles FFT time
   (constant ~2.1 ms/Nf_eff on GB10, ~0.36 ms/Nf_eff on GH200). This confirms
   efficient batching — no per-channel launch overhead.

2. **Memory-bandwidth limited**: At 32ch (Nf_eff=64), the 2D FFT processes
   4096×4096×64 complex FP16 elements (4.0 GB data).
   Minimum memory traffic (2 passes × read+write): 16.0 GB. Effective BW:
   GB10: 130 GB/s (24% of 546 GB/s peak),
   GH200: 730 GB/s (18% of 4000 GB/s peak).
   The 5.6x GB10→GH200 speedup closely matches the 7.3x
   memory BW ratio, confirming BW-limited behavior.

3. **No batching inefficiency**: The batched cuFFT call achieves consistent per-channel
   throughput regardless of batch size, indicating the FFT is already optimally batched.
   No opportunity to improve batching — the improvement path is higher BW hardware.

**FFT independence from K**: Tests at K=64, K=128, K=256 (1ch) show identical img_fft
times (~4.5 ms on GB10, ~0.72 ms on GH200). The FFT operates on the Ng×Ng×Nf_eff
imaging grid, which depends only on grid size and channel count, not on K.

**Implication for GB200/VR200**: Since the FFT is memory-BW-limited, it will scale
with the BW ratio: GB200 (8.0 TB/s) should give ~15x
over GB10, and VR200 (~13.0 TB/s) should give ~24x.
However, the large efficiency correction (3.9x) from our model suggests that
GH200's deeper memory hierarchy and HBM3 latency characteristics provide additional
benefits beyond raw BW. GB200's HBM3e and VR200's HBM4 may show similar or larger gains.

### VR200 Unknowns

- **Tensor core architecture**: Rubin may have different MMA instruction throughput
- **Memory subsystem**: HBM4 bandwidth and latency characteristics unknown
- **cuFFT/cuBLAS**: Library optimizations for new architecture not yet available
- **CUTLASS support**: SM_next tile shapes, stage counts, and cluster configs unknown

## 9. Limitations and Assumptions

1. **Linear scaling assumption**: Each stage scales by a single hardware metric.
   In reality, many stages have mixed compute/memory bottlenecks that shift
   with problem size.

2. **Constant efficiency correction**: We use the median efficiency across all
   benchmarks for each stage type. Problem-size-dependent effects (tile utilization,
   wave quantization) are averaged out.

3. **Same kernel selection**: We assume the autotuner makes equivalent kernel
   choices on GB200/VR200 as on the measured GPUs. In practice, different SM
   counts and SMEM sizes may change optimal tile/cluster configs.

4. **GB200 clock speed**: Estimated at 2.1 GHz. Actual boost clocks in
   production may differ by ±10%.

5. **VR200 FP8 TFLOPS**: The 16,000 estimate carries significant uncertainty.
   Glenn Klockwood's 4,000 figure may refer to per-die or a different precision.
   We use 16,000 with ±50% confidence intervals.

6. **No software optimization**: Predictions assume current code without
   architecture-specific optimizations for GB200/VR200.

7. **Two-point model**: With only 2 measured GPUs, we cannot fit nonlinear
   scaling curves. A third measured GPU would significantly improve confidence.

## Appendix: Full Prediction Table

| # | Benchmark | GB10 (ms) | GH200 (ms) | GH200 ×GB10 | GB200 (ms) | GB200 ×GB10 | GB200 ×GH200 | VR200 (ms) | VR200 ×GB10 | VR200 ×GH200 |
|---|-----------|-----------|-------------|-------------|------------|-------------|--------------|------------|-------------|--------------|
| 1 | dedisp_1600ch_1000dm_b64 | 39.0 | 4.5 | 8.6x | 1.7 | 22.6x | 2.6x | 0.7 | 58.8x | 6.8x |
| 2 | dedisp_1600ch_2000dm_b128 | 95.2 | 7.9 | 12.1x | 3.5 | 27.4x | 2.3x | 1.3 | 71.1x | 5.9x |
| 3 | dedisp_1600ch_2000dm_b16 | 36.4 | 5.2 | 6.9x | 1.9 | 18.7x | 2.7x | 0.6 | 56.1x | 8.1x |
| 4 | dedisp_1600ch_2000dm_b256 | 159.3 | 14.7 | 10.8x | 6.0 | 26.5x | 2.4x | 2.4 | 66.7x | 6.2x |
| 5 | dedisp_1600ch_2000dm_b32 | 45.2 | 5.7 | 7.9x | 2.2 | 20.5x | 2.6x | 0.8 | 58.9x | 7.4x |
| 6 | dedisp_1600ch_2000dm_b64 | 63.2 | 5.2 | 12.3x | 2.4 | 26.4x | 2.2x | 0.9 | 71.0x | 5.8x |
| 7 | dedisp_1600ch_2000dm_b64_pipe | 61.9 | 5.2 | 12.0x | 2.6 | 23.6x | 2.0x | 0.8 | 75.5x | 6.3x |
| 8 | dedisp_1600ch_512dm_b64 | 27.0 | 2.1 | 12.9x | 0.9 | 28.4x | 2.2x | 0.4 | 65.0x | 5.0x |
| 9 | visbf_16ch_K128 | 160.3 | 21.4 | 7.5x | 7.3 | 21.9x | 2.9x | 4.5 | 36.0x | 4.8x |
| 10 | visbf_1ch_K128 | 10.2 | 1.5 | 6.7x | 0.5 | 19.6x | 2.9x | 0.3 | 32.1x | 4.8x |
| 11 | visbf_1ch_K256 | 10.2 | 1.6 | 6.5x | 0.6 | 18.3x | 2.8x | 0.3 | 30.0x | 4.6x |
| 12 | visbf_1ch_K64 | 10.1 | 1.5 | 6.9x | 0.5 | 20.5x | 3.0x | 0.3 | 33.6x | 4.9x |
| 13 | visbf_2ch_K128 | 19.9 | 2.7 | 7.3x | 0.9 | 21.0x | 2.9x | 0.6 | 34.4x | 4.7x |
| 14 | visbf_32ch_K128 | 322.8 | 43.2 | 7.5x | 14.8 | 21.9x | 2.9x | 9.0 | 35.8x | 4.8x |
| 15 | visbf_4ch_K128 | 40.2 | 5.2 | 7.8x | 1.8 | 22.0x | 2.8x | 1.1 | 36.0x | 4.6x |
| 16 | visbf_8ch_K128 | 80.7 | 10.3 | 7.9x | 3.6 | 22.4x | 2.9x | 2.2 | 36.7x | 4.7x |
| 17 | voltbf_16ch | 207.5 | 22.8 | 9.1x | 11.5 | 18.0x | 2.0x | 5.4 | 38.1x | 4.2x |
| 18 | voltbf_1ch | 17.3 | 5.2 | 3.3x | 1.5 | 11.8x | 3.6x | 0.6 | 30.7x | 9.3x |
| 19 | voltbf_2ch | 34.4 | 9.6 | 3.6x | 2.8 | 12.4x | 3.5x | 1.0 | 33.4x | 9.3x |
| 20 | voltbf_32ch | 641.0 | 51.5 | 12.4x | 31.3 | 20.5x | 1.6x | 15.8 | 40.5x | 3.3x |
| 21 | voltbf_4ch | 44.6 | 5.5 | 8.1x | 2.6 | 17.4x | 2.1x | 1.1 | 38.8x | 4.8x |
| 22 | voltbf_8ch | 91.7 | 11.0 | 8.3x | 5.3 | 17.5x | 2.1x | 2.4 | 38.3x | 4.6x |
| 23 | voltbf_M10240 | 46.9 | 12.8 | 3.7x | 3.7 | 12.5x | 3.4x | 1.5 | 32.1x | 8.7x |
| 24 | voltbf_M12288 | 56.8 | 15.2 | 3.7x | 4.5 | 12.6x | 3.4x | 1.8 | 32.2x | 8.6x |
| 25 | voltbf_M14336 | 67.9 | 17.7 | 3.8x | 5.3 | 12.9x | 3.4x | 2.1 | 32.7x | 8.5x |
| 26 | voltbf_M16384 | 78.7 | 20.2 | 3.9x | 6.1 | 12.9x | 3.3x | 2.4 | 32.7x | 8.4x |
| 27 | voltbf_M4096 | 17.7 | 5.3 | 3.4x | 1.5 | 11.9x | 3.5x | 0.6 | 31.1x | 9.2x |
| 28 | voltbf_M6144 | 26.9 | 7.8 | 3.4x | 2.2 | 12.0x | 3.5x | 0.9 | 31.2x | 9.1x |
| 29 | voltbf_M8192 | 36.7 | 10.3 | 3.6x | 3.0 | 12.3x | 3.5x | 1.2 | 31.7x | 8.9x |
| 30 | voltbf_short_cf4 | 114.1 | 11.5 | 9.9x | 5.4 | 21.3x | 2.1x | 1.8 | 64.7x | 6.5x |
| 31 | voltbf_short_cf8 | 71.5 | 6.3 | 11.3x | 3.2 | 22.6x | 2.0x | 1.1 | 66.6x | 5.9x |
| 32 | voltbf_short_cf8_nb16 | 556.5 | 68.8 | 8.1x | 30.5 | 18.2x | 2.3x | 12.1 | 46.2x | 5.7x |
| 33 | voltbf_short_cf8_nb2 | 98.3 | 8.7 | 11.3x | 4.4 | 22.4x | 2.0x | 1.5 | 63.7x | 5.6x |
| 34 | voltbf_short_cf8_nb4 | 153.9 | 16.0 | 9.6x | 7.5 | 20.5x | 2.1x | 2.7 | 56.9x | 5.9x |
| 35 | voltbf_short_cf8_nb8 | 276.8 | 31.7 | 8.7x | 14.3 | 19.3x | 2.2x | 5.3 | 52.0x | 6.0x |
| 36 | voltbf_short_unfused | 50.3 | 4.3 | 11.7x | 2.2 | 23.0x | 2.0x | 0.7 | 70.0x | 6.0x |
