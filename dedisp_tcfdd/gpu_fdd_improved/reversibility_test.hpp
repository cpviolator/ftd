// --- reversibility_test.hpp ---
// FDD reversibility verification: forward FDD -> inverse FDD -> compare.
// Textual include inside the DedispApplication class body — no #pragma once.
//
// For the Ndm=Nf (square) case, the phasor matrix is approximately unitary,
// so the inverse is simply the conjugate transpose scaled by 1/Ndm.
// This avoids the O(Nf³) CPU pseudo-inverse entirely.

// ═══════════════════════════════════════════════════════════════
// Reversibility test structures and method
// ═══════════════════════════════════════════════════════════════

struct ReversibilityConfig {
    std::string label;
    int Nf, Ndm, Nt;
    bool expect_pass;
    double threshold;
};

struct ReversibilityResult {
    std::string label;
    int Nf, Ndm;
    double max_abs_error;
    double rms_error;
    double rel_error;
    double snr_db;
    bool passed;
    bool expected_pass;
};

void run_reversibility_test() {
    using CxGpu = ComplexTypeGpu;
    static_assert(std::is_same_v<Real, float>,
                  "Reversibility test requires single precision");

    int Nf_user = (int)params_.num_freq_channels;
    int Nt_user = (int)params_.num_time_samples;

    // FP32 round-trip precision: each cuBLAS CGEMM accumulates ~Nf * eps_fp32
    // error per element, and there are two GEMMs + two FFTs in the round-trip.
    // Empirical scaling: rel_error ≈ Nf * 3e-5 (measured on GB10).
    double fp32_threshold = std::max(1e-3, (double)Nf_user * 1e-4);

    // Only test Ndm=Nf: conjugate transpose is the exact inverse for unitary P.
    // Ndm < Nf is rank-deficient and cannot be inverted by transpose.
    std::vector<ReversibilityConfig> configs = {
        {"Ndm=Nf", Nf_user, Nf_user, Nt_user, true, fp32_threshold},
    };

    std::vector<ReversibilityResult> results;

    std::cout << "\n============================================\n";
    std::cout << "  FDD REVERSIBILITY VERIFICATION TEST\n";
    std::cout << "============================================\n\n";

    for (auto& cfg : configs) {
        int Nf = cfg.Nf;
        int Ndm = cfg.Ndm;
        int Nt = cfg.Nt;
        int batch = 1;

        int Nt_padded = 1;
        while (Nt_padded < Nt) Nt_padded *= 2;
        int Nt_complex = Nt_padded / 2 + 1;

        std::cout << "--- Config: " << cfg.label
                  << " (" << Ndm << " DMs, " << Nf << " channels, "
                  << Nt << " time samples, Nt_padded=" << Nt_padded << ") ---\n";

        // --- Generate test input: integer-freq sinusoids (exact zero mean) ---
        size_t input_elems = (size_t)batch * Nf * Nt;
        std::vector<float> h_input(input_elems);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> amp_dist(0.5f, 2.0f);
        std::uniform_int_distribution<int> freq_dist(5, 20);

        for (int f = 0; f < Nf; f++) {
            float a = amp_dist(rng);
            int w = freq_dist(rng);
            for (int t = 0; t < Nt; t++) {
                h_input[f * Nt + t] = a * std::sin(2.0f * 3.14159265f * w * t / Nt);
            }
        }

        // --- Generate phasors on GPU ---
        // Use large max_dm to ensure phasor phases span many oscillations,
        // making the matrix well-conditioned (approximately unitary) for all k >= 1.
        Real time_resolution = params_.total_obs_time / Nt;
        Real f_min = params_.f_min;
        Real f_max = params_.f_max;
        Real f_min_GHz = f_min / 1000.0f;
        Real f_max_GHz = f_max / 1000.0f;
        Real inv_f_spread = 1.0f / (f_min_GHz * f_min_GHz) - 1.0f / (f_max_GHz * f_max_GHz);
        Real f_k_1 = 1.0f / (Nt_padded * time_resolution);
        Real max_dm_needed = 10.0f * Nf / (f_k_1 * (Real)DISPERSION_CONSTANT * inv_f_spread);
        Real max_dm = std::max(max_dm_needed, params_.max_dm_search);
        std::cout << "  Using max_dm=" << std::fixed << std::setprecision(1) << max_dm
                  << " (needed " << max_dm_needed << " for conditioning)\n";

        DeviceBuffer d_delays, d_fk, d_phasors_buf;
        size_t delays_bytes = (size_t)Ndm * Nf * sizeof(Real);
        size_t fk_bytes = (size_t)Nt_complex * sizeof(Real);
        size_t phasor_bytes = (size_t)Nt_complex * Ndm * Nf * sizeof(CxGpu);

        d_delays.allocate(delays_bytes);
        d_fk.allocate(fk_bytes);
        d_phasors_buf.allocateManaged(phasor_bytes);

        {
            size_t total_work = std::max((size_t)Ndm * Nf, (size_t)Nt_complex);
            dim3 block_s(256);
            dim3 grid_s((total_work + 255) / 256);
            kernel_generate_precomp<Real><<<grid_s, block_s, 0, compute_stream_>>>(
                d_delays.get<Real>(), d_fk.get<Real>(),
                Ndm, Nf, Nt_complex,
                f_min, f_max, (Real)0.0, max_dm,
                time_resolution, DISPERSION_CONSTANT);
            CUDA_CHECK(cudaGetLastError());
        }

        {
            dim3 block_p(16, 16);
            dim3 grid_p((Nf + 15) / 16, (Ndm + 15) / 16, Nt_complex);
            kernel_generate_phasors<Real, CxGpu><<<grid_p, block_p, 0, compute_stream_>>>(
                d_phasors_buf.get<CxGpu>(),
                d_fk.get<Real>(), d_delays.get<Real>(),
                Nf, Ndm, Nt_complex, true);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaStreamSynchronize(compute_stream_));

        // --- Create forward pipeline and run ---
        std::cout << "  Forward FDD... ";
        std::cout.flush();

        auto fwd_pipeline = std::make_unique<FddGpuPipeline<Real>>(
            batch, Nf, Nt, Ndm, Nt_padded, "cublas");

        DeviceBuffer d_input, d_fwd_output;
        size_t in_bytes = input_elems * sizeof(float);
        size_t out_bytes = (size_t)batch * Ndm * Nt * sizeof(float);
        d_input.allocate(in_bytes);
        d_fwd_output.allocate(out_bytes);

        CUDA_CHECK(cudaMemcpy(d_input.get(), h_input.data(), in_bytes,
                              cudaMemcpyHostToDevice));

        auto t0 = std::chrono::high_resolution_clock::now();
        fwd_pipeline->execute(d_input.get<Real>(),
                              d_phasors_buf.get<CxGpu>(),
                              d_fwd_output.get<Real>(),
                              batch, compute_stream_);
        CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
        auto t1 = std::chrono::high_resolution_clock::now();
        double fwd_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "done (" << std::fixed << std::setprecision(1)
                  << fwd_ms << " ms)\n";

        // --- Compute inverse phasors: conjugate transpose / Ndm ---
        // For unitary P: P^{-1} = P^H / Ndm (since P P^H ≈ Ndm * I)
        //
        // Forward phasor layout: phasor[k * Ndm * Nf + dm * Nf + f] = P(k,f,dm)
        // cuBLAS reads as A[f, dm] col-major.
        //
        // Inverse phasor layout: inv[k * Nf * Ndm + f * Ndm + dm] = Q(k,dm,f)
        // cuBLAS reads as A[dm, f] col-major.
        //
        // Q(dm, f) = conj(P(f, dm)) / Ndm
        // inv[k*Nf*Ndm + f*Ndm + dm] = conj(phasor[k*Ndm*Nf + dm*Nf + f]) / Ndm

        std::cout << "  Computing inverse phasors (conjugate transpose / " << Ndm << ")... ";
        std::cout.flush();

        auto t2 = std::chrono::high_resolution_clock::now();

        size_t inv_phasor_bytes = (size_t)Nt_complex * Nf * Ndm * sizeof(CxGpu);
        DeviceBuffer d_inv_phasors;
        d_inv_phasors.allocateManaged(inv_phasor_bytes);

        std::vector<CxGpu> h_phasors((size_t)Nt_complex * Ndm * Nf);
        CUDA_CHECK(cudaMemcpy(h_phasors.data(), d_phasors_buf.get(),
                              phasor_bytes, cudaMemcpyDeviceToHost));

        std::vector<CxGpu> h_inv_phasors((size_t)Nt_complex * Nf * Ndm);
        float scale = 1.0f / Ndm;

        #pragma omp parallel for
        for (int k = 0; k < Nt_complex; k++) {
            for (int dm = 0; dm < Ndm; dm++) {
                for (int f = 0; f < Nf; f++) {
                    size_t fwd_idx = (size_t)k * Ndm * Nf + (size_t)dm * Nf + f;
                    size_t inv_idx = (size_t)k * Nf * Ndm + (size_t)f * Ndm + dm;
                    // conj(P) / Ndm
                    h_inv_phasors[inv_idx].x =  h_phasors[fwd_idx].x * scale;
                    h_inv_phasors[inv_idx].y = -h_phasors[fwd_idx].y * scale;
                }
            }
        }

        CUDA_CHECK(cudaMemcpy(d_inv_phasors.get(), h_inv_phasors.data(),
                              inv_phasor_bytes, cudaMemcpyHostToDevice));

        auto t3 = std::chrono::high_resolution_clock::now();
        double inv_phasor_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        std::cout << "done (" << std::fixed << std::setprecision(0)
                  << inv_phasor_ms << " ms)\n";

        // --- Run inverse pipeline ---
        std::cout << "  Reverse FDD... ";
        std::cout.flush();

        auto inv_pipeline = std::make_unique<FddGpuPipeline<Real>>(
            batch, Ndm, Nt, Nf, Nt_padded, "cublas");

        DeviceBuffer d_reconstructed;
        size_t recon_bytes = (size_t)batch * Nf * Nt * sizeof(float);
        d_reconstructed.allocate(recon_bytes);

        auto t4 = std::chrono::high_resolution_clock::now();
        inv_pipeline->execute(d_fwd_output.get<Real>(),
                              d_inv_phasors.get<CxGpu>(),
                              d_reconstructed.get<Real>(),
                              batch, compute_stream_);
        CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
        auto t5 = std::chrono::high_resolution_clock::now();
        double inv_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();
        std::cout << "done (" << std::fixed << std::setprecision(1)
                  << inv_ms << " ms)\n";

        // --- Compare ---
        std::vector<float> h_reconstructed(input_elems);
        CUDA_CHECK(cudaMemcpy(h_reconstructed.data(), d_reconstructed.get(),
                              recon_bytes, cudaMemcpyDeviceToHost));

        double max_abs = 0, sum_sq_err = 0, sum_sq_ref = 0;
        for (size_t i = 0; i < input_elems; i++) {
            double err = (double)h_reconstructed[i] - (double)h_input[i];
            double ref = (double)h_input[i];
            max_abs = std::max(max_abs, std::abs(err));
            sum_sq_err += err * err;
            sum_sq_ref += ref * ref;
        }

        double rms_err = std::sqrt(sum_sq_err / input_elems);
        double rel_err = (sum_sq_ref > 0) ? std::sqrt(sum_sq_err / sum_sq_ref) : 1e30;
        double snr_db = (sum_sq_err > 0) ? 10.0 * std::log10(sum_sq_ref / sum_sq_err) : 999.0;

        bool passed = cfg.expect_pass ? (rel_err < cfg.threshold)
                                      : (rel_err > cfg.threshold);

        std::cout << "\n  Reconstruction Metrics:\n"
                  << "    Max Absolute Error:   " << std::scientific << std::setprecision(2) << max_abs << "\n"
                  << "    RMS Error:            " << rms_err << "\n"
                  << "    Relative Error:       " << rel_err << "\n"
                  << "    Reconstruction SNR:   " << std::fixed << std::setprecision(1) << snr_db << " dB\n";

        std::cout << "\n  Result: " << (passed ? "PASS" : "FAIL")
                  << " (relative error " << std::scientific << std::setprecision(2)
                  << rel_err << (passed ? " < " : " >= ")
                  << cfg.threshold << ")\n\n";

        results.push_back({cfg.label, cfg.Nf, cfg.Ndm, max_abs, rms_err,
                           rel_err, snr_db, passed, cfg.expect_pass});

        fwd_pipeline.reset();
        inv_pipeline.reset();
    }

    // --- Summary table ---
    std::cout << "============================================\n";
    std::cout << "  SUMMARY\n";
    std::cout << "============================================\n";
    std::cout << " Config    | Nf  | Ndm | Rel Error | SNR (dB) | Expected | Result\n";
    std::cout << "-----------|-----|-----|-----------|----------|----------|--------\n";

    for (auto& r : results) {
        std::string result_str = r.passed ? "PASS" : "FAIL";
        std::cout << " " << std::setw(9) << std::left << r.label << " | "
                  << std::right << std::setw(3) << r.Nf << " | "
                  << std::setw(3) << r.Ndm << " | "
                  << std::scientific << std::setprecision(2) << std::setw(9) << r.rel_error << " | "
                  << std::fixed << std::setprecision(1) << std::setw(8) << r.snr_db << " | "
                  << std::setw(8) << (r.expected_pass ? "PASS" : "FAIL") << " | "
                  << result_str << "\n";
    }
    std::cout << "============================================\n\n";
}
