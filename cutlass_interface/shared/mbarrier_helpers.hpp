// mbarrier_helpers.hpp — Lightweight mbarrier PTX wrappers for SM90+ warp-specialized pipelines.
// No CUTLASS dependency. Used by herk_kernel_common.hpp for warp-specialized HERK kernel.
//
// Protocol: producer issues cp.async loads and arrives at full_barrier;
//           consumers wait on full_barrier, compute, then arrive at empty_barrier.
//           Producer waits on empty_barrier before reusing the buffer.

#pragma once

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

/// Initialize an mbarrier in shared memory with a given expected arrival count.
__device__ __forceinline__ void mbarrier_init(uint64_t* bar, int expected_count) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "mbarrier.init.shared.b64 [%0], %1;"
        :
        : "r"(smem_addr), "r"(expected_count)
    );
}

/// Signal one arrival at the given mbarrier.
__device__ __forceinline__ void mbarrier_arrive(uint64_t* bar) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n"
        "  .reg .b64 state;\n"
        "  mbarrier.arrive.shared.b64 state, [%0];\n"
        "}\n"
        :
        : "r"(smem_addr)
    );
}

/// Wait on an mbarrier with parity-based phase tracking.
/// phase_bit alternates 0/1 each time the barrier completes a full round.
__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* bar, int phase_bit) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n"
        "  .reg .pred P1;\n"
        "WAIT_LOOP_%=:\n"
        "  mbarrier.try_wait.parity.shared.b64 P1, [%0], %1;\n"
        "  @!P1 bra WAIT_LOOP_%=;\n"
        "}\n"
        :
        : "r"(smem_addr), "r"(phase_bit)
    );
}

/// Signal one arrival at the given mbarrier and expect a specific count of arrivals
/// before the barrier completes. This variant uses the cp.async.mbarrier.arrive
/// to combine cp.async completion notification with barrier arrival.
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar, int tx_count) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
        :
        : "r"(smem_addr), "r"(tx_count)
    );
}

/// Issue a cp.async.bulk.shared.global with mbarrier completion tracking.
/// Loads nbytes from global src to shared dst, signaling bar on completion.
/// SM90+: cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes
__device__ __forceinline__ void cp_async_bulk_global_to_shared(
    void* dst_shared, const void* src_global, int nbytes, uint64_t* bar)
{
    uint32_t smem_dst = static_cast<uint32_t>(__cvta_generic_to_shared(dst_shared));
    uint32_t smem_bar = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
        :
        : "r"(smem_dst), "l"(src_global), "r"(nbytes), "r"(smem_bar)
        : "memory"
    );
}

#else  // __CUDA_ARCH__ < 900

// Stubs for pre-SM90 compilation — these should never be called at runtime.
__device__ __forceinline__ void mbarrier_init(uint64_t*, int) {}
__device__ __forceinline__ void mbarrier_arrive(uint64_t*) {}
__device__ __forceinline__ void mbarrier_wait_parity(uint64_t*, int) {}
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t*, int) {}
__device__ __forceinline__ void cp_async_bulk_global_to_shared(void*, const void*, int, uint64_t*) {}

#endif  // __CUDA_ARCH__ >= 900
