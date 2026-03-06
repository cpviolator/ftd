// --- host_helpers.hpp ---
// Host utility functions and RAII device buffer.
// Textual include — no #pragma once.

// --- PATCH: Memory Check Helper ---
void check_memory_availability(size_t required_bytes) {
    size_t free_byte, total_byte;
    cudaError_t err = cudaMemGetInfo(&free_byte, &total_byte);

    if (err != cudaSuccess) {
        std::cerr << "[ERROR] Failed to get memory info: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    double req_gb = required_bytes / 1e9;
    double free_gb = free_byte / 1e9;

    // Guard: If we need more than 95% of FREE memory, warn or error.
    if (required_bytes > free_byte) {
        std::cerr << "\n[CRITICAL ERROR] Insufficient GPU Memory!" << std::endl;
        std::cerr << "  Required: " << std::fixed << std::setprecision(2) << req_gb << " GB" << std::endl;
        std::cerr << "  Available:" << std::fixed << std::setprecision(2) << free_gb << " GB" << std::endl;
        std::cerr << "  Action: Reduce --batch-size or --num-freq-channels.\n" << std::endl;
        exit(EXIT_FAILURE);
    }
}

// --- RAII WRAPPER FOR DEVICE MEMORY ---
struct DeviceBuffer {
    void* ptr = nullptr;

    DeviceBuffer() = default;

    // Destructor automatically frees memory
    ~DeviceBuffer() {
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }

    // Disable copy (No double-free)
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // --- ADDED: Move Constructor ---
    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr;
    }

    // --- ADDED: Move Assignment ---
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr) cudaFree(ptr);
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }

// Helper to allocate standard device memory
    void allocate(size_t size_bytes) {
        if (ptr) {
            CUDA_CHECK(cudaFree(ptr));
            ptr = nullptr;
        }
        if (size_bytes > 0) {
            // [PATCH] Check before alloc
            check_memory_availability(size_bytes);

            cudaError_t err = cudaMalloc(&ptr, size_bytes);
            if (err != cudaSuccess) {
                std::cerr << "[FATAL] cudaMalloc failed for " << (size_bytes/1e9)
                          << " GB: " << cudaGetErrorString(err) << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }

    // Helper to allocate Managed memory (Unified Memory)
    void allocateManaged(size_t size_bytes) {
        if (ptr) {
            CUDA_CHECK(cudaFree(ptr));
            ptr = nullptr;
        }
        if (size_bytes > 0) {
            // [PATCH] Check before alloc
            check_memory_availability(size_bytes);

            cudaError_t err = cudaMallocManaged(&ptr, size_bytes);
            if (err != cudaSuccess) {
                std::cerr << "[FATAL] cudaMallocManaged failed for " << (size_bytes/1e9)
                          << " GB: " << cudaGetErrorString(err) << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }

    template <typename T = void>
    T* get() const {
        return static_cast<T*>(ptr);
    }

    operator void*() const { return ptr; }

  void free() {
    if (ptr) {
      cudaFree(ptr);
      ptr = nullptr;
    }
  }
};

// --- MOVED PulsarParams to be a global struct ---
// This allows it to be used by __global__ kernels
template <typename Real>
struct PulsarParams {
  Real dm;
  Real width_s;
  Real scattering_s;
  Real amplitude;
  Real pulse_start_time; // ADDED: Kernel needs this
};
