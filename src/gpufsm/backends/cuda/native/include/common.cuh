// Shared definitions for the gpufsm CUDA translation units.
//
// The kernels are split one family per .cu (dense, bit-packed, worklist, DFA) and
// compiled WITHOUT relocatable device code, so a __device__ function cannot be called
// across translation units. Anything used by more than one family therefore lives in a
// header and is included into each TU that needs it; anything used by exactly one
// family stays in that family's .cu. Keep it that way: moving a __device__ helper into
// a .cu that another TU calls it from is a link error, not a warning.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// Wildcard symbol id: a transition labelled with it matches any input byte.
// Must stay in sync with gpufsm.core.nfa.ANY_SYMBOL.
static constexpr int ANY_SYMBOL = 256;

// Packed working set: 64-bit words, so 8 words covers up to 512 states.
static constexpr int BITPACKED_MAX_WORDS = 8;

// Word geometry of the packed state set. The Python side has named this since
// `gpufsm.core.packing.WORD_BITS`; the kernels spelled it out as bare `>> 6`, `& 63` and
// `* 64` in about seventy places, which is the one arithmetic in this file a reader has to
// re-derive every time. Same numbers, named once.
static constexpr int WORD_BITS = 64;
static constexpr int WORD_SHIFT = 6;              // log2(WORD_BITS)
static constexpr int WORD_MASK = WORD_BITS - 1;

// Which word of the packed set holds `state`, and the bit that selects it inside that word.
__device__ __host__ __forceinline__ int word_of(int state) { return state >> WORD_SHIFT; }
__device__ __host__ __forceinline__ unsigned long long bit_of(int state) {
    return 1ULL << (state & WORD_MASK);
}

// Words needed to hold `num_states` bits -- the host-side ceil(n/64) written once.
__host__ __forceinline__ int words_for(int num_states) {
    return (num_states + WORD_BITS - 1) / WORD_BITS;
}

// Wrap every CUDA call whose failure would corrupt a RESULT or a MEASUREMENT. That includes
// the timing events: `cudaEventElapsedTime` writes nothing on failure, so the usual
// `float ms = 0.0f; cudaEventElapsedTime(&ms, ...)` leaves a zero that travels downstream as a
// legitimate reading of zero milliseconds -- in a study whose numbers are kernel times.
//
// `cudaEventDestroy` is deliberately NOT wrapped: it runs on the way out, its failure cannot
// corrupt a value already computed, and throwing from a cleanup path replaces a harmless leak
// with a lost result.
#define CUDA_CHECK(call)                                                       \
    do {                                                                      \
        cudaError_t _err = (call);                                           \
        if (_err != cudaSuccess) {                                            \
            throw std::runtime_error(std::string("CUDA error at ") +          \
                __FILE__ ":" + std::to_string(__LINE__) + " -> " +            \
                cudaGetErrorString(_err));                                    \
        }                                                                     \
    } while (0)

// Frees every device allocation registered with it when it goes out of scope.
//
// The entry points used to collect pointers in a `std::vector<void*>` and drain it by hand
// at the end of the happy path. CUDA_CHECK throws, so any error after the first allocation
// — a launch failure, an out-of-memory on a later buffer — skipped the drain and leaked
// everything allocated so far. A destructor cannot be skipped by a throw.
//
// Only *device* memory. The async path additionally registers host pages and creates
// streams; those still unwind by hand.
class DeviceScope {
public:
    DeviceScope() = default;
    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;
    ~DeviceScope() {
        for (void* p : ptrs) cudaFree(p);
    }

    // Adopt an already-allocated pointer.
    template <typename T>
    T* own(T* p) {
        ptrs.push_back(static_cast<void*>(p));
        return p;
    }

    std::vector<void*> ptrs;
};

// Copies a numpy array to the device; the allocation is owned by `scope`.
template <typename T>
static const T* dev_copy(const py::array_t<T>& a, DeviceScope& scope) {
    auto buf = a.request();
    T* d = nullptr;
    size_t bytes = static_cast<size_t>(buf.size) * sizeof(T);
    CUDA_CHECK(cudaMalloc(&d, bytes ? bytes : 1));
    scope.own(d);
    if (bytes) CUDA_CHECK(cudaMemcpy(d, buf.ptr, bytes, cudaMemcpyHostToDevice));
    return d;
}
