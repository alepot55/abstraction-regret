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
