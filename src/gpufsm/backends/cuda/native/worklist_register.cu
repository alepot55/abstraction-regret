// Work-efficient worklist, register-resident working set (<= BITPACKED_MAX_WORDS states).
//
// Iterates only the active states plus a frontier-based epsilon closure, removing the
// O(n^2) full scan. The working set lives in registers, which is what makes it ~4-5x
// faster than the global-memory variant at equal state count; worklist_global.cu is the
// path with no state-count cap.
//
// Its __device__ helpers are private to this TU on purpose: no other family calls them.

#include "include/api.hpp"

// Block size for the register worklist. Named because __launch_bounds__ and the launch
// configuration have to agree, and they sit twenty-four lines apart.
static constexpr int WORKLIST_BLOCK = 256;

template <int NWORDS>
__device__ __forceinline__ void eps_closure_worklist(
    unsigned long long set[NWORDS], const int* eps_row_ptr, const int* eps_targets) {
    unsigned long long frontier[NWORDS];
#pragma unroll
    for (int w = 0; w < NWORDS; ++w) frontier[w] = set[w];
    bool any = true;
    while (any) {
        unsigned long long nb[NWORDS];
#pragma unroll
        for (int w = 0; w < NWORDS; ++w) nb[w] = 0ULL;
        for (int w = 0; w < NWORDS; ++w) {
            unsigned long long b = frontier[w];
            while (b) {
                int s = w * WORD_BITS + __ffsll(b) - 1;
                b &= b - 1;
                for (int k = eps_row_ptr[s]; k < eps_row_ptr[s + 1]; ++k) {
                    int t = eps_targets[k];
                    nb[word_of(t)] |= bit_of(t);
                }
            }
        }
        any = false;
#pragma unroll
        for (int w = 0; w < NWORDS; ++w) {
            nb[w] &= ~set[w];          // keep only genuinely new states
            set[w] |= nb[w];
            frontier[w] = nb[w];
            if (nb[w]) any = true;
        }
    }
}

// Work-efficient single-string simulation: iterate only ACTIVE states (set bits),
// not all num_states, and use a frontier epsilon-closure. Same verdict as
// simulate_one (latch-first-match) but O(active) per symbol instead of O(n^2) —
// the kernel that moves the workload toward the memory-bound regime where the
// memory-layout techniques matter.
template <int NWORDS>
__device__ __forceinline__ void simulate_one_worklist(
    const int* sym_row_ptr, const int* sym_targets, const int* sym_symbols,
    const int* eps_row_ptr, const int* eps_targets,
    const unsigned long long* accept_words,
    const int* input_symbols, int input_len,
    int num_states, int start_state, int uses_any,
    int& out_f, int& out_l) {
    (void)num_states;
    unsigned long long cur[NWORDS];
#pragma unroll
    for (int w = 0; w < NWORDS; ++w) cur[w] = 0ULL;
    cur[word_of(start_state)] |= bit_of(start_state);
    eps_closure_worklist<NWORDS>(cur, eps_row_ptr, eps_targets);

    out_f = 0; out_l = 0; int done = 0;
#pragma unroll
    for (int w = 0; w < NWORDS; ++w) if (cur[w] & accept_words[w]) done = 1;
    if (done) { out_f = 1; out_l = 0; }

    for (int pos = 0; pos < input_len && !done; ++pos) {
        int sym = input_symbols[pos];
        unsigned long long nxt[NWORDS];
#pragma unroll
        for (int w = 0; w < NWORDS; ++w) nxt[w] = 0ULL;
        for (int w = 0; w < NWORDS; ++w) {
            unsigned long long b = cur[w];
            while (b) {
                int s = w * WORD_BITS + __ffsll(b) - 1;
                b &= b - 1;
                for (int k = sym_row_ptr[s]; k < sym_row_ptr[s + 1]; ++k) {
                    int tsym = sym_symbols[k];
                    if (tsym == sym || (uses_any && tsym == ANY_SYMBOL)) {
                        int t = sym_targets[k];
                        nxt[word_of(t)] |= bit_of(t);
                    }
                }
            }
        }
        eps_closure_worklist<NWORDS>(nxt, eps_row_ptr, eps_targets);
#pragma unroll
        for (int w = 0; w < NWORDS; ++w) cur[w] = nxt[w];
        int m = 0;
#pragma unroll
        for (int w = 0; w < NWORDS; ++w) if (cur[w] & accept_words[w]) m = 1;
        if (m) { out_f = 1; out_l = pos + 1; done = 1; }
    }
}

// Multi-stream worklist: one thread/string, work-efficient kernel, global CSR.
// __launch_bounds__ raises occupancy for SMALL working sets (NWORDS<=2: cap registers
// to fit 6 blocks/SM — the kernel is latency-bound, so more resident warps hide latency,
// ~2x at <=64 states). For larger NWORDS the same cap forces register spills and hurts,
// so we relax to minBlocks=1 (effectively unconstrained). NWORDS is a compile-time
// template parameter, so the ternary is a constant expression.
template <int NWORDS>
// The first __launch_bounds__ argument is the block size this kernel is compiled for and
// MUST equal the launch's block size (WORKLIST_BLOCK below). Promising nvcc 256 and then
// launching something else costs occupancy silently, or fails the launch outright.
__global__ void __launch_bounds__(WORKLIST_BLOCK, (NWORDS <= 2 ? 6 : 1)) worklist_multistream_kernel(
    const int* sym_row_ptr, const int* sym_targets, const int* sym_symbols,
    const int* eps_row_ptr, const int* eps_targets,
    const unsigned long long* accept_words,
    const int* input_data, const int* input_offsets, int num_strings,
    int num_states, int start_state, int uses_any,
    int* out_flags, int* out_lens) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_strings) return;
    int out_f, out_l;
    simulate_one_worklist<NWORDS>(sym_row_ptr, sym_targets, sym_symbols, eps_row_ptr, eps_targets,
                                  accept_words, input_data + input_offsets[i],
                                  input_offsets[i + 1] - input_offsets[i],
                                  num_states, start_state, uses_any, out_f, out_l);
    out_flags[i] = out_f; out_lens[i] = out_l;
}

// Dispatch to the NWORDS specialisation of worklist_multistream_kernel. The working set is
// a compile-time register array, so the word count has to be a template argument.
static void launch_worklist(
    int nwords, int num_strings,
    const int* srp, const int* st, const int* ss, const int* erp, const int* et,
    const unsigned long long* acc, const int* in, const int* off,
    int num_states, int start_state, int uses_any, int* flags, int* lens) {
    int threads = WORKLIST_BLOCK;
    int blocks = (num_strings + threads - 1) / threads;
#define LAUNCH_WL(NW) worklist_multistream_kernel<NW><<<blocks, threads>>>( \
        srp, st, ss, erp, et, acc, in, off, num_strings, num_states, start_state, uses_any, flags, lens)
    switch (nwords) {
        case 1: LAUNCH_WL(1); break;
        case 2: LAUNCH_WL(2); break;
        case 3: LAUNCH_WL(3); break;
        case 4: LAUNCH_WL(4); break;
        case 5: LAUNCH_WL(5); break;
        case 6: LAUNCH_WL(6); break;
        case 7: LAUNCH_WL(7); break;
        case 8: LAUNCH_WL(8); break;
        default:
            throw std::runtime_error("worklist: num_states > " +
                std::to_string(BITPACKED_MAX_WORDS * 64) + " not supported (nwords=" +
                std::to_string(nwords) + ")");
    }
#undef LAUNCH_WL
}

// Work-efficient multi-stream. Returns (flags, lens, kernel_ms).
std::tuple<py::array_t<int>, py::array_t<int>, float> run_worklist(
    py::array_t<int> sym_row_ptr, py::array_t<int> sym_targets, py::array_t<int> sym_symbols,
    py::array_t<int> eps_row_ptr, py::array_t<int> eps_targets,
    py::array_t<unsigned long long> accept_words,
    py::array_t<int> input_data, py::array_t<int> input_offsets,
    int num_states, int start_state, int uses_any) {

    int nwords = words_for(num_states);
    int num_strings = static_cast<int>(input_offsets.request().size) - 1;

    DeviceScope scope;
    const int* d_srp = dev_copy(sym_row_ptr, scope);
    const int* d_st = dev_copy(sym_targets, scope);
    const int* d_ss = dev_copy(sym_symbols, scope);
    const int* d_erp = dev_copy(eps_row_ptr, scope);
    const int* d_et = dev_copy(eps_targets, scope);
    const unsigned long long* d_acc = dev_copy(accept_words, scope);
    const int* d_in = dev_copy(input_data, scope);
    const int* d_off = dev_copy(input_offsets, scope);

    int *d_flags, *d_lens;
    CUDA_CHECK(cudaMalloc(&d_flags, sizeof(int) * (num_strings ? num_strings : 1)));
    scope.own(d_flags);
    CUDA_CHECK(cudaMalloc(&d_lens, sizeof(int) * (num_strings ? num_strings : 1)));
    scope.own(d_lens);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    if (num_strings > 0) {
        launch_worklist(nwords, num_strings, d_srp, d_st, d_ss, d_erp, d_et, d_acc, d_in, d_off,
                        num_states, start_state, uses_any, d_flags, d_lens);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    float kernel_ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));

    py::array_t<int> flags(num_strings);
    py::array_t<int> lens(num_strings);
    if (num_strings > 0) {
        CUDA_CHECK(cudaMemcpy(flags.request().ptr, d_flags, sizeof(int) * num_strings, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lens.request().ptr, d_lens, sizeof(int) * num_strings, cudaMemcpyDeviceToHost));
    }
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return {flags, lens, kernel_ms};
}

