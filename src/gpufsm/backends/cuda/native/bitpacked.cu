// Bit-packed NFA techniques — the byte->bit axis and the multi-stream ablations.
//
// One packed bitmask (1 bit/state, 64-bit words) as the working set instead of one int8
// slot per state. For NWORDS==1 (<=64 states) it is a single register-resident
// `unsigned long long`. Same CSR algorithm as dense.cu, so the pair isolates the layout.
//
// Also holds the multi-stream family built on the same kernel: global CSR, CSR staged
// into shared memory, and the pinned/async transfer variant.

#include "include/api.hpp"

template <int NWORDS>
__device__ __forceinline__ void simulate_one(
    const int* sym_row_ptr, const int* sym_targets, const int* sym_symbols,
    const int* eps_row_ptr, const int* eps_targets,
    const unsigned long long* accept_words,
    const int* input_symbols, int input_len,
    int num_states, int start_state, int uses_any,
    int& out_f, int& out_l) {

    unsigned long long cur[NWORDS];
    unsigned long long nxt[NWORDS];
#pragma unroll
    for (int w = 0; w < NWORDS; ++w) cur[w] = 0ULL;
    cur[start_state >> 6] |= (1ULL << (start_state & 63));

    // Epsilon closure: num_states passes guarantee convergence.
    for (int it = 0; it < num_states; ++it) {
        for (int s = 0; s < num_states; ++s) {
            if (cur[s >> 6] & (1ULL << (s & 63))) {
                for (int k = eps_row_ptr[s]; k < eps_row_ptr[s + 1]; ++k) {
                    int t = eps_targets[k];
                    cur[t >> 6] |= (1ULL << (t & 63));
                }
            }
        }
    }

    out_f = 0; out_l = 0; int done = 0;
#pragma unroll
    for (int w = 0; w < NWORDS; ++w) if (cur[w] & accept_words[w]) done = 1;
    if (done) { out_f = 1; out_l = 0; }

    for (int pos = 0; pos < input_len && !done; ++pos) {
        int sym = input_symbols[pos];
#pragma unroll
        for (int w = 0; w < NWORDS; ++w) nxt[w] = 0ULL;
        for (int s = 0; s < num_states; ++s) {
            if (cur[s >> 6] & (1ULL << (s & 63))) {
                for (int k = sym_row_ptr[s]; k < sym_row_ptr[s + 1]; ++k) {
                    int tsym = sym_symbols[k];
                    if (tsym == sym || (uses_any && tsym == ANY_SYMBOL)) {
                        int t = sym_targets[k];
                        nxt[t >> 6] |= (1ULL << (t & 63));
                    }
                }
            }
        }
        for (int it = 0; it < num_states; ++it) {
            for (int s = 0; s < num_states; ++s) {
                if (nxt[s >> 6] & (1ULL << (s & 63))) {
                    for (int k = eps_row_ptr[s]; k < eps_row_ptr[s + 1]; ++k) {
                        int t = eps_targets[k];
                        nxt[t >> 6] |= (1ULL << (t & 63));
                    }
                }
            }
        }
#pragma unroll
        for (int w = 0; w < NWORDS; ++w) cur[w] = nxt[w];
        int m = 0;
#pragma unroll
        for (int w = 0; w < NWORDS; ++w) if (cur[w] & accept_words[w]) m = 1;
        if (m) { out_f = 1; out_l = pos + 1; done = 1; }
    }
}

template <int NWORDS>
__global__ void bitpacked_nfa_kernel(
    const int* sym_row_ptr, const int* sym_targets, const int* sym_symbols,
    const int* eps_row_ptr, const int* eps_targets,
    const unsigned long long* accept_words,
    const int* input_symbols, int input_len,
    int num_states, int start_state, int uses_any,
    int* out_flag, int* out_len) {
    int out_f, out_l;
    simulate_one<NWORDS>(sym_row_ptr, sym_targets, sym_symbols, eps_row_ptr, eps_targets,
                         accept_words, input_symbols, input_len,
                         num_states, start_state, uses_any, out_f, out_l);
    *out_flag = out_f; *out_len = out_l;
}

// Work-efficient frontier epsilon-closure: expand only NEW states (set bits in
// `frontier`) into `set`, until no new states appear. O(reachable) not O(n^2).

template <int NWORDS>
__global__ void bitpacked_multistream_kernel(
    const int* sym_row_ptr, const int* sym_targets, const int* sym_symbols,
    const int* eps_row_ptr, const int* eps_targets,
    const unsigned long long* accept_words,
    const int* input_data, const int* input_offsets, int num_strings,
    int num_states, int start_state, int uses_any,
    int* out_flags, int* out_lens) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_strings) return;
    int out_f, out_l;
    simulate_one<NWORDS>(sym_row_ptr, sym_targets, sym_symbols, eps_row_ptr, eps_targets,
                         accept_words, input_data + input_offsets[i],
                         input_offsets[i + 1] - input_offsets[i],
                         num_states, start_state, uses_any, out_f, out_l);
    out_flags[i] = out_f; out_lens[i] = out_l;
}

// Multi-stream + global->shared CSR ablation. Each block cooperatively stages the
// entire read-only CSR (+ accept words) into shared memory once; every thread then
// reads transitions from shared instead of global. **This layout cannot be
// expressed in Triton** (its compiler owns shared memory) — the core
// abstraction-regret demonstration on the CUDA side. Shared bytes are computed and
// requested by the host launcher; layout matches the host's size computation.
template <int NWORDS>
__global__ void bitpacked_multistream_shared_kernel(
    const int* sym_row_ptr, const int* sym_targets, const int* sym_symbols,
    const int* eps_row_ptr, const int* eps_targets,
    const unsigned long long* accept_words,
    const int* input_data, const int* input_offsets, int num_strings,
    int num_states, int start_state, int uses_any,
    int nnz_sym, int nnz_eps,
    int* out_flags, int* out_lens) {

    extern __shared__ unsigned char smem[];
    unsigned long long* s_acc = reinterpret_cast<unsigned long long*>(smem);
    int* s_srp = reinterpret_cast<int*>(s_acc + NWORDS);
    int* s_st  = s_srp + (num_states + 1);
    int* s_ss  = s_st + nnz_sym;
    int* s_erp = s_ss + nnz_sym;
    int* s_et  = s_erp + (num_states + 1);

    for (int j = threadIdx.x; j < NWORDS; j += blockDim.x) s_acc[j] = accept_words[j];
    for (int j = threadIdx.x; j < num_states + 1; j += blockDim.x) {
        s_srp[j] = sym_row_ptr[j];
        s_erp[j] = eps_row_ptr[j];
    }
    for (int j = threadIdx.x; j < nnz_sym; j += blockDim.x) {
        s_st[j] = sym_targets[j];
        s_ss[j] = sym_symbols[j];
    }
    for (int j = threadIdx.x; j < nnz_eps; j += blockDim.x) s_et[j] = eps_targets[j];
    __syncthreads();  // all threads must reach this before the bounds check / return

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_strings) return;
    int out_f, out_l;
    simulate_one<NWORDS>(s_srp, s_st, s_ss, s_erp, s_et, s_acc,
                         input_data + input_offsets[i],
                         input_offsets[i + 1] - input_offsets[i],
                         num_states, start_state, uses_any, out_f, out_l);
    out_flags[i] = out_f; out_lens[i] = out_l;
}


static void launch_bitpacked(
    int nwords,
    const int* srp, const int* st, const int* ss, const int* erp, const int* et,
    const unsigned long long* acc, const int* in, int input_len,
    int num_states, int start_state, int uses_any, int* flag, int* len) {
    switch (nwords) {
        case 1: bitpacked_nfa_kernel<1><<<1, 1>>>(srp, st, ss, erp, et, acc, in, input_len, num_states, start_state, uses_any, flag, len); break;
        case 2: bitpacked_nfa_kernel<2><<<1, 1>>>(srp, st, ss, erp, et, acc, in, input_len, num_states, start_state, uses_any, flag, len); break;
        case 3: bitpacked_nfa_kernel<3><<<1, 1>>>(srp, st, ss, erp, et, acc, in, input_len, num_states, start_state, uses_any, flag, len); break;
        case 4: bitpacked_nfa_kernel<4><<<1, 1>>>(srp, st, ss, erp, et, acc, in, input_len, num_states, start_state, uses_any, flag, len); break;
        case 5: bitpacked_nfa_kernel<5><<<1, 1>>>(srp, st, ss, erp, et, acc, in, input_len, num_states, start_state, uses_any, flag, len); break;
        case 6: bitpacked_nfa_kernel<6><<<1, 1>>>(srp, st, ss, erp, et, acc, in, input_len, num_states, start_state, uses_any, flag, len); break;
        case 7: bitpacked_nfa_kernel<7><<<1, 1>>>(srp, st, ss, erp, et, acc, in, input_len, num_states, start_state, uses_any, flag, len); break;
        case 8: bitpacked_nfa_kernel<8><<<1, 1>>>(srp, st, ss, erp, et, acc, in, input_len, num_states, start_state, uses_any, flag, len); break;
        default:
            throw std::runtime_error("bitpacked: num_states > " +
                std::to_string(BITPACKED_MAX_WORDS * 64) + " not supported (nwords=" +
                std::to_string(nwords) + ")");
    }
}

// Returns (accepted, match_len, kernel_ms). accept_words is the packed accept set.
std::tuple<bool, int, float> run_bitpacked(
    py::array_t<int> sym_row_ptr, py::array_t<int> sym_targets, py::array_t<int> sym_symbols,
    py::array_t<int> eps_row_ptr, py::array_t<int> eps_targets,
    py::array_t<unsigned long long> accept_words,
    py::array_t<int> input_symbols,
    int num_states, int start_state, int uses_any) {

    int nwords = (num_states + 63) / 64;
    DeviceScope scope;
    const int* d_srp = dev_copy(sym_row_ptr, scope);
    const int* d_st = dev_copy(sym_targets, scope);
    const int* d_ss = dev_copy(sym_symbols, scope);
    const int* d_erp = dev_copy(eps_row_ptr, scope);
    const int* d_et = dev_copy(eps_targets, scope);
    const unsigned long long* d_acc = dev_copy(accept_words, scope);
    const int* d_in = dev_copy(input_symbols, scope);
    int input_len = static_cast<int>(input_symbols.request().size);

    int *d_flag, *d_len;
    CUDA_CHECK(cudaMalloc(&d_flag, sizeof(int)));
    scope.own(d_flag);
    CUDA_CHECK(cudaMalloc(&d_len, sizeof(int)));
    scope.own(d_len);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    launch_bitpacked(nwords, d_srp, d_st, d_ss, d_erp, d_et, d_acc, d_in, input_len,
                     num_states, start_state, uses_any, d_flag, d_len);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    CUDA_CHECK(cudaDeviceSynchronize());
    float kernel_ms = 0.0f; cudaEventElapsedTime(&kernel_ms, start, stop);

    int h_flag = 0, h_len = 0;
    cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_len, d_len, sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return {h_flag != 0, h_len, kernel_ms};
}

static void launch_multistream(
    int nwords, int num_strings,
    const int* srp, const int* st, const int* ss, const int* erp, const int* et,
    const unsigned long long* acc, const int* in, const int* off,
    int num_states, int start_state, int uses_any, int* flags, int* lens) {
    int threads = 256;
    int blocks = (num_strings + threads - 1) / threads;
#define LAUNCH_MS(NW) bitpacked_multistream_kernel<NW><<<blocks, threads>>>( \
        srp, st, ss, erp, et, acc, in, off, num_strings, num_states, start_state, uses_any, flags, lens)
    switch (nwords) {
        case 1: LAUNCH_MS(1); break;
        case 2: LAUNCH_MS(2); break;
        case 3: LAUNCH_MS(3); break;
        case 4: LAUNCH_MS(4); break;
        case 5: LAUNCH_MS(5); break;
        case 6: LAUNCH_MS(6); break;
        case 7: LAUNCH_MS(7); break;
        case 8: LAUNCH_MS(8); break;
        default:
            throw std::runtime_error("multistream: num_states > " +
                std::to_string(BITPACKED_MAX_WORDS * 64) + " not supported (nwords=" +
                std::to_string(nwords) + ")");
    }
#undef LAUNCH_MS
}

// Returns (flags, lens, kernel_ms) for a batch of strings.
std::tuple<py::array_t<int>, py::array_t<int>, float> run_multistream(
    py::array_t<int> sym_row_ptr, py::array_t<int> sym_targets, py::array_t<int> sym_symbols,
    py::array_t<int> eps_row_ptr, py::array_t<int> eps_targets,
    py::array_t<unsigned long long> accept_words,
    py::array_t<int> input_data, py::array_t<int> input_offsets,
    int num_states, int start_state, int uses_any) {

    int nwords = (num_states + 63) / 64;
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
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    if (num_strings > 0) {
        launch_multistream(nwords, num_strings, d_srp, d_st, d_ss, d_erp, d_et, d_acc, d_in, d_off,
                           num_states, start_state, uses_any, d_flags, d_lens);
        CUDA_CHECK(cudaGetLastError());
    }
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    CUDA_CHECK(cudaDeviceSynchronize());
    float kernel_ms = 0.0f; cudaEventElapsedTime(&kernel_ms, start, stop);

    py::array_t<int> flags(num_strings);
    py::array_t<int> lens(num_strings);
    if (num_strings > 0) {
        cudaMemcpy(flags.request().ptr, d_flags, sizeof(int) * num_strings, cudaMemcpyDeviceToHost);
        cudaMemcpy(lens.request().ptr, d_lens, sizeof(int) * num_strings, cudaMemcpyDeviceToHost);
    }
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return {flags, lens, kernel_ms};
}

// Bytes of dynamic shared memory the shared-CSR kernel needs for this NFA.
static size_t shared_csr_bytes(int nwords, int num_states, int nnz_sym, int nnz_eps) {
    return static_cast<size_t>(nwords) * sizeof(unsigned long long)
         + static_cast<size_t>(num_states + 1) * sizeof(int) * 2
         + static_cast<size_t>(nnz_sym) * sizeof(int) * 2
         + static_cast<size_t>(nnz_eps) * sizeof(int);
}

static void launch_multistream_shared(
    int nwords, int num_strings,
    const int* srp, const int* st, const int* ss, const int* erp, const int* et,
    const unsigned long long* acc, const int* in, const int* off,
    int num_states, int start_state, int uses_any, int nnz_sym, int nnz_eps,
    size_t shared_bytes, int* flags, int* lens) {
    int threads = 256;
    int blocks = (num_strings + threads - 1) / threads;
#define LAUNCH_MSS(NW)                                                                              \
    do {                                                                                           \
        if (shared_bytes > 48 * 1024)                                                              \
            CUDA_CHECK(cudaFuncSetAttribute(bitpacked_multistream_shared_kernel<NW>,               \
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes)));     \
        bitpacked_multistream_shared_kernel<NW><<<blocks, threads, shared_bytes>>>(                \
            srp, st, ss, erp, et, acc, in, off, num_strings, num_states, start_state, uses_any,    \
            nnz_sym, nnz_eps, flags, lens);                                                        \
    } while (0)
    switch (nwords) {
        case 1: LAUNCH_MSS(1); break;
        case 2: LAUNCH_MSS(2); break;
        case 3: LAUNCH_MSS(3); break;
        case 4: LAUNCH_MSS(4); break;
        case 5: LAUNCH_MSS(5); break;
        case 6: LAUNCH_MSS(6); break;
        case 7: LAUNCH_MSS(7); break;
        case 8: LAUNCH_MSS(8); break;
        default:
            throw std::runtime_error("multistream_shared: num_states > " +
                std::to_string(BITPACKED_MAX_WORDS * 64) + " not supported (nwords=" +
                std::to_string(nwords) + ")");
    }
#undef LAUNCH_MSS
}

// Multi-stream + shared-CSR. Returns (flags, lens, kernel_ms). Throws if the CSR
// does not fit the device's opt-in shared-memory budget.
std::tuple<py::array_t<int>, py::array_t<int>, float> run_multistream_shared(
    py::array_t<int> sym_row_ptr, py::array_t<int> sym_targets, py::array_t<int> sym_symbols,
    py::array_t<int> eps_row_ptr, py::array_t<int> eps_targets,
    py::array_t<unsigned long long> accept_words,
    py::array_t<int> input_data, py::array_t<int> input_offsets,
    int num_states, int start_state, int uses_any) {

    int nwords = (num_states + 63) / 64;
    int num_strings = static_cast<int>(input_offsets.request().size) - 1;
    int nnz_sym = static_cast<int>(sym_targets.request().size);
    int nnz_eps = static_cast<int>(eps_targets.request().size);
    size_t shared_bytes = shared_csr_bytes(nwords, num_states, nnz_sym, nnz_eps);

    int max_optin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    if (static_cast<int>(shared_bytes) > max_optin) {
        throw std::runtime_error("multistream_shared: CSR needs " + std::to_string(shared_bytes) +
            " B shared mem > device opt-in max " + std::to_string(max_optin) + " B");
    }

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
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    if (num_strings > 0) {
        launch_multistream_shared(nwords, num_strings, d_srp, d_st, d_ss, d_erp, d_et, d_acc,
                                  d_in, d_off, num_states, start_state, uses_any, nnz_sym, nnz_eps,
                                  shared_bytes, d_flags, d_lens);
        CUDA_CHECK(cudaGetLastError());
    }
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    CUDA_CHECK(cudaDeviceSynchronize());
    float kernel_ms = 0.0f; cudaEventElapsedTime(&kernel_ms, start, stop);

    py::array_t<int> flags(num_strings);
    py::array_t<int> lens(num_strings);
    if (num_strings > 0) {
        cudaMemcpy(flags.request().ptr, d_flags, sizeof(int) * num_strings, cudaMemcpyDeviceToHost);
        cudaMemcpy(lens.request().ptr, d_lens, sizeof(int) * num_strings, cudaMemcpyDeviceToHost);
    }
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return {flags, lens, kernel_ms};
}

// Multi-stream + sync→async transfer ablation. Pinned host staging + N CUDA streams
// pipeline H2D(chunk) -> kernel(chunk) -> D2H(chunk) so input transfer overlaps
// compute (and vice-versa), hiding PCIe latency that a single blocking cudaMemcpy
// would expose. The read-only CSR is copied once up front; per-chunk launches reuse
// the global-CSR kernel by shifting the offsets/output pointers (offsets are absolute
// into input_data, so no kernel change is needed). Returns (flags, lens, total_ms)
// where total_ms is the overlapped end-to-end device time.
std::tuple<py::array_t<int>, py::array_t<int>, float> run_multistream_async(
    py::array_t<int> sym_row_ptr, py::array_t<int> sym_targets, py::array_t<int> sym_symbols,
    py::array_t<int> eps_row_ptr, py::array_t<int> eps_targets,
    py::array_t<unsigned long long> accept_words,
    py::array_t<int> input_data, py::array_t<int> input_offsets,
    int num_states, int start_state, int uses_any) {

    constexpr int N_STREAMS = 4;
    int nwords = (num_states + 63) / 64;
    int num_strings = static_cast<int>(input_offsets.request().size) - 1;
    int in_len = static_cast<int>(input_data.request().size);

    py::array_t<int> flags(num_strings < 0 ? 0 : num_strings);
    py::array_t<int> lens(num_strings < 0 ? 0 : num_strings);
    if (num_strings <= 0) return {flags, lens, 0.0f};

    // CSR copied once (small, read-only).
    DeviceScope scope;
    const int* d_srp = dev_copy(sym_row_ptr, scope);
    const int* d_st = dev_copy(sym_targets, scope);
    const int* d_ss = dev_copy(sym_symbols, scope);
    const int* d_erp = dev_copy(eps_row_ptr, scope);
    const int* d_et = dev_copy(eps_targets, scope);
    const unsigned long long* d_acc = dev_copy(accept_words, scope);

    // Pin the caller's host buffers IN PLACE (cudaHostRegister) instead of allocating
    // a second pinned buffer and copying into it — that extra full-input host memcpy
    // would dwarf any overlap benefit. Outputs are pinned likewise for async D2H.
    int* h_in = static_cast<int*>(input_data.request().ptr);
    int* h_off = static_cast<int*>(input_offsets.request().ptr);
    int* h_flags = static_cast<int*>(flags.request().ptr);
    int* h_lens = static_cast<int*>(lens.request().ptr);
    if (in_len) CUDA_CHECK(cudaHostRegister(h_in, sizeof(int) * in_len, cudaHostRegisterDefault));
    CUDA_CHECK(cudaHostRegister(h_flags, sizeof(int) * num_strings, cudaHostRegisterDefault));
    CUDA_CHECK(cudaHostRegister(h_lens, sizeof(int) * num_strings, cudaHostRegisterDefault));

    int *d_in, *d_off, *d_flags, *d_lens;
    CUDA_CHECK(cudaMalloc(&d_in, sizeof(int) * (in_len ? in_len : 1)));
    scope.own(d_in);
    CUDA_CHECK(cudaMalloc(&d_off, sizeof(int) * (num_strings + 1)));
    scope.own(d_off);
    CUDA_CHECK(cudaMalloc(&d_flags, sizeof(int) * num_strings));
    scope.own(d_flags);
    CUDA_CHECK(cudaMalloc(&d_lens, sizeof(int) * num_strings));
    scope.own(d_lens);

    cudaStream_t streams[N_STREAMS];
    for (int s = 0; s < N_STREAMS; ++s) CUDA_CHECK(cudaStreamCreate(&streams[s]));

    // Offsets are needed device-side; copy once (cheap) before the pipeline.
    CUDA_CHECK(cudaMemcpy(d_off, h_off, sizeof(int) * (num_strings + 1), cudaMemcpyHostToDevice));

    int chunk = (num_strings + N_STREAMS - 1) / N_STREAMS;
    int threads = 256;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int c = 0; c < N_STREAMS; ++c) {
        int lo = c * chunk;
        if (lo >= num_strings) break;
        int hi = lo + chunk; if (hi > num_strings) hi = num_strings;
        int nstr = hi - lo;
        int byte_lo = h_off[lo], byte_hi = h_off[hi];
        cudaStream_t st = streams[c % N_STREAMS];
        if (byte_hi > byte_lo) {
            CUDA_CHECK(cudaMemcpyAsync(d_in + byte_lo, h_in + byte_lo,
                sizeof(int) * (byte_hi - byte_lo), cudaMemcpyHostToDevice, st));
        }
        int blocks = (nstr + threads - 1) / threads;
#define LAUNCH_ASYNC(NW) bitpacked_multistream_kernel<NW><<<blocks, threads, 0, st>>>( \
        d_srp, d_st, d_ss, d_erp, d_et, d_acc, d_in, d_off + lo, nstr, \
        num_states, start_state, uses_any, d_flags + lo, d_lens + lo)
        switch (nwords) {
            case 1: LAUNCH_ASYNC(1); break;
            case 2: LAUNCH_ASYNC(2); break;
            case 3: LAUNCH_ASYNC(3); break;
            case 4: LAUNCH_ASYNC(4); break;
            case 5: LAUNCH_ASYNC(5); break;
            case 6: LAUNCH_ASYNC(6); break;
            case 7: LAUNCH_ASYNC(7); break;
            case 8: LAUNCH_ASYNC(8); break;
            default:
                throw std::runtime_error("multistream_async: num_states > " +
                    std::to_string(BITPACKED_MAX_WORDS * 64) + " not supported");
        }
#undef LAUNCH_ASYNC
        CUDA_CHECK(cudaMemcpyAsync(h_flags + lo, d_flags + lo, sizeof(int) * nstr,
            cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaMemcpyAsync(h_lens + lo, d_lens + lo, sizeof(int) * nstr,
            cudaMemcpyDeviceToHost, st));
    }
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop);
    CUDA_CHECK(cudaDeviceSynchronize());
    float total_ms = 0.0f; cudaEventElapsedTime(&total_ms, start, stop);

    // h_flags/h_lens ARE the numpy output buffers — async D2H already wrote them.
    for (int s = 0; s < N_STREAMS; ++s) cudaStreamDestroy(streams[s]);
    if (in_len) cudaHostUnregister(h_in);
    cudaHostUnregister(h_flags); cudaHostUnregister(h_lens);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return {flags, lens, total_ms};
}

