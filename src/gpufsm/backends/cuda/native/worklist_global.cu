// Work-efficient worklist variants with an out-of-register working set.
//
// Four kernels sharing this translation unit because they share __device__ helpers
// (worklist_shared_kernel calls eps_closure_warp, which device code cannot reach across
// a TU boundary without relocatable device code):
//
//   global   one thread/string, working set in global memory, no state-count cap
//   compact  one thread/string, frontier as an active-ID array instead of a bitmap scan
//   warp     one warp/string, lanes partition the state words, atomicOr into next-set
//   shared   block-cooperative, working set in dynamic shared memory. The hard cap is
//            98304 states (one warp's 4*nwords*8 bytes must fit 48 KB); well before that,
//            occupancy collapses to one warp per block, so the useful range is far lower.

#include "include/api.hpp"

__device__ __forceinline__ void eps_closure_global(
    unsigned long long* S, unsigned long long* F, unsigned long long* B, int nwords,
    const int* eps_row_ptr, const int* eps_targets) {
    for (int w = 0; w < nwords; ++w) F[w] = S[w];
    bool any = true;
    while (any) {
        for (int w = 0; w < nwords; ++w) B[w] = 0ULL;
        for (int w = 0; w < nwords; ++w) {
            unsigned long long b = F[w];
            while (b) {
                int s = w * WORD_BITS + __ffsll(b) - 1;
                b &= b - 1;
                for (int k = eps_row_ptr[s]; k < eps_row_ptr[s + 1]; ++k) {
                    int t = eps_targets[k];
                    B[word_of(t)] |= bit_of(t);
                }
            }
        }
        any = false;
        for (int w = 0; w < nwords; ++w) {
            B[w] &= ~S[w];
            S[w] |= B[w];
            F[w] = B[w];
            if (B[w]) any = true;
        }
    }
}

// Work-efficient worklist with the working set in GLOBAL memory — NO state-count cap
// (the register worklist is capped at 512). nwords words per string; cur/nxt/frontier/
// newb are per-string global slices. One thread per string. Same latch-first-match
// verdict as the reference. This is what scales the engine to large (ANMLZoo-sized) automata.
__global__ void worklist_global_kernel(
    const int* sym_row_ptr, const int* sym_targets, const int* sym_symbols,
    const int* eps_row_ptr, const int* eps_targets,
    const unsigned long long* accept_words,
    const int* input_data, const int* input_offsets, int num_strings,
    int num_states, int start_state, int uses_any, int nwords,
    unsigned long long* cur, unsigned long long* nxt,
    unsigned long long* frontier, unsigned long long* newb,
    int* out_flags, int* out_lens) {
    (void)num_states;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_strings) return;
    size_t off = (size_t)i * nwords;
    unsigned long long* C = cur + off;
    unsigned long long* N = nxt + off;
    unsigned long long* F = frontier + off;
    unsigned long long* B = newb + off;
    const int* input_symbols = input_data + input_offsets[i];
    int input_len = input_offsets[i + 1] - input_offsets[i];

    for (int w = 0; w < nwords; ++w) C[w] = 0ULL;
    C[word_of(start_state)] |= bit_of(start_state);
    eps_closure_global(C, F, B, nwords, eps_row_ptr, eps_targets);

    int out_f = 0, out_l = 0, done = 0;
    for (int w = 0; w < nwords; ++w) if (C[w] & accept_words[w]) done = 1;
    if (done) { out_f = 1; out_l = 0; }

    for (int pos = 0; pos < input_len && !done; ++pos) {
        int sym = input_symbols[pos];
        for (int w = 0; w < nwords; ++w) N[w] = 0ULL;
        for (int w = 0; w < nwords; ++w) {
            unsigned long long b = C[w];
            while (b) {
                int s = w * WORD_BITS + __ffsll(b) - 1;
                b &= b - 1;
                for (int k = sym_row_ptr[s]; k < sym_row_ptr[s + 1]; ++k) {
                    int tsym = sym_symbols[k];
                    if (tsym == sym || (uses_any && tsym == ANY_SYMBOL)) {
                        int t = sym_targets[k];
                        N[word_of(t)] |= bit_of(t);
                    }
                }
            }
        }
        eps_closure_global(N, F, B, nwords, eps_row_ptr, eps_targets);
        for (int w = 0; w < nwords; ++w) C[w] = N[w];
        int m = 0;
        for (int w = 0; w < nwords; ++w) if (C[w] & accept_words[w]) m = 1;
        if (m) { out_f = 1; out_l = pos + 1; done = 1; }
    }
    out_flags[i] = out_f; out_lens[i] = out_l;
}

// ---- Compacted active-ID worklist (one thread/string) ------------------------------
// The bitmap worklist iterates all nwords words per symbol even when the active set is
// tiny (measured: brill averages ~1.5 active states over 667 words -> ~99.8% wasted scan).
// This kernel keeps the active set as a COMPACTED array of state IDs (frontier), so per
// symbol it does O(active) work, not O(nwords). A per-string `visited` bitmap dedups the
// next frontier; we clear only the touched bits (O(active)), never the whole bitmap, so the
// per-symbol cost is independent of num_states. One thread per string (isolates the
// compaction effect vs worklist_global; both are 1-thread/string). Same latch-first-match.
// frontier_a/frontier_b are int32[num_states] per string; visited is ull[nwords] per string.
__device__ __forceinline__ int eps_closure_compact(
    int* F, int nf, unsigned long long* V,
    const int* eps_row_ptr, const int* eps_targets) {
    // BFS expansion in place: F[0..nf) is the queue; appends grow nf. V dedups.
    for (int j = 0; j < nf; ++j) {
        int s = F[j];
        for (int k = eps_row_ptr[s]; k < eps_row_ptr[s + 1]; ++k) {
            int t = eps_targets[k];
            if (!(V[word_of(t)] & bit_of(t))) {
                V[word_of(t)] |= bit_of(t);
                F[nf++] = t;
            }
        }
    }
    return nf;
}

__global__ void worklist_compact_kernel(
    const int* sym_row_ptr, const int* sym_targets, const int* sym_symbols,
    const int* eps_row_ptr, const int* eps_targets,
    const unsigned long long* accept_words,
    const int* input_data, const int* input_offsets, int num_strings,
    int num_states, int start_state, int uses_any, int nwords,
    int* frontier_a, int* frontier_b, unsigned long long* visited,
    int* out_flags, int* out_lens) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_strings) return;
    int* FA = frontier_a + (size_t)i * num_states;
    int* FB = frontier_b + (size_t)i * num_states;
    unsigned long long* V = visited + (size_t)i * nwords;
    const int* input_symbols = input_data + input_offsets[i];
    int input_len = input_offsets[i + 1] - input_offsets[i];

    for (int w = 0; w < nwords; ++w) V[w] = 0ULL;  // one-time O(nwords) zero
    int nf = 0;
    V[word_of(start_state)] |= bit_of(start_state);
    FA[nf++] = start_state;
    nf = eps_closure_compact(FA, nf, V, eps_row_ptr, eps_targets);

    int out_f = 0, out_l = 0, done = 0;
    for (int j = 0; j < nf && !done; ++j)
        if (accept_words[word_of(FA[j])] & bit_of(FA[j])) { out_f = 1; out_l = 0; done = 1; }

    for (int pos = 0; pos < input_len && !done; ++pos) {
        int sym = input_symbols[pos];
        // clear visited bits of the current frontier (O(active)) -> V all-zero
        for (int j = 0; j < nf; ++j) V[word_of(FA[j])] &= ~bit_of(FA[j]);
        int nfb = 0;
        for (int j = 0; j < nf; ++j) {
            int s = FA[j];
            for (int k = sym_row_ptr[s]; k < sym_row_ptr[s + 1]; ++k) {
                int tsym = sym_symbols[k];
                if (tsym == sym || (uses_any && tsym == ANY_SYMBOL)) {
                    int t = sym_targets[k];
                    if (!(V[word_of(t)] & bit_of(t))) {
                        V[word_of(t)] |= bit_of(t);
                        FB[nfb++] = t;
                    }
                }
            }
        }
        nfb = eps_closure_compact(FB, nfb, V, eps_row_ptr, eps_targets);
        for (int j = 0; j < nfb; ++j)
            if (accept_words[word_of(FB[j])] & bit_of(FB[j])) {
                out_f = 1; out_l = pos + 1; done = 1; break;
            }
        int* tmp = FA; FA = FB; FB = tmp;  // swap frontiers
        nf = nfb;
    }
    out_flags[i] = out_f; out_lens[i] = out_l;
}

// ---- Block-parallel (warp-per-string) work-efficient worklist ----------------------
// One *warp* (32 lanes) cooperates on one string instead of one thread. The 32 lanes
// partition the nwords words of the working set (lane handles words w with w%32==lane);
// cross-word transition/epsilon scatter uses atomicOr into the shared global next-set.
// Rationale: the 1-thread/string worklist under-utilizes the GPU at small batch (Nsight:
// 17% occupancy, 2 blocks) and issues one string's loads serially. A warp spreads those
// loads across 32 lanes -> more memory-level parallelism, the path toward the memory-bound
// regime for large (ANMLZoo-scale) automata. Same latch-first-match verdict as the oracle.

// Warp-cooperative frontier epsilon-closure over a GLOBAL bitset. All 32 lanes active.
__device__ __forceinline__ void eps_closure_warp(
    unsigned long long* S, unsigned long long* F, unsigned long long* B, int nwords,
    const int* eps_row_ptr, const int* eps_targets, int lane) {
    for (int w = lane; w < nwords; w += 32) F[w] = S[w];
    __syncwarp();
    bool any = true;
    while (any) {
        for (int w = lane; w < nwords; w += 32) B[w] = 0ULL;
        __syncwarp();
        for (int w = lane; w < nwords; w += 32) {
            unsigned long long b = F[w];
            while (b) {
                int s = w * WORD_BITS + __ffsll(b) - 1;
                b &= b - 1;
                for (int k = eps_row_ptr[s]; k < eps_row_ptr[s + 1]; ++k) {
                    int t = eps_targets[k];
                    atomicOr(&B[word_of(t)], bit_of(t));
                }
            }
        }
        __syncwarp();
        int any_local = 0;
        for (int w = lane; w < nwords; w += 32) {
            unsigned long long nb = B[w] & ~S[w];
            S[w] |= nb;
            F[w] = nb;
            if (nb) any_local = 1;
        }
        any = __any_sync(0xffffffffu, any_local) != 0;
        __syncwarp();
    }
}

// The body shared by worklist_warp_kernel and worklist_shared_kernel. The two differ only in
// their prologue -- where C/N/F/B point, global slices or dynamic shared memory -- and were
// 40 byte-identical lines apart before this was factored out. Keeping two copies of a
// simulation loop is how a semantics fix reaches one arm and not the other.
//
// __forceinline__ so the shared-memory variant still resolves its accesses to ld.shared /
// st.shared: inlining happens before nvcc's address-space inference. Both callers live in
// this translation unit, which is what makes a __device__ helper legal here at all (the
// kernels are compiled without relocatable device code).
__device__ __forceinline__ void simulate_one_warp(
    unsigned long long* C, unsigned long long* N,
    unsigned long long* F, unsigned long long* B,
    int lane, int nwords, int warp,
    const int* sym_row_ptr, const int* sym_targets, const int* sym_symbols,
    const int* eps_row_ptr, const int* eps_targets,
    const unsigned long long* accept_words,
    const int* input_data, const int* input_offsets,
    int start_state, int uses_any,
    int* out_flags, int* out_lens) {
    const int* input_symbols = input_data + input_offsets[warp];
    int input_len = input_offsets[warp + 1] - input_offsets[warp];

    for (int w = lane; w < nwords; w += 32) C[w] = 0ULL;
    __syncwarp();
    if (lane == 0) C[word_of(start_state)] |= bit_of(start_state);
    __syncwarp();
    eps_closure_warp(C, F, B, nwords, eps_row_ptr, eps_targets, lane);

    int out_f = 0, out_l = 0, done = 0, acc_local = 0;
    for (int w = lane; w < nwords; w += 32) if (C[w] & accept_words[w]) acc_local = 1;
    if (__any_sync(0xffffffffu, acc_local)) { out_f = 1; out_l = 0; done = 1; }

    for (int pos = 0; pos < input_len && !done; ++pos) {
        int sym = input_symbols[pos];
        for (int w = lane; w < nwords; w += 32) N[w] = 0ULL;
        __syncwarp();
        for (int w = lane; w < nwords; w += 32) {
            unsigned long long b = C[w];
            while (b) {
                int s = w * WORD_BITS + __ffsll(b) - 1;
                b &= b - 1;
                for (int k = sym_row_ptr[s]; k < sym_row_ptr[s + 1]; ++k) {
                    int tsym = sym_symbols[k];
                    if (tsym == sym || (uses_any && tsym == ANY_SYMBOL)) {
                        int t = sym_targets[k];
                        atomicOr(&N[word_of(t)], bit_of(t));
                    }
                }
            }
        }
        __syncwarp();
        eps_closure_warp(N, F, B, nwords, eps_row_ptr, eps_targets, lane);
        for (int w = lane; w < nwords; w += 32) C[w] = N[w];
        __syncwarp();
        int m_local = 0;
        for (int w = lane; w < nwords; w += 32) if (C[w] & accept_words[w]) m_local = 1;
        if (__any_sync(0xffffffffu, m_local)) { out_f = 1; out_l = pos + 1; done = 1; }
    }
    if (lane == 0) { out_flags[warp] = out_f; out_lens[warp] = out_l; }
}



__global__ void worklist_warp_kernel(
    const int* sym_row_ptr, const int* sym_targets, const int* sym_symbols,
    const int* eps_row_ptr, const int* eps_targets,
    const unsigned long long* accept_words,
    const int* input_data, const int* input_offsets, int num_strings,
    int start_state, int uses_any, int nwords,
    unsigned long long* cur, unsigned long long* nxt,
    unsigned long long* frontier, unsigned long long* newb,
    int* out_flags, int* out_lens) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= num_strings) return;
    size_t off = (size_t)warp * nwords;
    unsigned long long* C = cur + off;
    unsigned long long* N = nxt + off;
    unsigned long long* F = frontier + off;
    unsigned long long* B = newb + off;
    simulate_one_warp(C, N, F, B, lane, nwords, warp,
                      sym_row_ptr, sym_targets, sym_symbols, eps_row_ptr, eps_targets,
                      accept_words, input_data, input_offsets, start_state, uses_any,
                      out_flags, out_lens);
}

// ---- Shared-memory block-cooperative worklist -------------------------------------
// Same warp-per-string scheme as worklist_warp, but the per-string working set
// (cur/nxt/frontier/newb, 4*nwords words) lives in DYNAMIC SHARED memory instead of
// global. This privatizes the working-set traffic the warp kernel issues to global:
// a test of whether memory-layout privatization helps once the kernel is work-efficient
// (it did NOT in the compute-bound full-scan regime; see multistream_shared). Requires
// warps_per_block * 4 * nwords * 8 bytes of shared memory; the launcher picks
// warps_per_block to fit 48 KB and the technique is only offered when 1 warp/block fits.
__global__ void worklist_shared_kernel(
    const int* sym_row_ptr, const int* sym_targets, const int* sym_symbols,
    const int* eps_row_ptr, const int* eps_targets,
    const unsigned long long* accept_words,
    const int* input_data, const int* input_offsets, int num_strings,
    int start_state, int uses_any, int nwords,
    int* out_flags, int* out_lens) {
    extern __shared__ unsigned long long wl_smem[];
    int warps_per_block = blockDim.x >> 5;
    int warp_in_block = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int warp = blockIdx.x * warps_per_block + warp_in_block;
    if (warp >= num_strings) return;
    unsigned long long* base = wl_smem + (size_t)warp_in_block * 4 * nwords;
    unsigned long long* C = base;
    unsigned long long* N = base + nwords;
    unsigned long long* F = base + 2 * nwords;
    unsigned long long* B = base + 3 * nwords;
    simulate_one_warp(C, N, F, B, lane, nwords, warp,
                      sym_row_ptr, sym_targets, sym_symbols, eps_row_ptr, eps_targets,
                      accept_words, input_data, input_offsets, start_state, uses_any,
                      out_flags, out_lens);
}

// Host entry point for worklist_global_kernel: one thread per string, working set in
// global memory, no state-count cap. (The paragraph that used to sit here described the
// multi-stream ablation in bitpacked.cu, left behind when the monolith was split.)
std::tuple<py::array_t<int>, py::array_t<int>, float> run_worklist_global(
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
    unsigned long long *d_cur, *d_nxt, *d_fr, *d_nb;
    size_t ws = sizeof(unsigned long long) * (size_t)(num_strings ? num_strings : 1) * nwords;
    CUDA_CHECK(cudaMalloc(&d_cur, ws));
    scope.own(d_cur);
    CUDA_CHECK(cudaMalloc(&d_nxt, ws));
    scope.own(d_nxt);
    CUDA_CHECK(cudaMalloc(&d_fr, ws));
    scope.own(d_fr);
    CUDA_CHECK(cudaMalloc(&d_nb, ws));
    scope.own(d_nb);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    if (num_strings > 0) {
        int threads = 256, blocks = (num_strings + threads - 1) / threads;
        worklist_global_kernel<<<blocks, threads>>>(
            d_srp, d_st, d_ss, d_erp, d_et, d_acc, d_in, d_off, num_strings,
            num_states, start_state, uses_any, nwords, d_cur, d_nxt, d_fr, d_nb, d_flags, d_lens);
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

// Block-parallel (warp-per-string) worklist — same global working set as run_worklist_global,
// but launches 32 lanes per string. Returns (flags, lens, kernel_ms).
std::tuple<py::array_t<int>, py::array_t<int>, float> run_worklist_warp(
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
    unsigned long long *d_cur, *d_nxt, *d_fr, *d_nb;
    size_t ws = sizeof(unsigned long long) * (size_t)(num_strings ? num_strings : 1) * nwords;
    CUDA_CHECK(cudaMalloc(&d_cur, ws));
    scope.own(d_cur);
    CUDA_CHECK(cudaMalloc(&d_nxt, ws));
    scope.own(d_nxt);
    CUDA_CHECK(cudaMalloc(&d_fr, ws));
    scope.own(d_fr);
    CUDA_CHECK(cudaMalloc(&d_nb, ws));
    scope.own(d_nb);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    if (num_strings > 0) {
        int threads = 256;  // 8 warps/block
        int blocks = (num_strings * 32 + threads - 1) / threads;
        worklist_warp_kernel<<<blocks, threads>>>(
            d_srp, d_st, d_ss, d_erp, d_et, d_acc, d_in, d_off, num_strings,
            start_state, uses_any, nwords, d_cur, d_nxt, d_fr, d_nb, d_flags, d_lens);
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

// Compacted active-ID worklist (one thread/string): O(active) per symbol, not O(nwords).
// Returns (flags, lens, kernel_ms).
std::tuple<py::array_t<int>, py::array_t<int>, float> run_worklist_compact(
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

    int *d_flags, *d_lens, *d_fa, *d_fb;
    unsigned long long* d_vis;
    int ns1 = num_strings ? num_strings : 1;
    CUDA_CHECK(cudaMalloc(&d_flags, sizeof(int) * ns1));
    scope.own(d_flags);
    CUDA_CHECK(cudaMalloc(&d_lens, sizeof(int) * ns1));
    scope.own(d_lens);
    CUDA_CHECK(cudaMalloc(&d_fa, sizeof(int) * (size_t)ns1 * num_states));
    scope.own(d_fa);
    CUDA_CHECK(cudaMalloc(&d_fb, sizeof(int) * (size_t)ns1 * num_states));
    scope.own(d_fb);
    CUDA_CHECK(cudaMalloc(&d_vis, sizeof(unsigned long long) * (size_t)ns1 * nwords));
    scope.own(d_vis);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    if (num_strings > 0) {
        int threads = 256, blocks = (num_strings + threads - 1) / threads;
        worklist_compact_kernel<<<blocks, threads>>>(
            d_srp, d_st, d_ss, d_erp, d_et, d_acc, d_in, d_off, num_strings,
            num_states, start_state, uses_any, nwords, d_fa, d_fb, d_vis, d_flags, d_lens);
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

// Shared-memory block-cooperative worklist. Working set in dynamic shared memory; the
// launcher picks warps_per_block so warps_per_block*4*nwords*8 fits 48 KB. Raises if even
// one warp's working set (4*nwords*8 bytes) exceeds 48 KB. Returns (flags, lens, kernel_ms).
std::tuple<py::array_t<int>, py::array_t<int>, float> run_worklist_shared(
    py::array_t<int> sym_row_ptr, py::array_t<int> sym_targets, py::array_t<int> sym_symbols,
    py::array_t<int> eps_row_ptr, py::array_t<int> eps_targets,
    py::array_t<unsigned long long> accept_words,
    py::array_t<int> input_data, py::array_t<int> input_offsets,
    int num_states, int start_state, int uses_any) {

    int nwords = words_for(num_states);
    int num_strings = static_cast<int>(input_offsets.request().size) - 1;
    const size_t SMEM_CAP = 48 * 1024;  // bytes
    size_t per_warp = (size_t)4 * nwords * sizeof(unsigned long long);
    if (per_warp > SMEM_CAP) {
        // Derive the limit from the same constants the check uses. It used to be spelled
        // "num_states > ~1536", which is the cap on *nwords*, not on states -- wrong by a
        // factor of 64, and an error message that names the wrong quantity sends the reader
        // to change the wrong thing.
        const size_t max_states = (SMEM_CAP / (4 * sizeof(unsigned long long))) * 64;
        throw std::runtime_error(
            "worklist_shared: one warp's working set (" + std::to_string(per_warp) +
            " B) exceeds the " + std::to_string(SMEM_CAP / 1024) +
            " KB shared-memory budget; this technique supports num_states <= " +
            std::to_string(max_states) +
            ". Use worklist_warp or worklist_global above that.");
    }
    int warps_per_block = 1;
    while (warps_per_block < 8 && (warps_per_block + 1) * per_warp <= SMEM_CAP) warps_per_block++;
    int threads = warps_per_block * 32;
    size_t shared_bytes = (size_t)warps_per_block * per_warp;

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
        int blocks = (num_strings + warps_per_block - 1) / warps_per_block;
        worklist_shared_kernel<<<blocks, threads, shared_bytes>>>(
            d_srp, d_st, d_ss, d_erp, d_et, d_acc, d_in, d_off, num_strings,
            start_state, uses_any, nwords, d_flags, d_lens);
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

