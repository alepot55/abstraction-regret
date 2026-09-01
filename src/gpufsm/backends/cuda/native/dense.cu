// Dense NFA technique — one int8 slot per state.
//
// The faithful, correctness-first port of gpufsm.reference.simulate (latch-first-match),
// kept as the abstraction-regret baseline: 4 bytes per state where one bit suffices.
// bitpacked.cu is the same algorithm on the packed layout.

#include "include/api.hpp"

// One thread simulates the whole single stream (baseline). Working sets are
// int8 device buffers (one slot per state), mirroring the reference algorithm.
__global__ void dense_nfa_kernel(
    const int* sym_row_ptr, const int* sym_targets, const int* sym_symbols,
    const int* eps_row_ptr, const int* eps_targets,
    const signed char* accept,
    const int* input_symbols, int input_len,
    int num_states, int start_state, int uses_any,
    signed char* cur, signed char* nxt,
    int* out_flag, int* out_len) {

    for (int i = 0; i < num_states; ++i) cur[i] = 0;
    cur[start_state] = 1;

    // Epsilon closure: num_states passes guarantee convergence.
    for (int it = 0; it < num_states; ++it) {
        for (int s = 0; s < num_states; ++s) {
            if (cur[s]) {
                for (int k = eps_row_ptr[s]; k < eps_row_ptr[s + 1]; ++k)
                    cur[eps_targets[k]] = 1;
            }
        }
    }

    for (int s = 0; s < num_states; ++s) {
        if (cur[s] && accept[s]) { *out_flag = 1; *out_len = 0; return; }
    }

    for (int pos = 0; pos < input_len; ++pos) {
        int sym = input_symbols[pos];
        for (int i = 0; i < num_states; ++i) nxt[i] = 0;
        for (int s = 0; s < num_states; ++s) {
            if (cur[s]) {
                for (int k = sym_row_ptr[s]; k < sym_row_ptr[s + 1]; ++k) {
                    int tsym = sym_symbols[k];
                    if (tsym == sym || (uses_any && tsym == ANY_SYMBOL))
                        nxt[sym_targets[k]] = 1;
                }
            }
        }
        for (int it = 0; it < num_states; ++it) {
            for (int s = 0; s < num_states; ++s) {
                if (nxt[s]) {
                    for (int k = eps_row_ptr[s]; k < eps_row_ptr[s + 1]; ++k)
                        nxt[eps_targets[k]] = 1;
                }
            }
        }
        for (int i = 0; i < num_states; ++i) cur[i] = nxt[i];
        for (int s = 0; s < num_states; ++s) {
            if (cur[s] && accept[s]) { *out_flag = 1; *out_len = pos + 1; return; }
        }
    }
    *out_flag = 0; *out_len = 0;
}

// Bit-packed technique — the memory-centric thesis artifact.
// The active state-set is a packed bitmask (1 bit/state, 64-bit words) held in
// thread-local registers instead of an int8-per-state buffer in (global-backed)
// local memory. Templating on NWORDS makes the working set a compile-time array:
// for num_states <= 64 (NWORDS==1) it is a single register-resident
// `unsigned long long` with zero global traffic for the state vector — exactly the
// byte->bit + global->register ablation. Same CSR algorithm as the dense kernel.

std::tuple<bool, int, float> run_dense(
    py::array_t<int> sym_row_ptr, py::array_t<int> sym_targets, py::array_t<int> sym_symbols,
    py::array_t<int> eps_row_ptr, py::array_t<int> eps_targets,
    py::array_t<signed char> accept,
    py::array_t<int> input_symbols,
    int num_states, int start_state, int uses_any) {

    DeviceScope scope;
    const int* d_srp = dev_copy(sym_row_ptr, scope);
    const int* d_st = dev_copy(sym_targets, scope);
    const int* d_ss = dev_copy(sym_symbols, scope);
    const int* d_erp = dev_copy(eps_row_ptr, scope);
    const int* d_et = dev_copy(eps_targets, scope);
    const signed char* d_acc = dev_copy(accept, scope);
    const int* d_in = dev_copy(input_symbols, scope);
    int input_len = static_cast<int>(input_symbols.request().size);

    signed char *d_cur, *d_nxt;
    int *d_flag, *d_len;
    CUDA_CHECK(cudaMalloc(&d_cur, num_states));
    scope.own(d_cur);
    CUDA_CHECK(cudaMalloc(&d_nxt, num_states));
    scope.own(d_nxt);
    CUDA_CHECK(cudaMalloc(&d_flag, sizeof(int)));
    scope.own(d_flag);
    CUDA_CHECK(cudaMalloc(&d_len, sizeof(int)));
    scope.own(d_len);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    dense_nfa_kernel<<<1, 1>>>(d_srp, d_st, d_ss, d_erp, d_et, d_acc, d_in, input_len,
                               num_states, start_state, uses_any, d_cur, d_nxt, d_flag, d_len);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    float kernel_ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));

    int h_flag = 0, h_len = 0;
    CUDA_CHECK(cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_len, d_len, sizeof(int), cudaMemcpyDeviceToHost));
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return {h_flag != 0, h_len, kernel_ms};
}

// Launch the bitpacked kernel specialized for the NFA's word count.
