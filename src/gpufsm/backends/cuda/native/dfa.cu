// DFA simulation — the memory-bound face of the study.
//
// One dense-table gather per input byte (`cur = trans[cur * 256 + byte]`): a
// data-dependent load feeding a sequential chain, so throughput is set by the table
// layout and the memory system rather than by control flow.

#include "include/api.hpp"

__global__ void dfa_kernel(
    const int* trans, const signed char* accept,
    const int* input_data, const int* input_offsets, int num_strings,
    int num_states, int start_state,
    int* out_flags, int* out_lens) {
    (void)num_states;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_strings) return;
    const int* in = input_data + input_offsets[i];
    int len = input_offsets[i + 1] - input_offsets[i];
    int cur = start_state, out_f = 0, out_l = 0;
    if (accept[cur]) {
        out_f = 1;
    } else {
        for (int p = 0; p < len; ++p) {
            cur = trans[cur * 256 + in[p]];
            if (accept[cur]) { out_f = 1; out_l = p + 1; break; }
        }
    }
    out_flags[i] = out_f; out_lens[i] = out_l;
}

// Returns (flags, lens, kernel_ms) for a batch over a DFA.
std::tuple<py::array_t<int>, py::array_t<int>, float> run_dfa(
    py::array_t<int> trans, py::array_t<signed char> accept,
    py::array_t<int> input_data, py::array_t<int> input_offsets,
    int num_states, int start_state) {

    int num_strings = static_cast<int>(input_offsets.request().size) - 1;
    DeviceScope scope;
    const int* d_trans = dev_copy(trans, scope);
    const signed char* d_acc = dev_copy(accept, scope);
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
        int threads = 256, blocks = (num_strings + threads - 1) / threads;
        dfa_kernel<<<blocks, threads>>>(d_trans, d_acc, d_in, d_off, num_strings,
                                        num_states, start_state, d_flags, d_lens);
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

