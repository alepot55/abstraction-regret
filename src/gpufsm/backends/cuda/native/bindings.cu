// The Python module surface — nothing but the pybind11 block.
//
// Every entry point is implemented in a sibling .cu and declared in include/api.hpp.

#include "include/api.hpp"

PYBIND11_MODULE(_cuda, m) {
    m.doc() = "gpufsm CUDA backend (dense + bit-packed + multi-stream [+ shared-CSR/async/worklist] NFA kernels)";
    m.def("run_dense", &run_dense,
          "Simulate an NFA (CSR, int8 working set) over an input; returns (accepted, match_len, kernel_ms).");
    m.def("run_bitpacked", &run_bitpacked,
          "Simulate an NFA (CSR, packed-bitmask working set) over an input; returns (accepted, match_len, kernel_ms).");
    m.def("run_multistream", &run_multistream,
          "Simulate an NFA over a batch (one thread/string, global CSR); returns (flags, lens, kernel_ms).");
    m.def("run_multistream_shared", &run_multistream_shared,
          "Multi-stream with read-only CSR staged into shared memory; returns (flags, lens, kernel_ms).");
    m.def("run_multistream_async", &run_multistream_async,
          "Multi-stream with pinned host staging + streamed async H2D/kernel/D2H overlap; "
          "returns (flags, lens, total_ms).");
    m.def("run_worklist", &run_worklist,
          "Work-efficient multi-stream (iterate active states + frontier eps-closure); "
          "returns (flags, lens, kernel_ms).");
    m.def("run_worklist_global", &run_worklist_global,
          "Work-efficient worklist with a global working set — no state-count cap; "
          "returns (flags, lens, kernel_ms).");
    m.def("run_worklist_warp", &run_worklist_warp,
          "Block-parallel (warp-per-string) work-efficient worklist; no state-count cap; "
          "returns (flags, lens, kernel_ms).");
    m.def("run_worklist_compact", &run_worklist_compact,
          "Compacted active-ID worklist (one thread/string; O(active) per symbol, no "
          "O(nwords) bitmap scan); returns (flags, lens, kernel_ms).");
    m.def("run_worklist_shared", &run_worklist_shared,
          "Shared-memory block-cooperative worklist (working set in dynamic shared mem; "
          "num_states up to 98304, though occupancy collapses long before); returns "
          "(flags, lens, kernel_ms).");
    m.def("run_dfa", &run_dfa,
          "DFA simulation (dense transition-table lookup per byte, memory-bound); "
          "returns (flags, lens, kernel_ms).");
}
