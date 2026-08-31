// Host entry points exported by the kernel translation units to bindings.cu.
//
// Each is implemented in exactly one .cu; bindings.cu contains nothing but the
// PYBIND11_MODULE block, so adding a technique means one .cu, one declaration here and
// one m.def line.
//
// These declarations are transcribed verbatim from the definitions, spelling included.
// That is not pedantry: on LP64 Linux `uint64_t` is `unsigned long`, a *different* type
// from the `unsigned long long` the kernels declare, and the mismatch surfaces as an
// undefined symbol at link time rather than as a compiler error here.
//
// The signatures are also the extension's ABI, marshalled from
// gpufsm/backends/cuda/{nfa,dfa}.py; change one and both sides must move together.

#pragma once

#include "common.cuh"

#include <tuple>

// ---- dense.cu ----------------------------------------------------------------------

std::tuple<bool, int, float> run_dense(
    py::array_t<int> sym_row_ptr, py::array_t<int> sym_targets, py::array_t<int> sym_symbols,
    py::array_t<int> eps_row_ptr, py::array_t<int> eps_targets,
    py::array_t<signed char> accept,
    py::array_t<int> input_symbols,
    int num_states, int start_state, int uses_any);

// ---- bitpacked.cu ------------------------------------------------------------------

std::tuple<bool, int, float> run_bitpacked(
    py::array_t<int> sym_row_ptr, py::array_t<int> sym_targets, py::array_t<int> sym_symbols,
    py::array_t<int> eps_row_ptr, py::array_t<int> eps_targets,
    py::array_t<unsigned long long> accept_words,
    py::array_t<int> input_symbols,
    int num_states, int start_state, int uses_any);

std::tuple<py::array_t<int>, py::array_t<int>, float> run_multistream(
    py::array_t<int> sym_row_ptr, py::array_t<int> sym_targets, py::array_t<int> sym_symbols,
    py::array_t<int> eps_row_ptr, py::array_t<int> eps_targets,
    py::array_t<unsigned long long> accept_words,
    py::array_t<int> input_data, py::array_t<int> input_offsets,
    int num_states, int start_state, int uses_any);

std::tuple<py::array_t<int>, py::array_t<int>, float> run_multistream_shared(
    py::array_t<int> sym_row_ptr, py::array_t<int> sym_targets, py::array_t<int> sym_symbols,
    py::array_t<int> eps_row_ptr, py::array_t<int> eps_targets,
    py::array_t<unsigned long long> accept_words,
    py::array_t<int> input_data, py::array_t<int> input_offsets,
    int num_states, int start_state, int uses_any);

std::tuple<py::array_t<int>, py::array_t<int>, float> run_multistream_async(
    py::array_t<int> sym_row_ptr, py::array_t<int> sym_targets, py::array_t<int> sym_symbols,
    py::array_t<int> eps_row_ptr, py::array_t<int> eps_targets,
    py::array_t<unsigned long long> accept_words,
    py::array_t<int> input_data, py::array_t<int> input_offsets,
    int num_states, int start_state, int uses_any);

// ---- worklist_register.cu ----------------------------------------------------------

std::tuple<py::array_t<int>, py::array_t<int>, float> run_worklist(
    py::array_t<int> sym_row_ptr, py::array_t<int> sym_targets, py::array_t<int> sym_symbols,
    py::array_t<int> eps_row_ptr, py::array_t<int> eps_targets,
    py::array_t<unsigned long long> accept_words,
    py::array_t<int> input_data, py::array_t<int> input_offsets,
    int num_states, int start_state, int uses_any);

// ---- worklist_global.cu ------------------------------------------------------------

std::tuple<py::array_t<int>, py::array_t<int>, float> run_worklist_global(
    py::array_t<int> sym_row_ptr, py::array_t<int> sym_targets, py::array_t<int> sym_symbols,
    py::array_t<int> eps_row_ptr, py::array_t<int> eps_targets,
    py::array_t<unsigned long long> accept_words,
    py::array_t<int> input_data, py::array_t<int> input_offsets,
    int num_states, int start_state, int uses_any);

std::tuple<py::array_t<int>, py::array_t<int>, float> run_worklist_warp(
    py::array_t<int> sym_row_ptr, py::array_t<int> sym_targets, py::array_t<int> sym_symbols,
    py::array_t<int> eps_row_ptr, py::array_t<int> eps_targets,
    py::array_t<unsigned long long> accept_words,
    py::array_t<int> input_data, py::array_t<int> input_offsets,
    int num_states, int start_state, int uses_any);

std::tuple<py::array_t<int>, py::array_t<int>, float> run_worklist_compact(
    py::array_t<int> sym_row_ptr, py::array_t<int> sym_targets, py::array_t<int> sym_symbols,
    py::array_t<int> eps_row_ptr, py::array_t<int> eps_targets,
    py::array_t<unsigned long long> accept_words,
    py::array_t<int> input_data, py::array_t<int> input_offsets,
    int num_states, int start_state, int uses_any);

std::tuple<py::array_t<int>, py::array_t<int>, float> run_worklist_shared(
    py::array_t<int> sym_row_ptr, py::array_t<int> sym_targets, py::array_t<int> sym_symbols,
    py::array_t<int> eps_row_ptr, py::array_t<int> eps_targets,
    py::array_t<unsigned long long> accept_words,
    py::array_t<int> input_data, py::array_t<int> input_offsets,
    int num_states, int start_state, int uses_any);

// ---- dfa.cu ------------------------------------------------------------------------

std::tuple<py::array_t<int>, py::array_t<int>, float> run_dfa(
    py::array_t<int> trans, py::array_t<signed char> accept,
    py::array_t<int> input_data, py::array_t<int> input_offsets,
    int num_states, int start_state);
