cmake_minimum_required(VERSION 3.18)
project(gpufsm LANGUAGES CXX)

# -----------------------------------------------------------------------------
# Graceful CUDA build.
# The CUDA pybind11 extension is OPTIONAL. With -DGPUFSM_BUILD_CUDA=ON and a
# working CUDA toolkit it is compiled and installed into the package; otherwise
# the build is a no-op and the pure-Python package (CPU + Triton) still works.
# -----------------------------------------------------------------------------
option(GPUFSM_BUILD_CUDA "Build the CUDA pybind11 extension" OFF)

if(NOT GPUFSM_BUILD_CUDA)
    message(STATUS "gpufsm: GPUFSM_BUILD_CUDA=OFF -> skipping CUDA extension (pure-Python install).")
    return()
endif()

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(WARNING "gpufsm: CUDA requested but no CUDA compiler found -> skipping CUDA extension.")
    return()
endif()

# Architecture default MUST be set before enable_language(CUDA): that call seeds
# its own CMAKE_CUDA_ARCHITECTURES default, after which a `NOT DEFINED` guard is
# always false. Real SASS only (no embedded PTX) for Turing..Ada.
# Rationale: when the CUDA *toolkit* is newer than the max CUDA version the
# installed *driver* supports, any embedded PTX is rejected at module load
# ("the provided PTX was compiled with an unsupported toolchain"), even when
# matching SASS is present. Real-arch cubins still load under CUDA minor-version
# compatibility. The "-real" suffix suppresses PTX generation. Note: bare numbers
# ("89") and the "native" keyword DO embed PTX and would re-trigger the error on a
# toolkit/driver mismatch. Override with -DCMAKE_CUDA_ARCHITECTURES to add arches.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75-real 80-real 86-real 89-real)
endif()

enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# One translation unit per kernel family under native/, plus bindings.cu. They are
# compiled WITHOUT relocatable device code, so each __device__ helper must live in the
# same .cu as its callers; shared declarations go in native/include/.
file(GLOB_RECURSE GPUFSM_CUDA_SOURCES CONFIGURE_DEPENDS
    "src/gpufsm/backends/cuda/native/*.cu"
    "src/gpufsm/backends/cuda/native/*.cpp")

if(NOT GPUFSM_CUDA_SOURCES)
    message(WARNING "gpufsm: no CUDA sources found yet under src/gpufsm/backends/cuda -> skipping.")
    return()
endif()

pybind11_add_module(_cuda ${GPUFSM_CUDA_SOURCES})
target_include_directories(_cuda PRIVATE src/gpufsm/backends/cuda/native)
target_compile_options(_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --extended-lambda --expt-relaxed-constexpr>)
install(TARGETS _cuda DESTINATION gpufsm/backends/cuda)
