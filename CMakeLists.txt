cmake_minimum_required(VERSION 3.12...3.18)
include(GNUInstallDirs)
project(einsum_bit_cast LANGUAGES CXX)

message(STATUS "Using CMake version " ${CMAKE_VERSION})
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

include_directories(${CMAKE_CURRENT_LIST_DIR}/lib)

# abseil
include_directories(${ABSEIL_ROOT})
add_library(absl::base SHARED IMPORTED)

install(TARGETS einsum_bit_cast)