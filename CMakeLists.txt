cmake_minimum_required(VERSION 3.23.0)
project(Encodec LANGUAGES C CXX)

# Helpers
set(IS_GCC      "$<CXX_COMPILER_ID:GNU>")
set(IS_CLANG    "$<CXX_COMPILER_ID:Clang>")
set(IS_MSVC     "$<CXX_COMPILER_ID:MSVC>")
set(IS_NOT_MSVC "$<NOT:${IS_MSVC}>")

# Dependencies
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CPM_SOURCE_CACHE ${CMAKE_SOURCE_DIR}/.cache/cpm CACHE PATH "Directory to cache CPM packages" FORCE)
include(cmake/CPM.cmake)
include(CMakeDependentOption)
find_package(AVX)
find_package(NEON)

CPMAddPackage(
  NAME Eigen3
  VERSION 5.0.1
  URL https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz
  OPTIONS "EIGEN_BUILD_DOC OFF" "EIGEN_BUILD_PKGCONFIG OFF" "EIGEN_BUILD_TESTING OFF" "EIGEN_BUILD_DEMOS OFF" "EIGEN_BUILD_SPBENCH OFF")

# Options
option(ENCODEC_BUILD_TESTS       "Build unit tests" ${PROJECT_IS_TOP_LEVEL})
option(ENCODEC_BUILD_BENCH       "Build benchmarks" ${PROJECT_IS_TOP_LEVEL})
option(ENCODEC_BUILD_EXAMPLES    "Build examples"   ${PROJECT_IS_TOP_LEVEL})
option(ENCODEC_ENABLE_SANITIZERS "Build unit tests with sanitizers" OFF)

# Weights
function(add_zipped_weight_source out_var zip_file output_name)
  set(output_file "${CMAKE_CURRENT_BINARY_DIR}/generated/weights/${output_name}")
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated/weights")
  add_custom_command(
      OUTPUT "${output_file}"
      COMMAND ${CMAKE_COMMAND} -E tar xf "${zip_file}"
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated/weights"
      DEPENDS "${zip_file}"
      COMMENT "Extracting ${output_name}"
      VERBATIM)
  set(${out_var} "${output_file}" PARENT_SCOPE)
endfunction()

function(add_weight_target target_name weights_cpp) 
  add_library(${target_name} ${weights_cpp})
  target_compile_features(${target_name} PUBLIC cxx_std_20)
  target_include_directories(${target_name}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/lib>
    PUBLIC $<INSTALL_INTERFACE:include>)
endfunction()

add_zipped_weight_source(ENCODER24_WEIGHTS_CPP "${CMAKE_CURRENT_SOURCE_DIR}/src/lib/weights/encoder24.cpp.xz" "encoder24.cpp")
add_zipped_weight_source(DECODER24_WEIGHTS_CPP "${CMAKE_CURRENT_SOURCE_DIR}/src/lib/weights/decoder24.cpp.xz" "decoder24.cpp")
add_zipped_weight_source(RVQ24_WEIGHTS_CPP     "${CMAKE_CURRENT_SOURCE_DIR}/src/lib/weights/rvq24.cpp.xz"     "rvq24.cpp")

add_weight_target(encodec_encoder24 ${ENCODER24_WEIGHTS_CPP})
add_weight_target(encodec_decoder24 ${DECODER24_WEIGHTS_CPP})
add_weight_target(encodec_rvq24     ${RVQ24_WEIGHTS_CPP})
add_library(encodec::encoder24 ALIAS encodec_encoder24)
add_library(encodec::decoder24 ALIAS encodec_decoder24)
add_library(encodec::rvq24     ALIAS encodec_rvq24)

# Lib
add_library(encodec src/lib/encodec.cpp)
add_library(encodec::encodec ALIAS encodec)
target_compile_features(encodec PUBLIC cxx_std_20)

target_include_directories(encodec
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/lib>
  PUBLIC $<INSTALL_INTERFACE:include>)

target_compile_options(encodec 
  PRIVATE $<${IS_NOT_MSVC}:-Wall;-Wextra;-Wpedantic;-Werror>
  PRIVATE $<$<BOOL:${AVX_FOUND}>:${AVX_CFLAGS}>
  PRIVATE $<$<BOOL:${NEON_FOUND}>:${NEON_CFLAGS}>
  PRIVATE $<$<AND:${IS_NOT_MSVC},$<CONFIG:Release>>:-ffast-math>
  PRIVATE $<$<AND:${IS_MSVC},$<CONFIG:Release>>:/fp:fast>)
  
target_link_libraries(encodec PRIVATE Eigen3::Eigen)

if (ENCODEC_ENABLE_SANITIZERS)
  message(STATUS "Enabling sanitizers")
  target_compile_options(encodec PUBLIC $<$<OR:${IS_GCC},${IS_CLANG}>:-fsanitize=address,undefined;-fno-omit-frame-pointer;-fno-sanitize-recover=undefined>)
  target_link_options(encodec PUBLIC $<$<OR:${IS_GCC},${IS_CLANG}>:-fsanitize=address,undefined>)
endif()

# Test
if (ENCODEC_BUILD_TESTS)
  add_executable(encodec_tests 
    src/tests/doctest.cpp
    src/tests/encodec.cpp)

  target_link_libraries(encodec_tests PRIVATE
    encodec::encodec 
    encodec::encoder24
    encodec::decoder24
    encodec::rvq24)

  target_compile_options(encodec_tests PRIVATE $<${IS_NOT_MSVC}:-Wall -Wextra -Wpedantic -Werror>)
endif()

# Bench
if (ENCODEC_BUILD_BENCH)
  add_executable(encodec_bench
    src/bench/main.cpp
    src/bench/format.cpp
    src/bench/encodec.cpp)

  target_link_libraries(encodec_bench PRIVATE
    encodec::encodec 
    encodec::encoder24
    encodec::decoder24
    encodec::rvq24)

  target_compile_options(encodec_bench PRIVATE $<${IS_NOT_MSVC}:-Wall -Wextra -Wpedantic -Werror>)
endif()

# Example
if (ENCODEC_BUILD_EXAMPLES)
  add_executable(encodec_example src/examples/main.cpp)
  target_link_libraries(encodec_example PRIVATE 
    encodec::encodec 
    encodec::encoder24
    encodec::decoder24
    encodec::rvq24)
endif()