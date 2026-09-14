cmake_minimum_required(VERSION 3.23.0)
project(Encodec LANGUAGES C CXX)

# Helpers
set(IS_GCC      "$<CXX_COMPILER_ID:GNU>")
set(IS_CLANG    "$<CXX_COMPILER_ID:Clang>")
set(IS_MSVC     "$<CXX_COMPILER_ID:MSVC>")
set(IS_NOT_MSVC "$<NOT:${IS_MSVC}>")
set(IS_X86      "$<IN_LIST:$<LOWER_CASE:${CMAKE_SYSTEM_PROCESSOR}>,x86_64;amd64;i386;i486;i586;i686>")

# Dependencies
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CPM_SOURCE_CACHE ${CMAKE_SOURCE_DIR}/.cache/cpm CACHE PATH "Directory to cache CPM packages" FORCE)
include(cmake/CPM.cmake)
find_package(AVX)
find_package(NEON)
find_package(MDAZ)

CPMAddPackage(
  NAME Eigen3
  VERSION 5.0.1
  URL https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz
  OPTIONS "EIGEN_BUILD_DOC OFF" "EIGEN_BUILD_PKGCONFIG OFF" "EIGEN_BUILD_TESTING OFF" "EIGEN_BUILD_DEMOS OFF" "EIGEN_BUILD_SPBENCH OFF")

# Options
option(ENCODEC_BUILD_TESTS       "Build unit tests" ${PROJECT_IS_TOP_LEVEL})
option(ENCODEC_BUILD_BENCH       "Build benchmarks" ${PROJECT_IS_TOP_LEVEL})
option(ENCODEC_BUILD_EXAMPLES    "Build examples"   ${PROJECT_IS_TOP_LEVEL})
option(ENCODEC_ENABLE_SANITIZERS "Build with sanitizers" OFF)

# Helper
function(add_weight_target target_name zip_file) 
  cmake_path(ABSOLUTE_PATH zip_file BASE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} NORMALIZE OUTPUT_VARIABLE zip_path)
  cmake_path(GET zip_file STEM LAST_ONLY output_name)
  set(output_dir  ${CMAKE_CURRENT_BINARY_DIR}/generated/weights)
  set(output_file ${output_dir}/${output_name})
  file(MAKE_DIRECTORY ${output_dir})
  add_custom_command(
      OUTPUT            ${output_file}
      COMMAND           ${CMAKE_COMMAND} -E tar xf "${zip_path}"
      WORKING_DIRECTORY ${output_dir}
      DEPENDS           ${zip_path}
      COMMENT           "Extracting ${output_name}"
      VERBATIM)
  add_library(${target_name} ${output_file})
  target_compile_features(${target_name} PUBLIC cxx_std_20)
endfunction()

# Weights
add_weight_target(encodec_encoder24 src/lib/weights/encoder24.cpp.xz)
add_weight_target(encodec_decoder24 src/lib/weights/decoder24.cpp.xz)
add_weight_target(encodec_rvq24     src/lib/weights/rvq24.cpp.xz)
add_weight_target(encodec_encoder48 src/lib/weights/encoder48.cpp.xz)
add_weight_target(encodec_decoder48 src/lib/weights/decoder48.cpp.xz)
add_weight_target(encodec_rvq48     src/lib/weights/rvq48.cpp.xz)
add_library(encodec::encoder24 ALIAS encodec_encoder24)
add_library(encodec::decoder24 ALIAS encodec_decoder24)
add_library(encodec::rvq24     ALIAS encodec_rvq24)
add_library(encodec::encoder48 ALIAS encodec_encoder48)
add_library(encodec::decoder48 ALIAS encodec_decoder48)
add_library(encodec::rvq48     ALIAS encodec_rvq48)

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
  PRIVATE $<$<AND:${IS_NOT_MSVC},$<CONFIG:Release>>:-ffast-math;-fno-finite-math-only>
  PRIVATE $<$<AND:${IS_MSVC},$<CONFIG:Release>>:/fp:fast>)
  
target_link_libraries(encodec PRIVATE Eigen3::Eigen)

if (ENCODEC_ENABLE_SANITIZERS)
  message(STATUS "Enabling sanitizers")
  target_compile_options(encodec PUBLIC $<$<OR:${IS_GCC},${IS_CLANG}>:-fsanitize=address,undefined;-fno-omit-frame-pointer;-fno-sanitize-recover=undefined>)
  target_link_options(encodec PUBLIC $<$<OR:${IS_GCC},${IS_CLANG}>:-fsanitize=address,undefined>)
endif()

add_library(encodec24 INTERFACE)
target_link_libraries(encodec24 INTERFACE encodec encodec_encoder24 encodec_decoder24 encodec_rvq24)
add_library(encodec::encodec24 ALIAS encodec24)

add_library(encodec48 INTERFACE)
target_link_libraries(encodec48 INTERFACE encodec encodec_encoder48 encodec_decoder48 encodec_rvq48)
add_library(encodec::encodec48 ALIAS encodec48)

add_library(encodec_all INTERFACE)
target_link_libraries(encodec_all INTERFACE encodec24 encodec48)
add_library(encodec::encodec_all ALIAS encodec_all)

# Test
get_filename_component(ENCODEC_DATA_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/data/ REALPATH)

if (ENCODEC_BUILD_TESTS)
  add_executable(encodec_tests 
    src/tests/doctest.cpp
    src/tests/encodec.cpp)

  target_link_libraries(encodec_tests PRIVATE encodec::encodec_all)
  target_compile_options(encodec_tests PRIVATE $<${IS_NOT_MSVC}:-Wall;-Wextra;-Wpedantic;-Werror>)
  target_compile_definitions(encodec_tests PRIVATE ENCODEC_DATA_PATH="${ENCODEC_DATA_PATH}")
endif()

# Bench
if (ENCODEC_BUILD_BENCH)
  add_executable(encodec_bench
    src/bench/main.cpp
    src/bench/encodec.cpp)

  target_link_libraries(encodec_bench PRIVATE encodec::encodec_all)
  target_compile_options(encodec_bench PRIVATE $<${IS_NOT_MSVC}:-Wall;-Wextra;-Wpedantic;-Werror>)
  target_compile_options(encodec_bench PRIVATE PRIVATE $<$<BOOL:${MDAZ_FOUND}>:${MDAZ_CFLAGS}>)
  target_link_options   (encodec_bench PRIVATE PRIVATE $<$<BOOL:${MDAZ_FOUND}>:${MDAZ_LDFLAGS}>)
  target_compile_definitions(encodec_bench PRIVATE ENCODEC_DATA_PATH="${ENCODEC_DATA_PATH}")
endif()

# Example
if (ENCODEC_BUILD_EXAMPLES)
  add_executable(encodec_example src/examples/main.cpp)
  target_link_libraries(encodec_example PRIVATE encodec::encodec_all)
endif()
