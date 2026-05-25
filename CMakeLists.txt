cmake_minimum_required(VERSION 3.23.0)

project(Encodec LANGUAGES C CXX)

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

# Lib
get_filename_component(ENCODER_DATA ${CMAKE_CURRENT_SOURCE_DIR}/src/lib/weights/encoder.dat REALPATH)
get_filename_component(DECODER_DATA ${CMAKE_CURRENT_SOURCE_DIR}/src/lib/weights/decoder.dat REALPATH)
get_filename_component(RVQ_DATA     ${CMAKE_CURRENT_SOURCE_DIR}/src/lib/weights/rvq.dat     REALPATH)

add_library(encodec src/lib/encodec.cpp)
add_library(encodec::encodec ALIAS encodec)
target_compile_features(encodec PUBLIC cxx_std_20)

target_compile_definitions(encodec PRIVATE 
  ENCODER_DATA="${ENCODER_DATA}"
  DECODER_DATA="${DECODER_DATA}"
  RVQ_DATA="${RVQ_DATA}")

target_include_directories(encodec
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/lib>
  PUBLIC $<INSTALL_INTERFACE:include>)

target_compile_options(encodec 
  PRIVATE -Wall -Wpedantic
  PRIVATE $<$<BOOL:${AVX_FOUND}>:${AVX_CFLAGS}>
  PRIVATE $<$<BOOL:${NEON_FOUND}>:${NEON_CFLAGS}>
  PRIVATE $<$<CONFIG:Release>:-ffast-math>
)
  
target_link_libraries(encodec PRIVATE Eigen3::Eigen)

# Example
add_executable(encodec_example src/examples/main.cpp)
target_link_libraries(encodec_example PRIVATE encodec::encodec)