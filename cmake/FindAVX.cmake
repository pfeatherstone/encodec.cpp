include(FindPackageHandleStandardArgs)
include(CheckCSourceCompiles)
include(CheckTypeSize)
set(CMAKE_REQUIRED_QUIET_SAV ${CMAKE_REQUIRED_QUIET})
set(CMAKE_REQUIRED_QUIET     TRUE)
check_type_size( "void*" SIZE_OF_VOID_PTR)

# Test for AVX
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_REQUIRED_FLAGS "-mavx")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(CMAKE_REQUIRED_FLAGS "-xHost")
elseif(MSVC AND SIZE_OF_VOID_PTR EQUAL 4)
    set(CMAKE_REQUIRED_FLAGS "/arch:AVX")
endif()

check_c_source_compiles(
"
#include <immintrin.h>
int main()
{
    __m256 a = _mm256_set_ps (-1.0f, 2.0f, -3.0f, 4.0f, -1.0f, 2.0f, -3.0f, 4.0f);
    __m256 b = _mm256_set_ps (1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f);
    __m256 result = _mm256_add_ps (a, b);
    return 0;
}" HAVE_AVX)

if (HAVE_AVX)
    list(APPEND AVX_CFLAGS ${CMAKE_REQUIRED_FLAGS})
endif()

# Test for AVX2
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_REQUIRED_FLAGS "-mavx2")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(CMAKE_REQUIRED_FLAGS "-xHost")
elseif(MSVC AND SIZE_OF_VOID_PTR EQUAL 4)
    set(CMAKE_REQUIRED_FLAGS "/arch:AVX2")
endif()

check_c_source_compiles(
"
#include <immintrin.h>
int main()
{
    __m256i a       = _mm256_set_epi32 (-1, 2, -3, 4, -1, 2, -3, 4);
    __m256i result  = _mm256_abs_epi32 (a);
    return 0;
}
" HAVE_AVX2)

if (HAVE_AVX2)
    list(APPEND AVX_CFLAGS ${CMAKE_REQUIRED_FLAGS})
endif()

# Test for FMA
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_REQUIRED_FLAGS "-mfma")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(CMAKE_REQUIRED_FLAGS "-xHost")
elseif(MSVC AND SIZE_OF_VOID_PTR EQUAL 4)
    set(CMAKE_REQUIRED_FLAGS "/arch:AVX2")
endif()

check_c_source_compiles(
"#include <immintrin.h>
int main()
{
__m128d x;
x = _mm_fmadd_pd(x,x,x);

return 0;
}" HAVE_FMA)

if (HAVE_FMA)
    list(APPEND AVX_CFLAGS ${CMAKE_REQUIRED_FLAGS})
endif()

set(CMAKE_REQUIRED_QUIET ${CMAKE_REQUIRED_QUIET_SAV})
find_package_handle_standard_args(AVX DEFAULT_MSG AVX_CFLAGS)