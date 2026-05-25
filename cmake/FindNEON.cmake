include(FindPackageHandleStandardArgs)
include(CheckCSourceCompiles)
include(CheckTypeSize)
set(CMAKE_REQUIRED_QUIET_SAV ${CMAKE_REQUIRED_QUIET})
set(CMAKE_REQUIRED_QUIET     TRUE)
check_type_size( "void*" SIZE_OF_VOID_PTR)

# Test for NEON
if((CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang") AND SIZE_OF_VOID_PTR EQUAL 4)
    set(CMAKE_REQUIRED_FLAGS "-mfpu=neon")
else()
    set(CMAKE_REQUIRED_FLAGS "")
endif()

check_c_source_compiles(
"
#include <arm_neon.h>
int main(void){
uint8x8_t a = vdup_n_u8(9);
uint8x8_t b = vdup_n_u8(10);
uint8x8_t dst = vmul_u8(a, b);
int r = vget_lane_u8(dst, 0);
int s = vget_lane_u8(dst, 7);
return (r != 90 || s != 90);
}" HAVE_NEON)

if (HAVE_NEON)
    set(NEON_CFLAGS ${CMAKE_REQUIRED_FLAGS})
endif()

set(CMAKE_REQUIRED_QUIET ${CMAKE_REQUIRED_QUIET_SAV})
find_package_handle_standard_args(NEON DEFAULT_MSG NEON_CFLAGS)