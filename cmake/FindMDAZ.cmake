include(FindPackageHandleStandardArgs)
include(CheckCSourceCompiles)
set(CMAKE_REQUIRED_QUIET_SAV ${CMAKE_REQUIRED_QUIET})
set(CMAKE_REQUIRED_FLAGS_SAV ${CMAKE_REQUIRED_FLAGS})
set(CMAKE_REQUIRED_QUIET TRUE)
set(CMAKE_REQUIRED_FLAGS "-mdaz-ftz -Werror")

check_c_source_compiles(
  "int main() { return 0; }"
  HAS_MDAZ_FTZ
)

if(HAS_MDAZ_FTZ)
  set(MDAZ_CFLAGS "-mdaz-ftz")
  set(MDAZ_LDFLAGS "-mdaz-ftz")
endif()

set(CMAKE_REQUIRED_QUIET ${CMAKE_REQUIRED_QUIET_SAV})
set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAV})
find_package_handle_standard_args(MDAZ DEFAULT_MSG MDAZ_CFLAGS MDAZ_LDFLAGS)