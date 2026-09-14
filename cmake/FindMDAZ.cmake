include(FindPackageHandleStandardArgs)
include(CheckCXXCompilerFlag)
include(CheckLinkerFlag)

check_cxx_compiler_flag("-mdaz-ftz" HAS_MDAZ_FTZ_COMPILE)
check_linker_flag(CXX "-mdaz-ftz" HAS_MDAZ_FTZ_LINK)

if(HAS_MDAZ_FTZ_COMPILE AND HAS_MDAZ_FTZ_LINK)
    set(MDAZ_CFLAGS  "-mdaz-ftz")
    set(MDAZ_LDFLAGS "-mdaz-ftz")
endif()

find_package_handle_standard_args(MDAZ DEFAULT_MSG MDAZ_CFLAGS MDAZ_LDFLAGS)