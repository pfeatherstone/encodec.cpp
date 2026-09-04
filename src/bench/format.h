#pragma once

#include <string>

#if defined(__GNUC__) || defined(__clang__)
    #define PRINTF_FORMAT(fmt, args) __attribute__((format(printf, fmt, args)))
#else
    #define PRINTF_FORMAT(fmt, args)
#endif

std::string format(const char* fmt, ...) PRINTF_FORMAT(1, 2);