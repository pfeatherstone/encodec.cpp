#include <cstdarg>
#include "format.h"

std::string format(const char* fmt, ...)
{
    va_list args0, args1;
    va_start(args0, fmt);
    va_copy(args1, args0);
    const int size = std::vsnprintf(nullptr, 0, fmt, args0);
    va_end(args0);

    if (size < 0) {
        va_end(args1);
        return {};
    }

    std::string str(size+1, '\0');
    const int ret = std::vsnprintf(&str[0], str.size(), fmt, args1);
    va_end(args1);
    if (ret < 0) str.clear();
    else         str.resize(ret);
    return str;
}