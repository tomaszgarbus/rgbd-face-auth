#ifndef BASIC_TYPES_H
#define BASIC_TYPES_H

#include <array>
template<typename ElementT, size_t HEIGHT, size_t WIDTH>
using matrix = std::array< std::array<ElementT, WIDTH>, HEIGHT>;

#endif
