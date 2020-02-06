//
// Created by Daniel Simon on 7/18/19.
//

#ifndef CUDAPP_VECTOR_TYPE_TRAITS_H
#define CUDAPP_VECTOR_TYPE_TRAITS_H

#include <vector_types.h>

template <typename T, unsigned int N>
struct composed_vector_type {};

template <typename T>
struct composed_vector_type<T, 1> {
  using type = T;
};

#ifndef COMPOSED_VECTOR_TYPE_INSTANCE
#define COMPOSED_VECTOR_TYPE_INSTANCE(T, N, CUDA) \
template <>\
struct composed_vector_type<T, N> {\
  using type = CUDA##N;\
};
#endif

#ifndef COMPOSED_VECTOR_TYPE_INSTANCES
#define COMPOSED_VECTOR_TYPE_INSTANCES(T, CUDA) \
COMPOSED_VECTOR_TYPE_INSTANCE(T, 2, CUDA) \
COMPOSED_VECTOR_TYPE_INSTANCE(T, 3, CUDA) \
COMPOSED_VECTOR_TYPE_INSTANCE(T, 4, CUDA)
#endif

template <typename T, unsigned int N>
using composed_vector_type_t = typename composed_vector_type<T, N>::type;

template <typename T>
struct vector_type {};

#ifndef VECTOR_TYPE_INSTANCE
#define VECTOR_TYPE_INSTANCE(t, n, CUDA) \
template <> \
struct vector_type<CUDA##n> { \
  static constexpr unsigned int dims = n; \
  using type = t; \
};
#endif

#ifndef VECTOR_TYPE_INSTANCES
#define VECTOR_TYPE_INSTANCES(T, CUDA) \
VECTOR_TYPE_INSTANCE(T, 1, CUDA) \
VECTOR_TYPE_INSTANCE(T, 2, CUDA) \
VECTOR_TYPE_INSTANCE(T, 3, CUDA) \
VECTOR_TYPE_INSTANCE(T, 4, CUDA)
#endif

template <typename T>
using vector_type_t = typename vector_type<T>::type;

// NVCC doesnt have support for variable templates yet, so we have to protect this for the time being
#ifdef __cpp_variable_templates
template <typename T>
static constexpr unsigned int vector_channels_v = vector_type<T>::dims;
#endif

#ifndef VECTOR_TYPE_IS_FLOATING
#define VECTOR_TYPE_IS_FLOATING(t, n, CUDA) \
namespace std { \
template <> \
struct is_floating_point<CUDA##n> : std::true_type {}; \
}
#endif

#ifndef VECTOR_TYPE_TRAIT_FLOATING_INSTANCES
#define VECTOR_TYPE_TRAIT_FLOATING_INSTANCES(T, CUDA) \
VECTOR_TYPE_IS_FLOATING(T, 1, CUDA) \
VECTOR_TYPE_IS_FLOATING(T, 2, CUDA) \
VECTOR_TYPE_IS_FLOATING(T, 3, CUDA) \
VECTOR_TYPE_IS_FLOATING(T, 4, CUDA)
#endif

#ifndef VECTOR_TYPE_TRAIT
#define VECTOR_TYPE_TRAIT(T, CUDA) \
COMPOSED_VECTOR_TYPE_INSTANCES(T, CUDA) \
VECTOR_TYPE_INSTANCES(T, CUDA)
#endif

VECTOR_TYPE_TRAIT(unsigned char, uchar)
VECTOR_TYPE_TRAIT(char, char)
VECTOR_TYPE_TRAIT(unsigned short, ushort)
VECTOR_TYPE_TRAIT(short, short)
VECTOR_TYPE_TRAIT(unsigned int, uint)
VECTOR_TYPE_TRAIT(int, int)
VECTOR_TYPE_TRAIT(unsigned long, ulong)
VECTOR_TYPE_TRAIT(long, long)
VECTOR_TYPE_TRAIT(float, float)
VECTOR_TYPE_TRAIT_FLOATING_INSTANCES(float, float)
VECTOR_TYPE_TRAIT(unsigned long long, ulonglong)
VECTOR_TYPE_TRAIT(long long, longlong)
VECTOR_TYPE_TRAIT(double, double)
VECTOR_TYPE_TRAIT_FLOATING_INSTANCES(double, double)

#endif //CUDAPP_VECTOR_TYPE_TRAITS_H
