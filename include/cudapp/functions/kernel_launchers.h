//
// Created by Daniel Simon on 2/8/20.
//

#ifndef CUDAPP_KERNEL_LAUNCHERS_H
#define CUDAPP_KERNEL_LAUNCHERS_H

#include "cudapp/utilities/ide_helpers.h"

#include <array>
#include <functional>
#include <tuple>

#include "cudapp/exceptions/cuda_exception.h"

namespace cudapp {

namespace detail {

template <std::size_t N, std::size_t M, typename T>
inline void PointerToVal(std::array<void*, M>& array, T&& val) {
  static_assert(N == M - 1, "Error in array indexing");
  array.at(N) = reinterpret_cast<void*>(std::addressof(val));
}

template <std::size_t N, std::size_t M, typename T, typename ... Args>
inline void PointerToVal(std::array<void*, M>& array, T&& val, Args&& ... args) {
  static_assert(N < M, "Array index out of bounds");
  array.at(N) = PointerToVal<N + 1>(array, std::forward<Args>(args)...);
}

template <typename ... Args>
std::array<void*, sizeof...(Args)> PointerArray(Args&& ... args) {
  std::array<void*, sizeof...(Args)> array;
  detail::PointerToVal<0>(array, std::forward<Args>(args)...);
  return array;
}

template <typename T>
using ReferenceWrapWhenNeeded = typename
    std::conditional<std::is_lvalue_reference<T>::value, std::reference_wrapper<T>, T>::type;

template <typename ... FnArgs, typename ... PassedArgs, std::size_t ... Indices>
void LaunchFunctionHelper(std::index_sequence<Indices...>, dim3 grid, dim3 block,
    std::size_t shared_memory, cudaStream_t stream, void(*function)(FnArgs...), PassedArgs&& ... args) {
  static_assert(sizeof...(FnArgs) == sizeof...(PassedArgs), "Incorrect number of arguments passed for kernel");
  std::tuple<ReferenceWrapWhenNeeded<FnArgs>...> converted_arguments(std::forward<PassedArgs>(args)...);
  std::array<void*, sizeof...(FnArgs)> args_array = {std::addressof(std::get<Indices>(converted_arguments))...};
  auto ret = cudaLaunchKernel(reinterpret_cast<const void*>(function), grid, block, args_array.data(), shared_memory, stream);
  if (ret != cudaSuccess) {
    throw cudapp::CudaException(ret);
  }
  ret = cudaGetLastError();
  if (ret != cudaSuccess) {
    throw cudapp::CudaException(ret);
  }
}

template <typename ... FnArgs, typename ... PassedArgs>
void LaunchFunctionHelper(dim3 grid, dim3 block, std::size_t shared_memory, cudaStream_t stream, void(*function)(FnArgs...), PassedArgs&& ... args) {
  LaunchFunctionHelper(std::make_index_sequence<sizeof...(FnArgs)>{}, grid, block, shared_memory, stream, function, std::forward<PassedArgs>(args)...);
}

}

template <typename ... FnArgs, typename ... PassedArgs>
void LaunchFunction(dim3 grid, dim3 block, std::size_t shared_memory, cudaStream_t stream, void(*function)(FnArgs...), PassedArgs&& ... args) {
  detail::LaunchFunctionHelper(grid, block, shared_memory, stream, function, std::forward<PassedArgs>(args)...);
}

template <typename ... FnArgs, typename ... PassedArgs>
void LaunchFunction(dim3 grid, dim3 block, std::size_t shared_memory, void(*function)(FnArgs...), PassedArgs&& ... args) {
  detail::LaunchFunctionHelper(grid, block, shared_memory, cudaStream_t{0}, function, std::forward<PassedArgs>(args)...);
}

template <typename ... FnArgs, typename ... PassedArgs>
void LaunchFunction(dim3 grid, dim3 block, void(*function)(FnArgs...), PassedArgs&& ... args) {
  detail::LaunchFunctionHelper(grid, block, std::size_t{0}, cudaStream_t{0}, function, std::forward<PassedArgs>(args)...);
}

}

#endif //CUDAPP_KERNEL_LAUNCHERS_H
