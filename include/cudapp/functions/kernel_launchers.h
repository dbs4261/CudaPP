//
// Created by Daniel Simon on 2/8/20.
//

#ifndef CUDAPP_KERNEL_LAUNCHERS_H
#define CUDAPP_KERNEL_LAUNCHERS_H

#include "include/cudapp/utilities/ide_helpers.h"

#include <array>
#include <functional>

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
  array.at(N) = PointerToVal<N + 1>(array, std::forward(args...));
}

}

template <typename ... Args>
std::array<void*, sizeof...(Args)> PointerArray(Args&& ... args) {
  std::array<void*, sizeof...(Args)> array;
  detail::PointerToVal<0>(array, std::forward(args...));
  return array;
}

//TODO:(Daniel.Simon) Object lifetime will be a nightmare with this. Imagine a pass by value going out of scope before the stream launches the kernel.
template <typename F, typename ... Args>
cudaError_t LaunchFunction(dim3 grid, dim3 block, std::size_t shared_memory, cudaStream_t stream, std::function<void(Args...)> function, Args ... args) {
  auto args_array = PointerArray(std::forward(args...));
  cudaLaunchKernel(static_cast<const void*>(function), grid, block, args_array.data(), shared_memory, stream);
  return cudaGetLastError();
}

}

#endif //CUDAPP_KERNEL_LAUNCHERS_H
