//
// Created by Daniel Simon on 2/8/20.
//

#ifndef CUDAPP_KERNEL_LAUNCHERS_H
#define CUDAPP_KERNEL_LAUNCHERS_H

#include "cudapp/utilities/ide_helpers.h"

#include <array>
#include <functional>
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
  array.at(N) = PointerToVal<N + 1>(array, std::forward(args...));
}

template <typename ... Args>
std::array<void*, sizeof...(Args)> PointerArray(Args&& ... args) {
  std::array<void*, sizeof...(Args)> array;
  detail::PointerToVal<0>(array, std::forward(args...));
  return array;
}

}

template <typename F, typename ... Args>
void LaunchFunction(dim3 grid, dim3 block, std::function<void(Args...)> function, Args ... args) {
  LaunchFunction(grid, block, 0, function, args...);
}

template <typename F, typename ... Args>
void LaunchFunction(dim3 grid, dim3 block, std::size_t shared_memory, std::function<void(Args...)> function, Args ... args) {
  LaunchFunction(grid, block, shared_memory, 0, function, args...);
}

template <typename F, typename ... Args>
void LaunchFunction(dim3 grid, dim3 block, std::size_t shared_memory, cudaStream_t stream, std::function<void(Args...)> function, Args ... args) {
  //TODO:(Daniel.Simon) Object lifetime could be a nightmare with this. Imagine a pass by value going out of scope before the stream launches the kernel.
  // const lvalue
  // lvalue
  // const pointer
  // pointer
  // value
  // rvalue
  auto args_array = detail::PointerArray(std::forward(args...));
  auto ret = cudaLaunchKernel(static_cast<const void*>(function), grid, block, args_array.data(), shared_memory, stream);
  if (ret != cudaSuccess) {
    throw cudapp::CudaException(ret);
  }
  ret = cudaGetLastError();
  if (ret != cudaSuccess) {
    throw cudapp::CudaException(ret);
  }
}

}

#endif //CUDAPP_KERNEL_LAUNCHERS_H
