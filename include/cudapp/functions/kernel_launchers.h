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

template<typename T> struct identity {using type = T;};
template<typename T> using identity_t = typename identity<T>::type;

template <typename ... Args>
void LaunchFunctionHelper(dim3 grid, dim3 block, std::size_t shared_memory, cudaStream_t stream, void(*function)(Args...), identity_t<Args> ... args) {
  std::array<void*, sizeof...(Args)> args_array = {std::addressof(args)...};
  auto ret = cudaLaunchKernel(reinterpret_cast<const void*>(function), grid, block, args_array.data(), shared_memory, stream);
  if (ret != cudaSuccess) {
    throw cudapp::CudaException(ret);
  }
  ret = cudaGetLastError();
  if (ret != cudaSuccess) {
    throw cudapp::CudaException(ret);
  }
}

}

template <typename ... Args>
void LaunchFunction(dim3 grid, dim3 block, std::size_t shared_memory, cudaStream_t stream, void(*function)(Args...), detail::identity_t<Args>&& ... args) {
  detail::LaunchFunctionHelper(grid, block, shared_memory, stream, function, std::forward<Args>(args)...);
}

template <typename ... Args>
void LaunchFunction(dim3 grid, dim3 block, std::size_t shared_memory, void(*function)(Args...), detail::identity_t<Args>&& ... args) {
  detail::LaunchFunctionHelper(grid, block, shared_memory, cudaStream_t{0}, function, std::forward<Args>(args)...);
}

template <typename ... Args>
void LaunchFunction(dim3 grid, dim3 block, void(*function)(Args...), detail::identity_t<Args>&& ... args) {
  detail::LaunchFunctionHelper(grid, block, std::size_t{0}, cudaStream_t{0}, function, std::forward<Args>(args)...);
}

}

#endif //CUDAPP_KERNEL_LAUNCHERS_H
