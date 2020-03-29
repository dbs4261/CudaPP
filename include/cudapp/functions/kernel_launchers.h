//
// Created by Daniel Simon on 2/8/20.
//

#ifndef CUDAPP_KERNEL_LAUNCHERS_H
#define CUDAPP_KERNEL_LAUNCHERS_H

#include "cudapp/utilities/ide_helpers.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <tuple>
#include <vector>

#include "cudapp/exceptions/cuda_exception.h"

namespace cudapp {

enum struct CooperativeMultiDeviceFlags : unsigned int {
  BothSync   = 0,
  PreSync  = cudaCooperativeLaunchMultiDeviceNoPreSync,
  PostSync = cudaCooperativeLaunchMultiDeviceNoPostSync,
  NoSync   = cudaCooperativeLaunchMultiDeviceNoPreSync |
             cudaCooperativeLaunchMultiDeviceNoPostSync,
};

[[nodiscard]] constexpr inline CooperativeMultiDeviceFlags operator|(CooperativeMultiDeviceFlags a, CooperativeMultiDeviceFlags b) {
  using U = std::underlying_type_t<CooperativeMultiDeviceFlags>;
  return static_cast<CooperativeMultiDeviceFlags>(static_cast<U>(a) | static_cast<U>(b));
}

namespace detail {

template<typename T> struct identity {using type = T;};
template<typename T> using identity_t = typename identity<T>::type;

template <typename ... Args>
void LaunchKernelHelper(dim3 grid, dim3 block, std::size_t shared_memory, cudaStream_t stream, void(*function)(Args...), identity_t<Args> ... args) {
  std::array<void*, sizeof...(Args)> args_array = {std::addressof(args)...};
  cudaError_t ret = cudaLaunchKernel(reinterpret_cast<const void*>(function), grid, block, args_array.data(), shared_memory, stream);
  if (ret != cudaSuccess) {
    throw cudapp::CudaException(ret);
  }
  ret = cudaGetLastError();
  if (ret != cudaSuccess) {
    throw cudapp::CudaException(ret);
  }
}

template <typename ... Args>
void LaunchCooperativeKernelHelper(dim3 grid, dim3 block, std::size_t shared_memory, cudaStream_t stream, void(*function)(Args...), identity_t<Args> ... args) {
  std::array<void*, sizeof...(Args)> args_array = {std::addressof(args)...};
  cudaError_t ret = cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(function), grid, block, args_array.data(), shared_memory, stream);
  if (ret != cudaSuccess) {
    throw cudapp::CudaException(ret);
  }
  ret = cudaGetLastError();
  if (ret != cudaSuccess) {
    throw cudapp::CudaException(ret);
  }
}

}

//<editor-fold desc="SimpleKernelLaunchers">

template <typename ... Args>
void LaunchKernel(dim3 grid, dim3 block, std::size_t shared_memory, cudaStream_t stream, void(*function)(Args...), detail::identity_t<Args>&& ... args) {
  detail::LaunchKernelHelper(grid, block, shared_memory, stream, function, std::forward<Args>(args)...);
}

template <typename ... Args>
void LaunchKernel(dim3 grid, dim3 block, std::size_t shared_memory, void(*function)(Args...), detail::identity_t<Args>&& ... args) {
  detail::LaunchKernelHelper(grid, block, shared_memory, cudaStream_t{0}, function, std::forward<Args>(args)...);
}

template <typename ... Args>
void LaunchKernel(dim3 grid, dim3 block, void(*function)(Args...), detail::identity_t<Args>&& ... args) {
  detail::LaunchKernelHelper(grid, block, std::size_t{0}, cudaStream_t{0}, function, std::forward<Args>(args)...);
}
//</editor-fold>

//<editor-fold desc="CooperativeKernelLaunchers">

template <typename ... Args>
void LaunchCooperativeKernel(dim3 grid, dim3 block, std::size_t shared_memory, cudaStream_t stream, void(*function)(Args...), detail::identity_t<Args>&& ... args) {
  detail::LaunchCooperativeKernelHelper(grid, block, shared_memory, stream, function, std::forward<Args>(args)...);
}

template <typename ... Args>
void LaunchCooperativeKernel(dim3 grid, dim3 block, std::size_t shared_memory, void(*function)(Args...), detail::identity_t<Args>&& ... args) {
  detail::LaunchCooperativeKernelHelper(grid, block, shared_memory, cudaStream_t{0}, function, std::forward<Args>(args)...);
}

template <typename ... Args>
void LaunchCooperativeKernel(dim3 grid, dim3 block, void(*function)(Args...), detail::identity_t<Args>&& ... args) {
  detail::LaunchCooperativeKernelHelper(grid, block, std::size_t{0}, cudaStream_t{0}, function, std::forward<Args>(args)...);
}
//</editor-fold>

template <typename ... Args>
class MultiDeviceCoorperativeLauncher {
 public:
  explicit MultiDeviceCoorperativeLauncher(dim3 _grid, dim3 _block, std::size_t _shared_memory,
      void(*_function)(Args...), CooperativeMultiDeviceFlags _flags=CooperativeMultiDeviceFlags::BothSync) :
      grid(_grid), block(_block), shared_memory(_shared_memory), function(_function), flags(_flags), arguments() {}

  [[nodiscard]] auto SimultaneousStreams() const {
    return arguments.size();
  }

  MultiDeviceCoorperativeLauncher& AddStream(cudaStream_t stream, Args ... args) {
    arguments.emplace_back(stream, std::addressof(args)...);
    return *this;
  }

  [[nodiscard]] std::array<void*, sizeof...(Args)>& StreamArgumentPointers(cudaStream_t stream) {
    for (auto it = arguments.begin(); it != arguments.end(); ++it) {
      if (it->first == stream) {
        return it->second;
      }
    }
    throw std::out_of_range("This function does not have arguments to launch on the given stream");
  }

  [[nodiscard]] const std::array<void*, sizeof...(Args)>& StreamArgumentPointers(cudaStream_t stream) const {
    for (auto it = arguments.begin(); it != arguments.end(); ++it) {
      if (it->first == stream) {
        return it->second;
      }
    }
    throw std::out_of_range("This function does not have arguments to launch on the given stream");
  }

  void LaunchKernels() const {
    assert(arguments.size() > 0);
    std::vector<cudaLaunchParams> params(arguments.size());
    std::transform(arguments.begin(), arguments.end(), params.begin(),
        [this](const auto& args){
      return cudaLaunchParams{.func=this->function, .gridDim=this->grid, .blockDim=this->block,
          .args=args.second, .sharedMem=this->shared_memory, .stream=args.first};
    });
    cudaError_t ret = cudaLaunchCooperativeKernelMultiDevice(
        params.data(), params.size(), static_cast<unsigned int>(this->flags));
    if (ret != cudaSuccess) {
      throw CudaException(ret);
    }
    ret = cudaGetLastError();
    if (ret != cudaSuccess) {
      throw CudaException(ret);
    }
  }

 protected:
  dim3 grid;
  dim3 block;
  std::size_t shared_memory;
  CooperativeMultiDeviceFlags flags;
  void(*function)(Args...);
  std::vector<std::pair<cudaStream_t, std::array<void*, sizeof...(Args)>>> arguments;
};

}

#endif //CUDAPP_KERNEL_LAUNCHERS_H
