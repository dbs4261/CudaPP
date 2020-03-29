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
#include <unordered_map>
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

/**
 * @brief A helper function to combine flags.
 */
[[nodiscard]] constexpr inline CooperativeMultiDeviceFlags operator|(CooperativeMultiDeviceFlags a, CooperativeMultiDeviceFlags b) {
  using U = std::underlying_type_t<CooperativeMultiDeviceFlags>;
  return static_cast<CooperativeMultiDeviceFlags>(static_cast<U>(a) | static_cast<U>(b));
}

namespace detail {

/**
 * @brief A helper to prevent the function arguments being used for template argument deduction instead of the function.
 * @tparam T The type.
 */
template<typename T> struct identity {using type = T;};
template<typename T> using identity_t = typename identity<T>::type;

/**
 * @brief Does the actual work of launching a kernel.
 * @tparam Args The type of the arguments to be passed to the kernel.
 * @param grid The 3d grid size that the kernel will be launched with.
 * @param block The 3d block size that the kernel will be launched with.
 * @param shared_memory The amount of shared memory in bytes allocated per block.
 * @param stream The stream to launch the kernel on.
 * @param function The kernel to watch.
 * @param args The arguments to pack to the kernel.
 */
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

/**
 * @brief Does the actual work of launching the cooperative kernel.
 * @tparam Args The type of the arguments to be passed to the kernel.
 * @param grid The 3d grid size that the kernel will be launched with.
 * @param block The 3d block size that the kernel will be launched with.
 * @param shared_memory The amount of shared memory in bytes allocated per block.
 * @param stream The stream to launch the kernel on.
 * @param function The cooperative kernel to watch.
 * @param args The arguments to pack to the kernel.
 */
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

/**
 * @brief A pair consisting of a function and an array of pointers to the data that it is called with.
 */
template <typename ... Args>
using HostFunctionnData = std::pair<void(*)(Args...), std::array<void*, sizeof...(Args)>>;

/**
 * @brief A wrapper around a function to be launched from cudaLaunchHostFunc to allow more arguments to be passed.
 * @tparam Args The type of the arguments to be passed to the packed function.
 * @param user_data The wrapped function along with the calling parameters.
 */
template <typename ... Args>
void HostFunctionWrapper(void* user_data) {
  // TODO(Daniel.Simon): check if the passing of the pointer array causes issues with object lifetime. It may be necessary to forward as tuple.
  auto* actual_data = reinterpret_cast<HostFunctionnData<Args...>*>(user_data);
  actual_data->first(*reinterpret_cast<std::add_pointer_t<Args>>(actual_data->second)...);
}

}

//<editor-fold desc="SimpleKernelLaunchers">

/**
 * @brief Launches a standard cuda kernel.
 * @tparam Args The types of the arguments for the Kernel.
 * @param grid The 3d grid size that the kernel will be launched with.
 * @param block The 3d block size that the kernel will be launched with.
 * @param shared_memory The amount of shared memory in bytes allocated per block.
 * @param stream The stream to launch the kernel on.
 * @param function The kernel to launch.
 * @param args The arguments to pass to the kernel.
 */
template <typename ... Args>
void LaunchKernel(dim3 grid, dim3 block, std::size_t shared_memory, cudaStream_t stream, void(*function)(Args...), detail::identity_t<Args>&& ... args) {
  detail::LaunchKernelHelper(grid, block, shared_memory, stream, function, std::forward<Args>(args)...);
}

/**
 * @brief Launches a standard cuda kernel.
 * @tparam Args The types of the arguments for the Kernel.
 * @param grid The 3d grid size that the kernel will be launched with.
 * @param block The 3d block size that the kernel will be launched with.
 * @param shared_memory The amount of shared memory in bytes allocated per block.
 * @param function The kernel to launch.
 * @param args The arguments to pass to the kernel.
 */
template <typename ... Args>
void LaunchKernel(dim3 grid, dim3 block, std::size_t shared_memory, void(*function)(Args...), detail::identity_t<Args>&& ... args) {
  detail::LaunchKernelHelper(grid, block, shared_memory, cudaStream_t{0}, function, std::forward<Args>(args)...);
}

/**
 * @brief Launches a standard cuda kernel.
 * @tparam Args The types of the arguments for the Kernel.
 * @param grid The 3d grid size that the kernel will be launched with.
 * @param block The 3d block size that the kernel will be launched with.
 * @param function The kernel to launch.
 * @param args The arguments to pass to the kernel.
 */
template <typename ... Args>
void LaunchKernel(dim3 grid, dim3 block, void(*function)(Args...), detail::identity_t<Args>&& ... args) {
  detail::LaunchKernelHelper(grid, block, std::size_t{0}, cudaStream_t{0}, function, std::forward<Args>(args)...);
}
//</editor-fold>

//<editor-fold desc="CooperativeKernelLaunchers">

/**
 * @brief Launches a kernel that cooperates between blocks.
 * @tparam Args The types of the arguments for the Kernel.
 * @param grid The 3d grid size that the kernel will be launched with.
 * @param block The 3d block size that the kernel will be launched with.
 * @param shared_memory The amount of shared memory in bytes allocated per block.
 * @param stream The stream to launch the kernel on.
 * @param function The kernel to launch.
 * @param args The arguments to pass to the kernel.
 */
template <typename ... Args>
void LaunchCooperativeKernel(dim3 grid, dim3 block, std::size_t shared_memory, cudaStream_t stream, void(*function)(Args...), detail::identity_t<Args>&& ... args) {
  detail::LaunchCooperativeKernelHelper(grid, block, shared_memory, stream, function, std::forward<Args>(args)...);
}

/**
 * @brief Launches a kernel that cooperates between blocks.
 * @tparam Args The types of the arguments for the Kernel.
 * @param grid The 3d grid size that the kernel will be launched with.
 * @param block The 3d block size that the kernel will be launched with.
 * @param shared_memory The amount of shared memory in bytes allocated per block.
 * @param function The kernel to launch.
 * @param args The arguments to pass to the kernel.
 */
template <typename ... Args>
void LaunchCooperativeKernel(dim3 grid, dim3 block, std::size_t shared_memory, void(*function)(Args...), detail::identity_t<Args>&& ... args) {
  detail::LaunchCooperativeKernelHelper(grid, block, shared_memory, cudaStream_t{0}, function, std::forward<Args>(args)...);
}

/**
 * @brief Launches a kernel that cooperates between blocks.
 * @tparam Args The types of the arguments for the Kernel.
 * @param grid The 3d grid size that the kernel will be launched with.
 * @param block The 3d block size that the kernel will be launched with.
 * @param function The kernel to launch.
 * @param args The arguments to pass to the kernel.
 */
template <typename ... Args>
void LaunchCooperativeKernel(dim3 grid, dim3 block, void(*function)(Args...), detail::identity_t<Args>&& ... args) {
  detail::LaunchCooperativeKernelHelper(grid, block, std::size_t{0}, cudaStream_t{0}, function, std::forward<Args>(args)...);
}
//</editor-fold>

/**
 * @brief A factory style method for launching a cooperative kernel across multiple devices.
 * @tparam Args The types of the arguments of the kernel to be launched.
 */
template <typename ... Args>
class MultiDeviceCoorperativeLauncher {
 public:
  /**
   * @brief Create the launcher with the parameters that must be consistent across streams.
   * @param _grid The 3d grid size that the kerenels will be launched with.
   * @param _block The 3d block size that the kernels will be launched with.
   * @param _shared_memory The amount of shared memory in bytes that the kernels will be launched with.
   * @param _function The kernel that will be launched in each stream.
   * @param _flags The synchronization options.
   */
  explicit MultiDeviceCoorperativeLauncher(dim3 _grid, dim3 _block, std::size_t _shared_memory,
      void(*_function)(Args...), CooperativeMultiDeviceFlags _flags=CooperativeMultiDeviceFlags::BothSync) :
      grid(_grid), block(_block), shared_memory(_shared_memory), function(_function), flags(_flags), arguments() {}

  /**
   * @brief Gets the number of streams with arguments set to be launched.
   * @return The number of streams to be launched.
   */
  [[nodiscard]] auto CooperativeStreams() const {
    return arguments.size();
  }

  /**
   * @brief Attempts to add a new kernel launch to the launch list.
   * @param stream The stream to launch the processing on.
   * @param args The arguments passed to the kernel
   * @return A reference to this so that Construct().AddStream().AddStream().LaunchKernels();
   */
  MultiDeviceCoorperativeLauncher& AddStream(cudaStream_t stream, Args ... args) {
    // TODO: Replace with try emplace in c++17
    if (arguments.find(stream) == arguments.end()) {
      arguments.emplace(stream, std::addressof(args)...);
    }
    return *this;
  }

  /**
   * @brief Get the pointer array to the arguments for the given stream.
   * @param stream The stream the arguments are associated with.
   * @return A modifiable array of pointers to the arguments.
   */
  [[nodiscard]] std::array<void*, sizeof...(Args)>& StreamArgumentPointers(cudaStream_t stream) {
    return arguments.at(stream);
  }

  /**
   * @brief Get the pointer array to the arguments for the given stream.
   * @param stream The stream the arguments are associated with.
   * @return A constant array of pointers to the arguments.
   */
  [[nodiscard]] const std::array<void*, sizeof...(Args)>& StreamArgumentPointers(cudaStream_t stream) const {
    return arguments.at(stream);
  }

  /**
   * @brief Launches the given kernel with the arguments and streams given.
   */
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
  std::unordered_map<cudaStream_t, std::array<void*, sizeof...(Args)>> arguments;
};

/**
 * @brief Adds a CPU function to be executed on a stream.
 * @tparam Args The type of the arguments for the given function.
 * @param stream The stream that the host function should be queued in.
 * @param function A void returning function that runs on the CPU.
 * @param args The arguments needed to launch the host function.
 */
template <typename ... Args>
void LaunchHostFunction(cudaStream_t stream, void(*function)(Args...), detail::identity_t<Args> ... args) {
  std::array<void*, sizeof...(Args)> args_array = {std::addressof(args)...};
  detail::HostFunctionnData<Args...> user_data(function, args_array.data());
  cudaError_t ret = cudaLaunchHostFunc(stream, detail::HostFunctionWrapper<Args...>, user_data);
  if (ret != cudaSuccess) {
    throw CudaException(ret);
  }
  ret = cudaGetLastError();
  if (ret != cudaSuccess) {
    throw CudaException(ret);
  }
}

}

#endif //CUDAPP_KERNEL_LAUNCHERS_H
