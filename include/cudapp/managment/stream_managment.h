//
// Created by Daniel Simon on 3/31/20.
//

#ifndef CUDAPP_STREAM_MANAGMENT_H
#define CUDAPP_STREAM_MANAGMENT_H

#include "cudapp/utilities/ide_helpers.h"

#include "cudapp/exceptions/cuda_exception.h"
#include "cudapp/managment/device_managment.h"
#include "cudapp/utilities/capture_wrapper.h"
#include "cudapp/utilities/type_helpers.h"

namespace cudapp {

namespace detail {

template <typename ... Args>
void CallbackForwarder(cudaStream_t stream, cudaError_t error, void* user_data) {
  using function_t = void(*)(cudaStream_t, cudaError_t, Args...);
  using data_t = std::tuple<Args...>;
  std::pair<function_t, data_t>* pair = reinterpret_cast<std::pair<function_t, data_t>>(user_data);
  pair->first(stream, error, std::get<Args>(pair->second)...);
  delete pair;
}

}

class Stream {
 public:
  Stream() noexcept(false) : Stream(CurrentCudaDevice()) {}
  explicit Stream(const Device& _device) noexcept(false) : Stream(device, false) {}
  explicit Stream(const Device& _device, bool non_blocking) noexcept(false) : Stream(device, 0, non_blocking) {}
  explicit Stream(const Device& _device, int priority, bool non_blocking) noexcept(false) : stream(nullptr), device(_device) {
    auto push_pop = detail::MakeScopeBasedDevicePushPop(_device);
    cudaError_t ret = cudaStreamCreateWithPriority(&this->stream,
        non_blocking ? cudaStreamNonBlocking : cudaStreamDefault, priority);
    if (ret != cudaSuccess) {
      throw CudaException(ret);
    }
  }


  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;

  Stream(Stream&& other) noexcept : stream(other.stream), device(other.device) {
    other.stream = nullptr;
  }
  Stream& operator=(Stream&& other) noexcept(false) {
    if (this->device != other.device) {
      throw CudaException(cudaErrorInvalidDevice);
    }
    std::swap(this->stream, other.stream);
  }

  ~Stream() {
    cudaError_t ret = cudaStreamDestroy(this->stream);
    if (ret != cudaSuccess) {
      #pragma clang diagnostic push
      #pragma clang diagnostic ignored "-Wexceptions"
      throw CudaException(ret);
      #pragma clang diagnostic pop
    }
  }

  [[nodiscard]] const Device& getDevice() const noexcept {
    return device;
  }

  [[nodiscard]] int getPriority() const noexcept(false) {
    int priority;
    cudaError_t ret = cudaStreamGetPriority(this->stream, &priority);
    if (ret != cudaSuccess) {
      throw CudaException(ret);
    }
    return priority;
  }

  [[nodiscard]] bool isNonBlocking() const noexcept(false) {
    unsigned int flags;
    cudaError_t ret = cudaStreamGetFlags(this->stream, &flags);
    if (ret != cudaSuccess) {
      throw CudaException(ret);
    }
    return flags == cudaStreamNonBlocking;
  }

  template <typename ... Args>
  void AddCallback(void(*function)(cudaStream_t, cudaError_t, Args...), identity_t<Args>&& ... args) noexcept(false) {
    auto pushpop = detail::MakeScopeBasedDevicePushPop(this->device);
    using data_t = std::tuple<Args...>;
    using payload_t = std::pair<decltype(function), data_t>;
    void* user_data = new payload_t(function, data_t(capture(args)...));
    // The 4th flags parameter, as of cuda 10.2, is unused.
    cudaError_t ret = cudaStreamAddCallback(this->stream, detail::CallbackForwarder<Args...>, user_data, 0);
    if (ret != cudaSuccess) {
      throw CudaException(ret);
    }
  }

  [[nodiscard]] bool Completed() const noexcept(false) {
    cudaError_t ret = cudaStreamQuery(this->stream);
    if (ret == cudaSuccess) {
      return true;
    } else if (ret == cudaErrorNotReady) {
      return false;
    } else {
      throw CudaException(ret);
    }
  }

  void Synchronize() const noexcept(false) {
    cudaError_t ret = cudaStreamSynchronize(this->stream);
    if (ret != cudaSuccess) {
      throw CudaException(ret);
    }
  }

  // TODO(Dan.Simon): AttachMemAsync
  // TODO(Dan.Simon): WaitEvent

 protected:
  cudaStream_t stream;
  const Device& device;
};

}

#endif //CUDAPP_STREAM_MANAGMENT_H