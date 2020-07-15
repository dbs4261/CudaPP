//
// Created by Daniel Simon on 3/31/20.
//

#ifndef CUDAPP_STREAM_MANAGMENT_H
#define CUDAPP_STREAM_MANAGMENT_H

#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include "cudapp/exceptions/cuda_exception.h"
#include "cudapp/managment/device_managment.h"
#include "cudapp/utilities/capture_wrapper.h"
#include "cudapp/utilities/type_helpers.h"

namespace cudapp {

class Event;
class Stream;

namespace detail {

template <typename ... Args>
void CallbackForwarder(cudaStream_t stream, cudaError_t error, void* user_data) {
  using function_t = void(*)(cudaStream_t, cudaError_t, Args...);
  using data_t = std::tuple<Args...>;
  std::pair<function_t, data_t>* pair = reinterpret_cast<std::pair<function_t, data_t>*>(user_data);
  pair->first(stream, error, std::get<Args>(pair->second)...);
  delete pair;
}

std::vector<Stream> GenerateDefaultStreams() noexcept;

}

class Stream {
 public:
  explicit Stream(const Device& _device) noexcept(false) : Stream(_device, false) {}
  explicit Stream(const Device& _device, bool non_blocking) noexcept(false) : Stream(_device, 0, non_blocking) {}
  explicit Stream(const Device& _device, int priority, bool non_blocking) noexcept(false) : stream(nullptr), device_id(_device.getId()) {
    auto push_pop = detail::MakeScopeBasedDevicePushPop(this->device_id);
    cudaError_t ret = cudaStreamCreateWithPriority(&this->stream,
        non_blocking ? cudaStreamNonBlocking : cudaStreamDefault, priority);
    if (ret != cudaSuccess) {
      throw CudaException(ret);
    }
  }

  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;

  Stream(Stream&& other) noexcept : stream(other.stream), device_id(other.device_id) {
    other.stream = nullptr;
  }
  Stream& operator=(Stream&& other) noexcept {
    if (this != std::addressof(other)) {
      std::swap(this->device_id, other.device_id);
      std::swap(this->stream, other.stream);
    }
    return *this;
  }

  ~Stream() noexcept(false) {
    // Cant destroy default streams so just move on
    if (not (this->stream == nullptr or this->stream == cudaStreamLegacy or this->stream == cudaStreamPerThread)) {
      cudaError_t ret = cudaStreamDestroy(this->stream);
      if (ret != cudaSuccess) {
        throw CudaException(ret);
      }
    }
  }

  operator cudaStream_t() noexcept {
    return this->stream;
  }

  [[nodiscard]] const Device& getDevice() const noexcept(false) {
    return CudaDevices().findById(this->device_id);
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
    auto pushpop = detail::MakeScopeBasedDevicePushPop(this->device_id);
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

  friend std::vector<Stream> detail::GenerateDefaultStreams() noexcept;
  friend bool operator==(const Stream& a, const Stream& b) noexcept {
    return a.device_id == b.device_id and a.stream == b.stream;
  }
  friend bool operator!=(const Stream& a, const Stream& b) noexcept {
    return not (a == b);
  }
  friend cudapp::Event;

 protected:
  explicit Stream(const Device& _device, cudaStream_t _stream) noexcept : stream(_stream), device_id(_device.getId()) {}

  cudaStream_t stream;
  int device_id;
};

[[nodiscard]] std::vector<Stream> detail::GenerateDefaultStreams() noexcept {
  auto& devices = CudaDevices();
  std::vector<Stream> streams; streams.reserve(devices.size());
  std::transform(devices.begin(), devices.end(), std::back_inserter(streams),
      [](const Device& device)->Stream{
    // TODO(Dan.Simon): check if these streams are the proper way of generating default streams.
    #ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
    return Stream(device, cudaStreamPerThread);
    #else
    return Stream(device, cudaStreamLegacy);
    #endif
  });
  return streams;
}

[[nodiscard]] std::vector<Stream>& DefaultStreams() noexcept {
  static thread_local std::vector<Stream> streams = detail::GenerateDefaultStreams();
  return streams;
}

[[nodiscard]] Stream& DefaultStream() noexcept(false) {
  std::vector<Stream>& streams = DefaultStreams();
  int device_id;
  cudaError_t ret = cudaGetDevice(&device_id);
  if (ret != cudaSuccess) {
    throw CudaException(ret);
  }
  auto it = std::find_if(streams.begin(), streams.end(),
      [device_id](const Stream& stream)->bool{
    return stream.getDevice().getId() == device_id;
  });
  if (it == streams.end()) {
    throw std::out_of_range("Current active device is not a valid cuda device");
  }
  return *it;
}

}

#endif //CUDAPP_STREAM_MANAGMENT_H
