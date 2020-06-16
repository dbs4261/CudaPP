//
// Created by Daniel Simon on 4/23/20.
//

#ifndef CUDAPP_EVENT_MANAGMENT_H
#define CUDAPP_EVENT_MANAGMENT_H

#include <chrono>

#include <cuda_runtime_api.h>

#include "cudapp/exceptions/cuda_exception.h"
#include "cudapp/utilities/macros.h"

#include "device_managment.h"
#include "stream_managment.h"

namespace cudapp {

class Event {
 public:
  Event(const Device& device) noexcept(false) : Event(device, false, true) {}
  Event(const Device& device, bool blocking, bool timeable) noexcept(false)
    : event(nullptr), device_id(device.getId()) {
    int flags = (blocking ? cudaEventBlockingSync : 0) | (timeable ? 0: cudaEventDisableTiming); // NOLINT(hicpp-signed-bitwise)
    cudaError_t ret = cudaEventCreateWithFlags(&this->event, flags);
    if (ret != cudaSuccess) {
      throw CudaException(ret);
    }
  }
  ~Event() noexcept(false) {
    if (this->event != nullptr) {
      CudaCatchError(cudaEventDestroy(this->event));
    }
  }

  bool Query() noexcept(false) {
    // TODO(Daniel.Simon): Is this device aware?
    cudaError_t ret = cudaEventQuery(this->event);
    if (ret == cudaSuccess) {
      return true;
    } else if (ret == cudaErrorNotReady) {
      return false;
    } else {
      throw CudaException(ret);
    }
  }

  void Record(Stream& stream) noexcept(false) { // Maybe reverse this and put this functionality in stream
    // Tell the stream to record an event when its execution reaches this point.
    // Can be called multiple times, but only most recent call is recorded.
    if (stream.device_id != this->device_id) {
      throw CudaException(cudaErrorInvalidDevice);
    }
    cudaError_t ret = cudaEventRecord(this->event, stream.stream);
    if (ret != cudaSuccess) {
      throw CudaException(ret);
    }
  }

  void Synchronize() const noexcept(false) {
    // Not device aware
    // Without the blocking sync flag this will spin wait until stream reaches this event.
    cudaError_t ret = cudaEventSynchronize(this->event);
    if (ret != cudaSuccess) {
      throw CudaException(ret);
    }
  }

  float ElapsedTime(const Event& start) const noexcept(false) {
    float elapsed;
    cudaError_t ret = cudaEventElapsedTime(&elapsed, start.event, this->event);
    if (ret != cudaSuccess) {
      throw CudaException(ret);
    }
    return elapsed;
  }

 protected:
  cudaEvent_t event;
  int device_id;
};

}

#endif //CUDAPP_EVENT_MANAGMENT_H
