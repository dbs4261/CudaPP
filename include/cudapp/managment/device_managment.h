//
// Created by Daniel Simon on 3/31/20.
//

#ifndef CUDAPP_DEVICE_MANAGMENT_H
#define CUDAPP_DEVICE_MANAGMENT_H

#include <algorithm>
#include <array>
#include <vector>

#include <cuda_runtime_api.h>

#include "cudapp/exceptions/cuda_exception.h"
#include "cudapp/utilities/macros.h"

namespace cudapp {

class Devices;

class Device {
 public:
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  Device(Device&&) = default;
  Device& operator=(Device&&) = default;

  [[nodiscard]] int getId() const noexcept {
    return id;
  }

  [[nodiscard]] std::string PCIBusId() const noexcept(false) {
    std::array<char, 512> pci_id{};
    cudaError_t ret = cudaDeviceGetPCIBusId(pci_id.data(), pci_id.size() - 1, id);
    if (ret != cudaSuccess) {
      throw CudaException(ret);
    }
    auto end = pci_id.cbegin();
    while (*end++ != 0);
    auto len = std::min<std::ptrdiff_t>(std::distance(pci_id.cbegin(), end) - 1, pci_id.size());
    return std::string(pci_id.data(), len);
  }

  [[nodiscard]] bool isActive() const noexcept(false) {
    int active_id = -1;
    cudaError_t ret = cudaGetDevice(&active_id);
    if (ret != cudaSuccess) {
      throw CudaException(ret);
    }
    return this->id == active_id;
  }

  void setActive() const noexcept(false) {
    cudaError_t ret = cudaSetDevice(id);
    if (ret != cudaSuccess) {
      throw CudaException(ret);
    }
  }

  void BlockOnAllStreams() const noexcept(false) {
    cudaError_t ret = cudaDeviceSynchronize();
    if (ret != cudaSuccess) {
      throw CudaException(ret);
    }
  }

  void Reset() const noexcept(false) {
    cudaError_t ret = cudaDeviceReset();
    if (ret != cudaSuccess) {
      throw CudaException(ret);
    }
  }

  friend Devices;
  friend bool operator==(const Device& a, const Device& b) noexcept {
    return a.id == b.id;
  }
  friend bool operator!=(const Device& a, const Device& b) {
    return not (a == b);
  }

 protected:
  explicit Device(int _id) noexcept(false)
      : id(_id), lowest_stream_priority(-1), highest_stream_priority(-1) {
    cudaError_t ret = cudaDeviceGetStreamPriorityRange(&this->lowest_stream_priority, &this->highest_stream_priority);
    if (ret != cudaSuccess) {
      throw CudaException(ret);
    }
  }

  explicit Device(const std::string& pci_bus_id) noexcept(false)
      : id(-1), lowest_stream_priority(-1), highest_stream_priority(-1) {
    cudaError_t ret = cudaDeviceGetByPCIBusId(&id, pci_bus_id.c_str());
    if (ret != cudaSuccess) {
      throw CudaException(ret);
    }
    ret = cudaDeviceGetStreamPriorityRange(&this->lowest_stream_priority, &this->highest_stream_priority);
    if (ret != cudaSuccess) {
      throw CudaException(ret);
    }
  }

  int id;
  int lowest_stream_priority;
  int highest_stream_priority;
};

// Singleton
class Devices {
 public:
  Devices(const Devices&) = delete;
  Devices& operator=(const Devices&) = delete;

  Devices(Devices&&) = default;
  Devices& operator=(Devices&&) = default;

  [[nodiscard]] std::size_t size() const noexcept {
    return devices.size();
  }

  [[nodiscard]] const Device& at(std::size_t i) const noexcept(false) {
    return devices.at(i);
  }

  [[nodiscard]] const Device& findById(int id) const noexcept(false) {
    auto it = std::find_if(devices.begin(), devices.end(),
        [id](const Device& device)->bool{
      return device.id == id;
    });
    if (it != devices.end()) {
      return *it;
    } else {
      throw std::runtime_error("Device id: " + std::to_string(id) + " not found.");
    }
  }

  [[nodiscard]] const Device& findByPCIBusId(const std::string& bus_id) const noexcept(false) {
    auto it = std::find_if(devices.begin(), devices.end(),
        [bus_id](const Device& device)->bool{
      return device.PCIBusId() == bus_id;
    });
    if (it != devices.end()) {
      return *it;
    } else {
      throw std::runtime_error("Device with PCI Bus ID: " + bus_id + " not found.");
    }
  }

  template <typename Func>
  Devices& setValidDevices(Func function) noexcept(false) {
    devices.erase(std::remove_if(devices.begin(), devices.end(), function), devices.end());
    if (devices.empty()) {
      throw std::runtime_error("No cuda capable devices found!");
    }
    return *this;
  }

  Devices& resetValidDevices() noexcept(false) {
    devices.clear();
    int size;
    cudaError_t ret = cudaGetDeviceCount(&size);
    if (ret != cudaSuccess) {
      throw CudaException(ret);
    }
    devices.reserve(size);
    for (int i = 0; i < size; i++) {
      devices.emplace_back(Device(i));
    }
    if (devices.empty()) {
      throw std::runtime_error("No cuda capable devices found!");
    }
    return *this;
  }

  [[nodiscard]] std::vector<Device>::const_iterator begin() const noexcept {
    return devices.begin();
  }

  [[nodiscard]] std::vector<Device>::const_iterator end() const noexcept {
    return devices.end();
  }

  friend Devices& CudaDevices();

 protected:
  Devices() noexcept(false) : devices() {
    this->resetValidDevices();
  };

  ~Devices() = default;

  std::vector<Device> devices;
};

/**
 * @brief The only way to access the cuda devices.
 * @return The container of all devices, this is mutable and static.
 */
[[nodiscard]] Devices& CudaDevices() noexcept(false) {
  static Devices devices;
  return devices;
}

/**
 * @brief Access the current device.
 * @return A reference to the current active device.
 */
[[nodiscard]] const Device& CurrentCudaDevice() noexcept(false) {
  int device_id;
  cudaError_t ret = cudaGetDevice(&device_id);
  if (ret != cudaSuccess) {
    throw CudaException(ret);
  }
  return CudaDevices().findById(device_id);
}

namespace detail {

struct ScopeBasedDevicePushPop {
 protected:
  explicit ScopeBasedDevicePushPop(const Device& device) noexcept(false): previous(-1) {
    cudaError_t ret = cudaGetDevice(&this->previous);
    if (ret != cudaSuccess) {
      throw CudaException(ret);
    }
    if (device.getId() == previous) {
      this->previous = -1;
    } else {
      ret = cudaSetDevice(device.getId());
      throw CudaException(ret);
    }
  }
  explicit ScopeBasedDevicePushPop(int device_id) noexcept(false): previous(-1) {
    cudaError_t ret = cudaGetDevice(&this->previous);
    if (ret != cudaSuccess) {
      throw CudaException(ret);
    }
    if (device_id == previous) {
      this->previous = -1;
    } else {
      ret = cudaSetDevice(device_id);
      throw CudaException(ret);
    }
  }
  friend ScopeBasedDevicePushPop MakeScopeBasedDevicePushPop(const Device& device);
  friend ScopeBasedDevicePushPop MakeScopeBasedDevicePushPop(int device_id);

 public:
  ScopeBasedDevicePushPop(const ScopeBasedDevicePushPop&) = delete;
  ScopeBasedDevicePushPop& operator=(const ScopeBasedDevicePushPop&) = delete;

  ScopeBasedDevicePushPop(ScopeBasedDevicePushPop&& other) noexcept {
    this->previous = other.previous;
    other.previous = -1;
  }
  ScopeBasedDevicePushPop& operator=(ScopeBasedDevicePushPop&& other) noexcept {
    this->previous = other.previous;
    other.previous = -1;
    return *this;
  }

  ~ScopeBasedDevicePushPop() {
    if (this->previous >= 0) {
      CudaCatchError(cudaSetDevice(previous));
    }
  }

  int previous;
};

[[nodiscard]] inline ScopeBasedDevicePushPop MakeScopeBasedDevicePushPop(const Device& device) noexcept(false) {
  return ScopeBasedDevicePushPop(device);
}

[[nodiscard]] inline ScopeBasedDevicePushPop MakeScopeBasedDevicePushPop(int device_id) noexcept(false) {
  return ScopeBasedDevicePushPop(device_id);
}

}

namespace DeviceFilters {

struct ComputeCapability {
  ComputeCapability(int major, int minor) noexcept: required_major(major), required_minor(minor) {}

  bool operator()(const Device& device) noexcept(false) {
    int major, minor;
    cudaDeviceGetAttribute(&major, cudaDeviceAttr::cudaDevAttrComputeCapabilityMajor, device.getId());
    cudaDeviceGetAttribute(&minor, cudaDeviceAttr::cudaDevAttrComputeCapabilityMajor, device.getId());
    if (major > required_major) {
      return true;
    } else if (major == required_major) {
      return minor >= required_minor;
    } else {
      return false;
    }
  }

  int required_major;
  int required_minor;
};

} // DeviceFilters

}

#endif //CUDAPP_DEVICE_MANAGMENT_H
