//
// Created by Daniel Simon on 10/4/19.
//

#ifndef CUDAPP_SURFACE_VIEW_H
#define CUDAPP_SURFACE_VIEW_H

#include "cuda_ide_helpers.h"

#include <type_traits>

#include <surface_types.h>
#include <surface_indirect_functions.h>

namespace cuda {

namespace detail {

template <typename T, std::size_t Dimensions>
class SurfaceView {
  static_assert(sizeof(T) != 0 and Dimensions > 0 and Dimensions <= 3, "Surface must have 1, 2 or 3 dimensions");
 public:
  using value_type = T;
  static constexpr std::size_t Dims = Dimensions;
  __host__ __device__ SurfaceView() noexcept : surface(0), mode(cudaSurfaceBoundaryMode::cudaBoundaryModeTrap) {}
  __host__ __device__ explicit SurfaceView(cudaSurfaceObject_t _surface) noexcept
      : surface(_surface), mode(cudaSurfaceBoundaryMode::cudaBoundaryModeTrap) {}
  __host__ __device__ explicit SurfaceView(cudaSurfaceObject_t _surface, cudaSurfaceBoundaryMode _mode) noexcept
      : surface(_surface), mode(_mode) {}
  __host__ __device__ explicit SurfaceView(const SurfaceView<T, Dims>& _surf) noexcept
      : surface(_surf.surface), mode(_surf.mode) {}
  __host__ __device__ explicit SurfaceView(SurfaceView<T, Dims>&& _surf) noexcept
      : surface(std::move(_surf.surface)), mode(std::move(_surf.mode)) {}

  __host__ __device__ __forceinline__ SurfaceView& operator=(const SurfaceView<T, Dims>& other) noexcept {
    this->surface = other.surface;
    this->mode = other.mode;
  }
  __host__ __device__ __forceinline__ SurfaceView& operator=(SurfaceView<T, Dims>&& other) noexcept {
    this->surface = std::move(other.surface);
    this->mode = std::move(other.mode);
  }

  __host__ __device__ __forceinline__ operator SurfaceView<typename std::add_const<T>::type, Dimensions>() const noexcept { // NOLINT(hicpp-explicit-conversions,google-explicit-constructor)
    return SurfaceView<std::add_const_t<T>, Dimensions>(this->surface, this->mode);
  };

  __host__ __device__ __forceinline__ void Mode(cudaSurfaceBoundaryMode _mode) noexcept {
    this->mode = _mode;
  }

 protected:
  cudaSurfaceObject_t surface;
  cudaSurfaceBoundaryMode mode;
};

}

template <typename T, std::size_t Dimensions>
class SurfaceView : public detail::SurfaceView<T, 1> {
  static_assert(sizeof(T) != 0 and Dimensions > 0 and Dimensions <= 3, "Surface must have 1, 2 or 3 dimensions");
};

template <typename T>
class SurfaceView<T, 1> : public detail::SurfaceView<T, 1> {
 public:
  __host__ __device__ explicit SurfaceView(cudaSurfaceObject_t _surface) : detail::SurfaceView<T, 1>(_surface) {}
  __host__ __device__ explicit SurfaceView(cudaSurfaceObject_t _surface, cudaSurfaceBoundaryMode _mode)
      : detail::SurfaceView<T, 1>(_surface, _mode) {}

  __device__ __forceinline__ T Read(int x) const {
    #ifdef __CUDA_ARCH__
    return surf1Dread<T>(this->surface, sizeof(T) * x, this->mode);
    #endif
  }

  __device__ __forceinline__ void Write(int x, typename std::enable_if<not std::is_const<T>::value, T>::type data) {
    static_assert(not std::is_const<T>::value, "Write cannot be called for a const surface");
    #ifdef __CUDA_ARCH__
    surf1Dwrite<T>(data, this->surface, sizeof(T) * x, this->mode);
    #endif
  }
};

template <typename T>
class SurfaceView<T, 2> : public detail::SurfaceView<T, 2> {
 public:
  __host__ __device__ explicit SurfaceView(cudaSurfaceObject_t _surface) : detail::SurfaceView<T, 2>(_surface) {}
  __host__ __device__ explicit SurfaceView(cudaSurfaceObject_t _surface, cudaSurfaceBoundaryMode _mode)
      : detail::SurfaceView<T, 2>(_surface, _mode) {}

  __device__ __forceinline__ T Read(int x, int y) const {
    #ifdef __CUDA_ARCH__
    return surf2Dread<T>(this->surface, sizeof(T) * x, y, this->mode);
    #endif
  }

  __device__ __forceinline__ void Write(int x, int y, typename std::enable_if<not std::is_const<T>::value, T>::type data) {
    static_assert(not std::is_const<T>::value, "Write cannot be called for a const surface");
    #ifdef __CUDA_ARCH__
    surf2Dwrite<T>(data, this->surface, sizeof(T) * x, y, this->mode);
    #endif
  }
};

template <typename T>
class SurfaceView<T, 3> : public detail::SurfaceView<T, 3> {
 public:
  __host__ __device__ explicit SurfaceView(cudaSurfaceObject_t _surface) : detail::SurfaceView<T, 3>(_surface) {}
  __host__ __device__ explicit SurfaceView(cudaSurfaceObject_t _surface, cudaSurfaceBoundaryMode _mode)
      : detail::SurfaceView<T, 3>(_surface, _mode) {}

  __device__ __forceinline__ T Read(int x, int y, int z) const {
    #ifdef __CUDA_ARCH__
    return surf3Dread<T>(this->surface, sizeof(T) * x, y, z, this->mode);
    #endif
  }

  __device__ __forceinline__ void Write(int x, int y, int z, typename std::enable_if<not std::is_const<T>::value, T>::type data) {
    static_assert(not std::is_const<T>::value, "Write cannot be called for a const surface");
    #ifdef __CUDA_ARCH__
    surf3Dwrite<T>(data, this->surface, sizeof(T) * x, int y, int z, this->mode);
    #endif
  }
};

}

#endif //CUDAPP_SURFACE_VIEW_H
