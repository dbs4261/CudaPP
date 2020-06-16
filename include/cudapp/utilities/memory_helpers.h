//
// Created by Daniel Simon on 8/16/19.
//

#ifndef CUDAPP_CUDA_HELPERS_H
#define CUDAPP_CUDA_HELPERS_H

#include <algorithm>
#include <cassert>

#include <driver_functions.h>

namespace cudapp {

cudaMemcpy3DParms BlankMemcpy3DParams() {
  cudaMemcpy3DParms params;
  params.srcArray = nullptr;
  params.srcPos.x = 0;
  params.srcPos.y = 0;
  params.srcPos.z = 0;
  params.srcPtr.ptr = nullptr;
  params.srcPtr.pitch = 0;
  params.srcPtr.xsize = 0;
  params.srcPtr.ysize = 0;
  params.dstArray = nullptr;
  params.dstPos.x = 0;
  params.dstPos.y = 0;
  params.dstPos.z = 0;
  params.dstPtr.ptr = nullptr;
  params.dstPtr.pitch = 0;
  params.dstPtr.xsize = 0;
  params.dstPtr.ysize = 0;
  params.extent.width = 0;
  params.extent.height = 0;
  params.extent.depth = 0;
  params.kind = cudaMemcpyKind::cudaMemcpyDefault;
  return params;
}

template <typename T>
cudaMemcpy3DParms Memcpy3DParamsHD(const T* ptr, cudaArray* array, cudaExtent extent) {
  cudaMemcpy3DParms params = BlankMemcpy3DParams();
  params.kind = cudaMemcpyKind::cudaMemcpyHostToDevice;
  params.extent.width = extent.width;
  params.extent.height = std::max<std::size_t>(1, extent.height);
  params.extent.depth = std::max<std::size_t>(1, extent.depth);
  assert(params.extent.width * params.extent.height * params.extent.depth > 0);
  params.srcPtr.ptr = const_cast<T*>(ptr); // Casting away const because cuda isnt const correct :(
  params.srcPtr.pitch = extent.width * sizeof(T);
  params.srcPtr.xsize = extent.width;
  params.srcPtr.ysize = std::max<std::size_t>(1, extent.height);
  params.dstArray = array;
  return params;
}

template <typename T>
cudaMemcpy3DParms Memcpy3DParamsHD(const T* ptr, cudaArray* array, std::size_t w, std::size_t h, std::size_t d) {
  return Memcpy3DParamsHD<T>(ptr, array, make_cudaExtent(w, h, d));
}

template <typename T>
cudaMemcpy3DParms Memcpy3DParamsDH(cudaArray* array, T* ptr, cudaExtent extent) {
  cudaMemcpy3DParms params = BlankMemcpy3DParams();
  params.kind = cudaMemcpyKind::cudaMemcpyDeviceToHost;
  params.extent.width = extent.width;
  params.extent.height = std::max<std::size_t>(1, extent.height);
  params.extent.depth = std::max<std::size_t>(1, extent.depth);
  assert(params.extent.width * params.extent.height * params.extent.depth > 0);
  params.srcArray = array;
  params.dstPtr.ptr = ptr;
  params.dstPtr.pitch = extent.width * sizeof(T);
  params.dstPtr.xsize = extent.width;
  params.dstPtr.ysize = std::max<std::size_t>(1, extent.height);
  return params;
}

template <typename T>
cudaMemcpy3DParms Memcpy3DParamsDH(cudaArray* array, T* ptr, std::size_t w, std::size_t h, std::size_t d) {
  return Memcpy3DParamsDH<T>(array, ptr, make_cudaExtent(w, h, d));
}

cudaMemcpy3DParms Memcpy3DParamsDD(cudaArray* in, cudaArray* out, cudaExtent extent) {
  cudaMemcpy3DParms params = BlankMemcpy3DParams();
  params.kind = cudaMemcpyKind::cudaMemcpyDeviceToDevice;
  params.extent.width = extent.width;
  params.extent.height = std::max<std::size_t>(1, extent.height);
  params.extent.depth = std::max<std::size_t>(1, extent.depth);
  assert(params.extent.width * params.extent.height * params.extent.depth > 0);
  params.srcArray = in;
  params.dstArray = out;
  return params;
}

cudaMemcpy3DParms Memcpy3DParamsDD(cudaArray* in, cudaArray* out, std::size_t w, std::size_t h, std::size_t d) {
  return Memcpy3DParamsDD(in, out, make_cudaExtent(w, h, d));
}

cudaTextureDesc BlankTextureDesc() {
  cudaTextureDesc desc;
  desc.addressMode[0] = cudaTextureAddressMode::cudaAddressModeClamp;
  desc.addressMode[1] = cudaTextureAddressMode::cudaAddressModeClamp;
  desc.addressMode[2] = cudaTextureAddressMode::cudaAddressModeClamp;
  desc.filterMode = cudaTextureFilterMode::cudaFilterModePoint;
  desc.readMode = cudaTextureReadMode::cudaReadModeElementType;
  desc.sRGB = static_cast<int>(false);
  desc.borderColor[0] = 0.0f;
  desc.borderColor[1] = 0.0f;
  desc.borderColor[2] = 0.0f;
  desc.borderColor[3] = 0.0f;
  desc.normalizedCoords = static_cast<int>(false);
  desc.maxAnisotropy = 0;
  desc.mipmapFilterMode = cudaTextureFilterMode::cudaFilterModePoint;
  desc.mipmapLevelBias = 0.0f;
  desc.minMipmapLevelClamp = 0.0f;
  desc.maxMipmapLevelClamp = 0.0f;
  return desc;
}

cudaResourceDesc BlankResourceDesc() {
  cudaResourceDesc desc;
  desc.resType = cudaResourceType::cudaResourceTypeArray;
  desc.res.array.array = nullptr;
  return desc;
}

}

#endif //CUDAPP_CUDA_HELPERS_H
