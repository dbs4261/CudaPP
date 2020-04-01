//
// Created by Daniel Simon on 3/23/20.
//

#include "cudapp/utilities/ide_helpers.h"

#include <device_launch_parameters.h>

#include "gtest/gtest.h"

#include "cudapp/memory/allocators/device_allocator.h"

#include "cudapp_test/testing_helpers.h"

template <typename T>
__global__ void SaxpyKernel(T a, const T* x, const T* y, T* f, unsigned int n) {
  unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < n) {
    f[idx] = a * x[idx] + y[idx];
  }
}

template <typename T, typename Allocator>
std::vector<T, Allocator> SaxpyHost(T a, const std::vector<T, Allocator>& arr_x, const std::vector<T, Allocator>& arr_y) {
  assert(arr_x.size() == arr_y.size());
  std::vector<T, Allocator> out(arr_x.size());
  std::transform(arr_x.begin(), arr_x.end(), arr_y.begin(), out.begin(),
    [a](const auto& x, const auto& y)->float{
      return a * x + y;
  });
  return out;
}

TEST(DeviceAllocator, SaxpyRaw) {
  unsigned int num = 32;
  float minimum = 0.0f;
  float maximum = 1024.0f;
  float a = cudapp::test::CreateUniformRandom(minimum, maximum);
  std::vector<float> host_x = cudapp::test::GenerateUniformlyRandom(num, minimum, maximum);
  std::vector<float> host_y = cudapp::test::GenerateUniformlyRandom(num, minimum, maximum);
  float *device_x = cudapp::DeviceAllocator<float>().allocate(num);
  float *device_y = cudapp::DeviceAllocator<float>().allocate(num);
  float *device_f = cudapp::DeviceAllocator<float>().allocate(num);
  
  EXPECT_EQ(cudaMemcpy(device_x, host_x.data(), host_x.size() * sizeof(decltype(host_x)::value_type), cudaMemcpyHostToDevice), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(device_y, host_y.data(), host_y.size() * sizeof(decltype(host_y)::value_type), cudaMemcpyHostToDevice), cudaSuccess);

  dim3 block_size = dim3{32u, 1u, 1u};
  assert(num > 0);
  dim3 grid_size((num - 1u) / block_size.x + 1u, 1u, 1u);
  SaxpyKernel<<<grid_size, block_size>>>(a, device_x, device_y, device_f, num);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  cudapp::DeviceAllocator<float>().deallocate(device_x, num);
  cudapp::DeviceAllocator<float>().deallocate(device_y, num);

  std::vector<float> host_f(num);
  EXPECT_EQ(cudaMemcpy(host_f.data(), device_f, host_f.size() * sizeof(decltype(host_f)::value_type), cudaMemcpyDeviceToHost), cudaSuccess);
  cudapp::DeviceAllocator<float>().deallocate(device_f, num);

  std::vector<float> expected_f = SaxpyHost(a, host_x, host_y);

  for (unsigned int i = 0; i < num; i++) {
    EXPECT_FLOAT_EQ(expected_f.at(i), host_f.at(i)) << "Test failed at index: " << i;
  }
}
