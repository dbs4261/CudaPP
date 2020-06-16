//
// Created by Daniel Simon on 3/23/20.
//

#include "gtest/gtest.h"

#include "cudapp/memory/allocators/managed_allocator.h"

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

TEST(ManagedAllocator, SaxpyVector) {
  unsigned int num = 32;
  float minimum = 0.0f;
  float maximum = 1024.0f;
  float a = cudapp::test::CreateUniformRandom(minimum, maximum);
  auto managed_x = cudapp::test::GenerateUniformlyRandom<float, cudapp::ManagedAllocator<float>>(num, minimum, maximum);
  auto managed_y = cudapp::test::GenerateUniformlyRandom<float, cudapp::ManagedAllocator<float>>(num, minimum, maximum);
  std::vector<float, cudapp::ManagedAllocator<float>> managed_f(num);

  dim3 block_size = dim3{32u, 1u, 1u};
  assert(num > 0);
  dim3 grid_size((num - 1) / block_size.x + 1, 1u, 1u);
  SaxpyKernel<<<grid_size, block_size>>>(a, managed_x.data(), managed_y.data(), managed_f.data(), num);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  auto expected_f = SaxpyHost(a, managed_x, managed_y);

  for (unsigned int i = 0; i < num; i++) {
    EXPECT_FLOAT_EQ(expected_f.at(i), managed_f.at(i)) << "Test failed at index: " << i;
  }
}