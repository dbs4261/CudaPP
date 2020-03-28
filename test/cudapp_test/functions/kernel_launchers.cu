//
// Created by developer on 3/24/20.
//

#include "cudapp/utilities/ide_helpers.h"

#include <device_launch_parameters.h>

#include "gtest/gtest.h"

#include "cudapp/functions/kernel_launchers.h"
#include "cudapp/memory/allocators/managed_allocator.h"

#include "cudapp_test/testing_helpers.h"

template <typename A, typename B>
__global__ void CopyValueKernel(A value, B out, unsigned int n) {
  unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < n) {
    out[n] = value;
  }
}

using test_type = float;
static constexpr unsigned int num = 32;

TEST(KernelLauncher, CopyValue) {
  std::vector<test_type, cudapp::ManagedAllocator<test_type>> out(num, test_type{0});
  double val = 3.14;

  dim3 grid;
  dim3 block;
  auto function = CopyValueKernel<test_type, test_type*>;
  cudapp::LaunchFunction(grid, block, function, val, out.data(), out.size());
}