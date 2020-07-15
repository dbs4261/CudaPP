#include <cassert>
#include <cstdio>
#include <numeric>

#include <cuda_runtime_api.h>

#include "cudapp/utilities/special_registers.cuh"

__host__ void check(cudaError_t result, char const* const func, const char* const file, int const line) {
  if (result) {
    printf("CUDA error at %s:%d code=%d(%s) \"%s\" \n",
        file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
    cudaDeviceReset();
    // Make sure we call CUDA Device Reset before exiting
    exit(EXIT_FAILURE);
  }
}

#ifndef CudaCatchError
#define CudaCatchError(val) check((val), #val, __FILE__, __LINE__)
#endif

__device__ __constant__ float** scratch_memory;

template <int N>
__device__ __noinline__ float* Func(unsigned int rt_size) {
  static constexpr unsigned int np1 = N + 1;
  const int sizeof_scratch = np1 * rt_size;
  const int thread_location = cudapp::LaneId() + 32 * cudapp::WarpId();
  const int address_shift = sizeof_scratch * thread_location;
  float* scratch_start = scratch_memory[cudapp::MultiprocessorId()];
  return scratch_start + address_shift;
}

template <unsigned int N>
__device__ __noinline__ float* FuncAsm(unsigned int rt_size) {
  static constexpr unsigned int np1 = N + 1;
  float* matrix_start;
  asm volatile ("{\n"
                ".reg .u32 t1;"
                ".reg .u32 t2;"
                "mov.u32 t1, %%laneid;"
                "mov.u32 t2, %%warpid;"
                ".reg .u32 t3;"
                "shl.b32 t3, t2, 5;" // t3 = warpid << 5 = warpid * 32
                "add.u32 t2, t3, t1;" // t2 = warpid * 32 + laneid
                "mul.lo.u32 t1, %1, %2;" // t1 = (N + 1) * rt_size
                ".reg .u64 t4;"
                "mul.wide.u32 t4, t2, t1;" // t4 = (warpid * 32 + laneid) * ((N + 1) * rt_size) = address shift
                ".reg .u64 t5;"
                "ld.const.u64 t5, [%3];" // t5 = scratch_memory in generic space
                ".reg .u64 t6;"
                "cvta.to.global.u64 t6, t5;" // t6 = scratch_memory in global space
                "mov.u32 t1, %%smid;" // t1 = smid
                "mul.wide.u32 t5, t1, 8;" // t5 = smid * sizeof(float*)
                ".reg .u64 t7;"
                "add.u64 t7, t5, t6;" // t7 = (void*)(scratch_memory) + smid * sizeof(float*)
                "ld.global.u64 t5, [t7];" // t5 = *(void*(scratch_memory) + smid * sizeof(float*))
                "mul.lo.u64 t6, t4, 4;" // t6 = (warpid * 32 + laneid) * ((N + 1) * rt_size) * sizeof(float)
                "add.u64 %0, t5, t6;"
                "}"
                : "=l"(matrix_start)
                : "n"(np1), "r"(rt_size), "l"(scratch_memory));
  return matrix_start;
}

template <unsigned int constexpr_val>
__global__ void Kernel(char* valid, unsigned int runtime_val) {
    valid[threadIdx.x + blockDim.x * blockIdx.x] = Func<constexpr_val>(runtime_val) == Func<constexpr_val>(runtime_val);
}

template <unsigned int max_constexpr_val>
__host__ void AllocateScratchArray(int num_multiprocessors, int num_threads, unsigned int max_runtime_val) {
  float** scratchs_array_host = new float*[num_multiprocessors];
  float** scratchs_array_device = nullptr;
  CudaCatchError(cudaMalloc(&scratchs_array_device, sizeof(float*) * num_multiprocessors));
  std::size_t bytes_per_sm = sizeof(float) * num_threads * (max_constexpr_val + 1) * max_runtime_val;
  printf("Allocating %lu bytes per multiprocessor [%d] for a total of %lu bytes\n",
         bytes_per_sm, num_multiprocessors, (bytes_per_sm * num_multiprocessors));
  for (int i = 0; i < num_multiprocessors; i++) {
    CudaCatchError(cudaMalloc(&scratchs_array_host[i], bytes_per_sm));
  }
  CudaCatchError(cudaMemcpy(scratchs_array_device, scratchs_array_host, sizeof(float*) * num_multiprocessors, cudaMemcpyHostToDevice));
  CudaCatchError(cudaMemcpyToSymbol<float**>(scratch_memory, &scratchs_array_device, sizeof(float**)));
  delete[] scratchs_array_host;
}

void CleanupScratchArray(int num_multiprocessors) {
  float** scratchs_array_host = new float*[num_multiprocessors];
  float** scratchs_array_device = nullptr;
  CudaCatchError(cudaMemcpyFromSymbol(&scratchs_array_device, scratch_memory, sizeof(float**)));
  CudaCatchError(cudaMemcpy(scratchs_array_host, scratchs_array_device, sizeof(float*) * num_multiprocessors, cudaMemcpyDeviceToHost));
  for (int i = 0; i < num_multiprocessors; i++) {
    CudaCatchError(cudaFree(scratchs_array_host[i]));
  }
  CudaCatchError(cudaFree(scratchs_array_device));
  delete[] scratchs_array_host;
}

template <unsigned int constexpr_val>
__host__ void Test(int num_multiprocessors, int num_threads, unsigned int max_runtime_val) {
  char* success_array_d;
  CudaCatchError(cudaMalloc(&success_array_d, sizeof(char) * num_multiprocessors * num_threads * 2));
  char* success_array_h = new char[num_multiprocessors * num_threads * 2];
  int num_blocks_per_multiprocessor;
  for (unsigned int runtime_val = 1; runtime_val <= max_runtime_val; runtime_val++) {
    for (int block_size = 16; block_size <= 1024; block_size += 16) {
      CudaCatchError(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &num_blocks_per_multiprocessor, (const void*)&Kernel<constexpr_val>, block_size, 0));
      int grid_size = num_blocks_per_multiprocessor * num_multiprocessors * 2;
      CudaCatchError(cudaMemsetAsync(success_array_d, 0, sizeof(char) * num_multiprocessors * num_threads, cudaStreamDefault));
      CudaCatchError(cudaStreamSynchronize(cudaStreamDefault));
      Kernel<constexpr_val><<<grid_size, block_size>>>(success_array_d, runtime_val);
      CudaCatchError(cudaDeviceSynchronize());
      CudaCatchError(cudaMemcpy(success_array_h, success_array_d, sizeof(char) * grid_size * block_size, cudaMemcpyDeviceToHost));
      bool all_valid = std::accumulate(success_array_h, success_array_h + grid_size * block_size, true, [](bool out, char val){
        return out and val > 0;
      });
      assert(all_valid);
    }
  }
  CudaCatchError(cudaFree(success_array_d));
}

int main(int argc, char** argv) {
  static constexpr unsigned int max_constexpr_val = 8; // Inclusive
  static const unsigned int max_runtime_val = 24; // Inclusive
  int num_multiprocessors, num_threads, device;
  CudaCatchError(cudaGetDevice(&device));
  CudaCatchError(cudaDeviceGetAttribute(&num_multiprocessors, cudaDevAttrMultiProcessorCount, device));
  CudaCatchError(cudaDeviceGetAttribute(&num_threads, cudaDevAttrMaxThreadsPerMultiProcessor, device));
  AllocateScratchArray<max_constexpr_val>(num_multiprocessors, num_threads, max_runtime_val);
  Test<1>(num_multiprocessors, num_threads, max_runtime_val);
  Test<2>(num_multiprocessors, num_threads, max_runtime_val);
  Test<3>(num_multiprocessors, num_threads, max_runtime_val);
  Test<4>(num_multiprocessors, num_threads, max_runtime_val);
  Test<5>(num_multiprocessors, num_threads, max_runtime_val);
  Test<6>(num_multiprocessors, num_threads, max_runtime_val);
  Test<7>(num_multiprocessors, num_threads, max_runtime_val);
  Test<8>(num_multiprocessors, num_threads, max_runtime_val);
  CleanupScratchArray(num_multiprocessors);
}