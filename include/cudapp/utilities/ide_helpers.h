//
// Created by Daniel Simon on 2/5/20.
//

#ifndef CUDAPP_IDE_HELPERS_H
#define CUDAPP_IDE_HELPERS_H

#include <cstdio>
#include <cstdlib>

#ifdef __JETBRAINS_IDE__

#error "Compiler should never see this. This is just for static analysis."
#ifndef __cplusplus
#define __cplusplus 201703L
#endif
#define __CUDACC__ 1
#define __CUDA_ARCH__ 1

#endif // __JETBRAINS_IDE__

#include <cuda_runtime_api.h>

void check(cudaError_t result, char const* const func, const char* const file, int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
            file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
    cudaDeviceReset();
    // Make sure we call CUDA Device Reset before exiting
    exit(EXIT_FAILURE);
  }
}

#ifndef CudaCatchError
#define CudaCatchError(val) check((val), #val, __FILE__, __LINE__)
#endif

#endif //CUDAPP_IDE_HELPERS_H
