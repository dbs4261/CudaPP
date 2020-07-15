//
// Created by developer on 7/14/20.
//

#ifndef CUDAPP_SPECIAL_REGISTERS_CUH
#define CUDAPP_SPECIAL_REGISTERS_CUH

namespace cudapp {

__device__ unsigned int LaneId() {
  unsigned int lane_id;
  asm("mov.u32 %0, %%laneid;" : "=r"(lane_id));
  return lane_id;
}

__device__ unsigned int WarpId() {
  unsigned int warp_id;
  asm("mov.u32 %0, %%warpid;" : "=r"(warp_id));
  return warp_id;
}

__device__ unsigned int MultiprocessorId() {
  unsigned int multiprocessor_id;
  asm("mov.u32 %0, %%smid;" : "=r"(multiprocessor_id));
  return multiprocessor_id;
}

}

#endif //CUDAPP_SPECIAL_REGISTERS_CUH
