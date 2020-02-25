# CudaPP
CudaPP is a C++ library of wrappers around CUDA objects with (hopefully) low to no performance impact.
This is different from a library like Thrust or CUB as it aims to have the same direct low level functionality as writing raw CUDA with with more type safety and better interaction with modern C++ host code.


##### FAQ:
Q: Why did you choose to use exceptions? CUDA is for performance!

A: In most libraries cudaError_t returns are captured and if an error is detected the program exits. 
This prevents cleanup and makes tracing errors challenging.
In cases where the error returned could be used for control flow like cudaMemGetInfo CudaPP processes the useful error code and provides that information to the caller.

