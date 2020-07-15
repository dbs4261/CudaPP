//
// Created by Daniel Simon on 2/11/20.
//

#ifndef CUDAPP_CUDA_EXCEPTION_H
#define CUDAPP_CUDA_EXCEPTION_H

#include <exception>
#include <string>
#include <sstream>

namespace cudapp {

class CudaException : public std::exception {
 public:
  /**
   * @brief A wrapper that will read out a cuda error when captured.
   * @param _error The cuda error encountered.
   */
  explicit CudaException(cudaError_t _error) noexcept : error(_error) {
    std::stringstream ss;
    ss << cudaGetErrorName(error) << " (" << static_cast<int>(error) << "): " << cudaGetErrorString(error);
    info = ss.str();
  }

  /**
   * @brief Accesses the info on the cuda error captured.
   * @return A pointer to the beginning of that null terminated string.
   */
  const char* what() const noexcept override {
    return info.c_str();
  }

 protected:
  cudaError_t error;
  std::string info;
};

}

#endif //CUDAPP_CUDA_EXCEPTION_H
