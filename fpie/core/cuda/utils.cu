#include <cstdio>
#include <stdexcept>
#include <string>

#include "utils.h"

void cuda_check(const char* what) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(err));
  }
}

void print_cuda_info() {
  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);

  printf("---------------------------------------------------------\n");
  printf("Found %d CUDA devices\n", deviceCount);

  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, i);
    printf("Device %d: %s\n", i, deviceProps.name);
    printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
    printf("   Global mem: %.0f MB\n",
           static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
    printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
  }
  printf("---------------------------------------------------------\n");
}
