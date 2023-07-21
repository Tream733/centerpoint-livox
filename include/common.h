#pragma once
#include <assert.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "cuda_runtime_api.h"

#define NUM_3D_BOX_CORNERS 8
#define NUM_2D_BOX_CORNERS 4
#define NUM_THREADS 64  // need to be changed when num_threads_ is changed


#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
#define GPU_CHECK(ans) \
  { GPUAssert((ans), __FILE__, __LINE__); }

inline void GPUAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
};

template <typename T>
void HOST_SAVE(T *array, int row, int col, std::string filename) {
  std::ofstream out_file(filename, std::ios::out);
  if (out_file.is_open()) {
    for (int i = 0; i < row; ++i) {
      for (int j = 0; j < col; ++j) {
        out_file << array[i * col + j] << " ";
      }
      out_file << "\n";
    }
  }
  out_file.close();
  std::cout << "Data has been written in " << filename << std::endl;
};

template <typename T>
void DEVICE_SAVE(T *array, int row, int col, std::string filename) {
  T *temp_ = new T[row * col];
  cudaMemcpy(temp_, array, row * col * sizeof(T), cudaMemcpyDeviceToHost);
  HOST_SAVE<T>(temp_, row, col, filename);
  delete[] temp_;
};