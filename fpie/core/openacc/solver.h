#ifndef FPIE_CORE_OPENACC_SOLVER_H_
#define FPIE_CORE_OPENACC_SOLVER_H_

#include <tuple>

#include "base_solver.h"

// Red/black Gauss-Seidel equation solver. The arrays stay resident on the
// accelerator between step() calls; only the output image is copied back.
class OpenACCEquSolver : public EquSolver {
  int* maskbuf;
  unsigned char* imgbuf;
  int n_mid;
  // Host addresses currently mapped on the device, released on reset/exit.
  int* dev_A;
  float* dev_B;
  float* dev_X;
  unsigned char* dev_img;
  int dev_N;

  void unmap();

 public:
  OpenACCEquSolver();
  ~OpenACCEquSolver();

  py::array_t<int> partition(py::array_t<int> mask);
  void post_reset();

  std::tuple<py::array_t<unsigned char>, py::array_t<float>> step(
      int iteration);
};

// Jacobi grid solver over the masked bounding box.
class OpenACCGridSolver : public GridSolver {
  unsigned char* imgbuf;
  float* tmp;
  int* dev_mask;
  float* dev_tgt;
  float* dev_grad;
  float* dev_tmp;
  unsigned char* dev_img;
  int dev_size;

  void unmap();

 public:
  OpenACCGridSolver(int grid_x, int grid_y);
  ~OpenACCGridSolver();

  void post_reset();

  std::tuple<py::array_t<unsigned char>, py::array_t<float>> step(
      int iteration);
};

#endif  // FPIE_CORE_OPENACC_SOLVER_H_
