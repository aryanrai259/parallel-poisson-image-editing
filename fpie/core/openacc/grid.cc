#include <openacc.h>

#include <cstring>
#include <tuple>

#include "solver.h"

OpenACCGridSolver::OpenACCGridSolver(int grid_x, int grid_y)
    : imgbuf(NULL),
      tmp(NULL),
      dev_mask(NULL),
      dev_tgt(NULL),
      dev_grad(NULL),
      dev_tmp(NULL),
      dev_img(NULL),
      dev_size(0),
      GridSolver(grid_x, grid_y) {}

OpenACCGridSolver::~OpenACCGridSolver() {
  unmap();
  if (tmp != NULL) {
    delete[] tmp;
    delete[] imgbuf;
  }
}

void OpenACCGridSolver::unmap() {
  if (dev_size > 0) {
    acc_delete(dev_mask, sizeof(int) * dev_size);
    acc_delete(dev_tgt, sizeof(float) * dev_size * 3);
    acc_delete(dev_grad, sizeof(float) * dev_size * 3);
    acc_delete(dev_tmp, sizeof(float) * dev_size * 3);
    acc_delete(dev_img, sizeof(unsigned char) * dev_size * 3);
    dev_size = 0;
  }
}

void OpenACCGridSolver::post_reset() {
  unmap();
  if (tmp != NULL) {
    delete[] tmp;
    delete[] imgbuf;
  }
  tmp = new float[N * m3];
  imgbuf = new unsigned char[N * m3];
  memset(tmp, 0, sizeof(float) * N * m3);
  const int size = N * M;
  acc_copyin(mask, sizeof(int) * size);
  acc_copyin(tgt, sizeof(float) * size * 3);
  acc_copyin(grad, sizeof(float) * size * 3);
  acc_copyin(tmp, sizeof(float) * size * 3);
  acc_create(imgbuf, sizeof(unsigned char) * size * 3);
  dev_mask = mask;
  dev_tgt = tgt;
  dev_grad = grad;
  dev_tmp = tmp;
  dev_img = imgbuf;
  dev_size = size;
}

// grid_x / grid_y are CPU cache-tiling parameters; the accelerator version
// maps the 2-D image directly onto gangs and vector lanes instead.
std::tuple<py::array_t<unsigned char>, py::array_t<float>>
OpenACCGridSolver::step(int iteration) {
  const int n = N, m = M, s3 = m3, size = N * M;
  int* mk = mask;
  float* t = tgt;
  float* g = grad;
  float* w = tmp;
  unsigned char* img = imgbuf;

  for (int it = 0; it < iteration; ++it) {
#pragma acc parallel loop collapse(2) \
    present(mk[0 : size], t[0 : 3 * size], g[0 : 3 * size], w[0 : 3 * size])
    for (int r = 0; r < n; ++r) {
      for (int c = 0; c < m; ++c) {
        const int id = r * m + c;
        if (mk[id]) {
          const int o3 = id * 3;
          const int u = o3 - s3, l = o3 - 3, rr = o3 + 3, d = o3 + s3;
          w[o3 + 0] =
              (g[o3 + 0] + t[u + 0] + t[l + 0] + t[rr + 0] + t[d + 0]) / 4.0f;
          w[o3 + 1] =
              (g[o3 + 1] + t[u + 1] + t[l + 1] + t[rr + 1] + t[d + 1]) / 4.0f;
          w[o3 + 2] =
              (g[o3 + 2] + t[u + 2] + t[l + 2] + t[rr + 2] + t[d + 2]) / 4.0f;
        }
      }
    }
#pragma acc parallel loop present(mk[0 : size], t[0 : 3 * size], \
                                  w[0 : 3 * size])
    for (int id = 0; id < size; ++id) {
      if (mk[id]) {
        const int o3 = id * 3;
        t[o3 + 0] = w[o3 + 0];
        t[o3 + 1] = w[o3 + 1];
        t[o3 + 2] = w[o3 + 2];
      }
    }
  }

  float e0 = 0, e1 = 0, e2 = 0;
#pragma acc parallel loop reduction(+ : e0, e1, e2) \
    present(mk[0 : size], t[0 : 3 * size], g[0 : 3 * size])
  for (int id = 0; id < size; ++id) {
    if (mk[id]) {
      const int o3 = id * 3;
      const int u = o3 - s3, l = o3 - 3, rr = o3 + 3, d = o3 + s3;
      float r0 = g[o3 + 0] + t[u + 0] + t[l + 0] + t[rr + 0] + t[d + 0] -
                 t[o3 + 0] * 4.0f;
      float r1 = g[o3 + 1] + t[u + 1] + t[l + 1] + t[rr + 1] + t[d + 1] -
                 t[o3 + 1] * 4.0f;
      float r2 = g[o3 + 2] + t[u + 2] + t[l + 2] + t[rr + 2] + t[d + 2] -
                 t[o3 + 2] * 4.0f;
      e0 += r0 < 0 ? -r0 : r0;
      e1 += r1 < 0 ? -r1 : r1;
      e2 += r2 < 0 ? -r2 : r2;
    }
  }
  err[0] = e0;
  err[1] = e1;
  err[2] = e2;

#pragma acc parallel loop present(t[0 : 3 * size], img[0 : 3 * size])
  for (int i = 0; i < size * 3; ++i) {
    img[i] = t[i] < 0 ? 0 : t[i] > 255 ? 255 : t[i];
  }
#pragma acc update self(img[0 : 3 * size])
  return std::make_tuple(py::array({N, M, 3}, imgbuf), py::array(3, err));
}
