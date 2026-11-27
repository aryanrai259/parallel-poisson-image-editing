#include <openacc.h>

#include <cstring>
#include <tuple>

#include "solver.h"

OpenACCEquSolver::OpenACCEquSolver()
    : maskbuf(NULL),
      imgbuf(NULL),
      n_mid(0),
      dev_A(NULL),
      dev_B(NULL),
      dev_X(NULL),
      dev_img(NULL),
      dev_N(0),
      EquSolver() {}

OpenACCEquSolver::~OpenACCEquSolver() {
  unmap();
  if (maskbuf != NULL) {
    delete[] maskbuf;
  }
  if (imgbuf != NULL) {
    delete[] imgbuf;
  }
}

void OpenACCEquSolver::unmap() {
  if (dev_N > 0) {
    acc_delete(dev_A, sizeof(int) * dev_N * 4);
    acc_delete(dev_B, sizeof(float) * dev_N * 3);
    acc_delete(dev_X, sizeof(float) * dev_N * 3);
    acc_delete(dev_img, sizeof(unsigned char) * dev_N * 3);
    dev_N = 0;
  }
}

// Same red/black ordering as the OpenMP backend: all odd cells first, then
// all even cells, so each half-sweep only reads cells of the other colour.
py::array_t<int> OpenACCEquSolver::partition(py::array_t<int> mask) {
  auto arr = mask.unchecked<2>();
  int n = arr.shape(0), m = arr.shape(1);
  if (maskbuf != NULL) {
    delete[] maskbuf;
  }
  maskbuf = new int[n * m];
  int cnt = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      if ((i + j) % 2 == 1) {
        maskbuf[i * m + j] = arr(i, j) > 0 ? ++cnt : 0;
      }
    }
  }
  n_mid = cnt + 1;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      if ((i + j) % 2 == 0) {
        maskbuf[i * m + j] = arr(i, j) > 0 ? ++cnt : 0;
      }
    }
  }
  return py::array({n, m}, maskbuf);
}

void OpenACCEquSolver::post_reset() {
  unmap();
  if (imgbuf != NULL) {
    delete[] imgbuf;
  }
  imgbuf = new unsigned char[N * 3];
  acc_copyin(A, sizeof(int) * N * 4);
  acc_copyin(B, sizeof(float) * N * 3);
  acc_copyin(X, sizeof(float) * N * 3);
  acc_create(imgbuf, sizeof(unsigned char) * N * 3);
  dev_A = A;
  dev_B = B;
  dev_X = X;
  dev_img = imgbuf;
  dev_N = N;
}

std::tuple<py::array_t<unsigned char>, py::array_t<float>>
OpenACCEquSolver::step(int iteration) {
  const int n = N, nm = n_mid;
  int* a = A;
  float* b = B;
  float* x = X;
  unsigned char* img = imgbuf;

  for (int it = 0; it < iteration; ++it) {
    for (int half = 0; half < 2; ++half) {
      const int lo = half == 0 ? 1 : nm;
      const int hi = half == 0 ? nm : n;
#pragma acc parallel loop present(a[0 : 4 * n], b[0 : 3 * n], x[0 : 3 * n])
      for (int i = lo; i < hi; ++i) {
        const int o3 = i * 3, o4 = i * 4;
        const int i0 = a[o4 + 0] * 3, i1 = a[o4 + 1] * 3;
        const int i2 = a[o4 + 2] * 3, i3 = a[o4 + 3] * 3;
        x[o3 + 0] = (b[o3 + 0] + x[i0 + 0] + x[i1 + 0] + x[i2 + 0] + x[i3 + 0]) / 4;
        x[o3 + 1] = (b[o3 + 1] + x[i0 + 1] + x[i1 + 1] + x[i2 + 1] + x[i3 + 1]) / 4;
        x[o3 + 2] = (b[o3 + 2] + x[i0 + 2] + x[i1 + 2] + x[i2 + 2] + x[i3 + 2]) / 4;
      }
    }
  }

  float e0 = 0, e1 = 0, e2 = 0;
#pragma acc parallel loop reduction(+ : e0, e1, e2) \
    present(a[0 : 4 * n], b[0 : 3 * n], x[0 : 3 * n])
  for (int i = 1; i < n; ++i) {
    const int o3 = i * 3, o4 = i * 4;
    const int i0 = a[o4 + 0] * 3, i1 = a[o4 + 1] * 3;
    const int i2 = a[o4 + 2] * 3, i3 = a[o4 + 3] * 3;
    float r0 = b[o3 + 0] + x[i0 + 0] + x[i1 + 0] + x[i2 + 0] + x[i3 + 0] -
               x[o3 + 0] * 4.0f;
    float r1 = b[o3 + 1] + x[i0 + 1] + x[i1 + 1] + x[i2 + 1] + x[i3 + 1] -
               x[o3 + 1] * 4.0f;
    float r2 = b[o3 + 2] + x[i0 + 2] + x[i1 + 2] + x[i2 + 2] + x[i3 + 2] -
               x[o3 + 2] * 4.0f;
    e0 += r0 < 0 ? -r0 : r0;
    e1 += r1 < 0 ? -r1 : r1;
    e2 += r2 < 0 ? -r2 : r2;
  }
  err[0] = e0;
  err[1] = e1;
  err[2] = e2;

#pragma acc parallel loop present(x[0 : 3 * n], img[0 : 3 * n])
  for (int i = 0; i < n * 3; ++i) {
    img[i] = x[i] < 0 ? 0 : x[i] > 255 ? 255 : x[i];
  }
#pragma acc update self(img[0 : 3 * n])
  return std::make_tuple(py::array({N, 3}, imgbuf), py::array(3, err));
}
