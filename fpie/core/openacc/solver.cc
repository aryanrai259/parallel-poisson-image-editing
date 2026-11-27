#include "solver.h"

#ifndef FPIE_ACC_MODULE
#define FPIE_ACC_MODULE core_openacc
#endif

PYBIND11_MODULE(FPIE_ACC_MODULE, m) {
  py::class_<OpenACCEquSolver>(m, "EquSolver")
      .def(py::init<>())
      .def("partition", &OpenACCEquSolver::partition)
      .def("reset", &OpenACCEquSolver::reset)
      .def("sync", &OpenACCEquSolver::sync)
      .def("step", &OpenACCEquSolver::step);
  py::class_<OpenACCGridSolver>(m, "GridSolver")
      .def(py::init<int, int>())
      .def("reset", &OpenACCGridSolver::reset)
      .def("sync", &OpenACCGridSolver::sync)
      .def("step", &OpenACCGridSolver::step);
}
