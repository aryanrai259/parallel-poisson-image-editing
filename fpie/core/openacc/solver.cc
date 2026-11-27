#include "solver.h"

#ifndef FPIE_ACC_MODULE
#define FPIE_ACC_MODULE core_openacc
#endif

PYBIND11_MODULE(FPIE_ACC_MODULE, m) {
  // module_local(): core_openacc and core_openacc_mc both compile this same
  // solver.cc and register classes of the same name; without module_local()
  // pybind11 registers them in the global type registry, so importing both
  // modules in one process raises "generic_type: type X is already
  // registered!" on the second import.
  py::class_<OpenACCEquSolver>(m, "EquSolver", py::module_local())
      .def(py::init<>())
      .def("partition", &OpenACCEquSolver::partition)
      .def("reset", &OpenACCEquSolver::reset)
      .def("sync", &OpenACCEquSolver::sync)
      .def("step", &OpenACCEquSolver::step);
  py::class_<OpenACCGridSolver>(m, "GridSolver", py::module_local())
      .def(py::init<int, int>())
      .def("reset", &OpenACCGridSolver::reset)
      .def("sync", &OpenACCGridSolver::sync)
      .def("step", &OpenACCGridSolver::step);
}
