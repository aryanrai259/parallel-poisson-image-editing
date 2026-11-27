#include "solver.h"

PYBIND11_MODULE(core_openmp, m) {
  py::class_<OpenMPEquSolver>(m, "EquSolver")
      .def(py::init<int, int, int>(), py::arg("n_cpu"),
           py::arg("sched_kind") = 1, py::arg("sched_chunk") = 0)
      .def("partition", &OpenMPEquSolver::partition)
      .def("reset", &OpenMPEquSolver::reset)
      .def("sync", &OpenMPEquSolver::sync)
      .def("step", &OpenMPEquSolver::step);
  py::class_<OpenMPGridSolver>(m, "GridSolver")
      .def(py::init<int, int, int, int, int>(), py::arg("grid_x"),
           py::arg("grid_y"), py::arg("n_cpu"), py::arg("sched_kind") = 1,
           py::arg("sched_chunk") = 0)
      .def("reset", &OpenMPGridSolver::reset)
      .def("sync", &OpenMPGridSolver::sync)
      .def("step", &OpenMPGridSolver::step);
}
