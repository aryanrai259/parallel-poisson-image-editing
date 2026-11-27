"""Tests for the OpenMP schedule option and the OpenACC backends."""

import unittest

import numpy as np

from fpie.process import ALL_BACKEND, EquProcessor, GridProcessor


def _problem(seed: int = 0, size: int = 40):
    rng = np.random.default_rng(seed)
    src = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    tgt = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    inner = rng.random((size - 4, size - 4)) > 0.3
    mask[2:-2, 2:-2] = inner.astype(np.uint8) * 255
    return src, mask, tgt


def _run(proc, iters: int, seed: int = 0):
    src, mask, tgt = _problem(seed)
    proc.reset(src, mask, tgt.copy(), (0, 0), (0, 0))
    return proc.step(iters)


@unittest.skipUnless("openmp" in ALL_BACKEND, "OpenMP backend unavailable")
class OpenMPScheduleTest(unittest.TestCase):
    """Every schedule must give the same result as the static default."""

    def test_equ_schedules_identical(self) -> None:
        ref, ref_err = _run(EquProcessor(backend="openmp", n_cpu=4), 50)
        for schedule in ("static", "dynamic", "guided"):
            for chunk in (0, 1, 64):
                proc = EquProcessor(
                    backend="openmp",
                    n_cpu=4,
                    omp_schedule=schedule,
                    omp_chunk=chunk,
                )
                out, err = _run(proc, 50)
                np.testing.assert_array_equal(out, ref)
                np.testing.assert_allclose(err, ref_err, rtol=1e-6)

    def test_grid_schedules_identical(self) -> None:
        kw = {"backend": "openmp", "n_cpu": 4, "grid_x": 4, "grid_y": 4}
        ref, _ = _run(GridProcessor(**kw), 50)
        for schedule in ("static", "dynamic", "guided"):
            proc = GridProcessor(**kw, omp_schedule=schedule, omp_chunk=2)
            out, _ = _run(proc, 50)
            np.testing.assert_array_equal(out, ref)


@unittest.skipUnless("openacc" in ALL_BACKEND, "OpenACC backend unavailable")
class OpenACCTest(unittest.TestCase):
    """OpenACC solvers must reproduce the reference solvers."""

    def test_grid_matches_numpy(self) -> None:
        ref, ref_err = _run(GridProcessor(backend="numpy"), 30)
        out, err = _run(GridProcessor(backend="openacc"), 30)
        # fused multiply-add on the GPU may move a value across a rounding
        # boundary, so allow one intensity level.
        self.assertLessEqual(
            int(np.abs(out.astype(int) - ref.astype(int)).max()), 1
        )
        np.testing.assert_allclose(err, ref_err, rtol=1e-3)

    @unittest.skipUnless("openmp" in ALL_BACKEND, "OpenMP backend unavailable")
    def test_equ_matches_openmp(self) -> None:
        # Both use the same red/black ordering.
        ref, ref_err = _run(EquProcessor(backend="openmp", n_cpu=2), 30)
        out, err = _run(EquProcessor(backend="openacc"), 30)
        self.assertLessEqual(
            int(np.abs(out.astype(int) - ref.astype(int)).max()), 1
        )
        np.testing.assert_allclose(err, ref_err, rtol=1e-3)

    def test_reset_twice(self) -> None:
        # The device mapping must be released and rebuilt on a second reset.
        proc = EquProcessor(backend="openacc")
        first, _ = _run(proc, 20, seed=1)
        second, _ = _run(proc, 20, seed=1)
        np.testing.assert_array_equal(first, second)
        grid = GridProcessor(backend="openacc")
        g1, _ = _run(grid, 20, seed=2)
        g2, _ = _run(grid, 20, seed=2)
        np.testing.assert_array_equal(g1, g2)


@unittest.skipUnless("openacc-mc" in ALL_BACKEND, "OpenACC multicore unavailable")
class OpenACCMulticoreTest(unittest.TestCase):
    """The multicore build of the same source must agree with the GPU build."""

    def test_equ_matches_openmp(self) -> None:
        ref, _ = _run(EquProcessor(backend="openmp", n_cpu=2), 30)
        out, _ = _run(EquProcessor(backend="openacc-mc"), 30)
        self.assertLessEqual(
            int(np.abs(out.astype(int) - ref.astype(int)).max()), 1
        )


if __name__ == "__main__":
    unittest.main()
