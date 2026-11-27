#!/usr/bin/env python3
"""E7: compare every backend's output with the exact solution.

The exact solution solves the same sparse Poisson system (4 x_i - sum of its
neighbours = b_i over the masked pixels) directly with SciPy's sparse LU. Each
backend runs through the CLI, so MPI is included, and the comparison uses only
the blended (masked) pixels. Writes results/raw/accuracy.csv.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from fpie.io import read_images
from fpie.process import EquProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from examples import EXAMPLES, DATA, available, cli_args  # noqa: E402
from run_bench import fpie_cmd, logical_cpus, physical_cores, available_backends  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "raw" / "accuracy.csv"
FIELDS = ["example", "n_vars", "backend", "method", "iters", "time_s",
          "max_abs", "mean_abs", "psnr_db", "timestamp"]


class _Capture:
    """Stands in for a solver core and keeps the linear system."""

    def __init__(self, partition):
        self.partition = partition
        self.system = None

    def reset(self, n, a, x, b):
        self.system = (n, np.array(a), np.array(b))

    def sync(self):
        pass


def exact(name: str):
    """Exact blended image and the target indices of the unknowns."""
    src_f, mask_f, tgt_f, h0, w0, h1, w1, grad, _ = EXAMPLES[name]
    src, mask, tgt = read_images(str(DATA / src_f), str(DATA / mask_f) if mask_f else "", str(DATA / tgt_f))
    proc = EquProcessor(grad, backend="numpy")
    cap = _Capture(proc.core.partition)
    proc.core = cap
    proc.reset(src, mask, tgt, (h0, w0), (h1, w1))
    n, a, b = cap.system
    rows, cols, vals = [], [], []
    for i in range(1, n):
        rows.append(i - 1)
        cols.append(i - 1)
        vals.append(4.0)
        for j in a[i]:
            if j > 0:
                rows.append(i - 1)
                cols.append(j - 1)
                vals.append(-1.0)
    mat = coo_matrix((vals, (rows, cols)), shape=(n - 1, n - 1)).tocsc()
    lu = splu(mat)
    x = np.stack([lu.solve(b[1:, c].astype(np.float64)) for c in range(3)], axis=1)
    out = proc.tgt.copy()
    out[proc.tgt_index] = np.clip(x, 0, 255)
    return out, proc.tgt_index, n - 1


def run_cli(name: str, backend: str, method: str, iters: int, out_png: str) -> float:
    cmd = []
    if backend == "mpi":
        cmd += [shutil.which("mpiexec") or "mpiexec", "-np", str(min(4, physical_cores()))]
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            cmd += ["--allow-run-as-root"]
        if min(4, physical_cores()) > logical_cpus():
            cmd += ["--oversubscribe"]
    cmd += [fpie_cmd(), *cli_args(name), "-o", out_png, "-n", str(iters), "-b", backend, "--method", method]
    if backend == "openmp":
        cmd += ["-c", str(physical_cores())]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if res.returncode != 0:
        raise RuntimeError(res.stdout[-800:] + res.stderr[-800:])
    for line in res.stdout.splitlines():
        if line.startswith("Time elapsed"):
            return float(line.split(":")[1].strip().rstrip("s"))
    return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    examples = available()
    if not examples:
        print("accuracy: no photo examples in bench/data, skipped")
        return
    names = list(examples)[:2] if args.quick else list(examples)
    iters_list = [500, 2000] if args.quick else [500, 2000, 5000, 20000]
    backends = [b for b in ("gcc", "openmp", "mpi", "cuda", "openacc", "openacc-mc") if b in available_backends()]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    new = not OUT.exists()
    with open(OUT, "a", newline="") as f, tempfile.TemporaryDirectory() as tmp:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        for name in names:
            ref, idx, n_vars = exact(name)
            ref_px = ref[idx].astype(np.float64)
            for backend in backends:
                methods = ("equ",) if backend == "mpi" else ("equ", "grid")
                for method in methods:
                    for iters in iters_list:
                        out_png = os.path.join(tmp, "out.png")
                        try:
                            t = run_cli(name, backend, method, iters, out_png)
                        except Exception as exc:
                            print(f"accuracy {name} {backend} {method} failed: {exc}")
                            continue
                        img = cv2.imread(out_png)
                        diff = np.abs(img[idx].astype(np.float64) - ref_px)
                        mse = float((diff ** 2).mean())
                        psnr = float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
                        row = {"example": name, "n_vars": n_vars, "backend": backend, "method": method,
                               "iters": iters, "time_s": t, "max_abs": float(diff.max()),
                               "mean_abs": float(diff.mean()), "psnr_db": psnr,
                               "timestamp": time.strftime("%H:%M:%S")}
                        writer.writerow(row)
                        f.flush()
                        print(f"accuracy {name:<6} {backend:<10} {method} n={iters:<6} "
                              f"max={row['max_abs']:.0f} mean={row['mean_abs']:.3f} psnr={psnr:.1f} dB")


if __name__ == "__main__":
    main()
