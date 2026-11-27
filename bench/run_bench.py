#!/usr/bin/env python3
"""Run the benchmark experiments through the fpie CLI.

Every run is a separate process, timed by the CLI's own solver timer
("Time elapsed"), which excludes image I/O and system setup. Rows are appended
to results/raw/bench.csv; finished rows are skipped on a re-run.

Experiments
  E1 backends   all backends, equ method, square/circle sizes 6..10
  E2 threads    OpenMP strong scaling, equ and grid, square10
  E3 schedule   OpenMP static/dynamic/guided and chunk size, circle10
  E4 mpi        MPI ranks and sync interval, square10
  E5 gpu        CUDA block size (equ) and tile size (grid), square10
  E6 methods    equ vs grid for each compiled backend, square10/circle10
  E7 accuracy   every backend against the exact sparse solution (bench/accuracy.py)
"""

import argparse
import csv
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data"
RESULTS = ROOT / "results"
RAW = RESULTS / "raw"
CSV_PATH = RAW / "bench.csv"

FIELDS = [
    "exp", "backend", "method", "input", "n_vars", "iters", "threads", "ranks",
    "schedule", "chunk", "block", "grid", "sync", "repeat", "time_s",
    "err_r", "err_g", "err_b", "timestamp",
]
GPU_BACKENDS = {"cuda", "openacc", "taichi-gpu"}
RE_TIME = re.compile(r"Time elapsed: ([0-9.eE+-]+)s")
RE_VARS = re.compile(r"# of vars: (\d+)")
RE_ERR = re.compile(r"abs error \[([^\]]*)\]")


def fpie_cmd() -> str:
    exe = Path(sys.executable).parent / "fpie"
    return str(exe) if exe.exists() else "fpie"


def available_backends() -> list:
    out = subprocess.run([fpie_cmd(), "--check-backend"], capture_output=True, text=True)
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("[")][-1]
    return [b.strip(" '") for b in line.strip("[]").split(",")]


def logical_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def physical_cores() -> int:
    try:
        out = subprocess.run(["lscpu", "-p=CORE,SOCKET"], capture_output=True, text=True).stdout
        cores = {ln for ln in out.splitlines() if ln and not ln.startswith("#")}
        return max(1, len(cores))
    except FileNotFoundError:
        return max(1, logical_cpus() // 2)


def shape_args(name: str) -> list:
    path = str(DATA / f"{name}.png")
    return ["-s", path, "-t", path, "-m", path]


def key(cfg: dict, repeat: int) -> tuple:
    return tuple(str(cfg.get(f, "")) for f in FIELDS[:13]) + (str(repeat),)


def run_once(cfg: dict, timeout: float) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        cmd = []
        if cfg["backend"] == "mpi":
            cmd += [shutil.which("mpiexec") or "mpiexec", "-np", str(cfg["ranks"])]
            if hasattr(os, "geteuid") and os.geteuid() == 0:
                cmd += ["--allow-run-as-root"]
            if int(cfg["ranks"]) > logical_cpus():
                cmd += ["--oversubscribe"]
        cmd += [fpie_cmd(), *shape_args(cfg["input"]), "-o", os.path.join(tmp, "out.png"),
                "-n", str(cfg["iters"]), "-b", cfg["backend"], "--method", cfg["method"]]
        if cfg["backend"] in ("openmp", "taichi-cpu", "numba"):
            cmd += ["-c", str(cfg["threads"])]
        if cfg["backend"] == "openmp":
            cmd += ["--omp-schedule", cfg["schedule"], "--omp-chunk", str(cfg["chunk"])]
        if cfg["backend"] == "cuda" and cfg["method"] == "equ":
            cmd += ["-z", str(cfg["block"])]
        if cfg["method"] == "grid":
            gx, gy = cfg["grid"].split("x")
            cmd += ["--grid-x", gx, "--grid-y", gy]
        if cfg["backend"] == "mpi":
            cmd += ["--mpi-sync-interval", str(cfg["sync"])]
        env = dict(os.environ)
        env.setdefault("OMP_PROC_BIND", "close")
        env.setdefault("OMP_PLACES", "cores")
        if cfg["backend"] == "openacc-mc":
            env["ACC_NUM_CORES"] = str(cfg["threads"])
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        text = proc.stdout + proc.stderr
        m_time, m_vars = RE_TIME.search(text), RE_VARS.search(text)
        if proc.returncode != 0 or not m_time:
            raise RuntimeError(f"run failed ({proc.returncode}): {' '.join(cmd)}\n{text[-2000:]}")
        errs = RE_ERR.findall(text)
        err = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", errs[-1])] if errs else []
        err = (err + [float("nan")] * 3)[:3]
        return {"time_s": float(m_time.group(1)), "n_vars": int(m_vars.group(1)) if m_vars else "",
                "err_r": err[0], "err_g": err[1], "err_b": err[2]}


def base(exp: str, backend: str, method: str, inp: str, iters: int, **kw) -> dict:
    cfg = {"exp": exp, "backend": backend, "method": method, "input": inp, "iters": iters,
           "threads": "", "ranks": "", "schedule": "", "chunk": "", "block": "", "grid": "", "sync": ""}
    if method == "grid":
        cfg["grid"] = "8x8"
    if backend == "openmp":
        cfg.update(threads=kw.get("phys"), schedule="static", chunk=0)
    if backend in ("numba", "taichi-cpu", "openacc-mc"):
        cfg["threads"] = kw.get("phys")
    if backend == "mpi":
        cfg.update(ranks=kw.get("phys"), sync=100)
    if backend == "cuda" and method == "equ":
        cfg["block"] = 256
    for k, v in kw.items():
        if k in cfg:
            cfg[k] = v
    return cfg


def experiments(quick: bool, backends: list, phys: int, logical: int) -> list:
    iters = 500 if quick else 5000
    sizes = [6, 7, 8] if quick else [6, 7, 8, 9, 10]
    big = f"square{sizes[-1]}"
    big_circle = f"circle{sizes[-1]}"
    cfgs = []

    # E1: every backend on both shapes; interpreted backends only on small sizes
    for b in backends:
        if b in ("taichi-cpu", "taichi-gpu"):
            continue
        for shape in ("square", "circle"):
            for s in sizes:
                if b == "numpy" and s > 7:
                    continue
                if b == "numba" and s > 8:
                    continue
                cfgs.append(base("E1", b, "equ", f"{shape}{s}", iters, phys=phys))

    # E2: OpenMP strong scaling up to all logical CPUs
    if "openmp" in backends:
        threads = sorted({t for t in (1, 2, 4, 6, 8, 12, 16, phys, logical) if t <= logical})
        for method in ("equ", "grid"):
            for t in threads:
                cfgs.append(base("E2", "openmp", method, big, iters, phys=phys, threads=t))

    # E3: OpenMP schedules on the circle (uneven work per grid tile)
    if "openmp" in backends:
        it3 = 200 if quick else 1000
        for method in ("equ", "grid"):
            cfgs.append(base("E3", "openmp", method, big_circle, it3, phys=phys, schedule="static", chunk=0))
            for sched in ("static", "dynamic", "guided"):
                chunks = (1, 4, 32) if method == "grid" else (64, 1024, 16384)
                for c in chunks:
                    cfgs.append(base("E3", "openmp", method, big_circle, it3, phys=phys, schedule=sched, chunk=c))

    # E4: MPI ranks and synchronisation interval
    if "mpi" in backends:
        ranks = sorted({r for r in (1, 2, 4, 8, phys) if r <= phys})
        for r in ranks:
            cfgs.append(base("E4", "mpi", "equ", big, iters, phys=phys, ranks=r))
        for sync in (10, 100, 1000):
            cfgs.append(base("E4", "mpi", "equ", big, iters, phys=phys, ranks=phys, sync=sync))

    # E5: GPU launch geometry
    if "cuda" in backends:
        for z in (32, 64, 128, 256, 512, 1024):
            cfgs.append(base("E5", "cuda", "equ", big, iters, phys=phys, block=z))
        for g in ("4x4", "8x8", "16x16", "32x32"):
            cfgs.append(base("E5", "cuda", "grid", big, iters, phys=phys, grid=g))

    # E6: equ vs grid for each compiled backend
    for b in ("gcc", "openmp", "mpi", "cuda", "openacc", "openacc-mc"):
        if b not in backends:
            continue
        for inp in (big, big_circle):
            for method in ("equ", "grid"):
                cfgs.append(base("E6", b, method, inp, iters, phys=phys))
    return cfgs


def environment() -> dict:
    def cmd(*args):
        try:
            return subprocess.run(list(args), capture_output=True, text=True, timeout=30).stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return ""

    # WSL exposes nvidia-smi under /usr/lib/wsl/lib, which only lands on PATH
    # via the interactive-shell profile snippet that sources it; a plain
    # subprocess (no shell profile) doesn't see it, so a bare "nvidia-smi"
    # lookup silently raises FileNotFoundError here and env["gpu"] ends up "".
    nvidia_smi = shutil.which("nvidia-smi") or next(
        (p for p in ("/usr/lib/wsl/lib/nvidia-smi", "/usr/bin/nvidia-smi") if os.path.exists(p)),
        "nvidia-smi",
    )

    lscpu = cmd("lscpu")
    model = re.search(r"Model name:\s*(.*)", lscpu)
    env = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "cpu_model": model.group(1).strip() if model else platform.processor(),
        "logical_cpus": logical_cpus(),
        "physical_cores": physical_cores(),
        "mem_total": cmd("bash", "-c", "free -h | awk '/Mem:/ {print $2}'"),
        "gpu": cmd(nvidia_smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"),
        "nvcc": cmd("bash", "-c", "nvcc --version | tail -n 2 | head -n 1"),
        "nvcpp": cmd("bash", "-c", "nvc++ --version | sed -n 2p"),
        "gcc": cmd("bash", "-c", "gcc --version | head -n 1"),
        "mpi": cmd("bash", "-c", "mpiexec --version | head -n 1"),
        "python": sys.version.split()[0],
        "backends": available_backends(),
        "omp_env": {k: os.environ.get(k, "") for k in ("OMP_PROC_BIND", "OMP_PLACES")},
    }
    try:
        import fpie
        import numpy

        env["fpie"] = fpie.__version__
        env["numpy"] = numpy.__version__
    except ImportError:
        pass
    return env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small sizes, 1 repeat")
    parser.add_argument("--only", default="", help="comma-separated experiments, e.g. E1,E2")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--fresh", action="store_true", help="discard previous results")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--skip-accuracy", action="store_true")
    args = parser.parse_args()

    if not (DATA / "square6.png").exists():
        sys.exit("bench/data is empty: run bench/prepare_data.py first")
    RAW.mkdir(parents=True, exist_ok=True)
    if args.fresh and CSV_PATH.exists():
        CSV_PATH.unlink()

    backends = available_backends()
    phys, logical = physical_cores(), logical_cpus()
    repeats = args.repeats or (1 if args.quick else 3)
    env = environment()
    env["mode"] = "quick" if args.quick else "full"
    env["repeats"] = repeats
    (RESULTS / "env.json").write_text(json.dumps(env, indent=2))
    print(json.dumps(env, indent=2))

    only = {e.strip() for e in args.only.split(",") if e.strip()}
    cfgs = [c for c in experiments(args.quick, backends, phys, logical) if not only or c["exp"] in only]

    done = set()
    if CSV_PATH.exists():
        with open(CSV_PATH) as f:
            for row in csv.DictReader(f):
                done.add(tuple(row[k] for k in FIELDS[:13]) + (row["repeat"],))
    new_file = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        total = len(cfgs) * repeats
        n = 0
        for cfg in cfgs:
            warm = cfg["backend"] in GPU_BACKENDS
            for r in range(repeats):
                n += 1
                if key(cfg, r) in done:
                    continue
                try:
                    if warm and r == 0:
                        run_once(dict(cfg, iters=min(100, int(cfg["iters"]))), args.timeout)
                    res = run_once(cfg, args.timeout)
                except Exception as exc:  # keep going; failures are listed at the end
                    print(f"[{n}/{total}] FAILED {cfg['exp']} {cfg['backend']} {cfg['input']}: {exc}")
                    continue
                row = dict(cfg, repeat=r, timestamp=time.strftime("%H:%M:%S"), **res)
                writer.writerow(row)
                f.flush()
                print(f"[{n}/{total}] {cfg['exp']} {cfg['backend']:<10} {cfg['method']} {cfg['input']:<9} "
                      f"t={cfg['threads']} np={cfg['ranks']} sch={cfg['schedule']}/{cfg['chunk']} "
                      f"z={cfg['block']} g={cfg['grid']} -> {res['time_s']:.4f}s")

    if not args.skip_accuracy and (not only or "E7" in only):
        acc = [sys.executable, str(HERE / "accuracy.py")] + (["--quick"] if args.quick else [])
        subprocess.run(acc, check=False)


if __name__ == "__main__":
    main()
