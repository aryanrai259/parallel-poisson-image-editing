"""Demo: blend an example with several parallel backends and compare run times.

Run with demo/launch.sh (or: streamlit run demo/app.py).
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
from examples import DATA, EXAMPLES, available, cli_args  # noqa: E402
from run_bench import available_backends, fpie_cmd, logical_cpus, physical_cores  # noqa: E402

LABELS = {
    "gcc": "Serial C++", "openmp": "OpenMP", "mpi": "MPI", "cuda": "CUDA", "openacc": "OpenACC (GPU)",
    "openacc-mc": "OpenACC (multicore)", "numba": "Numba", "numpy": "NumPy",
}
COLORS = {
    "gcc": "#7F7F7F", "openmp": "#1F3FA3", "mpi": "#F36D25", "cuda": "#C00000",
    "openacc": "#2E8B57", "openacc-mc": "#17A2B8", "numba": "#8E44AD", "numpy": "#BFBFBF",
}
ORDER = ["gcc", "openmp", "mpi", "openacc-mc", "openacc", "cuda", "numba", "numpy"]

st.set_page_config(page_title="Parallel Poisson Image Editing", layout="wide")


@st.cache_data
def backend_list():
    return [b for b in ORDER if b in available_backends()]


def rgb(path_or_img):
    img = cv2.imread(str(path_or_img)) if isinstance(path_or_img, (str, Path)) else path_or_img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def direct_copy(name: str) -> np.ndarray:
    src_f, mask_f, tgt_f, h0, w0, h1, w1, _, _ = EXAMPLES[name]
    src = cv2.imread(str(DATA / src_f))
    tgt = cv2.imread(str(DATA / tgt_f))
    mask = cv2.imread(str(DATA / mask_f)) if mask_f else np.full_like(src, 255)
    m = mask.mean(-1)
    ys, xs = np.nonzero(m >= 128)
    ty, tx, sy, sx = ys + h1, xs + w1, ys + h0, xs + w0
    ok = (ty >= 0) & (ty < tgt.shape[0]) & (tx >= 0) & (tx < tgt.shape[1]) & \
         (sy >= 0) & (sy < src.shape[0]) & (sx >= 0) & (sx < src.shape[1])
    out = tgt.copy()
    out[ty[ok], tx[ok]] = src[sy[ok], sx[ok]]
    return out


def run(name, backend, method, iters, threads, ranks, block, out_png):
    cmd = []
    if backend == "mpi":
        cmd += [shutil.which("mpiexec") or "mpiexec", "-np", str(ranks)]
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            cmd += ["--allow-run-as-root"]
        if ranks > logical_cpus():
            cmd += ["--oversubscribe"]
    cmd += [fpie_cmd(), *cli_args(name), "-o", out_png, "-n", str(iters), "-b", backend, "--method", method]
    if backend == "openmp":
        cmd += ["-c", str(threads)]
    if backend == "cuda" and method == "equ":
        cmd += ["-z", str(block)]
    env = dict(os.environ, ACC_NUM_CORES=str(threads))
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    for line in res.stdout.splitlines():
        if line.startswith("Time elapsed"):
            return float(line.split(":")[1].strip().rstrip("s"))
    raise RuntimeError((res.stdout + res.stderr)[-600:])


examples = available()
if not examples:
    st.error("No examples found. Run: python bench/prepare_data.py")
    st.stop()

st.title("Parallel Poisson Image Editing")
st.caption("One Jacobi / red-black Gauss-Seidel solver, eight implementations: serial C++, OpenMP, MPI, CUDA, "
           "OpenACC on GPU and on CPU cores, Numba and NumPy.")

with st.sidebar:
    st.header("Settings")
    names = list(examples)
    name = st.selectbox("Example", names, index=names.index("test5") if "test5" in names else 0)
    iters = st.slider("Iterations", 500, 20000, EXAMPLES[name][8], step=500)
    method = st.radio("Solver", ["equ", "grid"], horizontal=True,
                      help="equ: red-black Gauss-Seidel over the masked pixels; grid: Jacobi over the bounding box")
    choices = backend_list()
    default = [b for b in ("gcc", "openmp", "mpi", "openacc", "cuda") if b in choices]
    chosen = st.multiselect("Backends", choices, default=default, format_func=lambda b: LABELS.get(b, b))
    threads = st.slider("CPU threads (OpenMP, OpenACC multicore)", 1, logical_cpus(), physical_cores())
    ranks = st.slider("MPI processes", 1, max(1, logical_cpus()), min(physical_cores(), 8))
    block = st.select_slider("CUDA block size", [32, 64, 128, 256, 512, 1024], value=256)

tab_blend, tab_scale, tab_results, tab_about = st.tabs(
    ["Blend and race", "Live thread scaling", "Benchmark results", "How it works"])

with tab_blend:
    src_f, mask_f, tgt_f = EXAMPLES[name][0], EXAMPLES[name][1], EXAMPLES[name][2]
    c1, c2, c3 = st.columns(3)
    c1.image(rgb(DATA / src_f), caption="source", width="stretch")
    c2.image(rgb(DATA / tgt_f), caption="target", width="stretch")
    c3.image(rgb(direct_copy(name)), caption="direct copy (no blending)", width="stretch")

    if st.button("Run selected backends", type="primary", disabled=not chosen):
        results = []
        progress = st.progress(0.0)
        with tempfile.TemporaryDirectory() as tmp:
            for i, b in enumerate(chosen):
                out = os.path.join(tmp, f"{b}.png")
                try:
                    t = run(name, b, method, iters, threads, ranks, block, out)
                    results.append((b, t, rgb(out)))
                except Exception as exc:
                    st.warning(f"{LABELS.get(b, b)} failed: {exc}")
                progress.progress((i + 1) / len(chosen))
        st.session_state["results"] = {"key": (name, method, iters), "rows": results}

    state = st.session_state.get("results")
    if state and state["rows"]:
        rows = state["rows"]
        ref = dict((b, t) for b, t, _ in rows).get("gcc")
        cols = st.columns(min(4, len(rows)))
        for i, (b, t, img) in enumerate(rows):
            note = f" · {ref / t:.1f}× vs serial" if ref and b != "gcc" else ""
            cols[i % len(cols)].image(img, caption=f"{LABELS.get(b, b)}: {t:.3f} s{note}", width="stretch")
        fig, ax = plt.subplots(figsize=(8, 0.5 + 0.45 * len(rows)))
        names = [LABELS.get(b, b) for b, _, _ in rows][::-1]
        times = [t for _, t, _ in rows][::-1]
        ax.barh(names, times, color=[COLORS.get(b, "#444") for b, _, _ in rows][::-1])
        for y, t in enumerate(times):
            ax.text(t, y, f" {t:.3f} s", va="center", fontsize=9)
        ax.set_xlabel("solver time (s)")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        k = state["key"]
        st.caption(f"{k[0]}, {k[1]} solver, {k[2]} iterations. Times exclude image loading and writing.")

with tab_scale:
    st.write("OpenMP speedup on this machine as threads are added (same example and settings).")
    inputs = list(examples) + [f"square{x}" for x in (8, 9, 10) if (DATA / f"square{x}.png").exists()]
    target = st.selectbox("Input", inputs, index=inputs.index("square9") if "square9" in inputs else 0)
    if st.button("Run thread sweep", disabled="openmp" not in choices):
        counts = sorted({t for t in (1, 2, 4, 6, 8, 12, 16, logical_cpus()) if t <= logical_cpus()})
        times = []
        bar = st.progress(0.0)
        with tempfile.TemporaryDirectory() as tmp:
            for i, t in enumerate(counts):
                out = os.path.join(tmp, "o.png")
                if target.startswith("square"):
                    p = str(DATA / f"{target}.png")
                    cmd = [fpie_cmd(), "-s", p, "-t", p, "-m", p, "-o", out, "-n", str(iters),
                           "-b", "openmp", "--method", method, "-c", str(t)]
                    res = subprocess.run(cmd, capture_output=True, text=True)
                    times.append(float([ln for ln in res.stdout.splitlines()
                                        if ln.startswith("Time")][0].split(":")[1].strip().rstrip("s")))
                else:
                    times.append(run(target, "openmp", method, iters, t, 1, 256, out))
                bar.progress((i + 1) / len(counts))
        sp = [times[0] / x for x in times]
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.plot(counts, counts, "--", color="#999", label="ideal")
        ax.plot(counts, sp, "o-", color=COLORS["openmp"], label="measured")
        ax.axvline(physical_cores(), color="#bbb", lw=0.8)
        ax.set_xlabel("threads")
        ax.set_ylabel("speedup over 1 thread")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        st.table({"threads": counts, "time (s)": [f"{x:.4f}" for x in times], "speedup": [f"{x:.2f}" for x in sp]})

with tab_results:
    res_dir = ROOT / "results"
    env_path = res_dir / "env.json"
    if env_path.exists():
        env = json.loads(env_path.read_text())
        st.write(f"Measured on {env.get('cpu_model', '')} ({env.get('physical_cores', '')} cores) "
                 f"and {env.get('gpu', '') or 'no GPU'}, {env.get('date', '')}.")
    figs = [("fig_speedup_largest.png", "Speedup over serial C++ at the largest size"),
            ("fig_backends_vs_size.png", "Run time against problem size"),
            ("fig_openmp_scaling.png", "OpenMP strong scaling with an Amdahl fit"),
            ("fig_openmp_schedule.png", "OpenMP loop schedules"),
            ("fig_mpi.png", "MPI processes and synchronisation interval"),
            ("fig_cuda_launch.png", "CUDA launch geometry"),
            ("fig_equ_vs_grid.png", "Equation solver against grid solver"),
            ("fig_accuracy.png", "Accuracy against the exact sparse solution")]
    shown = False
    for f, cap in figs:
        if (res_dir / "figures" / f).exists():
            st.subheader(cap)
            st.image(str(res_dir / "figures" / f), width="stretch")
            shown = True
    if not shown:
        st.info("No benchmark results yet. Run: python bench/run_bench.py && python bench/make_figures.py")

with tab_about:
    st.markdown(r"""
**Poisson image editing** (Pérez, Gangnet and Blake, 2003) pastes a source region into a target image by
solving for new pixel values whose gradients match the source while the border matches the target:

$\Delta f = \operatorname{div} \mathbf{v}$ inside the region, $f = f^{*}$ on its border.

On the pixel grid this becomes one linear equation per masked pixel,
$4 f_p - \sum_{q \in N(p)} f_q = b_p$, solved iteratively. Every pixel update needs only its four
neighbours, so the work is data-parallel.

| Implementation | Parallel model | Lab topic |
|---|---|---|
| Serial C++ | none (baseline) | |
| OpenMP | shared-memory threads, `parallel for`, schedules | Programs 1a, 2, 3, 4, 6 |
| MPI | distributed processes, Send/Recv and broadcast | Programs 1b, 5 |
| CUDA | GPU threads and blocks | GPU programming |
| OpenACC | directives, GPU and multicore from one source | Program 7 |

Based on [Fast-Poisson-Image-Editing](https://github.com/Trinkle23897/Fast-Poisson-Image-Editing) by
Jiayi Weng and Zixu Chen (MIT licence). This version adds the OpenACC backends, OpenMP schedule
selection, the benchmark harness, the accuracy check against the exact solution and this demo.
""")
