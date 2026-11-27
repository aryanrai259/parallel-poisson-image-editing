#!/usr/bin/env python3
"""Turn results/raw/*.csv into figures (results/figures) and summaries
(results/summary.md, results/summary.json)."""

import json
import math
import sys
from pathlib import Path

import cv2
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
FIG = RES / "figures"
sys.path.insert(0, str(HERE))

COLORS = {
    "gcc": "#7F7F7F", "openmp": "#1F3FA3", "mpi": "#F36D25", "cuda": "#C00000",
    "openacc": "#2E8B57", "openacc-mc": "#17A2B8", "numba": "#8E44AD", "numpy": "#BFBFBF",
}
LABELS = {
    "gcc": "Serial C++ (gcc)", "openmp": "OpenMP", "mpi": "MPI", "cuda": "CUDA",
    "openacc": "OpenACC (GPU)", "openacc-mc": "OpenACC (multicore)", "numba": "Numba", "numpy": "NumPy",
}
ORDER = ["numpy", "numba", "gcc", "openmp", "mpi", "openacc-mc", "openacc", "cuda"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9,
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def med(df: pd.DataFrame, by: list) -> pd.DataFrame:
    return df.groupby(by, dropna=False)["time_s"].median().reset_index()


def size_of(name: str) -> int:
    return int("".join(c for c in name if c.isdigit()))


def save(fig, name: str) -> None:
    fig.savefig(FIG / name)
    plt.close(fig)
    print(f"figure {name}")


def fig_backends(df, summary):
    d = df[df.exp == "E1"]
    if d.empty:
        return
    m = med(d, ["backend", "input"])
    n_vars = d.groupby("input")["n_vars"].first()
    m["n_vars"] = m["input"].map(n_vars)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, shape in zip(axes, ("square", "circle")):
        for b in ORDER:
            s = m[(m.backend == b) & m.input.str.startswith(shape)].sort_values("n_vars")
            if s.empty:
                continue
            ax.loglog(s.n_vars, s.time_s, "o-", color=COLORS[b], label=LABELS[b], ms=4, lw=1.6)
        ax.set_title(f"{shape} masks, equ method")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("unknowns (masked pixels)")
    axes[0].set_ylabel("time for the run (s)")
    axes[1].legend(loc="upper left", frameon=False)
    save(fig, "fig_backends_vs_size.png")

    largest = max(d.input.map(size_of))
    rows = []
    for shape in ("square", "circle"):
        inp = f"{shape}{largest}"
        s = m[m.input == inp].set_index("backend")["time_s"]
        if "gcc" not in s:
            continue
        for b in ORDER:
            if b in s:
                rows.append({"input": inp, "backend": b, "time_s": float(s[b]),
                             "speedup_vs_gcc": float(s["gcc"] / s[b])})
    sp = pd.DataFrame(rows)
    summary["E1"] = {"largest": largest, "rows": rows,
                     "n_vars": {k: int(v) for k, v in n_vars.items() if str(v) != "nan"}}
    if sp.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 3.8))
    inputs = list(dict.fromkeys(sp.input))
    bks = [b for b in ORDER if b in set(sp.backend)]
    w = 0.8 / len(inputs)
    for i, inp in enumerate(inputs):
        s = sp[sp.input == inp].set_index("backend").reindex(bks)
        x = np.arange(len(bks)) + (i - (len(inputs) - 1) / 2) * w
        bars = ax.bar(x, s.speedup_vs_gcc, w * 0.95, color=[COLORS[b] for b in bks],
                      alpha=1.0 if i == 0 else 0.55, label=inp)
        for xi, v in zip(x, s.speedup_vs_gcc):
            if not math.isnan(v):
                ax.text(xi, v, f"{v:.1f}×", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(np.arange(len(bks)), [LABELS[b] for b in bks], rotation=20, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("speedup over serial gcc")
    ax.set_title(f"Speedup at the largest size ({', '.join(inputs)}; solid = square, light = circle)")
    save(fig, "fig_speedup_largest.png")


def amdahl_fit(p, t):
    """Least-squares fit of T(p)/T(1) = f + (1 - f)/p."""
    y = np.asarray(t) / t[0]
    x = 1 / np.asarray(p)
    slope, intercept = np.polyfit(x, y, 1)
    f = float(np.clip(intercept / (intercept + slope), 0, 1))
    return f


def fig_threads(df, summary, env):
    d = df[df.exp == "E2"]
    if d.empty:
        return
    phys = int(env.get("physical_cores", 8))
    m = med(d, ["method", "threads"])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    out = {}
    pmax = int(m.threads.max())
    ax.plot([1, pmax], [1, pmax], "--", color="#999999", lw=1, label="ideal")
    for method, color in (("equ", "#1F3FA3"), ("grid", "#C00000")):
        s = m[m.method == method].sort_values("threads")
        if s.empty or 1 not in set(s.threads):
            continue
        t1 = float(s[s.threads == 1].time_s.iloc[0])
        sp = t1 / s.time_s
        ax.plot(s.threads, sp, "o-", color=color, label=f"{method} method", lw=1.8, ms=5)
        fit = s[s.threads <= phys]
        f = amdahl_fit(list(fit.threads), list(fit.time_s)) if len(fit) >= 3 else float("nan")
        ps = np.linspace(1, pmax, 100)
        if not math.isnan(f):
            ax.plot(ps, 1 / (f + (1 - f) / ps), ":", color=color, lw=1.2,
                    label=f"Amdahl fit, serial fraction {f:.3f}")
        rows = []
        for th, tt in zip(s.threads, s.time_s):
            spd = t1 / tt
            kf = (1 / spd - 1 / th) / (1 - 1 / th) if th > 1 else float("nan")
            rows.append({"threads": int(th), "time_s": float(tt), "speedup": float(spd),
                         "efficiency": float(spd / th), "karp_flatt": kf})
        out[method] = {"serial_fraction": f, "rows": rows, "t1": t1}
    ax.axvline(phys, color="#999999", lw=0.8)
    ax.text(phys, 0.5, f" {phys} physical cores", fontsize=8, color="#666666", va="bottom")
    ax.set_xlabel("OpenMP threads")
    ax.set_ylabel("speedup over 1 thread")
    ax.set_title(f"OpenMP strong scaling ({d.input.iloc[0]})")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "fig_openmp_scaling.png")
    summary["E2"] = out


def fig_schedule(df, summary):
    d = df[df.exp == "E3"]
    if d.empty:
        return
    m = med(d, ["method", "schedule", "chunk"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    out = {}
    colors = {"static": "#1F3FA3", "dynamic": "#C00000", "guided": "#F36D25"}
    for ax, method in zip(axes, ("equ", "grid")):
        s = m[m.method == method].copy()
        if s.empty:
            continue
        base_t = float(s[(s.schedule == "static") & (s.chunk.astype(int) == 0)].time_s.iloc[0])
        s["label"] = s.apply(lambda r: f"{r.schedule} {'default' if int(r.chunk) == 0 else int(r.chunk)}", axis=1)
        s = s.sort_values(["schedule", "chunk"], key=lambda c: c if c.name != "schedule" else c.map(
            {"static": 0, "dynamic": 1, "guided": 2}))
        ax.bar(range(len(s)), s.time_s / base_t, color=[colors[x] for x in s.schedule])
        ax.axhline(1, color="#666666", lw=0.8)
        ax.set_xticks(range(len(s)), s.label, fontsize=8, rotation=40, ha="right")
        ax.set_ylabel("time relative to static (default)")
        ax.set_title(f"{method} method, {d.input.iloc[0]}, {int(d.threads.iloc[0])} threads")
        out[method] = [{"schedule": r.schedule, "chunk": int(r.chunk), "time_s": float(r.time_s),
                        "relative": float(r.time_s / base_t)} for r in s.itertuples()]
    save(fig, "fig_openmp_schedule.png")
    summary["E3"] = out


def fig_mpi(df, summary):
    d = df[df.exp == "E4"]
    if d.empty:
        return
    m = med(d, ["ranks", "sync"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    s = m[m.sync.astype(int) == 100].sort_values("ranks")
    out = {}
    if not s.empty and 1 in set(s.ranks.astype(int)):
        t1 = float(s[s.ranks.astype(int) == 1].time_s.iloc[0])
        axes[0].plot(s.ranks, t1 / s.time_s, "o-", color=COLORS["mpi"], lw=1.8, label="MPI (sync every 100)")
        axes[0].plot([1, s.ranks.max()], [1, s.ranks.max()], "--", color="#999999", lw=1, label="ideal")
        axes[0].set_xlabel("MPI processes")
        axes[0].set_ylabel("speedup over 1 process")
        axes[0].legend(frameon=False)
        axes[0].set_title(f"MPI scaling ({d.input.iloc[0]})")
        out["ranks"] = [{"ranks": int(r.ranks), "time_s": float(r.time_s), "speedup": float(t1 / r.time_s)}
                        for r in s.itertuples()]
    pr = int(m.ranks.max())
    s2 = m[m.ranks.astype(int) == pr].sort_values("sync")
    axes[1].bar([str(int(v)) for v in s2.sync], s2.time_s, color=COLORS["mpi"])
    axes[1].set_xlabel("iterations between synchronisations")
    axes[1].set_ylabel("time (s)")
    axes[1].set_title(f"Synchronisation interval, {pr} processes")
    out["sync"] = [{"sync": int(r.sync), "time_s": float(r.time_s)} for r in s2.itertuples()]
    save(fig, "fig_mpi.png")
    summary["E4"] = out


def fig_gpu(df, summary):
    d = df[df.exp == "E5"]
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    out = {}
    e = med(d[d.method == "equ"], ["block"]).sort_values("block")
    if not e.empty:
        axes[0].plot(e.block.astype(int), e.time_s, "o-", color=COLORS["cuda"], lw=1.8)
        axes[0].set_xscale("log", base=2)
        axes[0].set_xlabel("threads per block (equ kernel)")
        axes[0].set_ylabel("time (s)")
        axes[0].set_title("CUDA block size")
        out["block"] = [{"block": int(r.block), "time_s": float(r.time_s)} for r in e.itertuples()]
    g = med(d[d.method == "grid"], ["grid"])
    if not g.empty:
        g["n"] = g.grid.str.split("x").str[0].astype(int)
        g = g.sort_values("n")
        axes[1].bar(g.grid, g.time_s, color=COLORS["cuda"])
        axes[1].set_xlabel("thread-block tile (grid kernel)")
        axes[1].set_ylabel("time (s)")
        axes[1].set_title("CUDA tile size")
        out["grid"] = [{"grid": r.grid, "time_s": float(r.time_s)} for r in g.itertuples()]
    save(fig, "fig_cuda_launch.png")
    summary["E5"] = out


def fig_methods(df, summary):
    d = df[df.exp == "E6"]
    if d.empty:
        return
    m = med(d, ["backend", "input", "method"])
    piv = m.pivot_table(index=["backend", "input"], columns="method", values="time_s").reset_index()
    if "equ" not in piv or "grid" not in piv:
        return
    piv["grid_over_equ"] = piv["grid"] / piv["equ"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    inputs = list(dict.fromkeys(piv.input))
    bks = [b for b in ORDER if b in set(piv.backend)]
    w = 0.8 / max(1, len(inputs))
    for i, inp in enumerate(inputs):
        s = piv[piv.input == inp].set_index("backend").reindex(bks)
        x = np.arange(len(bks)) + (i - (len(inputs) - 1) / 2) * w
        ax.bar(x, s.grid_over_equ, w * 0.95, color=[COLORS[b] for b in bks], alpha=1.0 if i == 0 else 0.55,
               label=inp)
    ax.axhline(1, color="#666666", lw=0.8)
    ax.set_xticks(np.arange(len(bks)), [LABELS[b] for b in bks], rotation=15, ha="right")
    ax.set_ylabel("grid time ÷ equ time")
    ax.set_title("Equation solver vs grid solver (below 1: grid is faster)")
    save(fig, "fig_equ_vs_grid.png")
    summary["E6"] = [{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()}
                     for r in piv.to_dict("records")]


def fig_accuracy(summary):
    path = RES / "raw" / "accuracy.csv"
    if not path.exists():
        return
    a = pd.read_csv(path)
    a = a.replace([np.inf], 99.0)
    examples = list(dict.fromkeys(a.example))
    fig, axes = plt.subplots(1, len(examples), figsize=(4.6 * len(examples), 3.8), squeeze=False)
    for ax, ex in zip(axes[0], examples):
        s = a[a.example == ex]
        for (b, method), g in s.groupby(["backend", "method"]):
            g = g.groupby("iters")["psnr_db"].median().reset_index()
            ax.semilogx(g.iters, g.psnr_db, "o-" if method == "equ" else "s--", color=COLORS.get(b, "k"),
                        label=f"{LABELS.get(b, b)} {method}", lw=1.4, ms=4)
        ax.set_title(f"{ex} ({int(s.n_vars.iloc[0])} unknowns)")
        ax.set_xlabel("iterations")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylabel("PSNR vs exact solution (dB)")
    axes[0][-1].legend(frameon=False, fontsize=7, loc="lower right")
    save(fig, "fig_accuracy.png")
    top = a[a.iters == a.iters.max()]
    summary["E7"] = {"iters": int(a.iters.max()),
                     "rows": top.groupby(["backend", "method"]).agg(
                         psnr_db=("psnr_db", "median"), max_abs=("max_abs", "max"),
                         mean_abs=("mean_abs", "median")).reset_index().to_dict("records"),
                     "all": a.to_dict("records")}


def fig_examples():
    try:
        from examples import EXAMPLES, DATA, available
        from fpie.io import read_images
        from fpie.process import ALL_BACKEND, EquProcessor
    except ImportError:
        return
    ex = list(available())[:3]
    if not ex:
        return
    backend = "openmp" if "openmp" in ALL_BACKEND else "numpy"
    fig, axes = plt.subplots(len(ex), 4, figsize=(12, 3.0 * len(ex)), squeeze=False)
    for row, name in zip(axes, ex):
        src_f, mask_f, tgt_f, h0, w0, h1, w1, grad, iters = EXAMPLES[name]
        src, mask, tgt = read_images(str(DATA / src_f), str(DATA / mask_f) if mask_f else "", str(DATA / tgt_f))
        m = mask.mean(-1) if mask.ndim == 3 else mask
        naive = tgt.copy()
        ys, xs = np.nonzero(m >= 128)
        ty, tx, sy, sx = ys + h1, xs + w1, ys + h0, xs + w0
        ok = (ty >= 0) & (ty < tgt.shape[0]) & (tx >= 0) & (tx < tgt.shape[1]) & \
             (sy >= 0) & (sy < src.shape[0]) & (sx >= 0) & (sx < src.shape[1])
        naive[ty[ok], tx[ok]] = src[sy[ok], sx[ok]]
        proc = EquProcessor(grad, backend=backend)
        proc.reset(src, mask, tgt, (h0, w0), (h1, w1))
        out, _ = proc.step(iters)
        for ax, img, title in zip(row, (src, tgt, naive, out),
                                  ("source", "target", "direct copy", "Poisson blend")):
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{name}: {title}" if title == "source" else title)
            ax.axis("off")
    save(fig, "fig_examples.png")


def write_markdown(summary, env):
    lines = ["# Results", "", f"Machine: {env.get('cpu_model', '')}, {env.get('physical_cores', '')} cores / "
             f"{env.get('logical_cpus', '')} threads; GPU: {env.get('gpu', 'none') or 'none'}.",
             f"Mode: {env.get('mode', '')}, {env.get('repeats', '')} repeats per configuration (medians).", ""]
    if "E1" in summary:
        lines += ["## E1 Backends at the largest size", "", "| input | backend | time (s) | speedup vs gcc |",
                  "|---|---|---|---|"]
        lines += [f"| {r['input']} | {LABELS[r['backend']]} | {r['time_s']:.4f} | {r['speedup_vs_gcc']:.1f}x |"
                  for r in summary["E1"]["rows"]]
        lines.append("")
    if "E2" in summary:
        lines += ["## E2 OpenMP strong scaling", ""]
        for method, v in summary["E2"].items():
            lines += [f"{method}: serial fraction (Amdahl fit, up to physical cores) = {v['serial_fraction']:.3f}",
                      "", "| threads | time (s) | speedup | efficiency |", "|---|---|---|---|"]
            lines += [f"| {r['threads']} | {r['time_s']:.4f} | {r['speedup']:.2f} | {r['efficiency']:.2f} |"
                      for r in v["rows"]]
            lines.append("")
    if "E3" in summary:
        lines += ["## E3 OpenMP schedules (time relative to static default)", ""]
        for method, rows in summary["E3"].items():
            lines.append(f"{method}: " + ", ".join(
                f"{r['schedule']}/{r['chunk'] or 'default'} {r['relative']:.2f}" for r in rows))
        lines.append("")
    if "E4" in summary:
        lines += ["## E4 MPI", ""]
        lines += [f"{r['ranks']} processes: {r['time_s']:.4f} s ({r['speedup']:.2f}x)"
                  for r in summary["E4"].get("ranks", [])]
        lines += [f"sync every {r['sync']}: {r['time_s']:.4f} s" for r in summary["E4"].get("sync", [])]
        lines.append("")
    if "E5" in summary:
        lines += ["## E5 CUDA launch geometry", ""]
        lines += [f"block {r['block']}: {r['time_s']:.4f} s" for r in summary["E5"].get("block", [])]
        lines += [f"tile {r['grid']}: {r['time_s']:.4f} s" for r in summary["E5"].get("grid", [])]
        lines.append("")
    if "E7" in summary:
        lines += [f"## E7 Accuracy against the exact solution ({summary['E7']['iters']} iterations)", "",
                  "| backend | method | PSNR (dB) | max abs | mean abs |", "|---|---|---|---|---|"]
        lines += [f"| {LABELS.get(r['backend'], r['backend'])} | {r['method']} | {r['psnr_db']:.1f} | "
                  f"{r['max_abs']:.0f} | {r['mean_abs']:.3f} |" for r in summary["E7"]["rows"]]
        lines.append("")
    (RES / "summary.md").write_text("\n".join(lines))


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    env = json.loads((RES / "env.json").read_text()) if (RES / "env.json").exists() else {}
    summary = {"env": env}
    bench = RES / "raw" / "bench.csv"
    if bench.exists():
        df = pd.read_csv(bench)
        for col in ("threads", "ranks", "chunk", "block", "sync"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        fig_backends(df, summary)
        fig_threads(df, summary, env)
        fig_schedule(df, summary)
        fig_mpi(df, summary)
        fig_gpu(df, summary)
        fig_methods(df, summary)
    fig_accuracy(summary)
    fig_examples()
    (RES / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    write_markdown(summary, env)
    print(f"summary in {RES / 'summary.md'}")


if __name__ == "__main__":
    main()
