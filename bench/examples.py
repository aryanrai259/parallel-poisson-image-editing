"""Photo examples and their placement arguments (from the upstream README)."""

from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"

# name: (source, mask or None, target, h0, w0, h1, w1, gradient, iterations)
EXAMPLES = {
    "test1": ("test1_src.jpg", "test1_mask.jpg", "test1_tgt.jpg", 0, 0, -150, -50, "max", 5000),
    "test2": ("test2_src.png", "test2_mask.png", "test2_tgt.png", 0, 0, 130, 130, "src", 5000),
    "test3": ("test3_src.jpg", "test3_mask.jpg", "test3_tgt.jpg", 0, 0, 100, 100, "max", 5000),
    "test4": ("test4_src.jpg", "test4_mask.jpg", "test4_tgt.jpg", 0, 0, 100, 100, "max", 5000),
    "test5": ("test5_src.jpg", "test5_mask.png", "test5_tgt.jpg", -70, 0, 50, 0, "max", 5000),
    "test6": ("test6_src.png", "test6_mask.png", "test6_tgt.png", 0, 0, 50, 0, "max", 5000),
    "test7": ("test7_src.jpg", None, "test7_tgt.jpg", 0, 0, 50, 30, "max", 5000),
    "test8": ("test8_src.jpg", None, "test8_tgt.jpg", 0, 0, 90, 90, "max", 10000),
}


def available() -> dict:
    """Examples whose files exist in bench/data."""
    out = {}
    for name, ex in EXAMPLES.items():
        files = [ex[0], ex[2]] + ([ex[1]] if ex[1] else [])
        if all((DATA / f).exists() and (DATA / f).stat().st_size > 0 for f in files):
            out[name] = ex
    return out


def cli_args(name: str) -> list:
    """fpie CLI arguments for an example (without backend/method/-n/-o)."""
    src, mask, tgt, h0, w0, h1, w1, grad, _ = EXAMPLES[name]
    args = ["-s", str(DATA / src), "-t", str(DATA / tgt)]
    if mask:
        args += ["-m", str(DATA / mask)]
    args += ["-h0", str(h0), "-w0", str(w0), "-h1", str(h1), "-w1", str(w1), "-g", grad]
    return args
