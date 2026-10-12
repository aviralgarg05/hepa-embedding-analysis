#!/usr/bin/env python3
"""
Substitute the 5-seed MLP results into report.tex and proposal.tex.

The two documents carry placeholders (MSCOS0, MSR1, PCOS0, ...) so the prose can
be written and laid out before the sweep finishes. This fills them from
out/multiseed.json and prints what it wrote, so the numbers in the PDFs can only
ever come from the results file.

Run:  ./.venv/bin/python code/fill_mlp_numbers.py
"""
import json, re, sys

MS = json.load(open("out/multiseed.json"))


def fmt(key, metric, dec):
    """mean +- s.d. as LaTeX, rounded to `dec` decimals."""
    m, s = MS[key][metric]["mean"], MS[key][metric]["std"]
    return f"${m:.{dec}f}{{\\pm}}{s:.{dec}f}$"


SUBS = {
    # report §5, raw target
    "MSCOS0": fmt("raw|0.0", "cos", 4),
    "MSR0":   fmt("raw|0.0", "rank", 1),
    "MSCOS1": fmt("raw|0.1", "cos", 4),
    "MSR1":   fmt("raw|0.1", "rank", 1),
    # report §5, whitened-target control
    "MSWC0":  fmt("whitened|0.0", "cos", 4),
    "MSWC1":  fmt("whitened|0.01", "cos", 4),
    "MSWR0":  fmt("whitened|0.0", "rank", 1),
    "MSWR1":  fmt("whitened|0.01", "rank", 1),
    # proposal, secondary direction
    "PCOS0":  fmt("raw|0.0", "cos", 4),
    "PCOS1":  fmt("raw|0.1", "cos", 4),
}

total = 0
for path in ("report.tex", "proposal.tex"):
    s = open(path).read()
    n = 0
    for k, v in SUBS.items():
        c = len(re.findall(rf"\b{k}\b", s))
        if c:
            s = re.sub(rf"\b{k}\b", v.replace("\\", "\\\\"), s)
            n += c
    open(path, "w").write(s)
    print(f"  {path}: {n} placeholders filled")
    total += n

left = []
for path in ("report.tex", "proposal.tex"):
    left += re.findall(r"\b(?:MS|P)(?:COS|R|WC|WR)\d\b", open(path).read())
print(f"\n  filled {total}; {len(left)} placeholders remaining {left}")
print(f"  seeds: {MS['_meta']['seeds']}")
for k, v in SUBS.items():
    print(f"    {k:7s} -> {v}")
sys.exit(1 if left else 0)
