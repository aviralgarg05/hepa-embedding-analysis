#!/usr/bin/env python3
"""
Multi-seed version of the MLP predictor sweep (report section 5, figure 4b).

analyze.py runs that sweep at one seed; this re-runs it at 5 seeds and writes
mean +- std to out/multiseed.json, so the one caveat the report declares about
itself can be removed. Everything else (ridge sweep, geometry, SIGReg, probes)
is closed-form or deterministic and does not need reseeding.

Run:  ./.venv/bin/python code/multiseed.py --npz /path/to/embeddings.npz
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import (nz, retrieval, var_entropy_rank, rankme, sigreg,
                     sigreg_torch, whiten)

ALPHAS = (0.0, 0.01, 0.05, 0.1, 0.3, 0.6, 0.9)
SEEDS = (0, 1, 2, 3, 4)
EPOCHS = 200
HID = 1024


def run_one(T, I_tgt, tr, te, alpha, seed):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    Xtr = torch.tensor(T[tr]); Ytr = torch.tensor(I_tgt[tr]); Xte = torch.tensor(T[te])
    net = torch.nn.Sequential(
        torch.nn.Linear(T.shape[1], HID), torch.nn.GELU(),
        torch.nn.Linear(HID, I_tgt.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
    for _ in range(EPOCHS):
        perm = torch.randperm(len(Xtr), generator=g)
        for b in range(0, len(Xtr), 256):
            ix = perm[b:b + 256]
            p = net(Xtr[ix])
            l1 = (p - Ytr[ix]).abs().mean()
            sg = sigreg_torch(p, gen=g) if alpha > 0 else torch.zeros((), dtype=l1.dtype)
            loss = (1 - alpha) * l1 + alpha * sg
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        P = net(Xte).numpy()
    r1, r5, _ = retrieval(P, I_tgt[te])
    return {"cos": float((nz(P) * nz(I_tgt[te])).sum(1).mean()), "R@1": r1, "R@5": r5,
            "rank": var_entropy_rank(P), "rankme": rankme(P), "sigreg_raw": sigreg(P, mode="raw")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.expanduser("~/Downloads/embeddings.npz"))
    ap.add_argument("--out", default="out/multiseed.json")
    a = ap.parse_args()

    d = np.load(a.npz)
    T, I = d["T"].astype(np.float32), d["I"].astype(np.float32)
    N = len(T)

    # identical split to analyze.py: default_rng(0).permutation after the calls
    # analyze.py makes before it -- so reuse the split it cached.
    import pickle
    cache = pickle.load(open("out/_cache.pkl", "rb"))
    tr, te = np.array(cache["_split_tr"]), np.array(cache["_split_te"])
    assert len(tr) == 2000 and len(te) == 500

    out, total = {}, len(ALPHAS) * len(SEEDS) * 2
    done = 0
    for tname, Itgt in (("raw", I), ("whitened", whiten(I).astype(np.float32))):
        for alpha in ALPHAS:
            runs = [run_one(T, Itgt, tr, te, alpha, s) for s in SEEDS]
            done += len(SEEDS)
            agg = {}
            for k in runs[0]:
                v = np.array([r[k] for r in runs])
                agg[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                          "min": float(v.min()), "max": float(v.max())}
            out[f"{tname}|{alpha}"] = agg
            print(f"[{done:3d}/{total}] {tname:9s} a={alpha:<5} "
                  f"cos={agg['cos']['mean']:+.4f}+-{agg['cos']['std']:.4f}  "
                  f"rank={agg['rank']['mean']:6.1f}+-{agg['rank']['std']:.1f}  "
                  f"R@5={agg['R@5']['mean']:.4f}+-{agg['R@5']['std']:.4f}", flush=True)
        out[f"target_rank_{tname}"] = var_entropy_rank(Itgt[te])

    out["_meta"] = {"seeds": list(SEEDS), "alphas": list(ALPHAS), "epochs": EPOCHS, "hid": HID}
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
