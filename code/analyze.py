#!/usr/bin/env python3
"""
Forgis screening task -- analysis of embeddings.npz

Reproducible end-to-end: writes out/results.json and figs/*.pdf.
Dependencies: numpy, scipy, torch, matplotlib. No sklearn.

Run:  ./.venv/bin/python code/analyze.py --npz /path/to/embeddings.npz
"""
import argparse, json, os, pickle, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import minimum_spanning_tree
import torch

SEED = 0
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

# ----------------------------------------------------------------------------- style
plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.015,
    "lines.linewidth": 1.1, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})
CT, CI, CG = "#2d6a9f", "#c1452b", "#8a8a8a"   # T, I, reference

R = {}            # results dict
def rec(k, v):
    R[k] = v
    return v

# ----------------------------------------------------------------------------- utils
def nz(X, eps=1e-12):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)

def spectrum(Z):
    """Singular values of the mean-centred matrix."""
    return np.linalg.svd(Z - Z.mean(0), compute_uv=False)

def rankme(Z):
    """RankMe (Garrido et al. 2022): entropy of L1-normalised singular values."""
    s = spectrum(Z); p = s / (np.abs(s).sum() + 1e-12); p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))

def participation_ratio(Z):
    lam = spectrum(Z) ** 2
    return float(lam.sum() ** 2 / (lam ** 2).sum())

def var_entropy_rank(Z):
    """Entropy of the *variance* spectrum (a stricter soft rank than RankMe)."""
    lam = spectrum(Z) ** 2; p = lam / lam.sum(); p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))

def dims_for_var(Z, frac):
    lam = spectrum(Z) ** 2
    return int(np.searchsorted(np.cumsum(lam) / lam.sum(), frac) + 1)

def twonn_id(Zn, discard=0.10):
    """TwoNN intrinsic dimension (Facco et al. 2017) on cosine geometry."""
    S = Zn @ Zn.T; np.fill_diagonal(S, -9.0)
    D = np.sqrt(np.maximum(2 - 2 * S, 1e-12)); np.fill_diagonal(D, np.inf)
    srt = np.sort(D, axis=1)
    mu = srt[:, 1] / np.maximum(srt[:, 0], 1e-12)
    mu = np.sort(mu[np.isfinite(mu) & (mu > 1)])
    k = int((1 - discard) * len(mu)); mu = mu[:k]
    F = np.arange(1, k + 1) / len(mu)
    x, y = np.log(mu), -np.log(np.clip(1 - F, 1e-12, None))
    return float((x @ y) / (x @ x))

def mst_id(Z, sizes=(300, 500, 800, 1200, 1800, 2500), reps=3):
    """
    MST-based intrinsic dimension. For n points on a d-manifold the total
    Euclidean MST edge length scales as L_n ~ n^((d-1)/d), so a log-log slope
    b gives d = 1/(1-b). Preferred over TwoNN for SSL features (IdEst, 2026).
    """
    N = len(Z); Ls, ns = [], []
    for n in sizes:
        n = min(n, N)
        for r in range(reps):
            idx = np.random.default_rng(1000 * n + r).choice(N, n, replace=False)
            A = Z[idx]
            D = np.sqrt(np.maximum(
                (A * A).sum(1)[:, None] + (A * A).sum(1)[None, :] - 2 * A @ A.T, 0))
            Ls.append(float(minimum_spanning_tree(D).sum())); ns.append(n)
    x, y = np.log(np.array(ns)), np.log(np.array(Ls))
    b = np.polyfit(x, y, 1)[0]
    return float(1.0 / max(1e-6, 1.0 - b))

def sigreg(Z, n_dir=1024, n_quad=17, seed=0, mode="raw", chunk=128):
    """
    SIGReg with the Epps-Pulley statistic, following algorithm 1 of LeJEPA
    (Balestriero & LeCun, arXiv:2511.08544) exactly: random unit directions,
    *unstandardised* projections, theoretical CF exp(-t^2/2), Gaussian window
    exp(-t^2/2), trapezoidal quadrature on [-5,5] with 17 knots, times N, then
    averaged over directions (def. 2).

    Because the projections are not rescaled, the statistic is sensitive to the
    full departure from N(0, I_d) -- LeJEPA notes SIGReg "will enforce in
    expectation that E[Z]=0 and Cov(Z)=I_d". We therefore report it at three
    levels to separate the sources of that departure:
      mode="raw"      : as the loss sees it (global scale + anisotropy + shape)
      mode="global"   : after one global rescale to unit mean projection
                        variance   (anisotropy + shape)
      mode="perproj"  : after standardising every projection (shape only)
    """
    g = np.random.default_rng(seed)
    A = g.standard_normal((Z.shape[1], n_dir)); A /= np.linalg.norm(A, axis=0, keepdims=True)
    P = (Z - Z.mean(0)) @ A
    if mode == "global":
        P = P / np.sqrt(P.var(0).mean())
    elif mode == "perproj":
        P = P / (P.std(0) + 1e-12)
    elif mode != "raw":
        raise ValueError(mode)
    t = np.linspace(-5, 5, n_quad)
    phi = np.exp(-0.5 * t ** 2)                     # CF of N(0,1) == window
    stats = np.empty(n_dir)
    for s in range(0, n_dir, chunk):
        Pc = P[:, s:s + chunk]
        ecf = np.exp(1j * (Pc[:, :, None] * t[None, None, :])).mean(0)   # (m, T)
        err = np.abs(ecf - phi[None, :]) ** 2 * phi[None, :]
        stats[s:s + chunk] = np.trapezoid(err, t, axis=1) * len(Z)
    return float(stats.mean())

def proj_moments(Z, n_dir=512, seed=0):
    g = np.random.default_rng(seed)
    Zc = Z - Z.mean(0)
    A = g.standard_normal((Z.shape[1], n_dir)); A /= np.linalg.norm(A, axis=0, keepdims=True)
    P = Zc @ A; P = (P - P.mean(0)) / (P.std(0) + 1e-12)
    return (P ** 3).mean(0), (P ** 4).mean(0)

def whiten(Z):
    Zc = Z - Z.mean(0)
    U, s, Vt = np.linalg.svd(Zc, full_matrices=False)
    return (U * np.sqrt(len(Z))) @ Vt

def ridge_fit(X, Y, lam):
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ Y)

def retrieval(Q, K, labels=None):
    """Instance R@1/R@5 and (optional) class-match@1 of query rows against key rows."""
    S = nz(Q) @ nz(K).T
    order = np.argsort(-S, axis=1)
    n = len(Q); tgt = np.arange(n)
    r1 = float((order[:, 0] == tgt).mean())
    r5 = float(np.mean([tgt[i] in order[i, :5] for i in range(n)]))
    cls = float((labels[order[:, 0]] == labels).mean()) if labels is not None else None
    return r1, r5, cls

def procrustes(A, B):
    """Orthogonal R minimising ||A R - B||_F, plus the optimal-map spectrum."""
    U, s, Vt = np.linalg.svd(A.T @ B)
    return U @ Vt, s

def knn_acc(Zn, Y, k=1):
    S = Zn @ Zn.T; np.fill_diagonal(S, -9.0)
    nb = np.argsort(-S, axis=1)[:, :k]
    if k == 1:
        return float((Y[nb[:, 0]] == Y).mean())
    out = []
    for i in range(len(Y)):
        v, c = np.unique(Y[nb[i]], return_counts=True)
        out.append(v[c.argmax()] == Y[i])
    return float(np.mean(out))

def linear_probe(X, Y, tr, te, lam=1.0):
    classes = np.unique(Y); cmap = {c: j for j, c in enumerate(classes)}
    Yi = np.array([cmap[y] for y in Y])
    Oh = np.eye(len(classes))[Yi]
    W = ridge_fit(X[tr], Oh[tr], lam)
    return float(((X[te] @ W).argmax(1) == Yi[te]).mean())


# ============================================================== 0. load + forensics
def forensics(T, I, Y):
    N, D = T.shape
    print("\n[0] FORENSICS")
    rec("n_samples", int(N)); rec("dim", int(D))
    cls, cnt = np.unique(Y, return_counts=True)
    rec("n_classes_present", int(len(cls)))
    rec("class_labels", cls.tolist())
    rec("class_label_min", int(cls.min())); rec("class_label_max", int(cls.max()))
    rec("missing_labels_in_range", sorted(set(range(int(cls.min()), int(cls.max()) + 1)) - set(cls.tolist())))
    rec("class_count_min", int(cnt.min())); rec("class_count_max", int(cnt.max()))
    for nm, X in (("T", T), ("I", I)):
        nr = np.linalg.norm(X, axis=1)
        rec(f"{nm}_norm_mean", float(nr.mean())); rec(f"{nm}_norm_std", float(nr.std()))
        rec(f"{nm}_unique_rows", int(len(np.unique(X, axis=0))))
        rec(f"{nm}_exact_zeros_frac", float((X == 0).mean()))
    # duplicate-row structure in T
    _, inv, cnts = np.unique(T, axis=0, return_inverse=True, return_counts=True)
    dup = np.where(cnts > 1)[0]
    same = [len(set(Y[np.where(inv == g)[0]].tolist())) == 1 for g in dup]
    rec("T_duplicate_groups", int(len(dup)))
    rec("T_duplicate_max_multiplicity", int(cnts.max()))
    rec("T_duplicate_groups_within_one_class_frac", float(np.mean(same)) if len(same) else None)
    # is the row order temporal?
    Tn, In = nz(T), nz(I)
    for nm, Xn in (("T", Tn), ("I", In)):
        lag = {int(L): float((Xn[:-L] * Xn[L:]).sum(1).mean()) for L in (1, 2, 5, 10, 50)}
        base = float((Xn * Xn[rng.permutation(N)]).sum(1).mean())
        rec(f"{nm}_lag_cos", lag); rec(f"{nm}_random_cos", base)
    rec("Y_n_runs", int(1 + (np.diff(Y) != 0).sum()))
    # are T and I in the same basis?
    rec("per_dim_var_corr_T_I", float(np.corrcoef(T.var(0), I.var(0))[0, 1]))
    dcorr = np.array([np.corrcoef(T[:, j], I[:, j])[0, 1] for j in range(D)])
    rec("per_dim_corr_T_I_mean", float(dcorr.mean()))
    rec("per_dim_corr_T_I_n_above_0.5", int((np.abs(dcorr) > 0.5).sum()))
    print(f"    N={N} D={D} classes={len(cls)} labels {cls.min()}..{cls.max()} "
          f"missing={R['missing_labels_in_range']}")
    print(f"    norms  T={R['T_norm_mean']:.2f}  I={R['I_norm_mean']:.2f}   (unit-norm? no)")
    print(f"    unique rows  T={R['T_unique_rows']}/{N}  I={R['I_unique_rows']}/{N}")
    print(f"    T dup groups={R['T_duplicate_groups']} (max mult {R['T_duplicate_max_multiplicity']}), "
          f"{R['T_duplicate_groups_within_one_class_frac']:.3f} inside one class")
    print(f"    temporal?  I lag1={R['I_lag_cos'][1]:.4f} vs random={R['I_random_cos']:.4f}  -> no")
    print(f"    same basis? per-dim var corr={R['per_dim_var_corr_T_I']:.4f} -> no")


# ===================================================================== 1. geometry
def geometry(T, I):
    print("\n[1] GEOMETRY / RANK")
    G = rng.standard_normal((len(T), T.shape[1])).astype(np.float32)
    for nm, X in (("T", T), ("I", I), ("G", G)):
        rec(f"{nm}_rankme", rankme(X))
        rec(f"{nm}_partratio", participation_ratio(X))
        rec(f"{nm}_var_entropy_rank", var_entropy_rank(X))
        rec(f"{nm}_dims90", dims_for_var(X, .90)); rec(f"{nm}_dims99", dims_for_var(X, .99))
        rec(f"{nm}_id_twonn", twonn_id(nz(X)))
        rec(f"{nm}_id_mst", mst_id(X))
        print(f"    {nm}: RankMe={R[f'{nm}_rankme']:6.1f}  PR={R[f'{nm}_partratio']:6.1f}  "
              f"varEnt={R[f'{nm}_var_entropy_rank']:5.1f}  dims90={R[f'{nm}_dims90']:3d}  "
              f"ID_mst={R[f'{nm}_id_mst']:5.2f}  ID_2nn={R[f'{nm}_id_twonn']:5.2f}")
    rec("spec_T", spectrum(T).tolist()); rec("spec_I", spectrum(I).tolist())
    rec("spec_G", spectrum(G).tolist())


# ===================================================================== 2. isotropy
def isotropy(T, I):
    print("\n[2] ISOTROPY  (SIGReg / Epps-Pulley, read-only, LeJEPA algorithm 1)")
    G = rng.standard_normal((len(T), T.shape[1])).astype(np.float32)
    variants = {"T": T, "I": I, "N(0,I)": G, "T whitened": whiten(T), "I whitened": whiten(I)}
    sg, ku = {}, {}
    print(f"    {'':12s} {'raw':>10s} {'global-scaled':>14s} {'shape-only':>11s}   kurtosis")
    for nm, X in variants.items():
        sg[nm] = {m: sigreg(X, mode=m) for m in ("raw", "global", "perproj")}
        skew, k4 = proj_moments(X)
        ku[nm] = (float(k4.mean()), float(k4.std()), float(np.abs(skew).mean()))
        print(f"    {nm:12s} {sg[nm]['raw']:10.3f} {sg[nm]['global']:14.3f} "
              f"{sg[nm]['perproj']:11.3f}   {ku[nm][0]:.3f}+-{ku[nm][1]:.3f} (Gauss 3)")
    rec("sigreg", sg); rec("proj_kurtosis", ku)
    _, k4T = proj_moments(T); _, k4I = proj_moments(I); _, k4G = proj_moments(G)
    rec("_kurt_hist", {"T": k4T.tolist(), "I": k4I.tolist(), "G": k4G.tolist()})
    for nm, X in (("T", T), ("I", I)):
        v = np.sort(X.var(0))[::-1]
        rec(f"{nm}_var_ratio_top_bottom", float(v[0] / v[-1]))
        rec(f"{nm}_top10_var_share", float(v[:10].sum() / v.sum()))
        rec(f"{nm}_mean_proj_var", float(np.trace(np.cov(X, rowvar=False)) / X.shape[1]))


# ================================================================ 3. modality gap
def modality_gap(T, I, Y):
    print("\n[3] MODALITY GAP")
    Tn, In = nz(T), nz(I)
    rec("paired_cos", float((Tn * In).sum(1).mean()))
    rec("random_cos", float((Tn * In[rng.permutation(len(T))]).sum(1).mean()))
    sc = []
    for c in np.unique(Y):
        ix = np.where(Y == c)[0]
        if len(ix) > 1:
            sc.append((Tn[ix] * In[ix[rng.permutation(len(ix))]]).sum(1))
    rec("same_class_cos", float(np.concatenate(sc).mean()))
    rec("gap_mean_distance", float(np.linalg.norm(Tn.mean(0) - In.mean(0))))
    rec("gap_mean_cos", float((nz(Tn.mean(0)[None]) @ nz(In.mean(0)[None]).T)[0, 0]))
    print(f"    cos paired={R['paired_cos']:.4f}  same-class={R['same_class_cos']:.4f}  "
          f"random={R['random_cos']:.4f}")
    print(f"    cos(mean T, mean I)={R['gap_mean_cos']:.4f}  ||mean diff||={R['gap_mean_distance']:.4f}")

    stages = {}
    stages["raw cosine"] = retrieval(In, Tn, Y)
    Tc, Ic = nz(Tn - Tn.mean(0)), nz(In - In.mean(0))
    stages["mean-centred"] = retrieval(Ic, Tc, Y)
    Rm, sv = procrustes(Ic, Tc)
    stages["+ Procrustes"] = retrieval(Ic @ Rm, Tc, Y)
    W = ridge_fit(Ic, Tc, 1.0)
    stages["+ ridge map"] = retrieval(Ic @ W, Tc, Y)
    rec("retrieval_stages", {k: {"R@1": v[0], "R@5": v[1], "class@1": v[2]} for k, v in stages.items()})
    for k, v in stages.items():
        print(f"    I->T  {k:14s} R@1={v[0]:.4f}  R@5={v[1]:.4f}  class@1={v[2]:.4f}")
    # how close to orthogonal is the optimal *unconstrained* linear map?
    sW = np.linalg.svd(W, compute_uv=False)
    rec("procrustes_corr_spectrum", sv[:20].tolist())
    rec("ridge_map_sv_mean", float(sW.mean())); rec("ridge_map_sv_std", float(sW.std()))
    rec("ridge_map_sv", sW.tolist())
    rec("chance_R1", 1.0 / len(T))
    rec("majority_class_frac", float(np.bincount(Y).max() / len(Y)))
    # 2-D joint PCA for the figure
    Z = np.vstack([Tn, In]); Z = Z - Z.mean(0)
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    P = (U[:, :2] * s[:2])
    rec("_pca2_T", P[:len(T)].tolist()); rec("_pca2_I", P[len(T):].tolist())
    rec("_pca2_varshare", (s[:2] ** 2 / (s ** 2).sum()).tolist())


# ============================================================== 4. ridge predictor
def ridge_predictor(T, I):
    print("\n[4] LINEAR PREDICTOR  T->I  (cosine vs rank)")
    N = len(T); idx = rng.permutation(N); tr, te = idx[:2000], idx[2000:]
    lams = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    rows = []
    tgt_rank = var_entropy_rank(I[te])
    for lam in lams:
        W = ridge_fit(T[tr], I[tr], lam)
        P = T[te] @ W
        cos = float((nz(P) * nz(I[te])).sum(1).mean())
        r2 = float(1 - ((P - I[te]) ** 2).sum() / ((I[te] - I[tr].mean(0)) ** 2).sum())
        r1, r5, _ = retrieval(P, I[te])
        rows.append({"lam": lam, "cos": cos, "r2": r2, "R@1": r1,
                     "rank": var_entropy_rank(P), "rankme": rankme(P)})
        print(f"    lam={lam:<8g} cos={cos:.4f} R2={r2:+.4f} R@1={r1:.4f} "
              f"varEntRank={rows[-1]['rank']:5.1f} (target {tgt_rank:.1f})")
    rec("ridge_sweep", rows); rec("ridge_target_rank", tgt_rank)
    rec("ridge_target_rankme", rankme(I[te]))
    # reverse direction, one setting
    W = ridge_fit(I[tr], T[tr], 1.0); P = I[te] @ W
    rec("ridge_I_to_T", {"cos": float((nz(P) * nz(T[te])).sum(1).mean()),
                         "R@1": retrieval(P, T[te])[0],
                         "rank": var_entropy_rank(P),
                         "target_rank": var_entropy_rank(T[te])})
    rec("_split_tr", tr.tolist()); rec("_split_te", te.tolist())


# ======================================================== 5. JEPA predictor + SIGReg
def sigreg_torch(Z, n_dir=256, n_quad=17, gen=None):
    """
    Differentiable SIGReg, a direct transcription of LeJEPA algorithm 1
    (single-GPU, no all_reduce). Projections are NOT standardised, so the term
    penalises mean and covariance departure as well as shape -- this is what
    makes it a collapse preventer rather than a shape matcher.
    """
    A = torch.randn(Z.shape[1], n_dir, generator=gen, device=Z.device)
    A = A / A.norm(p=2, dim=0, keepdim=True)
    t = torch.linspace(-5, 5, n_quad, device=Z.device)
    phi = torch.exp(-0.5 * t ** 2)
    xt = (Z @ A).unsqueeze(2) * t                      # (N, M, T)
    ecf = torch.exp(1j * xt).mean(0)                   # (M, T)
    err = (ecf - phi).abs().square().mul(phi)
    return (torch.trapezoid(err, t, dim=1) * Z.shape[0]).mean()

def jepa_predictor(T, I, alphas=(0.0, 0.01, 0.05, 0.1, 0.3, 0.6, 0.9), epochs=200, hid=1024):
    """
    Minimal HEPA-shaped predictor on the frozen dump: 2-layer MLP, L1 loss,
    SIGReg on the predictor output, exactly HEPA's eq. (2):
        L = (1-alpha)*||g(T) - I||_1 + alpha*SIGReg(g(T))
    alpha=0.1 is HEPA's own value (their table 7). We sweep alpha because the two
    terms have different natural scales; both raw magnitudes are logged so the
    trade-off is readable rather than hidden in the weighting.
    """
    print("\n[5] JEPA PREDICTOR  (2-layer MLP, L1 + SIGReg, alpha sweep)")
    tr = np.array(R["_split_tr"]); te = np.array(R["_split_te"])
    out = {}
    for tgt_name, Itgt in (("raw", I), ("whitened", whiten(I).astype(np.float32))):
        Xtr = torch.tensor(T[tr]); Ytr = torch.tensor(Itgt[tr]); Xte = torch.tensor(T[te])
        print(f"  -- target = {tgt_name}  (mean proj. var "
              f"{np.trace(np.cov(Itgt, rowvar=False))/Itgt.shape[1]:.4f}; SIGReg wants 1.0)")
        for alpha in alphas:
            g = torch.Generator().manual_seed(SEED); torch.manual_seed(SEED)
            net = torch.nn.Sequential(
                torch.nn.Linear(T.shape[1], hid), torch.nn.GELU(),
                torch.nn.Linear(hid, Itgt.shape[1]))
            opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
            l1_last = sig_last = 0.0
            for ep in range(epochs):
                perm = torch.randperm(len(Xtr), generator=g)
                for b in range(0, len(Xtr), 256):
                    ix = perm[b:b + 256]
                    p = net(Xtr[ix])
                    l1 = (p - Ytr[ix]).abs().mean()
                    sg = sigreg_torch(p, gen=g) if alpha > 0 else torch.zeros((), dtype=l1.dtype)
                    loss = (1 - alpha) * l1 + alpha * sg
                    opt.zero_grad(); loss.backward(); opt.step()
                    l1_last, sig_last = float(l1), float(sg)
            with torch.no_grad():
                P = net(Xte).numpy()
            r1, r5, _ = retrieval(P, Itgt[te])
            o = {"cos": float((nz(P) * nz(Itgt[te])).sum(1).mean()), "R@1": r1, "R@5": r5,
                 "rank": var_entropy_rank(P), "rankme": rankme(P),
                 "sigreg_raw": sigreg(P, mode="raw"), "sigreg_shape": sigreg(P, mode="perproj"),
                 "l1_term": l1_last, "sigreg_term": sig_last}
            out[f"{tgt_name}|{alpha}"] = o
            print(f"     alpha={alpha:<5} cos={o['cos']:+.4f} R@1={r1:.4f} R@5={r5:.4f} "
                  f"varEntRank={o['rank']:5.1f} RankMe={o['rankme']:5.1f} "
                  f"SIGReg={o['sigreg_raw']:8.3f}   [L1={l1_last:.4f} SIGRegTerm={sig_last:.4f}]")
        rec(f"jepa_target_rank_{tgt_name}", var_entropy_rank(Itgt[te]))
        rec(f"jepa_target_sigreg_{tgt_name}", sigreg(Itgt[te], mode="raw"))
    rec("jepa_mlp", out); rec("jepa_alphas", list(alphas))
    rec("jepa_target_rank", R["jepa_target_rank_raw"])


# ======================================================================= 6. probes
def probes(T, I, Y):
    print("\n[6] PROBES")
    tr = np.array(R["_split_tr"]); te = np.array(R["_split_te"])
    lin, knn = {}, {}
    for nm, X in (("T", T), ("I", I), ("T+I", np.hstack([T, I]))):
        lin[nm] = linear_probe(X, Y, tr, te)
        knn[nm] = knn_acc(nz(X), Y, k=1)
        print(f"    {nm:4s} linear={lin[nm]:.4f}  1-NN={knn[nm]:.4f}")
    rec("linear_probe", lin); rec("knn_probe", knn)
    # zero-shot: class-mean T prototype classifies I
    Tn, In = nz(T), nz(I)
    cls = np.unique(Y)
    proto = nz(np.stack([Tn[Y == c].mean(0) for c in cls]))
    rec("zeroshot_acc", float((cls[(In @ proto.T).argmax(1)] == Y).mean()))
    # same, after Procrustes alignment
    Tc, Ic = nz(Tn - Tn.mean(0)), nz(In - In.mean(0))
    Rm, _ = procrustes(Ic, Tc)
    proto_c = nz(np.stack([Tc[Y == c].mean(0) for c in cls]))
    rec("zeroshot_acc_procrustes", float((cls[(nz(Ic @ Rm) @ proto_c.T).argmax(1)] == Y).mean()))
    rec("chance_acc", float(1 / len(cls)))
    print(f"    zero-shot (T class prototype -> I): {R['zeroshot_acc']:.4f}  "
          f"after Procrustes: {R['zeroshot_acc_procrustes']:.4f}  chance={R['chance_acc']:.4f}")


# ====================================================== 7. is this a HEPA dump?
def hepa_check(T, I):
    print("\n[7] IS THIS A HEPA-STYLE (h_t, h*) DUMP?  -- three falsification tests")
    tests = {
        "dim matches HEPA d=256": {"observed": int(T.shape[1]), "expected": 256, "pass": T.shape[1] == 256},
        "row order temporal": {"observed": round(R["I_lag_cos"][1], 4),
                               "expected": f"> random ({R['I_random_cos']:.4f})",
                               "pass": R["I_lag_cos"][1] > R["I_random_cos"] + 0.02},
        "SIGReg-trained (near-isotropic)": {"observed": round(R["sigreg"]["I"]["global"], 3),
                                            "expected": f"~ N(0,I) ({R['sigreg']['N(0,I)']['global']:.3f})",
                                            "pass": R["sigreg"]["I"]["global"] < 3 * R["sigreg"]["N(0,I)"]["global"]},
    }
    rec("hepa_checks", tests)
    for k, v in tests.items():
        print(f"    [{'PASS' if v['pass'] else 'FAIL'}] {k}: observed {v['observed']}, expected {v['expected']}")
    rec("hepa_dump_verdict", "not a HEPA dump" if not any(v["pass"] for v in tests.values()) else "inconclusive")
    print(f"    verdict: {R['hepa_dump_verdict']}")


# ======================================================================= figures
def figures(T, I, Y, figdir):
    print("\n[8] FIGURES")
    os.makedirs(figdir, exist_ok=True)
    LC = dict(frameon=False, handlelength=1.4, borderpad=.2, labelspacing=.3)

    # ---------------------------------------------------------------- Fig 1
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.05), layout="constrained")
    sT, sI, sG = (np.array(R[k]) for k in ("spec_T", "spec_I", "spec_G"))
    for sp, c, l, ls in ((sT, CT, "T", "-"), (sI, CI, "I", "-"), (sG, CG, r"$\mathcal{N}(0,I)$", "--")):
        ax[0].semilogy(np.arange(1, len(sp) + 1), sp ** 2 / (sp ** 2).max(), color=c, label=l, ls=ls)
    ax[0].set_xlabel("PCA component"); ax[0].set_ylabel(r"$\lambda_k/\lambda_1$")
    ax[0].set_title("(a) Eigenspectrum"); ax[0].legend(**LC); ax[0].set_xlim(0, 512)
    ax[0].set_ylim(1e-8, 2)

    labs = ["RankMe", "Partic.\nratio", "Var-entr.\nrank", "ID\n(MST)", "ID\n(TwoNN)"]
    keys = ["rankme", "partratio", "var_entropy_rank", "id_mst", "id_twonn"]
    x = np.arange(len(keys)); w = .27
    for off, nm, c, lb in ((-w, "T", CT, "T"), (0., "I", CI, "I"), (w, "G", CG, r"$\mathcal{N}(0,I)$")):
        ax[1].bar(x + off, [R[f"{nm}_{k}"] for k in keys], w, color=c, label=lb)
    ax[1].set_yscale("log"); ax[1].set_xticks(x); ax[1].set_xticklabels(labs)
    ax[1].set_ylabel("estimated dimension"); ax[1].set_ylim(1.5, 2200)
    ax[1].axhline(512, color="k", lw=.5, ls=":")
    ax[1].text(len(keys) - .55, 560, "$D$=512", fontsize=6, ha="right")
    ax[1].set_title(r"(b) Five estimates disagree by $>$10$\times$")
    ax[1].legend(ncol=3, loc="upper center", fontsize=6, **LC)
    fig.savefig(f"{figdir}/fig1_geometry.pdf"); plt.close(fig)

    # ---------------------------------------------------------------- Fig 2
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.05), layout="constrained")
    sg = R["sigreg"]; names = [r"$\mathcal{N}(0,I)$", "I whitened", "I", "T", "T whitened"]
    keyn = ["N(0,I)", "I whitened", "I", "T", "T whitened"]
    modes = [("raw", "raw: scale+aniso+shape", "#3f3f3f"),
             ("global", "global-scaled: aniso+shape", CI),
             ("perproj", "shape only", CT)]
    yy = np.arange(len(names)); h = .26
    for j, (m, lab, c) in enumerate(modes):
        ax[0].barh(yy + (j - 1) * h, [sg[n][m] for n in keyn], h, color=c, label=lab)
    ax[0].set_yticks(yy); ax[0].set_yticklabels(names); ax[0].set_xscale("log")
    ax[0].set_xlim(.2, 1e4)
    ax[0].set_xlabel(r"SIGReg distance to $\mathcal{N}(0,I)$  (lower = closer)")
    ax[0].set_title("(a) HEPA's regulariser, read-only")
    ax[0].legend(loc="lower right", fontsize=5.8, **LC)

    kh = R["_kurt_hist"]; bins = np.linspace(2.2, 5.0, 44)
    for k, c, l in (("G", CG, r"$\mathcal{N}(0,I)$"), ("I", CI, "I"), ("T", CT, "T")):
        ax[1].hist(kh[k], bins=bins, color=c, alpha=.62, label=l)
    ax[1].axvline(3, color="k", lw=.7, ls=":")
    ax[1].set_xlabel("kurtosis of a random 1-D projection"); ax[1].set_ylabel("count")
    ax[1].set_title("(b) Projections are near-Gaussian in shape")
    ax[1].legend(**LC)
    fig.savefig(f"{figdir}/fig2_isotropy.pdf"); plt.close(fig)

    # ---------------------------------------------------------------- Fig 3
    fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.0), layout="constrained")
    PT, PI = np.array(R["_pca2_T"]), np.array(R["_pca2_I"]); vs = R["_pca2_varshare"]
    ax[0].scatter(PT[:, 0], PT[:, 1], s=1.1, c=CT, alpha=.42, lw=0, label="T")
    ax[0].scatter(PI[:, 0], PI[:, 1], s=1.1, c=CI, alpha=.42, lw=0, label="I")
    ax[0].set_xlabel(f"PC1 ({vs[0]*100:.0f}%)"); ax[0].set_ylabel(f"PC2 ({vs[1]*100:.0f}%)")
    ax[0].set_title("(a) Two disjoint cones"); ax[0].legend(markerscale=6, **LC)

    st = R["retrieval_stages"]; order = ["raw cosine", "mean-centred", "+ Procrustes", "+ ridge map"]
    sh = ["raw\ncosine", "mean-\ncentred", "+ Pro-\ncrustes", "+ ridge\nmap"]
    xx = np.arange(len(order))
    ax[1].bar(xx - .2, [st[k]["R@1"] * 100 for k in order], .4, color=CT, label="instance R@1")
    ax[1].bar(xx + .2, [st[k]["class@1"] * 100 for k in order], .4, color=CI, label="class match@1")
    for i, k in enumerate(order):
        ax[1].text(i - .2, st[k]["R@1"] * 100 + 2, f"{st[k]['R@1']*100:.1f}", ha="center", fontsize=5.4)
        ax[1].text(i + .2, st[k]["class@1"] * 100 + 2, f"{st[k]['class@1']*100:.0f}", ha="center", fontsize=5.4)
    ax[1].set_xticks(xx); ax[1].set_xticklabels(sh, fontsize=5.8)
    ax[1].set_ylabel(r"I$\to$T retrieval (%)"); ax[1].set_ylim(0, 116)
    ax[1].set_title("(b) The gap is largely a rotation")
    ax[1].legend(loc="upper left", fontsize=5.8, **LC)

    nms = ["T", "I", "T+I"]; x3 = np.arange(3)
    ax[2].bar(x3 - .2, [R["linear_probe"][n] * 100 for n in nms], .4, color=CT, label="linear probe")
    ax[2].bar(x3 + .2, [R["knn_probe"][n] * 100 for n in nms], .4, color=CI, label="1-NN probe")
    for i, n in enumerate(nms):
        ax[2].text(i - .2, R["linear_probe"][n] * 100 + 2, f"{R['linear_probe'][n]*100:.0f}", ha="center", fontsize=5.4)
        ax[2].text(i + .2, R["knn_probe"][n] * 100 + 2, f"{R['knn_probe'][n]*100:.0f}", ha="center", fontsize=5.4)
    ax[2].axhline(R["zeroshot_acc"] * 100, color="#2e7d32", ls="--", lw=.9)
    ax[2].text(-.95, R["zeroshot_acc"] * 100 + 2, f"zero-shot {R['zeroshot_acc']*100:.0f}",
               fontsize=5.6, ha="left", color="#2e7d32")
    ax[2].axhline(R["chance_acc"] * 100, color=CG, ls=":", lw=.8)
    ax[2].text(-.95, R["chance_acc"] * 100 + 2.5, f"chance {R['chance_acc']*100:.1f}",
               fontsize=5.6, ha="left", color="#555")
    ax[2].set_xticks(x3); ax[2].set_xticklabels(nms); ax[2].set_ylabel("31-class accuracy (%)")
    ax[2].set_ylim(0, 132); ax[2].set_xlim(-1.0, 2.5)
    ax[2].set_title("(c) Both carry class structure")
    ax[2].legend(loc="upper center", ncol=2, fontsize=5.6, **LC)
    fig.savefig(f"{figdir}/fig3_gap.pdf"); plt.close(fig)

    # ---------------------------------------------------------------- Fig 4
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.05), layout="constrained")
    sw = R["ridge_sweep"]; lam = [r["lam"] for r in sw]
    a0 = ax[0]; a1 = a0.twinx()
    a0.semilogx(lam, [r["cos"] for r in sw], "o-", color=CT, ms=2.6)
    a1.semilogx(lam, [r["rank"] for r in sw], "s-", color=CI, ms=2.6)
    a1.axhline(R["ridge_target_rank"], color=CG, ls=":", lw=.8)
    a1.text(lam[-1], R["ridge_target_rank"] + 2, f"target {R['ridge_target_rank']:.0f}",
            fontsize=5.8, color="#555", ha="right")
    a0.set_xlabel(r"ridge $\lambda$"); a0.set_ylabel("cosine to target", color=CT)
    a1.set_ylabel("var-entropy rank", color=CI); a1.set_ylim(0, 70)
    a0.tick_params(axis="y", colors=CT); a1.tick_params(axis="y", colors=CI)
    a1.spines["right"].set_visible(True); a1.spines["top"].set_visible(False)
    a0.set_title("(a) Cosine rises as rank collapses")

    al = R["jepa_alphas"]; xa = np.arange(len(al))
    b0 = ax[1]; b1 = b0.twinx()
    ms = None
    if os.path.exists("out/multiseed.json"):
        ms = json.load(open("out/multiseed.json"))
        nseed = len(ms["_meta"]["seeds"])
    jm = R["jepa_mlp"]
    for tn, mk, alph in (("raw", "o-", 1.0), ("whitened", "^--", .5)):
        if ms is not None:
            cm = np.array([ms[f"{tn}|{a}"]["cos"]["mean"] for a in al])
            cs = np.array([ms[f"{tn}|{a}"]["cos"]["std"] for a in al])
            rm = np.array([ms[f"{tn}|{a}"]["rank"]["mean"] for a in al])
            rs = np.array([ms[f"{tn}|{a}"]["rank"]["std"] for a in al])
            b0.errorbar(xa, cm, yerr=cs, fmt=mk, color=CT, ms=2.6, alpha=alph,
                        lw=1.1, elinewidth=.8, capsize=1.6, label=f"cosine ({tn} target)")
            b1.errorbar(xa, rm, yerr=rs, fmt=mk.replace("o", "s").replace("^", "D"),
                        color=CI, ms=2.4, alpha=alph, lw=1.1, elinewidth=.8, capsize=1.6,
                        label=f"rank ({tn} target)")
        else:
            b0.plot(xa, [jm[f"{tn}|{a}"]["cos"] for a in al], mk, color=CT, ms=2.6, alpha=alph,
                    label=f"cosine ({tn} target)")
            b1.plot(xa, [jm[f"{tn}|{a}"]["rank"] for a in al],
                    mk.replace("o", "s").replace("^", "D"), color=CI, ms=2.4, alpha=alph,
                    label=f"rank ({tn} target)")
    tr_rank = (ms["target_rank_raw"] if ms is not None else R["jepa_target_rank_raw"])
    b1.axhline(tr_rank, color=CG, ls=":", lw=.8)
    b1.text(len(al) - 1.05, tr_rank * 1.04, f"target rank {tr_rank:.0f}",
            fontsize=5.6, ha="right", color="#555")
    b0.axvline(al.index(0.1), color="k", lw=.5, ls="--")
    b0.text(al.index(0.1) + .12, .55, r"HEPA $\alpha$=0.1", fontsize=5.8, rotation=90)
    b0.set_xticks(xa); b0.set_xticklabels([str(a) for a in al], fontsize=5.8)
    b0.set_xlabel(r"SIGReg weight $\alpha$  (HEPA eq. 2)")
    b0.set_ylabel("cosine to target", color=CT); b0.set_ylim(-.05, 1.0)
    b1.set_ylabel("var-entropy rank", color=CI); b1.set_ylim(0, 140)
    b0.tick_params(axis="y", colors=CT); b1.tick_params(axis="y", colors=CI)
    b1.spines["right"].set_visible(True); b1.spines["top"].set_visible(False)
    ttl = "(b) On a frozen target, SIGReg fights prediction"
    if ms is not None:
        ttl += f"  ($\\pm$1 s.d., {nseed} seeds)"
    b0.set_title(ttl)
    h0, l0 = b0.get_legend_handles_labels(); h1, l1 = b1.get_legend_handles_labels()
    b0.legend(h0 + h1, l0 + l1, loc="upper right", fontsize=5.2, **LC)
    fig.savefig(f"{figdir}/fig4_collapse.pdf"); plt.close(fig)

    print(f"    wrote 4 figures to {figdir}")


# ========================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.expanduser("~/Downloads/embeddings.npz"))
    ap.add_argument("--out", default="out"); ap.add_argument("--figs", default="figs")
    ap.add_argument("--figs-only", action="store_true",
                    help="redraw figures from out/_cache.pkl without recomputing")
    a = ap.parse_args()
    t0 = time.time()

    d = np.load(a.npz)
    print(f"keys: {d.files}")
    T, I, Y = d["T"].astype(np.float32), d["I"].astype(np.float32), d["Y"]

    os.makedirs(a.out, exist_ok=True)
    cache = f"{a.out}/_cache.pkl"
    if a.figs_only:
        R.update(pickle.load(open(cache, "rb")))
        print(f"loaded {len(R)} cached entries from {cache}")
        figures(T, I, Y, a.figs)
        return

    forensics(T, I, Y)
    geometry(T, I)
    isotropy(T, I)
    modality_gap(T, I, Y)
    ridge_predictor(T, I)
    jepa_predictor(T, I)
    probes(T, I, Y)
    hepa_check(T, I)
    pickle.dump(R, open(cache, "wb"))
    figures(T, I, Y, a.figs)

    pub = {k: v for k, v in R.items() if not k.startswith("_")}
    with open(f"{a.out}/results.json", "w") as f:
        json.dump(pub, f, indent=2, sort_keys=True)
    print(f"\nwrote {a.out}/results.json  ({len(pub)} entries)  in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
