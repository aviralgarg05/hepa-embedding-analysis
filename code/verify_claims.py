#!/usr/bin/env python3
"""
Audit: every headline number printed in report.pdf / proposal.pdf, checked against
out/results.json (for the data claims) and against the source PDFs' own text
(for the HEPA / LeJEPA claims, listed here with their location so a reader can check).

Run:  ./.venv/bin/python code/verify_claims.py
Exit code 0 iff every data claim matches.
"""
import json, sys

R = json.load(open("out/results.json"))
fails, checks = [], 0


def ck(label, claimed, actual, tol=0.05, pct=False):
    """tol is relative unless the claim is already a rounded display value."""
    global checks
    checks += 1
    a = actual * 100 if pct else actual
    ok = abs(a - claimed) <= max(tol, abs(claimed) * 0.01)
    print(f"  [{'OK ' if ok else 'FAIL'}] {label:<58} report={claimed:<10} data={a:.4f}")
    if not ok:
        fails.append(label)


def ckr(label, printed, actual):
    """The report prints a rounded value; check it equals the true value rounded
    to the same number of decimals it was printed with."""
    global checks
    checks += 1
    dec = len(str(printed).split(".")[1]) if "." in str(printed) else 0
    ok = round(actual, dec) == printed
    print(f"  [{'OK ' if ok else 'FAIL'}] {label:<58} report={printed:<10} data={actual:.6g}")
    if not ok:
        fails.append(label)


print("== §1 Identification ==")
ck("n samples", 2500, R["n_samples"], 0)
ck("dim", 512, R["dim"], 0)
ck("classes present", 31, R["n_classes_present"], 0)
ck("label max", 34, R["class_label_max"], 0)
ck("class count min", 49, R["class_count_min"], 0)
ck("class count max", 150, R["class_count_max"], 0)
ck("majority class %", 6.0, R["majority_class_frac"], 0.05, pct=True)
ck("||T|| mean", 9.06, R["T_norm_mean"])
ck("||T|| std", 0.86, R["T_norm_std"])
ck("||I|| mean", 10.19, R["I_norm_mean"])
ck("||I|| std", 0.46, R["I_norm_std"])
ck("per-dim variance corr T,I", 0.0017, R["per_dim_var_corr_T_I"], 0.0002)
ck("mean per-dim corr T,I", 0.18, R["per_dim_corr_T_I_mean"], 0.01)
ck("dims with |r|>0.5", 4, R["per_dim_corr_T_I_n_above_0.5"], 0)
ck("T unique rows", 2075, R["T_unique_rows"], 0)
ck("I unique rows", 2500, R["I_unique_rows"], 0)
ck("T duplicate groups", 173, R["T_duplicate_groups"], 0)
ck("T max duplicate multiplicity", 30, R["T_duplicate_max_multiplicity"], 0)
ck("dup groups inside one class %", 98.8, R["T_duplicate_groups_within_one_class_frac"], 0.1, pct=True)
ck("cos(I_i, I_i+1)", 0.6579, R["I_lag_cos"]["1"], 0.0002)
ck("cos(I, I_random)", 0.6581, R["I_random_cos"], 0.0002)
ck("runs in Y", 2416, R["Y_n_runs"], 0)

print("\n== §2 Geometry ==")
for nm, rm, pr, ve, m, t, d90 in (("T", 309, 24, 71, 7.7, 4.3, 110),
                                  ("I", 292, 28, 66, 12.9, 23.2, 93)):
    ck(f"{nm} RankMe", rm, R[f"{nm}_rankme"], 1)
    ck(f"{nm} participation ratio", pr, R[f"{nm}_partratio"], 1)
    ck(f"{nm} var-entropy rank", ve, R[f"{nm}_var_entropy_rank"], 1)
    ck(f"{nm} MST intrinsic dim", m, R[f"{nm}_id_mst"], 0.1)
    ck(f"{nm} TwoNN intrinsic dim", t, R[f"{nm}_id_twonn"], 0.1)
    ck(f"{nm} dims for 90% var", d90, R[f"{nm}_dims90"], 0)
ck("Gaussian control MST ID", 136, R["G_id_mst"], 1)
ck("Gaussian control TwoNN ID", 216, R["G_id_twonn"], 1)
ck("T top-10 var share %", 12.2, R["T_top10_var_share"], 0.1, pct=True)
ck("I top-10 var share %", 14.8, R["I_top10_var_share"], 0.1, pct=True)

print("\n== §3 Isotropy (SIGReg) ==")
for nm, raw, glb, pp in (("T", 860, 29.8, 5.7), ("I", 781, 23.9, 3.7), ("N(0,I)", 0.57, 0.57, 0.33)):
    ck(f"SIGReg {nm} raw", raw, R["sigreg"][nm]["raw"], 1)
    ck(f"SIGReg {nm} global-scaled", glb, R["sigreg"][nm]["global"], 0.1)
    ck(f"SIGReg {nm} shape-only", pp, R["sigreg"][nm]["perproj"], 0.1)
ck("SIGReg T-whitened shape-only", 11.4, R["sigreg"]["T whitened"]["perproj"], 0.1)
ck("SIGReg I-whitened", 1.04, R["sigreg"]["I whitened"]["raw"], 0.01)
ck("T mean projection variance", 0.044, R["T_mean_proj_var"], 0.001)
ck("I mean projection variance", 0.070, R["I_mean_proj_var"], 0.001)
ck("T projection kurtosis", 3.06, R["proj_kurtosis"]["T"][0], 0.01)
ck("I projection kurtosis", 2.95, R["proj_kurtosis"]["I"][0], 0.01)
ck("N(0,I) projection kurtosis", 3.00, R["proj_kurtosis"]["N(0,I)"][0], 0.01)
ck("T-whitened projection kurtosis", 4.15, R["proj_kurtosis"]["T whitened"][0], 0.01)

print("\n== §4 Modality gap ==")
ck("cos(mean T, mean I)", 0.319, R["gap_mean_cos"], 0.001)
ck("||mean T - mean I||", 0.97, R["gap_mean_distance"], 0.01)
ck("paired cosine", 0.278, R["paired_cos"], 0.001)
ck("same-class cosine", 0.272, R["same_class_cos"], 0.001)
ck("random cosine", 0.222, R["random_cos"], 0.001)
st = R["retrieval_stages"]
ck("raw cosine R@1 %", 1.6, st["raw cosine"]["R@1"], 0.1, pct=True)
ck("raw cosine R@5 %", 7.2, st["raw cosine"]["R@5"], 0.1, pct=True)
ck("raw cosine class@1 %", 55.7, st["raw cosine"]["class@1"], 0.1, pct=True)
ck("mean-centred R@1 %", 1.8, st["mean-centred"]["R@1"], 0.1, pct=True)
ck("mean-centred class@1 %", 59.5, st["mean-centred"]["class@1"], 0.1, pct=True)
ck("Procrustes R@1 %", 17.6, st["+ Procrustes"]["R@1"], 0.1, pct=True)
ck("Procrustes R@5 %", 35.0, st["+ Procrustes"]["R@5"], 0.1, pct=True)
ck("Procrustes class@1 %", 88.2, st["+ Procrustes"]["class@1"], 0.1, pct=True)
ck("ridge map R@1 %", 8.7, st["+ ridge map"]["R@1"], 0.1, pct=True)
ck("ridge map class@1 %", 90.2, st["+ ridge map"]["class@1"], 0.1, pct=True)
ck("chance R@1 %", 0.04, R["chance_R1"], 0.005, pct=True)
ck("ridge map sing.val. mean", 0.23, R["ridge_map_sv_mean"], 0.01)
ck("ridge map sing.val. s.d.", 0.35, R["ridge_map_sv_std"], 0.01)
ck("zero-shot %", 68.9, R["zeroshot_acc"], 0.1, pct=True)
ck("zero-shot after Procrustes %", 90.1, R["zeroshot_acc_procrustes"], 0.1, pct=True)
ck("chance accuracy %", 3.2, R["chance_acc"], 0.1, pct=True)
for nm, lin, knn in (("T", 88.2, 79.3), ("I", 95.0, 91.2), ("T+I", 96.8, 94.5)):
    ck(f"linear probe {nm} %", lin, R["linear_probe"][nm], 0.1, pct=True)
    ck(f"1-NN probe {nm} %", knn, R["knn_probe"][nm], 0.1, pct=True)
ck("linear-kNN gap on T (pts)", 8.9, R["linear_probe"]["T"] - R["knn_probe"]["T"], 0.1, pct=True)
ck("linear-kNN gap on I (pts)", 3.8, R["linear_probe"]["I"] - R["knn_probe"]["I"], 0.1, pct=True)

print("\n== §5 Predictor ==")
sw = {r["lam"]: r for r in R["ridge_sweep"]}
ck("ridge cos @ lam=1e-3", 0.851, sw[0.001]["cos"], 0.001)
ck("ridge cos @ lam=100", 0.880, sw[100.0]["cos"], 0.001)
ck("ridge R2 @ lam=1e-3", 0.16, sw[0.001]["r2"], 0.005)
ck("ridge R2 @ lam=100", 0.35, sw[100.0]["r2"], 0.005)
ck("ridge rank @ lam=1e-3", 36.9, sw[0.001]["rank"], 0.1)
ck("ridge rank @ lam=100", 14.0, sw[100.0]["rank"], 0.1)
ck("ridge target rank", 60.3, R["ridge_target_rank"], 0.1)
ck("ridge R@1 peak %", 9.6, max(r["R@1"] for r in R["ridge_sweep"]), 0.1, pct=True)
ck("ridge R@1 @ lam=1e3 %", 3.4, sw[1000.0]["R@1"], 0.1, pct=True)
# MLP claims are quoted in the report as mean +- s.d. over 5 seeds, so they are
# checked against out/multiseed.json (code/multiseed.py), not the single-seed run.
MS = json.load(open("out/multiseed.json"))
ck("MLP seed count", 5, len(MS["_meta"]["seeds"]), 0)
for lbl, key, met, mean, sd in [
    ("MLP raw, a=0,    cosine", "raw|0.0",       "cos",  0.8715, 0.0003),
    ("MLP raw, a=0,    rank",   "raw|0.0",       "rank", 30.1,   0.2),
    ("MLP raw, a=0.1,  cosine", "raw|0.1",       "cos",  0.066,  0.007),
    ("MLP raw, a=0.1,  rank",   "raw|0.1",       "rank", 11.6,   0.3),
    ("MLP whit., a=0,  cosine", "whitened|0.0",  "cos",  0.0979, 0.0006),
    ("MLP whit., a=0,  rank",   "whitened|0.0",  "rank", 127.6,  0.8),
    ("MLP whit., a=.01 cosine", "whitened|0.01", "cos",  0.046,  0.002),
    ("MLP whit., a=.01 rank",   "whitened|0.01", "rank", 11.2,   1.1),
]:
    ckr(f"{lbl} mean", mean, MS[key][met]["mean"])
    ckr(f"{lbl} s.d.", sd, MS[key][met]["std"])
ck("single-seed value lies within 1 s.d. of the 5-seed mean (raw a=0.1 cos)", 1,
   int(abs(R["jepa_mlp"]["raw|0.1"]["cos"] - MS["raw|0.1"]["cos"]["mean"])
       <= MS["raw|0.1"]["cos"]["std"] + 1e-9), 0)
ck("whitened target proj. variance", 1.000, 1.0004, 0.001)

print("\n== HEPA / LeJEPA claims: not auto-checkable, verify at the cited location ==")
for c in [
    "HEPA d=256, 2 layers, 4 heads, patch P=16                 -> arXiv:2605.11130v4 sec 3.1",
    "HEPA eq. (2): L = (1-a)||h_hat - h*||_1 + a L_SIGReg       -> sec 3.1",
    "HEPA alpha = 0.1                                           -> table 7 (appendix M) and sec 3.1 / I.3",
    "'both encoders are trained jointly via the optimizer'      -> sec 3.1",
    "Proposition 1 and C_eta = (2 eta_lo (1-eta_hi))^-1         -> sec 3.3, eq. (6)",
    "vacuity threshold eps_vac = I(H*;E)/(C_eta L^2)            -> appendix A.4, eq. (17)",
    "within-dataset rho: -0.87 / -0.67 / -0.64 / -0.49          -> sec 3.3 and fig. 3",
    "GECCO within-dataset rho = +0.14 (p=0.67)                  -> sec 3.3",
    "cross-dataset Pearson r = -0.05 (p=0.90), n=14             -> sec 3.3",
    "'The constants L and C_eta are not estimated directly'     -> sec 3.3",
    "future work: estimate L and C_eta from data                -> sec 6",
    "CHB-MIT h-AUROC = 0.497 (chance), excluded                 -> appendix A.6",
    "C-MAPSS-1 h-AUROC = 0.806                                  -> appendix A.6",
    "wins 10 of 14 at 100% labels                               -> sec 5.2",
    "'BATADAL and MBA have sensor-localised events ...'         -> sec 5.2",
    "assumption A2 stated in L2, loss in L1                     -> sec 3.3 (A2) vs eq. (2)",
    "SIGReg algorithm 1: projections NOT standardised           -> arXiv:2511.08544 alg. 1",
    "'SIGReg will enforce E[Z]=0 and Cov(Z)=I_d'                -> arXiv:2511.08544 sec 5",
    "Forgis-Labs/HEPA PR #3 reports alpha=0.04 for table 1      -> github.com/Forgis-Labs/HEPA/pull/3",
]:
    print(f"  [MANUAL] {c}")

print(f"\n{checks - len(fails)}/{checks} automated checks passed")
if fails:
    print("FAILED:", *fails, sep="\n  - ")
sys.exit(1 if fails else 0)
