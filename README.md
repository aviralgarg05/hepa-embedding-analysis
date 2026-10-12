# Forgis screening task, Aviral Garg

Two deliverables, plus the code that produces every number in them.

| File | What it is |
|---|---|
| `report.pdf` | **2 pages.** What is in `embeddings.npz`, what I did, what I found. Bullets and plots only. |
| `proposal.pdf` | **1 page.** A research proposal extending HEPA (Petersen et al., [arXiv:2605.11130](https://arxiv.org/abs/2605.11130)), stated as falsifiable hypotheses. |

## Reproducing

```bash
python3 -m venv .venv && ./.venv/bin/pip install matplotlib numpy scipy torch
./.venv/bin/python code/analyze.py    --npz /path/to/embeddings.npz   # ~4 min
./.venv/bin/python code/multiseed.py  --npz /path/to/embeddings.npz   # ~25 min, 5 seeds
```

Runs in about 4 minutes on CPU. Seeded (`SEED = 0`); no sklearn. Writes:

- `out/results.json`: every computed quantity
- `figs/fig1..4.pdf`: the four figures used in the report
- `out/_cache.pkl`: lets `--figs-only` redraw figures without recomputing
- `out/multiseed.json`: the section-5 MLP sweep at 5 seeds (mean/std/min/max). When present,
  figure 4b is drawn with error bars.

Then:

```bash
./.venv/bin/python code/verify_claims.py   # 112/112 checks, exit 0
tectonic -X compile report.tex   --outdir out
tectonic -X compile proposal.tex --outdir out
```

`code/verify_claims.py` re-checks **every headline number printed in the two PDFs** against
`out/results.json`, and lists the HEPA/LeJEPA claims that are not auto-checkable together with the
section where each can be verified in the source papers.

## What the data turned out to be

`T (2500,512) float32`, `I (2500,512) float32`, `Y (2500,) int64`. 31 of 34 classes are present.

Working hypothesis: a CLIP-style dual encoder, `T` = text, `I` = image, `Y` = class label. Reported in
the report as a hypothesis, with the evidence for and against it laid out. The decisive
test (encoding known image/caption pairs with ViT-B/32) needs the source data, which I did not have.

It is **not** a HEPA representation dump. Three falsification tests, all of them negative (`report.pdf` §1).

## Protocol

One 80/20 split at seed 0. Every fitted object in report §4 and §5 — the centring, the Procrustes rotation,
the ridge map, the class prototypes, the probes, the MLP — is fitted on the 2000 training rows and scored on
the 500 held out. Retrieval queries the 500 held-out rows against the full 2500-row gallery, so chance R@1
stays 1/2500. Report §2 and §3 are descriptive statistics of all 2500 rows and need no split.

## Headline findings

1. **The modality gap is a rotation for class structure, not for instances.** One orthogonal Procrustes map
   lifts held-out I→T class match from 55.4% to 84.6% and zero-shot from 68.4% to 89.0%, but instance R@1 only
   from 1.6% to 3.2% (80× chance). That split is the finding: the row pairing is class-level to begin with.
2. **Cosine improves while nothing downstream does.** A ridge predictor's cosine *rises* 0.848 → 0.882 while
   its prediction rank *falls* 37.0 → 14.2 against a target rank of 58.9 — and held-out retrieval stays flat
   at ~8% R@1 across that whole range. Cosine buys nothing; rank tracks the collapse but not the task.
3. **On a frozen target, SIGReg fights the prediction term.** HEPA's target is never frozen — their appendix
   I.3 shows the method is robust to an EMA target too, and picks joint training "for its simplicity" — so
   this is a regime HEPA has not tested, not a contradiction of it. That gap is where the proposal starts.

## Two corrections worth flagging

**The §4 numbers were in-sample in an earlier version.** The Procrustes map, the ridge map and the class
prototypes were fitted on all 2500 rows and then evaluated on the same 2500 rows. A 512×512 orthogonal map has
~131k free parameters against 2500 pairs, and the inflation was about 5× at instance level: the headline read
"1.6% → 17.6%" where the held-out number is 1.6% → 3.2%. The class-level result survived (88.2% in-sample vs
84.6% held-out), which is why the finding stands but in a narrower form.

**My first SIGReg implementation was not SIGReg.** It standardised each random projection before the
Epps–Pulley statistic.
That silently removes the scale and covariance information, which is exactly the part that makes SIGReg
prevent collapse (LeJEPA, [arXiv:2511.08544](https://arxiv.org/abs/2511.08544), algorithm 1 does *not*
standardise; §5 of that paper: "SIGReg will enforce in expectation that E[Z]=0 and Cov(Z)=I_d"). With the
standardisation in place, α had almost no effect and the result looked clean and meant nothing. The
numbers in the report come from a faithful transcription of algorithm 1, and the finding reversed.

## Open question

The paper reports α = 0.1 (table 7, and §3.1/§I.3), while `Forgis-Labs/HEPA` PR #3 states the table-1
runs used 0.04. I have used 0.1 throughout, since that is what the paper says. Worth confirming.
