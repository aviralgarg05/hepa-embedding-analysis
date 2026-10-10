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

## Headline findings

1. **The modality gap is largely a rotation.** One orthogonal Procrustes map lifts I→T instance R@1 from
   1.6% to 17.6%, class match from 55.7% to 88.2%, and zero-shot accuracy from 68.9% to 90.1%.
2. **Cosine hides the collapse.** A ridge predictor's cosine to the target *rises* 0.851 → 0.880 while its
   prediction rank *falls* 36.9 → 14.0 against a target rank of 60.3. They move in opposite directions across
   the whole useful range of λ.
3. **On a frozen target, SIGReg fights the prediction term.** At HEPA's own α=0.1 the cosine drops from
   0.8715 ± 0.0003 to 0.066 ± 0.007 (5 seeds). HEPA avoids this only because it trains both encoders jointly,
   which lets the target migrate toward the isotropic Gaussian SIGReg demands. That's where the proposal starts.

## A correction worth flagging

My first SIGReg implementation standardised each random projection before the Epps–Pulley statistic.
That silently removes the scale and covariance information, which is exactly the part that makes SIGReg
prevent collapse (LeJEPA, [arXiv:2511.08544](https://arxiv.org/abs/2511.08544), algorithm 1 does *not*
standardise; §5 of that paper: "SIGReg will enforce in expectation that E[Z]=0 and Cov(Z)=I_d"). With the
standardisation in place, α had almost no effect and the result looked clean and meant nothing. The
numbers in the report come from a faithful transcription of algorithm 1, and the finding reversed.

## Open question

The paper reports α = 0.1 (table 7, and §3.1/§I.3), while `Forgis-Labs/HEPA` PR #3 states the table-1
runs used 0.04. I have used 0.1 throughout, since that is what the paper says. Worth confirming.
