**To:** Pratik Jawahar <pratik.jawahar@forgis.com>
**Subject:** Screening task — report and proposal

---

Hi Pratik,

Thanks for the task, it was a good one to sit with. Report and proposal are attached.

The report is on `embeddings.npz`. I've written it up as a CLIP-style dual encoder, with `T` as text, `I` as image and `Y` a class label, and I've put the evidence for and against it in the report rather than stating it flatly, since the decisive test needs the source data. The main finding is that the modality gap is a rotation for class structure and barely one for instances: an orthogonal Procrustes map fitted on 2000 rows takes held-out class match from 55.4% to 84.6%, while instance R@1 moves only from 1.6% to 3.2%.

The proposal builds on Proposition 1. You validate the bound within a dataset but report r = −0.05 across the 14, and say L and C_η aren't estimated directly. I want to estimate them and test whether the normalised slack recovers the ordering that raw ε loses, with the controls and the failure modes I can already see written in.

The code is at github.com/aviralgarg05/hepa-embedding-analysis if you want to re-run any of it. There's a script in there that re-checks every number printed in the two PDFs against the saved results.

One question: the paper gives α = 0.1 in table 7, while PR #3 on the repo says the table-1 runs used 0.04. I've worked from 0.1 throughout.

Best,
Aviral
