# ERA V5 · Session 1 — Four Proofs

> A single, self-contained web app that **proves the four core claims of Session 1
> ("From Neural Networks to the Transformer")** by training tiny neural networks
> **live in the browser** with [TensorFlow.js](https://www.tensorflow.org/js).
> No backend, no build step — one `index.html`.

Built for the Axiom **Learning OS** · ERA V5 · Session 1 Assignment QnA (500 pts).

---

## The four proofs

| # | Claim | What the app shows |
|---|-------|--------------------|
| **S1-1** | *Activations exist for a reason.* | The same 300-point concentric-ring dataset is fed to a **linear + sigmoid** model and a **one-hidden-layer ReLU** model. The linear decision boundary stays a straight line stuck near ~55%; the ReLU model wraps the ring at ~99%. Only the activation changed. |
| **S1-2** | *Depth without nonlinearity is a lie.* | **1 linear layer** vs **5 stacked linear layers (no activation)** vs **5 layers + ReLU**. The first two produce the *identical* straight-line boundary and accuracy. The app then **multiplies the five trained weight matrices** and prints the single collapsed `2×1` map — proof that 5 linear layers ≡ 1 linear layer. ReLU breaks the tie. |
| **S1-3** | *Embeddings learn similarity from nothing but next-token.* | A tiny `embedding → softmax` model is trained **only** to predict the next token in a toy grammar (`animal → verb → fruit`). The 2-D embedding table visibly separates into three category clusters (animals / fruits / verbs). Similarity was never supplied — only co-occurrence. |
| **S1-4** | *Memorization vs generalization — data closes the gap.* | The **same over-parameterized 64×64 net** is trained at dataset sizes **20 / 200 / 2000**. At n=20 it memorizes (train ≈ 100%, test poor); at n=2000 the train/test gap nearly vanishes. A bar chart plots the generalization gap shrinking with data. |

---

## Why this design

The assignment asks for "a beautiful web app which proves these 4 points." Rather than
ship static screenshots, every claim is settled by a model that **actually trains in
front of the viewer** — the most convincing possible proof, and faithful to the spirit of
Session 1 (a forward pass, a loss, a backward pass, repeated).

All four experiments were **validated twice before shipping**:
1. Reproduced numerically in Python (NumPy / scikit-learn).
2. Re-run through the **actual TensorFlow.js model code in Node** to confirm the in-browser
   API calls are correct and produce the claimed numbers. (This caught a real bug: TF.js
   `sparseCategoricalCrossentropy` requires float32 labels — fixed before release.)

### Validated results

```
S1-1   linear ~52–56%   |  ReLU ~99–100%
S1-2   1-linear ≈ 5-linear (both a straight line)  |  5+ReLU ~100%
       W1·W2·W3·W4·W5  →  single 2×1 matrix
S1-3   next-token loss → ln(3) ≈ 1.10 (theoretical floor)  |  9/9 same-category neighbors
S1-4   gap: ~0.23 (n=20) → ~0.07 (n=200) → ~0.00 (n=2000)
```

> Note on S1-3: the loss settles near **1.10**, not 0 — that is the information-theoretic
> minimum because the next token is genuinely 1-of-3. The **clustering** is the proof, not loss→0.

---

## Tech stack

- **TensorFlow.js 4.22** (via CDN) — in-browser model definition + training.
- **Vanilla JS + HTML5 Canvas** — custom decision-surface heatmaps, scatter plots, and
  the generalization-gap bar chart (no charting library).
- **Fonts:** Fraunces (display serif) + Inter (body) + JetBrains Mono (code), Google Fonts.
- **Theme:** Axiom-inspired warm cream / ink palette to match the Learning OS brand.
- Seeded PRNG (mulberry32) so datasets are reproducible across reloads.

No frameworks, no bundler, no server. The entire app is one `index.html`.

---

## Run locally

Just open `index.html` in any modern browser, **or** serve the folder:

```bash
cd webapp
python3 -m http.server 8000
# visit http://localhost:8000
```

Click **"Run all four experiments"**, or train each proof individually. Reload to reset.

## Deploy

See [`DEPLOY.md`](./DEPLOY.md). Fastest path: drag the `webapp` folder onto
**https://app.netlify.com/drop**, then sign in to make the URL permanent.

---

## File structure

```
webapp/
├── index.html     # the entire app (markup + styles + TF.js logic)
├── README.md      # this file
└── DEPLOY.md      # Netlify deployment instructions
```

## Notes

- First load fetches TensorFlow.js (~1 MB) and Google Fonts; afterwards it runs offline.
- Numbers vary slightly run-to-run (random initialization) — expected, and reinforces that
  training is genuinely happening live rather than being replayed.

---

*Course: ERA V5 · Session 1 — From Neural Networks to the Transformer · Axiom Learning OS.*
