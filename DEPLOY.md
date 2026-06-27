# Session 1 — Four Proofs · Deploy to Netlify

This is a **single self-contained `index.html`** (no build step, no backend). Everything trains live
in the browser with TensorFlow.js loaded from a CDN.

## Option A — Netlify Drop (fastest, ~30 seconds)
1. Go to **https://app.netlify.com/drop**
2. Drag the **`webapp` folder** (the one containing `index.html`) onto the page.
3. Netlify gives you a live URL instantly (e.g. `https://something-random.netlify.app`).
4. To keep it permanent: click **"Sign up / Log in"** and the site saves to your account.
   You can rename it under **Site settings → Change site name**.

## Option B — Netlify account (recommended for the submission)
1. Create a free account at **https://netlify.com**.
2. **Add new site → Deploy manually** → drag the `webapp` folder.
3. Copy the live link into the assignment's **Netlify Link** field, add the caption
   (e.g. "Session 1 — four live in-browser proofs"), and tick the incognito-tested box.

## Test before submitting
- Open the live URL in an **incognito window** (assignment requires it to be publicly accessible).
- Click **"Run all four experiments"** and confirm:
  - S1-1: linear stalls ~50–58%, ReLU reaches ~99–100%.
  - S1-2: the 1-linear and 5-linear panels show the **same straight line**; the matrix box prints a single 2×1 `W_eff`.
  - S1-3: three colored clusters (animals / fruits / verbs) separate; neighbors → 9/9.
  - S1-4: gap shrinks from ~20%+ at n=20 to ~0 at n=2000.

## Notes
- First load fetches TensorFlow.js (~1 MB) and the Google fonts; after that it runs offline.
- Numbers vary slightly run-to-run (random init) — that's expected and reinforces that it's training live.
- Reload the page to reset everything and re-run from scratch.
