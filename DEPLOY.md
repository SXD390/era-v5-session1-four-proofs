# Deploy to Netlify

⚠️ **Deploy the whole `webapp` folder, not just `index.html`.** The app loads its
precomputed training data from `data.js`, so both files must be served together.

## Option A — Netlify Drop (fastest, ~30 seconds)
1. Go to **https://app.netlify.com/drop**
2. Drag the **`webapp` folder** (the one containing `index.html` **and** `data.js`) onto the page.
3. Netlify gives you a live URL instantly.
4. To keep it permanent: **Sign up / Log in** and the site saves to your account.
   Rename it under **Site settings → Change site name**.

## Option B — Netlify account (recommended for submission)
1. Create a free account at **https://netlify.com**.
2. **Add new site → Deploy manually** → drag the `webapp` folder.
3. Paste the live link into the assignment's **Netlify Link** field, add a caption
   (e.g. "Session 1 — four interactive, scrubbable proofs"), and tick the incognito-tested box.

## Test before submitting (in an incognito window)
- The page loads instantly in **Precomputed** mode (default). Drag any training timeline; press ▶ to animate.
- Confirm each proof:
  - **S1-1** — left (linear) stays a straight line ~50%; right (ReLU/Tanh) wraps the ring ~100%.
  - **S1-2** — "1 linear" and "5 linear" trace the same line; the matrix box shows a single 2×1 `W_eff`; "5 + ReLU" wins.
  - **S1-3** — scrub the timeline; nine words separate into three colored clusters (9/9 neighbors).
  - **S1-4** — switch n = 20 / 200 / 2000; the gap bars shrink to ~0 at 2000.
- Flip the top-right glass toggle to **Live · TF.js** to confirm real in-browser training works (loads TF.js ~1 MB the first time).

## Notes
- Precomputed mode is fully static and offline-capable — safe on any laptop, even an i3/i5.
- Live mode is optional and only loads TensorFlow.js when selected.
- Files served: `index.html`, `data.js` (required); `README.md`, `DEPLOY.md` are informational.
