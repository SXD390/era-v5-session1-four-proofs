#!/usr/bin/env bash
# One-shot: clean any half-written repo, commit, create the GitHub repo, and push.
# Requires: gh authenticated as SXD390 (gh auth status).
set -e
cd "$(dirname "$0")"

echo "→ Removing any partial .git left by the sandbox…"
rm -rf .git

echo "→ Creating a fresh local repo…"
git init -q
git add -A
git -c user.name="SXD390" -c user.email="sudarshanravi13@gmail.com" \
    commit -q -m "Session 1 — four live in-browser proofs (TensorFlow.js): working version"
git branch -M main

echo "→ Creating GitHub repo and pushing…"
gh repo create era-v5-session1-four-proofs --public --source=. --remote=origin --push

echo "✓ Done. Repo: https://github.com/SXD390/era-v5-session1-four-proofs"
