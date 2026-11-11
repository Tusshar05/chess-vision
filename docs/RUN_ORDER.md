# What to do next (exact order)

Everything is built and tested with smoke weights. These steps produce the real numbers.

1. **Open the notebook in Colab.** Upload/open `notebooks/train_colab.ipynb` (File -> Open notebook
   -> GitHub tab -> `Tusshar05/chess-vision`, or upload the file). Set **Runtime -> Change runtime type
   -> T4 GPU**.
2. **Run all cells** (Runtime -> Run all). Authorise Google Drive when asked. Total ~2.5 h; the
   four runs are capped at <= 45 min each. If the session disconnects, re-run all cells - it resumes.
3. **Download the weights zip.** The last cell downloads `chessvision_weights.zip` (also saved at
   `MyDrive/chessvision/chessvision_weights.zip`).
4. **Unzip into `release_assets/`** in your local repo:
   ```bash
   cd chess-vision && unzip ~/Downloads/chessvision_weights.zip -d release_assets/
   ```
   (`release_assets/manifest.json` and `release_assets/det_n_640/...` should now exist.)
5. **Regenerate everything:**
   ```bash
   make data        # only if you have not run it (needs ~1.2 GB for the test images)
   make weights eval export bench report
   ```
   - `make weights` verifies SHA-256 + model tasks, picks the best detector by *validation*
     mAP50-95 and activates it.
   - `make eval` -> `reports/eval/`, `make export` -> ONNX/INT8/CoreML, `make bench` ->
     `reports/benchmark.md`, `make report` rewrites the README results tables
     (also run `.venv/bin/python -m ml.training_runs` to refresh the training-runs table - `make report`
     does this for you).
6. **Check results.** Open `reports/eval/e2e_summary.json`, `reports/eval/error_analysis.md` (fill in
   the "Analysis" section from the failure gallery), `reports/benchmark.md`, and the README.
   Sanity: detector mAP50 should be well above 0.5 and boards-with-0-errors clearly above 0. If not,
   look at `reports/eval/confusion_matrix.png` and the failure gallery before anything else.
7. **Commit and push:**
   ```bash
   git checkout -b results/trained-weights
   git add reports README.md docs && git commit -m "docs: results from trained weights" 
   git checkout main && git merge --no-ff results/trained-weights && git push origin main
   ```
   (Weights stay out of git; `release_assets/` and `weights/*.pt|onnx` are gitignored.)
8. **Upload weights as a GitHub Release:** GitHub -> Releases -> Draft a new release -> tag `v0.1.0`
   -> paste the notice from `LICENSE-DATA-AND-WEIGHTS.md` into the release description (weights are
   CC BY-NC-SA 4.0), attach `chessvision_weights.zip`, and `weights/detector.onnx`, `weights/corners.onnx`,
   `weights/manifest.json` (used by `scripts/download_weights.py` and the Kubernetes load job).
9. **Deploy the demo** per [DEPLOY_DEMO.md](DEPLOY_DEMO.md).
10. **Fill the resume bullets** in [RESUME.md](RESUME.md) using only numbers from `reports/`.

Handy: `docker compose up -d` after step 5 serves the trained models (the `weights/` folder is a
mounted volume: `docker compose restart api` after re-exporting).
