# Deploying the Gradio demo to Hugging Face Spaces

> Do **not** deploy smoke weights publicly. `scripts/build_demo_bundle.py` refuses unless the
> active weights are `kind: trained`.

## Prerequisites
- Trained weights installed and exported (`make weights export`), i.e. `weights/detector.onnx`,
  `weights/corners.onnx` (and `*_int8.onnx`) exist and `weights/manifest.json` says `"kind": "trained"`.
- A free account at <https://huggingface.co> and an access token with **write** permission
  (Settings -> Access Tokens).

## Steps
1. **Check locally first**
   ```bash
   .venv/bin/python demo/app.py        # http://localhost:7860 - upload a sample, confirm sane output
   ```
2. **Build the self-contained bundle** (app, ONNX models, `ml/chessvision`, samples, `packages.txt`
   which installs Stockfish via apt on the Space, Space README with the YAML header):
   ```bash
   .venv/bin/python scripts/build_demo_bundle.py --precision fp32     # -> demo_build/
   ```
   INT8 (`--precision int8`) is smaller but check `reports/benchmark.md` first: on some CPUs it is
   not faster and costs accuracy.
3. **Create the Space**: huggingface.co -> New -> Space -> SDK **Gradio**, hardware **CPU basic
   (free)**, name e.g. `chessvision`, public.
4. **Upload the bundle** (handles large files via LFS for you):
   ```bash
   .venv/bin/pip install -U huggingface_hub
   huggingface-cli login                      # paste the write token
   huggingface-cli upload <your-user>/chessvision demo_build . --repo-type=space
   ```
5. The Space builds for a few minutes (Logs tab). When it shows *Running*, open it and test with the
   sample photos.
6. Put the Space URL in the README (demo badge) and on your resume.

## Notes
- Free CPU Spaces sleep after inactivity; the first request wakes them (~30-60 s).
- **Licensing (required):** the weights and sample images are CC BY-NC-SA 4.0 (non-commercial,
  share-alike, attribution to Masouris & van Gemert / TU Delft, DOI 10.4121/99b5c721-280b-450b-b058-b2900b69a90f.v2)
  and the code is AGPL-3.0. The bundle's Space README carries this notice; also copy
  `LICENSE-DATA-AND-WEIGHTS.md` into the Space (the bundle script does). Do not use a commercial/paid Space for it.
- (old note) Ultralytics-derived weights are AGPL-3.0, the dataset is
  CC BY-NC-SA 4.0 (non-commercial) - keep the attribution in the Space README (the bundle script
  writes it).
- To update: rebuild the bundle and re-run the `upload` command.
