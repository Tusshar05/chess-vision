# License notice: dataset, sample images and trained weights

The **source code** in this repository is licensed under **AGPL-3.0** (see [LICENSE](LICENSE)).

The following are **not** covered by the AGPL grant; they are under
**[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)**:

1. **The Chess Recognition Dataset (ChessReD)** and everything derived from it by the scripts here
   (YOLO-format conversions, resized images, `reports/data/` figures showing dataset images).
2. **The sample photos in `samples/`** (resized ChessReD test-split images).
3. **Any model weights trained on ChessReD** (`.pt`, `.onnx`, `.mlpackage`, release assets,
   Hugging Face Space models), because they are derived from the dataset. Ultralytics' own
   AGPL-3.0 terms also apply to the YOLO11 code/weights these were fine-tuned from.

Under CC BY-NC-SA 4.0 you must: give attribution (below), use the material **non-commercially**,
and share adaptations under the **same license**. Any redistribution (GitHub Release, Hugging Face
Space, Docker image containing weights) must carry this notice.

## Attribution
Athanasios Masouris and Jan van Gemert (Delft University of Technology), *Chess Recognition
Dataset (ChessReD)*, 4TU.ResearchData, DOI
[10.4121/99b5c721-280b-450b-b058-b2900b69a90f.v2](https://doi.org/10.4121/99b5c721-280b-450b-b058-b2900b69a90f.v2).
Paper: Masouris & van Gemert, "End-to-End Chess Recognition", VISAPP 2024,
DOI 10.5220/0012370200003660.
Changes made: images resized (<= 1280 px), annotations converted to YOLO format, models trained on it.

Note: the task brief described the dataset as CC BY 4.0; the 4TU.ResearchData record (checked
2026-11-11) lists CC BY-NC-SA 4.0, which is what applies.

## Other third parties
Stockfish: GPL-3.0, executed as a separate binary. Lichess links are plain URLs.
