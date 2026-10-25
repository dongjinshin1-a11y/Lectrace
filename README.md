# PathMNIST colour robustness — LecTrace

A ten-minute, five-state execution explanation of the PathMNIST colorectal-tissue colour augmentation experiment. This is **not** a model training or inference service: it presents the saved notebook's results and executes small clearly labelled teaching calculations to explain what the original PyTorch model did.

## Presentation

[Open the LecTrace presentation](https://dongjinshin1-a11y.github.io/Lectrace/#/?trace=01_pathmnist_colour_robustness)

Use the right arrow key to advance through the code and inspect variables. Read the timed `note(...)` blocks and the [ten-minute speaking guide](PRESENTER_SCRIPT_10_MIN.md) as the presenter script. A video recording is optional.

| Time | Segment | What changes and why |
|---|---|---|
| 0:00–2:20 | Introduction | Pathology problem, two research questions, controlled experiment |
| 2:20–3:15 | State 1 — input | Real split counts, binary filtering, 2×2 averaging, balanced batches |
| 3:15–4:35 | State 2 — colour | The actual augmentation parameter ranges, RGB stress operator, reconstruction comparator |
| 4:35–6:40 | State 3 — CNN | Executable nine-layer tensor-shape and parameter-count derivation, plus actual seed-2026 error counts; checkpoint replay appears when exported |
| 6:40–7:50 | State 4 — validation | Actual eight-epoch BCE curve, minimum-loss checkpoint selection, clean AUROCs and baselines |
| 7:50–9:20 | State 5 — robustness | Averaged gains, worst-condition AUROCs in both seeds, H&E comparator caveat |
| 9:20–10:00 | Conclusion | What the results support, what remains unproven, next experiments |

## Scientific provenance

All measured values are transcribed from `PathMNIST_Tellez_Inspired_(1).ipynb` and cross-checked against `PathMNIST_Colour_Robustness_Report(5).docx`. Dataset counts, architecture, training configuration, confusion-matrix errors and robustness metrics come from the executed experiment. The two image panels are embedded PNG outputs from notebook cells 5 and 9 (zero-based). There are no invented pixel values or logits. CNN layer-shape and parameter arithmetic is executed in the presentation and checked against the implemented architecture (1,243,985 parameters). It is not a recorded trained forward pass. The validation-loss plot uses the notebook's saved rounded epoch logs. Checkpoint selection is replayed from rounded validation logs, with the same selected epoch as the saved checkpoint summary.

The actual evaluation includes six training conditions, two paired seeds, eight epochs, 1,974 test patches, and 18 test conditions per trained CNN. HSV training improved observed synthetic RGB robustness; no statistical significance for the RGB–HSV contrasts or real-world clinical generalisation is claimed. H&E effects are small and should be interpreted against the HED reconstruction comparator as well as the unaugmented model.

The charts are built from the saved numerical outputs. The site does not rerun PyTorch training or load checkpoints. If `real_forward_trace.json` is present, State 3 replays recorded actual PyTorch checkpoint inference for the SAME official test patch, clean and RGB severity-4. It shows three recorded feature-map summaries, the actual logits, executable sigmoid calculations, scores and predicted classes. The source notebook is the scientific implementation; the LecTrace Python file is an educational execution explanation.

## Verify a real trained CNN forward pass (without retraining)

The notebook's output ZIP deliberately excluded all **.pt** checkpoints, so a trained forward pass cannot be reconstructed from charts or confusion counts. The repository includes [colab_export_real_forward.py](colab_export_real_forward.py) to export an authentic inference trace from the **existing** Google Drive checkpoint and official PathMNIST test patch.

In Google Colab, with the same Google account that ran the original notebook:

```python
# Run in a Colab cell; this does not train a model.
!pip -q install medmnist pillow
!curl -fsSL https://raw.githubusercontent.com/dongjinshin1-a11y/Lectrace/main/colab_export_real_forward.py -o colab_export_real_forward.py
!python colab_export_real_forward.py
from google.colab import files
files.download('real_forward_trace.json')
```

This checks for the original `cnn_hsv_strong_seed2026_best.pt` **and `experiment_config.json`** under `MyDrive/PathMNIST_HE_Robustness/tellez-inspired-gn-v5-budget64/final/`. The exporter checks the final experiment signature saved after training rather than the preliminary configuration signature. Exported data include the **actual** sample, nine recorded convolution activations, actual logits, sigmoid scores and predictions for clean versus severity-4 RGB-stressed input. It will refuse to invent values if the checkpoint is missing.

Upload **only** `real_forward_trace.json` to the GitHub repository root. GitHub Actions will then rebuild the page with the trained-forward trace. Do not upload a multi-GB dataset, raw checkpoints, or Google Drive credentials to make the page work. Until this export is supplied, the page explicitly labels the CNN dimension calculation as architecture-only. This is the outstanding evidence gap for a full trained inference replay until the real JSON has been exported and committed.

The repeated CNN `for`-loop that previously made navigation appear frozen near step 97 was replaced with nine distinct executable convolution-shape steps. GitHub Actions now checks five states, seven timing cues and a reachable ending. Use the **beginning** link above; URLs with `step=80` will jump into the middle.

## Build locally

Requires Python 3.11+:

```bash
pip install lectrace
lectrace run 01_pathmnist_colour_robustness.py
lectrace serve 01_pathmnist_colour_robustness.py
```

GitHub Actions automatically builds and deploys the site after a push to `main`. GitHub Pages must use **GitHub Actions** as the source.

## Source identity and remaining evidence requirement

Notebook SHA-256: `2745a3597c474efbc5091b5215db7e11535d7763107af79bf861f900bc9d3127`. Source filenames were updated to the files supplied for the 25 October review. All reported aggregate results remain unchanged. AI assisted with the presentation edits; researcher review is required.

To demonstrate an actual trained forward pass, export a real patch, layer activations/shapes, logit and score from an existing saved checkpoint. Neither the notebook attachment nor this repository includes those weights. Do not describe the current shape calculation as trained inference.

Cover-page state list: input/data; colour transformations; CNN computation and prediction errors; checkpoint selection; robustness evaluation. Use the beginning link above, not a URL containing `step=80`.
