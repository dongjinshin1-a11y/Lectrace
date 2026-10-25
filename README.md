# PathMNIST colour robustness — LecTrace

A ten-minute, five-state execution explanation of the PathMNIST colorectal-tissue colour augmentation experiment. This is **not** a model training or inference service: it presents the saved notebook's results and executes small clearly labelled teaching calculations to explain what the original PyTorch model did.

## Presentation

[Open the LecTrace presentation](https://dongjinshin1-a11y.github.io/Lectrace/#/?trace=01_pathmnist_colour_robustness)

Use the right arrow key to advance through the code and inspect variables. Read the timed `note(...)` blocks as the presenter script.

| Time | Segment | What changes and why |
|---|---|---|
| 0:00–2:00 | Introduction | Pathology problem, two research questions, controlled experiment |
| 2:00–3:00 | State 1 — input | Real split counts, binary filtering, 2×2 averaging, balanced batches |
| 3:00–4:40 | State 2 — colour | The actual augmentation parameter ranges, RGB stress operator, reconstruction comparator |
| 4:40–6:10 | State 3 — CNN | Executable nine-layer tensor-shape and parameter-count derivation, plus actual seed-2026 error counts |
| 6:10–7:25 | State 4 — validation | Actual eight-epoch BCE curve, minimum-loss checkpoint selection, clean AUROCs and baselines |
| 7:25–9:15 | State 5 — robustness | Averaged gains, worst-condition AUROCs in both seeds, H&E comparator caveat |
| 9:15–10:00 | Conclusion | What the results support, what remains unproven, next experiments |

## Scientific provenance

All measured values are transcribed from `PathMNIST_Tellez_Inspired_.ipynb` and cross-checked against `PathMNIST_Colour_Robustness_Report(4).docx`. Dataset counts, architecture, training configuration, confusion-matrix errors and robustness metrics come from the executed experiment. The two image panels are embedded PNG outputs from notebook cells 5 and 9 (zero-based). There are no invented pixel values or logits. CNN layer-shape and parameter arithmetic is executed in the presentation and checked against the implemented architecture (1,243,985 parameters). It is not a recorded trained forward pass. The validation-loss plot uses the notebook's saved rounded epoch logs. Checkpoint selection is replayed from rounded validation logs, with the same selected epoch as the saved checkpoint summary.

The actual evaluation includes six training conditions, two paired seeds, eight epochs, 1,974 test patches, and 18 test conditions per trained CNN. HSV training improved observed synthetic RGB robustness; no statistical significance for the RGB–HSV contrasts or real-world clinical generalisation is claimed. H&E effects are small and should be interpreted against the HED reconstruction comparator as well as the unaugmented model.

The charts are built from the saved numerical outputs. The site does not rerun PyTorch training or load checkpoints. The source notebook is the scientific implementation; the LecTrace Python file is an educational execution explanation.

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
