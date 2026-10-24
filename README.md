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
| 4:40–6:10 | State 3 — CNN | Nine-layer model, sigmoid illustration, actual seed-2026 error counts |
| 6:10–7:25 | State 4 — validation | Saved BCE change, fair checkpoint selection, clean AUROCs and baselines |
| 7:25–9:15 | State 5 — robustness | Averaged gains, worst-condition AUROCs in both seeds, H&E comparator caveat |
| 9:15–10:00 | Conclusion | What the results support, what remains unproven, next experiments |

## Scientific provenance

All measured values are transcribed from `PathMNIST_Tellez_Inspired_Reviewed_3(4).ipynb` and cross-checked against `PathMNIST_Colour_Robustness_Report(3).docx`. Dataset counts, architecture, training configuration, confusion-matrix errors and robustness metrics come from the executed experiment. Numeric RGB pixels, 2×2 block values and sigmoid inputs in this script are **illustrative**, not claims about specific test patches or loaded neural weights.

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
