# PathMNIST Colour Augmentation — LecTrace

**A 10-minute execution-explanation presentation for MD6329: Deep Learning for Healthcare AI.**

The presentation explains the completed PathMNIST experiment about binary classification of normal colorectal mucosa vs tumour epithelium, with colour augmentation and synthetic colour robustness testing.

## View the presentation

Once GitHub Pages is enabled (**Settings → Pages → Source → GitHub Actions**), the site is built on every push to `main`.

Expected URL: https://dongjinshin1-a11y.github.io/Lectrace/

A successful GitHub Actions deployment is required before the URL works.

## Present for ~10 minutes

- 0:00–2:30: healthcare motivation, dataset, experimental question and six CNN training conditions.
- 2:30–3:45: **State 1** data filtering and 128×128 → 64×64 preprocessing.
- 3:45–5:15: **State 2** HSV and H&E colour augmentation.
- 5:15–7:00: **State 3** convolution, feature extraction and sigmoid.
- 7:00–8:15: **State 4** BCE and validation-based checkpoint selection.
- 8:15–9:15: **State 5** AUROC robustness under synthetic RGB shifts.
- 9:15–10:00: interpretation and limitations.

Use the right arrow key to move forward through the source and speaker notes. Skip mathematical micro-steps when needed to stay within 10 minutes.

## Scientific accuracy

All dataset counts, training configurations, validation numbers and results reflect the supplied executed project notebook. Individual example pixel values, a demonstration convolution kernel, and sigmoid/BCE calculations are **pedagogical examples, not outputs from a saved trained model**. This is a standalone teaching trace, not retraining of the 12 models. HSV RGB-effect confidence intervals were not displayed in the report, so no statistical significance is claimed.

## Local preview (optional)

Requires Python 3.11+:

```bash
python3 -m pip install lectrace
lectrace serve 01_pathmnist_colour_robustness.py
```

Open http://localhost:7000.
