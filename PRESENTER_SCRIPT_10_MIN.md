# LecTrace — 10-minute rehearsal script (MD6329)

**Lecturer's structure:** 2–3 minutes for the healthcare problem and approach, then **3–5 selective execution states**, then findings and healthcare meaning. We use **five states**. Present by stepping through LecTrace, not by reading every source line. **No video is required.**

| Clock | Segment | Duration |
|---|---|---|
| 0:00–2:20 | Problem, motivation, research questions, six training conditions | 2:20 |
| 2:20–3:15 | State 1: data entering the CNN | 0:55 |
| 3:15–4:35 | State 2: colour manipulation | 1:20 |
| 4:35–6:40 | State 3: CNN computation and prediction | 2:05 |
| 6:40–7:50 | State 4: training and checkpoint choice | 1:10 |
| 7:50–9:20 | State 5: robustness metrics | 1:30 |
| 9:20–10:00 | Conclusion and clinical relevance | 0:40 |

The spoken material below is intentionally concise: use the remaining seconds within each segment to **advance through code and point to the changing variables or plots**. Rehearse with a clock because real speaking speed differs.

## 0:00–2:20 | Introduction

“Haematoxylin and eosin staining is widely used to inspect colorectal tissue. The colours of stained images can vary with laboratory procedures and image acquisition, even where the underlying tissue category is unchanged. If an AI model performs very well on familiar images but is sensitive to colour changes, that could be problematic for pathology support.

“My project investigates **binary classification** of normal colon mucosa, or NORM, versus colorectal tumour epithelium, or TUM, using PathMNIST. This is patch classification, not patient diagnosis.

“I ask two research questions. First, how does weak or strong H&E or HSV colour augmentation affect classification on original images? Second, does augmentation reduce loss of discrimination under progressively stronger synthetic colour shifts? Clean-image AUROC and performance under shifts must be distinguished.

“We trained a nine-convolution CNN under six conditions: unaugmented control, HED conversion-and-reconstruction control, weak/strong H&E augmentation and weak/strong HSV augmentation. Architecture, sampling and training budget were matched across two random seeds, for **twelve models**. Each model was evaluated under 18 conditions: the original image, HED reconstruction and 16 nonzero synthetic perturbations. I will step through five execution states that explain the input, processing, prediction, model selection and robustness result.”

**Display/click cue:** point to both research questions, the six-condition design and the five-state outline. Do not rush this opening: teacher expects 2–3 minutes.

## 2:20–3:15 | State 1: Input

“First, we filter PathMNIST to labels six and eight. The executed partitions contain **25,054** patches, including **1,974** held-out test patches. The original 128-by-128 RGB images are reduced to 64-by-64 by averaging each two-by-two block, then scaled to zero-to-one. This reduces computation, though fine detail is lost. Training batches have 32 examples from each class, while validation and test use their natural proportions.”

**Display/click cue:** inspect total patches, input dimensions, four-pixel averaging and real training examples. **What changes:** size and pixel scale. **Why it matters:** reproducible, faster CNN input.

## 3:15–4:35 | State 2: Colour operations

“Our intervention changes image colours during training. Weak H&E multiplies estimated H/E stain concentrations by 0.95–1.05; strong H&E uses 0.80–1.20. HSV instead changes hue, saturation and brightness within weak or strong predefined ranges.

“Watch this severity-four stress calculation: red-dominant RGB shifts multiply channels by **1.16, 1.00 and 0.84**. H-dominant H&E shifts use factors **1.32 and 0.84**. They alter colour without moving the image's tissue structures. We use HED reconstruction as the primary H&E comparator, because RGB-to-HED conversion itself may change pixel values; HSV primarily uses the unaugmented comparator.”

**Display/click cue:** advance through severity, gains and H&E factors. **What changes:** colour intensity. **Why it matters:** controlled stress tests, not proof of real cross-hospital robustness.

## 4:35–6:40 | State 3: Actual CNN computation

“The CNN has nine 3-by-3 convolution layers, each followed by GroupNorm and LeakyReLU. Layers 2, 4, 6 and 8 use stride two, reducing spatial resolution. Advancing through these calculations, the input **3×64×64** becomes a **256×4×4** feature map. Global average pooling gives 256 features, and the dense head gives one logit. Overall there are **1,243,985** trained parameters.

“Convolutions transform local image information into learned features. The final logit is converted into a score with the **sigmoid function**, and a score of 0.5 or above predicts TUM. The score is not a calibrated clinical probability.”

**If the exported REAL trace appears:** “This is the same actual test patch before and after RGB severity-four stress. Inspect recorded convolution 1, 5 and 9 activations; then watch LecTrace calculate sigmoid from the **actual trained checkpoint logits** and apply the prediction rule. The score can shift without necessarily changing the class. A single patch does not prove overall robustness.”

**If no trace appears:** “These are executed **architecture calculations**, not a replay of learned activations; a trained-checkpoint trace was not available to the site. We must not invent its logit.”

“In the real clean-test predictions for seed 2026, unaugmented training produced **31 false positives and 41 false negatives**; strong HSV produced **68 false positives and 24 false negatives**. The 0.5 threshold therefore trades fewer missed tumours for more normal-patch alerts.”

**Display/click cue:** use most of these 2:05 to step through 9 layers, the genuine logit/sigmoid if available, and the two error pairs. **Why it matters:** this is how predictions arise and why mistakes have different clinical consequences.

## 6:40–7:50 | State 4: Checkpoint selection

“Each condition uses the same eight-epoch budget and Adam optimiser. Checkpoints were selected using **lowest clean validation binary cross-entropy**, not by test AUROC.

“For weak H&E, seed 2026, validation BCE rose from **0.0738 at epoch six** to **0.2374 at epoch seven**. Our executed minimum calculation selects epoch six. A fluctuating loss curve is a training-stability observation, not by itself proof of a software bug. The unaugmented CNN had the highest mean clean AUROC, **0.9928**, but clean discrimination alone cannot tell us whether it is robust.”

**Display/click cue:** run the saved BCE list/minimum operation and point to the AUROC plot. **Why it matters:** comparison without test-based checkpoint cherry-picking.

## 7:50–9:20 | State 5: Robustness comparison

“We define degradation as reference AUROC minus stressed AUROC. A positive augmentation gain means **less** degradation than the appropriate comparator, but we must also check absolute stressed AUROC.

“Under synthetic RGB stress, weak and strong HSV reduced average degradation by **0.0108 and 0.0104 AUROC**, and increased stressed AUROC by **0.0071 and 0.0090** relative to unaugmented training. Strong HSV gave the highest **minimum** RGB-stressed AUROC in both tested seeds: **0.9762 and 0.9833**.

“In contrast, H&E augmentation improved H&E-stress degradation by just **0.0006 and 0.0004**, and those tiny effects are relative to **HED reconstruction**, not evidence of gains over unaugmented training.

“Only two seeds were trained; perturbations were synthetic; and displayed uncertainty does not establish statistical significance for RGB-HSV differences. The result is exploratory, not proof of clinical generalisation.”

**Display/click cue:** execute the seed-average gain, show both absolute-gain numbers and the worst-condition figure. **What changes:** AUROC under colour stress. **Why it matters:** ranking shifts when robustness, not clean AUROC, is the objective.

## 9:20–10:00 | Conclusion

“The key finding is **perturbation-specific robustness**. The unaugmented CNN had the highest clean AUROC, but HSV training performed better under our tested synthetic RGB shifts. H&E benefits were very small and comparator-dependent.

“For healthcare AI, high accuracy on familiar images is not sufficient for safe deployment. More seeds, a locked evaluation protocol, patient-aware analysis and independent multi-laboratory data are needed before any clinical conclusion. Thank you.”

---

**Before submission:** open LecTrace from the first step; check it reaches the conclusion; optionally enable real forward replay by uploading the genuine `real_forward_trace.json` exported from your own checkpoint. Only present numerical values that were recorded or recalculated from those records.
