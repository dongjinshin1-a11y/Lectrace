# Ten-minute LecTrace speaking guide (MD6329)

**Format required by the lecturer:** approximately 10 minutes, with a 2–3 minute introduction, 3–5 carefully chosen *execution states*, and a concluding result with a healthcare interpretation. This guide uses **five states**. Speak to an imaginary audience and advance through the code, stopping at the indicated variables; do not click through every trace step at uniform speed. **A video is optional.**

**Timing:** 0:00–2:20 introduction; 2:20–3:15 State 1; 3:15–4:35 State 2; 4:35–6:40 State 3; 6:40–7:50 State 4; 7:50–9:20 State 5; 9:20–10:00 conclusion. **Total: exactly 10:00.**

## 0:00–2:20 — Healthcare problem, motivation and approach

“Pathology tissue sections are commonly stained with haematoxylin and eosin. Haematoxylin highlights nuclei and eosin highlights other tissue structures. Unfortunately, the appearance of the resulting colour image can vary with laboratory staining procedures and acquisition settings. A CNN might score well on images resembling the data it saw in training and still be sensitive to those appearance changes. In a future pathology-support workflow that would be undesirable, because reliable tissue discrimination is more important than good performance on just one familiar colour distribution.

“My mini-project focuses on two categories: normal colon mucosa, NORM, and colorectal tumour epithelium, TUM. The dataset is PathMNIST. This is a **binary patch-classification research exercise**, not a clinical diagnostic system or patient-level classifier.

“I ask two questions. First, does training with colour augmentation preserve the ability to distinguish NORM from TUM on original images? Second, does augmentation prevent a drop in discrimination when test-image colours are perturbed? These are different questions: a model can have nearly perfect original-image AUROC while being comparatively fragile under a specific synthetic shift.

“I compare six training conditions: no colour transformation, an HED conversion-and-reconstruction control, weak and strong H&E augmentation, and weak and strong HSV augmentation. The CNN architecture and eight-epoch budget are kept fixed across two paired random seeds, giving twelve trained models. The evaluation has eighteen conditions per model: original RGB, HED reconstruction, and sixteen nonzero stress conditions across two families, two directions and four levels. I will now show **five selective execution states**: input, colour processing, CNN computation, checkpoint selection and robustness evaluation. The aim is to explain how a neural model produces and is evaluated on its output, not to display every training iteration.”

**Cue:** stop on the two research questions and six-condition design, then advance into State 1.

## 2:20–3:15 — State 1: What enters the model?

“First, we filter the official PathMNIST partitions to labels 6 and 8. The executed data contain 25,054 eligible patches, with 20,771 used for training, 2,309 for validation and 1,974 for testing. About 62% are tumour patches. The test images remain separate from training.

“The original 128 by 128 patches are reduced to 64 by 64 by averaging each two-by-two block. Pixel values are then scaled to zero to one. **What changes** here is resolution and numerical scale, not the class label. **Why it matters** is computational feasibility, although some fine image detail is lost. During training each batch has 32 NORM and 32 TUM patches; the natural class proportions remain in validation and test. These displayed example tissue patches come from training.”

**Cue:** step through dataset counts and the 128-to-64 calculation; point to the genuine training images.

## 3:15–4:35 — State 2: What changes during colour augmentation?

“Second, colour processing is the intervention. In H&E augmentation I change estimated stain strengths after converting RGB into HED stain space and reconstructing RGB. Weak H&E samples factors between 0.95 and 1.05; strong H&E between 0.80 and 1.20. For HSV augmentation the weak setting changes hue by up to 0.05 turns and saturation/value by up to ten percent; the strong setting allows 0.15 hue turns and thirty percent changes to saturation/value. Training applies its assigned colour operation with probability 0.8.

“Look at the executed severity-four calculation. For a red-dominant RGB stress, the gains become 1.16, 1.00 and 0.84. In the H-dominant H&E stress, the factors become 1.32 and 0.84. **What changes** is pixel colour intensity; the spatial location of the tissue is unchanged. **Why it matters** is that we can compare model sensitivity under a controlled perturbation.

“One important detail is the comparator: H&E augmentation should principally be compared against HED reconstruction alone, because conversion and reconstruction can themselves alter image values. HSV is principally compared against the unaugmented model. These are controlled synthetic stresses, not observed differences between real hospitals.”

**Cue:** step through severity, gains and H&E factors; point to the original versus transformed training-image panel.

## 4:35–6:40 — State 3: How does the CNN produce a prediction?

“The actual PyTorch model uses nine 3-by-3 convolutions. Every layer uses GroupNorm and LeakyReLU. Convolutions 2, 4, 6 and 8 have stride two. Let me advance through these shape calculations: starting from 3 by 64 by 64, the model increases the number of feature channels while reducing the spatial width, reaching 256 by 4 by 4. After global average pooling, 256 values feed a small dense classification head which produces **one logit**. This model has 1,243,985 trainable parameters. GroupNorm and LayerNorm do not require BatchNorm running statistics.

“**What changes** across the convolutional layers is the representation: local colour and spatial patterns are transformed into higher-dimensional features. **Why it matters** is that the final classification score comes from a sequence of learned transformations, not just a hand-written image rule.

**When the genuine exported forward trace is present:** “Here is an actual held-out PathMNIST test patch, shown unmodified and with the defined level-four RGB shift. These displayed convolution activation summaries, the two logits and the scores came from the saved trained strong-HSV checkpoint. Watch the recorded output after convolution 1, 5 and 9; the means can change even though this is the same underlying tissue. Next, the trace *executes sigmoid on each real logit*. The predicted label is TUM if the score is at least 0.5, otherwise NORM. Read the actual on-screen values rather than memorising numbers. A change in score does not necessarily mean the predicted class flips. One patch alone cannot establish population-level robustness.”

**If the exported forward trace is NOT present:** “The nine shape calculations are executed, but without the trained weights I cannot reproduce the actual per-patch activations or logits on this webpage. I therefore show the saved error counts instead and do not call the shape calculation a trained forward pass.”

“On the full clean test, seed 2026, the unaugmented CNN marked 31 normal patches as tumour and missed 41 tumour patches. Strong HSV instead marked 68 normal patches as tumour and missed 24 tumour patches. **Why it matters:** at the fixed 0.5 threshold, higher tumour recall can be accompanied by more unnecessary normal-patch alerts. The raw sigmoid score is not established as a calibrated clinical probability.”

**Cue:** move deliberately through the nine architecture lines, then (if available) clean and stressed recorded forward values, logits, sigmoid and predictions; finish on real confusion counts.

## 6:40–7:50 — State 4: How do we choose a model checkpoint?

“Fourth, the models all use the same paired seeds, eight epochs, balanced batches, optimiser and core architecture. Training uses Adam with learning rate 0.0003 and binary cross-entropy with logits. Importantly, we select each checkpoint by the **lowest clean validation BCE**, never by whichever model happened to perform best on the test set.

“This is a real curve for weak H&E, seed 2026. At epoch 6, clean validation BCE was approximately 0.0738, but at epoch 7 it increased to 0.2374. Our executed minimum-loss calculation therefore chooses epoch 6. **What changes** is the saved validation loss across epochs. **Why it matters** is that using the final epoch may be worse than selecting the best validation checkpoint. Fluctuating validation loss alone is not a proven code bug.

“The clean-test AUROC of the unaugmented model was 0.9928, the highest mean among the CNN settings. A colour-only logistic-regression baseline nevertheless scored 0.9320, reminding us that colour statistics alone contain predictive information. High clean AUROC, however, does not answer the question of how the models behave under colour stress.”

**Cue:** step the BCE list and selected-epoch calculation; point to the clean-AUROC comparison.

## 7:50–9:20 — State 5: Did augmentation reduce stress degradation?

“Finally, the robust comparison distinguishes absolute performance from degradation. Degradation is reference AUROC minus stressed AUROC. A positive augmentation gain means the augmented model loses **less** AUROC than its comparator. We also examine absolute stressed AUROC to avoid mistaking a weak baseline for true robustness.

“Across the synthetic RGB stress levels and directions, weak HSV reduced mean degradation by 0.0108 AUROC units compared with no colour augmentation; strong HSV reduced it by 0.0104. Both also improved **absolute stressed AUROC**, by 0.0071 and 0.0090 respectively. These are averages over both directions, four levels and two seeds.

“Looking at the worst tested RGB condition in each seed, strong HSV achieved a minimum AUROC of 0.9762 and 0.9833, higher than the corresponding unaugmented results of 0.9209 and 0.9601. The plotted vertical axis is zoomed: look at the numerical labels before interpreting visual distances.

“Under H&E stress, gains were much smaller: about 0.0006 and 0.0004 for weak and strong H&E, and those are **relative to the reconstruction control**, not evidence of superiority over unaugmented training. **What changes** in this state is the metric under progressively altered colours. **Why it matters** is that resilience depends on the augmentation and the kind of shift tested.

“Both HSV seeds showed descriptive improvements, but with only two training seeds, synthetic stresses and no displayed confidence intervals for the RGB gain, I cannot claim statistical significance or real-world hospital robustness.”

**Cue:** calculate seed-mean gain, discuss absolute gains, then stop on the worst-condition comparison and uncertainty.

## 9:20–10:00 — Key result and healthcare interpretation

“My main finding is not that one colour augmentation is universally best. The more defensible result is that the unaugmented CNN had the highest clean-image AUROC, while HSV-augmented models were more robust to our specific synthetic RGB changes. H&E augmentation offered only tiny robustness gains relative to its HED reconstruction control.

“For healthcare AI, this illustrates why evaluating a familiar test set is not enough: we should examine anticipated appearance changes and clinically relevant errors. The next steps would be more training seeds, a locked protocol, patient-aware uncertainty analysis and independent real-laboratory data. These experiments are exploratory and do not establish clinical safety. Thank you.”

---

## Submission checklist

- Use the **opening URL** of LecTrace, not a URL with a saved step number.
- Walk the audience through **five** computational states, not every training iteration.
- Do not present hypothetical logits or colour-perturbed data as clinical validation.
- To enable the **real checkpoint forward replay**, run the exporter in the original Google Colab account, download `real_forward_trace.json`, then upload it to the GitHub repository root.
- The Python source is the submission; this Markdown speaking guide is supplementary. No video is required.
