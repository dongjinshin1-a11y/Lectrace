"""LecTrace: Colour augmentation and synthetic robustness in PathMNIST.

This is an execution-explanation of the *completed* experiment, not CNN retraining.
All reported findings come from PathMNIST_Tellez_Inspired_Reviewed_3(3).ipynb.
The small pixel, convolution, and sigmoid calculations are teaching examples.
"""
import colorsys
import math
from lectrace import text, note, plot


def main():
    text("# Colour Augmentation and Robustness in Colorectal Tissue Classification")
    text("## PathMNIST: normal colon mucosa (NORM) vs tumour epithelium (TUM)")
    text("**Ten-minute execution explanation:** introduction, five states, and healthcare interpretation.")
    text("**Research questions:** Does colour augmentation change clean discrimination? Does it reduce AUROC degradation under controlled H&E and RGB shifts?")
    note("""0:00–2:30 | INTRODUCTION
    Imagine an AI system that helps identify tumour epithelium in colorectal histology images. H&E staining reveals cellular and tissue structures, but laboratories and scanners can produce different colours. A CNN can achieve excellent AUROC on familiar images yet still be sensitive to colour changes. This project isolates normal mucosa and tumour epithelium from PathMNIST; it is patch classification, not clinical diagnosis. We trained six comparable CNN conditions: unaugmented; HED conversion-and-reconstruction only; weak and strong H&E; and weak and strong HSV. Each uses the same nine-convolution network, balanced training batches, eight epochs and two paired random seeds. We evaluate clean accuracy separately from robustness to synthetic colour shifts. In the next five states, we trace the computational choices that explain the experiment, then discuss what the results do and do not establish.""")

    state_1_prepare_input()
    state_2_augment_colour()
    state_3_cnn_prediction()
    state_4_select_checkpoint()
    state_5_evaluate_robustness()

    text("# Healthcare interpretation")
    text("**Key finding:** HSV augmentation improved *observed* AUROC under the tested synthetic RGB changes while clean-test AUROCs stayed high.")
    text("**What it does not prove:** robustness to real hospitals or scanners, patient-level clinical performance, or statistically significant HSV effects.")
    note("""9:15–10:00 | CONCLUSION
    Our highest clean AUROC came from the unaugmented CNN, but HSV-trained models suffered less AUROC degradation under RGB channel shifts. This demonstrates why strong performance on unmodified histology images is insufficient for deployment claims. H&E augmentation offered only tiny benefits relative to the HED reconstruction control, not a consistent benefit against unaugmented training. Limitations include two training seeds, synthetic rather than measured inter-lab colour variation, and missing displayed RGB-effect confidence intervals. The next study would use a locked protocol, more seeds, and external slide- or patient-level testing. The clinical message is methodological: always test robustness before trusting clean-image metrics.""")


def state_1_prepare_input():
    text("# Execution state 1 / 5 — Prepare PathMNIST input")
    text("**Question:** Which data enter the network, and what changes during preprocessing?")
    train_norm, train_tum = 7886, 12885
    val_norm, val_tum = 877, 1432
    test_norm, test_tum = 741, 1233
    train_n = train_norm + train_tum
    val_n = val_norm + val_tum
    test_n = test_norm + test_tum
    total_n = train_n + val_n + test_n
    example_2x2 = [60, 80, 100, 120]  # @inspect example_2x2
    averaged_value = round(sum(example_2x2) / 4)  # @inspect averaged_value
    text(f"Two retained classes, {total_n:,} patches: train {train_n:,}; validation {val_n:,}; test {test_n:,}.")
    text("The notebook loads 128×128 RGB patches and block-averages each non-overlapping 2×2 region to 64×64.")
    text(f"**Illustrative single-channel 2×2 block:** {example_2x2} → mean intensity {averaged_value}. These are demonstration pixels, not measured PathMNIST pixels.")
    text("CNN inputs are float RGB intensities scaled to [0,1], with shape (3, 64, 64).")
    note("""2:30–3:45 | STATE 1
    We preserve the official PathMNIST train, validation and test partitions, then keep only original class 6 (NORM) and class 8 (TUM). The total is 25,054 patches. The network receives downsampled 64 by 64 colour images to meet the computing budget. Follow the four invented numbers in example_2x2: the 2 by 2 block becomes one averaged pixel. The same operation is applied to the three colour channels. All models receive the same resizing, so the robustness comparison is controlled. Training batches are balanced between classes; validation and test retain their natural class proportions. This step matters because the conclusions are about only two selected tissue classes and a particular input resolution.""")


def state_2_augment_colour():
    text("# Execution state 2 / 5 — Change colour at training time")
    text("**Question:** How can one tissue image be presented with different colours?")
    rgb_before = (0.55, 0.30, 0.45)  # @inspect rgb_before
    hue, saturation, value = colorsys.rgb_to_hsv(*rgb_before)
    shifted_hue = (hue + 0.10) % 1.0  # @inspect shifted_hue
    shifted_saturation = min(1.0, saturation * 1.10)
    rgb_after = colorsys.hsv_to_rgb(shifted_hue, shifted_saturation, value)  # @inspect rgb_after
    text(f"**Illustrative HSV calculation:** RGB {tuple(round(x, 3) for x in rgb_before)} → RGB {tuple(round(x, 3) for x in rgb_after)}.")
    text("Actual strong HSV: hue shift ±0.15 turns; saturation/value ×0.70–1.30.")
    text("Actual weak HSV: hue shift ±0.05 turns; saturation/value ×0.90–1.10.")
    text("Actual H&E augmentation varies estimated haematoxylin/eosin stain strengths: weak ×0.95–1.05; strong ×0.80–1.20.")
    text("The HED reconstruction control applies stain-space conversion without deliberate stain-strength changes.")
    note("""3:45–5:15 | STATE 2
    We are now changing colour, not the image's spatial layout. HSV converts a representative RGB pixel into hue, saturation and value. The trace shifts hue and adjusts saturation, then converts to RGB. This is an illustrative pixel, not one retrieved from the dataset. In the actual experiment, independent training-time transformations are sampled from the displayed weak or strong ranges. H&E augmentation instead converts to a stain representation, changes estimated H and E concentrations, and reconstructs the image. An important methodological point is the reconstruction-only control: colour conversion itself might affect performance, so H&E augmentation should be compared with that control as well as with the unaugmented CNN. All CNN conditions also share the same rotations and flips. This state shows precisely what the proposed intervention changes.""")


def state_3_cnn_prediction():
    text("# Execution state 3 / 5 — CNN feature extraction and prediction")
    text("**Question:** How does a transformed image become a tumour score?")
    patch = [[1, 2, 0], [0, 3, 1], [2, 1, 0]]
    kernel = [[1, 0, -1], [1, 0, -1], [1, 0, -1]]
    response = sum(patch[i][j] * kernel[i][j] for i in range(3) for j in range(3))  # @inspect response
    leaky_response = response if response >= 0 else 0.01 * response
    channels = [16, 32, 32, 64, 64, 128, 128, 256, 256]
    final_spatial_size = 64 // (2 ** 4)
    illustrative_logit = 1.2  # @inspect illustrative_logit
    illustrative_score = 1 / (1 + math.exp(-illustrative_logit))  # @inspect illustrative_score
    text(f"**Teaching-only convolution:** weighted 3×3 sum {response}; after LeakyReLU {leaky_response:.2f}.")
    text(f"**Actual CNN architecture:** 9 convolution layers, channel counts {channels}; stride-2 at convolutions 2, 4, 6, 8.")
    text(f"64×64 input → {final_spatial_size}×{final_spatial_size} feature map with 256 channels → global-average-pool 256 features → one logit.")
    text(f"**Teaching-only sigmoid:** logit {illustrative_logit} → tumour score {illustrative_score:.3f}; NOT a saved model prediction.")
    note("""5:15–7:00 | STATE 3
    A convolution computes weighted sums over local image regions, detecting patterns that later layers can combine into tissue features. The 3 by 3 matrices are a mathematical demonstration; these are not the learned CNN weights. Our actual network has nine convolutions with GroupNorm and LeakyReLU. Downsampling at every second layer reduces spatial size from 64 to 4, while the channels expand to 256. Global average pooling compresses the 4 by 4 maps into 256 features. A classification head, including dropout and LayerNorm, outputs one logit. The sigmoid converts a logit into a score between zero and one, and a threshold of 0.5 can turn this score into a tumour/not-tumour classification. Notice that the sigmoid example is deliberately illustrative. We do not pretend to load or retrain the saved CNN in this LecTrace explanation.""")


def state_4_select_checkpoint():
    text("# Execution state 4 / 5 — Validation loss and model selection")
    text("**Question:** Which epoch's parameters should be evaluated?")
    true_tumour_label = 1
    example_probability = 1 / (1 + math.exp(-1.2))
    example_bce = -math.log(example_probability)  # @inspect example_bce
    actual_epoch_6_bce = 0.0738
    actual_epoch_7_bce = 0.2374
    observed_difference = actual_epoch_7_bce - actual_epoch_6_bce  # @inspect observed_difference
    text(f"Teaching-only binary cross-entropy for y={true_tumour_label}: {example_bce:.4f}.")
    text(f"**Actual notebook logs (weak H&E, seed 2026):** epoch 6 validation BCE {actual_epoch_6_bce}; epoch 7 {actual_epoch_7_bce}.")
    text(f"Validation loss worsened by {observed_difference:.4f} between those epochs.")
    text("Checkpoint rule: retain the **lowest clean-validation BCE among all 8 epochs**; test AUROC is not the checkpoint-selection metric.")
    note("""7:00–8:15 | STATE 4
    Training minimises binary cross-entropy with logits. The example computes a positive-label loss from a made-up sigmoid score. In the actual notebook, the model is evaluated on unchanged validation images after each epoch. The training state chosen for final testing is the one with lowest validation BCE, not necessarily the final epoch. Our saved logs include an instructive case: weak H&E, seed 2026, changes from 0.0738 to 0.2374 between epochs six and seven. This illustrates why checkpoint selection matters and why fair comparisons should not choose a model after seeing held-out test performance. We used the same eight-epoch budget for every augmentation condition.""")


def state_5_evaluate_robustness():
    text("# Execution state 5 / 5 — Quantify RGB robustness")
    text("**Question:** Does the model preserve ranking performance when RGB colour shifts become stronger?")
    control_clean_auc = 0.9928
    weak_hsv_clean_auc = 0.9890
    strong_hsv_clean_auc = 0.9914
    weak_hsv_degradation_gain = 0.0108  # @inspect weak_hsv_degradation_gain
    strong_hsv_degradation_gain = 0.0104  # @inspect strong_hsv_degradation_gain
    weak_hsv_absolute_gain = 0.0071
    strong_hsv_absolute_gain = 0.0090
    worst_control_seed2026 = 0.9209
    worst_strong_hsv_seed2026 = 0.9762
    worst_difference = worst_strong_hsv_seed2026 - worst_control_seed2026
    text("Define **AUROC degradation** = reference-condition AUROC − stressed AUROC.")
    text("Define **robustness gain** = control degradation − augmented-model degradation; positive values mean less degradation.")
    text(f"Clean-test AUROC: unaugmented {control_clean_auc:.4f}, weak HSV {weak_hsv_clean_auc:.4f}, strong HSV {strong_hsv_clean_auc:.4f}.")
    text(f"Mean RGB reduction in degradation: weak HSV +{weak_hsv_degradation_gain:.4f}, strong HSV +{strong_hsv_degradation_gain:.4f}.")
    text(f"Absolute stressed-AUROC advantage: weak HSV +{weak_hsv_absolute_gain:.4f}, strong HSV +{strong_hsv_absolute_gain:.4f}.")
    text(f"Seed-2026 *minimum* RGB AUROC: unaugmented {worst_control_seed2026:.4f}; strong HSV {worst_strong_hsv_seed2026:.4f} (different worst conditions possible).")
    plot({
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "width": 440,
        "height": 180,
        "data": {"values": [
            {"training": "Weak HSV", "gain": 0.0108},
            {"training": "Strong HSV", "gain": 0.0104}
        ]},
        "mark": "bar",
        "encoding": {
            "x": {"field": "training", "type": "nominal", "title": "Training condition"},
            "y": {"field": "gain", "type": "quantitative", "title": "Reduction in RGB AUROC degradation"},
            "tooltip": [{"field": "training"}, {"field": "gain", "format": ".4f"}]
        }
    })
    text("**Uncertainty:** only two paired seeds, synthetic stresses, and no displayed RGB-HSV confidence intervals. Interpret results descriptively.")
    note("""8:15–9:15 | STATE 5
    Each trained model was evaluated on unmodified images, HED identity reconstruction, and 16 non-zero synthetic stresses in two transformation families, two directions, and four levels. A clean AUROC of almost 0.99 is impressive, but it does not tell us how the model reacts to colour shifts. For RGB perturbations, both HSV training variants showed less average AUROC degradation than unaugmented training: +0.0108 for weak HSV and +0.0104 for strong HSV. Importantly, absolute stressed AUROC was also higher, by +0.0071 and +0.0090, respectively. Strong HSV also had the highest worst-condition RGB AUROC in both tested seeds. We should not call these statistically significant findings: the displayed report does not include confidence intervals for these RGB contrasts. The evidence is exploratory, not proof of clinical generalisation.""")


if __name__ == "__main__":
    main()
