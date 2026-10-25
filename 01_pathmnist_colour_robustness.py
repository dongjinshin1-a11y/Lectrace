"""LecTrace | PathMNIST colour robustness: five states, ten minutes.

Grounding: PathMNIST_Tellez_Inspired_.ipynb saved outputs and
PathMNIST_Colour_Robustness_Report(4).docx, reviewed 25 October 2026.
Images are the notebook's embedded figures. Results are saved observations;
shape arithmetic is derived from the implemented architecture. No invented
pixel intensities, logits or trained activations are used. This presentation
does not load trained weights or claim to replay a trained forward pass.
AI assisted with this presentation; the researcher must review and understand it.
"""
import json
from pathlib import Path
from lectrace import text, note, plot

FIGURES = json.loads((Path(__file__).parent / "notebook_figures.json").read_text())


def main():
    text('# When colour changes, can a pathology CNN still recognise tumour tissue?')
    text('## PathMNIST | H&E / HSV augmentation | 10-minute execution explanation')
    text('**Clinical motivation.** H&E colour varies with staining and image acquisition. A high AUROC on familiar images does not guarantee reliable predictions after colour changes.')
    text('**Two research questions.** (1) Does colour augmentation preserve clean NORM-versus-TUM discrimination? (2) Does it reduce AUROC degradation under controlled H&E and RGB shifts?')
    text('**Design:** 6 training conditions × 2 paired seeds × 8 epochs, tested against original images, HED identity, and 16 synthetic colour stresses.')
    note('0:00–2:00 | Set the clinical scene: NORM means normal colon mucosa; TUM means colorectal tumour epithelium. This is patch classification, not patient diagnosis. Motivate why scanners and stains matter, but emphasise that this project tests synthetic perturbations rather than real inter-hospital generalisation. Explain the two RQs. Introduce the six conditions: unaugmented, HED reconstruction, weak/strong H&E, weak/strong HSV. Then walk through five computational states, not every training iteration.')

    state_1_input_and_data()
    state_2_colour_transforms()
    state_3_cnn_and_predictions()
    state_4_training_and_validation()
    state_5_robustness_evaluation()

    text('# Conclusion | Colour-robustness depends on *which* colour shift')
    text('**Observed:** HSV training reduced degradation under synthetic RGB shifts in both seeds and improved absolute stressed AUROC. H&E training gave only tiny gains relative to HED reconstruction, not consistent gains relative to unaugmented training.')
    text('**Not established:** statistical significance of RGB-HSV effects, real multi-hospital robustness, calibration or clinical diagnostic usefulness.')
    text('**Next step:** lock the protocol, run more seeds, report RGB-effect confidence intervals and test a genuinely external, slide/patient-aware cohort.')
    note('9:15–10:00 | Our unaugmented model had the highest mean clean AUROC, but that did not make it most robust. Strong HSV had the best *minimum* RGB AUROC in both observed seeds. The descriptive pattern depends on augmentation and stress family; no formal interaction test or general stain-invariance claim is made. Two seeds, synthetic stresses and earlier test-informed revisions make this exploratory. End on the clinical lesson: clean accuracy alone is not a deployment safety evaluation.')


def state_1_input_and_data():
    text('# State 1 / 5 | What exactly enters the CNN?')
    train_norm, train_tum = 7886, 12885  # @inspect train_norm train_tum
    val_norm, val_tum = 877, 1432
    test_norm, test_tum = 741, 1233  # @inspect test_norm test_tum
    total = sum((train_norm, train_tum, val_norm, val_tum, test_norm, test_tum))  # @inspect total
    tumour_share = (train_tum + val_tum + test_tum) / total  # @inspect tumour_share
    text(f'**Actual dataset after filtering:** {total:,} patches; train {train_norm + train_tum:,}, validation {val_norm + val_tum:,}, test {test_norm + test_tum:,}. TUM prevalence: {tumour_share:.1%}.')
    plot({'$schema':'https://vega.github.io/schema/vega-lite/v5.json','width':450,'height':155,
          'data':{'values':[{'split':'Train','class':'NORM','count':7886},{'split':'Train','class':'TUM','count':12885},
                            {'split':'Validation','class':'NORM','count':877},{'split':'Validation','class':'TUM','count':1432},
                            {'split':'Test','class':'NORM','count':741},{'split':'Test','class':'TUM','count':1233}]},
          'mark':'bar','encoding':{'x':{'field':'split','type':'nominal','sort':['Train','Validation','Test'],'title':'Official split'},
                                   'y':{'field':'count','type':'quantitative','title':'Patches'},
                                   'color':{'field':'class','type':'nominal','title':'Class'},
                                   'xOffset':{'field':'class'},'tooltip':[{'field':'class'},{'field':'split'},{'field':'count'}]}})
    source_size, input_size = 128, 64  # @inspect source_size input_size
    pixels_per_output = (source_size // input_size) ** 2  # @inspect pixels_per_output
    text(f'**Actual preprocessing:** each output pixel averages {pixels_per_output} source pixels within the same colour channel, rounds to uint8, then divides by 255 before CNN input.')
    plot({'width':500, 'height':210, 'data':{'values':[{'url':FIGURES['training_examples']}]}, 'mark':{'type':'image','width':500,'height':210}, 'encoding':{'x':{'value':250},'y':{'value':105},'url':{'field':'url','type':'nominal'}}})
    text('**Saved notebook figure:** real NORM (top) and TUM (bottom) training patches after resizing. These are not test examples or newly generated images.')
    text('All six CNNs use the same partitions and 64×64 inputs scaled to [0, 1]. Balanced training batches contain 32 NORM + 32 TUM; validation/test retain natural prevalence.')
    note('2:00–3:00 | What happens: filter original labels 6 and 8, resize 128 to 64, and scale intensities. What changes: four source pixels contribute to each output pixel. Why it matters: lower compute cost at the expense of spatial detail. Show the actual training-patch figure. Explain 32+32 sampling only during training; evaluation retains natural prevalence. Source images were already colour-normalised.')


def state_2_colour_transforms():
    text('# State 2 / 5 | What is the intervention, and what is the stress test?')
    text('**Training intervention:** weak H&E multiplies estimated H/E concentrations by 0.95–1.05; strong H&E by 0.80–1.20. Weak HSV shifts hue ±0.05 turns and scales S/V ×0.90–1.10; strong HSV uses ±0.15 turns and ×0.70–1.30.')
    plot({'width':500, 'height':210, 'data':{'values':[{'url':FIGURES['he_examples']}]}, 'mark':{'type':'image','width':500,'height':210}, 'encoding':{'x':{'value':250},'y':{'value':105},'url':{'field':'url','type':'nominal'}}})
    text('**Saved notebook transformation figure:** original training patches and H/E multipliers (1.15,0.95), (0.95,1.15), (0.88,0.88), (1.12,1.12). These are qualitative demonstrations, not the fixed evaluation stress levels or measured laboratory distributions.')
    severity = 4  # @inspect severity
    delta = 0.04 * severity  # @inspect delta
    gains = (1 + delta, 1.0, 1 - delta)  # @inspect gains
    he_delta = 0.08 * severity  # @inspect he_delta
    he_factors = (1 + he_delta, 1 - he_delta / 2)  # @inspect he_factors
    text(f'**Executed stress-parameter calculations, level {severity}:** RGB red-dominant gains = {gains}; H-dominant H/E multipliers = {he_factors}. These reproduce the notebook operators; RGB values are clipped to [0,1].')
    text('**What changes:** colour intensities; pixel locations stay fixed. The stressed appearance is synthetic and is not proof of real scanner or laboratory variation.')
    plot({'$schema':'https://vega.github.io/schema/vega-lite/v5.json','width':450,'height':145,
          'data':{'values':[{'severity':s,'channel':ch,'gain':round(g,2)} for s in range(5) for ch,g in [('R',1+.04*s),('G',1),('B',1-.04*s)]]},
          'mark':{'type':'line','point':True},'encoding':{'x':{'field':'severity','type':'quantitative','title':'RGB stress severity','axis':{'tickMinStep':1}},
          'y':{'field':'gain','type':'quantitative','title':'Channel multiplier','scale':{'domain':[.8,1.2]}},
          'color':{'field':'channel','type':'nominal'},'tooltip':[{'field':'severity'},{'field':'channel'},{'field':'gain'}]}})
    text('**Fair comparator:** H&E augmentation vs HED conversion-and-reconstruction at factors (1,1). HSV augmentation vs the unaugmented CNN. H&E stress level 0 is reconstructed RGB; RGB stress level 0 is untouched RGB.')
    note('3:00–4:40 | What happens: training samples random colour changes with probability 0.8; testing uses fixed stresses. What changes: execute RGB gain and H/E factor calculations at severity four. Why it matters: H increases to 1.32, exceeding the strong-training maximum of 1.20. Explain HED reconstruction as the reference for H&E stresses; RGB uses untouched input. The saved figure demonstrates transformations on training patches, not unseen clinical data.')


def state_3_cnn_and_predictions():
    text('# State 3 / 5 | CNN computation and observed prediction errors')
    channels = (16, 32, 32, 64, 64, 128, 128, 256, 256)  # @inspect channels
    height, width, in_channels = 64, 64, 3  # @inspect height width in_channels
    layer_shapes = []  # @inspect layer_shapes
    convolution_parameters = 0  # @inspect convolution_parameters
    for layer, out_channels in enumerate(channels, start=1):
        stride = 2 if layer in (2, 4, 6, 8) else 1
        height = (height + 2 - 3) // stride + 1  # 3×3 kernel, padding 1
        width = (width + 2 - 3) // stride + 1
        layer_shapes.append((layer, out_channels, height, width))
        convolution_parameters += (out_channels * in_channels * 3 * 3) + (2 * out_channels)  # conv weights, GroupNorm scale/bias
        in_channels = out_channels
    head_parameters = (256 * 256 + 256) + (2 * 256) + (256 + 1)  # linear, LayerNorm, output linear
    total_parameters = convolution_parameters + head_parameters  # @inspect total_parameters
    assert (height, width, in_channels) == (4, 4, 256)
    assert total_parameters == 1_243_985, 'Architecture does not match the notebook'
    spatial_positions = height * width  # @inspect spatial_positions
    shape_trace = ' → '.join(f'L{layer}: {c}×{h}×{w}' for layer, c, h, w in layer_shapes)
    text(f'**Executable CNN shape trace (channels × height × width):** 3×64×64 → {shape_trace}. Stride 2 at convolution layers 2, 4, 6 and 8 halves the spatial dimensions.')
    text(f'**Global average pooling:** the final 256 channels each contain {spatial_positions} spatial values, averaged into a 256-element feature vector. These are architecture-derived shapes, **not** saved trained activations.')
    text('**Actual model design:** nine 3×3 convolutions with GroupNorm(8) + LeakyReLU; dropout(0.5) → Linear(256,256) → LayerNorm → LeakyReLU → one logit.')
    text(f'**Verified model-size arithmetic:** {convolution_parameters:,} convolution/GroupNorm parameters + {head_parameters:,} classifier/LayerNorm parameters = **{total_parameters:,}** trainable parameters. The model was trained from scratch using binary cross-entropy.')
    text('**Prediction rule in the notebook:** score = sigmoid(logit); score ≥ 0.5 predicts TUM, otherwise NORM. Dropout is disabled in evaluation. A score is not an established clinical probability.')
    text('**Evidence boundary:** trained weights and per-patch logits are not embedded here. The following calculation uses the actual saved confusion counts rather than an invented example score.')
    norm_total, tumour_total = 741, 1233
    fp_control, fn_control = 31, 41  # @inspect fp_control fn_control
    fp_hsv, fn_hsv = 68, 24  # @inspect fp_hsv fn_hsv
    control_recall = (tumour_total - fn_control) / tumour_total  # @inspect control_recall
    hsv_recall = (tumour_total - fn_hsv) / tumour_total  # @inspect hsv_recall
    control_specificity = (norm_total - fp_control) / norm_total  # @inspect control_specificity
    hsv_specificity = (norm_total - fp_hsv) / norm_total  # @inspect hsv_specificity
    text(f'**Calculated from saved counts:** specificity falls from {control_specificity:.1%} to {hsv_specificity:.1%} as tumour recall rises. At the chosen threshold, the model would flag more normal patches as TUM.')
    text(f'**Real test-set predictions at threshold 0.5, seed 2026:** unaugmented: 31 normal patches labelled tumour (FP), 41 tumours missed (FN). Strong HSV: 68 FP, 24 FN. Tumour recall {control_recall:.1%} → {hsv_recall:.1%}; false positives also increase.')
    plot({'$schema':'https://vega.github.io/schema/vega-lite/v5.json','width':450,'height':145,
          'data':{'values':[{'model':'Unaugmented','error':'FP: NORM called TUM','patches':31},
                            {'model':'Unaugmented','error':'FN: missed TUM','patches':41},
                            {'model':'Strong HSV','error':'FP: NORM called TUM','patches':68},
                            {'model':'Strong HSV','error':'FN: missed TUM','patches':24}]},
          'mark':'bar','encoding':{'x':{'field':'model','type':'nominal','title':'Seed 2026 / clean test'},
                                  'y':{'field':'patches','type':'quantitative','title':'Misclassified patches'},
                                  'color':{'field':'error','type':'nominal'},'xOffset':{'field':'error'},
                                  'tooltip':[{'field':'model'},{'field':'error'},{'field':'patches'}]}})
    note('4:40–6:10 | What happens: convolution extracts spatial features, pooling reduces each channel to one value, and a dense head produces a logit. What changes: execute the full nine-layer shape/parameter calculation from the actual architecture, then compute recall and specificity from saved seed-2026 predictions. Why it matters: a threshold changes the missed-tumour versus unnecessary-review trade-off. Clearly identify this as architecture arithmetic plus observed evaluation, not a replay of trained activations.')


def state_4_training_and_validation():
    text('# State 4 / 5 | How were checkpoints selected fairly?')
    text('**Matched budget:** 6 conditions × 2 seeds (2026, 2027) × 8 epochs; 325 balanced batches per epoch; batch size 64. Adam learning rate 0.0003; gradient clipping 1.0.')
    val_bce_epoch6 = 0.0738  # @inspect val_bce_epoch6
    val_bce_epoch7 = 0.2374  # @inspect val_bce_epoch7
    loss_increase = val_bce_epoch7 - val_bce_epoch6  # @inspect loss_increase
    text(f'**Actual log example:** weak H&E, seed 2026: clean validation BCE at epoch 6 = {val_bce_epoch6:.4f}, epoch 7 = {val_bce_epoch7:.4f} (worsened by {loss_increase:.4f}).')
    validation_bce = [0.4420, 0.2259, 0.1615, 0.1404, 0.1488, 0.0738, 0.2374, 0.1023]  # @inspect validation_bce
    selected_epoch = min(range(1, 9), key=lambda epoch: validation_bce[epoch - 1])  # @inspect selected_epoch
    text(f'**Replay of checkpoint selection from saved rounded logs:** minimum validation BCE selects epoch {selected_epoch} for weak H&E / seed 2026, consistent with the saved checkpoint summary. The training code selects using unrounded BCE across all eight epochs, never the best test score.')
    plot({'$schema':'https://vega.github.io/schema/vega-lite/v5.json','width':450,'height':145,
          'data':{'values':[{'epoch':i + 1,'val_bce':v} for i,v in enumerate(validation_bce)]},
          'mark':{'type':'line','point':True},
          'encoding':{'x':{'field':'epoch','type':'quantitative','title':'Epoch (weak H&E, seed 2026)','axis':{'tickMinStep':1}},
                      'y':{'field':'val_bce','type':'quantitative','title':'Clean validation BCE','scale':{'zero':True}},
                      'tooltip':[{'field':'epoch'},{'field':'val_bce','format':'.4f'}]}})
    text(f'**Read this saved training curve:** epoch {selected_epoch} has the lowest validation BCE ({min(validation_bce):.4f}). Epoch 7 worsened despite a lower training loss; this does not by itself identify a software bug.')
    text('**Training stability:** fluctuating validation loss prompted investigation; it is not by itself a confirmed software bug. GroupNorm/LayerNorm, finite-gradient checks and batch-independence tests supported the revised implementation; their individual effects were not isolated.')
    plot({'$schema':'https://vega.github.io/schema/vega-lite/v5.json','width':460,'height':200,
          'data':{'values':[{'method':'Majority','auc':.5}, {'method':'Colour-only LR','auc':.9320},
                            {'method':'Unaugmented CNN','auc':.9928},{'method':'HED reconstruction','auc':.9877},
                            {'method':'Weak H&E','auc':.9926},{'method':'Strong H&E','auc':.9926},
                            {'method':'Weak HSV','auc':.9890},{'method':'Strong HSV','auc':.9914}]},
          'mark':{'type':'bar'},'encoding':{'x':{'field':'auc','type':'quantitative','title':'Clean-test AUROC','scale':{'domain':[0,1]}},
          'y':{'field':'method','type':'nominal','title':None,'sort':['Majority','Colour-only LR','Unaugmented CNN','HED reconstruction','Weak H&E','Strong H&E','Weak HSV','Strong HSV']},
          'tooltip':[{'field':'method'},{'field':'auc','format':'.4f'}]}})
    text('**Observed clean AUROC:** unaugmented 0.9928 (highest mean), weak H&E 0.9926, strong H&E 0.9926, weak HSV 0.9890, strong HSV 0.9914. High *clean* AUROC alone cannot answer the robustness question.')
    note('6:10–7:25 | A valid model comparison uses the same architecture, sampling schedule and budget. Execute the minimum-loss selection over all eight saved values, rather than selecting the final epoch. The fluctuation between epochs six and seven is not proof of a bug. We select each model using minimum validation BCE across *all* eight epochs; we do not pick the best test AUROC. Point to the saved validation-BCE curve (epoch 6 is best) before reading the clean-test AUROC chart. The AUROC chart uses observed saved means, with majority and 30-feature colour-only logistic regression as reference baselines. The colour-only AUROC of 0.932 shows global colour carries predictive signal, but does not prove that the CNN learned morphology independently of colour.')


def state_5_robustness_evaluation():
    text('# State 5 / 5 | Does robustness survive synthetic colour shifts?')
    text('**Definition:** degradation = reference AUROC − stressed AUROC. Augmentation gain = comparator degradation − augmented-model degradation. A positive gain means *less* degradation, not necessarily higher absolute stressed AUROC.')
    strong_seed_gains = (0.0146, 0.0062)  # @inspect strong_seed_gains
    mean_from_rounded_seeds = sum(strong_seed_gains) / len(strong_seed_gains)  # @inspect mean_from_rounded_seeds
    text(f'**Aggregation example:** strong HSV RGB gains are positive in both saved seeds; their rounded values average to {mean_from_rounded_seeds:.4f}. Final reported means below come from the notebook summary, calculated before rounding.')
    weak_gain, strong_gain = 0.0108, 0.0104  # @inspect weak_gain strong_gain
    weak_absolute, strong_absolute = 0.0071, 0.0090  # @inspect weak_absolute strong_absolute
    text(f'**Observed averaged RGB contrasts vs unaugmented:** weak HSV gain +{weak_gain:.4f}, absolute stressed AUROC +{weak_absolute:.4f}; strong HSV gain +{strong_gain:.4f}, absolute stressed AUROC +{strong_absolute:.4f}. Each averages both directions, levels 1–4, and the two seeds.')
    plot({'$schema':'https://vega.github.io/schema/vega-lite/v5.json','width':450,'height':170,
          'data':{'values':[{'seed':'2026','model':'Unaugmented','worst':.9209},
                            {'seed':'2026','model':'Weak HSV','worst':.9660},
                            {'seed':'2026','model':'Strong HSV','worst':.9762},
                            {'seed':'2027','model':'Unaugmented','worst':.9601},
                            {'seed':'2027','model':'Weak HSV','worst':.9823},
                            {'seed':'2027','model':'Strong HSV','worst':.9833}]},
          'mark':{'type':'point','filled':True,'size':125},
          'encoding':{'x':{'field':'seed','type':'nominal','title':'Training seed'},
          'y':{'field':'worst','type':'quantitative','scale':{'domain':[.90,1]},'title':'Minimum AUROC across RGB stresses (zoomed)'},
          'color':{'field':'model','type':'nominal','title':'Training'},
          'xOffset':{'field':'model'},'tooltip':[{'field':'seed'},{'field':'model'},{'field':'worst','format':'.4f'}]}})
    text('**Observed minimum RGB-stressed AUROC:** Strong HSV is highest in both seeds (2026: 0.9762; 2027: 0.9833). The minimum may occur at a different stress for each model; the chart uses a deliberately zoomed y-axis.')
    text('**H&E result for context:** weak/strong H&E improved mean H&E degradation by only +0.0006/+0.0004 **vs reconstruction**; comparisons against the unaugmented CNN were approximately −0.0005/−0.0007.')
    text('**Uncertainty:** two seeds; synthetic, not real-lab stresses; prior test-informed revisions; patch-bootstrap intervals (H&E gains were small) do not capture patient/training uncertainty. RGB-HSV confidence intervals were computed in the notebook but are not displayed in the supplied outputs, so do **not** claim statistical significance.')
    note('7:25–9:15 | Compare clean AUROC with the tested stress results. Read out the paired differences carefully: weak HSV +0.0108 and strong HSV +0.0104 in *degradation reduction*, plus positive absolute stressed-AUROC differences. On the zoomed plot, strong HSV has the highest minimum RGB AUROC in both seeds; identify the truncation and avoid overclaiming magnitudes. Then contrast the very small H&E effect relative to the *weaker HED reconstruction comparator*, which does not establish an improvement over unaugmented training. Do not equate the conditional patch-bootstrap result for H&E with proof of clinical significance. HSV RGB effects are descriptive because displayed uncertainty is absent.')


if __name__ == '__main__':
    main()
