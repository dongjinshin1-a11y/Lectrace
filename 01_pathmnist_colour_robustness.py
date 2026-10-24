"""LecTrace | PathMNIST colour robustness (10-minute execution explanation).

Grounding: saved outputs of PathMNIST_Tellez_Inspired_Reviewed_3(4).ipynb
and PathMNIST_Colour_Robustness_Report(3).docx. This lecture does NOT
retrain models or reload checkpoints. Counts, architecture, confusion
matrices, loss values and results are observed. Small pixel, kernel,
and sigmoid calculations explicitly demonstrate the executed operations
rather than claiming to be saved trained-model inferences.
"""
import colorsys
import math
from lectrace import text, note, plot


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
    note('9:15–10:00 | Our unaugmented model had the highest mean clean AUROC, but that did not make it most robust. Strong HSV had the best *minimum* RGB AUROC in both observed seeds. This is evidence for an augmentation-by-stress interaction, not a claim of general stain invariance. Two seeds, synthetic stresses and earlier test-informed revisions make this exploratory. End on the clinical lesson: clean accuracy alone is not a deployment safety evaluation.')


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
    illustrative_block = (60, 80, 100, 120)  # @inspect illustrative_block
    averaged_pixel = round(sum(illustrative_block) / 4)  # @inspect averaged_pixel
    text(f'**Implemented preprocessing:** 128×128 RGB → 64×64 by 2×2 block averaging; illustrative one-channel block {illustrative_block} → {averaged_pixel}. The four intensities are **not extracted from PathMNIST**.')
    text('All six CNNs use the same partitions and 64×64 inputs scaled to [0, 1]. Balanced training batches contain 32 NORM + 32 TUM; validation/test retain natural prevalence.')
    note('2:00–3:00 | Point to the measured class counts, including 1,974 held-out test patches. The classes are original PathMNIST labels 6 and 8, rather than all nine tissues. Follow the small 2×2 averaging computation: it demonstrates exactly how the real code downsamples each colour channel, but these four example intensities are invented. Balance only the training sampler, not the test set. Mention that PathMNIST source imagery was already colour-normalised; no new offline stain normalisation was added.')


def state_2_colour_transforms():
    text('# State 2 / 5 | What is the intervention, and what is the stress test?')
    text('**Training intervention:** weak H&E multiplies estimated H/E concentrations by 0.95–1.05; strong H&E by 0.80–1.20. Weak HSV shifts hue ±0.05 turns and scales S/V ×0.90–1.10; strong HSV uses ±0.15 turns and ×0.70–1.30.')
    rgb_example = (0.55, 0.30, 0.45)  # @inspect rgb_example
    hue, saturation, value = colorsys.rgb_to_hsv(*rgb_example)
    hsv_changed = colorsys.hsv_to_rgb((hue + 0.10) % 1, min(1, saturation * 1.10), value)  # @inspect hsv_changed
    text('**Executed teaching example:** RGB → HSV → hue-shifted RGB. These input values are illustrative; the conversion and hue wrapping are real Python operations.')
    severity = 4  # @inspect severity
    delta = 0.04 * severity  # @inspect delta
    gains = (1 + delta, 1.0, 1 - delta)  # @inspect gains
    stressed_rgb = tuple(min(1, max(0, c * g)) for c, g in zip(rgb_example, gains))  # @inspect stressed_rgb
    text(f'**Actual RGB stress recipe, red-dominant level {severity}:** channel gains {gains} → illustrative pixel {tuple(round(x, 3) for x in stressed_rgb)}. Increasing severity changes colour, not spatial tissue structure.')
    plot({'$schema':'https://vega.github.io/schema/vega-lite/v5.json','width':450,'height':145,
          'data':{'values':[{'severity':s,'channel':ch,'gain':round(g,2)} for s in range(5) for ch,g in [('R',1+.04*s),('G',1),('B',1-.04*s)]]},
          'mark':{'type':'line','point':True},'encoding':{'x':{'field':'severity','type':'quantitative','title':'RGB stress severity','axis':{'tickMinStep':1}},
          'y':{'field':'gain','type':'quantitative','title':'Channel multiplier','scale':{'domain':[.8,1.2]}},
          'color':{'field':'channel','type':'nominal'},'tooltip':[{'field':'severity'},{'field':'channel'},{'field':'gain'}]}})
    text('**Fair comparator:** H&E augmentation vs HED conversion-and-reconstruction at factors (1,1). HSV augmentation vs the unaugmented CNN. H&E stress level 0 is reconstructed RGB; RGB stress level 0 is untouched RGB.')
    note('3:00–4:40 | Distinguish what the model *sees while training* (randomly sampled H&E or HSV augmentation at 80% probability) from how *robustness is tested* (fixed RGB channel-gain or H&E shifts). The small HSV example runs a real coloursys transformation but is not a sampled PathMNIST pixel. For the RGB stress, execute delta = 0.04 × level. At level four R goes up 16%, B goes down 16%. This is the exact form used in the notebook. Explain the reconstruction control: HED conversion can itself modify RGB, so comparing H&E training only with the untouched model may conflate two effects.')


def state_3_cnn_and_predictions():
    text('# State 3 / 5 | From an augmented patch to a tumour score')
    channels = (16, 32, 32, 64, 64, 128, 128, 256, 256)  # @inspect channels
    sizes = (64, 32, 32, 16, 16, 8, 8, 4, 4)  # @inspect sizes
    text('**Actually trained:** nine 3×3 convolutions (stride 2 at layers 2, 4, 6, 8), each with GroupNorm(8) + LeakyReLU. Feature tensor: 256 × 4 × 4; global average pooling → 256 features; dropout(0.5) → Linear(256,256) → LayerNorm → LeakyReLU → 1 logit.')
    text('**Model size:** 1,243,985 trainable parameters, trained from scratch with binary cross-entropy. A sigmoid maps the logit to a score in [0,1].')
    illustrative_logit = 1.2  # @inspect illustrative_logit
    illustrative_score = 1 / (1 + math.exp(-illustrative_logit))  # @inspect illustrative_score
    text(f'**Math demonstration only:** sigmoid({illustrative_logit}) = {illustrative_score:.3f}. This is **not** a checkpoint prediction and is not calibrated as a clinical probability.')
    norm_total, tumour_total = 741, 1233
    fp_control, fn_control = 31, 41  # @inspect fp_control fn_control
    fp_hsv, fn_hsv = 68, 24  # @inspect fp_hsv fn_hsv
    control_recall = (tumour_total - fn_control) / tumour_total  # @inspect control_recall
    hsv_recall = (tumour_total - fn_hsv) / tumour_total  # @inspect hsv_recall
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
    note('4:40–6:10 | Point out that the *architecture* and parameter count are from the actual PyTorch notebook. Trace the progression from image through convolutional feature maps, global pooling and a one-logit classifier. There are no saved weights in this presentation, so the 1.2-logit example explains sigmoid only. Then show genuinely observed predictions: both models rank clean examples very well, yet at the fixed threshold their tumour recall and false-positive behaviour differ. A stronger recall does not automatically make a model clinically preferable without threshold selection and calibration.')


def state_4_training_and_validation():
    text('# State 4 / 5 | How were checkpoints selected fairly?')
    text('**Matched budget:** 6 conditions × 2 seeds (2026, 2027) × 8 epochs; 325 balanced batches per epoch; batch size 64. Adam learning rate 0.0003; gradient clipping 1.0.')
    val_bce_epoch6 = 0.0738  # @inspect val_bce_epoch6
    val_bce_epoch7 = 0.2374  # @inspect val_bce_epoch7
    loss_increase = val_bce_epoch7 - val_bce_epoch6  # @inspect loss_increase
    text(f'**Actual log example:** weak H&E, seed 2026: clean validation BCE at epoch 6 = {val_bce_epoch6:.4f}, epoch 7 = {val_bce_epoch7:.4f} (worsened by {loss_increase:.4f}).')
    text('**Selection rule:** after all eight epochs, retain whichever checkpoint had the **lowest clean validation BCE**. This example does **not** assert epoch 6 was globally optimal. Test scores never select checkpoints in the revised run.')
    plot({'$schema':'https://vega.github.io/schema/vega-lite/v5.json','width':460,'height':200,
          'data':{'values':[{'method':'Majority','auc':.5}, {'method':'Colour-only LR','auc':.9320},
                            {'method':'Unaugmented CNN','auc':.9928},{'method':'HED reconstruction','auc':.9877},
                            {'method':'Weak H&E','auc':.9926},{'method':'Strong H&E','auc':.9926},
                            {'method':'Weak HSV','auc':.9890},{'method':'Strong HSV','auc':.9914}]},
          'mark':{'type':'bar'},'encoding':{'x':{'field':'auc','type':'quantitative','title':'Clean-test AUROC','scale':{'domain':[0,1]}},
          'y':{'field':'method','type':'nominal','title':None,'sort':['Majority','Colour-only LR','Unaugmented CNN','HED reconstruction','Weak H&E','Strong H&E','Weak HSV','Strong HSV']},
          'tooltip':[{'field':'method'},{'field':'auc','format':'.4f'}]}})
    text('**Observed clean AUROC:** unaugmented 0.9928 (highest mean), weak H&E 0.9926, strong H&E 0.9926, weak HSV 0.9890, strong HSV 0.9914. High *clean* AUROC alone cannot answer the robustness question.')
    note('6:10–7:25 | A valid model comparison uses the same architecture, sampling schedule and budget. Demonstrate the real validation loss fluctuation between epochs six and seven. We select each model using minimum validation BCE across *all* eight epochs; we do not pick the best test AUROC. The chart uses observed saved clean-test means, with majority and 30-feature colour-only logistic regression as reference baselines. The colour-only AUROC of 0.932 shows global colour carries predictive signal, but does not prove that the CNN learned morphology independently of colour.')


def state_5_robustness_evaluation():
    text('# State 5 / 5 | Does robustness survive synthetic colour shifts?')
    text('**Definition:** degradation = reference AUROC − stressed AUROC. Augmentation gain = comparator degradation − augmented-model degradation. A positive gain means *less* degradation, not necessarily higher absolute stressed AUROC.')
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
