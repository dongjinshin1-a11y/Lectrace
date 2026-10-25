"""Export a REAL inference trace from the user's saved PathMNIST PyTorch checkpoint.

Run this file in Google Colab; it loads an EXISTING best checkpoint and an actual
official test patch. It DOES NOT train any model. The resulting JSON can be
uploaded as real_forward_trace.json to the LecTrace GitHub repository.

Quick Colab use (same Google account as the completed notebook):
  !pip -q install medmnist pillow
  !python colab_export_real_forward.py
The original best checkpoint and experiment_config.json MUST both exist.
The notebook's Drive location is used by default. Mount Google Drive when prompted.
"""
from pathlib import Path
from io import BytesIO
import base64
import hashlib
import json
import os

import numpy as np
import torch
from torch import nn
from PIL import Image
from medmnist import PathMNIST


MODEL_MODE = "hsv_strong"
SEED = 2026
PATCH_INDEX = 0                       # zero-based index after NORM/TUM filtering
EXPERIMENT_ID = "tellez-inspired-gn-v5-budget64"
ROOT = Path(os.environ.get(
    "PATHMNIST_ROOT", "/content/drive/MyDrive/PathMNIST_HE_Robustness"))
RUN_OUTPUT = ROOT / EXPERIMENT_ID / "final"
DEST = Path("real_forward_trace.json")


class TissueCNN(nn.Module):
    """Identical structure to the executed PathMNIST notebook."""
    def __init__(self):
        super().__init__()
        layers = []
        incoming = 3
        for number, outgoing in enumerate(
            (16, 32, 32, 64, 64, 128, 128, 256, 256), start=1
        ):
            layers.extend([
                nn.Conv2d(incoming, outgoing, 3,
                          stride=2 if number % 2 == 0 else 1,
                          padding=1, bias=False),
                nn.GroupNorm(8, outgoing),
                nn.LeakyReLU(0.01, inplace=True),
            ])
            incoming = outgoing
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.5),
            nn.Linear(256, 256), nn.LayerNorm(256),
            nn.LeakyReLU(0.01, inplace=True), nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


def png_data_url(rgb_uint8):
    stream = BytesIO()
    Image.fromarray(rgb_uint8, mode="RGB").save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def load_real_test_patch():
    # This replicates the notebook's binary subset order and its 2x2 area resize.
    cache = Path.home() / ".medmnist"
    cache.mkdir(parents=True, exist_ok=True)
    source_name = "pathmnist_128.npz"
    if not (cache / source_name).exists():
        backup = ROOT / "dataset_cache" / source_name
        if backup.exists():
            import shutil
            shutil.copy2(backup, cache / source_name)
    source = PathMNIST(split="test", size=128, root=str(cache), download=True)
    labels = source.labels.reshape(-1)
    indices = np.flatnonzero(np.isin(labels, (6, 8)))
    assert 0 <= PATCH_INDEX < len(indices)
    ix = int(indices[PATCH_INDEX])
    full_size_patch = np.asarray(source.imgs[ix], dtype=np.uint8)
    resized = np.rint(full_size_patch.reshape(64, 2, 64, 2, 3)
                       .mean(axis=(1, 3))).astype(np.uint8)
    target_label = "TUM" if int(labels[ix]) == 8 else "NORM"
    return resized, target_label, ix


@torch.inference_mode()
def trace_case(model, image, condition):
    observations = []
    hooks = []

    def record(layer_number):
        def hook(module, inputs, output):
            a = output.detach().float()
            observations.append({
                "name": f"conv{layer_number}",
                "shape": list(a.shape),
                "mean": float(a.mean().item()),
                "std": float(a.std(unbiased=False).item()),
                "min": float(a.min().item()),
                "max": float(a.max().item()),
                "sample_values": [float(v) for v in a.flatten()[:4].tolist()],
            })
        return hook

    for num in range(1, 10):
        conv = model.features[(num - 1) * 3]
        hooks.append(conv.register_forward_hook(record(num)))
    try:
        logit = float(model(image).item())
    finally:
        for hook in hooks:
            hook.remove()
    score = float(torch.sigmoid(torch.tensor(logit)).item())
    assert len(observations) == 9 and np.isfinite(logit)
    return {
        "condition": condition,
        "features": observations,
        "logit": logit,
        "score": score,
        "prediction": "TUM" if score >= 0.5 else "NORM",
    }


def main():
    # Mount only when executing in a standalone Google Colab environment.
    if Path("/content").exists() and not ROOT.exists():
        try:
            from google.colab import drive
            drive.mount("/content/drive")
        except ImportError:
            pass

    checkpoint = RUN_OUTPUT / f"cnn_{MODEL_MODE}_seed{SEED}_best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Trained checkpoint missing: {checkpoint}\\n"
            "Open the Drive account used for the original experiment. "
            "Do not rerun training just to construct a made-up trace."
        )
    # Only load a checkpoint produced by your own trusted notebook.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not (saved.get("mode") == MODEL_MODE and int(saved.get("seed")) == SEED):
        raise ValueError("Checkpoint mode / seed mismatch")
    # The notebook changes CONFIG_SIGNATURE after fingerprinting the actual
    # training code/data and again after locking the eight-epoch budget.
    # The value printed in its first configuration cell is NOT the final value.
    config_path = RUN_OUTPUT / "experiment_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Missing {config_path}. Export experiment_config.json from the "
            "original completed notebook run; do not invent a signature."
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_signature = config.get("config_signature")
    if (not expected_signature or config.get("run_mode") != "final" or
            config.get("version") != EXPERIMENT_ID):
        raise ValueError("Experiment metadata is incomplete or mismatched")
    if saved.get("signature") != expected_signature:
        raise ValueError(
            f"Trained checkpoint signature {saved.get('signature')!r} does "
            f"not match final experiment metadata {expected_signature!r}"
        )
    model = TissueCNN().eval()
    model.load_state_dict(saved["model"], strict=True)
    assert sum(p.numel() for p in model.parameters()) == 1_243_985

    raw_patch, label, original_source_ix = load_real_test_patch()
    clean = (torch.from_numpy(raw_patch.copy()).permute(2, 0, 1)
             .unsqueeze(0).float() / 255.0)
    # Identical to original notebook's RGB red-dominant severity-4 operator.
    gains = torch.tensor([1.16, 1.0, 0.84]).view(1, 3, 1, 1)
    stressed = (clean * gains).clamp(0, 1)
    stressed_display = (stressed[0].permute(1, 2, 0).numpy() * 255
                        ).round().clip(0, 255).astype(np.uint8)

    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    trace = {
        "source": "saved_trained_checkpoint",
        "experiment": EXPERIMENT_ID,
        "model_mode": MODEL_MODE,
        "seed": SEED,
        "patch_index": PATCH_INDEX,
        "source_test_index": original_source_ix,
        "true_label": label,
        "checkpoint_sha256": digest,
        "config_signature": expected_signature,
        "provenance": "Real PyTorch evaluation-mode inference; no training or synthetic logits",
        "patch_png_data_url": png_data_url(raw_patch),
        "stressed_png_data_url": png_data_url(stressed_display),
        "stress_gains": [1.16, 1.0, 0.84],
        "cases": [
            trace_case(model, clean, "clean (original RGB)"),
            trace_case(model, stressed, "RGB red-dominant level 4"),
        ],
    }
    assert trace["cases"][0]["features"][-1]["shape"] == [1, 256, 4, 4]
    assert trace["cases"][1]["features"][-1]["shape"] == [1, 256, 4, 4]
    for case in trace["cases"]:
        assert np.isfinite(case["score"]) and 0.0 <= case["score"] <= 1.0
        assert case["prediction"] == ("TUM" if case["score"] >= 0.5 else "NORM")
    DEST.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    print(f"Exported {DEST.resolve()}")
    for case in trace["cases"]:
        print(case["condition"], "logit", case["logit"], "sigmoid", case["score"], "prediction", case["prediction"])
    print("Upload ONLY real_forward_trace.json to your LecTrace repository root.")
    print("No trained weights or private account credentials are included in the JSON.")


if __name__ == "__main__":
    main()
