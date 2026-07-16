from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .config import save_project_config
from .geometry import roi_to_spherical_ranges


GENERATE_SCRIPT = '''from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from trainset.config import load_project_config
from trainset.generator import DatasetGenerator

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml")
parser.add_argument("--samples", type=int, default=None)
parser.add_argument("--mode", choices=("dry", "full"), default="full")
parser.add_argument("--output", default="dataset")
args = parser.parse_args()
config = load_project_config(ROOT / args.config)
count = args.samples or int(config["dataset"]["number_of_samples"])
task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
output = ROOT / args.output
if task_id is not None:
    task_index = int(task_id)
    shard_size = int(config["dataset"]["samples_per_shard"])
    total = int(config["dataset"]["number_of_samples"])
    count = min(shard_size, max(0, total - task_index * shard_size))
    if count <= 0:
        raise SystemExit(f"array task {task_index} has no samples to generate")
    config["project"]["seed"] = int(config["project"]["seed"]) + task_index
    output = output / f"array_{task_index:04d}"
files = DatasetGenerator(config).write_hdf5_shards(output, count)
print(f"generated_files={len(files)} samples={count} output={output}")
'''

VALIDATE_SCRIPT = '''from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from trainset.config import load_project_config, validate_project_config
from trainset.geometry import roi_to_spherical_ranges
cfg = load_project_config(ROOT / "config.yaml")
valid, errors, warnings = validate_project_config(cfg)
print("valid=", valid)
print("spherical_ranges=", roi_to_spherical_ranges(cfg))
for item in warnings: print("WARNING:", item)
for item in errors: print("ERROR:", item)
raise SystemExit(0 if valid else 1)
'''

TRAIN_SCRIPT = '''from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from trainset.config import load_project_config

cfg = load_project_config(ROOT / "config.yaml")
try:
    import tensorflow as tf
except Exception as exc:
    raise SystemExit(f"TensorFlow is required by the cnn_2d template: {exc}")

files = sorted((ROOT / "dataset").rglob("*.h5"))
if not files:
    raise SystemExit("No HDF5 dataset shards found under dataset/.")

parameter_names = list(cfg["parameters"])
target_min = np.asarray([cfg["parameters"][name]["minimum"] for name in parameter_names], dtype=np.float32)
target_max = np.asarray([cfg["parameters"][name]["maximum"] for name in parameter_names], dtype=np.float32)

def records():
    for path in files:
        with h5py.File(path, "r") as handle:
            for image, label in zip(handle["images"], handle["labels"]):
                normalized_label = (label.astype("float32") - target_min) / np.maximum(target_max - target_min, 1e-12)
                yield image[..., None].astype("float32"), normalized_label

with h5py.File(files[0], "r") as first:
    image_shape = tuple(first["images"].shape[1:]) + (1,)
    output_size = int(first["labels"].shape[1])
signature = (
    tf.TensorSpec(shape=image_shape, dtype=tf.float32),
    tf.TensorSpec(shape=(output_size,), dtype=tf.float32),
)
dataset = tf.data.Dataset.from_generator(records, output_signature=signature)
total_records = 0
for path in files:
    with h5py.File(path, "r") as handle:
        total_records += int(handle["images"].shape[0])
if total_records < 2:
    raise SystemExit("At least two generated samples are required for training.")

split = cfg["dataset"]["split"]
train_count = max(1, int(total_records * float(split["train"])))
validation_count = int(total_records * float(split["validation"]))
test_count = max(0, total_records - train_count - validation_count)
shuffled = dataset.shuffle(
    min(total_records, 8192),
    seed=int(cfg["project"]["seed"]),
    reshuffle_each_iteration=False,
)
batch_size = int(cfg["training"]["batch_size"])
train_dataset = shuffled.take(train_count).batch(batch_size).prefetch(tf.data.AUTOTUNE)
validation_dataset = shuffled.skip(train_count).take(validation_count).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = shuffled.skip(train_count + validation_count).take(test_count).batch(batch_size).prefetch(tf.data.AUTOTUNE)

inputs = tf.keras.Input(shape=image_shape)
x = inputs
for channels in cfg["model"]["channels"]:
    x = tf.keras.layers.Conv2D(int(channels), int(cfg["model"]["kernel_size"]), padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(float(cfg["model"]["dropout"]))(x)
outputs = tf.keras.layers.Dense(output_size)(x)
model = tf.keras.Model(inputs, outputs)
optimizer_name = str(cfg["training"].get("optimizer", "adam")).lower()
learning_rate = float(cfg["training"]["learning_rate"])
if optimizer_name == "sgd":
    optimizer = tf.keras.optimizers.SGD(learning_rate)
elif optimizer_name == "adamw" and hasattr(tf.keras.optimizers, "AdamW"):
    optimizer = tf.keras.optimizers.AdamW(learning_rate)
else:
    optimizer = tf.keras.optimizers.Adam(learning_rate)
model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

results = ROOT / "results"
results.mkdir(exist_ok=True)
metrics_path = results / "metrics.jsonl"
if metrics_path.exists():
    metrics_path.unlink()
class JsonlCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        payload = {"epoch": epoch + 1, **{key: float(value) for key, value in (logs or {}).items()}}
        with metrics_path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(payload) + "\\n")

monitor = "val_loss" if validation_count else "loss"
callbacks = [
    JsonlCallback(),
    tf.keras.callbacks.ModelCheckpoint(results / "best_model.keras", monitor=monitor, save_best_only=True),
]
scheduler = str(cfg["training"].get("scheduler", "constant")).lower()
if scheduler == "cosine":
    epochs = max(1, int(cfg["training"]["epochs"]))
    callbacks.append(tf.keras.callbacks.LearningRateScheduler(lambda epoch: learning_rate * 0.5 * (1.0 + np.cos(np.pi * epoch / epochs))))
elif scheduler == "plateau":
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, patience=5))

history = model.fit(
    train_dataset,
    validation_data=validation_dataset if validation_count else None,
    epochs=int(cfg["training"]["epochs"]),
    callbacks=callbacks,
)
model.save(results / "last_checkpoint.keras")
test_metrics = model.evaluate(test_dataset, return_dict=True, verbose=0) if test_count else {}
(results / "test_metrics.json").write_text(json.dumps({key: float(value) for key, value in test_metrics.items()}, indent=2), encoding="utf-8")
if test_count:
    predictions = model.predict(test_dataset, verbose=0)
    targets = np.concatenate([labels.numpy() for _images, labels in test_dataset], axis=0)
    with h5py.File(results / "predictions.h5", "w") as handle:
        handle.create_dataset("predictions_normalized", data=predictions, compression="gzip")
        handle.create_dataset("targets_normalized", data=targets, compression="gzip")
        handle.attrs["parameter_names"] = np.asarray(parameter_names, dtype="S")
try:
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(history.history.get("loss", []), label="train")
    if history.history.get("val_loss"):
        axis.plot(history.history["val_loss"], label="validation")
    axis.set(xlabel="Epoch", ylabel="Loss", title="Training summary")
    axis.legend()
    figure.tight_layout()
    figure.savefig(results / "training_summary.png", dpi=160)
    plt.close(figure)
except Exception as exc:
    print(f"training summary plot unavailable: {exc}")
shutil.copy2(ROOT / "config.yaml", results / "config_used.yaml")
print(f"split train={train_count} validation={validation_count} test={test_count}")
'''


def _write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def _git_commit(project_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()
    except Exception:
        return "unknown"


def prepare_job_package(config: Dict[str, Any], output_root: str | Path, project_root: str | Path) -> Path:
    project_root = Path(project_root)
    destination = Path(output_root) / str(config["project"]["name"])
    if destination.exists():
        shutil.rmtree(destination)
    (destination / "src").mkdir(parents=True)
    shutil.copytree(project_root / "trainset", destination / "src" / "trainset", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    save_project_config(config, destination / "config.yaml")
    _write(destination / "generate_dataset.py", GENERATE_SCRIPT)
    _write(destination / "validate_config.py", VALIDATE_SCRIPT)
    _write(destination / "train.py", TRAIN_SCRIPT)
    _write(
        destination / "environment.yml",
        """name: gimap-trainset\nchannels:\n  - conda-forge\ndependencies:\n  - python=3.10\n  - numpy\n  - scipy\n  - h5py\n  - pyyaml\n  - tensorflow=2.15\n  - keras=2.15\n  - matplotlib\n  - pip\n  - pip:\n      - BornAgain==24.1\n""",
    )
    dataset = config["dataset"]
    hpc = config["hpc"]
    python_command = str(hpc.get("python_command", "python")).strip() or "python"
    shard_count = max(1, int(np_ceil(int(dataset["number_of_samples"]) / int(dataset["samples_per_shard"]))))
    job_array = bool(hpc.get("job_array", True))
    array_line = f"#SBATCH --array=0-{shard_count - 1}" if job_array else ""
    generation_count = int(dataset["samples_per_shard"] if job_array else dataset["number_of_samples"])
    _write(
        destination / "slurm_generate.sh",
        f"""#!/bin/bash\n#SBATCH --job-name={config['project']['name']}-generate\n#SBATCH --partition={hpc['partition']}\n#SBATCH --cpus-per-task={hpc['cpus']}\n#SBATCH --mem={hpc['memory']}\n#SBATCH --time={hpc['time']}\n{array_line}\n#SBATCH --output=logs/generate_%A_%a.out\n#SBATCH --error=logs/generate_%A_%a.err\nset -euo pipefail\nmkdir -p logs dataset\n{python_command} generate_dataset.py --config config.yaml --samples {generation_count} --output dataset\n""",
    )
    _write(
        destination / "slurm_train.sh",
        f"""#!/bin/bash\n#SBATCH --job-name={config['project']['name']}-train\n#SBATCH --partition={hpc['partition']}\n#SBATCH --gres=gpu:{hpc['gpus']}\n#SBATCH --cpus-per-task={hpc['cpus']}\n#SBATCH --mem={hpc['memory']}\n#SBATCH --time={hpc['time']}\n#SBATCH --output=logs/train_%j.out\n#SBATCH --error=logs/train_%j.err\nset -euo pipefail\nmkdir -p logs results\n{python_command} train.py\n""",
    )
    (destination / "logs").mkdir()
    (destination / "dataset").mkdir()
    (destination / "results").mkdir()

    file_hashes = {}
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            file_hashes[str(path.relative_to(destination)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "seed": int(config["project"]["seed"]),
        "roi_spherical_ranges": roi_to_spherical_ranges(config),
        "files": file_hashes,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return destination


def np_ceil(value: float) -> int:
    return int(value) if int(value) == value else int(value) + 1
