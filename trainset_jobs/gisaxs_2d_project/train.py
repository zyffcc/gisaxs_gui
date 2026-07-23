from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from trainset.config import load_project_config, synchronize_parameter_specs, trainable_parameter_names
from trainset.modeling import build_keras_model, build_optimizer, resolve_keras_api

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true", help="Cap records/model size for a quick local pipeline test")
parser.add_argument("--epochs", type=int, default=None)
args = parser.parse_args()

cfg = synchronize_parameter_specs(load_project_config(ROOT / "config.yaml"))
try:
    import tensorflow as tf
except Exception as exc:
    raise SystemExit(f"TensorFlow is required by the cnn_2d template: {exc}")
keras_api = resolve_keras_api(tf)

files = sorted((ROOT / "dataset").rglob("*.h5"))
if not files:
    raise SystemExit("No HDF5 dataset shards found under dataset/.")

parameter_names = trainable_parameter_names(cfg)
target_min = np.asarray([cfg["parameters"][name]["minimum"] for name in parameter_names], dtype=np.float32)
target_max = np.asarray([cfg["parameters"][name]["maximum"] for name in parameter_names], dtype=np.float32)

def records():
    emitted = 0
    for path in files:
        with h5py.File(path, "r") as handle:
            stored_names = [value.decode() if isinstance(value, bytes) else str(value) for value in handle.attrs["label_names"]]
            label_indices = [stored_names.index(name) for name in parameter_names]
            for image, label in zip(handle["images"], handle["labels"]):
                selected = label.astype("float32")[label_indices]
                normalized_label = (selected - target_min) / np.maximum(target_max - target_min, 1e-12)
                yield image[..., None].astype("float32"), normalized_label
                emitted += 1
                if args.smoke and emitted >= int(cfg["training"].get("smoke_samples", 64)):
                    return

with h5py.File(files[0], "r") as first:
    image_shape = tuple(first["images"].shape[1:]) + (1,)
    output_size = len(parameter_names)
signature = (
    tf.TensorSpec(shape=image_shape, dtype=tf.float32),
    tf.TensorSpec(shape=(output_size,), dtype=tf.float32),
)
total_records = 0
for path in files:
    with h5py.File(path, "r") as handle:
        total_records += int(handle["images"].shape[0])
if args.smoke:
    total_records = min(total_records, int(cfg["training"].get("smoke_samples", 64)))
if total_records < 2:
    raise SystemExit("At least two generated samples are required for training.")

split = cfg["dataset"]["split"]
train_count = max(1, int(total_records * float(split["train"])))
validation_count = int(total_records * float(split["validation"]))
test_count = max(0, total_records - train_count - validation_count)
batch_size = int(cfg["training"]["batch_size"])
if args.smoke:
    materialized = list(records())
    all_images = np.stack([item[0] for item in materialized])
    all_labels = np.stack([item[1] for item in materialized])
    order = np.random.default_rng(int(cfg["project"]["seed"])).permutation(total_records)
    all_images, all_labels = all_images[order], all_labels[order]
    train_data = (all_images[:train_count], all_labels[:train_count])
    validation_data = (all_images[train_count:train_count + validation_count], all_labels[train_count:train_count + validation_count]) if validation_count else None
    test_data = (all_images[train_count + validation_count:], all_labels[train_count + validation_count:]) if test_count else None
else:
    dataset = tf.data.Dataset.from_generator(records, output_signature=signature)
    shuffled = dataset.shuffle(
        min(total_records, 8192),
        seed=int(cfg["project"]["seed"]),
        reshuffle_each_iteration=False,
    )
    train_data = shuffled.take(train_count).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    validation_data = shuffled.skip(train_count).take(validation_count).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_data = shuffled.skip(train_count + validation_count).take(test_count).batch(batch_size).prefetch(tf.data.AUTOTUNE)

model = build_keras_model(tf, image_shape, output_size, cfg["model"], smoke=args.smoke)
optimizer_name = str(cfg["training"].get("optimizer", "adam")).lower()
learning_rate = float(cfg["training"]["learning_rate"])
optimizer = build_optimizer(keras_api, optimizer_name, learning_rate)
model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

results = ROOT / "results"
results.mkdir(exist_ok=True)
metrics_path = results / "metrics.jsonl"
if metrics_path.exists():
    metrics_path.unlink()
class JsonlCallback(keras_api.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        payload = {"epoch": epoch + 1, **{key: float(value) for key, value in (logs or {}).items()}}
        with metrics_path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(payload) + "\n")

monitor = "val_loss" if validation_count else "loss"
callbacks = [
    JsonlCallback(),
    keras_api.callbacks.ModelCheckpoint(str(results / "best_model.h5"), monitor=monitor, save_best_only=True),
]
epochs = args.epochs or (int(cfg["training"].get("smoke_epochs", 2)) if args.smoke else int(cfg["training"]["epochs"]))
scheduler = str(cfg["training"].get("scheduler", "constant")).lower()
if scheduler == "cosine":
    epochs = max(1, int(epochs))
    callbacks.append(keras_api.callbacks.LearningRateScheduler(lambda epoch: learning_rate * 0.5 * (1.0 + np.cos(np.pi * epoch / epochs))))
elif scheduler == "plateau":
    callbacks.append(keras_api.callbacks.ReduceLROnPlateau(monitor=monitor, patience=5))

if args.smoke:
    history = model.fit(
        train_data[0], train_data[1],
        validation_data=validation_data,
        batch_size=min(batch_size, train_count),
        epochs=epochs,
        callbacks=callbacks,
    )
else:
    history = model.fit(train_data, validation_data=validation_data, epochs=epochs, callbacks=callbacks)
model.save(str(results / "last_checkpoint.h5"))
if test_count and args.smoke:
    values = model.evaluate(test_data[0], test_data[1], verbose=0)
    test_metrics = dict(zip(model.metrics_names, np.atleast_1d(values)))
elif test_count:
    test_metrics = model.evaluate(test_data, return_dict=True, verbose=0)
else:
    test_metrics = {}
(results / "test_metrics.json").write_text(json.dumps({key: float(value) for key, value in test_metrics.items()}, indent=2), encoding="utf-8")
if test_count:
    predictions = model.predict(test_data[0] if args.smoke else test_data, verbose=0)
    targets = test_data[1] if args.smoke else np.concatenate([labels.numpy() for _images, labels in test_data], axis=0)
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
