from __future__ import annotations
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
        with metrics_path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(payload) + "\n")

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
