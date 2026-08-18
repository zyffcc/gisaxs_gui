"""Train the 1D GISAXS slot model."""

from __future__ import annotations

import json
import re
import shutil
import signal
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Training import data_loader
from Training.losses import LossWeights, compute_losses
from Training.model import build_model
from TrainSetBuild import schema


from Training.training_artifacts import (
    count_samples,
    load_json_list,
    mean_metrics,
    scalar_dict,
    update_runtime_status,
    write_json_atomic,
    write_training_artifacts,
)
from Training.training_cli import parse_args


STOP_REQUESTED = False


def request_graceful_stop(signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"Received signal {signum}; checkpoint will be saved after the current step.", flush=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
    args = parse_args()
    signal.signal(signal.SIGTERM, request_graceful_stop)
    signal.signal(signal.SIGINT, request_graceful_stop)
    if args.max_points != schema.MAX_POINTS:
        raise ValueError(
            f"This first version expects max_points={schema.MAX_POINTS}; got {args.max_points}"
        )
    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    if args.reconstruction_loss_weight < 0:
        raise ValueError("--reconstruction_loss_weight must be >= 0")
    if args.reconstruction_start_epoch < 1:
        raise ValueError("--reconstruction_start_epoch must be >= 1")
    if args.reconstruction_ramp_epochs < 1:
        raise ValueError("--reconstruction_ramp_epochs must be >= 1")
    if args.reconstruction_q_stride < 1:
        raise ValueError("--reconstruction_q_stride must be >= 1")
    if args.reconstruction_samples_per_batch < 1:
        raise ValueError("--reconstruction_samples_per_batch must be >= 1")
    if args.max_skipped_nonfinite_batches < 0:
        raise ValueError("--max_skipped_nonfinite_batches must be >= 0")

    tf.keras.utils.set_random_seed(args.seed)
    dataset_dir = Path(args.dataset_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "checkpoints").mkdir(exist_ok=True)
    (model_dir / "logs").mkdir(exist_ok=True)

    train_count = count_samples(dataset_dir, "train")
    val_count = count_samples(dataset_dir, "val")
    if args.quick_test:
        train_count = min(train_count, 64)
        val_count = min(val_count, 32)
        args.epochs = min(args.epochs, 2)
        args.save_interval = min(args.save_interval, 2)
    train_steps = max(1, train_count // args.batch_size)
    val_steps = max(1, val_count // args.batch_size)

    train_ds = data_loader.make_dataset(
        dataset_dir,
        "train",
        args.batch_size,
        shuffle=True,
        seed=args.seed,
        max_samples=train_count,
        drop_remainder=True,
    )
    val_ds = data_loader.make_dataset(
        dataset_dir,
        "val",
        args.batch_size,
        shuffle=False,
        seed=args.seed + 1,
        max_samples=val_count,
        drop_remainder=True,
    )
    # Apply cardinality limits while these are still regular tf.data.Dataset
    # objects. DistributedDataset intentionally does not expose .take().
    train_ds = train_ds.take(train_steps)
    val_ds = val_ds.take(val_steps)

    logical_gpus = tf.config.list_logical_devices("GPU")
    strategy = None
    if args.multi_gpu:
        if len(logical_gpus) > 1:
            strategy = tf.distribute.MirroredStrategy()
            print(
                f"Using MirroredStrategy with {strategy.num_replicas_in_sync} replicas.", flush=True
            )
        else:
            raise RuntimeError(
                f"--multi_gpu requires at least two visible GPUs; found {len(logical_gpus)}"
            )

    if strategy is not None:
        with strategy.scope():
            model = build_model()
            optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=1.0)
            optimizer.build(model.trainable_variables)
            reconstruction_weight_var = tf.Variable(
                0.0, dtype=tf.float32, trainable=False, name="reconstruction_loss_weight"
            )
    else:
        model = build_model()
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=1.0)
        optimizer.build(model.trainable_variables)
        reconstruction_weight_var = tf.Variable(
            0.0, dtype=tf.float32, trainable=False, name="reconstruction_loss_weight"
        )
    loss_weights = LossWeights(
        reconstruction=reconstruction_weight_var,
        reconstruction_q_stride=args.reconstruction_q_stride,
        reconstruction_samples_per_batch=args.reconstruction_samples_per_batch,
    )

    ckpt_epoch = tf.Variable(1, dtype=tf.int64, trainable=False)
    ckpt_step = tf.Variable(0, dtype=tf.int64, trainable=False)
    global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
    ckpt = tf.train.Checkpoint(
        model=model, optimizer=optimizer, epoch=ckpt_epoch, step=ckpt_step, global_step=global_step
    )
    manager = tf.train.CheckpointManager(ckpt, str(model_dir / "checkpoints"), max_to_keep=20)
    writer = tf.summary.create_file_writer(str(model_dir / "logs"))

    history = load_json_list(model_dir / "history.json")
    step_history = load_json_list(model_dir / "step_history.json")
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        if (
            int(ckpt_epoch.numpy()) == 1
            and int(ckpt_step.numpy()) == 0
            and int(global_step.numpy()) == 0
        ):
            match = re.search(r"ckpt-(\d+)$", manager.latest_checkpoint)
            if match and not step_history:
                legacy_epoch = int(match.group(1))
                ckpt_epoch.assign(legacy_epoch + 1)
                global_step.assign(legacy_epoch * train_steps)
                print(
                    f"Interpreting legacy checkpoint {manager.latest_checkpoint} as completed epoch {legacy_epoch}.",
                    flush=True,
                )
        print(
            f"Restored checkpoint {manager.latest_checkpoint}: "
            f"epoch={int(ckpt_epoch.numpy())}, step={int(ckpt_step.numpy())}, global_step={int(global_step.numpy())}",
            flush=True,
        )
    elif step_history:
        global_step.assign(int(step_history[-1]["global_step"]))
        print(
            f"No checkpoint found, but loaded existing step history through global_step={int(global_step.numpy())}.",
            flush=True,
        )

    update_runtime_status(
        model_dir,
        "initialized",
        epoch=int(ckpt_epoch.numpy()),
        step=int(ckpt_step.numpy()),
        global_step=int(global_step.numpy()),
        train_steps=train_steps,
        val_steps=val_steps,
        replicas=1 if strategy is None else strategy.num_replicas_in_sync,
    )

    def train_step_fn(inputs, labels):
        with tf.GradientTape() as tape:
            preds = model(inputs, training=True)
            losses = compute_losses(labels, preds, loss_weights)
            loss = losses["total_loss"]
            optimization_loss = (
                loss if strategy is None else loss / float(strategy.num_replicas_in_sync)
            )
        grads = tape.gradient(optimization_loss, model.trainable_variables)
        finite_grads = [tf.reduce_all(tf.math.is_finite(g)) for g in grads if g is not None]
        local_finite = tf.reduce_all(
            tf.stack([tf.reduce_all(tf.math.is_finite(v)) for v in losses.values()] + finite_grads)
        )
        if strategy is not None:
            replica_context = tf.distribute.get_replica_context()
            # TensorFlow 2.15 only exposes SUM and MEAN for distributed
            # reductions.  Every replica must report a finite result.
            finite_replica_count = replica_context.all_reduce(
                tf.distribute.ReduceOp.SUM, tf.cast(local_finite, tf.int32)
            )
            globally_finite = finite_replica_count == strategy.num_replicas_in_sync
        else:
            globally_finite = local_finite

        # Keras' distributed optimizer performs a merge_call internally, so it
        # cannot live inside tf.cond while tracing strategy.run.  Keep one
        # unconditional optimizer path and feed it zero gradients whenever any
        # replica is non-finite.  The outer loop records/rejects such batches.
        safe_grads = [
            None if grad is None else tf.where(globally_finite, grad, tf.zeros_like(grad))
            for grad in grads
        ]
        optimizer.apply_gradients(zip(safe_grads, model.trainable_variables))
        update_applied = tf.cast(globally_finite, tf.float32)
        result = dict(losses)
        result["gradient_global_norm"] = tf.linalg.global_norm([g for g in grads if g is not None])
        result["update_applied"] = update_applied
        return result

    def val_step_fn(inputs, labels):
        preds = model(inputs, training=False)
        return compute_losses(labels, preds, loss_weights)

    if strategy is not None:
        train_ds = strategy.experimental_distribute_dataset(train_ds)
        val_ds = strategy.experimental_distribute_dataset(val_ds)

        @tf.function
        def train_step(inputs, labels):
            per_replica = strategy.run(train_step_fn, args=(inputs, labels))
            return {
                k: strategy.reduce(tf.distribute.ReduceOp.MEAN, v, axis=None)
                for k, v in per_replica.items()
            }

        @tf.function
        def val_step(inputs, labels):
            per_replica = strategy.run(val_step_fn, args=(inputs, labels))
            return {
                k: strategy.reduce(tf.distribute.ReduceOp.MEAN, v, axis=None)
                for k, v in per_replica.items()
            }
    else:
        train_step = tf.function(train_step_fn)
        val_step = tf.function(val_step_fn)

    print(
        f"Training samples={train_count}, val samples={val_count}, steps={train_steps}/{val_steps}",
        flush=True,
    )
    print(
        f"Intervals: log_interval={args.log_interval}, save_interval={args.save_interval}",
        flush=True,
    )
    print(
        f"Fixed train batches: drop_remainder=True, discarded_per_epoch={train_count - train_steps * args.batch_size}",
        flush=True,
    )
    print(
        f"Fixed validation batches: drop_remainder=True, discarded_per_epoch={val_count - val_steps * args.batch_size}",
        flush=True,
    )

    start_epoch = int(ckpt_epoch.numpy())
    resume_step = int(ckpt_step.numpy())
    if resume_step >= train_steps:
        start_epoch += 1
        resume_step = 0
        ckpt_epoch.assign(start_epoch)
        ckpt_step.assign(0)
    if start_epoch > args.epochs:
        print(
            f"Checkpoint already reached epoch {start_epoch}; requested epochs={args.epochs}. Nothing to train.",
            flush=True,
        )
        write_training_artifacts(model_dir, history, step_history)
        return

    nonfinite_events = load_json_list(model_dir / "nonfinite_batches.json")
    skipped_nonfinite_batches = len(nonfinite_events)

    for epoch in range(start_epoch, args.epochs + 1):
        if args.reconstruction_loss_weight > 0.0 and epoch >= args.reconstruction_start_epoch:
            ramp_step = epoch - args.reconstruction_start_epoch + 1
            physical_weight = args.reconstruction_loss_weight * min(
                ramp_step / args.reconstruction_ramp_epochs, 1.0
            )
        else:
            physical_weight = 0.0
        reconstruction_weight_var.assign(physical_weight)
        print(f"epoch {epoch}: reconstruction_loss_weight={physical_weight:.6g}", flush=True)
        update_runtime_status(
            model_dir,
            "running",
            epoch=epoch,
            step=resume_step if epoch == start_epoch else 0,
            global_step=int(global_step.numpy()),
            train_steps=train_steps,
            reconstruction_loss_weight=float(physical_weight),
        )
        train_metrics = []
        for step, (inputs, labels) in enumerate(train_ds, start=1):
            if epoch == start_epoch and step <= resume_step:
                continue
            m = train_step(inputs, labels)
            raw_metrics = {k: v.numpy() for k, v in m.items()}
            train_row = scalar_dict(raw_metrics)
            update_applied = train_row.pop("update_applied")
            loss_value = float(train_row["total_loss"])
            if update_applied < 0.5:
                skipped_nonfinite_batches += 1
                if strategy is None:
                    bad_slot_type = np.asarray(labels["slot_type"]).astype(int)
                    bad_slot_exist = np.asarray(labels["slot_exist"]).astype(float)
                else:
                    bad_slot_type = np.concatenate(
                        [
                            np.asarray(v).astype(int)
                            for v in strategy.experimental_local_results(labels["slot_type"])
                        ],
                        axis=0,
                    )
                    bad_slot_exist = np.concatenate(
                        [
                            np.asarray(v).astype(float)
                            for v in strategy.experimental_local_results(labels["slot_exist"])
                        ],
                        axis=0,
                    )
                event = {
                    "epoch": int(epoch),
                    "step": int(step),
                    "global_step": int(global_step.numpy()),
                    "losses": train_row,
                    "slot_type": bad_slot_type.tolist(),
                    "slot_exist": bad_slot_exist.tolist(),
                }
                nonfinite_events.append(event)
                write_json_atomic(model_dir / "nonfinite_batches.json", nonfinite_events)
                update_runtime_status(
                    model_dir,
                    "skipped_nonfinite_batch",
                    epoch=epoch,
                    step=step,
                    global_step=int(global_step.numpy()),
                    train_steps=train_steps,
                    skipped_nonfinite_batches=skipped_nonfinite_batches,
                    diagnostic=event,
                )
                print(
                    f"WARNING: skipped non-finite batch epoch={epoch} step={step}; "
                    f"skipped_total={skipped_nonfinite_batches}",
                    flush=True,
                )
                if skipped_nonfinite_batches > args.max_skipped_nonfinite_batches:
                    raise RuntimeError(
                        f"Exceeded --max_skipped_nonfinite_batches={args.max_skipped_nonfinite_batches}; "
                        f"see {model_dir / 'nonfinite_batches.json'}"
                    )
                continue
            if not np.isfinite(loss_value):
                raise RuntimeError(
                    f"Non-finite total_loss at epoch={epoch}, step={step}: {loss_value}"
                )
            train_metrics.append(train_row)
            global_step_value = int(global_step.assign_add(1).numpy())
            step_history.append(
                {
                    "global_step": global_step_value,
                    "epoch": int(epoch),
                    "step": int(step),
                    "train": train_row,
                }
            )
            with writer.as_default():
                for k, v in train_row.items():
                    tf.summary.scalar(f"train_step/{k}", v, step=global_step_value)
            if args.log_interval > 0 and (step % args.log_interval == 0 or step == train_steps):
                print(
                    f"epoch {epoch} train step {step}/{train_steps} global_step={global_step_value} loss={loss_value:.5f}",
                    flush=True,
                )
                update_runtime_status(
                    model_dir,
                    "running",
                    epoch=epoch,
                    step=step,
                    global_step=global_step_value,
                    train_steps=train_steps,
                    reconstruction_loss_weight=float(physical_weight),
                    latest_train=train_row,
                )
            if args.save_interval > 0 and (
                global_step_value % args.save_interval == 0 or step == train_steps
            ):
                ckpt_epoch.assign(epoch)
                ckpt_step.assign(step)
                manager.save(checkpoint_number=global_step_value)
                write_training_artifacts(model_dir, history, step_history)
                writer.flush()
                model.save(model_dir / "model.keras", overwrite=True)
                print(
                    f"saved progress at epoch {epoch} step {step}/{train_steps} global_step={global_step_value}",
                    flush=True,
                )
            if STOP_REQUESTED:
                ckpt_epoch.assign(epoch)
                ckpt_step.assign(step)
                manager.save(checkpoint_number=global_step_value)
                write_training_artifacts(model_dir, history, step_history)
                writer.flush()
                model.save(model_dir / "model.keras", overwrite=True)
                update_runtime_status(
                    model_dir,
                    "interrupted_checkpoint_saved",
                    epoch=epoch,
                    step=step,
                    global_step=global_step_value,
                    train_steps=train_steps,
                    latest_train=train_row,
                )
                print(f"Graceful stop checkpoint saved at epoch {epoch} step {step}.", flush=True)
                return

        if not train_metrics:
            print(
                f"epoch {epoch}: no new training steps after resume skip; moving to validation.",
                flush=True,
            )

        val_metrics = []
        for step, (inputs, labels) in enumerate(val_ds, start=1):
            m = val_step(inputs, labels)
            loss_value = float(m["total_loss"].numpy())
            if not np.isfinite(loss_value):
                raise RuntimeError(
                    f"Non-finite validation total_loss at epoch={epoch}, step={step}: {loss_value}"
                )
            val_metrics.append(scalar_dict({k: v.numpy() for k, v in m.items()}))
            if STOP_REQUESTED:
                ckpt_epoch.assign(epoch)
                ckpt_step.assign(train_steps)
                manager.save(checkpoint_number=int(global_step.numpy()))
                write_training_artifacts(model_dir, history, step_history)
                writer.flush()
                model.save(model_dir / "model.keras", overwrite=True)
                update_runtime_status(
                    model_dir,
                    "interrupted_checkpoint_saved",
                    epoch=epoch,
                    step=train_steps,
                    global_step=int(global_step.numpy()),
                    train_steps=train_steps,
                )
                print(
                    f"Graceful stop checkpoint saved during epoch {epoch} validation.", flush=True
                )
                return

        tr = mean_metrics(train_metrics)
        va = mean_metrics(val_metrics)
        row = {"epoch": epoch, "train": tr, "val": va}
        history.append(row)
        print(
            f"epoch {epoch}: train_loss={tr['total_loss']:.5f} val_loss={va['total_loss']:.5f} "
            f"val_type_acc={va['slot_type_accuracy']:.3f} val_nonempty_acc={va['nonempty_type_accuracy']:.3f} "
            f"val_K_acc={va['component_count_accuracy']:.3f}",
            flush=True,
        )
        best_row = min(history, key=lambda item: float(item["val"]["total_loss"]))
        update_runtime_status(
            model_dir,
            "epoch_complete",
            epoch=epoch,
            step=train_steps,
            global_step=int(global_step.numpy()),
            train_steps=train_steps,
            reconstruction_loss_weight=float(physical_weight),
            latest_train=tr,
            latest_val=va,
            best_val_loss=float(best_row["val"]["total_loss"]),
            best_val_epoch=int(best_row["epoch"]),
        )
        with writer.as_default():
            for k, v in tr.items():
                tf.summary.scalar(f"train/{k}", v, step=epoch)
            for k, v in va.items():
                tf.summary.scalar(f"val/{k}", v, step=epoch)
        writer.flush()
        ckpt_epoch.assign(epoch + 1)
        ckpt_step.assign(0)
        manager.save(checkpoint_number=int(global_step.numpy()))
        write_training_artifacts(model_dir, history, step_history)
        model.save(model_dir / "model.keras", overwrite=True)
        print(f"saved epoch {epoch} checkpoint/model artifacts", flush=True)

    config = {
        "max_points": schema.MAX_POINTS,
        "max_slots": schema.MAX_SLOTS,
        "num_types": schema.NUM_TYPES,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "reconstruction_loss_weight": args.reconstruction_loss_weight,
        "reconstruction_start_epoch": args.reconstruction_start_epoch,
        "reconstruction_ramp_epochs": args.reconstruction_ramp_epochs,
        "reconstruction_q_stride": args.reconstruction_q_stride,
        "reconstruction_samples_per_batch": args.reconstruction_samples_per_batch,
        "d_constraint_model": {
            "explicit_presence_head": True,
            "spacing_rules": schema.D_RULE_NAMES,
        },
    }
    with (model_dir / "model_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    src_meta = dataset_dir / "metadata.json"
    if src_meta.exists():
        shutil.copy2(src_meta, model_dir / "dataset_metadata.json")

    model.save(model_dir / "model.keras", overwrite=True)
    model.save(model_dir / "saved_model")
    write_training_artifacts(model_dir, history, step_history)
    update_runtime_status(
        model_dir,
        "complete",
        epoch=args.epochs,
        step=train_steps,
        global_step=int(global_step.numpy()),
        train_steps=train_steps,
        latest_train=history[-1]["train"] if history else None,
        latest_val=history[-1]["val"] if history else None,
    )
    print(f"Training complete. Model written to {model_dir}", flush=True)


if __name__ == "__main__":
    main()
