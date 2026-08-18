"""Curve reconstruction and reporting for the overfit diagnostic."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from TrainSetBuild import schema
from TrainSetBuild.physics_adapter import (
    component_array_to_dict,
    evaluate_clean,
    global_array_to_dict,
)


def denormalized_sample_from_label(labels: dict, sample_idx: int):
    slot_exist = np.asarray(labels["slot_exist"][sample_idx])
    slot_type = np.asarray(labels["slot_type"][sample_idx])
    slot_params_norm = np.asarray(labels["slot_params_norm"][sample_idx])
    slot_weight = np.asarray(labels["slot_weight"][sample_idx])
    slot_param_mask = np.asarray(labels["slot_param_mask"][sample_idx])
    global_params_norm = np.asarray(labels["global_params_norm"][sample_idx])

    true_slot = int(np.argmax(slot_exist))
    true_type = int(slot_type[true_slot])
    true_slot_weight = float(slot_weight[true_slot])
    true_params_norm = slot_params_norm[true_slot]
    true_param_mask = slot_param_mask[true_slot]
    if "slot_params_phys" in labels:
        true_param_mask = true_param_mask * schema.effective_param_mask(
            true_type,
            np.asarray(labels["slot_params_phys"][sample_idx, true_slot]),
        )
    true_params_phys = schema.denormalize_params_with_mask(
        true_params_norm, true_type, true_param_mask
    )
    true_global_phys = schema.denormalize_global_with_optional_zero(global_params_norm)
    return true_slot, true_type, true_params_phys, true_global_phys, true_slot_weight


def predicted_sample_from_preds(preds: dict):
    exist_logits = preds["exist_logit"].numpy()[0]
    type_logits = preds["type_logits"].numpy()[0]
    param_mu_norm = preds["param_mu_norm"].numpy()[0]
    global_mu_norm = preds["global_mu_norm"].numpy()[0]
    weight_logits = preds["weight_logit"].numpy()[0]

    pred_slot = int(np.argmax(exist_logits))
    type_scores = np.asarray(type_logits[pred_slot], dtype=np.float64)
    type_scores[schema.TYPE_EMPTY] = -1e9
    pred_type = int(np.argmax(type_scores))
    pred_params_norm = param_mu_norm[pred_slot, pred_type]
    pred_params_phys = schema.denormalize_params_with_mask(
        pred_params_norm, pred_type, schema.type_param_mask(pred_type)
    )
    pred_global_phys = schema.denormalize_global_with_optional_zero(global_mu_norm)
    weight_scores = np.asarray(weight_logits, dtype=np.float64)
    weight_scores = weight_scores - np.max(weight_scores)
    pred_weights = np.exp(weight_scores)
    pred_weights = pred_weights / np.maximum(np.sum(pred_weights), 1e-300)
    pred_slot_weight = float(pred_weights[pred_slot])
    return pred_slot, pred_type, pred_params_phys, pred_global_phys, pred_slot_weight


def array_json(values):
    return [float(v) for v in np.asarray(values, dtype=np.float64).tolist()]


def curve_metrics(pred_curve, saved_curve):
    pred_curve = np.asarray(pred_curve, dtype=np.float64)
    saved_curve = np.asarray(saved_curve, dtype=np.float64)
    eps = 1e-30
    return {
        "curve_rmse_linear": float(np.sqrt(np.mean(np.square(pred_curve - saved_curve)))),
        "curve_log_rmse": float(
            np.sqrt(
                np.mean(
                    np.square(
                        np.log(np.maximum(pred_curve, eps)) - np.log(np.maximum(saved_curve, eps))
                    )
                )
            )
        ),
    }


def prefixed_curve_metrics(prefix: str, pred_curve, saved_curve):
    metrics = curve_metrics(pred_curve, saved_curve)
    return {
        f"{prefix}_linear_rmse": metrics["curve_rmse_linear"],
        f"{prefix}_log_rmse": metrics["curve_log_rmse"],
    }


def plot_curves(
    model,
    batch_inputs: dict,
    batch_labels: dict,
    q_list: list[np.ndarray],
    i_list: list[np.ndarray],
    out_dir: Path,
    eval_batch_size: int,
    plot_n: int,
    use_true_global_for_plot: bool,
):
    curve_dir = out_dir / "curves"
    curve_dir.mkdir(parents=True, exist_ok=True)

    n = min(len(q_list), int(plot_n))
    for i in range(n):
        q = q_list[i]
        saved_curve = i_list[i]

        xb = {k: tf.convert_to_tensor(v[i : i + 1]) for k, v in batch_inputs.items()}
        preds = model(xb, training=False)

        true_slot, true_type, true_params_phys, true_global_phys, true_slot_weight = (
            denormalized_sample_from_label(batch_labels, i)
        )
        pred_slot, pred_type, pred_params_phys, pred_global_phys, pred_slot_weight = (
            predicted_sample_from_preds(preds)
        )
        true_weight_warning = None
        if abs(true_slot_weight - 1.0) >= 1e-6:
            true_weight_warning = f"K=1 true slot weight is {true_slot_weight:.9g}, expected 1.0"
            print(f"WARNING sample_{i:03d}: {true_weight_warning}", flush=True)

        oracle_component = component_array_to_dict(true_type, true_params_phys, true_slot_weight)
        oracle_curve = evaluate_clean(q, [oracle_component], global_array_to_dict(true_global_phys))

        pred_component_forced_weight = component_array_to_dict(pred_type, pred_params_phys, 1.0)
        pred_component_pred_weight = component_array_to_dict(
            pred_type, pred_params_phys, pred_slot_weight
        )
        pred_params_true_global_curve = evaluate_clean(
            q, [pred_component_forced_weight], global_array_to_dict(true_global_phys)
        )
        pred_params_pred_global_curve = evaluate_clean(
            q, [pred_component_forced_weight], global_array_to_dict(pred_global_phys)
        )
        pred_params_true_global_pred_weight_curve = evaluate_clean(
            q, [pred_component_pred_weight], global_array_to_dict(true_global_phys)
        )
        pred_params_pred_global_pred_weight_curve = evaluate_clean(
            q, [pred_component_pred_weight], global_array_to_dict(pred_global_phys)
        )
        selected_pred_curve = (
            pred_params_true_global_curve
            if use_true_global_for_plot
            else pred_params_pred_global_curve
        )
        selected_source = (
            "pred_params_true_global_curve"
            if use_true_global_for_plot
            else "pred_params_pred_global_curve"
        )
        metrics = curve_metrics(selected_pred_curve, saved_curve)
        curve_metric_block = {
            **prefixed_curve_metrics("oracle_curve_from_true_label", oracle_curve, saved_curve),
            **prefixed_curve_metrics(
                "pred_params_true_global_curve", pred_params_true_global_curve, saved_curve
            ),
            **prefixed_curve_metrics(
                "pred_params_pred_global_curve", pred_params_pred_global_curve, saved_curve
            ),
            **prefixed_curve_metrics(
                "pred_params_true_global_pred_weight_curve",
                pred_params_true_global_pred_weight_curve,
                saved_curve,
            ),
            **prefixed_curve_metrics(
                "pred_params_pred_global_pred_weight_curve",
                pred_params_pred_global_pred_weight_curve,
                saved_curve,
            ),
        }

        info = {
            "sample_index": int(i),
            "true_slot": true_slot,
            "pred_slot": pred_slot,
            "true_type": true_type,
            "pred_type": pred_type,
            "true_type_name": schema.TYPE_NAMES.get(true_type, str(true_type)),
            "pred_type_name": schema.TYPE_NAMES.get(pred_type, str(pred_type)),
            "true_params_phys": array_json(true_params_phys),
            "pred_params_phys": array_json(pred_params_phys),
            "true_global_phys": array_json(true_global_phys),
            "pred_global_phys": array_json(pred_global_phys),
            "true_slot_weight": float(true_slot_weight),
            "pred_slot_weight": float(pred_slot_weight),
            "forced_slot_weight": 1.0,
            "true_slot_weight_is_one": bool(abs(true_slot_weight - 1.0) < 1e-6),
            "true_slot_weight_warning": true_weight_warning,
            "curve_metric_source": selected_source,
            **metrics,
            **curve_metric_block,
        }
        with (curve_dir / f"sample_{i:03d}_info.json").open("w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        np.savez_compressed(
            curve_dir / f"sample_{i:03d}_curves.npz",
            q=q,
            saved_curve=saved_curve,
            oracle_curve_from_true_label=oracle_curve,
            pred_params_true_global_curve=pred_params_true_global_curve,
            pred_params_pred_global_curve=pred_params_pred_global_curve,
            pred_params_true_global_pred_weight_curve=pred_params_true_global_pred_weight_curve,
            pred_params_pred_global_pred_weight_curve=pred_params_pred_global_pred_weight_curve,
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.loglog(q, saved_curve, label="saved_curve", lw=1.2)
        ax.loglog(q, oracle_curve, label="oracle_curve_from_true_label", lw=1.0, ls=":")
        ax.loglog(
            q, pred_params_true_global_curve, label="pred_true_global_forced_w", lw=1.0, ls="--"
        )
        ax.loglog(
            q, pred_params_pred_global_curve, label="pred_pred_global_forced_w", lw=1.0, ls="-."
        )
        ax.loglog(
            q,
            pred_params_true_global_pred_weight_curve,
            label="pred_true_global_pred_w",
            lw=0.9,
            ls=(0, (3, 1, 1, 1)),
        )
        ax.loglog(
            q,
            pred_params_pred_global_pred_weight_curve,
            label="pred_pred_global_pred_w",
            lw=0.9,
            ls=(0, (1, 1)),
        )
        ax.set_xlabel("q")
        ax.set_ylabel("I")
        ax.set_title(
            f"sample_{i:03d} true={info['true_type_name']} pred={info['pred_type_name']} "
            f"{selected_source} logRMSE={metrics['curve_log_rmse']:.3g}"
        )
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(curve_dir / f"sample_{i:03d}.png", dpi=140)
        plt.close(fig)
