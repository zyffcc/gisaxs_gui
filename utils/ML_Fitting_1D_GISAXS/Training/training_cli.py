"""Command-line contract for slot-model training."""

import argparse

from TrainSetBuild import schema


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset_dir", default="/data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS"
    )
    p.add_argument("--model_dir", default="/data/dust/user/zhaiyufe/Models/ML_1D_Fitting_GISAXS")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--max_points", type=int, default=schema.MAX_POINTS)
    p.add_argument("--quick_test", action="store_true")
    p.add_argument("--reconstruction_loss_weight", type=float, default=0.0)
    p.add_argument("--reconstruction_start_epoch", type=int, default=6)
    p.add_argument("--reconstruction_ramp_epochs", type=int, default=5)
    p.add_argument("--reconstruction_q_stride", type=int, default=16)
    p.add_argument("--reconstruction_samples_per_batch", type=int, default=2)
    p.add_argument("--mixed_precision", action="store_true")
    p.add_argument("--multi_gpu", action="store_true")
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_skipped_nonfinite_batches", type=int, default=10)
    return p.parse_args()
