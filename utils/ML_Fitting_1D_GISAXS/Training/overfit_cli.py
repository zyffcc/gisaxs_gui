"""Command-line contract for the K=1 overfit diagnostic."""

import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train_dir",
        default="/data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS_K1/train",
        help="K=1 training split directory containing TFRecord shards",
    )
    p.add_argument("--num_samples", type=int, default=256)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument(
        "--noise_frac", type=float, default=0.01, help="Low-noise sigma as fraction of I_clean"
    )
    p.add_argument(
        "--debug_max_points",
        type=int,
        default=256,
        help="Cap effective max points for debug overfit runs",
    )
    p.add_argument(
        "--simple_k1_loss",
        action="store_true",
        help="Use a direct K=1 supervised loss instead of permutation matching.",
    )
    p.add_argument(
        "--plot_n", type=int, default=16, help="Number of samples to render after training."
    )
    p.add_argument(
        "--skip_model_save", action="store_true", help="Skip saving overfit_debug_model.keras."
    )
    p.add_argument(
        "--use_true_global_for_plot",
        action="store_true",
        help="Use true global parameters for the main predicted-curve plot/debug metric.",
    )
    p.add_argument("--print_every", type=int, default=10)
    p.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation/plot inference to avoid OOM",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output_dir",
        default="/home/zhaiyufe/PycharmProjects/ML_Fitting_1D_GISAXS/Training/overfit_debug_output",
    )
    return p.parse_args()
