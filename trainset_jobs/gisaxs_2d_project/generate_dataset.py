from __future__ import annotations
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
parser.add_argument("--mode", choices=("dry", "full", "demo"), default="full")
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
files = DatasetGenerator(config).write_hdf5_shards(output, count, mode=args.mode)
print(f"generated_files={len(files)} samples={count} output={output}")
