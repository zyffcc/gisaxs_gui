# Training

Train a TensorFlow 2.15 slot model for 1D GISAXS/SAXS fitting. `Training/train.py` automatically uses TFRecord shards when `train/*.tfrecord` exists, with NPZ kept only as a debug fallback.

Quick training:

```bash
conda activate tf
python Training/train.py \
  --dataset_dir /data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS_QUICK \
  --model_dir /data/dust/user/zhaiyufe/Models/ML_1D_Fitting_GISAXS_QUICK \
  --epochs 2 \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --quick_test
```

Full training:

```bash
sbatch Training/training.sbatch 50 64 0.0001
```

Training resumes automatically from the latest checkpoint in `MODEL_DIR/checkpoints`.
Progress is printed every `--log_interval` steps and saved every `--save_interval` steps;
the defaults are both 10. Each save updates `model.keras`, `history.json`,
`step_history.json`, `step_history.csv`, and `loss_curve.png`.

Train a curriculum dataset after its build job finishes:

```bash
# K = 1 model, after build array 22984596 finishes
sbatch --dependency=afterany:22984596 Training/training.sbatch \
  50 64 0.0001 \
  /data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS_K1 \
  /data/dust/user/zhaiyufe/Models/ML_1D_Fitting_GISAXS_K1 \
  --log_interval 10 \
  --save_interval 10

# K = 3/4 model, after build array 22984619 finishes
sbatch --dependency=afterany:22984619 Training/training.sbatch \
  50 64 0.0001 \
  /data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS_K3K4 \
  /data/dust/user/zhaiyufe/Models/ML_1D_Fitting_GISAXS_K3K4 \
  --log_interval 10 \
  --save_interval 10
```

Predict TOP 20:

```bash
python Training/predict_topk.py \
  --model_dir /data/dust/user/zhaiyufe/Models/ML_1D_Fitting_GISAXS_QUICK \
  --input_csv /data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS_QUICK/inspection/example_curve.csv \
  --output_dir /data/dust/user/zhaiyufe/Models/ML_1D_Fitting_GISAXS_QUICK/prediction_example \
  --num_samples 200 \
  --top_k 20
```
