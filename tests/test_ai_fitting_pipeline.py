from pathlib import Path

from utils.ai_fitting_pipeline import FittingPipeline, FittingRequest
from utils.ai_fitting_profiles import profile_registry


def test_all_profiles_use_the_same_pipeline_script(tmp_path):
    script = tmp_path / "predict.py"
    script.write_text("print('ok')", encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.keras").write_bytes(b"placeholder")
    input_csv = tmp_path / "curve.csv"
    input_csv.write_text("q,I,sigma\n0.1,1,0.1\n", encoding="utf-8")
    pipeline = FittingPipeline(script)
    commands = []
    for name in ("Fast", "Balanced", "Exhaustive"):
        request = FittingRequest(model_dir, input_csv, tmp_path / name, profile_registry.get(name))
        commands.append(pipeline.build_args(request))
    assert all(command[0] == str(script) for command in commands)
    assert [int(command[command.index("--num_samples") + 1]) for command in commands] == [48, 192, 512]
    assert [int(command[command.index("--refine_top_n") + 1]) for command in commands] == [0, 2, 6]
    assert [command[command.index("--sampling_scales") + 1] for command in commands] == [
        "1.0",
        "0.5,1.0,2.0",
        "0.5,1.0,2.0,4.0",
    ]
    assert all(command[command.index("--score_mode") + 1] == "hybrid_log_relative" for command in commands)


def test_pipeline_normalizes_file_and_saved_model_paths(tmp_path):
    model_dir = tmp_path / "model"
    saved = model_dir / "saved_model"
    saved.mkdir(parents=True)
    (saved / "saved_model.pb").write_bytes(b"pb")
    keras = model_dir / "model.keras"
    keras.write_bytes(b"keras")
    assert FittingPipeline.normalize_model_dir(keras) == model_dir
    assert FittingPipeline.normalize_model_dir(saved) == model_dir
