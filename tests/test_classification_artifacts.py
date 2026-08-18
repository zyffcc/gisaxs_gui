import csv

from src.gimap.features.classification.infrastructure import (
    LocalClassificationArtifactRepository,
)


def test_local_classification_artifacts_preserve_json_and_csv_shapes(tmp_path) -> None:
    repository = LocalClassificationArtifactRepository()
    session_path = tmp_path / "session.json"
    csv_path = tmp_path / "results.csv"

    repository.save_session(session_path, {"sources": [], "ranking_metric": "macro_f1"})
    repository.export_csv(csv_path, ("rank", "name"), ((1, "SVM"),))

    assert repository.load_session(session_path) == {
        "sources": [],
        "ranking_metric": "macro_f1",
    }
    with csv_path.open(newline="", encoding="utf-8") as handle:
        assert list(csv.reader(handle)) == [["rank", "name"], ["1", "SVM"]]
