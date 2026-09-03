from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PosteriorV8.artifact import (
    MANIFEST_FILE,
    TOPOLOGY_CATALOG_SHA256,
    ArtifactContract,
    ArtifactValidationError,
    canonical_json_bytes,
    validate_manifest,
    write_manifest_atomic,
)
from PosteriorV8.contract import FORWARD_MODEL_VERSION, TOPOLOGIES


class PosteriorV8ArtifactTests(unittest.TestCase):
    def make_artifact(self, root: Path) -> ArtifactContract:
        model_path = root / "model.keras"
        model_path.write_bytes(b"deterministic-posterior-v8-model")
        contract = ArtifactContract.from_model_file(model_path)
        write_manifest_atomic(root / MANIFEST_FILE, contract)
        return contract

    def rewrite_payload(self, root: Path, update) -> None:
        path = root / MANIFEST_FILE
        payload = json.loads(path.read_text(encoding="utf-8"))
        update(payload)
        path.write_bytes(canonical_json_bytes(payload) + b"\n")

    def test_valid_manifest_round_trip_is_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = self.make_artifact(root)

            actual = validate_manifest(root)

            self.assertEqual(actual, expected)
            self.assertEqual(actual.model_file, "model.keras")
            self.assertEqual(actual.topology_catalog_sha256, TOPOLOGY_CATALOG_SHA256)
            with self.assertRaises(FrozenInstanceError):
                actual.model_file = "other.keras"
            with self.assertRaises(TypeError):
                actual.manifest_payload()["model_file"] = "other.keras"

    def test_topology_hash_is_stable_canonical_json_of_id_order(self):
        payload = {
            "topologies": [
                {"id": topology_id, "shapes": list(shapes)}
                for topology_id, shapes in enumerate(TOPOLOGIES)
            ]
        }
        import hashlib

        expected = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        self.assertEqual(TOPOLOGY_CATALOG_SHA256, expected)

    def test_model_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_artifact(root)
            with (root / "model.keras").open("ab") as handle:
                handle.write(b"tampered")

            with self.assertRaisesRegex(ArtifactValidationError, "checksum mismatch"):
                validate_manifest(root)

    def test_missing_manifest_field_and_model_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_artifact(root)
            self.rewrite_payload(root, lambda value: value.pop("codec_version"))
            with self.assertRaisesRegex(ArtifactValidationError, "missing required fields"):
                validate_manifest(root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_artifact(root)
            (root / "model.keras").unlink()
            with self.assertRaisesRegex(ArtifactValidationError, "model file is missing"):
                validate_manifest(root)

    def test_unknown_field_and_manifest_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_artifact(root)
            self.rewrite_payload(root, lambda value: value.__setitem__("future_magic", True))
            with self.assertRaisesRegex(ArtifactValidationError, "unsupported fields"):
                validate_manifest(root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_artifact(root)
            manifest = root / MANIFEST_FILE
            actual = root / "actual-manifest.json"
            manifest.replace(actual)
            manifest.symlink_to(actual.name)
            with self.assertRaisesRegex(ArtifactValidationError, "must not be a symbolic link"):
                validate_manifest(root)

    def test_version_and_topology_hash_mismatches_are_rejected(self):
        mutations = (
            ("forward_model_version", FORWARD_MODEL_VERSION + "_stale"),
            ("topology_catalog_sha256", "0" * 64),
        )
        for field, value in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.make_artifact(root)
                self.rewrite_payload(root, lambda payload: payload.__setitem__(field, value))
                with self.assertRaisesRegex(ArtifactValidationError, "incompatible"):
                    validate_manifest(root)

    def test_atomic_write_refuses_overwrite_and_preserves_existing_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = self.make_artifact(root)
            path = root / MANIFEST_FILE
            original = path.read_bytes()

            with self.assertRaises(FileExistsError):
                write_manifest_atomic(path, contract)

            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(root.glob(f".{MANIFEST_FILE}.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
