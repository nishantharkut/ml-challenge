"""Tests for provenance-safe run manifests and checkpoint reuse."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


class RunContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._temporary_directory.name)

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def test_canonical_json_sha256_is_order_independent_and_utf8_stable(self) -> None:
        from src.run_context import canonical_json, canonical_json_sha256

        left = {"z": [3, 2, 1], "name": "café", "nested": {"b": 2, "a": 1}}
        right = {"nested": {"a": 1, "b": 2}, "name": "café", "z": [3, 2, 1]}
        expected_json = '{"name":"café","nested":{"a":1,"b":2},"z":[3,2,1]}'
        self.assertEqual(canonical_json(left), expected_json)
        self.assertEqual(canonical_json(right), expected_json)
        self.assertEqual(
            canonical_json_sha256(left),
            hashlib.sha256(expected_json.encode("utf-8")).hexdigest(),
        )

    def test_file_fingerprint_hashes_entire_file_and_records_size(self) -> None:
        from src.run_context import fingerprint_file

        artifact = self.tmp_path / "large.bin"
        content = (b"a" * 1_100_000) + b"tail-one"
        artifact.write_bytes(content)
        first = fingerprint_file(artifact)
        self.assertEqual(
            first,
            {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()},
        )
        artifact.write_bytes((b"a" * 1_100_000) + b"tail-two")
        self.assertNotEqual(fingerprint_file(artifact)["sha256"], first["sha256"])

    def test_run_context_resolves_explicit_portable_roots(self) -> None:
        from src.run_context import RunContext

        implementation = self.tmp_path / "Implementation"
        dataset = self.tmp_path / "dataset"
        runs = self.tmp_path / "runs"
        with mock.patch.dict(
            "os.environ",
            {
                "AMAZON_ML_DATASET_DIR": str(dataset),
                "AMAZON_ML_RUNS_DIR": str(runs),
                "AMAZON_ML_RUN_ID": "trial-01",
            },
            clear=False,
        ):
            context = RunContext.from_environment(implementation_root=implementation)

        self.assertEqual(context.run_id, "trial-01")
        self.assertEqual(context.data_root, dataset.resolve())
        self.assertEqual(context.implementation_root, implementation.resolve())
        self.assertEqual(context.runs_root, runs.resolve())
        self.assertEqual(context.run_root, runs.resolve() / "trial-01")

    def test_save_manifest_is_versioned_atomic_and_does_not_mutate_input(self) -> None:
        from src.checkpoint import save_manifest
        from src.run_context import MANIFEST_SCHEMA_VERSION, canonical_json_sha256

        manifest_path = self.tmp_path / "stage_manifest.json"
        manifest_path.write_text('{"legacy":true}', encoding="utf-8")
        artifact = self.tmp_path / "model.bin"
        artifact.write_bytes(b"model-v1")
        metadata = {
            "status": "completed",
            "configuration_hash": "cfg-1",
            "input_hashes": {"train": "train-1"},
        }
        original = json.loads(json.dumps(metadata))
        saved_path = save_manifest(
            manifest_path, metadata, artifacts={"model": artifact}
        )
        self.assertEqual(saved_path, manifest_path)
        self.assertEqual(metadata, original)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], MANIFEST_SCHEMA_VERSION)
        self.assertIs(manifest["completed"], True)
        self.assertEqual(manifest["configuration_hash"], "cfg-1")
        self.assertEqual(manifest["artifacts"]["model"]["size"], len(b"model-v1"))
        self.assertEqual(
            manifest["artifacts"]["model"]["sha256"],
            hashlib.sha256(b"model-v1").hexdigest(),
        )
        unsigned = dict(manifest)
        content_hash = unsigned.pop("content_hash")
        self.assertEqual(content_hash, canonical_json_sha256(unsigned))
        self.assertEqual(list(self.tmp_path.glob("*.tmp")), [])

    def test_checkpoint_accepts_matching_expected_fields_and_artifact(self) -> None:
        from src.checkpoint import is_checkpoint_valid, save_manifest

        manifest_path = self.tmp_path / "manifest.json"
        artifact = self.tmp_path / "features.parquet"
        artifact.write_bytes(b"rows")
        save_manifest(
            manifest_path,
            {
                "configuration_hash": "cfg-1",
                "input_hashes": {"train": "abc", "labels": "def"},
                "producer": "stage_30",
            },
            artifacts={"features": artifact},
        )
        self.assertTrue(
            is_checkpoint_valid(
                manifest_path,
                expected={
                    "configuration_hash": "cfg-1",
                    "input_hashes": {"train": "abc"},
                },
                artifacts={"features": artifact},
            )
        )

    def test_checkpoint_rejects_incomplete_missing_or_mismatched_expected(self) -> None:
        from src.checkpoint import is_checkpoint_valid, save_manifest

        manifest_path = self.tmp_path / "manifest.json"
        save_manifest(manifest_path, {"configuration_hash": "cfg-1"})
        self.assertFalse(
            is_checkpoint_valid(manifest_path, expected={"configuration_hash": "cfg-2"})
        )
        self.assertFalse(is_checkpoint_valid(manifest_path, expected={"run_id": "run-1"}))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["completed"] = False
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.assertFalse(is_checkpoint_valid(manifest_path))
        self.assertFalse(is_checkpoint_valid(self.tmp_path / "missing.json"))

    def test_checkpoint_rejects_changed_missing_and_unrecorded_artifacts(self) -> None:
        from src.checkpoint import is_checkpoint_valid, save_manifest

        manifest_path = self.tmp_path / "manifest.json"
        artifact = self.tmp_path / "model.bin"
        artifact.write_bytes(b"v1")
        save_manifest(manifest_path, {}, artifacts={"model": artifact})
        self.assertTrue(is_checkpoint_valid(manifest_path))
        artifact.write_bytes(b"v2")
        self.assertFalse(is_checkpoint_valid(manifest_path))
        artifact.write_bytes(b"v1")
        self.assertFalse(
            is_checkpoint_valid(manifest_path, artifacts={"unrecorded": artifact})
        )
        artifact.unlink()
        self.assertFalse(is_checkpoint_valid(manifest_path))

    def test_legacy_manifest_is_never_silently_reused(self) -> None:
        from src.checkpoint import is_checkpoint_valid

        manifest_path = self.tmp_path / "legacy.json"
        manifest_path.write_text(json.dumps({"completed": True}), encoding="utf-8")
        artifact = self.tmp_path / "legacy.bin"
        artifact.write_bytes(b"legacy")
        self.assertFalse(is_checkpoint_valid(manifest_path))
        self.assertFalse(
            is_checkpoint_valid(manifest_path, expected={"run_id": "new-run"})
        )
        self.assertFalse(
            is_checkpoint_valid(manifest_path, artifacts={"artifact": artifact})
        )

    def test_artifact_can_be_relocated_by_logical_name(self) -> None:
        from src.checkpoint import is_checkpoint_valid, save_manifest

        manifest_path = self.tmp_path / "manifest.json"
        original = self.tmp_path / "old" / "model.bin"
        original.parent.mkdir()
        original.write_bytes(b"portable-model")
        save_manifest(manifest_path, {"run_id": "r"}, artifacts={"model": original})
        relocated = self.tmp_path / "new" / "model.bin"
        relocated.parent.mkdir()
        relocated.write_bytes(original.read_bytes())
        original.unlink()

        self.assertTrue(
            is_checkpoint_valid(manifest_path, artifacts={"model": relocated})
        )

    def test_checkpoint_rejects_manifest_whose_signed_content_was_modified(self) -> None:
        from src.checkpoint import is_checkpoint_valid, save_manifest

        manifest_path = self.tmp_path / "manifest.json"
        save_manifest(manifest_path, {"run_id": "original"})
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["run_id"] = "tampered"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.assertFalse(is_checkpoint_valid(manifest_path))


if __name__ == "__main__":
    unittest.main()
