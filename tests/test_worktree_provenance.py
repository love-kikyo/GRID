"""Path relocation must not weaken training-source content verification."""
import json

from omegaconf import OmegaConf
import pytest

from scripts import export_paired_evaluation as export


SOURCE_NAMES = (
    "src/data/loading/components/dataloading.py",
    "src/data/loading/components/interfaces.py",
    "src/models/modules/semantic_id/tiger_generation_model.py",
)


@pytest.fixture
def relocated_case(tmp_path, monkeypatch):
    original, execution, data = (tmp_path / name for name in ("original", "worktree", "data"))
    data.mkdir()
    for root in (original, execution):
        for name in SOURCE_NAMES:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# frozen {name}\n")
    monkeypatch.setattr(export, "ROOT", execution)
    manifests = {}
    for name, basename in (("train", "protocol_manifest_train.json"),
                           ("test", "protocol_manifest_test.json"),
                           ("audit", "protocol_audit.json")):
        payload = {"protocol": "history-last-user-v1"}
        if name == "audit":
            payload.update(status="passed", destinations={"test": {"records": 5, "unique_users": 5}})
        path = data / basename
        path.write_text(json.dumps(payload))
        manifests[name] = {"sha256": export.file_sha256(path)}
    preflight = {
        "status": "passed", "protocol": "history-last-user-v1", "data_root": str(data),
        "counts": {"test": 5}, "source_manifests": manifests,
        "source_files": [{"path": str(original / name),
                          "sha256": export.file_sha256(original / name)} for name in SOURCE_NAMES],
        "initialization_checkpoint": None,
    }
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(preflight))
    cfg = OmegaConf.create({"data_dir": str(data), "model": {
        "_target_": "src.models.SemanticIDEncoderDecoder", "manual_ckpt_path": None}})
    kwargs = dict(cfg=cfg, split="test", expected_data_root=data,
                  expected_protocol="history-last-user-v1", training_preflight=preflight_path)
    return original, execution, preflight, kwargs


def test_explicit_worktree_relocation_hashes_execution_sources(relocated_case):
    original, execution, _, kwargs = relocated_case
    # Historic source contents are not the executing contents: only their
    # recorded hashes matter. No fallback to reading the original source.
    (original / SOURCE_NAMES[0]).write_text("# changed after training\n")
    result, count = export.validate_provenance(**kwargs, preflight_source_root=original)
    assert count == 5
    relocation = result["source_root_relocation"]
    assert relocation["execution_root"] == str(execution)
    assert relocation["preflight_source_root"] == str(original)
    assert len(relocation["files"]) == 3
    for name, binding in zip(SOURCE_NAMES, relocation["files"]):
        expected = export.file_sha256(execution / name)
        assert binding == {"execution_path": str(execution / name),
                           "preflight_path": str(original / name),
                           "execution_sha256": expected, "training_sha256": expected}
        assert result["verified_training_sources"][str(execution / name)] == expected


def test_worktree_content_change_is_rejected(relocated_case):
    original, execution, _, kwargs = relocated_case
    (execution / SOURCE_NAMES[1]).write_text("# unreviewed execution change\n")
    with pytest.raises(ValueError, match="Training source changed since preflight"):
        export.validate_provenance(**kwargs, preflight_source_root=original)


def test_missing_recorded_source_is_rejected(relocated_case):
    original, _, preflight, kwargs = relocated_case
    preflight["source_files"].pop()
    kwargs["training_preflight"].write_text(json.dumps(preflight))
    with pytest.raises(ValueError, match="lacks required source file"):
        export.validate_provenance(**kwargs, preflight_source_root=original)


def test_missing_execution_source_does_not_fall_back(relocated_case):
    original, execution, _, kwargs = relocated_case
    (execution / SOURCE_NAMES[0]).unlink()
    with pytest.raises(FileNotFoundError):
        export.validate_provenance(**kwargs, preflight_source_root=original)


def test_default_does_not_infer_relocation(relocated_case):
    _, _, _, kwargs = relocated_case
    with pytest.raises(ValueError, match="lacks required source file"):
        export.validate_provenance(**kwargs)


def test_default_same_root_preserves_report_shape(relocated_case):
    _, execution, preflight, kwargs = relocated_case
    for name, record in zip(SOURCE_NAMES, preflight["source_files"]):
        record["path"] = str(execution / name)
    kwargs["training_preflight"].write_text(json.dumps(preflight))
    result, count = export.validate_provenance(**kwargs)
    assert count == 5
    assert "source_root_relocation" not in result


@pytest.mark.parametrize("wrong_hash", [False, True])
def test_relocation_preserves_exact_compatibility_checks(relocated_case, tmp_path, wrong_hash):
    original, execution, preflight, kwargs = relocated_case
    name = SOURCE_NAMES[0]
    old_hash = preflight["source_files"][0]["sha256"]
    (execution / name).write_text("# reviewed execution change\n")
    new_hash = export.file_sha256(execution / name)
    record = {"schema_version": 1,
              "training_preflight_sha256": export.file_sha256(kwargs["training_preflight"]),
              "changes": [{"path": name, "training_sha256": old_hash,
                           "evaluation_sha256": "a" * 64 if wrong_hash else new_hash,
                           "reason": "reviewed test transition", "change_commit": "test-commit"}]}
    compatibility = tmp_path / "compatibility.json"
    compatibility.write_text(json.dumps(record))
    if wrong_hash:
        with pytest.raises(ValueError, match="Training source changed since preflight"):
            export.validate_provenance(**kwargs, preflight_source_root=original,
                                       source_compatibility=compatibility)
    else:
        result, _ = export.validate_provenance(**kwargs, preflight_source_root=original,
                                              source_compatibility=compatibility)
        binding = result["source_root_relocation"]["files"][0]
        assert binding["execution_sha256"] == new_hash
        assert binding["training_sha256"] == old_hash
        assert result["source_compatibility"]["changes"] == record["changes"]
