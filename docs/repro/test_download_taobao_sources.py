"""Offline, standard-library-only downloader tests; no ML imports or network."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SPEC = importlib.util.spec_from_file_location("taobao_source_download", Path(__file__).with_name("download_taobao_sources.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def example(tmp_path):
    content = {"train/train-shard-000000.parquet": b"train record",
               "test/test-shard-000000.parquet": b"test record",
               "feature_map/example.npy": b"fixed feature map"}
    def entry(name):
        return dict(path=name, size_bytes=len(content[name]), sha256=hashlib.sha256(content[name]).hexdigest())
    manifest = dict(repo_id=MODULE.REPO_ID, revision=MODULE.REVISION,
                    required_files=[entry(name) for name in content if name.endswith(".parquet")],
                    optional_feature_maps=[entry("feature_map/example.npy")],
                    frozen_assets={name: dict(sha256=hashlib.sha256(name.encode()).hexdigest())
                                   for name in ("item2row", "sid")})
    manifest_path = tmp_path / "sources.json"
    manifest_path.write_text(json.dumps(manifest))
    destination = tmp_path / "raw"
    calls = []
    def download(**kwargs):
        assert kwargs["repo_id"] == MODULE.REPO_ID
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["revision"] == MODULE.REVISION
        calls.append(kwargs["filename"])
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content[kwargs["filename"]])
        return str(path)
    return manifest_path, destination, manifest, content, calls, download


def test_download_uses_frozen_revision_and_reuses_valid_existing_files(example):
    manifest, root, _, content, calls, download = example
    report = MODULE.run(manifest, root, downloader=download)
    assert report["status"] == "passed" and report["downloaded_files"] == 2
    assert report["shard_counts"] == {"train": 1, "test": 1}
    assert calls == list(content)[:2]
    assert not (root / "feature_map").exists()
    calls.clear()
    assert MODULE.run(manifest, root, downloader=download)["downloaded_files"] == 0
    assert not calls


def test_verify_only_never_imports_downloader_and_allows_metadata(example, monkeypatch):
    manifest, root, _, _, _, download = example
    MODULE.run(manifest, root, downloader=download)
    (root / "README.md").write_text("dataset description")
    (root / "train" / "metadata.json").write_text("{}")
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    result = MODULE.run(manifest, root, verify_only=True)
    assert result["verified_files"] == 2 and not result["frozen_assets_verified"]


def test_optional_maps_and_paired_fixed_assets_are_verified_without_deserialization(example):
    manifest, root, _, _, calls, download = example
    item2row, sid = root.parent / "mapping.pkl", root.parent / "sid.npy"
    item2row.write_bytes(b"item2row")  # Deliberately not a pickle.
    sid.write_bytes(b"sid")  # Deliberately not a NumPy array.
    report = MODULE.run(manifest, root, include_feature_maps=True,
                        item2row=item2row, sid=sid, downloader=download)
    assert report["verified_files"] == 3 and report["frozen_assets_verified"]
    assert calls[-1] == "feature_map/example.npy"
    with pytest.raises(ValueError, match="together"):
        MODULE.run(manifest, root, verify_only=True, item2row=item2row)
    sid.write_bytes(b"modified")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        MODULE.run(manifest, root, verify_only=True, item2row=item2row, sid=sid)


@pytest.mark.parametrize("bad", [b"edited", b"X" * len(b"test record")])
def test_preflight_rejects_mismatch_before_any_download_or_overwrite(example, bad):
    manifest, root, _, _, calls, download = example
    target = root / "test/test-shard-000000.parquet"
    target.parent.mkdir(parents=True)
    target.write_bytes(bad)
    with pytest.raises(ValueError, match="mismatch"):
        MODULE.run(manifest, root, downloader=download)
    assert target.read_bytes() == bad and calls == []
    assert not (root / "train").exists()


def test_unexpected_shard_is_rejected_not_removed(example):
    manifest, root, _, _, calls, download = example
    extra = root / "train/train-shard-999999.parquet"
    extra.parent.mkdir(parents=True)
    extra.write_bytes(b"other user's data")
    with pytest.raises(ValueError, match="Unexpected Parquet"):
        MODULE.run(manifest, root, downloader=download)
    assert extra.is_file() and not calls


def test_verify_only_missing_file_does_not_download(example):
    manifest, root, _, _, calls, download = example
    with pytest.raises(ValueError, match="Missing Parquet"):
        MODULE.run(manifest, root, verify_only=True, downloader=download)
    assert not root.exists() and not calls


@pytest.mark.parametrize("bad_path", ["../escape.parquet", "/absolute.parquet", "train/../escape.parquet",
                                      "train\\escape.parquet", "train//train-shard-000000.parquet"])
def test_manifest_rejects_unsafe_paths_before_writes(example, bad_path):
    manifest, root, payload, _, calls, download = example
    payload["required_files"][0]["path"] = bad_path
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        MODULE.run(manifest, root, downloader=download)
    assert not root.exists() and not calls


@pytest.mark.parametrize("field,value", [("repo_id", "other/repo"), ("revision", "0" * 40)])
def test_manifest_repository_and_revision_are_pinned(example, field, value):
    manifest, root, payload, _, calls, download = example
    payload[field] = value
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Unexpected upstream"):
        MODULE.run(manifest, root, downloader=download)
    assert not calls


def test_manifest_duplicate_filename_is_rejected(example):
    manifest, root, payload, _, calls, download = example
    payload["required_files"].append(payload["required_files"][0])
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="repeated source path"):
        MODULE.run(manifest, root, downloader=download)
    assert not calls


def test_symlink_destination_component_is_rejected(example):
    manifest, root, _, _, calls, download = example
    root.mkdir()
    outside = root.parent / "outside"
    outside.mkdir()
    (root / "train").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        MODULE.run(manifest, root, downloader=download)
    assert not calls and list(outside.iterdir()) == []


def test_download_corruption_fails_verification_and_is_not_deleted(example):
    manifest, root, _, _, _, download = example
    def corrupt(**kwargs):
        path = Path(download(**kwargs))
        path.write_bytes(b"bad download")
        return str(path)
    with pytest.raises(ValueError, match="mismatch"):
        MODULE.run(manifest, root, downloader=corrupt)
    assert (root / "train/train-shard-000000.parquet").read_bytes() == b"bad download"
