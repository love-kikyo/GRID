#!/usr/bin/env python3
"""Download or verify the frozen joined TAOBAO-MM source files, never rebuild data.

Existing matching files are reused. Existing mismatches, symlinks and unexpected
Parquet shards are rejected before any download. No files are deleted, and fixed
SID/item2row assets are only hashed, never deserialized or regenerated.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys


REPO_ID = "TaoBao-MM/Taobao-MM"
REVISION = "f185841b6e9612edd06f5112e06416f1bdf6c6d0"
SHARD = re.compile(r"(?:train/train-shard-|test/test-shard-)\d{6}\.parquet")
SHA256 = re.compile(r"[0-9a-f]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_manifest(path):
    raw = Path(path).read_bytes()
    manifest = json.loads(raw)
    require(manifest.get("repo_id") == REPO_ID, "Unexpected upstream repository")
    require(manifest.get("revision") == REVISION, "Unexpected upstream revision")
    required = manifest.get("required_files")
    optional = manifest.get("optional_feature_maps", [])
    require(isinstance(required, list) and required, "Missing required source files")
    require(isinstance(optional, list), "Invalid optional feature-map list")
    names = set()
    for entry in required + optional:
        require(isinstance(entry, dict), "Invalid source-file entry")
        name = entry.get("path")
        require(isinstance(name, str) and name and "\\" not in name and "\0" not in name,
                "Invalid relative source path")
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and ".." not in relative.parts
                and relative.as_posix() == name and name not in names,
                f"Unsafe, noncanonical or repeated source path: {name}")
        names.add(name)
        require(type(entry.get("size_bytes")) is int and entry["size_bytes"] > 0,
                f"Invalid size: {name}")
        require(isinstance(entry.get("sha256"), str) and SHA256.fullmatch(entry["sha256"]),
                f"Invalid SHA256: {name}")
    require(all(SHARD.fullmatch(entry["path"]) for entry in required),
            "Required files must be joined train/test Parquet shards")
    require({entry["path"].split("/")[0] for entry in required} == {"train", "test"},
            "Both official splits are required")
    require(all(entry["path"].startswith("feature_map/") for entry in optional),
            "Optional feature maps must stay under feature_map/")
    assets = manifest.get("frozen_assets", {})
    for name in ("item2row", "sid"):
        entry = assets.get(name, {})
        require(isinstance(entry, dict) and isinstance(entry.get("sha256"), str)
                and SHA256.fullmatch(entry["sha256"]), f"Missing fixed {name} SHA256")
    return manifest, hashlib.sha256(raw).hexdigest()


def target_path(root, relative):
    target = root / relative
    for path in (target, *target.parents):
        if path == root:
            break
        require(not path.is_symlink(), f"Refusing symlink in destination: {path}")
    require(target.resolve().is_relative_to(root), f"Destination escapes source root: {target}")
    return target


def identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def check_file(path, expected_hash, expected_size=None):
    require(path.is_file(), f"Required file is missing or not regular: {path}")
    before = identity(path)
    if expected_size is not None:
        require(before[2] == expected_size, f"Size mismatch; refusing to overwrite: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    require(identity(path) == before, f"File changed during verification: {path}")
    require(digest.hexdigest() == expected_hash, f"SHA256 mismatch; refusing to overwrite: {path}")
    return before


def check_shard_set(root, expected, complete):
    present = set()
    for split in ("train", "test"):
        directory = root / split
        require(not directory.is_symlink(), f"Refusing symlink directory: {directory}")
        if directory.exists():
            require(directory.is_dir(), f"Expected source directory: {directory}")
            present.update(str(path.relative_to(root)) for path in directory.glob("*.parquet"))
    unexpected = present - expected
    require(not unexpected, f"Unexpected Parquet shards: {sorted(unexpected)[:5]}")
    if complete:
        missing = expected - present
        require(not missing, f"Missing Parquet shards ({len(missing)}): {sorted(missing)[:5]}")


def run(manifest_path, destination, *, verify_only=False, include_feature_maps=False,
        item2row=None, sid=None, downloader=None):
    manifest, manifest_hash = load_manifest(manifest_path)
    require((item2row is None) == (sid is None), "--item2row and --sid must be supplied together")
    destination = Path(destination).expanduser()
    require(not destination.is_symlink(), "Destination itself must not be a symlink")
    root = destination.resolve()
    require(not root.exists() or root.is_dir(), "Destination must be a directory")
    files = list(manifest["required_files"])
    if include_feature_maps:
        files.extend(manifest.get("optional_feature_maps", []))
    expected_shards = {entry["path"] for entry in manifest["required_files"]}
    check_shard_set(root, expected_shards, complete=verify_only)

    # Preflight all existing files/assets first, so a late mismatch cannot cause
    # earlier missing files to be downloaded before we discover user modifications.
    verified = {}
    for entry in files:
        target = target_path(root, entry["path"])
        if target.exists() or verify_only:
            verified[entry["path"]] = check_file(target, entry["sha256"], entry["size_bytes"])
    asset_paths = {"item2row": item2row, "sid": sid}
    asset_identities = {}
    if item2row is not None:
        for name, path in asset_paths.items():
            asset_identities[name] = check_file(Path(path), manifest["frozen_assets"][name]["sha256"])

    downloaded = 0
    if not verify_only:
        missing = [entry for entry in files if entry["path"] not in verified]
        if missing and downloader is None:
            from huggingface_hub import hf_hub_download
            downloader = hf_hub_download
        for entry in missing:
            target = target_path(root, entry["path"])
            # Also refuse a mismatching file created after the preflight scan.
            if not target.exists():
                print(f"Downloading {entry['path']}", file=sys.stderr, flush=True)
                result = downloader(repo_id=REPO_ID, repo_type="dataset", revision=REVISION,
                                    filename=entry["path"], local_dir=str(root))
                require(Path(result).resolve() == target.resolve(), "Download returned an unexpected path")
                downloaded += 1
            target = target_path(root, entry["path"])
            verified[entry["path"]] = check_file(target, entry["sha256"], entry["size_bytes"])

    check_shard_set(root, expected_shards, complete=True)
    for name, before in verified.items():
        require(identity(target_path(root, name)) == before, f"Verified source changed: {name}")
    for name, before in asset_identities.items():
        require(identity(Path(asset_paths[name])) == before, f"Fixed asset changed: {name}")
    return dict(status="passed", repo_id=REPO_ID, revision=REVISION,
                manifest_sha256=manifest_hash, destination=str(root),
                verified_files=len(verified), downloaded_files=downloaded,
                shard_counts={split: sum(name.startswith(split + "/") for name in expected_shards)
                              for split in ("train", "test")},
                optional_feature_maps_included=bool(include_feature_maps),
                frozen_assets_verified=bool(asset_identities),
                mode="verify_only" if verify_only else "download_and_verify",
                data_rebuilt=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).with_name("taobao_history_last_user_v1_sources.json"))
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--include-feature-maps", action="store_true")
    parser.add_argument("--item2row", type=Path)
    parser.add_argument("--sid", type=Path)
    args = parser.parse_args()
    try:
        report = run(args.manifest, args.destination, verify_only=args.verify_only,
                     include_feature_maps=args.include_feature_maps, item2row=args.item2row, sid=args.sid)
    except Exception as error:
        parser.exit(1, f"Source verification/download failed: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
