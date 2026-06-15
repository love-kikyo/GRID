import json
from pathlib import Path
from typing import Any, Dict


def build_candidate_manifest(
    output_path: str,
    root_output_dir: str,
    num_hierarchies: int,
    num_embeddings_per_hierarchy: int,
    top_k_for_generation: int,
    train_split: str = "training",
    val_split: str = "evaluation",
    test_split: str = "testing",
) -> Dict[str, Any]:
    root = Path(root_output_dir)
    train_dir = root / f"{train_split}_candidate_cache"
    val_dir = root / f"{val_split}_candidate_cache"
    test_dir = root / f"{test_split}_candidate_cache"

    def _load_split_manifest(split_dir: Path) -> Dict[str, Any]:
        manifest_path = split_dir / "shard_manifest.json"
        with manifest_path.open("r", encoding="utf-8") as handle:
            shard_manifest = json.load(handle)
        shard_manifest["manifest_path"] = str(manifest_path.resolve())
        shard_manifest["shard_paths"] = [
            str((split_dir / shard).resolve()) for shard in shard_manifest["shards"]
        ]
        shard_manifest["shard_lengths"] = [
            shard_info["num_rows"] for shard_info in shard_manifest.get("shard_sizes", [])
        ]
        return shard_manifest

    manifest = {
        "root_output_dir": str(root.resolve()),
        "num_hierarchies": num_hierarchies,
        "num_embeddings_per_hierarchy": num_embeddings_per_hierarchy,
        "num_candidates": top_k_for_generation,
        "splits": {
            "train": _load_split_manifest(train_dir),
            "val": _load_split_manifest(val_dir),
            "test": _load_split_manifest(test_dir),
        },
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)

    return manifest


def load_candidate_manifest_value(manifest_path: str, key_path: str) -> Any:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    value: Any = manifest
    for part in key_path.split("."):
        value = value[part]
    return value
