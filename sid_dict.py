import torch
import argparse
import os
from collections import defaultdict

BASE_PATH = "./logs/inference/runs/"
BASE = 1024


def build_prefix_container(sid_code, k):
    """
    sid_code: [N] 完整 sid_code
    k: 使用前 k 个 sid
    """
    prefix_code = sid_code % (BASE ** k)

    code2items = defaultdict(list)
    for item_id, code in enumerate(prefix_code.tolist()):
        code2items[code].append(item_id)

    codes = list(code2items.keys())
    code2idx = {c: i for i, c in enumerate(codes)}

    dense_sid = torch.empty_like(prefix_code)
    for c, items in code2items.items():
        dense_sid[items] = code2idx[c]

    offsets = torch.zeros(len(codes), 2, dtype=torch.long)
    flat_items = []
    cur = 0
    for c in codes:
        items = code2items[c]
        idx = code2idx[c]
        offsets[idx] = torch.tensor([cur, cur + len(items)])
        flat_items.extend(items)
        cur += len(items)

    return {
        "sid_code": dense_sid,
        "offsets": offsets,
        "items": torch.tensor(flat_items, dtype=torch.long),
    }


def build_sid_dict(map_path):
    id_map = torch.load(map_path)  # [D, N]
    D, N = id_map.shape

    max_sid = id_map.max().item()
    assert max_sid < BASE, f"sid overflow: max_sid={max_sid} >= base={BASE}"

    sid_code = torch.zeros(N, dtype=torch.long)
    mult = 1
    for d in range(D):
        sid_code += id_map[d].long() * mult
        mult *= BASE

    prefix_dict = {}
    for k in range(1, D):
        prefix_dict[k] = build_prefix_container(
            sid_code, k
        )

    code2items = defaultdict(list)
    for item_id, code in enumerate(sid_code.tolist()):
        code2items[code].append(item_id)

    codes = list(code2items.keys())
    num_sid = len(codes)

    code2idx = {code: i for i, code in enumerate(codes)}

    dense_sid = torch.empty_like(sid_code)
    for code, items in code2items.items():
        idx = code2idx[code]
        dense_sid[items] = idx

    offsets = torch.zeros(num_sid, 2, dtype=torch.long)

    flat_items = []
    cur = 0
    for code in codes:
        idx = code2idx[code]
        items = code2items[code]
        offsets[idx, 0] = cur
        offsets[idx, 1] = cur + len(items)
        flat_items.extend(items)
        cur += len(items)

    container = {
        "sid_code": dense_sid,          # [num_item] dense index
        "offsets": offsets,             # [num_sid, 2]
        "items": torch.tensor(flat_items, dtype=torch.long),

        "prefix": prefix_dict,
        "code2idx": code2idx,
        "idx2code": codes,
        "base": BASE,
    }

    return container


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True,
        help="logs path, e.g., 2026-01-01/00-00-00"
    )
    args = parser.parse_args()

    sid_map_path = os.path.join(
        BASE_PATH, args.path, "pickle", "merged_predictions_tensor.pt"
    )

    if not os.path.isfile(sid_map_path):
        raise ValueError(f"path does not exist: {sid_map_path}")

    print(f"Loading map from: {sid_map_path}")
    container = build_sid_dict(sid_map_path)

    output_path = os.path.join(
        BASE_PATH, args.path, "pickle", "sid2items_container.pt"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(container, output_path)

    print(f"Saved dict to: {output_path}")


if __name__ == "__main__":
    main()
