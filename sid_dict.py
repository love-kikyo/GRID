import torch
import argparse
import os
from collections import defaultdict

BASE_PATH = "./logs/inference/runs/"


def build_sid_dict(map_path):
    id_map = torch.load(map_path)  # [D, N]
    D, N = id_map.shape

    base = 1024
    max_sid = id_map.max().item()
    assert max_sid < base, f"sid overflow: max_sid={max_sid} >= base={base}"

    # --------------------------------------------------
    # 1. build sid_code  [num_item]
    # --------------------------------------------------
    sid_code = torch.zeros(N, dtype=torch.long)
    mult = 1
    for d in range(D):
        sid_code += id_map[d].long() * mult
        mult *= base

    # --------------------------------------------------
    # 2. group items by sid
    # --------------------------------------------------
    code2items = defaultdict(list)
    for item_id, code in enumerate(sid_code.tolist()):
        code2items[code].append(item_id)

    # --------------------------------------------------
    # 3. flatten items + build tensor offsets
    # --------------------------------------------------
    num_sid = sid_code.max().item() + 1
    offsets = torch.zeros(num_sid, 2, dtype=torch.long)

    flat_items = []
    cur = 0
    for code, items in code2items.items():
        l = cur
        r = cur + len(items)
        offsets[code, 0] = l
        offsets[code, 1] = r
        flat_items.extend(items)
        cur = r

    # --------------------------------------------------
    # 4. container (ALL tensors, no dict)
    # --------------------------------------------------
    container = {
        # [num_item]
        "sid_code": sid_code.long(),
        # [num_sid, 2]  (l, r)
        "offsets": offsets,
        # [total_items]
        "items": torch.tensor(flat_items, dtype=torch.long),
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
