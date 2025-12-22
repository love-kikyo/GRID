import os
import argparse
import torch
import tensorflow as tf
from tqdm import tqdm

BASE_PATH = "./data/amazon_data"
def load_item_embeddings(item_dir):
    id_to_emb = {}

    files = sorted([f for f in os.listdir(item_dir)
                   if f.endswith(".tfrecord.gz")])
    print("Found", len(files), "files.")

    for fname in tqdm(files):
        path = os.path.join(item_dir, fname)
        dataset = tf.data.TFRecordDataset(path, compression_type="GZIP")

        for raw_example in dataset:
            ex = tf.train.Example.FromString(raw_example.numpy())
            fid = ex.features.feature["id"].int64_list.value[0]
            emb = ex.features.feature["embedding"].float_list.value

            id_to_emb[fid] = emb

    return id_to_emb


def build_embedding_table(id_to_emb):
    max_id = max(id_to_emb.keys())
    dim = len(next(iter(id_to_emb.values())))

    print("Max item id:", max_id, "embedding dim:", dim)
    table = torch.zeros((max_id + 1, dim), dtype=torch.float32)

    for fid, emb in id_to_emb.items():
        table[fid] = torch.tensor(emb, dtype=torch.float32)

    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, required=True,
                        help="domain name, e.g., beauty / toys / sports")
    args = parser.parse_args()
    item_dir = os.path.join(BASE_PATH, args.domain, "items")
    if not os.path.isdir(item_dir):
        raise ValueError(f"Invalid domain or path does not exist: {item_dir}")

    print(f"Loading embeddings from: {item_dir}")
    id_to_emb = load_item_embeddings(item_dir)
    table = build_embedding_table(id_to_emb)

    output_path = f"./data/amazon_data/{args.domain}/item_embedding_table.pt"
    torch.save(table, output_path)
    print(f"Saved embedding table to: {output_path}")


if __name__ == "__main__":
    main()
