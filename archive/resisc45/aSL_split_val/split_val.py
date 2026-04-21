"""
Split a COCO-format train.json into train.json and val.json.

Usage:
    python split_val.py <train_json> <n> [--seed SEED]

Arguments:
    train_json   Path to the input train annotation JSON (will be overwritten).
    n            Number of images per class to move into val.json.
    --seed       Random seed for reproducibility (default: 42).

Output:
    - <train_json>          Overwritten with the remaining train annotations.
    - <train_json_dir>/val.json   Written with the held-out val annotations.
"""

import argparse
import json
import os
import random
from collections import defaultdict 


def split_val(train_json_path, n, seed=42):
    with open(train_json_path) as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]

    # Build lookup: image_id -> annotation
    ann_by_image = {ann["image_id"]: ann for ann in annotations}

    # Group image_ids by category
    by_category = defaultdict(list)
    for ann in annotations:
        by_category[ann["category_id"]].append(ann["image_id"])

    rng = random.Random(seed)
    val_image_ids = set()

    for cat_id, img_ids in by_category.items():
        if len(img_ids) < n:
            raise ValueError(
                f"Category {cat_id} only has {len(img_ids)} images, "
                f"cannot split {n} into val."
            )
        chosen = rng.sample(img_ids, n)
        val_image_ids.update(chosen)

    # Partition images and annotations
    train_images = [img for img in images if img["id"] not in val_image_ids]
    val_images   = [img for img in images if img["id"] in val_image_ids]

    train_anns = [ann for ann in annotations if ann["image_id"] not in val_image_ids]
    val_anns   = [ann for ann in annotations if ann["image_id"] in val_image_ids]

    train_out = {"images": train_images, "annotations": train_anns, "categories": categories}
    val_out   = {"images": val_images,   "annotations": val_anns,   "categories": categories}

    val_json_path = os.path.join(os.path.dirname(train_json_path), "val.json")

    with open(train_json_path, "w") as f:
        json.dump(train_out, f)
    with open(val_json_path, "w") as f:
        json.dump(val_out, f)

    print(f"Train: {len(train_images)} images, {len(train_anns)} annotations")
    print(f"Val:   {len(val_images)} images, {len(val_anns)} annotations")
    print(f"Written: {train_json_path}")
    print(f"Written: {val_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split train.json into train/val by n per class.")
    parser.add_argument("train_json", help="Path to train annotation JSON (will be overwritten).")
    parser.add_argument("n", type=int, help="Number of images per class to move to val.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    args = parser.parse_args()

    split_val(args.train_json, args.n, seed=args.seed)
