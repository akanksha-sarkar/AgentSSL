"""
Evaluator for classification programs using a single-image predict interface.
"""
import sys
import os

# Add the Apptainer bind mount path to Python's module search
sys.path.insert(0, "/work")

print("✅ Added /work to sys.path")
print("Working dir:", os.getcwd())

import json
import importlib.util
from PIL import Image


def load_program(program_path):
    spec = importlib.util.spec_from_file_location("program", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_ground_truth(val_ann_file):
    with open(val_ann_file, "r") as f:
        coco = json.load(f)

    # map image_id -> file_name
    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

    # map file_name -> category_id
    gt = {}
    for ann in coco["annotations"]:
        image_id = ann["image_id"]
        category_id = ann["category_id"]

        if image_id in id_to_file:
            file_name = id_to_file[image_id]
            gt[file_name] = category_id

    return gt


def evaluate_agent(agent, val_img_dir, ground_truth):
    correct = 0
    total = 0

    for file_name, true_cat in ground_truth.items():
        img_path = os.path.join(val_img_dir, file_name)

        if not os.path.isfile(img_path):
            continue

        image = Image.open(img_path).convert("RGB")
        pred = agent.predict(image)

        if not isinstance(pred, dict):
            continue
        if "category_id" not in pred:
            continue

        total += 1
        if pred["category_id"] == true_cat:
            correct += 1

    acc = correct / total if total > 0 else 0.0

    return {
        "top1_accuracy": acc,
        "fitness": acc,
    }



if __name__ == "__main__":

    root_dir = os.environ.get("ROOT_DIR", "/work")
    program_path = os.environ.get("PROGRAM_PATH", os.path.join(root_dir, "program.py"))
    data_dir = os.environ.get("DATA_DIR", os.path.join("/data"))

    train_img_dir = os.path.join(data_dir, "images", "train")
    train_ann_file = os.path.join(root_dir, "annotations", "train", "train.json")

    val_img_dir = os.path.join(data_dir, "images", "val")
    val_ann_file = os.path.join(data_dir, "annotations", "val", "val.json")

    program = load_program(program_path)
    agent = program.ClassificationAgent()

    agent.fit(train_img_dir, train_ann_file)

    ground_truth = load_ground_truth(val_ann_file)
    metrics = evaluate_agent(agent, val_img_dir, ground_truth)

    print("METRICS:", metrics)