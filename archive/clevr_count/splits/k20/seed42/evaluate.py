"""
Evaluator for classification programs using the new ClassificationDataset.
"""
import sys
import os
# Add the Apptainer bind mount path to Python's module search
sys.path.insert(0, "/work")

print("✅ Added /work to sys.path")
print("Working dir:", os.getcwd())

import time
import json
import logging
import importlib.util
from pycocotools.coco import COCO
from src.eval import evaluate as evaluate_coco
from src.dataset import ClassificationDataset, UnlabelledDataset
import traceback

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def evaluate(program_path, train_img_dir, val_img_dir, train_ann_dir, val_ann_dir, unlabelled_img_dir, unlabelled_ann_dir, seed=42):
    """
    Main evaluation function that tests the classification method
    on the validation set and computes the composite performance metric.

    Args:
        program_path (str): Path to user program (must implement finetune()).
        img_dir (str): Path to image root directory (contains train/, val/).
        ann_dir (str): Path to annotation directory (contains train.json, val.json, etc.).
        seed (int): Random seed.
    """
    start_time = time.time()

    # Dynamically import the user program
    logging.info(f"Loading program from {program_path}")
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)

    if not hasattr(program, "classify"):
        raise AttributeError("❌ The program must define a 'classify(train_dataset, val_dataset)' function.")

    # Load datasets
    logging.info("Loading COCO-format datasets...")
    train_dataset = ClassificationDataset(
        img_dir=train_img_dir,
        ann_dir=train_ann_dir,
    )
    val_dataset = ClassificationDataset(
        img_dir=val_img_dir,
        ann_dir=val_ann_dir,
    )
    unlabelled_dataset = UnlabelledDataset(
        img_dir=unlabelled_img_dir,
        ann_dir=unlabelled_ann_dir,
    )
    # Run the user program
    logging.info("Starting classify()...")
    try:
        coco_results = program.classify(train_dataset, val_dataset, unlabelled_dataset)
    except Exception as e:
        logging.error("❌ Error during classify() execution:")
        logging.error(traceback.format_exc())
        return {"error": str(e)}

    if not coco_results:
        return {"error": "No classification results returned by classify()."}

    # Load COCO ground truth and detection results (any .json in val_ann_dir)
    val_ann_path = os.path.abspath(val_ann_dir)
    if os.path.isfile(val_ann_path):
        ann_file = val_ann_path
    else:
        jsons = sorted([f for f in os.listdir(val_ann_dir) if f.endswith(".json") and not f.startswith(".")])
        if not jsons:
            raise FileNotFoundError(f"No .json file found in {val_ann_dir}")
        ann_file = os.path.join(val_ann_dir, jsons[0])
    coco_gt = COCO(ann_file)
    # pycocotools expects dataset['info']; add if missing (e.g. minimal COCO JSON)
    if "info" not in coco_gt.dataset:
        coco_gt.dataset["info"] = {"description": "", "version": "1.0", "year": ""}
    coco_dt = coco_gt.loadRes(coco_results)

    # Evaluate using COCO metrics
    logging.info("Evaluating predictions...")
    total_metrics = evaluate_coco(coco_gt, coco_dt)
    total_metrics["time_minutes"] = round((time.time() - start_time) / 60, 3)
    total_metrics["program_path"] = program_path

    logging.info("✅ Evaluation complete.")
    return total_metrics


if __name__ == "__main__":
    # Default paths are Apptainer-friendly


    root_dir = os.environ.get("ROOT_DIR", "/work")
    program_path = os.environ.get("PROGRAM_PATH", os.path.join(root_dir, "program.py"))
    data_dir = os.environ.get("DATA_DIR", os.path.join("/data"))

    train_img_dir = os.path.join(data_dir, "images/train")
    val_img_dir = os.path.join(data_dir, "images/val")
    train_ann_dir = os.path.join(root_dir, "annotations/train")
    val_ann_dir = os.path.join(data_dir, "annotations/val")

    unlabelled_img_dir = os.path.join(data_dir, "images/train")
    unlabelled_ann_dir = os.path.join(data_dir, "annotations/train")
    metrics = evaluate(program_path, train_img_dir, val_img_dir, train_ann_dir, val_ann_dir, unlabelled_img_dir, unlabelled_ann_dir)
    metrics["fitness"] = metrics.get("top1_accuracy", 0.0)
    print("METRICS:", metrics)
