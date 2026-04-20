"""
Evaluator for classification programs using a single-image predict interface.
"""
import sys
import os

# Before numpy/sklearn/torch: avoid OpenBLAS/MKL threading segfaults on some clusters.
for _k, _v in (
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
):
    os.environ.setdefault(_k, _v)

# Add the Apptainer bind mount path to Python's module search
sys.path.insert(0, "/work")

import faulthandler

faulthandler.enable()

print("✅ Added /work to sys.path")
print("Working dir:", os.getcwd())

import json
import importlib.util
from PIL import Image

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import ClassificationDataset

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
        x = agent.test_transform(image).unsqueeze(0)
        _, _, predictions = agent.predict(x)
        pred = predictions[0]

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

def _eval(model, loader, sup_metric=False, scores=['rankme', 'ami', 'ari', 'v_measure', 'fmi', 'silhouette', 'dbi', 'chi', 'bnm', 'snd']):
    #model.eval()
    acc = 0.0
    dset_len = len(loader.dataset)
    
    y_feats = []
    y_logits = []
    y_pred = []
    y_probs = []
    y_labels = []
    
    _n_data_processed = 0
    dset = loader.dataset
    print(f'len(dset): {len(dset)}')
    print(f'transform: {dset.transform}')
    n_processed = 0

    with torch.no_grad():
        # for data in loader:
        for image, target in tqdm(loader):

            _n_data_processed += len(image)

            image = image.float().to(model.device)

            logits, feat, predictions = model.predict(image)
            prob = logits.softmax(dim=-1)
            pred = prob.argmax(1)

            pred_cat_ids = torch.tensor([p["category_id"] for p in predictions])
            acc += pred_cat_ids.eq(target).sum().item()

            y_feats.append(feat.cpu())
            y_logits.append(logits.cpu())
            y_pred.append(pred.cpu())
            y_probs.append(prob.cpu())
            y_labels.append(target)

            n_processed += len(image)
    
    y_feats  = torch.cat(y_feats,  dim=0)
    y_logits = torch.cat(y_logits, dim=0)
    y_pred  = torch.cat(y_pred,  dim=0)
    y_probs  = torch.cat(y_probs,  dim=0)
    y_labels = torch.cat(y_labels, dim=0)

    acc = acc / dset_len
    assert n_processed == dset_len, f"n_processed: {n_processed}, dset_len: {dset_len}"
    
    # Lazy import: pulls sklearn + pytorch_adapt validators (heavy; can segfault if threaded BLAS misconfigured).
    #from metrics.metrics import unsupervised_scores

    # eval_dict = unsupervised_scores(y_feats, y_logits, y_probs, scores)
    # if sup_metric:
    eval_dict = {}
    eval_dict["test_acc"] = acc
    return eval_dict

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--program-path", required=True, help="Path to the program .py file to evaluate")
    parser.add_argument("--root-dir", required=True, help="Root dir for the experiment (contains annotations/)")
    parser.add_argument("--setting", default=None, help="Experiment setting: aSSL, aSL, agentSL. Inferred from --root-dir if omitted.")
    parser.add_argument("--out-file", default=None, help="If set, write JSON result to this file instead of (only) stdout")
    args = parser.parse_args()

    root_dir = args.root_dir
    program_path = args.program_path
    data_dir = "/share/j_sun/agentSSL/clevr_count/data"

    # Infer setting from root_dir path if not provided
    if args.setting is not None:
        setting = args.setting
    else:
        parts = os.path.normpath(root_dir).split(os.sep)
        setting = next((p for p in parts if p in ("aSSL", "aSL", "agentSL")), "aSL")
        

    train_img_dir = os.path.join(data_dir, "images", "train")
    train_ann_file = os.path.join(root_dir, "annotations", "train", "train.json")

    test_dir = "/share/j_sun/agentSSL/clevr_count/test"

    test_img_dir = os.path.join(test_dir, "images")
    test_ann_file = os.path.join(test_dir, "annotations", "test.json")

    unlabel_ann_file = os.path.join(root_dir, "annotations", "unlabelled", "unlabelled.json")

    print(f"setting:      {setting}")
    print(f"root_dir:     {root_dir}")
    print(f"program_path: {program_path}")

    program = load_program(program_path)
    agent = program.ClassificationAgent()

    agent.fit(train_img_dir, train_ann_file, unlabel_ann_file)

    test_dataset = ClassificationDataset(test_img_dir, test_ann_file, transform=agent.transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    eval_dict = _eval(agent, test_loader)

    print("METRICS:", eval_dict)

    if args.out_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
        with open(args.out_file, "w") as f:
            json.dump(eval_dict, f, indent=2)