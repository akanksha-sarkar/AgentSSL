"""
Evaluator for classification programs using a single-image predict interface.
"""
import sys
import os
import time
import numpy as np

# Replaced by setup/setup.py from setup/dataset.json (num_classes for this dataset)
NUM_CLASSES = None

exp_setting = "SSL"
eval_setting = "noisy_val"
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

try:
    from utils.dataset import ClassificationDataset
except ImportError:
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
        x = agent.transform(image).unsqueeze(0)
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

def calculate_fitness(eval_dict, scores_to_use):
    """
    Normalize each metric using predefined global ranges and compute mean fitness.
    """

    # Define ranges and direction (True = higher is better)
    METRIC_INFO = {
        "ami":        {"range": (-1.0, 1.0), "higher_is_better": True},
        "ari":        {"range": (-1.0, 1.0), "higher_is_better": True},
        "v_measure":  {"range": (0.0, 1.0),  "higher_is_better": True},
        "fmi":        {"range": (0.0, 1.0),  "higher_is_better": True},
        "silhouette": {"range": (-1.0, 1.0), "higher_is_better": True},
        #"dbi":        {"range": (0.0, 10.0), "higher_is_better": False},  # capped to 10
        #"chi":        {"range": (0.0, 1000.0), "higher_is_better": True}, # capped to 1000
        #"rankme":     {"range": (0.0, 1.0),  "higher_is_better": True},
        "bnm":        {"range": (0.0, 1.0),  "higher_is_better": True},  
        "snd":        {"range": (0.0, 1.0),  "higher_is_better": True},  
    }

    normalized = []

    for s in scores_to_use:
        if s not in METRIC_INFO:
            raise ValueError(f"No normalization info for metric: {s}")

        v = eval_dict[s]
        min_val, max_val = METRIC_INFO[s]["range"]
        higher_is_better = METRIC_INFO[s]["higher_is_better"]

        # clip to range to avoid exploding values
        v = max(min(v, max_val), min_val)

        # normalize to [0,1]
        if max_val - min_val == 0:
            norm = 0.0
        else:
            norm = (v - min_val) / (max_val - min_val)

        # invert if lower is better
        if not higher_is_better:
            norm = 1.0 - norm

        normalized.append(norm)

    fitness = sum(normalized) / len(normalized)
    eval_dict["fitness"] = fitness
    return eval_dict

def _eval(model, loader, sup_metric=True, scores=['rankme', 'ami', 'ari', 'v_measure', 'fmi', 'silhouette', 'dbi', 'chi', 'bnm', 'snd']):
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

    print("[eval] forward pass start", flush=True)
    t0 = time.perf_counter()
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
    print(f"[eval] forward pass done in {time.perf_counter() - t0:.2f}s", flush=True)

    t1 = time.perf_counter()
    y_feats  = torch.cat(y_feats,  dim=0)
    y_logits = torch.cat(y_logits, dim=0)
    y_pred  = torch.cat(y_pred,  dim=0)
    y_probs  = torch.cat(y_probs,  dim=0)
    y_labels = torch.cat(y_labels, dim=0)
    print(f"[eval] stacked tensors in {time.perf_counter() - t1:.2f}s", flush=True)

    acc = acc / dset_len
    assert n_processed == dset_len, f"n_processed: {n_processed}, dset_len: {dset_len}"
    
    eval_dict = {}
    if not sup_metric:
        # Lazy import: pulls sklearn + pytorch_adapt validators (heavy; can segfault if threaded BLAS misconfigured).
        from metrics.metrics import unsupervised_scores

        print("[eval] unsupervised metrics start", flush=True)
        t2 = time.perf_counter()
        eval_dict = unsupervised_scores(y_feats, y_logits, y_probs, scores)
        print(f"[eval] unsupervised metrics done in {time.perf_counter() - t2:.2f}s", flush=True)
    else:
        eval_dict["acc"] = acc
    return eval_dict, y_feats, y_logits, y_pred, y_probs, y_labels


def run_noisy_val(agent, val_img_dir, val_ann_file): 
    print(f"[noisy_val] val_img_dir={val_img_dir}", flush=True)
    print(f"[noisy_val] val_ann_file={val_ann_file}", flush=True)
    t0 = time.perf_counter()
    val_dataset = ClassificationDataset(val_img_dir, val_ann_file, transform=agent.transform)
    print(f"[noisy_val] ClassificationDataset built len={len(val_dataset)} in {time.perf_counter() - t0:.2f}s", flush=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    eval_dict, y_feats, y_logits, y_pred, y_probs, y_labels = _eval(agent, val_loader)
    return eval_dict["acc"] 

if __name__ == "__main__":

    root_dir = os.environ.get("ROOT_DIR", "/work")
    program_path = os.environ.get("PROGRAM_PATH", os.path.join(root_dir, "program.py"))
    data_dir = os.environ.get("DATA_DIR", os.path.join("/data"))

    train_img_dir = os.path.join(data_dir, "images", "train")
    train_ann_file = os.path.join(root_dir, "annotations", "train", "train.json")
    unlabel_ann_file = os.path.join(root_dir, "annotations", "unlabelled", "unlabelled.json")

    print(f"[main] ROOT_DIR={root_dir}", flush=True)
    print(f"[main] DATA_DIR={data_dir}", flush=True)
    print(f"[main] PROGRAM_PATH={program_path}", flush=True)
    print(f"[main] train_img_dir={train_img_dir}", flush=True)
    print(f"[main] train_ann_file={train_ann_file}", flush=True)
    print(f"[main] unlabel_ann_file={unlabel_ann_file}", flush=True)

    t0 = time.perf_counter()
    program = load_program(program_path)
    print(f"[main] load_program done in {time.perf_counter() - t0:.2f}s", flush=True)

    if eval_setting == "noisy_val":
        acc_lst = []
        for n in range(4):
            print(f"[main] --- noisy_val seed {n} ---", flush=True)
            train_img_dir = os.path.join(data_dir, "images", "train")
            train_ann_file = os.path.join(root_dir, "annotations", "val", f"seed{n}", "annotations",  "train", "train.json")

            val_img_dir = os.path.join(data_dir, "images", "train")
            val_ann_file = os.path.join(root_dir, "annotations", "val", f"seed{n}", "annotations", "noisy_val", "noisy_val.json")

            from nets.net_builder import get_net_builder
            from nets.peft import get_peft_config
            agent = program.ClassificationAgent(net_builder_fn=get_net_builder, get_peft_config_fn=get_peft_config, num_classes=NUM_CLASSES)

            print(f"[main] refit train_ann_file={train_ann_file}", flush=True)
            print(f"[main] refit val_ann_file={val_ann_file}", flush=True)
            t_fit = time.perf_counter()
            if exp_setting == "SL": 
                agent.fit(train_img_dir, train_ann_file)
            else:
                agent.fit(train_img_dir, train_ann_file, unlabel_ann_file) 
            print(f"[main] refit done in {time.perf_counter() - t_fit:.2f}s", flush=True)
         
            t_nv = time.perf_counter()
            acc = run_noisy_val(agent, val_img_dir, val_ann_file)
            print(f"[main] run_noisy_val acc={acc} in {time.perf_counter() - t_nv:.2f}s", flush=True)
            acc_lst.append(acc)

        eval_dict = {"fitness": np.mean(acc_lst)}
        print("METRICS:", eval_dict)

    else: 

        print("[main] initial fit start", flush=True)
        t1 = time.perf_counter()
        if exp_setting == "SL": 
            agent.fit(train_img_dir, train_ann_file)
        else:
            agent.fit(train_img_dir, train_ann_file, unlabel_ann_file) 
        print(f"[main] initial fit done in {time.perf_counter() - t1:.2f}s", flush=True)
        val_img_dir = os.path.join(data_dir, "images", "val")
        val_ann_file = os.path.join(data_dir, "annotations", "val", "val.json")

        val_dataset = ClassificationDataset(val_img_dir, val_ann_file, transform=agent.transform)
        val_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

        possible_scores = ['ami', 'ari', 'v_measure', 'fmi', 'silhouette', 'bnm', 'snd']
        scores_to_use = ['ami', 'ari', 'v_measure', 'fmi', 'silhouette', 'bnm', 'snd']

        eval_dict, _, _, _, _, _ = _eval(agent, val_loader, scores=scores_to_use)
        eval_dict = calculate_fitness(eval_dict, scores_to_use)

        print("METRICS:", eval_dict)