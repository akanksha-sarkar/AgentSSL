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
print("Imported dataset...")
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
    print("Calculated acc...")
    print("acc: ", acc)
    assert n_processed == dset_len, f"n_processed: {n_processed}, dset_len: {dset_len}"
    
    # Lazy import: pulls sklearn + pytorch_adapt validators (heavy; can segfault if threaded BLAS misconfigured).
    # from metrics.metrics import unsupervised_scores

    # eval_dict = unsupervised_scores(y_feats, y_logits, y_probs, scores)
    eval_dict = {"acc": acc}
        # if sup_metric:
        #     eval_dict["acc"] = acc
    return eval_dict, y_feats, y_logits, y_pred, y_probs, y_labels

if __name__ == "__main__":

    # root_dir = os.environ.get("ROOT_DIR", "/work")
    # program_path = os.environ.get("PROGRAM_PATH", os.path.join(root_dir, "program.py"))
    # data_dir = os.environ.get("DATA_DIR", os.path.join("/data"))

    root_dir = "/home/eyl45/Sun/AgentSSL/resisc45/aStructSSL_clip/k2/seed0"
    data_dir = "/share/j_sun/agentSSL/resisc45/data"
    program_path = os.path.join(root_dir, "program2.py")
    
    train_img_dir = os.path.join(data_dir, "images", "train")
    train_ann_file = os.path.join(root_dir, "annotations", "train", "train.json")
    unlabel_ann_file = os.path.join(root_dir, "annotations", "unlabelled", "unlabelled.json")
    val_img_dir = os.path.join(data_dir, "images", "val")
    val_ann_file = os.path.join(data_dir, "annotations", "val", "val.json")


    program = load_program(program_path)

    print("Loaded program...")
    net_name_options = ["timm/vit_base_patch16_clip_224.openai","timm/vit_base_patch14_reg4_dinov2.lvd142m"]
    agent = program.SSL_Algorithm(net_name_options=net_name_options)
    print("Created agent...")
    agent.fit(train_img_dir, train_ann_file, unlabel_ann_file)
    print("Fitted agent...")
    val_dataset = ClassificationDataset(val_img_dir, val_ann_file, transform=agent.transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    print("Created val loader...")
    possible_scores = ['rankme', 'ami', 'ari', 'v_measure', 'fmi', 'silhouette', 'dbi', 'chi', 'bnm', 'snd']
    scores_to_use = []
    eval_dict, _, _, _, _, _ = _eval(agent, val_loader, sup_metric=True,scores=scores_to_use)

    ground_truth = load_ground_truth(val_ann_file)
    #eval_dict2 = evaluate_agent(agent, val_img_dir, ground_truth)
    eval_dict["fitness"] = eval_dict["acc"]

    print("METRICS:", eval_dict)
    #print("METRICS2:", eval_dict2)