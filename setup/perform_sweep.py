import sys
import os

import faulthandler

faulthandler.enable()

import json
import importlib.util
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import torchvision.transforms.functional as TF
from utils.dataset import ClassificationDataset
import numpy as np

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
        "rankme":     {"range": (0.0, 1.0),  "higher_is_better": True},
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
    from metrics.metrics import unsupervised_scores

    eval_dict = unsupervised_scores(y_feats, y_logits, y_probs, scores)
    if sup_metric:
        eval_dict["acc"] = acc
    return eval_dict, y_feats, y_logits, y_pred, y_probs, y_labels


HYPERPARAMETERS = {
    "net_name": ["timm/vit_base_patch14_reg4_dinov2.lvd142m"],
    "peft_config": [
        {"method_name": "lora_1", "lora_bottleneck": 4},
        {"method_name": "adaptformer", "ft_mlp_module": "adapter", "ft_mlp_mode": "parallel", "ft_mlp_ln": "before", "adapter_init": "lora_kaiming", "adapter_bottleneck": 4, "adapter_scaler": 0.1},
    ],
    "train_epochs": [50,100],
}
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--shot", type=int, required=True, help="Shot")
    parser.add_argument("--seed", type=int, required=True, help="Seed")
    parser.add_argument("--eval_method", type=str, required=True, help="Evaluation method")
    args = parser.parse_args()
    dataset = args.dataset
    shot = args.shot
    seed = args.seed
    eval_method = args.eval_method
    if eval_method == "unsupervised":
        from utils.evaluate.unsupervised_metric.SSL.evaluate import _eval as unsup_eval   
    elif eval_method == "noisy_val":
        from utils.evaluate.noisy_val.SSL.evaluate import run_noisy_val
        from utils.evaluate.unsupervised_metric.SSL.evaluate import _eval as eval_fn
    else:
        raise ValueError(f"Invalid evaluation method: {eval_method}")
    if dataset == "clevr_count":
        NUM_CLASSES = 8
    elif dataset == "dtd":
        NUM_CLASSES = 47
    elif dataset == "resisc45":
        NUM_CLASSES = 45
    elif dataset == "kitti":
        NUM_CLASSES = 4
    elif dataset == "sun397":
        NUM_CLASSES = 397
    elif dataset == "retino":
        raise ValueError(f"Invalid dataset: {dataset}")
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    root_dir = f"/home/eyl45/Sun/AgentSSL/{dataset}/aSSL_backbone_{eval_method}/k{shot}/seed{seed}"
    data_dir = f"/share/j_sun/agentSSL/{dataset}/data"
    program_path = os.path.join("setup/warmstart/warm_start_program.py")

    save_path = os.path.join("setup", "warmstart", dataset, eval_method, f"k{shot}", "results.json")
    if os.path.exists(save_path):
        print(f"Results file already exists: {save_path}")
        exit()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    train_img_dir = os.path.join(data_dir, "images", "train")
    train_ann_file = os.path.join(root_dir, "annotations", "train", "train.json")
    unlabel_ann_file = os.path.join(root_dir, "annotations", "unlabelled", "unlabelled.json")
    val_img_dir = os.path.join(data_dir, "images", "val")
    val_ann_file = os.path.join(data_dir, "annotations", "val", "val.json")

    program = load_program(program_path)
    results = []
    # Perform sweep
    from nets.net_builder import get_net_builder
    from nets.peft import get_peft_config
    for net_name in HYPERPARAMETERS["net_name"]:
        for peft_config in HYPERPARAMETERS["peft_config"]:
            for train_epochs in HYPERPARAMETERS["train_epochs"]:
                print("Running: ", net_name, peft_config, train_epochs)
                if eval_method == "unsupervised":
                    agent = program.ClassificationAgent(net_builder_fn=get_net_builder, 
                                                        get_peft_config_fn=get_peft_config, 
                                                        num_classes=NUM_CLASSES, 
                                                        net_name=net_name, 
                                                        peft_config=peft_config, 
                                                        train_epochs=train_epochs)
                    agent.fit(train_img_dir, train_ann_file, unlabel_ann_file)
                    val_dataset = ClassificationDataset(val_img_dir, val_ann_file, transform=agent.transform)
                    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
                    scores_to_use = ['rankme', 'ami', 'ari', 'v_measure', 'fmi', 'silhouette', 'bnm', 'snd']

                    eval_dict, _, _, _, _, _ = unsup_eval(agent, val_loader, sup_metric=True, scores=scores_to_use)
                    eval_dict = calculate_fitness(eval_dict, scores_to_use)
                elif eval_method == "noisy_val":
                    acc_lst = []
                    for n in range(4):
                        train_ann_file_partial = os.path.join(root_dir, "annotations", "val", f"seed{n}", "annotations",  "train", "train.json")
                        val_ann_file_partial = os.path.join(root_dir, "annotations", "val", f"seed{n}", "annotations", "noisy_val", "noisy_val.json")
                        agent = program.ClassificationAgent(net_builder_fn=get_net_builder, 
                                                            get_peft_config_fn=get_peft_config, 
                                                            num_classes=NUM_CLASSES, 
                                                            net_name=net_name, 
                                                            peft_config=peft_config, 
                                                            train_epochs=train_epochs)
                        agent.fit(train_img_dir, train_ann_file_partial, unlabel_ann_file)
                        val_dataset = ClassificationDataset(val_img_dir, val_ann_file_partial, transform=agent.transform)
                        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
                        acc = run_noisy_val(agent, val_img_dir, val_ann_file_partial)
                        print(f"Noisy val accuracy: {acc}")
                        acc_lst.append(acc)
                    eval_dict = {"fitness": np.mean(acc_lst)}
                    print(f"Noisy val accuracy: {eval_dict['fitness']}")
                    
                    # Compute full val set accuracy
                    agent = program.ClassificationAgent(net_builder_fn=get_net_builder, 
                                                        get_peft_config_fn=get_peft_config, 
                                                        num_classes=NUM_CLASSES, 
                                                        net_name=net_name, 
                                                        peft_config=peft_config, 
                                                        train_epochs=train_epochs)
                    agent.fit(train_img_dir, train_ann_file, unlabel_ann_file)
                    val_dataset = ClassificationDataset(val_img_dir, val_ann_file, transform=agent.transform)
                    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
                    acc_dict, _, _, _, _, _ = eval_fn(agent, val_loader, sup_metric=True, scores=[])
                    eval_dict["acc"] = acc_dict["acc"]
                    print(f"Full val set accuracy: {eval_dict['acc']}")
                print(f"Result: Net name: {net_name}, Peft config: {peft_config}, Train epochs: {train_epochs}, Fitness: {eval_dict['fitness']}, Acc: {eval_dict['acc']}")
                results.append({
                    "net_name": net_name,
                    "peft_config": peft_config,
                    "train_epochs": train_epochs,
                    "fitness": eval_dict['fitness'],
                    "acc": eval_dict['acc']
                })
    # Save results to json
    with open(save_path, "w") as f:
        json.dump(results, f)