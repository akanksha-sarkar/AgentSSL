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


def _first_json_in_ann_dir(ann_dir: Path) -> Path:
    """ann_dir is either a .json file or a directory containing one."""
    ann_dir = Path(ann_dir)
    if ann_dir.is_file():
        return ann_dir
    jsons = sorted(ann_dir.glob("*.json"))
    if not jsons:
        raise FileNotFoundError(f"No .json found under {ann_dir}")
    return jsons[0]


def _filename_to_image_id_from_ann(data) -> dict:
    """Build basename -> COCO image id from annotations only (no labels)."""
    if isinstance(data, dict) and "images" in data:
        return {
            Path(img["file_name"]).name: img["id"]
            for img in data["images"]
            if "file_name" in img and "id" in img
        }
    if isinstance(data, list):
        m = {}
        for i, r in enumerate(data):
            name = Path(r.get("image_path", r.get("file_name", ""))).name
            if name:
                m[name] = r.get("id", i + 1)
        return m
    raise ValueError("Annotations must be a COCO dict with 'images' or a list of records")


class UnlabelledDataset(Dataset):
    """
    Images that appear in the annotation file and exist on disk.

    Loads the real JSON on disk (including any annotations block), but only the
    ``images`` / list-record identity fields are used to build ``file_to_coco_id``.
    Nothing on this object exposes ground-truth category_id to user code: ``coco``
    is always None; ``__getitem__`` meta is only file_name and image_id.
    """

    def __init__(self, img_dir, ann_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.coco = None

        with open(_first_json_in_ann_dir(Path(ann_dir))) as f:
            self.file_to_coco_id = _filename_to_image_id_from_ann(json.load(f))

        on_disk = {
            p.name for p in self.img_dir.iterdir()
            if p.is_file() and not p.name.startswith(".")
        }
        self.img_ids = sorted(on_disk & self.file_to_coco_id.keys())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        file_name = self.img_ids[idx]
        img_path = self.img_dir / file_name

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = TF.to_tensor(image)

        meta = {
            "file_name": file_name,
            "image_id": self.file_to_coco_id.get(file_name, None),
        }
        return image, meta


def collate_unlabeled(batch):
    images, metas = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(metas)

class ClassificationDataset(Dataset):
    """
    A dataset for classification in COCO format.
    """
    def __init__(self, img_dir, ann_dir, transform=None,):
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.transform = transform
        self.img_ids = os.listdir(self.img_dir)

        # ann_dir may be a directory (first *.json) or a direct path to a .json file
        p = self.ann_dir
        if p.suffix.lower() == ".json":
            if not p.exists():
                raise FileNotFoundError(f"Annotation file not found: {p}")
            if not p.is_file():
                raise FileNotFoundError(f"Annotation path is not a file: {p}")
            ann_path = p
        elif p.is_dir():
            ann_files = sorted(p.glob("*.json"))
            if len(ann_files) == 0:
                raise FileNotFoundError(f"No .json files found in {p}")
            ann_path = ann_files[0]
        else:
            raise FileNotFoundError(
                f"Annotation path not found (use a .json file or a directory): {p}"
            )

        # ---------- NEW: accept list-format annotations by converting to COCO ----------
        with open(ann_path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            # Convert list-of-records -> COCO dict
            images = []
            annotations = []
            label_set = set()

            for i, r in enumerate(data):
                img_id = i + 1
                file_name = Path(r["image_path"]).name  # basename only (matches your img_dir listing)
                w = int(r.get("width", 0))
                h = int(r.get("height", 0))
                label = int(r["label"])

                images.append({
                    "id": img_id,
                    "file_name": file_name,
                    "width": w,
                    "height": h,
                })

                annotations.append({
                    "id": i + 1,
                    "image_id": img_id,
                    "category_id": label,
                })

                label_set.add(label)

            categories = [{"id": lab, "name": str(lab)} for lab in sorted(label_set)]
            coco_dict = {"images": images, "annotations": annotations, "categories": categories}

            converted_path = ann_path.with_name(ann_path.stem + "_coco.json")
            if not converted_path.exists():  # avoid rewriting every time
                with open(converted_path, "w") as f:
                    json.dump(coco_dict, f)

            ann_path = converted_path
        # ---------------------------------------------------------------------------

        self.coco = COCO(str(ann_path))

        # map filename -> COCO image_id for quick lookup (basename matches os.listdir)
        self.file_to_coco_id = {
            Path(img["file_name"]).name: img["id"]
            for img in self.coco.dataset["images"]
        }

        # keep only images that both exist on disk and have annotations
        annotated_image_ids = {
            ann["image_id"] for ann in self.coco.dataset.get("annotations", [])
        }
        self.img_ids = [
            file_name for file_name in self.img_ids
            if file_name in self.file_to_coco_id
            and self.file_to_coco_id[file_name] in annotated_image_ids
        ]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        file_name = self.img_ids[idx]
        img_path = self.img_dir / file_name

        image = Image.open(img_path).convert("RGB")

        # get COCO image_id from filename
        if file_name not in self.file_to_coco_id:
            raise KeyError(f"{file_name} not found in COCO annotations")
        coco_img_id = self.file_to_coco_id[file_name]

        # get annotations for this image, pick first category_id as label
        ann_ids = self.coco.getAnnIds(imgIds=[coco_img_id])
        anns = self.coco.loadAnns(ann_ids)
        if len(anns) == 0:
            raise RuntimeError(f"No annotations found for image: {file_name}")

        label = int(anns[0]["category_id"])

        # apply transform or default to tensor
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = TF.to_tensor(image)  # float32 in [0,1]

        return image, label

    
def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels
    

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
    "net_name": ["timm/vit_base_patch16_clip_224.openai", "timm/vit_base_patch14_reg4_dinov2.lvd142m"],
    "peft_config": [
        {"method_name": "lora_1", "lora_bottleneck": 4},
        {"method_name": "adaptformer", "ft_mlp_module": "adapter", "ft_mlp_mode": "parallel", "ft_mlp_ln": "before", "adapter_init": "lora_kaiming", "adapter_bottleneck": 4, "adapter_scaler": 0.1},
    ],
    "train_epochs": [50, 100],
}
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--shot", type=str, required=True, help="Setting name")
    parser.add_argument("--seed", type=int, required=True, help="Seed")
    args = parser.parse_args()
    dataset = args.dataset
    shot = args.shot
    seed = args.seed

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

    root_dir = f"/home/eyl45/Sun/AgentSSL/{dataset}/aSSL_backbone/k{shot}/seed{seed}"
    data_dir = f"/share/j_sun/agentSSL/{dataset}/data"
    program_path = os.path.join("setup/warmstart/warm_start_program.py")

    save_path = os.path.join("setup", "warmstart", dataset, f"k{shot}", "results.json")
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

                eval_dict, _, _, _, _, _ = _eval(agent, val_loader, sup_metric=True, scores=scores_to_use)
                eval_dict = calculate_fitness(eval_dict, scores_to_use)

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