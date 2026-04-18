import json
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import timm
from PIL import Image
from torchvision import transforms


# ============================
# Dataset Loading
# ============================

def load_categories(data):
    categories = sorted(data["categories"], key=lambda x: x["id"])
    cat_ids = [c["id"] for c in categories]
    cat_id_to_name = {c["id"]: c["name"] for c in categories}
    cat_id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
    idx_to_cat_id = {i: cid for cid, i in cat_id_to_idx.items()}

    return cat_ids, cat_id_to_name, cat_id_to_idx, idx_to_cat_id


def load_labeled_data(ann_file: str):
    with open(ann_file, "r") as f:
        data = json.load(f)

    cat_ids, cat_id_to_name, cat_id_to_idx, idx_to_cat_id = load_categories(data)

    id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

    samples = []
    for ann in data["annotations"]:
        samples.append({
            "image_id": ann["image_id"],
            "file_name": id_to_file[ann["image_id"]],
            "label": cat_id_to_idx[ann["category_id"]],
        })

    return samples, cat_ids, cat_id_to_name, cat_id_to_idx, idx_to_cat_id


def load_unlabeled_data(ann_file: str):
    with open(ann_file, "r") as f:
        data = json.load(f)

    samples = []
    for img in data["images"]:
        samples.append({
            "image_id": img["id"],
            "file_name": img["file_name"],
            "label": None,
        })

    return samples


# ============================
# Dataset Classes
# ============================

class ImageDataset(Dataset):
    def __init__(self, img_dir, samples, transform):
        self.img_dir = Path(img_dir)
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(self.img_dir / s["file_name"]).convert("RGB")
        img = self.transform(img)

        if s["label"] is None:
            return img
        return img, s["label"]


# ============================
# Model
# ============================

class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # self.backbone = timm.create_model(
        #     "hf_hub:timm/vit_base_patch16_clip_224.openai",
        #     pretrained=True,
        #     num_classes=0,  # remove head
        # )
        model = timm.create_model("vit_base_patch16_clip_224_petl", 
                                  pretrained=False,
                                  num_classes=0)
        model.load_pretrained(
            '/home/eyl45/SSL-Foundation-Models/pretrain_weight/vit_base_patch16_clip_224_openai.bin')
        model.reset_classifier(num_classes)
        self.backbone = model
        self.head = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits, feats


# ============================
# Agent
# ============================

class ClassificationAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch_size = 32
        self.lr = 1e-3
        self.epochs = 32

        self.use_pseudo = False
        self.pseudo_thresh = 0.95

        self.transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.481, 0.457, 0.408),
                std=(0.268, 0.261, 0.275),
            ),
        ])

        self.transform_eval = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.481, 0.457, 0.408),
                std=(0.268, 0.261, 0.275),
            ),
        ])
        self.test_transform = self.transform_eval

    def fit(self, img_dir, train_ann_file, unlabel_ann_file):

        # ---- Load data ----
        train_samples, cat_ids, cat_id_to_name, cat_id_to_idx, idx_to_cat_id = \
            load_labeled_data(train_ann_file)

        unlabeled_samples = load_unlabeled_data(unlabel_ann_file)

        self.cat_ids = cat_ids
        self.cat_id_to_name = cat_id_to_name
        self.idx_to_cat_id = idx_to_cat_id

        num_classes = len(cat_ids)

        # ---- Datasets ----
        train_ds = ImageDataset(img_dir, train_samples, self.transform_train)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        unlabeled_ds = ImageDataset(img_dir, unlabeled_samples, self.transform_eval)
        unlabeled_loader = DataLoader(unlabeled_ds, batch_size=self.batch_size)

        # ---- Model ----
        self.model = Model(num_classes).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        # ============================
        # Supervised training
        # ============================
        for epoch in range(self.epochs):
            self.model.train()

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits, _ = self.model(images)
                loss = F.cross_entropy(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1} loss: {loss.item()}")
        # ============================
        # Pseudo-labeling
        # ============================
        if self.use_pseudo and len(unlabeled_samples) > 0:
            pseudo_samples = []

            self.model.eval()
            with torch.no_grad():
                for images in unlabeled_loader:
                    images = images.to(self.device)

                    logits, _ = self.model(images)
                    probs = F.softmax(logits, dim=1)

                    conf, pred = probs.max(dim=1)

                    for i in range(len(conf)):
                        if conf[i] > self.pseudo_thresh:
                            pseudo_samples.append({
                                "image_id": -1,
                                "file_name": unlabeled_samples[i]["file_name"],
                                "label": pred[i].item(),
                            })

            if len(pseudo_samples) > 0:
                print(f"Pseudo labels: {len(pseudo_samples)}")

                pseudo_ds = ImageDataset(img_dir, pseudo_samples, self.transform_train)
                pseudo_loader = DataLoader(pseudo_ds, batch_size=self.batch_size, shuffle=True)

                for epoch in range(3):
                    self.model.train()
                    for images, labels in pseudo_loader:
                        images = images.to(self.device)
                        labels = labels.to(self.device)

                        logits, _ = self.model(images)
                        loss = F.cross_entropy(logits, labels)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

    def predict(self, image_batch):
        self.model.eval()

        image_batch = image_batch.to(self.device)
        logits, feats = self.model(image_batch)

        preds = logits.argmax(dim=1)

        outputs = []
        for i in range(len(preds)):
            idx = preds[i].item()
            outputs.append({
                "category_id": int(self.idx_to_cat_id[idx]),
                "score": float(F.softmax(logits, dim=1)[i, idx]),
            })

        return logits, feats, outputs