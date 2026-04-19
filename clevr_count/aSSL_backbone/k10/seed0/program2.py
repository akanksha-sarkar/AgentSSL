import json
import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class LabeledDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        with open(ann_file, "r") as f:
            coco = json.load(f)

        self.img_dir = img_dir
        self.transform = transform

        id_to_fname = {img["id"]: img["file_name"] for img in coco["images"]}

        cat_ids = sorted(cat["id"] for cat in coco["categories"])
        self.cat_id_to_label = {cid: i for i, cid in enumerate(cat_ids)}
        self.label_to_cat_id = {i: cid for i, cid in enumerate(cat_ids)}

        self.samples = []
        for ann in coco["annotations"]:
            fname = id_to_fname[ann["image_id"]]
            label = self.cat_id_to_label[ann["category_id"]]
            self.samples.append((fname, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        path = os.path.join(self.img_dir, fname)
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def get_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])


class ClassificationAgent:
    def __init__(self, net_builder_fn, get_peft_config_fn, num_classes=8):
        self.num_classes = num_classes
        self.net_builder_fn = net_builder_fn
        self.get_peft_config_fn = get_peft_config_fn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = get_val_transform()
        self.label_to_cat_id = None

        self.model = self._build_model()

    def _build_model(self):
        net_name = "timm/vit_base_patch16_clip_224.openai"

        peft_config = self.get_peft_config_fn({
            "method_name": "lora_1",
            "lora_bottleneck": 4,
        })

        vit_config = {"drop_path_rate": 0.2}
        net_builder = self.net_builder_fn(net_name, peft_config, vit_config)

        model = net_builder(
            num_classes=self.num_classes,
            pretrained=True,
            pretrained_path="",
        ).to(self.device)

        return model

    def fit(self, img_dir, train_ann_file, unlabel_ann_file):
        train_transform = get_train_transform()
        dataset = LabeledDataset(img_dir, train_ann_file, transform=train_transform)

        # Fix 1: derive class count from the dataset
        self.num_classes = len(dataset.cat_id_to_label)
        self.label_to_cat_id = dataset.label_to_cat_id

        # Fix 2: rebuild model so head matches dataset labels
        self.model = self._build_model()

        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
        )

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        print(f"[fit] trainable params: {sum(p.numel() for p in trainable):,}")
        print(f"[fit] num_classes: {self.num_classes}")

        optimizer = torch.optim.AdamW(trainable, lr=2e-3)
        criterion = nn.CrossEntropyLoss()

        num_epochs = 1000
        self.model.train()
        best_loss = float("inf")
        best_state = None

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            for imgs, labels in loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                out = self.model(imgs)
                logits = out["logits"]

                # Optional safety checks
                assert labels.dtype == torch.long
                assert labels.min().item() >= 0
                assert labels.max().item() < logits.shape[1], (
                    f"Label out of range: min={labels.min().item()}, "
                    f"max={labels.max().item()}, num_classes={logits.shape[1]}"
                )

                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                epoch_loss += loss.item()
            train_acc = correct / total

            avg_loss = epoch_loss / len(loader)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = copy.deepcopy(self.model.state_dict())

            print(f"[fit] epoch {epoch+1:3d}/{num_epochs}  loss={avg_loss:.4f} acc={train_acc:.3f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        print(f"[fit] done — best loss: {best_loss:.4f}")

    def predict(self, image_batch):
        self.model.eval()
        with torch.no_grad():
            if not isinstance(image_batch, torch.Tensor):
                image_batch = torch.stack([self.transform(img) for img in image_batch])
            image_batch = image_batch.to(self.device)
            out = self.model(image_batch)

        logits = out["logits"]
        features = out["feat"]

        probs = F.softmax(logits, dim=-1)
        scores, pred_labels = probs.max(dim=-1)

        # Fix 3: map internal label index back to original COCO category_id
        predictions = [
            {
                "category_id": int(self.label_to_cat_id[int(pred_labels[i].item())]),
                "score": float(scores[i].item()),
            }
            for i in range(len(pred_labels))
        ]

        return logits, features, predictions