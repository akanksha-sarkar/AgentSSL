
"""
Baseline Classification Agent
-------------------------------
Strategy:
1. ResNet-18 (torchvision) — smaller and simpler than a ViT + PEFT stack
2. ImageNet-pretrained weights, replace the final FC for num_classes
3. Full fine-tune (all conv + head parameters) with data augmentation
4. Label smoothing + light regularization; no unlabeled / pseudo-labeling
"""

import json
import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

try:
    from torchvision.models import ResNet18_Weights
except Exception:  # older torchvision
    ResNet18_Weights = None


# ---------------------------------------------------------------------------
# Small CNN wrapper: same output contract as the ViT (dict with logits, feat)
# ---------------------------------------------------------------------------

class ResNetClassifier(nn.Module):
    """ResNet-18 with features before the final linear, dict forward."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        if ResNet18_Weights is not None and pretrained:
            w = ResNet18_Weights.IMAGENET1K_V1
            backbone = models.resnet18(weights=w)
        else:
            backbone = models.resnet18(pretrained=pretrained)
        in_f = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Linear(in_f, num_classes)
        self.feat_dim = in_f

    def forward(self, x: torch.Tensor):
        # backbone ends with global pool + identity -> (B, 512)
        z = self.backbone(x)
        if z.dim() > 2:
            z = z.flatten(1)
        logits = self.head(z)
        return {"logits": logits, "feat": z}


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

class LabeledDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        with open(ann_file, "r") as f:
            coco = json.load(f)

        self.img_dir = img_dir
        self.transform = transform

        # Build id -> filename map
        id_to_fname = {img["id"]: img["file_name"] for img in coco["images"]}

        # Build category id -> contiguous label
        cat_ids = sorted(cat["id"] for cat in coco["categories"])
        self.cat_id_to_label = {cid: i for i, cid in enumerate(cat_ids)}

        # Build samples list
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


# ---------------------------------------------------------------------------
# Training transforms
# ---------------------------------------------------------------------------

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
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Classification Agent
# ---------------------------------------------------------------------------

class ClassificationAgent:
    def __init__(self, num_classes: int = 45):
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Inference transform (used by the evaluation harness)
        self.transform = get_val_transform()

        # Build model
        self.model = self._build_model()

    def _build_model(self):
        return ResNetClassifier(
            num_classes=self.num_classes,
            pretrained=True,
        ).to(self.device)

    def fit(self, img_dir, train_ann_file, unlabel_ann_file):
        """Fine-tune on labeled data with augmentation (unlabel_ann_file ignored)."""
        _ = unlabel_ann_file  # not used in this baseline

        train_transform = get_train_transform()
        dataset = LabeledDataset(img_dir, train_ann_file, transform=train_transform)

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

        optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0.01)

        num_epochs = 5
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.model.train()
        best_loss = float("inf")
        best_state = None

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for imgs, labels in loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                out = self.model(imgs)
                loss = criterion(out["logits"], labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            avg_loss = epoch_loss / len(loader)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = copy.deepcopy(self.model.state_dict())

            if (epoch + 1) % 2 == 0 or (epoch + 1) == num_epochs:
                print(
                    f"[fit] epoch {epoch + 1:3d}/{num_epochs}  loss={avg_loss:.4f}"
                )

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        print(f"[fit] done — best loss: {best_loss:.4f}")

    def predict(self, image_batch):
        """
        Args:
            image_batch: torch.Tensor of shape (B, C, H, W), already transformed.
        Returns:
            logits:       (B, num_classes)
            features:     (B, 512)  (ResNet-18 pooled dim)
            predictions:  list[dict] with "category_id" and "score"
        """
        self.model.eval()
        with torch.no_grad():
            if not isinstance(image_batch, torch.Tensor):
                image_batch = torch.stack(
                    [self.transform(img) for img in image_batch]
                )
            image_batch = image_batch.to(self.device)
            out = self.model(image_batch)

        logits = out["logits"]
        features = out["feat"]

        probs = F.softmax(logits, dim=-1)
        scores, pred_labels = probs.max(dim=-1)

        predictions = [
            {"category_id": int(pred_labels[i].item()), "score": float(scores[i].item())}
            for i in range(len(pred_labels))
        ]

        return logits, features, predictions
