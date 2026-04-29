# import json
# import os

# from PIL import Image
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Dataset
# import torchvision.transforms as T


# def _coco_label_mapping(train_ann_file: str):
#     with open(train_ann_file, "r") as f:
#         coco = json.load(f)

#     cat_ids = sorted(c["id"] for c in coco["categories"])
#     cat_id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
#     idx_to_cat_id = {i: cid for cid, i in cat_id_to_idx.items()}

#     return cat_id_to_idx, idx_to_cat_id, len(cat_ids)


# def _labeled_samples(img_dir: str, train_ann_file: str, cat_id_to_idx: dict):
#     with open(train_ann_file, "r") as f:
#         coco = json.load(f)

#     id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}

#     samples = []
#     for ann in coco["annotations"]:
#         fn = id_to_name.get(ann["image_id"])
#         if fn is None:
#             continue

#         path = os.path.join(img_dir, fn)
#         if not os.path.isfile(path):
#             continue

#         cid = ann["category_id"]
#         if cid not in cat_id_to_idx:
#             continue

#         samples.append((path, cat_id_to_idx[cid]))

#     return samples


# class _LabeledDataset(Dataset):
#     def __init__(self, samples, transform):
#         self.samples = samples
#         self.transform = transform

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         path, y = self.samples[idx]
#         img = Image.open(path).convert("RGB")
#         return self.transform(img), y


# class ClassificationAgent:
#     def __init__(
#         self,
#         net_builder_fn,
#         get_peft_config_fn,
#         num_classes=45,
#         train_epochs=100,
#         batch_size=32,
#         lr=2e-4,
#     ):
#         self.train_epochs = train_epochs
#         self.batch_size = batch_size
#         self.lr = lr
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         self.transform = None
#         self.idx_to_cat_id = None

#         self.net_name = "timm/vit_base_patch16_clip_224.openai"

#         self.peft_config = get_peft_config_fn(
#             {
#                 "method_name": "lora",
#                 "freeze_backbone": False,  # full finetune
#                 "lora_bottleneck": 8,
#             }
#         )

#         self.vit_config = {"drop_path_rate": 0}

#         net_builder = net_builder_fn(
#             self.net_name,
#             peft_config=self.peft_config,
#             vit_config=self.vit_config,
#         )

#         self.num_classes = num_classes
#         self.model = net_builder(
#             num_classes=self.num_classes,
#             pretrained=True,
#             pretrained_path="",
#         ).to(self.device)


#     def _normalize(self):       
#         return T.Normalize(
#                     mean=[0.48145466, 0.4578275, 0.40821073],
#                     std=[0.26862954, 0.26130258, 0.27577711],
#         )

#     def _train_transform(self):
#         return T.Compose(
#             [
#                 T.RandomResizedCrop(224, scale=(0.7, 1.0)),
#                 T.RandomHorizontalFlip(),
#                 T.ToTensor(),
#                 self._normalize(),
#             ]
#         )

#     def _eval_transform(self):
#         return T.Compose(
#             [
#                 T.Resize(256),
#                 T.CenterCrop(224),
#                 T.ToTensor(),
#                 self._normalize(),
#             ]
#         )

#     def fit(self, img_dir, train_ann_file, unlabel_ann_file):

#         cat_id_to_idx, idx_to_cat_id, num_classes = _coco_label_mapping(train_ann_file)
#         self.idx_to_cat_id = idx_to_cat_id
#         self.num_classes = num_classes

#         train_tf = self._train_transform()
#         self.transform = self._eval_transform()

#         samples = _labeled_samples(img_dir, train_ann_file, cat_id_to_idx)

#         if len(samples) == 0:
#             raise RuntimeError("No labeled samples found.")

#         ds = _LabeledDataset(samples, train_tf)

#         loader = DataLoader(
#             ds,
#             batch_size=min(self.batch_size, len(ds)),
#             shuffle=True,
#             num_workers=4,
#             pin_memory=True,
#         )

#         params = [p for p in self.model.parameters() if p.requires_grad]
#         opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=0.05)

#         self.model.train()

#         for epoch in range(self.train_epochs):
#             total_loss = 0

#             for x, y in loader:
#                 x = x.to(self.device)
#                 y = y.to(self.device)

#                 opt.zero_grad()

#                 out = self.model(x)
#                 loss = F.cross_entropy(out["logits"], y)

#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(params, 1.0)

#                 opt.step()

#                 total_loss += loss.item()

#             print(f"Epoch {epoch+1}: loss = {total_loss / len(loader):.4f}")

#         self.model.eval()

#     @torch.no_grad()
#     def predict(self, image_batch):
#         self.model.eval()

#         if isinstance(image_batch, torch.Tensor):
#             x = image_batch.to(self.device)
#         else:
#             x = torch.stack(
#                 [self.transform(im.convert("RGB")) for im in image_batch]
#             ).to(self.device)

#         out = self.model(x)

#         logits = out["logits"]
#         feat = out["feat"]

#         pred_idx = logits.argmax(dim=-1).cpu().tolist()

#         predictions = [
#             {"category_id": int(self.idx_to_cat_id[i])} for i in pred_idx
#         ]
#         return logits, feat, predictions

# """
# ClassificationAgent — simple supervised fine-tuning.

# Strategy:
#   - Backbone: DINOv2 (strong pretrained features)
#   - Fine-tune LoRA + linear head on the 90 labeled images
#   - Heavy augmentation to combat the tiny training set
#   - Predict with a single forward pass
# """

# import os
# import json

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image


# # ---------------------------------------------------------------------------
# # Dataset
# # ---------------------------------------------------------------------------

# class LabeledDataset(Dataset):
#     def __init__(self, img_dir, file_names, labels, transform=None):
#         self.img_dir = img_dir
#         self.file_names = file_names
#         self.labels = labels
#         self.transform = transform

#     def __len__(self):
#         return len(self.file_names)

#     def __getitem__(self, idx):
#         path = os.path.join(self.img_dir, self.file_names[idx])
#         img = Image.open(path).convert("RGB")
#         if self.transform:
#             img = self.transform(img)
#         return img, self.labels[idx]


# # ---------------------------------------------------------------------------
# # Agent
# # ---------------------------------------------------------------------------

# class ClassificationAgent:

#     def __init__(self, net_builder_fn, get_peft_config_fn=None, num_classes=45):
#         if get_peft_config_fn is None:
#             get_peft_config_fn = lambda cfg: cfg

#         self.num_classes = num_classes
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # --- Build model ---
#         net_name = "timm/vit_base_patch16_clip_224.openai"

#         peft_config = get_peft_config_fn({
#             "method_name": "lora_1",
#             "lora_bottleneck": 16,
#         })

#         vit_config = {"drop_path_rate": 0.0}

#         net_builder = net_builder_fn(net_name, peft_config, vit_config)
#         self.model = net_builder(
#             num_classes=num_classes,
#             pretrained=True,
#             pretrained_path=""
#         ).to(self.device)

#         # --- Transforms ---
#         mean = [0.48145466, 0.4578275, 0.40821073]
#         std  = [0.26862954, 0.26130258, 0.27577711]

#         self.transform = transforms.Compose([
#             transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std),
#         ])

#         self._train_transform = transforms.Compose([
#             transforms.RandomResizedCrop(224, scale=(0.4, 1.0),
#                                          interpolation=transforms.InterpolationMode.BICUBIC),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomVerticalFlip(),
#             transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
#             transforms.RandomGrayscale(p=0.1),
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std),
#         ])

#     # ------------------------------------------------------------------

#     def fit(self, img_dir, train_ann_file, unlabel_ann_file):
#         # --- Parse annotations ---
#         with open(train_ann_file, "r") as f:
#             train_ann = json.load(f)

#         id2fn = {im["id"]: im["file_name"] for im in train_ann["images"]}
#         cat_ids = sorted({c["id"] for c in train_ann["categories"]})
#         cat_id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
#         self.num_classes = len(cat_ids)

#         file_names, labels = [], []
#         for ann in train_ann["annotations"]:
#             file_names.append(id2fn[ann["image_id"]])
#             labels.append(cat_id_to_idx[ann["category_id"]])

#         print(f"[fit] Training on {len(file_names)} labeled images, {self.num_classes} classes")

#         # --- Dataset / loader ---
#         dataset = LabeledDataset(img_dir, file_names, labels, transform=self._train_transform)
#         loader  = DataLoader(dataset, batch_size=32, shuffle=True,
#                              num_workers=4, pin_memory=True, drop_last=False)
        
#         trainable = [p for p in self.model.parameters() if p.requires_grad]
#         print(f"[fit] Trainable params: {sum(p.numel() for p in trainable):,}")

#         optimizer = torch.optim.AdamW(trainable, lr=3e-3, weight_decay=0.05)
#         num_epochs = 100
#         # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
#         criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

#         self.model.train()
#         for epoch in range(num_epochs):
#             total_loss, correct, total = 0.0, 0, 0
#             for imgs, lbls in loader:
#                 imgs, lbls = imgs.to(self.device), lbls.to(self.device)
#                 out = self.model(imgs)
#                 logits = out["logits"] if isinstance(out, dict) else out
#                 loss = criterion(logits, lbls)

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#                 total_loss += loss.item() * len(lbls)
#                 correct += (logits.argmax(1) == lbls).sum().item()
#                 total += len(lbls)

            
#             print(f"Epoch {epoch+1}/{num_epochs} | loss={total_loss/total:.4f} | acc={correct/total:.3f}")

#         self.model.eval()
#         print("[fit] Done.")

#     # ------------------------------------------------------------------

#     @torch.no_grad()
#     def predict(self, image_batch):
#         """
#         Args:
#             image_batch: torch.Tensor of shape (B, C, H, W), already transformed.
#         Returns:
#             logits:      torch.Tensor (B, num_classes)
#             features:    torch.Tensor (B, D)
#             predictions: list of dicts with 'label' and 'score'
#         """
#         self.model.eval()
#         image_batch = image_batch.to(self.device)
#         out = self.model(image_batch)
#         logits   = out["logits"] if isinstance(out, dict) else out
#         features = out["feat"]   if isinstance(out, dict) else None

#         probs = torch.softmax(logits, dim=-1)
#         pred_labels = logits.argmax(dim=-1).tolist()
#         pred_scores = probs.max(dim=-1).values.tolist()

#         predictions = [
#             {"category_id": lbl, "score": score}
#             for lbl, score in zip(pred_labels, pred_scores)
#         ]

#         return logits, features, predictions

"""
Baseline Classification Agent
-------------------------------
Strategy:
1. Use DINOv2 backbone (strong visual features out-of-the-box)
2. Fine-tune with LoRA adapters on the 90 labeled images
3. Use heavy augmentation to combat the tiny training set
4. Apply label smoothing + class-balanced sampling
5. No pseudo-labeling in this baseline — keep it simple and stable
"""

import json
import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


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
# Training transforms (heavy augmentation for tiny labeled set)
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
    def __init__(self, net_builder_fn, get_peft_config_fn, num_classes=45):
        self.num_classes = num_classes
        self.net_builder_fn = net_builder_fn
        self.get_peft_config_fn = get_peft_config_fn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Inference transform (used by the evaluation harness)
        self.transform = get_val_transform()

        # Build model
        self.model = self._build_model()

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(self):
        net_name = "timm/vit_base_patch14_reg4_dinov2.lvd142m"  # DINOv2

        peft_config = self.get_peft_config_fn({
            "method_name": "lora_1",
            "lora_bottleneck": 16,
        })

        vit_config = {"drop_path_rate": 0.1}

        net_builder = self.net_builder_fn(net_name, peft_config, vit_config)

        model = net_builder(
            num_classes=self.num_classes,
            pretrained=True,
            pretrained_path="",
        ).to(self.device)

        return model

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, img_dir, train_ann_file, unlabel_ann_file):
        """Fine-tune on labeled data with augmentation."""

        train_transform = get_train_transform()
        dataset = LabeledDataset(img_dir, train_ann_file, transform=train_transform)

        # With only 90 samples we can afford a small batch and many epochs
        loader = DataLoader(
            dataset,
            batch_size=10,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
        )

        # Only update parameters marked as trainable by the builder
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        print(f"[fit] trainable params: {sum(p.numel() for p in trainable):,}")

        optimizer = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=0.05)

        num_epochs = 2
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6
        )

        # Label-smoothing cross-entropy
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

            if (epoch + 1) % 10 == 0:
                print(f"[fit] epoch {epoch+1:3d}/{num_epochs}  loss={avg_loss:.4f}")

        # Restore best checkpoint
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        print(f"[fit] done — best loss: {best_loss:.4f}")

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, image_batch):
        """
        Args:
            image_batch: torch.Tensor of shape (B, C, H, W), already transformed.
        Returns:
            logits     : torch.Tensor (B, num_classes)
            features   : torch.Tensor (B, 768)
            predictions: list[dict]  each {"label": int, "score": float}
        """
        self.model.eval()
        with torch.no_grad():
            if not isinstance(image_batch, torch.Tensor):
                image_batch = torch.stack(
                    [self.transform(img) for img in image_batch]
                )
            image_batch = image_batch.to(self.device)
            out = self.model(image_batch)

        logits = out["logits"]       # (B, num_classes)
        features = out["feat"]       # (B, 768)

        probs = F.softmax(logits, dim=-1)
        scores, pred_labels = probs.max(dim=-1)

        predictions = [
            {"category_id": int(pred_labels[i].item()), "score": float(scores[i].item())}
            for i in range(len(pred_labels))
        ]

        return logits, features, predictions