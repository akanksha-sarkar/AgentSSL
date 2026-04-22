import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import math


class LabeledDataset(Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


def mixup_data(x, y, alpha=0.4):
    """Returns mixed inputs, pairs of targets, and lambda."""
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def _extract_logits(out):
    """TimmViTWrapper returns dict(feat=..., logits=...); raw timm may return a tensor or tuple."""
    if isinstance(out, dict):
        if "logits" not in out:
            raise KeyError(f"Expected 'logits' in model output dict, got keys={list(out.keys())}")
        return out["logits"]
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


class ClassificationAgent:
    def __init__(self, net_builder_fn, get_peft_config_fn, num_classes=45):
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # LoRA-style adapter on attention + MLP for lightweight fine-tuning
        peft_config = get_peft_config_fn({
            "freeze_backbone": False,
            "ft_attn_module": "adapter",
            "ft_attn_mode": "parallel",
            "ft_attn_ln": "before",
            "ft_mlp_module": "adapter",
            "ft_mlp_mode": "parallel",
            "ft_mlp_ln": "before",
            "adapter_bottleneck": 32,
            "adapter_init": "lora_kaiming",
            "adapter_scaler": 0.1,
            "lora_bottleneck": 8,
            "ln": True,       # fine-tune layer norms
            "bitfit": True,   # fine-tune biases
        })

        net_name = "timm/vit_base_patch14_reg4_dinov2.lvd142m"
        net_builder = net_builder_fn(net_name, peft_config=peft_config)
        self.model = net_builder(
            num_classes=num_classes,
            pretrained=True,
            pretrained_path="",
        ).to(self.device)

        # Standard DINOv2 inference transform (no augmentation)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # Heavy augmentation transform for training
        self._train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                   saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
        ])

        self.category_id_to_name = {}
        self.category_name_to_idx = {}
        self.idx_to_category_id = {}

    def fit(self, img_dir, train_ann_file, unlabel_ann_file,
            epochs=40, batch_size=16, lr=3e-4, warmup_epochs=4):
        # unlabel_ann_file intentionally ignored
        print("[ClassificationAgent] Loading annotations...")

        with open(train_ann_file, "r") as f:
            train_ann = json.load(f)

        for cat in train_ann["categories"]:
            idx = len(self.category_name_to_idx)
            self.category_id_to_name[cat["id"]] = cat["name"]
            self.category_name_to_idx[cat["name"]] = idx
            self.idx_to_category_id[idx] = cat["id"]

        id_to_file = {img["id"]: img["file_name"] for img in train_ann["images"]}

        image_paths, labels = [], []
        for ann in train_ann["annotations"]:
            cat_name = self.category_id_to_name[ann["category_id"]]
            image_paths.append(os.path.join(img_dir, id_to_file[ann["image_id"]]))
            labels.append(self.category_name_to_idx[cat_name])

        labels_tensor = torch.tensor(labels, dtype=torch.long)
        print(f"[ClassificationAgent] Labeled samples: {len(image_paths)}")

        # Balanced sampler — ensures every class seen each effective epoch
        class_counts = torch.bincount(labels_tensor, minlength=self.num_classes).float()
        sample_weights = 1.0 / class_counts[labels_tensor]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(image_paths), replacement=True)

        dataset = LabeledDataset(image_paths, labels_tensor, self._train_transform)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                            num_workers=4, pin_memory=True, drop_last=True)

        # Only optimize adapter/head parameters + layer norms + biases
        # Frozen base weights stay frozen via requires_grad=False set by PEFT
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=0.05
        )

        total_steps = epochs * len(loader)
        warmup_steps = warmup_epochs * len(loader)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        print(f"[ClassificationAgent] Training for {epochs} epochs...")
        self.model.train()

        for epoch in range(epochs):
            total_loss, correct, total = 0.0, 0, 0

            for imgs, lbls in loader:
                imgs, lbls = imgs.to(self.device), lbls.to(self.device)

                # Mixup augmentation
                imgs, lbls_a, lbls_b, lam = mixup_data(imgs, lbls, alpha=0.4)

                optimizer.zero_grad()
                logits = _extract_logits(self.model(imgs))

                loss = mixup_criterion(criterion, logits, lbls_a, lbls_b, lam)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, self.model.parameters()), 1.0
                )
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                # Accuracy against the dominant mixup label
                correct += (lam * (preds == lbls_a).float() +
                            (1 - lam) * (preds == lbls_b).float()).sum().item()
                total += imgs.size(0)

            if (epoch + 1) % 20 == 0:
                acc = 100.0 * correct / total
                avg_loss = total_loss / len(loader)
                cur_lr = scheduler.get_last_lr()[0]
                print(f"  Epoch [{epoch+1:>3}/{epochs}]  "
                      f"loss: {avg_loss:.4f}  acc: {acc:.1f}%  lr: {cur_lr:.2e}")

        self.model.eval()
        print("[ClassificationAgent] Training complete.")

    @torch.no_grad()
    def predict(self, image_batch):
        if image_batch.ndim == 3:
            image_batch = image_batch.unsqueeze(0)
        image_batch = image_batch.to(self.device)

        logits = _extract_logits(self.model(image_batch))

        # Extract backbone features for the metrics that need them
        feat = self._get_backbone_features(image_batch)

        pred_scores = logits.softmax(dim=-1)
        pred_labels = logits.argmax(dim=1)

        predictions = []
        for b in range(image_batch.shape[0]):
            cls_idx = pred_labels[b].item()
            cat_id = self.idx_to_category_id.get(cls_idx, cls_idx)
            predictions.append({
                "category_id": cat_id,
                "category_name": self.category_id_to_name.get(cat_id, str(cat_id)),
                "score": pred_scores[b, cls_idx].item(),
            })

        return logits.cpu(), feat.cpu(), predictions

    @torch.no_grad()
    def _get_backbone_features(self, imgs):
        if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'forward_features'):
            feat = self.model.backbone.forward_features(imgs)
        elif hasattr(self.model, 'forward_features'):
            feat = self.model.forward_features(imgs)
        else:
            features = {}
            hook = None
            for name, module in reversed(list(self.model.named_modules())):
                if isinstance(module, nn.LayerNorm) and 'head' not in name:
                    hook = module.register_forward_hook(
                        lambda m, inp, out: features.update({'feat': out})
                    )
                    break
            self.model(imgs)
            if hook:
                hook.remove()
            feat = features.get('feat', self.model(imgs))

        if isinstance(feat, dict):
            feat = feat.get('x_norm_clstoken', next(iter(feat.values())))
        elif isinstance(feat, (tuple, list)):
            feat = feat[0]
        if feat.ndim == 3:
            feat = feat[:, 0]
        return F.normalize(feat, dim=-1)