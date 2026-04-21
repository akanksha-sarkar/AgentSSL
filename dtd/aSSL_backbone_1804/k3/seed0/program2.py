from __future__ import annotations

import copy
import json
import math
import os
from collections import deque

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def _build_label_mappings(train_ann_file: str):
    coco = _read_json(train_ann_file)
    cat_ids = sorted([c["id"] for c in coco["categories"]])
    cat_id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
    idx_to_cat_id = {i: cid for cid, i in cat_id_to_idx.items()}
    return cat_id_to_idx, idx_to_cat_id, len(cat_ids)


def _load_labeled_samples(img_dir: str, train_ann_file: str, cat_id_to_idx: dict):
    coco = _read_json(train_ann_file)
    id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}
    samples = []
    for ann in coco["annotations"]:
        fn = id_to_name.get(ann["image_id"], None)
        if fn is None:
            continue
        path = os.path.join(img_dir, fn)
        if not os.path.isfile(path):
            continue
        cid = ann["category_id"]
        if cid not in cat_id_to_idx:
            continue
        samples.append((path, cat_id_to_idx[cid]))
    return samples


def _load_unlabeled_samples(img_dir: str, unlabel_ann_file: str):
    coco = _read_json(unlabel_ann_file)
    samples = []
    for img in coco["images"]:
        fn = img["file_name"]
        path = os.path.join(img_dir, fn)
        if os.path.isfile(path):
            samples.append(path)
    return samples


class _LabeledDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, y


class _UnlabeledPairDataset(Dataset):
    def __init__(self, samples, weak_transform, strong_transform):
        self.samples = samples
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        img = Image.open(path).convert("RGB")
        xw = self.weak_transform(img)
        xs = self.strong_transform(img)
        return xw, xs


class _SoftTargetCrossEntropy(nn.Module):
    def forward(self, logits, target_probs):
        logp = F.log_softmax(logits, dim=-1)
        return -(target_probs * logp).sum(dim=-1).mean()


class ClassificationAgent:
    def __init__(self, net_builder_fn, get_peft_config_fn, num_classes):
        self.net_builder_fn = net_builder_fn
        self.get_peft_config_fn = get_peft_config_fn
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.idx_to_cat_id = {i: i for i in range(num_classes)}
        self.transform = self._build_eval_transform()

        net_name = "timm/vit_base_patch14_reg4_dinov2.lvd142m"
        peft_config = self.get_peft_config_fn(
            {
                "method_name": "adaptformer",
                "ft_mlp_module": "adapter",
                "ft_mlp_mode": "parallel",
                "ft_mlp_ln": "before",
                "adapter_init": "lora_kaiming",
                "adapter_bottleneck": 16,
                "adapter_scaler": 0.1,
                "freeze_backbone": False,
            }
        )
        vit_config = {"drop_path_rate": 0.0}
        net_builder = self.net_builder_fn(net_name, peft_config, vit_config)
        self.model = net_builder(
            num_classes=num_classes,
            pretrained=True,
            pretrained_path="",
        ).to(self.device)
        self.ema_model = copy.deepcopy(self.model).to(self.device)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

        self.sup_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.unsup_criterion = _SoftTargetCrossEntropy()

    def _build_eval_transform(self, img_size: int = 224):
        resize = int(img_size * 256 / 224)
        return T.Compose(
            [
                T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(img_size),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def _build_weak_transform(self, img_size: int = 224):
        return T.Compose(
            [
                T.RandomResizedCrop(
                    img_size,
                    scale=(0.7, 1.0),
                    interpolation=T.InterpolationMode.BICUBIC,
                ),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def _build_strong_transform(self, img_size: int = 224):
        return T.Compose(
            [
                T.RandomResizedCrop(
                    img_size,
                    scale=(0.5, 1.0),
                    interpolation=T.InterpolationMode.BICUBIC,
                ),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
                T.RandomAutocontrast(p=0.2),
                T.RandomEqualize(p=0.1),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    @torch.no_grad()
    def _update_ema(self, decay=0.999):
        msd = self.model.state_dict()
        esd = self.ema_model.state_dict()
        for k in esd.keys():
            if esd[k].dtype.is_floating_point:
                esd[k].mul_(decay).add_(msd[k], alpha=1.0 - decay)
            else:
                esd[k].copy_(msd[k])

    def fit(self, img_dir, train_ann_file, unlabel_ann_file):
        cat_id_to_idx, idx_to_cat_id, num_classes = _build_label_mappings(
            train_ann_file
        )
        self.idx_to_cat_id = idx_to_cat_id
        self.num_classes = num_classes

        labeled_samples = _load_labeled_samples(img_dir, train_ann_file, cat_id_to_idx)
        unlabeled_samples = _load_unlabeled_samples(img_dir, unlabel_ann_file)

        if len(labeled_samples) == 0:
            raise RuntimeError("No labeled samples found.")
        if len(unlabeled_samples) == 0:
            raise RuntimeError("No unlabeled samples found.")

        weak_tf = self._build_weak_transform()
        strong_tf = self._build_strong_transform()
        self.transform = self._build_eval_transform()

        labeled_ds = _LabeledDataset(labeled_samples, weak_tf)
        unlabeled_ds = _UnlabeledPairDataset(unlabeled_samples, weak_tf, strong_tf)

        labeled_bs = min(32, len(labeled_ds))
        unlabeled_bs = 96 if len(unlabeled_ds) >= 96 else max(16, len(unlabeled_ds))

        labeled_loader = DataLoader(
            labeled_ds,
            batch_size=labeled_bs,
            shuffle=True,
            num_workers=min(8, os.cpu_count() or 1),
            pin_memory=self.device.type == "cuda",
            drop_last=True if len(labeled_ds) >= labeled_bs else False,
        )
        unlabeled_loader = DataLoader(
            unlabeled_ds,
            batch_size=unlabeled_bs,
            shuffle=True,
            num_workers=min(8, os.cpu_count() or 1),
            pin_memory=self.device.type == "cuda",
            drop_last=True if len(unlabeled_ds) >= unlabeled_bs else False,
        )

        class_counts = torch.zeros(num_classes, dtype=torch.float32)
        for _, y in labeled_samples:
            class_counts[y] += 1
        target_prior = (class_counts / class_counts.sum()).to(self.device)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=0.05)

        total_epochs = 20
        warmup_epochs = 5
        conf_thresh = 0.90
        lambda_u_max = 2.0
        entropy_weight = 0.05
        dist_hist = deque(maxlen=128)

        self.model.train()
        self.ema_model.load_state_dict(copy.deepcopy(self.model.state_dict()))

        # Stage 1: supervised warmup
        for epoch in range(warmup_epochs):
            self.model.train()
            total_loss = 0.0
            total_correct = 0
            total_count = 0

            for x, y in labeled_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                out = self.model(x)
                logits = out["logits"]
                loss = self.sup_criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                self._update_ema(decay=0.995)

                total_loss += loss.item() * y.size(0)
                total_correct += (logits.argmax(dim=-1) == y).sum().item()
                total_count += y.size(0)

            print(
                f"[warmup {epoch+1}/{warmup_epochs}] "
                f"loss={total_loss/max(total_count,1):.4f} "
                f"acc={total_correct/max(total_count,1):.4f}"
            )

        # Stage 2: semi-supervised consistency learning
        unlabeled_iter = iter(unlabeled_loader)
        steps_per_epoch = max(len(labeled_loader), 1)

        for epoch in range(warmup_epochs, total_epochs):
            self.model.train()
            total_sup = 0.0
            total_unsup = 0.0
            total_ent = 0.0
            total_acc = 0
            total_lab = 0
            total_mask = 0.0
            total_u = 0

            lambda_u = lambda_u_max * min(
                1.0,
                (epoch - warmup_epochs + 1)
                / max(1, (total_epochs - warmup_epochs) * 0.4),
            )

            for x_l, y_l in labeled_loader:
                try:
                    x_uw, x_us = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    x_uw, x_us = next(unlabeled_iter)

                x_l = x_l.to(self.device, non_blocking=True)
                y_l = y_l.to(self.device, non_blocking=True)
                x_uw = x_uw.to(self.device, non_blocking=True)
                x_us = x_us.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                out_l = self.model(x_l)
                logits_l = out_l["logits"]
                sup_loss = self.sup_criterion(logits_l, y_l)

                with torch.no_grad():
                    teacher_out = self.ema_model(x_uw)
                    logits_uw = teacher_out["logits"]
                    probs_uw = F.softmax(logits_uw, dim=-1)

                    batch_mean = probs_uw.mean(dim=0)
                    dist_hist.append(batch_mean.detach())
                    running_model_dist = torch.stack(list(dist_hist), dim=0).mean(dim=0)

                    aligned = probs_uw * (
                        target_prior / running_model_dist.clamp_min(1e-6)
                    ).unsqueeze(0)
                    aligned = aligned / aligned.sum(dim=-1, keepdim=True).clamp_min(
                        1e-6
                    )

                    conf, _ = aligned.max(dim=-1)
                    mask = (conf >= conf_thresh).float()

                out_us = self.model(x_us)
                logits_us = out_us["logits"]

                if mask.sum() > 0:
                    per_sample_unsup = -(
                        aligned * F.log_softmax(logits_us, dim=-1)
                    ).sum(dim=-1)
                    unsup_loss = (per_sample_unsup * mask).sum() / mask.sum().clamp_min(
                        1.0
                    )

                    probs_us = F.softmax(logits_us, dim=-1)
                    ent = -(probs_us * torch.log(probs_us.clamp_min(1e-8))).sum(dim=-1)
                    ent_loss = (ent * mask).sum() / mask.sum().clamp_min(1.0)
                else:
                    unsup_loss = logits_us.sum() * 0.0
                    ent_loss = logits_us.sum() * 0.0

                loss = sup_loss + lambda_u * unsup_loss + entropy_weight * ent_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                self._update_ema(decay=0.999)

                total_sup += sup_loss.item() * y_l.size(0)
                total_unsup += unsup_loss.item() * x_uw.size(0)
                total_ent += ent_loss.item() * x_uw.size(0)
                total_acc += (logits_l.argmax(dim=-1) == y_l).sum().item()
                total_lab += y_l.size(0)
                total_mask += mask.sum().item()
                total_u += mask.numel()

            print(
                f"[ssl {epoch+1}/{total_epochs}] "
                f"sup={total_sup/max(total_lab,1):.4f} "
                f"unsup={total_unsup/max(total_u,1):.4f} "
                f"ent={total_ent/max(total_u,1):.4f} "
                f"acc={total_acc/max(total_lab,1):.4f} "
                f"mask_rate={total_mask/max(total_u,1):.4f} "
                f"lambda_u={lambda_u:.3f}"
            )

        self.model.load_state_dict(self.ema_model.state_dict())
        self.model.eval()

    @torch.no_grad()
    def predict(self, image_batch):
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")

        self.model.eval()

        if isinstance(image_batch, torch.Tensor):
            x = image_batch.to(self.device, non_blocking=True).float()
            if x.ndim == 3:
                x = x.unsqueeze(0)
        elif isinstance(image_batch, list):
            x = torch.stack(
                [self.transform(im.convert("RGB")) for im in image_batch], dim=0
            ).to(self.device, non_blocking=True)
        else:
            raise TypeError("image_batch must be a torch.Tensor or list of PIL images")

        out = self.model(x)
        logits = out["logits"]
        feat = out["feat"]
        pred_idx = logits.argmax(dim=-1).detach().cpu().tolist()
        predictions = [{"category_id": int(self.idx_to_cat_id[i])} for i in pred_idx]
        return logits, feat, predictions