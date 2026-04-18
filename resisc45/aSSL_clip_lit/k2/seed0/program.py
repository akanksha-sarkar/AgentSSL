import os
import json
import copy
import random
from itertools import cycle

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import timm


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def find_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def ensure_rgb(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def default_num_workers():
    try:
        return min(4, max(0, os.cpu_count() or 0))
    except Exception:
        return 0


def pil_list_collate(batch):
    return batch


class CocoLabeledDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None, cat_id_to_idx=None):
        data = read_json(ann_file)
        self.img_dir = img_dir
        self.transform = transform

        images = {x["id"]: x["file_name"] for x in data["images"]}
        anns = data["annotations"]
        categories = data["categories"]

        if cat_id_to_idx is None:
            cat_ids = sorted([c["id"] for c in categories])
            self.cat_id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
        else:
            self.cat_id_to_idx = cat_id_to_idx

        self.idx_to_cat_id = {v: k for k, v in self.cat_id_to_idx.items()}

        self.samples = []
        for ann in anns:
            image_id = ann["image_id"]
            cat_id = ann["category_id"]
            self.samples.append(
                {
                    "image_id": image_id,
                    "file_name": images[image_id],
                    "label": self.cat_id_to_idx[cat_id],
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        path = os.path.join(self.img_dir, s["file_name"])
        img = ensure_rgb(Image.open(path))
        x = self.transform(img) if self.transform is not None else img
        y = s["label"]
        return x, y


class CocoUnlabeledDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None, return_meta=False):
        data = read_json(ann_file)
        self.img_dir = img_dir
        self.transform = transform
        self.return_meta = return_meta
        self.samples = [
            {"image_id": x["id"], "file_name": x["file_name"]} for x in data["images"]
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        path = os.path.join(self.img_dir, s["file_name"])
        img = ensure_rgb(Image.open(path))
        x = self.transform(img) if self.transform is not None else img
        if self.return_meta:
            return x, s["image_id"], s["file_name"]
        return x


class PseudoLabeledDataset(Dataset):
    def __init__(self, img_dir, pseudo_records, transform=None):
        self.img_dir = img_dir
        self.samples = pseudo_records
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        path = os.path.join(self.img_dir, s["file_name"])
        img = ensure_rgb(Image.open(path))
        x = self.transform(img) if self.transform is not None else img
        y = int(s["label"])
        w = float(s.get("weight", 1.0))
        return x, y, w


class CocoImageDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        data = read_json(ann_file)
        self.img_dir = img_dir
        self.transform = transform
        self.images = data["images"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        rec = self.images[idx]
        path = os.path.join(self.img_dir, rec["file_name"])
        img = ensure_rgb(Image.open(path))
        x = self.transform(img) if self.transform is not None else img
        return x, rec["id"], rec["file_name"]


class ViTClassifier(nn.Module):
    def __init__(
        self, model_name="vit_base_patch16_224", num_classes=45, pretrained=True
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="token"
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Linear(feat_dim, num_classes)
        self.num_features = feat_dim

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits, feats

    def forward_head(self, feats):
        return self.head(feats)


def freeze_vit_partial(model: ViTClassifier, train_last_blocks: int = 4):
    for p in model.backbone.parameters():
        p.requires_grad = False

    if hasattr(model.backbone, "blocks"):
        blocks = model.backbone.blocks
        for b in blocks[-train_last_blocks:]:
            for p in b.parameters():
                p.requires_grad = True

    if hasattr(model.backbone, "norm"):
        for p in model.backbone.norm.parameters():
            p.requires_grad = True

    for p in model.head.parameters():
        p.requires_grad = True


class BalancedSoftmaxCE(nn.Module):
    def __init__(self, class_counts, label_smoothing=0.0):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32).clamp_min(1.0)
        self.register_buffer("log_counts", counts.log())
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, target, sample_weights=None):
        log_counts = self.log_counts.to(device=logits.device, dtype=logits.dtype)
        adjusted_logits = logits + log_counts.view(1, -1)

        if self.label_smoothing > 0.0:
            num_classes = adjusted_logits.size(1)
            with torch.no_grad():
                true_dist = torch.full_like(
                    adjusted_logits, self.label_smoothing / max(1, (num_classes - 1))
                )
                true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)
            log_probs = F.log_softmax(adjusted_logits, dim=1)
            loss = -(true_dist * log_probs).sum(dim=1)
        else:
            loss = F.cross_entropy(adjusted_logits, target, reduction="none")

        if sample_weights is not None:
            sw = sample_weights.to(device=loss.device, dtype=loss.dtype)
            return (loss * sw).sum() / sw.sum().clamp_min(1.0)
        return loss.mean()


class BalancedFocalSoftmaxCE(nn.Module):
    def __init__(self, class_counts, gamma=1.5, label_smoothing=0.0):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32).clamp_min(1.0)
        self.register_buffer("log_counts", counts.log())
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, target, sample_weights=None):
        log_counts = self.log_counts.to(device=logits.device, dtype=logits.dtype)
        adjusted_logits = logits + log_counts.view(1, -1)
        log_probs = F.log_softmax(adjusted_logits, dim=1)
        probs = log_probs.exp()

        if self.label_smoothing > 0.0:
            num_classes = adjusted_logits.size(1)
            with torch.no_grad():
                true_dist = torch.full_like(
                    adjusted_logits, self.label_smoothing / max(1, (num_classes - 1))
                )
                true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)
            ce = -(true_dist * log_probs).sum(dim=1)
            pt = (true_dist * probs).sum(dim=1).clamp_min(1e-6)
        else:
            ce = F.nll_loss(log_probs, target, reduction="none")
            pt = probs.gather(1, target.unsqueeze(1)).squeeze(1).clamp_min(1e-6)

        focal = (1.0 - pt).pow(self.gamma)
        loss = focal * ce

        if sample_weights is not None:
            sw = sample_weights.to(device=loss.device, dtype=loss.dtype)
            return (loss * sw).sum() / sw.sum().clamp_min(1.0)
        return loss.mean()


class LabelGenerator:
    def __init__(self, num_classes=45, ema_decay=0.999, alpha=0.7, proto_temp=0.07):
        self.teacher_model = None
        self.ema_decay = ema_decay
        self.alpha = alpha
        self.proto_temp = proto_temp
        self.num_classes = num_classes
        self.prototypes = None

    def fit(self, model, labeled_loader, device):
        self.teacher_model = copy.deepcopy(model).to(device)
        self.teacher_model.eval()
        self.refresh_prototypes(model, labeled_loader, device)

    @torch.no_grad()
    def refresh_prototypes(self, model, labeled_loader, device):
        model.eval()
        feat_sum = None
        count = None
        for x, y in labeled_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            _, feats = model(x)
            feats = F.normalize(feats, dim=1)

            if feat_sum is None:
                d = feats.size(1)
                feat_sum = torch.zeros(self.num_classes, d, device=device)
                count = torch.zeros(self.num_classes, device=device)

            feat_sum.index_add_(0, y, feats)
            count.index_add_(0, y, torch.ones_like(y, dtype=torch.float))

        count = count.clamp_min(1.0).unsqueeze(1)
        protos = feat_sum / count
        self.prototypes = F.normalize(protos, dim=1)
        model.train()

    @torch.no_grad()
    def generate(self, model, x):
        self.teacher_model.eval()
        logits_t, feats_t = self.teacher_model(x)
        probs_t = torch.softmax(logits_t, dim=1)

        feats_t_n = F.normalize(feats_t, dim=1)
        if self.prototypes is not None:
            sim = feats_t_n @ self.prototypes.t()
            proto_probs = torch.softmax(sim / self.proto_temp, dim=1)
        else:
            proto_probs = probs_t

        combined = self.alpha * probs_t + (1.0 - self.alpha) * proto_probs
        conf, pseudo = combined.max(dim=1)
        top2 = combined.topk(k=2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]

        metadata = {
            "confidence": conf,
            "margin": margin,
            "teacher_probs": probs_t,
            "proto_probs": proto_probs,
            "teacher_features": feats_t_n,
            "combined_probs": combined,
        }
        return pseudo, metadata

    @torch.no_grad()
    def update(self, student_model, step):
        if self.teacher_model is None:
            return
        d = self.ema_decay
        for t_p, s_p in zip(
            self.teacher_model.parameters(), student_model.parameters()
        ):
            t_p.data.mul_(d).add_(s_p.data, alpha=1.0 - d)


class Augmentation:
    def __init__(self, image_size=224):
        self.image_size = image_size
        self.resize_size = int(image_size * 256 / 224)
        self.mean = (0.48145466, 0.4578275, 0.40821073)
        self.std  = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(self.mean, self.std)]
        )

        self.weak = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.7, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

        self.strong = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.5, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

        self.masked_strong = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.55, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.35, 0.35, 0.35, 0.08)], p=0.7
                ),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
                transforms.RandomErasing(
                    p=0.95, scale=(0.10, 0.35), ratio=(0.3, 3.3), value="random"
                ),
                transforms.RandomErasing(
                    p=0.60, scale=(0.04, 0.12), ratio=(0.3, 3.3), value="random"
                ),
            ]
        )

        self.eval_tf = transforms.Compose(
            [
                transforms.Resize(
                    self.resize_size, interpolation=InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

        self.sup = transforms.Compose(
            [
                transforms.Resize(
                    self.resize_size, interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.5
                ),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

        self.tta_weak = transforms.Compose(
            [
                transforms.Resize(
                    self.resize_size, interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

        mean_t = torch.tensor(self.mean).view(1, 3, 1, 1)
        std_t = torch.tensor(self.std).view(1, 3, 1, 1)
        self.registered_mean = mean_t
        self.registered_std = std_t

    def generate_views(self, x):
        weak = torch.stack([self.weak(img) for img in x], dim=0)
        strong = torch.stack([self.strong(img) for img in x], dim=0)
        return {"weak": weak, "strong": strong}

    def generate_masked_consistency_views(self, x):
        teacher = torch.stack([self.weak(img) for img in x], dim=0)
        student = torch.stack([self.masked_strong(img) for img in x], dim=0)
        return {"teacher": teacher, "student": student}

    def generate_supervised_pair(self, x):
        v1 = torch.stack([self.sup(img) for img in x], dim=0)
        v2 = torch.stack([self.sup(img) for img in x], dim=0)
        return v1, v2

    def generate_tta_adapt_views(self, x):
        eval_v = torch.stack([self.eval_tf(img) for img in x], dim=0)
        weak_v = torch.stack([self.tta_weak(img) for img in x], dim=0)
        return {"eval": eval_v, "weak": weak_v}

    def _resize_only(self, img):
        return transforms.functional.resize(
            img, self.resize_size, interpolation=InterpolationMode.BICUBIC
        )

    def _crop_box(self, img, top, left):
        return transforms.functional.crop(
            img, top, left, self.image_size, self.image_size
        )

    def tta_views(self, imgs):
        views = [[] for _ in range(6)]
        margin = self.resize_size - self.image_size

        for img in imgs:
            img_r = self._resize_only(img)
            c = margin // 2
            v0 = self._crop_box(img_r, c, c)
            v1 = transforms.functional.hflip(v0)
            v2 = self._crop_box(img_r, 0, 0)
            v3 = self._crop_box(img_r, 0, margin)
            v4 = self._crop_box(img_r, margin, 0)
            v5 = self._crop_box(img_r, margin, margin)

            raw_views = [v0, v1, v2, v3, v4, v5]
            for k in range(6):
                views[k].append(self.normalize(raw_views[k]))

        return [torch.stack(v, dim=0) for v in views]

    def preprocess_tensor_batch(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Expected image tensor batch with shape [B, C, H, W].")

        if x.shape[1] != 3 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)

        if x.shape[1] != 3:
            raise ValueError("Expected 3-channel RGB tensor input.")

        x = x.float()
        if x.max() > 1.5:
            x = x / 255.0

        x = F.interpolate(
            x,
            size=(self.resize_size, self.resize_size),
            mode="bicubic",
            align_corners=False,
        )

        margin = self.resize_size - self.image_size
        top = margin // 2
        left = margin // 2
        x = x[:, :, top : top + self.image_size, left : left + self.image_size]

        mean = self.registered_mean.to(device=x.device, dtype=x.dtype)
        std = self.registered_std.to(device=x.device, dtype=x.dtype)

        x_min = float(x.min().detach().cpu())
        x_max = float(x.max().detach().cpu())
        already_normalized = (x_min < -0.5) or (x_max > 1.5)
        if not already_normalized:
            x = (x - mean) / std

        return x


class Filter:
    def __init__(self, warmup_steps=100, low=0.75, high=0.92, proto_min=0.35):
        self.warmup_steps = warmup_steps
        self.low = low
        self.high = high
        self.proto_min = proto_min

    def apply(self, pseudo_labels, metadata, step=0):
        thr = self.low if step < self.warmup_steps else self.high
        conf = metadata["confidence"]
        proto_conf = metadata["proto_probs"].max(dim=1).values
        mask = (conf >= thr) & (proto_conf >= self.proto_min)
        return mask


class SSL_Loss:
    def __init__(
        self,
        num_classes=45,
        label_smoothing=0.1,
        lambda_u=1.5,
        lambda_c=0.5,
        lambda_sc=0.15,
        lambda_mm=0.25,
        lambda_mask=0.7,
        manifold_mix_alpha=2.0,
        supcon_temp=0.10,
        class_counts=None,
        device=None,
        focal_gamma=1.5,
    ):
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.lambda_u = lambda_u
        self.lambda_c = lambda_c
        self.lambda_sc = lambda_sc
        self.lambda_mm = lambda_mm
        self.lambda_mask = lambda_mask
        self.manifold_mix_alpha = manifold_mix_alpha
        self.supcon_temp = supcon_temp
        self.focal_gamma = focal_gamma
        self.balanced_ce = None
        self.balanced_focal_ce = None
        self.device = device
        if class_counts is not None:
            self.set_class_counts(class_counts)

    def set_device(self, device):
        self.device = device
        if self.balanced_ce is not None:
            self.balanced_ce = self.balanced_ce.to(device)
        if self.balanced_focal_ce is not None:
            self.balanced_focal_ce = self.balanced_focal_ce.to(device)

    def set_class_counts(self, class_counts):
        self.balanced_ce = BalancedSoftmaxCE(
            class_counts=class_counts,
            label_smoothing=self.label_smoothing,
        )
        self.balanced_focal_ce = BalancedFocalSoftmaxCE(
            class_counts=class_counts,
            gamma=self.focal_gamma,
            label_smoothing=self.label_smoothing,
        )
        if self.device is not None:
            self.balanced_ce = self.balanced_ce.to(self.device)
            self.balanced_focal_ce = self.balanced_focal_ce.to(self.device)

    def supervised(self, preds, y_true):
        if self.balanced_focal_ce is None:
            return F.cross_entropy(preds, y_true, label_smoothing=self.label_smoothing)
        return self.balanced_focal_ce(preds, y_true)

    def unsupervised(self, preds, targets, mask):
        if mask.sum() == 0:
            return preds.new_tensor(0.0)
        loss = F.cross_entropy(preds, targets, reduction="none")
        return (loss * mask.float()).sum() / mask.float().sum().clamp_min(1.0)

    def consistency(self, student_features, teacher_features, mask):
        if mask.sum() == 0:
            return student_features.new_tensor(0.0)
        sf = F.normalize(student_features, dim=1)
        tf = F.normalize(teacher_features, dim=1)
        mse = ((sf - tf) ** 2).mean(dim=1)
        return (mse * mask.float()).sum() / mask.float().sum().clamp_min(1.0)

    def masked_consistency(self, student_logits, teacher_probs, sample_weights=None):
        log_q = F.log_softmax(student_logits, dim=1)
        kl = F.kl_div(log_q, teacher_probs, reduction="none").sum(dim=1)
        if sample_weights is None:
            return kl.mean()
        w = sample_weights.to(device=kl.device, dtype=kl.dtype)
        return (kl * w).sum() / w.sum().clamp_min(1e-6)

    def supervised_contrastive(self, feats, labels):
        feats = F.normalize(feats, dim=1)
        device = feats.device
        n = feats.size(0)

        sim = torch.matmul(feats, feats.t()) / self.supcon_temp
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.t()).float().to(device)

        logits_mask = torch.ones_like(mask) - torch.eye(n, device=device)
        mask = mask * logits_mask

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-12))

        pos_count = mask.sum(dim=1)
        valid = pos_count > 0
        if valid.sum() == 0:
            return feats.new_tensor(0.0)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / pos_count.clamp_min(1.0)
        loss = -mean_log_prob_pos[valid].mean()
        return loss

    def manifold_mixup(self, feats, labels):
        b = feats.size(0)
        if b < 2:
            return None, None

        alpha = self.manifold_mix_alpha
        lam = float(np.random.beta(alpha, alpha))
        lam = max(lam, 1.0 - lam)

        perm = torch.randperm(b, device=feats.device)
        feats_perm = feats[perm]
        labels_perm = labels[perm]

        feats_n = F.normalize(feats, dim=1)
        feats_perm_n = F.normalize(feats_perm, dim=1)
        mixed_feats = lam * feats_n + (1.0 - lam) * feats_perm_n

        y1 = F.one_hot(labels, num_classes=self.num_classes).float()
        y2 = F.one_hot(labels_perm, num_classes=self.num_classes).float()
        mixed_targets = lam * y1 + (1.0 - lam) * y2

        if self.label_smoothing > 0.0:
            mixed_targets = (
                mixed_targets * (1.0 - self.label_smoothing)
                + self.label_smoothing / self.num_classes
            )

        return mixed_feats, mixed_targets

    def soft_ce(self, logits, soft_targets):
        log_probs = F.log_softmax(logits, dim=1)
        return -(soft_targets * log_probs).sum(dim=1).mean()

    def total(self, L_sup, L_sc, L_mm, L_unsup, L_cons):
        return (
            L_sup
            + self.lambda_sc * L_sc
            + self.lambda_mm * L_mm
            + self.lambda_u * L_unsup
            + self.lambda_c * L_cons
        )

    def weighted_supervised(self, preds, y_true, sample_weights):
        if self.balanced_ce is None:
            loss = F.cross_entropy(
                preds, y_true, reduction="none", label_smoothing=self.label_smoothing
            )
            sw = sample_weights.float()
            return (loss * sw).sum() / sw.sum().clamp_min(1.0)
        return self.balanced_ce(preds, y_true, sample_weights=sample_weights)


class SSL_Algorithm:
    def __init__(self, num_classes=45, image_size=224, device=None):
        self.num_classes = num_classes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.label_gen = LabelGenerator(
            num_classes=num_classes, ema_decay=0.999, alpha=0.7, proto_temp=0.07
        )
        self.aug = Augmentation(image_size=image_size)
        self.filter = Filter(warmup_steps=100, low=0.75, high=0.92, proto_min=0.35)
        self.ssl = SSL_Loss(
            num_classes=num_classes,
            label_smoothing=0.1,
            lambda_u=1.5,
            lambda_c=0.5,
            lambda_sc=0.15,
            lambda_mm=0.25,
            lambda_mask=0.7,
            manifold_mix_alpha=2.0,
            supcon_temp=0.10,
            device=self.device,
            focal_gamma=1.5,
        )
        self.model = None
        self.predict_model = None
        self.transform = None
        self.idx_to_cat_id = None
        self.cat_id_to_idx = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        self.final_centroids = None
        self.centroid_alpha = 2.0
        self.labeled_centroids_for_pseudo = None
        self.labeled_eval_loader_for_adapt = None
        self.class_counts = None
        self.labeled_repeat_factor = 6

    def _make_repeated_labeled_loader(
        self, labeled_ds, batch_size, num_workers, pin_memory, drop_last=True
    ):
        labels = torch.tensor(
            [s["label"] for s in labeled_ds.samples], dtype=torch.long
        )
        class_counts = (
            torch.bincount(labels, minlength=self.num_classes).float().clamp_min(1.0)
        )
        sample_weights = (1.0 / class_counts[labels]).double()
        num_samples = max(len(labeled_ds), self.labeled_repeat_factor * len(labeled_ds))
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=num_samples,
            replacement=True,
        )
        loader = DataLoader(
            labeled_ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last and num_samples >= batch_size,
            collate_fn=lambda batch: (
                [b[0] for b in batch],
                torch.tensor([b[1] for b in batch], dtype=torch.long),
            ),
        )
        return loader

    def _manifold_mix_loss_from_feats(self, feats1, feats2, y, model):
        mixed_feats_1, mixed_targets_1 = self.ssl.manifold_mixup(feats1, y)
        mixed_feats_2, mixed_targets_2 = self.ssl.manifold_mixup(feats2, y)
        losses = []
        if mixed_feats_1 is not None:
            logits_m1 = model.forward_head(mixed_feats_1)
            losses.append(self.ssl.soft_ce(logits_m1, mixed_targets_1))
        if mixed_feats_2 is not None:
            logits_m2 = model.forward_head(mixed_feats_2)
            losses.append(self.ssl.soft_ce(logits_m2, mixed_targets_2))
        if len(losses) == 0:
            return feats1.new_tensor(0.0)
        return sum(losses) / len(losses)

    def ssl_step(self, model, batch_l, batch_u, step, optimizer):
        imgs_l, y_l = batch_l
        y_l = y_l.to(self.device, non_blocking=True)

        x_l1, x_l2 = self.aug.generate_supervised_pair(imgs_l)
        x_l1 = x_l1.to(self.device, non_blocking=True)
        x_l2 = x_l2.to(self.device, non_blocking=True)

        imgs_u = batch_u
        views = self.aug.generate_views(imgs_u)
        x_uw = views["weak"].to(self.device, non_blocking=True)
        x_us = views["strong"].to(self.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits_l1, feats_l1 = model(x_l1)
            logits_l2, feats_l2 = model(x_l2)

            L_sup = 0.5 * (
                self.ssl.supervised(logits_l1, y_l)
                + self.ssl.supervised(logits_l2, y_l)
            )

            feats_supcon = torch.cat([feats_l1, feats_l2], dim=0)
            labels_supcon = torch.cat([y_l, y_l], dim=0)
            L_sc = self.ssl.supervised_contrastive(feats_supcon, labels_supcon)
            L_mm = self._manifold_mix_loss_from_feats(feats_l1, feats_l2, y_l, model)

            pseudo_labels, metadata = self.label_gen.generate(model, x_uw)
            mask = self.filter.apply(pseudo_labels, metadata, step=step)

            logits_us, feats_us = model(x_us)
            L_unsup = self.ssl.unsupervised(logits_us, pseudo_labels, mask)
            L_cons = self.ssl.consistency(feats_us, metadata["teacher_features"], mask)

            loss = self.ssl.total(L_sup, L_sc, L_mm, L_unsup, L_cons)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        self.scaler.step(optimizer)
        self.scaler.update()

        self.label_gen.update(model, step)

        return {
            "loss": float(loss.detach().cpu()),
            "mask_ratio": float(mask.float().mean().detach().cpu()),
        }

    def _masked_consistency_refine(
        self,
        labeled_loader,
        unlabeled_loader,
        epochs=2,
        lr=5e-5,
    ):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.03)
        steps_per_epoch = max(
            1, min(len(unlabeled_loader), max(1, len(labeled_loader)))
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs * steps_per_epoch)
        )

        self.model.train()
        for _ in range(epochs):
            l_iter = cycle(labeled_loader)
            u_iter = iter(unlabeled_loader)

            for _ in range(steps_per_epoch):
                try:
                    imgs_u = next(u_iter)
                except StopIteration:
                    break

                imgs_l, y_l = next(l_iter)
                y_l = y_l.to(self.device, non_blocking=True)

                mv = self.aug.generate_masked_consistency_views(imgs_u)
                x_t = mv["teacher"].to(self.device, non_blocking=True)
                x_s = mv["student"].to(self.device, non_blocking=True)

                x_l1, x_l2 = self.aug.generate_supervised_pair(imgs_l)
                x_l1 = x_l1.to(self.device, non_blocking=True)
                x_l2 = x_l2.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    self.label_gen.teacher_model.eval()
                    logits_t, _ = self.label_gen.teacher_model(x_t)
                    teacher_probs = torch.softmax(logits_t / 0.8, dim=1)
                    conf = teacher_probs.max(dim=1).values
                    weights = ((conf - 0.45).clamp_min(0.0) / 0.55).pow(1.5)

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits_s, _ = self.model(x_s)
                    loss_mask = self.ssl.masked_consistency(
                        logits_s, teacher_probs, sample_weights=weights
                    )

                    logits_l1, feats_l1 = self.model(x_l1)
                    logits_l2, feats_l2 = self.model(x_l2)
                    loss_sup = 0.5 * (
                        self.ssl.supervised(logits_l1, y_l)
                        + self.ssl.supervised(logits_l2, y_l)
                    )
                    feats_supcon = torch.cat([feats_l1, feats_l2], dim=0)
                    labels_supcon = torch.cat([y_l, y_l], dim=0)
                    loss_sc = self.ssl.supervised_contrastive(
                        feats_supcon, labels_supcon
                    )
                    loss_mm = self._manifold_mix_loss_from_feats(
                        feats_l1, feats_l2, y_l, self.model
                    )

                    loss = (
                        0.6 * loss_sup
                        + 0.5 * self.ssl.lambda_sc * loss_sc
                        + 0.5 * self.ssl.lambda_mm * loss_mm
                        + self.ssl.lambda_mask * loss_mask
                    )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
                scheduler.step()
                self.label_gen.update(self.model, 0)

        self.label_gen.teacher_model = copy.deepcopy(self.model).to(self.device)
        self.label_gen.teacher_model.eval()

    @torch.no_grad()
    def _extract_features_labels(self, loader):
        self.model.eval()
        feats_all = []
        labels_all = []
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            _, feats = self.model(x)
            feats_all.append(F.normalize(feats, dim=1))
            labels_all.append(y.to(self.device, non_blocking=True))
        return torch.cat(feats_all, dim=0), torch.cat(labels_all, dim=0)

    @torch.no_grad()
    def _extract_features_unlabeled(self, loader):
        self.model.eval()
        feats_all = []
        for x in loader:
            x = x.to(self.device, non_blocking=True)
            _, feats = self.model(x)
            feats_all.append(F.normalize(feats, dim=1))
        return torch.cat(feats_all, dim=0)

    @torch.no_grad()
    def _build_initial_centroids(self, labeled_eval_loader):
        feats_l, y_l = self._extract_features_labels(labeled_eval_loader)
        d = feats_l.size(1)
        centroids = torch.zeros(self.num_classes, d, device=self.device)
        counts = torch.zeros(self.num_classes, device=self.device)
        centroids.index_add_(0, y_l, feats_l)
        counts.index_add_(0, y_l, torch.ones_like(y_l, dtype=torch.float))
        centroids = centroids / counts.clamp_min(1.0).unsqueeze(1)
        centroids = F.normalize(centroids, dim=1)
        return centroids, feats_l, y_l

    @torch.no_grad()
    def _refine_centroids_transductive(
        self, labeled_eval_loader, unlabeled_eval_loader, iters=3, margin_thr=0.08
    ):
        centroids, feats_l, y_l = self._build_initial_centroids(labeled_eval_loader)
        feats_u = self._extract_features_unlabeled(unlabeled_eval_loader)

        d = feats_l.size(1)
        ones_l = torch.ones(feats_l.size(0), device=self.device)

        for _ in range(iters):
            sims = feats_u @ centroids.t()
            top2_vals, top2_idx = sims.topk(k=2, dim=1)
            pseudo = top2_idx[:, 0]
            margin = top2_vals[:, 0] - top2_vals[:, 1]
            keep = margin >= margin_thr

            sum_feats = torch.zeros(self.num_classes, d, device=self.device)
            counts = torch.zeros(self.num_classes, device=self.device)

            sum_feats.index_add_(0, y_l, feats_l)
            counts.index_add_(0, y_l, ones_l)

            if keep.any():
                feats_sel = feats_u[keep]
                pseudo_sel = pseudo[keep]
                sum_feats.index_add_(0, pseudo_sel, feats_sel)
                counts.index_add_(
                    0, pseudo_sel, torch.ones_like(pseudo_sel, dtype=torch.float)
                )

            centroids = sum_feats / counts.clamp_min(1.0).unsqueeze(1)
            centroids = F.normalize(centroids, dim=1)

        self.final_centroids = centroids

    @torch.no_grad()
    def _build_agreement_balanced_pseudo_records(
        self,
        img_dir,
        unlabel_ann_file,
        labeled_eval_loader,
        batch_size=64,
        max_per_class=100,
        min_teacher_conf=0.55,
        min_centroid_sim=0.20,
        min_teacher_margin=0.03,
        min_centroid_margin=0.02,
    ):
        if self.label_gen.teacher_model is None:
            return []

        labeled_centroids, _, _ = self._build_initial_centroids(labeled_eval_loader)
        self.labeled_centroids_for_pseudo = labeled_centroids

        ds = CocoUnlabeledDataset(
            img_dir, unlabel_ann_file, transform=self.aug.eval_tf, return_meta=True
        )
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=default_num_workers(),
            pin_memory=torch.cuda.is_available(),
            collate_fn=lambda batch: (
                torch.stack([b[0] for b in batch], dim=0),
                [b[1] for b in batch],
                [b[2] for b in batch],
            ),
        )

        self.label_gen.teacher_model.eval()
        candidates = [[] for _ in range(self.num_classes)]

        for x, image_ids, file_names in loader:
            x = x.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits_t, feats_t = self.label_gen.teacher_model(x)

            probs_t = torch.softmax(logits_t, dim=1)
            t_top2_vals, t_top2_idx = probs_t.topk(k=2, dim=1)
            t_pred = t_top2_idx[:, 0]
            t_conf = t_top2_vals[:, 0]
            t_margin = t_top2_vals[:, 0] - t_top2_vals[:, 1]

            feats_n = F.normalize(feats_t.float(), dim=1)
            sims = feats_n @ labeled_centroids.t()
            c_top2_vals, c_top2_idx = sims.topk(k=2, dim=1)
            c_pred = c_top2_idx[:, 0]
            c_conf = c_top2_vals[:, 0]
            c_margin = c_top2_vals[:, 0] - c_top2_vals[:, 1]

            agree = t_pred.eq(c_pred)
            quality_mask = (
                agree
                & (t_conf >= min_teacher_conf)
                & (c_conf >= min_centroid_sim)
                & (t_margin >= min_teacher_margin)
                & (c_margin >= min_centroid_margin)
            )

            if quality_mask.any():
                idxs = torch.nonzero(quality_mask, as_tuple=False).squeeze(1)
                joint_score = (
                    0.45 * t_conf[idxs]
                    + 0.30 * c_conf[idxs]
                    + 0.15 * t_margin[idxs]
                    + 0.10 * c_margin[idxs]
                )
                for local_j, score in zip(idxs.tolist(), joint_score.tolist()):
                    cls = int(t_pred[local_j].item())
                    candidates[cls].append(
                        {
                            "image_id": int(image_ids[local_j]),
                            "file_name": file_names[local_j],
                            "label": cls,
                            "weight": float(max(0.2, min(1.0, score))),
                            "score": float(score),
                            "teacher_conf": float(t_conf[local_j].item()),
                            "centroid_conf": float(c_conf[local_j].item()),
                            "teacher_margin": float(t_margin[local_j].item()),
                            "centroid_margin": float(c_margin[local_j].item()),
                        }
                    )

        selected = []
        for cls in range(self.num_classes):
            if not candidates[cls]:
                continue
            candidates[cls].sort(
                key=lambda r: (
                    r["score"],
                    r["teacher_conf"],
                    r["centroid_conf"],
                    r["teacher_margin"],
                    r["centroid_margin"],
                ),
                reverse=True,
            )
            selected.extend(candidates[cls][:max_per_class])

        return selected

    def _update_balanced_counts_with_pseudo(self, pseudo_records, pseudo_weight=0.5):
        counts = self.class_counts.clone().float()
        if pseudo_records is not None and len(pseudo_records) > 0:
            pseudo_labels = torch.tensor(
                [int(r["label"]) for r in pseudo_records], dtype=torch.long
            )
            pseudo_hist = torch.bincount(
                pseudo_labels, minlength=self.num_classes
            ).float()
            counts = counts + pseudo_weight * pseudo_hist
        self.ssl.set_class_counts(counts.tolist())

    def _finetune_on_expanded_set(
        self, labeled_loader, pseudo_loader, pseudo_records, epochs=4, lr=8e-5
    ):
        self._update_balanced_counts_with_pseudo(pseudo_records, pseudo_weight=0.5)

        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.05)
        steps_per_epoch = max(len(labeled_loader), len(pseudo_loader))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs * steps_per_epoch)
        )

        self.model.train()
        for _ in range(epochs):
            real_iter_cycle = cycle(labeled_loader)
            pseudo_iter_cycle = cycle(pseudo_loader)

            for _ in range(steps_per_epoch):
                imgs_r, y_r = next(real_iter_cycle)
                x_r1, x_r2 = self.aug.generate_supervised_pair(imgs_r)
                x_r1 = x_r1.to(self.device, non_blocking=True)
                x_r2 = x_r2.to(self.device, non_blocking=True)
                y_r = y_r.to(self.device, non_blocking=True)

                x_p, y_p, w_p = next(pseudo_iter_cycle)
                x_p = x_p.to(self.device, non_blocking=True)
                y_p = y_p.to(self.device, non_blocking=True)
                w_p = w_p.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits_r1, feats_r1 = self.model(x_r1)
                    logits_r2, feats_r2 = self.model(x_r2)
                    logits_p, _ = self.model(x_p)

                    loss_r_ce = 0.5 * (
                        self.ssl.supervised(logits_r1, y_r)
                        + self.ssl.supervised(logits_r2, y_r)
                    )
                    feats_supcon = torch.cat([feats_r1, feats_r2], dim=0)
                    labels_supcon = torch.cat([y_r, y_r], dim=0)
                    loss_r_sc = self.ssl.supervised_contrastive(
                        feats_supcon, labels_supcon
                    )
                    loss_r_mm = self._manifold_mix_loss_from_feats(
                        feats_r1, feats_r2, y_r, self.model
                    )

                    loss_p = self.ssl.weighted_supervised(
                        logits_p, y_p, sample_weights=0.5 + 0.5 * w_p
                    )
                    loss = (
                        loss_r_ce
                        + self.ssl.lambda_sc * loss_r_sc
                        + self.ssl.lambda_mm * loss_r_mm
                        + 0.7 * loss_p
                    )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
                scheduler.step()

        self.label_gen.teacher_model = copy.deepcopy(self.model).to(self.device)
        self.label_gen.teacher_model.eval()

    def _enable_tta_adaptation_params(self, model):
        for p in model.parameters():
            p.requires_grad = False

        if hasattr(model.backbone, "norm"):
            for p in model.backbone.norm.parameters():
                p.requires_grad = True

        if hasattr(model.backbone, "blocks") and len(model.backbone.blocks) > 0:
            for p in model.backbone.blocks[-1].parameters():
                p.requires_grad = True

        for p in model.head.parameters():
            p.requires_grad = True

    @torch.no_grad()
    def _adapted_feature_centroids(self, adapted_model, labeled_eval_loader):
        adapted_model.eval()
        feats_all = []
        labels_all = []
        for x, y in labeled_eval_loader:
            x = x.to(self.device, non_blocking=True)
            _, feats = adapted_model(x)
            feats_all.append(F.normalize(feats, dim=1))
            labels_all.append(y.to(self.device, non_blocking=True))
        feats_l = torch.cat(feats_all, dim=0)
        y_l = torch.cat(labels_all, dim=0)
        d = feats_l.size(1)
        centroids = torch.zeros(self.num_classes, d, device=self.device)
        counts = torch.zeros(self.num_classes, device=self.device)
        centroids.index_add_(0, y_l, feats_l)
        counts.index_add_(0, y_l, torch.ones_like(y_l, dtype=torch.float))
        centroids = centroids / counts.clamp_min(1.0).unsqueeze(1)
        centroids = F.normalize(centroids, dim=1)
        return centroids

    def adapt_for_prediction(
        self,
        img_dir,
        val_ann_file,
        labeled_eval_loader,
        epochs=2,
        batch_size=64,
        lr=2e-5,
    ):
        if val_ann_file is None or (not os.path.exists(val_ann_file)):
            self.predict_model = self.model
            return

        adapt_ds = CocoImageDataset(img_dir, val_ann_file, transform=None)
        adapt_loader = DataLoader(
            adapt_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=default_num_workers(),
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            collate_fn=lambda batch: [b[0] for b in batch],
        )

        adapted = copy.deepcopy(self.model).to(self.device)
        self._enable_tta_adaptation_params(adapted)
        adapted.train()

        trainable = [p for p in adapted.parameters() if p.requires_grad]
        if len(trainable) == 0:
            self.predict_model = self.model
            return

        anchor_params = {
            n: p.detach().clone().to(self.device)
            for n, p in adapted.named_parameters()
            if p.requires_grad
        }
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        lambda_cons = 0.5
        lambda_anchor = 1e-3

        for _ in range(epochs):
            for imgs in adapt_loader:
                views = self.aug.generate_tta_adapt_views(imgs)
                x1 = views["eval"].to(self.device, non_blocking=True)
                x2 = views["weak"].to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits1, _ = adapted(x1)
                    logits2, _ = adapted(x2)

                    p1 = torch.softmax(logits1, dim=1)
                    p2 = torch.softmax(logits2, dim=1)

                    ent1 = -(p1 * torch.log(p1.clamp_min(1e-8))).sum(dim=1).mean()
                    ent2 = -(p2 * torch.log(p2.clamp_min(1e-8))).sum(dim=1).mean()
                    loss_ent = 0.5 * (ent1 + ent2)

                    log_p1 = F.log_softmax(logits1, dim=1)
                    log_p2 = F.log_softmax(logits2, dim=1)
                    loss_cons = 0.5 * (
                        F.kl_div(log_p1, p2.detach(), reduction="batchmean")
                        + F.kl_div(log_p2, p1.detach(), reduction="batchmean")
                    )

                    anchor = logits1.new_tensor(0.0)
                    for n, p in adapted.named_parameters():
                        if p.requires_grad:
                            anchor = anchor + (p - anchor_params[n]).pow(2).mean()

                    loss = loss_ent + lambda_cons * loss_cons + lambda_anchor * anchor

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()

        adapted.eval()
        self.predict_model = adapted
        self.final_centroids = self._adapted_feature_centroids(
            adapted, labeled_eval_loader
        )

    def fit(self, img_dir, train_ann_file, unlabel_ann_file):
        train_data = read_json(train_ann_file)
        cat_ids = sorted([c["id"] for c in train_data["categories"]])
        self.cat_id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
        self.idx_to_cat_id = {i: cid for cid, i in self.cat_id_to_idx.items()}

        labeled_ds = CocoLabeledDataset(
            img_dir, train_ann_file, transform=None, cat_id_to_idx=self.cat_id_to_idx
        )
        labeled_eval_ds = CocoLabeledDataset(
            img_dir,
            train_ann_file,
            transform=self.aug.eval_tf,
            cat_id_to_idx=self.cat_id_to_idx,
        )
        unlabeled_ds = CocoUnlabeledDataset(img_dir, unlabel_ann_file, transform=None)
        unlabeled_eval_ds = CocoUnlabeledDataset(
            img_dir, unlabel_ann_file, transform=self.aug.eval_tf
        )

        labels_all = torch.tensor(
            [s["label"] for s in labeled_ds.samples], dtype=torch.long
        )
        self.class_counts = torch.bincount(
            labels_all, minlength=self.num_classes
        ).float()
        self.ssl.set_device(self.device)
        self.ssl.set_class_counts(self.class_counts.tolist())

        nw = default_num_workers()
        pin = torch.cuda.is_available()

        labeled_loader = self._make_repeated_labeled_loader(
            labeled_ds=labeled_ds,
            batch_size=32,
            num_workers=nw,
            pin_memory=pin,
            drop_last=True,
        )
        labeled_eval_loader = DataLoader(
            labeled_eval_ds,
            batch_size=64,
            shuffle=False,
            num_workers=nw,
            pin_memory=pin,
        )
        unlabeled_loader = DataLoader(
            unlabeled_ds,
            batch_size=64,
            shuffle=True,
            num_workers=nw,
            pin_memory=pin,
            drop_last=len(unlabeled_ds) >= 64,
            collate_fn=pil_list_collate,
        )
        unlabeled_eval_loader = DataLoader(
            unlabeled_eval_ds,
            batch_size=64,
            shuffle=False,
            num_workers=nw,
            pin_memory=pin,
        )

        self.model = ViTClassifier(
            model_name="vit_large_patch14_clip_224.openai",
            num_classes=self.num_classes,
            pretrained=True,
        ).to(self.device)
        freeze_vit_partial(self.model, train_last_blocks=4)

        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=2e-4,
            weight_decay=0.05,
        )

        warmup_epochs = 3
        ssl_epochs = 7
        total_train_steps = warmup_epochs * max(
            1, len(labeled_loader)
        ) + ssl_epochs * max(1, len(labeled_loader))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_train_steps)
        )

        total_steps = 0
        self.model.train()

        for _ in range(warmup_epochs):
            for imgs, y in labeled_loader:
                y = y.to(self.device, non_blocking=True)
                x1, x2 = self.aug.generate_supervised_pair(imgs)
                x1 = x1.to(self.device, non_blocking=True)
                x2 = x2.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits1, feats1 = self.model(x1)
                    logits2, feats2 = self.model(x2)

                    loss_ce = 0.5 * (
                        self.ssl.supervised(logits1, y)
                        + self.ssl.supervised(logits2, y)
                    )
                    feats_supcon = torch.cat([feats1, feats2], dim=0)
                    labels_supcon = torch.cat([y, y], dim=0)
                    loss_sc = self.ssl.supervised_contrastive(
                        feats_supcon, labels_supcon
                    )
                    loss_mm = self._manifold_mix_loss_from_feats(
                        feats1, feats2, y, self.model
                    )
                    loss = (
                        loss_ce
                        + self.ssl.lambda_sc * loss_sc
                        + self.ssl.lambda_mm * loss_mm
                    )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
                scheduler.step()
                total_steps += 1

        self.label_gen.fit(self.model, labeled_eval_loader, self.device)

        self.model.train()
        for epoch in range(ssl_epochs):
            if epoch == max(1, ssl_epochs // 2):
                self.label_gen.refresh_prototypes(
                    self.model, labeled_eval_loader, self.device
                )

            u_iter = cycle(unlabeled_loader)
            for batch_l in labeled_loader:
                batch_u = next(u_iter)
                self.ssl_step(self.model, batch_l, batch_u, total_steps, optimizer)
                scheduler.step()
                total_steps += 1

        self.label_gen.refresh_prototypes(self.model, labeled_eval_loader, self.device)

        self._masked_consistency_refine(
            labeled_loader=labeled_loader,
            unlabeled_loader=unlabeled_loader,
            epochs=2,
            lr=5e-5,
        )
        self.label_gen.refresh_prototypes(self.model, labeled_eval_loader, self.device)

        pseudo_records = self._build_agreement_balanced_pseudo_records(
            img_dir=img_dir,
            unlabel_ann_file=unlabel_ann_file,
            labeled_eval_loader=labeled_eval_loader,
            batch_size=64,
            max_per_class=100,
            min_teacher_conf=0.55,
            min_centroid_sim=0.20,
            min_teacher_margin=0.03,
            min_centroid_margin=0.02,
        )

        if len(pseudo_records) > 0:
            pseudo_ds = PseudoLabeledDataset(
                img_dir, pseudo_records, transform=self.aug.sup
            )
            pseudo_loader = DataLoader(
                pseudo_ds,
                batch_size=32,
                shuffle=True,
                num_workers=nw,
                pin_memory=pin,
                drop_last=len(pseudo_ds) >= 32,
            )
            self._finetune_on_expanded_set(
                labeled_loader,
                pseudo_loader,
                pseudo_records=pseudo_records,
                epochs=4,
                lr=8e-5,
            )
            self.label_gen.refresh_prototypes(
                self.model, labeled_eval_loader, self.device
            )
        else:
            self.ssl.set_class_counts(self.class_counts.tolist())

        self.model.eval()
        self._refine_centroids_transductive(
            labeled_eval_loader=labeled_eval_loader,
            unlabeled_eval_loader=unlabeled_eval_loader,
            iters=3,
            margin_thr=0.08,
        )
        self.predict_model = self.model
        self.labeled_eval_loader_for_adapt = labeled_eval_loader

    @torch.no_grad()
    def predict(self, image_batch):
        model = self.predict_model if self.predict_model is not None else self.model
        model.eval()

        if isinstance(image_batch, list):
            tta_batches = self.aug.tta_views(image_batch)
            logits_acc = None
            feats_acc = None

            for xb in tta_batches:
                xb = xb.to(self.device, non_blocking=True)
                logits_v, feats_v = model(xb)
                if logits_acc is None:
                    logits_acc = logits_v
                    feats_acc = feats_v
                else:
                    logits_acc = logits_acc + logits_v
                    feats_acc = feats_acc + feats_v

            logits = logits_acc / len(tta_batches)
            features = feats_acc / len(tta_batches)

        elif isinstance(image_batch, torch.Tensor):
            x = self.aug.preprocess_tensor_batch(image_batch)
            x = x.to(self.device, non_blocking=True)
            logits, features = model(x)
        else:
            raise TypeError(
                "image_batch must be a list of PIL images or a torch.Tensor"
            )

        if self.final_centroids is not None:
            feat_n = F.normalize(features, dim=1)
            centroid_scores = feat_n @ self.final_centroids.t()
            final_scores = logits + self.centroid_alpha * centroid_scores
        else:
            final_scores = logits

        pred_idx = final_scores.argmax(dim=1).tolist()
        predictions = [{"category_id": self.idx_to_cat_id[i]} for i in pred_idx]
        return final_scores, features, predictions


def main():
    set_seed(42)

    img_dir = find_existing(
        ["/data/images", "/data/imgs", "/data/train_images", "/data"]
    )

    train_ann = find_existing(
        [
            "/data/train.json",
            "/data/train_ann.json",
            "/data/annotations/train.json",
            "/data/annotations/train_ann.json",
        ]
    )

    unlabel_ann = find_existing(
        [
            "/data/unlabeled.json",
            "/data/unlabel.json",
            "/data/annotations/unlabeled.json",
            "/data/annotations/unlabel.json",
        ]
    )

    val_ann = find_existing(
        [
            "/data/val.json",
            "/data/valid.json",
            "/data/validation.json",
            "/data/annotations/val.json",
            "/data/annotations/valid.json",
            "/data/annotations/validation.json",
        ]
    )

    if img_dir is None or train_ann is None or unlabel_ann is None:
        raise FileNotFoundError("Could not locate required dataset files under /data.")

    algo = SSL_Algorithm(num_classes=45, image_size=224)
    algo.fit(img_dir, train_ann, unlabel_ann)

    if val_ann is not None:
        algo.adapt_for_prediction(
            img_dir=img_dir,
            val_ann_file=val_ann,
            labeled_eval_loader=algo.labeled_eval_loader_for_adapt,
            epochs=2,
            batch_size=64,
            lr=2e-5,
        )

        val_ds = CocoImageDataset(img_dir, val_ann, transform=None)
        val_loader = DataLoader(
            val_ds,
            batch_size=32,
            shuffle=False,
            num_workers=default_num_workers(),
            pin_memory=False,
            collate_fn=lambda batch: (
                [b[0] for b in batch],
                torch.tensor([b[1] for b in batch], dtype=torch.long),
                [b[2] for b in batch],
            ),
        )

        results = []
        with torch.no_grad():
            for imgs, image_ids, file_names in val_loader:
                _, _, preds = algo.predict(imgs)
                for image_id, file_name, pred in zip(
                    image_ids.tolist(), file_names, preds
                ):
                    results.append(
                        {
                            "image_id": image_id,
                            "file_name": file_name,
                            "category_id": int(pred["category_id"]),
                        }
                    )

        out_path = "/work/predictions.json"
        with open(out_path, "w") as f:
            json.dump(results, f)
    else:
        out_path = "/work/predictions.json"
        with open(out_path, "w") as f:
            json.dump([], f)


if __name__ == "__main__":
    main()