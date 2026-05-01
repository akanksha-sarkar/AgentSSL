# THIS PROGRAM GOT A FITNESS SCORE OF 0.668
from __future__ import annotations
import json
import os
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

def _coco_label_mapping(train_ann_file: str):
    with open(train_ann_file, "r") as f:
        coco = json.load(f)
    cat_ids = sorted(c["id"] for c in coco["categories"])
    cat_id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
    idx_to_cat_id = {i: cid for cid, i in cat_id_to_idx.items()}
    return cat_id_to_idx, idx_to_cat_id, len(cat_ids)


def _labeled_samples(img_dir: str, train_ann_file: str, cat_id_to_idx: dict):
    with open(train_ann_file, "r") as f:
        coco = json.load(f)
    id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}
    samples = []
    for ann in coco["annotations"]:
        fn = id_to_name.get(ann["image_id"])
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


class _LabeledDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), y


class ClassificationAgent:
    NET_NAME = "timm/vit_base_patch16_clip_224.openai"
    PEFT_CONFIG = {
    "method_name": "lora_1",
    "lora_bottleneck": 4
}
    TRAIN_EPOCHS = 100
    def __init__(self, net_builder_fn, get_peft_config_fn, num_classes=45):
        self.net_builder_fn = net_builder_fn
        self.get_peft_config_fn = get_peft_config_fn
        self.batch_size = 32
        self.lr = 1e-3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None
        self.idx_to_cat_id = None
        self.num_classes = num_classes
        
        self.train_epochs = self.TRAIN_EPOCHS
        self.net_name = self.NET_NAME
        self.peft_config = self.get_peft_config_fn(self.PEFT_CONFIG)

        self.vit_config = {"drop_path_rate": 0}
        self.net_builder = self.net_builder_fn(self.net_name, self.peft_config, self.vit_config)
        self.model = self.net_builder(
            num_classes=self.num_classes,
            pretrained=True,
            pretrained_path="",
        ).to(self.device)

    def _eval_transform(self, img_size: int = 224):
        resize = int(img_size * 256 / 224)
        return T.Compose(
            [
                T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(img_size),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

    def fit(self, img_dir, train_ann_file, unlabel_ann_file):

        cat_id_to_idx, idx_to_cat_id, num_classes = _coco_label_mapping(train_ann_file)
        self.idx_to_cat_id = idx_to_cat_id
        self.num_classes = num_classes

        train_tf = T.Compose(
            [
                T.RandomResizedCrop(224, scale=(0.7, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                T.RandomRotation(degrees=10),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        self.transform = self._eval_transform(224)

        samples = _labeled_samples(img_dir, train_ann_file, cat_id_to_idx)
        if len(samples) == 0:
            raise RuntimeError("No labeled samples found; check img_dir and train_ann_file.")

        ds = _LabeledDataset(samples, train_tf)
        bs = min(self.batch_size, len(ds))
        loader = DataLoader(
            ds,
            batch_size=bs,
            shuffle=True,
            num_workers=min(4, os.cpu_count() or 0),
            pin_memory=self.device.type == "cuda",
            drop_last=False,
        )

        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=0.05)
        self.model.train()
        for _ in range(self.train_epochs):
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                out = self.model(x)
                loss = F.cross_entropy(out["logits"], y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
            print(f"Epoch {_ + 1} loss: {loss.item()}")

        self.model.eval()

    @torch.no_grad()
    def predict(self, image_batch):
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")

        self.model.eval()
        if isinstance(image_batch, torch.Tensor):
            x = image_batch.float().to(self.device, non_blocking=True)
        elif isinstance(image_batch, list):
            x = torch.stack([self.transform(im.convert("RGB")) for im in image_batch], dim=0).to(
                self.device, non_blocking=True
            )
        else:
            raise TypeError("image_batch must be a Tensor or list of PIL images")

        out = self.model(x)
        logits = out["logits"]
        feat = out["feat"]
        pred_idx = logits.argmax(dim=-1).detach().cpu().tolist()
        predictions = [{"category_id": int(self.idx_to_cat_id[i])} for i in pred_idx]
        return logits, feat, predictions