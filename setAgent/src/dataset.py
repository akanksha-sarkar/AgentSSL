import os
import json
from pathlib import Path

from PIL import Image
import torch
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

        # load the first json file in ann_dir
        ann_files = sorted(self.ann_dir.glob("*.json"))
        if len(ann_files) == 0:
            raise FileNotFoundError(f"No .json files found in {self.ann_dir}")

        ann_path = ann_files[0]

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
    

if __name__ == "__main__": 
    test_dataset = ClassificationDataset(
        img_dir="images_mini",
        ann_dir="k10/annotations/train",
    )
    print(f"Dataset size: {len(test_dataset)}")
    x, y = test_dataset[0]
    print("One sample:", x.shape, y)