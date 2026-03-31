import json
from pathlib import Path

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class TrainClassificationDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = Path(img_dir)
        self.transform = transform

        with open(ann_file, "r") as f:
            coco = json.load(f)

        images = coco.get("images", [])
        annotations = coco.get("annotations", [])
        categories = coco.get("categories", [])

        self.id_to_file = {img["id"]: img["file_name"] for img in images}

        cat_ids = sorted(cat["id"] for cat in categories)
        if not cat_ids:
            cat_ids = sorted({ann["category_id"] for ann in annotations})

        self.cat_id_to_idx = {cat_id: i for i, cat_id in enumerate(cat_ids)}
        self.idx_to_cat_id = {i: cat_id for cat_id, i in self.cat_id_to_idx.items()}

        self.samples = []
        for ann in annotations:
            image_id = ann["image_id"]
            category_id = ann["category_id"]
            if image_id in self.id_to_file and category_id in self.cat_id_to_idx:
                self.samples.append({
                    "file_name": self.id_to_file[image_id],
                    "label": self.cat_id_to_idx[category_id],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = self.img_dir / sample["file_name"]

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, sample["label"]


class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


class ClassificationAgent:
    def __init__(self):
        self.image_size = 64
        self.batch_size = 32
        self.num_epochs = 5
        self.lr = 1e-3

        self.device = "cpu"
        self.model = None
        self.idx_to_cat_id = None

        self.train_transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
        ])

        self.test_transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
        ])

    def fit(self, train_img_dir, train_ann_file):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        train_dataset = TrainClassificationDataset(
            train_img_dir,
            train_ann_file,
            transform=self.train_transform,
        )

        num_classes = len(train_dataset.idx_to_cat_id)
        self.idx_to_cat_id = train_dataset.idx_to_cat_id

        self.model = SmallCNN(num_classes=num_classes).to(self.device)

        loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for _ in range(self.num_epochs):
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                logits = self.model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

    def predict(self, image):
        """
        Predict a label for a single PIL image.

        Input:
            image: PIL.Image.Image

        Returns:
            {
                "category_id": int,
                "score": float,
            }
        """
        self.model.eval()

        if image.mode != "RGB":
            image = image.convert("RGB")

        x = self.test_transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            score, pred_idx = probs.max(dim=1)

        category_id = self.idx_to_cat_id[int(pred_idx.item())]

        return {
            "category_id": int(category_id),
            "score": float(score.item()),
        }