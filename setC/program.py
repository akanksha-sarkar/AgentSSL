"""
Simple classification: train on train_dataset, return COCO-format results for val_dataset.
Uses dataset.py only. Minimal, for testing.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

from src.dataset import collate_fn, collate_unlabeled


def classify(train_dataset, val_dataset, unlabelled_dataset):
    """
    Train on train_dataset, return a list of COCO format classification results
    for each image in the val_dataset. Uses dataset.py only.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # dataset.py exposes .coco (no get_cat_ids)
    cat_ids = sorted(train_dataset.coco.getCatIds())
    num_classes = len(cat_ids)
    if num_classes == 0:
        return []

    cat_id_to_idx = {c: i for i, c in enumerate(cat_ids)}
    idx_to_cat_id = {i: c for c, i in cat_id_to_idx.items()}

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset.transform = trans
    val_dataset.transform = trans

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=4,
        pin_memory=(device.type == "cuda"),  # faster CPU->GPU transfer
    )

    unlabelled_loader = DataLoader(
        unlabelled_dataset, batch_size=64, shuffle=True, num_workers=0, collate_fn=collate_unlabeled,
    )

    # Val: same loader as train. image_id list matches dataset order (shuffle=False).
    val_image_ids = [val_dataset.file_to_coco_id[f] for f in val_dataset.img_ids]
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False, collate_fn=collate_unlabeled, num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Train (minimal, for testing)
    model.train()
    for epoch in range(3):
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            targets = torch.tensor(
                [cat_id_to_idx[l.item()] for l in labels], dtype=torch.long, device=device
            )
            optimizer.zero_grad()
            criterion(model(images), targets).backward()
            optimizer.step()

    # Predict on val, build COCO results (pycocotools loadRes expects each record to have "id")
    model.eval()
    coco_results = []
    res_id = 1
    idx = 0
    with torch.no_grad():
        for images, _metas in val_loader:
            preds = model(images.to(device, non_blocking=True)).argmax(dim=1)
            n = images.size(0)
            for i in range(n):
                coco_results.append({
                    "id": res_id,
                    "image_id": val_image_ids[idx + i],
                    "category_id": idx_to_cat_id[preds[i].item()],
                })
                res_id += 1
            idx += n

    return coco_results
