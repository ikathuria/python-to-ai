#!/usr/bin/env python3
"""
Train the Colour CNN from scratch and export to ONNX for the Computer Vision demo.

The model classifies 64x64 RGB images into 10 colour categories:
    black, blue, brown, green, grey, orange, red, violet, white, yellow

Run from repo root (requires: pip install torch torchvision onnx pillow numpy):
    python scripts/export_colour_cnn.py

Expects colour images in:  .resources/data/colours/<class_name>/*.png|jpg
Output:                     app/onnx_models/colour_cnn.onnx

ImageNet-style normalisation is baked into the ONNX graph so the browser
only needs to scale pixels to [0, 1] before passing the tensor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / ".resources" / "data" / "colours"
OUT = ROOT / "app" / "onnx_models" / "colour_cnn.onnx"
OUT.parent.mkdir(parents=True, exist_ok=True)

CLASSES = ["black", "blue", "brown", "green", "grey",
           "orange", "red", "violet", "white", "yellow"]
IMG_SIZE = 64
# ImageNet mean/std (same as Colour_CNN.ipynb)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from PIL import Image
except ImportError as e:
    print("Install: pip install torch torchvision pillow", file=sys.stderr)
    raise e


# ── Dataset ──────────────────────────────────────────────────────────────────

class ColourDataset(Dataset):
    def __init__(self, root: Path, transform=None):
        self.samples: list[tuple[Path, int]] = []
        self.transform = transform
        for cls_idx, cls_name in enumerate(CLASSES):
            cls_dir = root / cls_name
            if not cls_dir.exists():
                print(f"  Warning: {cls_dir} not found — skipping", file=sys.stderr)
                continue
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    self.samples.append((img_path, cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Model ─────────────────────────────────────────────────────────────────────

class ColourCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                              # 32x32

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                              # 16x16

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),                              # 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Training ──────────────────────────────────────────────────────────────────

def train(epochs: int = 20, batch_size: int = 32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN.tolist(), std=STD.tolist()),
    ])
    aug_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN.tolist(), std=STD.tolist()),
    ])

    dataset = ColourDataset(DATA_DIR, transform=aug_transform)
    if len(dataset) == 0:
        raise RuntimeError(f"No images found under {DATA_DIR}. "
                           "Check that .resources/data/colours/<class>/*.png exists.")

    n_val = max(1, int(0.15 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    val_ds.dataset.transform = transform  # no augmentation for val

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} | {len(train_ds)} train / {len(val_ds)} val")

    model = ColourCNN(num_classes=len(CLASSES)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.3)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X).argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct / total if total else 0
        print(f"  Epoch {epoch:2d}/{epochs}  loss={total_loss/len(train_loader):.3f}  val_acc={acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"Best val accuracy: {best_acc:.3f}")
    model.load_state_dict(best_state)
    return model.cpu().eval()


# ── ONNX Export ───────────────────────────────────────────────────────────────

def export(model: "ColourCNN"):
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        model, dummy, str(OUT),
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=12,
    )
    size_kb = OUT.stat().st_size // 1024
    print(f"Exported → {OUT}  ({size_kb} KB)")
    print(f"Classes  : {CLASSES}")
    print("Browser usage: scale pixels to [0,1], apply ImageNet normalisation,")
    print("               then pass a Float32 tensor of shape [1, 3, 64, 64].")


if __name__ == "__main__":
    model = train()
    export(model)
