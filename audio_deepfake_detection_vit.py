import os
import random
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import torchaudio
import torchvision.transforms as T
import timm

import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

from tqdm import tqdm

# ============================================================
# 1. Configuration
# ============================================================

# 🔧 Set these to your dataset paths
REAL_DIR = "data/real"   # folder containing real .wav files
FAKE_DIR = "data/fake"   # folder containing fake .wav files

SAMPLE_RATE = 16000
N_MELS = 128
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-4
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ============================================================
# 2. Reproducibility
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ============================================================
# 3. Dataset: Audio → Mel-Spectrogram → 3-channel Image
# ============================================================

class AudioDeepfakeDataset(Dataset):
    """
    Expects two folders:
        REAL_DIR: path to real wav files
        FAKE_DIR: path to fake wav files

    Label mapping:
        real -> 0
        fake -> 1
    """

    def __init__(self, real_dir: str, fake_dir: str, transform=None):
        self.samples: List[str] = []
        self.labels: List[int] = []
        self.transform = transform

        # Collect real audio paths
        self._collect_files(real_dir, label=0)
        # Collect fake audio paths
        self._collect_files(fake_dir, label=1)

        print(f"Loaded {len(self.samples)} total audio files "
              f"({sum(np.array(self.labels) == 0)} real, "
              f"{sum(np.array(self.labels) == 1)} fake)")

        # Audio transforms (fixed)
        self.resampler = torchaudio.transforms.Resample(orig_freq=SAMPLE_RATE, new_freq=SAMPLE_RATE)
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_mels=N_MELS
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

        # Image transforms (resize + optional normalization)
        self.to_image = T.Compose([
            T.Resize((224, 224)),  # ViT expects 224x224
            # Optional: normalize like ImageNet
            # T.Normalize(mean=[0.485, 0.456, 0.406],
            #             std=[0.229, 0.224, 0.225]),
        ])

    def _collect_files(self, root_dir: str, label: int):
        if not os.path.exists(root_dir):
            print(f"⚠️ Warning: directory not found: {root_dir}")
            return
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(".wav"):
                    self.samples.append(os.path.join(root, f))
                    self.labels.append(label)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_waveform(self, path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(path)  # shape: (channels, time)
        # Convert to mono by averaging channels if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample to uniform SAMPLE_RATE
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
        return waveform

    def _waveform_to_mel_image(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: (1, time)
        returns: (3, 224, 224) tensor
        """
        mel = self.melspec(waveform)                 # (1, n_mels, time)
        mel_db = self.db_transform(mel)              # log scale

        # Normalize to [0, 1] for visualization / stability
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-9)

        # Resize to (224, 224)
        mel_resized = self.to_image(mel_db)          # still 1-channel
        # Repeat to 3 channels for ViT
        mel_3ch = mel_resized.repeat(3, 1, 1)        # (3, 224, 224)
        return mel_3ch

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.samples[idx]
        label = self.labels[idx]

        waveform = self._load_waveform(path)
        mel_img = self._waveform_to_mel_image(waveform)

        if self.transform:
            mel_img = self.transform(mel_img)

        return mel_img, torch.tensor(label, dtype=torch.long)

# ============================================================
# 4. Data Splits & Dataloaders
# ============================================================

def create_dataloaders(real_dir: str, fake_dir: str):
    dataset = AudioDeepfakeDataset(real_dir, fake_dir)

    if len(dataset) == 0:
        raise RuntimeError("No audio files found. Please check REAL_DIR and FAKE_DIR paths.")

    total_len = len(dataset)
    test_len = int(TEST_SPLIT * total_len)
    val_len = int(VAL_SPLIT * total_len)
    train_len = total_len - val_len - test_len

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(SEED)
    )

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader

# ============================================================
# 5. Model: ViT (Vision Transformer)
# ============================================================

def create_model(num_classes: int = 2) -> nn.Module:
    """
    Uses timm's ViT base model (224x224 input).
    """
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=True,
        num_classes=num_classes
    )
    return model

# ============================================================
# 6. Training & Evaluation Loops
# ============================================================

def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            outputs = model(X)
            loss = criterion(outputs, y)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / len(loader)
    acc = 100.0 * correct / total
    return avg_loss, acc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-4,
    save_best_path: str = "best_model.pth"
):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for X, y in loop:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_best_path)
            print(f"✅ New best model saved with Val Acc = {best_val_acc:.2f}%")

    return history, criterion

# ============================================================
# 7. Test Evaluation: ROC, AUC, Confusion Matrix
# ============================================================

def evaluate_on_test(
    model: nn.Module,
    test_loader: DataLoader,
    criterion,
    device: torch.device
):
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for X, y in tqdm(test_loader, desc="Testing"):
            X = X.to(device)
            y = y.to(device)

            outputs = model(X)
            loss = criterion(outputs, y)

            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            labels = y.cpu().numpy()

            y_prob.extend(probs)
            y_pred.extend(preds)
            y_true.extend(labels)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Real vs Fake)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return roc_auc

# ============================================================
# 8. Single Sample Inference & Visualization
# ============================================================

def visualize_random_test_sample(model, test_loader, device: torch.device):
    # Collect all samples from test loader into a list
    all_samples = []
    all_labels = []
    for X, y in test_loader:
        all_samples.append(X)
        all_labels.append(y)

    X_all = torch.cat(all_samples, dim=0)
    y_all = torch.cat(all_labels, dim=0)

    idx = random.randint(0, len(X_all) - 1)
    sample = X_all[idx:idx + 1].to(device)  # shape (1,3,224,224)
    label = y_all[idx].item()

    model.eval()
    with torch.no_grad():
        outputs = model(sample)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_label = int(np.argmax(probs))

    label_str = "Real" if label == 0 else "Fake"
    pred_str = "Real" if pred_label == 0 else "Fake"
    confidence = probs[pred_label] * 100.0

    print(f"True Label:      {label_str}")
    print(f"Predicted Label: {pred_str}")
    print(f"Confidence:      {confidence:.2f}%")

    # Show spectrogram image
    img = sample[0].cpu().permute(1, 2, 0).numpy()  # (224,224,3)
    plt.imshow(img)
    plt.title(f"Pred: {pred_str} ({confidence:.1f}% conf)")
    plt.axis("off")
    plt.show()

# ============================================================
# 9. Main
# ============================================================

def main():
    # 1) Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(REAL_DIR, FAKE_DIR)

    # 2) Create model
    model = create_model(num_classes=2)

    # 3) Train
    history, criterion = train_model(
        model,
        train_loader,
        val_loader,
        device=DEVICE,
        epochs=EPOCHS,
        lr=LR,
        save_best_path="best_model.pth",
    )

    # 4) Load best model
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    model.to(DEVICE)

    # 5) Evaluate on test set
    roc_auc = evaluate_on_test(model, test_loader, criterion, DEVICE)
    print(f"\n✅ Test ROC-AUC: {roc_auc:.3f}")

    # 6) Visualize one prediction
    visualize_random_test_sample(model, test_loader, DEVICE)


if __name__ == "__main__":
    main()
