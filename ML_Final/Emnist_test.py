import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import EMNIST
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt


# -------------------------
# EMNIST Fix (correct orientation)
# -------------------------
class EMNISTFix:
    def __call__(self, img):
        img = F.rotate(img, -90)
        img = F.hflip(img)
        return img


# -------------------------
# Transforms
# -------------------------
train_transform = transforms.Compose([
    EMNISTFix(),
    transforms.RandomAffine(
        degrees=5,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1)
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    EMNISTFix(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# -------------------------
# Model
# -------------------------
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def plot_losses(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")

    plt.xlabel("Epoch (iterations over dataset)")
    plt.ylabel("Loss (Cross-Entropy)")
    plt.title("Training and Validation Loss Over Time")

    plt.legend()
    plt.grid(True)
    plt.show()

# -------------------------
# Load EMNIST
# -------------------------
train_dataset = EMNIST(
    root="./data",
    split="byclass",
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = EMNIST(
    root="./data",
    split="byclass",
    train=False,
    download=True,
    transform=test_transform
)


# -------------------------
# Split train/val
# -------------------------
train_size = int(0.8 * len(train_dataset))
print(train_size)
val_size = len(train_dataset) - train_size
print(val_size)

generator = torch.Generator().manual_seed(42)
train_ds, val_ds = random_split(
    train_dataset,
    [train_size, val_size],
    generator=generator
)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)


# -------------------------
# Setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = len(train_dataset.classes)
model = CNN(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=2, factor=0.3
)



# EMNIST label mapping
idx_to_class = {i: train_dataset.classes[i] for i in range(num_classes)}


# -------------------------
# Training
# -------------------------

train_losses = []
val_losses = []

epochs = 10
print("Training")

for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item()

    # validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    scheduler.step(avg_val_loss)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    print(
        f"Epoch {epoch + 1} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Val Acc: {correct / total:.3f}"
    )

# -------------------------
# Save model
# -------------------------
save_path = "emnist_model.pth"

torch.save({
    "model_state_dict": model.state_dict(),
    "classes": train_dataset.classes,
    "model_config": {
        "num_classes": num_classes
    }
}, save_path)

plot_losses(train_losses, val_losses)

print(f"Model saved to {save_path}")