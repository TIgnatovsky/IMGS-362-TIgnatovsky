import os
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools



emnist_byclass_map = {

    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E',
    15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O',
    25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',

    36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e',
    41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j',
    46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o',
    51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't',
    56: 'u', 57: 'v', 58: 'w', 59: 'x', 60: 'y',
    61: 'z'
}

char_to_emnist = {v: k for k, v in emnist_byclass_map.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

used_classes = sorted(emnist_byclass_map.keys())

old_to_new = {old: i for i, old in enumerate(used_classes)}
new_to_old = {i: old for old, i in old_to_new.items()}

num_classes = len(used_classes)


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
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)



train_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomRotation(5),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.Resize((28, 28)),
    transforms.CenterCrop(28),
    transforms.Lambda(lambda x: F.invert(x)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])



class HandwrittenDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        df = pd.read_csv(csv_file)

        valid_rows = []
        self.labels = []

        for _, row in df.iterrows():
            char = chr(int(row["label_ascii"]))
            if char in char_to_emnist:
                valid_rows.append(row)
                self.labels.append(char_to_emnist[char])

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.img_dir, row["filename"])
        img = Image.open(img_path).convert("L")

        char = chr(int(row["label_ascii"]))
        old_label = char_to_emnist[char]
        label = old_to_new[old_label]

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=6)
    plt.yticks(tick_marks, class_names, fontsize=6)

    thresh = cm.max() / 2.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] > 0:
            plt.text(
                j, i, str(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=6
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()


dataset = HandwrittenDataset(
    csv_file="dataset/labels.csv",
    img_dir="dataset/images",
    transform=train_transform
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

checkpoint = torch.load("emnist_model.pth", map_location=device)
model = CNN(num_classes=num_classes).to(device)

state_dict = checkpoint["model_state_dict"]

# remove classifier layer weights
state_dict = {k: v for k, v in state_dict.items() if "classifier" not in k}


model.load_state_dict(state_dict, strict=False)


# Freeze first two convolutional layers
# Freeze Block 1 completely
for param in model.features[0:7].parameters():
    param.requires_grad = False

for param in model.features[7:].parameters():
    param.requires_grad = True


all_labels = []

for _, labels in train_loader:
    all_labels.extend(labels.tolist())

counts = Counter(all_labels)

weights = torch.zeros(num_classes)

for cls, count in counts.items():
    weights[cls] = 1.0 / max(count, 1)

weights = weights / weights.sum()
weights = weights.to(device)

criterion = nn.CrossEntropyLoss(weight=weights)
val_criterion = nn.CrossEntropyLoss()



optimizer = torch.optim.Adam([
    {"params": model.features[7:].parameters(), "lr": 2e-4},
    {"params": model.classifier.parameters(), "lr": 5e-4}
])



train_losses = []
val_losses = []
val_accuracies = []

epochs = 35
print("Fine-tuning")

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    correct, total = 0, 0
    val_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            # 🔹 use val_criterion here
            loss = val_criterion(outputs, labels)
            val_loss += loss.item()

            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    val_acc = correct / total
    val_accuracies.append(val_acc)

    print(
        f"Epoch {epoch + 1} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Val Acc: {val_acc:.3f}"
    )

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
class_names = [emnist_byclass_map[new_to_old[i]] for i in range(num_classes)]
plot_confusion_matrix(cm, class_names)


plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

# Accuracy curve
plt.figure()
plt.plot(val_accuracies)
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()

plt.show()


idx_to_class = {i: emnist_byclass_map[new_to_old[i]] for i in range(num_classes)}

torch.save({
    "model_state_dict": model.state_dict(),
    "classes": checkpoint["classes"],
    "idx_to_class": idx_to_class,
    "img_size": 28
}, "emnist_finetuned.pth")

print("Saved model → emnist_finetuned.pth")