from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms



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


def emnist_preprocess(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    img = img.convert("L")

    img = transforms.functional.resize(img, (28, 28))
    img = transforms.functional.to_tensor(img)
    img = transforms.functional.normalize(img, (0.5,), (0.5,))
    return img
#"""






def split_by_projection(char_img):
    projection = np.sum(char_img, axis=0)

    max_val = np.max(projection)
    if max_val > 0:
        projection = projection / max_val

    splits = []
    threshold = 0.2

    in_gap = False
    start = 0

    for i, val in enumerate(projection):
        if val < threshold and not in_gap:
            in_gap = True
            start = i
        elif val >= threshold and in_gap:
            in_gap = False
            end = i
            if end - start > 2:
                splits.append((start, end))

    cut_points = [(s + e) // 2 for (s, e) in splits]

    segments = []
    prev = 0
    for cp in cut_points:
        segments.append(char_img[:, prev:cp])
        prev = cp
    segments.append(char_img[:, prev:])

    return segments

def split_with_watershed(binary_img):
    dist = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)

    # find peaks (centers of characters)
    _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # background
    sure_bg = cv2.dilate(binary_img, np.ones((3,3), np.uint8), iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # markers
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # watershed
    color = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color, markers)

    segments = []
    for label in range(2, np.max(markers) + 1):
        mask = np.uint8(markers == label) * 255
        x, y, w, h = cv2.boundingRect(mask)
        segments.append(mask[y:y+h, x:x+w])

    return segments

def merge_boxes(boxes, overlap_thresh=0.3):
    if not boxes:
        return []

    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = x1 + boxes[:, 2]
    y2 = y1 + boxes[:, 3]

    areas = boxes[:, 2] * boxes[:, 3]
    order = np.argsort(x1)

    merged = []
    used = set()

    for i in order:
        if i in used:
            continue

        cur = [x1[i], y1[i], x2[i], y2[i]]
        used.add(i)

        for j in order:
            if j in used:
                continue

            xx1 = max(cur[0], x1[j])
            yy1 = max(cur[1], y1[j])
            xx2 = min(cur[2], x2[j])
            yy2 = min(cur[3], y2[j])

            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            overlap = (w * h) / areas[j]

            if overlap > overlap_thresh:
                # merge
                cur[0] = min(cur[0], x1[j])
                cur[1] = min(cur[1], y1[j])
                cur[2] = max(cur[2], x2[j])
                cur[3] = max(cur[3], y2[j])
                used.add(j)

        merged.append((cur[0], cur[1], cur[2] - cur[0], cur[3] - cur[1]))

    return merged


def center_to_28(img, padding=4):
    h, w = img.shape

    if h == 0 or w == 0:
        return np.zeros((28, 28), dtype=np.uint8)

    # Available drawing area
    inner_size = 28 - 2 * padding
    if inner_size <= 0:
        return np.zeros((28, 28), dtype=np.uint8)

    # Scale while preserving aspect ratio
    scale = inner_size / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(img, (new_w, new_h))

    # Create blank canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)

    # Center the character
    y_offset = padding + (inner_size - new_h) // 2
    x_offset = padding + (inner_size - new_w) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    return canvas

def show_grid(images, cols=8):
    if len(images) == 0:
        return

    h, w = images[0].shape
    rows = (len(images) + cols - 1) // cols

    grid = np.zeros((rows * h, cols * w), dtype=np.uint8)

    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = img

    cv2.imshow("64x64 chars", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def segment_characters(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # binarize
    _, thresh = cv2.threshold(
        img, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # light cleanup
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # find contours
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # filter noise
        if w < 5 or h < 5:
            continue

        boxes.append((x, y, w, h))

    # merge overlapping boxes
    boxes = merge_boxes(boxes)

    widths = [w for (_, _, w, _) in boxes]
    avg_width = np.median(widths) if widths else 0

    multi_boxes = []
    single_boxes = []

    for b in boxes:
        if b[2] > 1.5 * avg_width:  # heuristic
            multi_boxes.append(b)
        else:
            single_boxes.append(b)

    # sort left-to-right
    boxes = sorted(boxes, key=lambda b: b[0])

    chars = []
    debug = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for (x, y, w, h) in boxes:
        char_img = thresh[y:y + h, x:x + w]

        if w > 1.5 * avg_width:
            # try splitting
            pieces = split_by_projection(char_img)

            # fallback if projection fails
            if len(pieces) <= 1:
                pieces = split_with_watershed(char_img)

            for p in pieces:
                if p.shape[1] > 3 and p.shape[0] > 5:
                    chars.append(center_to_28(p, 5))
        else:
            chars.append(center_to_28(char_img, 5))

        # draw debug boxes
        cv2.rectangle(debug, (x, y), (x+w, y+h), (0, 255, 0), 1)

    cv2.imshow("segments", debug)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    show_grid(chars)

    return chars



def predict_line(image_path, model, idx_to_class, transform, device):

    chars = segment_characters(image_path)
    assert len(chars) > 0, "Segmentation failed — no characters found."

    batch = torch.stack([
        transform(c) for c in chars
    ]).to(device)

    with torch.no_grad():
        outputs = model(batch)
        outputs = outputs[:, 10:]
        preds = torch.argmax(outputs, dim=1) + 10
        probs = torch.softmax(outputs, dim=1)
        confs, _ = torch.max(probs, dim=1)

    result = ""
    confidences = []

    for i in range(len(chars)):
        char = idx_to_class[preds[i].item()]
        confidence = confs[i].item()
        confidences.append(confidence)

        print(f"{char} ({confidence:.2f})")

        if confidence < 0.7:
            char = "?"

        # add to string
        result += char

    return result, confidences

test_transform = emnist_preprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("emnist_finetuned.pth", map_location=device)


classes = checkpoint["classes"]
idx_to_class = {i: c for i, c in enumerate(classes)}


model = CNN(num_classes=len(classes))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

text, accuracies = predict_line(
    "dataset/test_line.png",
    model,
    idx_to_class,
    test_transform,
    device
)

accuracy_avg = np.mean(accuracies)

print("Recognized:", text)
print("Average accuracy = ", accuracy_avg)