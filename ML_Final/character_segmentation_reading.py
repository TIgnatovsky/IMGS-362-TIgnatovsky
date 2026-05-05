import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy import ndimage
from torchvision import transforms
import torchvision.transforms.functional as F

# config

DEBUG = True

# model load
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


# preprocess and helpers



def emnist_preprocess(img, debug=False):



    if isinstance(img, Image.Image):
        img = np.array(img)

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # light smoothing (closer to EMNIST look)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)


    # soft threshold only for locating the character
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = cv2.findNonZero(thresh)
    if coords is None:
        raise ValueError("No character found")

    x, y, w, h = cv2.boundingRect(coords)
    cropped = gray[y:y+h, x:x+w]


    h, w = cropped.shape

    if h > w:
        new_h = 20
        new_w = int(round(w * (20.0 / h)))
    else:
        new_w = 20
        new_h = int(round(h * (20.0 / w)))

    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)


    canvas = np.zeros((28, 28), dtype=np.uint8)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized


    cy, cx = ndimage.center_of_mass(canvas)

    shiftx = int(np.round(14 - cx))
    shifty = int(np.round(14 - cy))

    canvas = np.roll(canvas, shiftx, axis=1)
    canvas = np.roll(canvas, shifty, axis=0)


    img = Image.fromarray(canvas)


    img = F.rotate(img, -90)
    img = F.hflip(img)


    img = F.to_tensor(img)
    img = F.normalize(img, (0.5,), (0.5,))


    if debug:
        cv2.imshow("Final EMNIST Input", canvas)
        cv2.waitKey(0)

    return img

def show_grid(images, cols=8):

    if len(images) == 0:
        return

    norm_imgs = []

    # convert tensors → numpy if needed
    for img in images:
        if isinstance(img, torch.Tensor):
            img = img.squeeze().cpu().numpy()

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        norm_imgs.append(img)

    # find max height and width
    max_h = max(img.shape[0] for img in norm_imgs)
    max_w = max(img.shape[1] for img in norm_imgs)

    rows = (len(norm_imgs) + cols - 1) // cols

    grid = np.zeros((rows * max_h, cols * max_w), dtype=np.uint8)

    for idx, img in enumerate(norm_imgs):
        r = idx // cols
        c = idx % cols

        h, w = img.shape

        # center image in its cell
        y_offset = (max_h - h) // 2
        x_offset = (max_w - w) // 2

        grid[
            r * max_h + y_offset : r * max_h + y_offset + h,
            c * max_w + x_offset : c * max_w + x_offset + w
        ] = img

    cv2.imshow("Grid", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def binarize(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary
def merge_boxes(boxes, overlap_thresh=0.3):
    if not boxes:
        return []

    boxes = np.array(boxes)
    x1, y1, w, h = boxes.T
    x2 = x1 + w
    y2 = y1 + h

    areas = w * h
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

            w_ = max(0, xx2 - xx1)
            h_ = max(0, yy2 - yy1)
            overlap = (w_ * h_) / areas[j]

            if overlap > overlap_thresh:
                cur[0] = min(cur[0], x1[j])
                cur[1] = min(cur[1], y1[j])
                cur[2] = max(cur[2], x2[j])
                cur[3] = max(cur[3], y2[j])
                used.add(j)

        merged.append((cur[0], cur[1], cur[2]-cur[0], cur[3]-cur[1]))

    return merged


def word_seg(image, padding=0):

    # input = cv2 image
    binary = binarize(image)

    # horizontal + vertical projections
    h_proj = np.sum(binary > 0, axis=0)
    v_proj = np.sum(binary > 0, axis=1)


    y_min = 0
    y_max = binary.shape[0] - 1

    # find first non-empty row
    for i, val in enumerate(v_proj):
        if val > 0:
            y_min = i - padding
            break

    # find last non-empty row
    for i, val in enumerate(v_proj[::-1]):
        if val > 0:
            y_max = binary.shape[0] - 1 - i + padding
            break


    whitespace_runs = []
    run = 0

    for col in h_proj:
        if col == 0:
            run += 1
        else:
            if run > 0:
                whitespace_runs.append(run)
                run = 0

    # catch trailing whitespace
    if run > 0:
        whitespace_runs.append(run)

    avg_whitespace = np.mean(whitespace_runs) if whitespace_runs else 0

    threshold = 1.1 * avg_whitespace


    word_imgs = []
    h, w = binary.shape

    in_word = False
    start_x = 0
    whitespace_count = 0

    for x in range(w):
        if h_proj[x] == 0:
            whitespace_count += 1
        else:
            if not in_word:
                start_x = x
                in_word = True
            whitespace_count = 0

        # boundary condition: large whitespace ends word
        if in_word and whitespace_count > threshold:
            end_x = x - whitespace_count

            if end_x > start_x:
                crop = image[y_min:y_max, (start_x - padding):(end_x + padding)]
                word_imgs.append(crop)

            in_word = False
            whitespace_count = 0

    # catch last word
    if in_word:
        word_imgs.append(image[y_min:y_max, start_x:w])

    if DEBUG == True:
        show_grid(word_imgs)

    return word_imgs

def draw_segmentation_debug(word, runs):

    vis = cv2.cvtColor(word.copy(), cv2.COLOR_GRAY2BGR)

    for val, start, end in runs:
        color = (0, 255, 0) if val == 1 else (0, 0, 255)

        # draw boundaries
        cv2.line(vis, (start, 0), (start, vis.shape[0]), color, 1)
        cv2.line(vis, (end, 0), (end, vis.shape[0]), color, 1)

    cv2.imshow("Segmentation Debug", vis)
    cv2.waitKey(0)

def character_segmentation(word, word_img):

    # input: grayscale cv2 image
    binary = binarize(word)
    binary_line = binarize(word_img)

    h_proj = np.sum(binary > 0, axis=0)
    h_proj_line = np.sum(binary_line > 0, axis=0)

    # binarize projection
    binary_graph = np.array([1 if col > 1 else 0 for col in h_proj])
    binary_graph_line = np.array([1 if col > 1 else 0 for col in h_proj_line])

    def get_runs(arr):
        runs = []
        start = 0
        current = arr[0]

        for i in range(1, len(arr)):
            if arr[i] != current:
                runs.append((current, start, i - 1))
                start = i
                current = arr[i]
        runs.append((current, start, len(arr) - 1))
        return runs

    word_runs = get_runs(binary_graph)
    line_runs = get_runs(binary_graph_line)


    total_widths = 0
    n = 0

    for peak, start, end in line_runs:
        if peak == 1:
            width = end - start
            total_widths += width
            n += 1

    if n == 0:
        return []

    avg_width = total_widths / n
    k = 0.7


    filtered_runs = []
    for peak, start, end in word_runs:
        if peak == 1:
            width = end - start
            if width >= k * avg_width:
                filtered_runs.append((peak, start, end))

        else:
            filtered_runs.append((peak, start, end))  # keep gaps


    char_runs = [(start, end) for peak, start, end in filtered_runs if peak == 1]

    if len(char_runs) == 0:
        return []


    boundaries = []

    # left edge
    boundaries.append(char_runs[0][0])

    for i in range(len(char_runs) - 1):
        end_curr = char_runs[i][1]
        start_next = char_runs[i + 1][0]

        mid = end_curr + (start_next - end_curr) // 2
        boundaries.append(mid)

    # right edge
    boundaries.append(char_runs[-1][1])


    char_images = []
    for i in range(len(boundaries) - 1):
        x1 = boundaries[i]
        x2 = boundaries[i + 1]

        char = word[:, x1:x2]
        if char.size > 0:
            char_images.append(char)

    if DEBUG:
        show_grid(char_images)
        draw_segmentation_debug(word, word_runs)

    return char_images

def predict_sentence(words, word_img):

    # input a list of word images

    sentence = ""

    for word in words:

        # find image of words
        chars = character_segmentation(word, word_img)

        if len(chars) == 0:
            print("Empty word")
            return ""



        batch = torch.stack([emnist_preprocess(c, DEBUG) for c in chars]).to(device)

        with torch.no_grad():
            out = model(batch)
            probs = torch.softmax(out, dim=1)
            confs, preds = torch.max(probs, dim=1)


        for i in range(len(chars)):
            char = idx_to_class[preds[i].item()]
            conf = confs[i].item()

            print(f"{char} ({conf:.2f})")

            if conf < 0.6:
                char = "?"

            sentence += char


        # add space after every word
        sentence += " "

    return sentence

if __name__ == "__main__":

    # load image
    img = cv2.imread("dataset/test_line.png", cv2.IMREAD_GRAYSCALE)
    img_inv = cv2.bitwise_not(img)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load("emnist_model.pth", map_location=device)

    classes = checkpoint["classes"]  # already characters
    idx_to_class = {i: c for i, c in enumerate(classes)}

    model = CNN(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # run word segmentation
    words = word_seg(img_inv, 5)
    sentence = predict_sentence(words, img_inv)
    print(f"Recognized text: {sentence}")
    #cv2.imshow("words", words)
    #cv2.waitKey(0)

    # run character segmentation

    # recognize characters, recombine into sentence

    # load model

    # evaluate code segmentation