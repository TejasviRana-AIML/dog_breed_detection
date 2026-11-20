import os
from pathlib import Path
from collections import defaultdict, Counter

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO


# ---------- PATHS AND SETTINGS ----------
BASE = Path(__file__).resolve().parent
DOWNLOADS = Path.home() / "Downloads"
OUTPUT_DIR = BASE / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

INPUT = DOWNLOADS / "dog6.mp4"
OUTPUT = OUTPUT_DIR / "dog6_local.mp4"

LABELS = BASE / "data" / "dataset" / "labels.csv"
WEIGHTS = BASE / "dog_breed_model_ft.pth"
YOLO_WEIGHTS = BASE / "yolov8m.pt"
TRACKER = BASE / "my_tracker.yaml"

device = torch.device("cpu")

# ---------- BREED CLASSIFIER ----------
df = pd.read_csv(LABELS)
classes = sorted(df["breed"].unique())

norm = transforms.Normalize(
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225]
)

val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    norm
])

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load(WEIGHTS, map_location=device))
model = model.to(device)
model.eval()

# ---------- YOLO DETECTOR ----------
detector = YOLO(str(YOLO_WEIGHTS))

ANIMALS = [15, 16, 17, 18, 19, 20, 21, 22, 23]
DOG = 16


def process_video(INPUT, OUTPUT, MIN_FRAMES=25, DET_CONF=0.4,
                  SURE_CONF=0.35, PAD=0.10, MAX_GAP_SEC=3.0):

    model.eval()
    detector.predictor = None

    # ---------- PASS 1 ----------
    cap = cv2.VideoCapture(str(INPUT))

    if not cap.isOpened():
        raise FileNotFoundError(f"Video not opened: {INPUT}")

    w = int(cap.get(3))
    h = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    prob_sum = {}
    prob_n = defaultdict(int)
    info = {}
    frame_boxes = []

    MIN_SIDE = 40
    n = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        boxes_here = []

        res = detector.track(
            frame,
            persist=True,
            classes=ANIMALS,
            tracker=str(TRACKER),
            iou=0.5,
            agnostic_nms=True,
            verbose=False
        )[0]

        if res.boxes.id is not None:
            for b in res.boxes:
                if b.id is None or float(b.conf) < DET_CONF:
                    continue

                if int(b.cls) != DOG:
                    continue

                tid = int(b.id)

                x1, y1, x2, y2 = map(int, b.xyxy[0])
                x1 = max(x1, 0)
                y1 = max(y1, 0)
                x2 = min(x2, w)
                y2 = min(y2, h)

                bw = x2 - x1
                bh = y2 - y1

                if bw < MIN_SIDE or bh < MIN_SIDE:
                    continue

                boxes_here.append((tid, x1, y1, x2, y2))

                px = int(bw * PAD)
                py = int(bh * PAD)

                crop = frame[
                    max(y1 - py, 0):min(y2 + py, h),
                    max(x1 - px, 0):min(x2 + px, w)
                ]

                img = Image.fromarray(
                    cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                )

                with torch.no_grad():
                    inp = val_tf(img).unsqueeze(0).to(device)
                    p = torch.softmax(model(inp), 1)[0].cpu().numpy()

                if tid not in prob_sum:
                    prob_sum[tid] = np.zeros(len(classes))
                    info[tid] = [
                        n, n,
                        (x1, y1, x2, y2),
                        (x1, y1, x2, y2)
                    ]

                prob_sum[tid] += p
                prob_n[tid] += 1
                info[tid][1] = n
                info[tid][3] = (x1, y1, x2, y2)

        frame_boxes.append(boxes_here)
        n += 1

        if n % 30 == 0:
            print(f"Pass 1: {n}/{total} frames")

    cap.release()

    # ---------- MERGE TRACKS ----------
    parent = {t: t for t in prob_sum}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    cand = [t for t in prob_sum if prob_n[t] >= 5]
    cand.sort(key=lambda t: info[t][0])

    used_pred = set()
    max_gap = int(MAX_GAP_SEC * fps)

    for b in cand:
        bx1, by1, bx2, by2 = info[b][2]
        bc = ((bx1 + bx2) / 2, (by1 + by2) / 2)

        best = None
        best_d = None

        for a in cand:
            if a == b or a in used_pred:
                continue

            if info[a][1] >= info[b][0]:
                continue

            if info[b][0] - info[a][1] > max_gap:
                continue

            if find(a) == find(b):
                continue

            ax1, ay1, ax2, ay2 = info[a][3]
            ac = ((ax1 + ax2) / 2, (ay1 + ay2) / 2)

            d = ((ac[0] - bc[0]) ** 2 + (ac[1] - bc[1]) ** 2) ** 0.5
            size = max(ax2 - ax1, ay2 - ay1)

            if d <= 1.2 * size and (best_d is None or d < best_d):
                best = a
                best_d = d

        if best is not None:
            parent[find(b)] = find(best)
            used_pred.add(best)

    groups = defaultdict(list)

    for t in prob_sum:
        groups[find(t)].append(t)

    # ---------- FINAL BREED PER DOG ----------
    final = {}
    root_of = {}

    for root, tids in groups.items():
        frames = sum(prob_n[t] for t in tids)

        if frames < MIN_FRAMES:
            continue

        s = sum(prob_sum[t] for t in tids)
        mean = s / frames
        idx = int(mean.argmax())

        final[root] = (
            classes[idx].replace("_", " "),
            float(mean[idx])
        )

        for t in tids:
            root_of[t] = root

    counts = Counter(v[0] for v in final.values())

    first_frame = {
        r: min(info[t][0] for t in groups[r])
        for r in final
    }

    order = {
        r: i + 1
        for i, r in enumerate(sorted(final, key=lambda r: first_frame[r]))
    }

    # ---------- PASS 2 ----------
    cap = cv2.VideoCapture(str(INPUT))

    out = cv2.VideoWriter(
        str(OUTPUT),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    if not out.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video")

    scale = max(w, h) / 1000.0
    font_scale = max(0.5, 0.6 * scale)
    thick = max(1, int(1.5 * scale))

    def draw_panel(frame, lines):
        pad = 10
        fs = font_scale * 0.8

        sizes = [
            cv2.getTextSize(
                t, cv2.FONT_HERSHEY_SIMPLEX, fs, thick
            )
            for t in lines
        ]

        pw = max(s[0][0] for s in sizes) + 2 * pad
        lh = max(s[0][1] + s[1] for s in sizes) + 8
        ph = lh * len(lines) + 2 * pad

        cv2.rectangle(
            frame, (10, 10), (10 + pw, 10 + ph), (0, 0, 0), -1
        )

        for i, t in enumerate(lines):
            cv2.putText(
                frame,
                t,
                (10 + pad, 10 + pad + lh * (i + 1) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                fs,
                (255, 255, 255),
                thick
            )

    i = 0

    while True:
        ok, frame = cap.read()

        if not ok or i >= len(frame_boxes):
            break

        in_frame = 0

        for tid, x1, y1, x2, y2 in frame_boxes[i]:
            if tid not in root_of:
                continue

            root = root_of[tid]
            name, conf = final[root]
            in_frame += 1

            if conf >= SURE_CONF:
                label = f"Dog {order[root]}: {name} {conf*100:.0f}%"
                color = (0, 255, 0)
            else:
                label = f"Dog {order[root]}: {name}? {conf*100:.0f}%"
                color = (0, 200, 255)

            cv2.rectangle(
                frame, (x1, y1), (x2, y2), color, thick + 1
            )

            (tw, th), bl = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick
            )

            ty = max(y1, th + bl + 6)

            cv2.rectangle(
                frame,
                (x1, ty - th - bl - 6),
                (x1 + tw + 10, ty),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                frame,
                label,
                (x1 + 5, ty - bl - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thick
            )

        lines = [
            f"Dogs in frame: {in_frame}",
            f"Total dogs: {len(final)}"
        ]

        for breed, c in counts.most_common():
            lines.append(f"{breed}: {c}")

        draw_panel(frame, lines)
        out.write(frame)

        i += 1

        if i % 30 == 0:
            print(f"Pass 2: {i}/{total} frames")

    cap.release()
    out.release()

    print("\nProcessing complete!")
    print("Total dogs:", len(final))
    print("Final breed count:", dict(counts))
    print("Output saved at:", OUTPUT)


if __name__ == "__main__":
    print("Device:", device)
    print("Breed classes:", len(classes))
    print("Loading YOLOv8m and processing first test video...")
    process_video(INPUT, OUTPUT)
