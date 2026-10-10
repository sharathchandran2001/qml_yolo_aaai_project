"""
data_pipeline.py - crop extraction, caching and splitting.

Crops come from data/images/{train,val}/ (YOLO layout). Each crop is converted to a
32x32 grayscale vector (1024-D, values in [0,1]) exactly as in the original project.

crop sources
  labels : ground-truth boxes + classes from the YOLO label .txt files   (default, recommended)
  yolo   : boxes AND classes predicted by the trained detector, falling back to the label
           file when the detector finds nothing. NOTE: with this source the "label" is the
           detector's own prediction, so a classifier is partly learning the detector.

Binary mapping: the LESS frequent class overall becomes label 1 ("positive/minority").
"""
import os
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def label_path_for(img_path: Path):
    parts = list(img_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            cand = Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")
            if cand.exists():
                return cand
    cand = img_path.with_suffix(".txt")
    return cand if cand.exists() else None


def crop_to_vector(crop, size=(32, 32)):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return (cv2.resize(gray, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0).ravel()


def gt_boxes(img_path: Path, w, h):
    lp = label_path_for(img_path)
    boxes = []
    if lp is None:
        return boxes
    for line in lp.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cls = int(p[0]); cx, cy, bw, bh = map(float, p[1:5])
        x1, y1 = max(0, int((cx - bw / 2) * w)), max(0, int((cy - bh / 2) * h))
        x2, y2 = min(w, int((cx + bw / 2) * w)), min(h, int((cy + bh / 2) * h))
        boxes.append((cls, x1, y1, x2, y2))
    return boxes


def list_images(images_root: Path):
    found = []
    for sub in ("train", "val"):
        d = images_root / sub
        if d.exists():
            for p in sorted(d.rglob("*")):
                if p.suffix.lower() in IMG_EXTS:
                    found.append((sub, p))
    if not found and images_root.exists():
        for p in sorted(images_root.rglob("*")):
            if p.suffix.lower() in IMG_EXTS and "processed" not in p.parts:
                found.append(("all", p))
    return found


def extract_dataset(images_root, source="labels", weights=None, size=(32, 32), conf=0.25):
    images_root = Path(images_root)
    items = list_images(images_root)
    if not items:
        raise FileNotFoundError(f"No images found under {images_root}")
    detector = None
    if source == "yolo":
        from ultralytics import YOLO  # lazy import: not needed for source=labels
        detector = YOLO(str(weights))
    X, y, groups, split = [], [], [], []
    for sp, path in items:
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = []
        if detector is not None:
            for r in detector(img, verbose=False, conf=conf):
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    boxes.append((int(b.cls[0]), x1, y1, x2, y2))
        if not boxes:
            boxes = gt_boxes(path, w, h)
        for cls, x1, y1, x2, y2 in boxes:
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            X.append(crop_to_vector(crop, size)); y.append(cls)
            groups.append(f"{sp}/{path.stem}"); split.append(sp)
    if not X:
        raise RuntimeError("No crops extracted - check that label .txt files exist next to images.")
    return {"X": np.asarray(X, np.float32), "y": np.asarray(y, int),
            "groups": np.asarray(groups), "split": np.asarray(split)}


def load_or_build(cache_path, builder, refresh=False):
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        z = np.load(cache_path, allow_pickle=False)
        print(f"[+] Loaded cached crops: {cache_path} ({len(z['y'])} crops)")
        return {k: z[k] for k in z.files}
    data = builder()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **data)
    print(f"[+] Extracted {len(data['y'])} crops from {len(set(data['groups']))} images -> cached at {cache_path}")
    return data


def to_binary(y):
    """Minority class -> 1. Returns (y_bin, info dict)."""
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        raise ValueError(f"Expected a binary problem, found classes {classes.tolist()}")
    pos = classes[np.argmin(counts)]
    return (y == pos).astype(int), {"positive_class_id": int(pos),
                                    "class_counts": {int(c): int(n) for c, n in zip(classes, counts)}}


def synthetic_dataset(n_images=40, seed=0):
    """SMOKE-TEST ONLY: random 1024-D data with a weak low-rank class signal. Never use for the paper."""
    rng = np.random.default_rng(seed)
    basis = rng.normal(size=(8, 1024))
    X, y, g, s = [], [], [], []
    for i in range(n_images):
        sp = "val" if i >= int(n_images * 0.75) else "train"
        labels = [0] + [1] * 8
        for lab in labels:
            z = rng.normal(size=8) + (1.5 if lab == 0 else -1.5) * np.eye(8)[0]
            X.append(np.clip(0.5 + 0.05 * (z @ basis) + 0.05 * rng.normal(size=1024), 0, 1))
            y.append(lab); g.append(f"{sp}/img{i:03d}"); s.append(sp)
    return {"X": np.asarray(X, np.float32), "y": np.asarray(y, int),
            "groups": np.asarray(g), "split": np.asarray(s)}
