"""
Evaluation script for RT-DETR-based soybean pod detection with DAC post-processing.

This script performs:
1. Pre-cropping (horizontal crop + HSV-based region extraction)
2. Model inference with density-aware confidence (DAC) filtering
3. mAP evaluation on pre-cropped validation data
4. Count-based metrics (MAE, RMSE, F1, R2)
"""

import argparse
import gc
import glob
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from ultralytics import RTDETR

# ---------------------------------------------------------------------------
# Pre-cropping constants
# ---------------------------------------------------------------------------
HSV_LOW = np.array([15, 55, 100])
HSV_HIGH = np.array([30, 170, 245])

TOP_CUT = 667      # y-coordinate for top horizontal crop
BOTTOM_CUT = 4013  # y-coordinate for bottom horizontal crop

LEFT_MASK_WIDTH = 100
MIN_AREA = 5000
PAD = 50

# ---------------------------------------------------------------------------
# DAC hyperparameters (tuned on validation set)
# ---------------------------------------------------------------------------
OPTIMAL = {
    "T_base": 0.5888,
    "k": 0.3673,
    "sigma": 0.1437,
    "theta_base": 0.6206,
    "lambda_iou": 0.1579,
    "T_min": 0.20,
    "T_max": 0.95,
}


# ===========================================================================
# Density-Aware Confidence (DAC) post-processing
# ===========================================================================

def kde_density_estimate(
    boxes_xywh_norm: np.ndarray,
    bandwidth: float = 0.1437,
    grid_size: int = 50,
) -> np.ndarray:
    """Estimate spatial density of detections via Gaussian KDE on a grid."""
    if len(boxes_xywh_norm) == 0:
        return np.zeros((grid_size, grid_size))

    centers = boxes_xywh_norm[:, :2]
    x_grid = np.linspace(0, 1, grid_size)
    y_grid = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)

    norm_const = 1.0 / (np.sqrt(2.0 * np.pi) * bandwidth)
    density = np.zeros(len(grid_points))
    for cx, cy in centers:
        dist_sq = (grid_points[:, 0] - cx) ** 2 + (grid_points[:, 1] - cy) ** 2
        density += np.exp(-dist_sq / (2 * bandwidth ** 2))

    if len(centers) > 0:
        density *= norm_const / len(centers)

    return density.reshape(grid_size, grid_size)


def query_density_at_boxes(
    density_map: np.ndarray, boxes_xywh_norm: np.ndarray
) -> np.ndarray:
    """Query KDE density values at detection centers."""
    h, w = density_map.shape
    centers = boxes_xywh_norm[:, :2]
    grid_x = np.clip((centers[:, 0] * (w - 1)).astype(int), 0, w - 1)
    grid_y = np.clip((centers[:, 1] * (h - 1)).astype(int), 0, h - 1)
    return density_map[grid_y, grid_x]


def normalize_density(densities: np.ndarray) -> np.ndarray:
    """Normalize densities to [0, 1] using max-normalization."""
    if len(densities) == 0:
        return densities
    max_d = np.max(densities)
    return densities / max(max_d, 1e-8)


def compute_adaptive_thresholds(
    density_norm: np.ndarray,
    T_base: float,
    k: float,
    T_min: float,
    T_max: float,
) -> np.ndarray:
    """Compute confidence threshold for each detection based on local density."""
    thresholds = T_base - k * density_norm
    return np.clip(thresholds, T_min, T_max)


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute intersection-over-union for two boxes in xyxy format."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / max(union, 1e-8)


def density_aware_deduplication(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    density_norm: np.ndarray,
    theta_base: float,
    lambda_iou: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply density-aware NMS with adaptive IoU thresholds."""
    n = len(boxes_xyxy)
    if n == 0:
        return (
            np.array([], dtype=int),
            np.zeros((0, 4)),
            np.array([]),
        )

    iou_thresholds = theta_base - lambda_iou * density_norm
    iou_thresholds = np.clip(iou_thresholds, 0.05, 0.80)

    order = np.argsort(-scores)
    suppressed = np.zeros(n, dtype=bool)
    keep = []

    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        curr_thresh = iou_thresholds[i]
        for j in order:
            if j == i or suppressed[j]:
                continue
            iou = compute_iou(boxes_xyxy[i], boxes_xyxy[j])
            if iou > curr_thresh:
                suppressed[j] = True

    keep = np.array(keep, dtype=int)
    return keep, boxes_xyxy[keep], scores[keep]


def dac_postprocess(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    img_w: int,
    img_h: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Full DAC pipeline: density estimation -> adaptive threshold -> IoU deduplication."""
    n = len(boxes_xyxy)
    if n == 0:
        return np.zeros((0, 4)), np.array([])

    xywh = np.zeros((n, 4))
    xywh[:, 0] = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2
    xywh[:, 1] = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2
    xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
    xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
    xywh[:, [0, 2]] /= img_w
    xywh[:, [1, 3]] /= img_h
    xywh = np.clip(xywh, 0.0, 1.0)

    density_map = kde_density_estimate(xywh, bandwidth=OPTIMAL["sigma"], grid_size=50)
    densities = query_density_at_boxes(density_map, xywh)
    density_norm = normalize_density(densities)

    thresholds = compute_adaptive_thresholds(
        density_norm,
        OPTIMAL["T_base"],
        OPTIMAL["k"],
        OPTIMAL["T_min"],
        OPTIMAL["T_max"],
    )
    keep = scores >= thresholds
    if not np.any(keep):
        return np.zeros((0, 4)), np.array([])

    _, final_boxes, final_scores = density_aware_deduplication(
        boxes_xyxy[keep],
        scores[keep],
        density_norm[keep],
        theta_base=OPTIMAL["theta_base"],
        lambda_iou=OPTIMAL["lambda_iou"],
    )
    return final_boxes, final_scores


# ===========================================================================
# Pre-cropping (horizontal crop + HSV mask)
# ===========================================================================

def get_crop_region(img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop image by removing top/bottom rulers and extracting plant region via HSV masking."""
    h, w = img.shape[:2]

    # Horizontal crop
    y_top = min(TOP_CUT, h)
    y_bot = min(BOTTOM_CUT, h)
    if y_bot <= y_top:
        y_top, y_bot = 0, h
    band = img[y_top:y_bot, :]
    bh = band.shape[0]

    # HSV masking with left-side exclusion
    masked = band.copy()
    cv2.rectangle(masked, (0, 0), (LEFT_MASK_WIDTH, bh), (0, 0, 0), -1)

    hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    plant_mask = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(plant_mask)
    filtered_mask = np.zeros_like(plant_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA:
            filtered_mask[labels == i] = 255

    cols = np.where(filtered_mask.sum(axis=0) > 0)[0]
    rows = np.where(filtered_mask.sum(axis=1) > 0)[0]

    if len(cols) == 0 or len(rows) == 0:
        bx1, by1, bx2, by2 = 0, 0, w, bh
    else:
        bx1 = max(0, cols[0] - PAD)
        by1 = max(0, rows[0] - PAD)
        bx2 = min(w, cols[-1] + PAD)
        by2 = min(bh, rows[-1] + PAD)

    cx1, cy1, cx2, cy2 = bx1, by1 + y_top, bx2, by2 + y_top
    return img[cy1:cy2, cx1:cx2], (cx1, cy1, cx2, cy2)


# ===========================================================================
# Cropped evaluation dataset construction
# ===========================================================================

def build_crop_sample(args: Tuple[str, str, str, str]) -> Optional[str]:
    """Crop a single image and convert labels to the cropped coordinate system."""
    img_path, label_path, out_img_dir, out_label_dir = args
    fname = os.path.splitext(os.path.basename(img_path))[0]

    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]

    crop_img, (cx1, cy1, cx2, cy2) = get_crop_region(img)
    if crop_img.size == 0:
        return None
    ch, cw = crop_img.shape[:2]

    out_img_path = os.path.join(out_img_dir, f"{fname}.jpg")
    cv2.imwrite(out_img_path, crop_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    lines_out = []
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f.read().strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split()
                cls = parts[0]
                cxn, cyn, wn, hn = map(float, parts[1:5])
                x1 = (cxn - wn / 2) * w
                y1 = (cyn - hn / 2) * h
                x2 = (cxn + wn / 2) * w
                y2 = (cyn + hn / 2) * h
                nx1 = max(x1, cx1)
                ny1 = max(y1, cy1)
                nx2 = min(x2, cx2)
                ny2 = min(y2, cy2)
                if nx2 - nx1 < 2 or ny2 - ny1 < 2:
                    continue
                ncx = ((nx1 + nx2) / 2 - cx1) / cw
                ncy = ((ny1 + ny2) / 2 - cy1) / ch
                nw = (nx2 - nx1) / cw
                nh = (ny2 - ny1) / ch
                lines_out.append(f"{cls} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}")

    with open(os.path.join(out_label_dir, f"{fname}.txt"), "w") as f:
        f.write("\n".join(lines_out))

    return out_img_path


def build_crop_eval_dataset(
    tasks: List[Tuple[str, str, str, int]],
    crop_eval_root: str,
    num_workers: int,
) -> Dict[str, List[str]]:
    """Build pre-cropped evaluation dataset in parallel."""
    print("\nBuilding pre-cropped evaluation dataset...")

    build_tasks = []
    for img_path, label_path, group, _ in tasks:
        out_img_dir = os.path.join(crop_eval_root, group, "images")
        out_label_dir = os.path.join(crop_eval_root, group, "labels")
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_label_dir, exist_ok=True)
        build_tasks.append((img_path, label_path, out_img_dir, out_label_dir))

    group_cropped: Dict[str, List[str]] = {}
    done = 0
    with ProcessPoolExecutor(max_workers=num_workers,
                             mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(build_crop_sample, t): t for t in build_tasks}
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                group = os.path.basename(os.path.dirname(os.path.dirname(res)))
                group_cropped.setdefault(group, []).append(res)
            done += 1
            if done % 200 == 0 or done == len(build_tasks):
                print(f"  crop [{done}/{len(build_tasks)}]")

    for g, paths in group_cropped.items():
        paths.sort()
        print(f"  {g}: {len(paths)} cropped images")

    return group_cropped


# ===========================================================================
# Metrics
# ===========================================================================

def get_gt_count(label_path: str) -> int:
    """Count ground-truth objects from a YOLO-format label file."""
    if not os.path.exists(label_path):
        return 0
    with open(label_path) as f:
        return len([l for l in f.read().strip().split("\n") if l.strip()])


def compute_r2(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute coefficient of determination (R2)."""
    ss_res = np.sum((pred - gt) ** 2)
    ss_tot = np.sum((gt - np.mean(gt)) ** 2)
    return 1.0 - ss_res / max(ss_tot, 1e-8)


def compute_f1_from_counts(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1 from count vectors."""
    tp = np.sum(np.minimum(pred, gt))
    fp = np.sum(np.maximum(pred - gt, 0))
    fn = np.sum(np.maximum(gt - pred, 0))
    precision = tp / max(tp + fp, 1e-8)
    recall = tp / max(tp + fn, 1e-8)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return precision, recall, f1


def print_metrics(
    name: str,
    pred_list: List[int],
    gt_list: List[int],
    map50: Optional[float] = None,
    map50_75: Optional[float] = None,
) -> None:
    """Print evaluation metrics in a formatted table."""
    pred = np.array(pred_list)
    gt = np.array(gt_list)
    errors = np.abs(pred - gt)
    n_total = len(errors)

    print(f"\n{'=' * 55}")
    print(f"{name:^55}")
    print(f"{'=' * 55}")

    if map50 is not None:
        print(f"{'mAP@0.5':<20} {map50:>34.4f}")
    if map50_75 is not None:
        print(f"{'mAP@0.5:0.75':<20} {map50_75:>34.4f}")
    if map50 is not None or map50_75 is not None:
        print(f"{'-' * 55}")

    if n_total == 0:
        print("No valid samples")
        return

    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    precision, recall, f1 = compute_f1_from_counts(pred, gt)
    r2 = compute_r2(pred, gt)

    stats = {
        "error=0": int(np.sum(errors == 0)),
        "error<=1": int(np.sum(errors <= 1)),
        "error<=2": int(np.sum(errors <= 2)),
        "error<=3": int(np.sum(errors <= 3)),
        "error<=5": int(np.sum(errors <= 5)),
    }

    print(f"{'Images:':<20} {n_total:>34}")
    print(f"{'-' * 55}")
    print(f"{'MAE':<20} {mae:>34.4f}")
    print(f"{'RMSE':<20} {rmse:>34.4f}")
    print(f"{'Precision (count)':<20} {precision:>34.4f}")
    print(f"{'Recall (count)':<20} {recall:>34.4f}")
    print(f"{'F1':<20} {f1:>34.4f}")
    print(f"{'R^2':<20} {r2:>34.4f}")
    print(f"{'-' * 55}")
    print(f"{'Error Range':<15s} {'Count':>8s} {'Ratio':>8s}")
    print(f"{'-' * 35}")
    for label, count in stats.items():
        print(f"{label:<15s} {count:>8d} {count / n_total * 100:>7.1f}%")
    print(f"{'=' * 55}")


# ===========================================================================
# mAP evaluation (using ultralytics model.val)
# ===========================================================================

def _val_one(
    model: RTDETR,
    val_entry: str,
    path: str,
    imgsz: int,
    val_batch: int,
) -> Tuple[Optional[float], Optional[float]]:
    """Run model.val and return (mAP@0.5, mAP@0.5:0.75)."""
    tmp_yaml = f"/tmp/val_{abs(hash(str(val_entry)))}.yaml"
    with open(tmp_yaml, "w") as f:
        yaml.dump(
            {
                "path": path,
                "train": val_entry,
                "val": val_entry,
                "names": {0: "pod"},
            },
            f,
        )

    results = model.val(
        data=tmp_yaml,
        imgsz=imgsz,
        batch=val_batch,
        device=0,
        workers=0,
        verbose=False,
        save=False,
    )

    map50 = float(results.box.map50) if hasattr(results.box, "map50") else None

    map50_75 = None
    if (
        hasattr(results.box, "all_ap")
        and results.box.all_ap is not None
        and len(results.box.all_ap) > 0
    ):
        all_ap = results.box.all_ap
        map50_75 = float(np.mean(np.mean(all_ap[:, :6], axis=1)))

    del results
    torch.cuda.empty_cache()
    gc.collect()
    return map50, map50_75


def run_map_eval(
    model_path: str,
    group_cropped: Dict[str, List[str]],
    img_dirs: List[str],
    crop_eval_root: str,
    imgsz: int,
    val_batch: int,
) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """Evaluate mAP on pre-cropped datasets per group and overall."""
    print("\nRunning model.val on pre-cropped images...")
    model = RTDETR(model_path)
    maps: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

    for img_dir in img_dirs:
        group = os.path.basename(img_dir.rstrip("/"))
        if group not in group_cropped:
            continue
        maps[group] = _val_one(
            model, "images", os.path.join(crop_eval_root, group), imgsz, val_batch
        )
        m50, m5075 = maps[group]
        print(f"  {group}: mAP@0.5={m50:.4f}, mAP@0.5:0.75={m5075:.4f}")

    # Combined evaluation
    all_cropped = []
    for img_dir in img_dirs:
        group = os.path.basename(img_dir.rstrip("/"))
        all_cropped.extend(group_cropped.get(group, []))
    all_txt = os.path.join(crop_eval_root, "all_images.txt")
    with open(all_txt, "w") as f:
        f.write("\n".join(all_cropped))

    maps["ALL"] = _val_one(model, all_txt, crop_eval_root, imgsz, val_batch)
    m50, m5075 = maps["ALL"]
    print(f"  ALL: mAP@0.5={m50:.4f}, mAP@0.5:0.75={m5075:.4f}")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return maps


# ===========================================================================
# Inference
# ===========================================================================

_MODEL = None


def init_worker(model_path: str) -> None:
    """Initialize worker process with a model instance."""
    global _MODEL
    cv2.setNumThreads(1)
    _MODEL = RTDETR(model_path)


def infer_single(
    args: Tuple[str, str, str, int, float]
) -> Optional[Dict[str, object]]:
    """Run inference on a single image with DAC post-processing."""
    img_path, label_path, group, imgsz, conf = args
    fname = os.path.splitext(os.path.basename(img_path))[0]

    img = cv2.imread(img_path)
    if img is None:
        return None

    crop_img, (_, _, _, _) = get_crop_region(img)
    if crop_img.size == 0:
        return None
    ch, cw = crop_img.shape[:2]

    results = _MODEL(crop_img, imgsz=imgsz, conf=conf, verbose=False)

    coarse_boxes_list, coarse_scores_list = [], []
    for r in results:
        if r.boxes is None:
            continue
        coarse_boxes_list.append(r.boxes.xyxy.cpu().numpy())
        coarse_scores_list.append(r.boxes.conf.cpu().numpy())

    if coarse_boxes_list:
        coarse_boxes = np.concatenate(coarse_boxes_list, axis=0)
        coarse_scores = np.concatenate(coarse_scores_list, axis=0)
        final_boxes, _ = dac_postprocess(coarse_boxes, coarse_scores, cw, ch)
        pred_count = len(final_boxes)
    else:
        pred_count = 0

    gt_count = get_gt_count(label_path)

    return {
        "fname": fname,
        "group": group,
        "pred": pred_count,
        "gt": gt_count,
    }


# ===========================================================================
# Main
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RT-DETR with DAC post-processing for soybean pod counting"
    )
    parser.add_argument("--img-dirs", nargs="+", required=True,
                        help="List of image directories (each containing images/ and labels/)")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model weights")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for coarse detection")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image size")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers")
    parser.add_argument("--val-batch", type=int, default=8,
                        help="Batch size for validation")
    parser.add_argument("--crop-eval-root", type=str, default="/tmp/cropped_eval",
                        help="Temporary directory for pre-cropped evaluation data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    exts = ("*.JPG", "*.jpg", "*.jpeg", "*.JPEG", "*.png", "*.PNG")
    tasks = []
    for img_dir in args.img_dirs:
        images_dir = os.path.join(img_dir, "images")
        labels_dir = os.path.join(img_dir, "labels")
        group = os.path.basename(img_dir.rstrip("/"))

        img_paths = []
        for ext in exts:
            img_paths.extend(glob.glob(os.path.join(images_dir, ext)))
        img_paths = sorted(list(set(img_paths)))
        print(f"{group}: {len(img_paths)} images")
        for p in img_paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            label_path = os.path.join(labels_dir, f"{stem}.txt")
            tasks.append((p, label_path, group, args.imgsz, args.conf))

    if not tasks:
        print("No images found.")
        return

    print(f"Total: {len(tasks)} images, Model: {args.model}")

    # Step 1: Build pre-cropped evaluation dataset
    group_cropped = build_crop_eval_dataset(
        tasks, args.crop_eval_root, args.workers
    )

    # Step 2: Evaluate mAP on pre-cropped images
    maps = run_map_eval(
        args.model, group_cropped, args.img_dirs,
        args.crop_eval_root, args.imgsz, args.val_batch,
    )

    # Step 3: Inference with DAC post-processing
    t0 = time.time()
    done = 0
    results = []

    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=mp.get_context("spawn"),
        initializer=init_worker,
        initargs=(args.model,),
    ) as pool:
        futures = {pool.submit(infer_single, t): t for t in tasks}
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                results.append(res)
            done += 1
            if done % 10 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                speed = done / elapsed if elapsed > 0 else 0
                remain = len(tasks) - done
                eta = remain / speed if speed > 0 else 0
                print(f"  [{done}/{len(tasks)}] {speed:.1f} img/s  ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\nInference done: {len(results)}/{len(tasks)} images, {elapsed:.1f}s")

    # Group results
    group_results: Dict[str, Dict[str, List[int]]] = {}
    for r in results:
        group_results.setdefault(r["group"], {"pred": [], "gt": []})
        group_results[r["group"]]["pred"].append(r["pred"])
        group_results[r["group"]]["gt"].append(r["gt"])

    # Per-group metrics
    all_pred, all_gt = [], []
    for img_dir in args.img_dirs:
        group = os.path.basename(img_dir.rstrip("/"))
        if group not in group_results:
            continue
        pred_list = group_results[group]["pred"]
        gt_list = group_results[group]["gt"]
        m50, m5075 = maps.get(group, (None, None))
        print_metrics(group, pred_list, gt_list, map50=m50, map50_75=m5075)
        all_pred.extend(pred_list)
        all_gt.extend(gt_list)

    # Overall metrics
    m50, m5075 = maps.get("ALL", (None, None))
    print_metrics("ALL", all_pred, all_gt, map50=m50, map50_75=m5075)


if __name__ == "__main__":
    main()
