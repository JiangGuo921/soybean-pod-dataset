"""
Training script for RT-DETR-based soybean pod detection.

This script trains an RT-DETR model with a custom callback that saves
the best checkpoint based on mAP@0.5 (COCO metric).
"""

import argparse
import copy
from pathlib import Path

import torch
from ultralytics import RTDETR


class Map50BestSaver:
    """Callback to save the best model checkpoint based on mAP@0.5."""

    def __init__(self) -> None:
        self.best_map50 = 0.0

    def __call__(self, trainer) -> None:
        map50 = trainer.metrics.get("metrics/mAP50(B)", 0)
        if isinstance(map50, torch.Tensor):
            map50 = map50.item()

        if map50 > self.best_map50:
            self.best_map50 = map50
            ckpt = {
                "epoch": trainer.epoch,
                "model": copy.deepcopy(trainer.model).half(),
                "best_map50": map50,
                "date": None,
            }
            save_path = trainer.last.parent / "best_map50.pt"
            torch.save(ckpt, save_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RT-DETR for soybean pod detection")
    parser.add_argument("--model-cfg", type=str, required=True,
                        help="Path to RT-DETR model YAML config")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to pretrained weights")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to dataset YAML config")
    parser.add_argument("--project", type=str, default="./runs",
                        help="Directory to save training outputs")
    parser.add_argument("--name", type=str, default="soybean_pod",
                        help="Experiment name")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    parser.add_argument("--device", type=int, default=0,
                        help="GPU device index")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--lr0", type=float, default=1e-4,
                        help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01,
                        help="Final learning rate factor")
    parser.add_argument("--warmup-epochs", type=int, default=20,
                        help="Number of warmup epochs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = RTDETR(args.model_cfg)
    model.load(args.weights)

    saver = Map50BestSaver()
    model.add_callback("on_fit_epoch_end", saver)

    model.train(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.epochs,
        patience=args.patience,
        amp=True,
        optimizer="AdamW",
        copy_paste=0.5,
        lr0=args.lr0,
        lrf=args.lrf,
        warmup_epochs=args.warmup_epochs,
        warmup_momentum=0.5,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.5,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.5,
        box=10.0,
        cls=0.8,
        mosaic=0,
        mixup=0.0,
        close_mosaic=0,
        cos_lr=True,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        save=True,
        val=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
