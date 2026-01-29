from __future__ import annotations

import os
import time
from typing import Dict, Any
from collections import defaultdict

import torch
from torch.cuda.amp import GradScaler
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
import numpy as np
from lib.vis import Visualizer

from .checkpoint import save_checkpoint


class Trainer:
    def __init__(
        self,
        cfg: Dict[str, Any],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss_fn: torch.nn.Module,
        train_loader,
        logger,
        device: torch.device,
        val_loader=None,
    ):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.device = device
        self.epoch = cfg.get("trainer", {}).get("start_epoch", 0)
        self.max_epoch = cfg.get("trainer", {}).get("max_epoch", 1)
        self.save_frequency = cfg.get("trainer", {}).get("save_frequency", 1)
        self.log_interval = cfg.get("trainer", {}).get("log_interval", 10)
        self.output_dir = cfg.get("output_dir", "./runs/exp")
        self.amp = cfg.get("amp", True)
        self.scaler = GradScaler(enabled=self.amp)

        trainer_cfg = cfg.get("trainer", {})
        self.eval_interval = trainer_cfg.get("eval_interval", 1)
        self.eval_max_batches = trainer_cfg.get("eval_max_batches", None)
        self.vis_interval = trainer_cfg.get("vis_interval", 1)
        self.eval_pred_thr = trainer_cfg.get("eval_pred_thr", 0.25)
        self.eval_pred_topk = trainer_cfg.get("eval_pred_topk", 300)
        self.eval_iou_thr = trainer_cfg.get("eval_iou_thr", 0.5)
        self.eval_nms_thr = trainer_cfg.get("eval_nms_thr", 0.5)
        self.miou_thr = trainer_cfg.get("miou_thr", 0.5)

        self.best_metric = None
        self.best_map = -1.0
        self.best_miou = -1.0
        self.best_epoch = -1

        class_map = cfg.get("dataset", {}).get("args", {}).get("class_map", {})
        self.class_id_to_name = {int(v): str(k) for k, v in class_map.items()}
        self.visualizer = Visualizer(
            cfg=cfg,
            output_dir=self.output_dir,
            class_id_to_name=self.class_id_to_name,
            logger=self.logger,
        )

    def train(self):
        self.model.train()
        for epoch in range(self.epoch, self.max_epoch):
            epoch_start = time.perf_counter()
            self._train_one_epoch(epoch)
            self.scheduler.step()

            if self.val_loader and self.eval_interval and (epoch + 1) % self.eval_interval == 0:
                eval_result = self._evaluate(epoch)
                if eval_result is not None:
                    self._maybe_save_best(epoch, eval_result)

            if self.val_loader and self.vis_interval and (epoch + 1) % self.vis_interval == 0:
                self._visualize(epoch)

            if not self.val_loader:
                self._maybe_save_best(epoch, None)

            self.logger.info(
                f"epoch {epoch+1}/{self.max_epoch} finished, time: {time.perf_counter() - epoch_start:.2f}s"
            )

    def _train_one_epoch(self, epoch: int):
        self.model.train()
        total = len(self.train_loader)
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]train[/bold]"),
            TextColumn("epoch {task.fields[epoch]}/{task.fields[max_epoch]}"),
            BarColumn(bar_width=None),
            TextColumn("{task.percentage:>3.0f}%"),
            TextColumn("loss={task.fields[loss]}") ,
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=True,
        )
        with progress:
            task_id = progress.add_task(
                "train",
                total=total,
                epoch=epoch + 1,
                max_epoch=self.max_epoch,
                loss="--",
            )
            for step, (inputs, targets, info) in enumerate(self.train_loader, start=1):
                inputs = inputs.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}

                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type="cuda" if torch.cuda.is_available() else "cpu",
                    enabled=self.amp,
                ):
                    outputs = self.model(inputs)
                    loss, loss_info = self.loss_fn(outputs, targets)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                loss_value = loss.item() if hasattr(loss, "item") else float(loss)
                progress.update(task_id, advance=1, loss=f"{loss_value:.4f}")

                if step % self.log_interval == 0:
                    if isinstance(loss_info, dict):
                        def _to_float(val):
                            if hasattr(val, "item"):
                                return float(val.item())
                            return float(val)

                        log_detail = ", ".join([f"{k}={_to_float(v):.4f}" for k, v in loss_info.items()])
                        self.logger.info(
                            f"epoch {epoch+1} step {step}: loss={loss_value:.4f} {log_detail}"
                        )

    @torch.no_grad()
    def _evaluate(self, epoch: int) -> dict | None:
        self.model.eval()
        total_loss = 0.0
        loss_sums = defaultdict(float)
        count = 0

        num_classes = len(self.class_id_to_name)
        pred_by_class = defaultdict(list)
        gt_by_class = defaultdict(lambda: defaultdict(list))
        miou_inter = np.zeros((num_classes,), dtype=np.float64)
        miou_union = np.zeros((num_classes,), dtype=np.float64)

        total = len(self.val_loader)
        if self.eval_max_batches:
            total = min(total, int(self.eval_max_batches))

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]eval[/bold]"),
            TextColumn("epoch {task.fields[epoch]}/{task.fields[max_epoch]}"),
            BarColumn(bar_width=None),
            TextColumn("{task.percentage:>3.0f}%"),
            TextColumn("loss={task.fields[loss]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=True,
        )
        with progress:
            task_id = progress.add_task(
                "eval",
                total=total,
                epoch=epoch + 1,
                max_epoch=self.max_epoch,
                loss="--",
            )
            for step, (inputs, targets, info) in enumerate(self.val_loader, start=1):
                inputs = inputs.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}

                with torch.autocast(
                    device_type="cuda" if torch.cuda.is_available() else "cpu",
                    enabled=self.amp,
                ):
                    outputs = self.model(inputs)
                    loss, loss_info = self.loss_fn(outputs, targets)

                loss_value = loss.item() if hasattr(loss, "item") else float(loss)
                total_loss += loss_value
                if isinstance(loss_info, dict):
                    for k, v in loss_info.items():
                        loss_sums[k] += float(v.item()) if hasattr(v, "item") else float(v)

                yolo_outputs = outputs.get("yolo2d")
                image_size = targets.get("image_size")
                boxes2d = targets.get("boxes2d")
                labels = targets.get("labels")

                if yolo_outputs is not None and image_size is not None:
                    batch_size = inputs.shape[0]
                    for b in range(batch_size):
                        img_size = image_size if image_size.ndim == 1 else image_size[b]
                        preds = self._decode_pred_boxes_eval(
                            yolo_outputs=yolo_outputs,
                            image_size=img_size,
                            image_index=b,
                        )
                        preds = self._apply_nms(preds, self.eval_nms_thr)
                        for x1, y1, x2, y2, score, class_id in preds:
                            pred_by_class[class_id].append(
                                {
                                    "image_id": int(info["image_id"][b]) if isinstance(info, dict) and "image_id" in info else int(b),
                                    "bbox": [x1, y1, x2, y2],
                                    "score": float(score),
                                }
                            )

                if boxes2d is not None and labels is not None:
                    batch_size = boxes2d.shape[0]
                    for b in range(batch_size):
                        for j in range(boxes2d.shape[1]):
                            cls_id = int(labels[b, j].item())
                            if cls_id < 0:
                                continue
                            box = boxes2d[b, j].detach().cpu().numpy().tolist()
                            gt_by_class[cls_id][int(info["image_id"][b]) if isinstance(info, dict) and "image_id" in info else int(b)].append(box)

                heatmap_pred = outputs.get("heatmap3d")
                heatmap_gt = targets.get("heatmap3D")
                if heatmap_pred is not None and heatmap_gt is not None:
                    pred_bin = (heatmap_pred.sigmoid() > float(self.miou_thr))
                    gt_bin = heatmap_gt > 0
                    inter = (pred_bin & gt_bin).sum(dim=(0, 2, 3)).detach().cpu().numpy()
                    union = (pred_bin | gt_bin).sum(dim=(0, 2, 3)).detach().cpu().numpy()
                    miou_inter += inter
                    miou_union += union

                count += 1
                progress.update(task_id, advance=1, loss=f"{loss_value:.4f}")
                if self.eval_max_batches and count >= self.eval_max_batches:
                    break

        if count == 0:
            self.logger.info(f"eval epoch {epoch+1}: no batches")
        else:
            avg_loss = total_loss / count
            detail = ", ".join([f"{k}={loss_sums[k]/count:.4f}" for k in sorted(loss_sums.keys())])
            map50 = self._compute_map(pred_by_class, gt_by_class, num_classes, self.eval_iou_thr)
            miou = self._compute_miou(miou_inter, miou_union)
            self.logger.info(
                f"eval epoch {epoch+1}: loss={avg_loss:.4f} mAP@{self.eval_iou_thr:.2f}={map50:.4f} mIoU3D={miou:.4f} {detail}"
            )

        self.model.train()
        if count == 0:
            return None
        return {
            "loss": total_loss / count,
            "map": map50,
            "miou": miou,
        }

    def _maybe_save_best(self, epoch: int, eval_result: dict | None) -> None:
        if eval_result is None:
            if self.best_metric is None:
                self.best_metric = float("inf")
        else:
            current_map = float(eval_result.get("map", -1.0))
            current_miou = float(eval_result.get("miou", -1.0))
            if current_map < self.best_map:
                return
            if abs(current_map - self.best_map) <= 1e-12 and current_miou <= self.best_miou:
                return
            self.best_map = current_map
            self.best_miou = current_miou
            self.best_metric = eval_result.get("loss", None)

        ckpt_path = os.path.join(self.output_dir, "checkpoints", "best")
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "best_metric": self.best_metric,
                "best_map": self.best_map,
                "best_miou": self.best_miou,
            },
            ckpt_path,
        )
        self.best_epoch = epoch + 1
        if eval_result is None:
            self.logger.info(f"checkpoint saved: {ckpt_path}.pth (latest)")
        else:
            self.logger.info(
                f"checkpoint saved: {ckpt_path}.pth (best mAP {self.best_map:.4f}, mIoU {self.best_miou:.4f} at epoch {self.best_epoch})"
            )

    @torch.no_grad()
    def _visualize(self, epoch: int):
        self.visualizer.visualize(epoch, self.model, self.val_loader, self.device)

    def _decode_pred_boxes_eval(
        self,
        yolo_outputs,
        image_size: torch.Tensor | None,
        image_index: int,
    ) -> list[tuple[int, int, int, int, float, int]]:
        if yolo_outputs is None or image_size is None:
            return []

        img_h = float(image_size[0].item())
        img_w = float(image_size[1].item())

        candidates: list[tuple[int, int, int, int, float, int]] = []
        for out in yolo_outputs:
            cls_logits = out["cls"][image_index]
            box_pred = out["box"][image_index]

            scores = torch.sigmoid(cls_logits)
            c, h, w = scores.shape
            scores_flat = scores.reshape(-1)
            if self.eval_pred_thr is not None:
                keep = scores_flat >= float(self.eval_pred_thr)
                if keep.sum() == 0:
                    continue
                scores_flat = scores_flat[keep]
                keep_idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
            else:
                keep_idx = torch.arange(scores_flat.numel(), device=scores_flat.device)

            topk = min(int(self.eval_pred_topk), scores_flat.numel())
            if topk <= 0:
                continue
            conf_topk, topk_idx = torch.topk(scores_flat, k=topk)
            sel = keep_idx[topk_idx]

            cls_id = (sel // (h * w)).long()
            rem = sel % (h * w)
            gy = (rem // w).long()
            gx = (rem % w).long()

            stride_h = img_h / float(h)
            stride_w = img_w / float(w)

            cx = box_pred[0, gy, gx] * stride_w
            cy = box_pred[1, gy, gx] * stride_h
            bw = box_pred[2, gy, gx] * stride_w
            bh = box_pred[3, gy, gx] * stride_h

            x1 = (cx - bw / 2.0).clamp(0, img_w - 1)
            y1 = (cy - bh / 2.0).clamp(0, img_h - 1)
            x2 = (cx + bw / 2.0).clamp(0, img_w - 1)
            y2 = (cy + bh / 2.0).clamp(0, img_h - 1)

            for j in range(conf_topk.numel()):
                candidates.append(
                    (
                        int(x1[j].item()),
                        int(y1[j].item()),
                        int(x2[j].item()),
                        int(y2[j].item()),
                        float(conf_topk[j].item()),
                        int(cls_id[j].item()),
                    )
                )

        candidates.sort(key=lambda x: x[4], reverse=True)
        return candidates

    def _apply_nms(
        self,
        boxes: list[tuple[int, int, int, int, float, int]],
        iou_thr: float,
    ) -> list[tuple[int, int, int, int, float, int]]:
        if not boxes:
            return boxes

        results = []
        boxes_by_class: dict[int, list[tuple[int, int, int, int, float, int]]] = defaultdict(list)
        for b in boxes:
            boxes_by_class[int(b[5])].append(b)

        for cls_id, bxs in boxes_by_class.items():
            bxs = sorted(bxs, key=lambda x: x[4], reverse=True)
            kept: list[tuple[int, int, int, int, float, int]] = []
            while bxs:
                best = bxs.pop(0)
                kept.append(best)
                bxs = [
                    b
                    for b in bxs
                    if self._box_iou(best[:4], b[:4]) < float(iou_thr)
                ]
            results.extend(kept)

        results.sort(key=lambda x: x[4], reverse=True)
        return results

    def _box_iou(self, a: list[int] | tuple[int, int, int, int], b: list[int] | tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter + 1e-6
        return inter / union

    def _compute_map(
        self,
        pred_by_class,
        gt_by_class,
        num_classes: int,
        iou_thr: float,
    ) -> float:
        ap_list = []
        for cls_id in range(num_classes):
            preds = pred_by_class.get(cls_id, [])
            gts = gt_by_class.get(cls_id, {})
            num_gt = sum(len(v) for v in gts.values())
            if num_gt == 0:
                continue

            preds = sorted(preds, key=lambda x: x["score"], reverse=True)
            tp = np.zeros(len(preds), dtype=np.float32)
            fp = np.zeros(len(preds), dtype=np.float32)

            matched = {img_id: np.zeros(len(boxes), dtype=bool) for img_id, boxes in gts.items()}
            for i, p in enumerate(preds):
                img_id = p["image_id"]
                gt_boxes = gts.get(img_id, [])
                if len(gt_boxes) == 0:
                    fp[i] = 1.0
                    continue
                ious = [self._box_iou(p["bbox"], gt) for gt in gt_boxes]
                max_iou = max(ious) if ious else 0.0
                max_idx = int(np.argmax(ious)) if ious else -1
                if max_iou >= float(iou_thr) and not matched[img_id][max_idx]:
                    tp[i] = 1.0
                    matched[img_id][max_idx] = True
                else:
                    fp[i] = 1.0

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recall = tp_cum / max(num_gt, 1)
            precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)

            ap = self._compute_ap(recall, precision)
            ap_list.append(ap)

        if len(ap_list) == 0:
            return 0.0
        return float(np.mean(ap_list))

    def _compute_ap(self, recall: np.ndarray, precision: np.ndarray) -> float:
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
        return float(ap)

    def _compute_miou(self, inter: np.ndarray, union: np.ndarray) -> float:
        valid = union > 0
        if valid.sum() == 0:
            return 0.0
        iou = np.zeros_like(inter, dtype=np.float64)
        iou[valid] = inter[valid] / (union[valid] + 1e-6)
        return float(iou[valid].mean())
