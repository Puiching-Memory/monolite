from __future__ import annotations

import os
from typing import Any, Dict, Iterable
from collections import defaultdict

import random
import cv2
import numpy as np
import torch


class Visualizer:
    def __init__(
        self,
        cfg: Dict[str, Any],
        output_dir: str,
        class_id_to_name: Dict[int, str],
        logger=None,
    ) -> None:
        trainer_cfg = cfg.get("trainer", {})
        self.vis_num = int(trainer_cfg.get("vis_num", 2))
        self.vis_boxes = bool(trainer_cfg.get("vis_boxes", True))
        self.vis_pred_num = int(trainer_cfg.get("vis_pred_num", 20))
        self.vis_pred_thr = trainer_cfg.get("vis_pred_thr", 0.25)
        self.vis_nms_thr = trainer_cfg.get("vis_nms_thr", 0.5)

        self.output_dir = output_dir
        self.class_id_to_name = class_id_to_name
        self.logger = logger
        
        self.mean_size = np.array([
            [1.52563191, 1.62856739, 3.52588311], # Car
            [1.76255119, 0.66068622, 0.84422524], # Pedestrian
            [1.73698127, 0.59706367, 1.76282397], # Cyclist
        ], dtype=np.float32)

    @torch.no_grad()
    def visualize(
        self,
        epoch: int,
        model: torch.nn.Module,
        val_loader: Iterable,
        device: torch.device,
    ) -> None:
        if self.vis_num <= 0:
            return

        save_dir = os.path.join(self.output_dir, "visuals", f"epoch_{epoch+1}")
        os.makedirs(save_dir, exist_ok=True)

        model.eval()
        saved = 0

        # Randomly sample indices for visualization
        total_samples = len(val_loader.dataset)
        sample_indices = set(random.sample(range(total_samples), min(total_samples, self.vis_num)))

        for inputs, targets, info in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            batch_size = inputs.shape[0]
            for i in range(batch_size):
                image_id = None
                if isinstance(info, dict) and "image_id" in info:
                    try:
                        image_id = info["image_id"][i].item()
                    except Exception:
                        image_id = None
                
                if image_id not in sample_indices:
                    continue

                img = self._tensor_to_bgr(inputs[i])
                img_3d = img.copy()

                heatmap = outputs.get("heatmap3d")
                if heatmap is not None:
                    hm = heatmap[i]
                    if hm.ndim == 3:
                        hm = hm.sigmoid().max(dim=0).values
                    hm = hm.clamp(0, 1).cpu().numpy()
                    hm = (hm * 255).astype(np.uint8)
                    img_hm = cv2.resize(hm, (img.shape[1], img.shape[0]))
                    img_hm = cv2.applyColorMap(img_hm, cv2.COLORMAP_VIRIDIS)
                    img = cv2.addWeighted(img, 0.6, img_hm, 0.4, 0)
                
                # Draw 3D Predictions
                if "calib" in info:
                    P2 = info["calib"][i].cpu().numpy()
                    preds_3d = self._decode_3d_predictions(outputs, i, P2)
                    for corners in preds_3d:
                        img_3d = self._draw_3d_box(img_3d, corners, (0, 255, 0))

                # Draw 3D Ground Truth
                if "calib" in info and "gt_boxes3d" in targets:
                    P2 = info["calib"][i].cpu().numpy()
                    gt_boxes3d = targets["gt_boxes3d"][i].cpu().numpy()
                    for b3d in gt_boxes3d:
                        if b3d[2] <= 0:  # depth check
                            continue
                        pos = b3d[:3]
                        size = b3d[3:6]
                        ry = b3d[6]
                        corners = self._project_3d_box(P2, pos, size, ry)
                        img_3d = self._draw_3d_box(img_3d, corners, (0, 0, 255))

                if self.vis_boxes and "boxes2d" in targets:
                    boxes = targets["boxes2d"][i].detach().cpu().numpy()
                    labels = targets.get("labels")
                    label_array = None
                    if labels is not None:
                        label_array = labels[i].detach().cpu().numpy()

                    for idx, box in enumerate(boxes):
                        if box.sum() <= 0:
                            continue

                        class_id = None
                        if label_array is not None:
                            class_id = int(label_array[idx])
                            if class_id < 0:
                                continue

                        x1, y1, x2, y2 = box.astype(int).tolist()
                        img = self._draw_box(img, x1, y1, x2, y2, (0, 0, 255))

                        label_name = self._get_class_name(class_id)
                        label_text = f"GT {label_name} 1.00"
                        img = self._draw_label(img, label_text, x1, y1, (0, 0, 255))

                yolo_outputs = outputs.get("yolo2d")
                image_size = targets.get("image_size")
                img_size = None
                if image_size is not None:
                    if image_size.ndim == 1:
                        img_size = image_size
                    else:
                        img_size = image_size[i]

                pred_boxes = self._decode_pred_boxes(
                    yolo_outputs=yolo_outputs,
                    image_size=img_size,
                    image_index=i,
                )
                for x1, y1, x2, y2, score, class_id in pred_boxes:
                    img = self._draw_box(img, x1, y1, x2, y2, (0, 255, 0))
                    label_name = self._get_class_name(class_id)
                    label_text = f"PR {label_name} {score:.2f}"
                    img = self._draw_label(img, label_text, x1, y1, (0, 255, 0))

                name = (
                    f"vis_{saved}.jpg"
                    if image_id is None
                    else f"vis_{saved}_id_{image_id}.jpg"
                )
                cv2.imwrite(os.path.join(save_dir, name), img)
                if "calib" in info:
                    cv2.imwrite(os.path.join(save_dir, f"3d_{name}"), img_3d)
                saved += 1

            if saved >= self.vis_num:
                break

        if self.logger is not None:
            self.logger.info(f"visuals saved: {save_dir} ({saved} images)")
        model.train()

    def _decode_3d_predictions(self, outputs, batch_idx, P2):
        heatmap = outputs["heatmap3d"][batch_idx].sigmoid()
        depth_map = outputs["depth"][batch_idx]
        size_map = outputs["size_3d"][batch_idx]
        heading_map = outputs["heading"][batch_idx]
        offset_map = outputs["offset_3d"][batch_idx]
        
        c, h, w = heatmap.shape
        hmax = torch.nn.functional.max_pool2d(heatmap.unsqueeze(0), kernel_size=3, stride=1, padding=1)
        keep = (hmax == heatmap.unsqueeze(0)).float()
        heatmap = heatmap * keep.squeeze(0)
        
        scores, indices = torch.topk(heatmap.view(-1), k=20)
        valid = scores > self.vis_pred_thr
        scores = scores[valid]
        indices = indices[valid]
        
        if len(indices) == 0:
            return []
            
        cls_ids = indices // (h * w)
        indices = indices % (h * w)
        ys = (indices // w).long()
        xs = (indices % w).long()
        
        fu, fv = P2[0, 0], P2[1, 1]
        cu, cv = P2[0, 2], P2[1, 2]
        tx, ty = P2[0, 3], P2[1, 3]
        
        stride = 8.0 
        all_corners = []
        
        for i in range(len(indices)):
            gy, gx = ys[i], xs[i]
            cid = cls_ids[i].item()
            if cid >= len(self.mean_size): cid = 0 # fall back
            
            d = depth_map[0, gy, gx].item()
            s = size_map[:, gy, gx].cpu().numpy() + self.mean_size[cid]
            off = offset_map[:, gy, gx].cpu().numpy()
            
            h_bin = heading_map[:12, gy, gx].argmax().item()
            h_res = heading_map[12 + h_bin, gy, gx].item()
            angle_per_class = 2 * np.pi / 12.0
            angle = h_bin * angle_per_class + h_res
            if angle > np.pi: angle -= 2 * np.pi
            
            px = (gx.float() + 0.5 + off[0]) * stride
            py = (gy.float() + 0.5 + off[1]) * stride
            
            z = d
            x = (px.item() * z - cu * z - tx) / fu
            y = (py.item() * z - cv * z - ty) / fv
            pos = np.array([x, y, z])

            h, w_box, l = s[0], s[1], s[2]
            pts_2d = self._project_3d_box(P2, pos, (h, w_box, l), angle)
            all_corners.append(pts_2d)

        return all_corners

    def _draw_3d_box(self, img, corners_3d, color=(0, 255, 0), thickness=2):
        corners_3d = corners_3d.astype(np.int32)
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            cv2.line(img, (corners_3d[i, 0], corners_3d[i, 1]), (corners_3d[j, 0], corners_3d[j, 1]), color, thickness)
            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(img, (corners_3d[i, 0], corners_3d[i, 1]), (corners_3d[j, 0], corners_3d[j, 1]), color, thickness)
            i, j = k, k + 4
            cv2.line(img, (corners_3d[i, 0], corners_3d[i, 1]), (corners_3d[j, 0], corners_3d[j, 1]), color, thickness)
        return img

    def _project_3d_box(self, P2, pos, size, ry):
        h, w, l = size
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        corners = np.vstack([x_corners, y_corners, z_corners])

        rot_mat = np.array(
            [
                [np.cos(ry), 0, np.sin(ry)],
                [0, 1, 0],
                [-np.sin(ry), 0, np.cos(ry)],
            ]
        )
        corners = np.dot(rot_mat, corners)
        corners += pos.reshape(3, 1)

        corners_hom = np.vstack([corners, np.ones((1, 8))])
        pts_2d = np.dot(P2, corners_hom)
        pts_2d = pts_2d[:2] / pts_2d[2]
        return pts_2d.T

    def _tensor_to_bgr(self, tensor: torch.Tensor) -> np.ndarray:
        mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
        img = tensor.detach().float() * std + mean
        img = img.clamp(0, 1)
        img = (img * 255).byte().permute(1, 2, 0).cpu().numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def _draw_box(
        self,
        img: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: tuple[int, int, int],
        thickness: int = 2,
    ) -> np.ndarray:
        h, w = img.shape[:2]
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(0, min(int(x2), w - 1))
        y2 = max(0, min(int(y2), h - 1))
        if x2 <= x1 or y2 <= y1:
            return img
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        return img

    def _draw_label(
        self,
        img: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
        font_scale: float = 0.5,
        thickness: int = 1,
    ) -> np.ndarray:
        h, w = img.shape[:2]
        x = max(0, min(int(x), w - 1))
        y = max(0, min(int(y), h - 1))

        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        y1 = max(0, y - text_h - baseline - 2)
        y2 = min(h - 1, y)
        x2 = min(w - 1, x + text_w + 2)

        cv2.rectangle(img, (x, y1), (x2, y2), color, -1)
        cv2.putText(
            img,
            text,
            (x + 1, y2 - baseline - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            lineType=cv2.LINE_AA,
        )
        return img

    def _get_class_name(self, class_id: int | None) -> str:
        if class_id is None:
            return "class"
        return self.class_id_to_name.get(int(class_id), f"class_{int(class_id)}")

    def _decode_pred_boxes(
        self,
        yolo_outputs,
        image_size: torch.Tensor | None,
        image_index: int,
    ) -> list[tuple[int, int, int, int, float, int]]:
        if yolo_outputs is None or image_size is None:
            return []

        img_h = float(image_size[0].item())
        img_w = float(image_size[1].item())

        candidates = []
        for out in yolo_outputs:
            cls_logits = out["cls"][image_index]
            box_pred = out["box"][image_index]

            scores = torch.sigmoid(cls_logits)
            conf, cls_id = scores.max(dim=0)

            conf_flat = conf.reshape(-1)
            if self.vis_pred_thr is not None:
                keep = conf_flat >= float(self.vis_pred_thr)
                if keep.sum() == 0:
                    continue
                conf_flat = conf_flat[keep]
                keep_idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
            else:
                keep_idx = torch.arange(conf_flat.numel(), device=conf_flat.device)

            topk = min(int(self.vis_pred_num), conf_flat.numel())
            if topk <= 0:
                continue
            conf_topk, topk_idx = torch.topk(conf_flat, k=topk)
            sel = keep_idx[topk_idx]

            h, w = cls_logits.shape[1], cls_logits.shape[2]
            gy = (sel // w).long()
            gx = (sel % w).long()

            stride_h = img_h / float(h)
            stride_w = img_w / float(w)

            # box_pred format: [dist_l, dist_t, dist_r, dist_b] in stride units
            l = box_pred[0, gy, gx] * stride_w
            t = box_pred[1, gy, gx] * stride_h
            r = box_pred[2, gy, gx] * stride_w
            b = box_pred[3, gy, gx] * stride_h

            # grid center in image coordinates
            gx_coord = (gx.float() + 0.5) * stride_w
            gy_coord = (gy.float() + 0.5) * stride_h

            x1 = (gx_coord - l).clamp(0, img_w - 1)
            y1 = (gy_coord - t).clamp(0, img_h - 1)
            x2 = (gx_coord + r).clamp(0, img_w - 1)
            y2 = (gy_coord + b).clamp(0, img_h - 1)

            for j in range(conf_topk.numel()):
                cls_idx = int(cls_id[gy[j], gx[j]].item())
                candidates.append(
                    (
                        int(x1[j].item()),
                        int(y1[j].item()),
                        int(x2[j].item()),
                        int(y2[j].item()),
                        float(conf_topk[j].item()),
                        cls_idx,
                    )
                )

        candidates.sort(key=lambda x: x[4], reverse=True)
        candidates = self._apply_nms(candidates, self.vis_nms_thr)
        return candidates[: int(self.vis_pred_num)]

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

    def _box_iou(self, box1: Iterable, box2: Iterable) -> float:
        x1_1, y1_1, x1_2, y1_2 = box1
        x2_1, y2_1, x2_2, y2_2 = box2

        i_x1 = max(x1_1, x2_1)
        i_y1 = max(y1_1, y2_1)
        i_x2 = min(x1_2, x2_2)
        i_y2 = min(y1_2, y2_2)

        i_area = max(0, i_x2 - i_x1) * max(0, i_y2 - i_y1)
        b1_area = (x1_2 - x1_1) * (y1_2 - y1_1)
        b2_area = (x2_2 - x2_1) * (y2_2 - y2_1)

        union = b1_area + b2_area - i_area
        if union <= 0:
            return 0.0
        return i_area / union
