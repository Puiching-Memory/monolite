from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.engine.registry import LOSSES


class VarifocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target, mask):
        pred_sig = pred.sigmoid()
        target_val = target * mask.unsqueeze(-1)
        focal_weight = target_val * (target_val > 0.0).float() + \
                       self.alpha * (pred_sig - target_val).abs().pow(self.gamma) * (target_val <= 0.0).float()
        loss = F.binary_cross_entropy_with_logits(pred, target_val, reduction='none') * focal_weight
        norm = target_val.sum()
        if norm < 1.0:
            norm = 1.0
        return loss.sum() / norm


class CIoULoss(nn.Module):
    def forward(self, b1, b2):
        # b1, b2: [N, 4] in x1,y1,x2,y2
        w1, h1 = (b1[:, 2] - b1[:, 0]).clamp(min=1e-6), (b1[:, 3] - b1[:, 1]).clamp(min=1e-6)
        w2, h2 = (b2[:, 2] - b2[:, 0]).clamp(min=1e-6), (b2[:, 3] - b2[:, 1]).clamp(min=1e-6)
        inter_x1 = torch.max(b1[:, 0], b2[:, 0])
        inter_y1 = torch.max(b1[:, 1], b2[:, 1])
        inter_x2 = torch.min(b1[:, 2], b2[:, 2])
        inter_y2 = torch.min(b1[:, 3], b2[:, 3])
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        union_area = w1 * h1 + w2 * h2 - inter_area + 1e-7
        iou = inter_area / union_area
        cw = torch.max(b1[:, 2], b2[:, 2]) - torch.min(b1[:, 0], b2[:, 0])
        ch = torch.max(b1[:, 3], b2[:, 3]) - torch.min(b1[:, 1], b2[:, 1])
        c2 = cw**2 + ch**2 + 1e-7
        b1_cx, b1_cy = (b1[:, 0] + b1[:, 2]) / 2, (b1[:, 1] + b1[:, 3]) / 2
        b2_cx, b2_cy = (b2[:, 0] + b2[:, 2]) / 2, (b2[:, 1] + b2[:, 3]) / 2
        rho2 = (b1_cx - b2_cx)**2 + (b1_cy - b2_cy)**2
        v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
        with torch.no_grad():
            alpha = v / (1 - iou + v + 1e-7)
        return 1.0 - iou + rho2 / c2 + alpha * v


@LOSSES.register("monolite_loss")
class MonoliteLoss(nn.Module):
    def __init__(
        self,
        heatmap3d_weight: float = 1.0,
        yolo2d_weight: float = 1.0,
        use_2d: bool = False,
        reg_max: int = 16,
    ):
        super().__init__()
        self.heatmap3d_weight = heatmap3d_weight
        self.yolo2d_weight = yolo2d_weight
        self.use_2d = use_2d
        self.reg_max = reg_max
        self.heatmap_loss = nn.BCEWithLogitsLoss()
        self.cls_loss = VarifocalLoss()
        self.iou_loss = CIoULoss()

    def forward(self, outputs: dict, targets: dict):
        losses = {}
        heatmap_pred = outputs.get("heatmap3d")
        heatmap_target = targets.get("heatmap3D")
        if heatmap_pred is not None and heatmap_target is not None:
            heatmap_target = heatmap_target.to(heatmap_pred.device)
            loss_heatmap = self.heatmap_loss(heatmap_pred.reshape(-1), heatmap_target.reshape(-1))
            losses["loss_heatmap3d"] = loss_heatmap * self.heatmap3d_weight
        
        # 3D regression losses
        if "indices" in targets and "mask_3d" in targets:
            indices = targets["indices"].to(torch.long)
            mask = targets["mask_3d"]
            
            if mask.sum() > 0:
                # depth
                if "depth" in outputs:
                    pred_depth = self._transpose_and_gather(outputs["depth"], indices)
                    target_depth = targets["depth"].to(pred_depth.device)
                    losses["loss_depth"] = F.l1_loss(pred_depth[mask.bool()], target_depth[mask.bool()]) * 1.0
                
                # size_3d
                if "size_3d" in outputs:
                    pred_size = self._transpose_and_gather(outputs["size_3d"], indices)
                    target_size = targets["size_3d"].to(pred_size.device)
                    losses["loss_size3d"] = F.l1_loss(pred_size[mask.bool()], target_size[mask.bool()]) * 1.0
                
                # offset_3d
                if "offset_3d" in outputs:
                    pred_offset = self._transpose_and_gather(outputs["offset_3d"], indices)
                    target_offset = targets["offset_3d"].to(pred_offset.device)
                    losses["loss_offset3d"] = F.l1_loss(pred_offset[mask.bool()], target_offset[mask.bool()]) * 1.0

                # heading
                if "heading" in outputs:
                    pred_heading = self._transpose_and_gather(outputs["heading"], indices) # [B, M, 24]
                    target_bin = targets["heading_bin"].to(torch.long).to(pred_heading.device)
                    target_res = targets["heading_res"].to(pred_heading.device)
                    
                    # Bin classification
                    pred_bin = pred_heading[..., :12]
                    loss_heading_bin = F.cross_entropy(pred_bin[mask.bool()], target_bin[mask.bool()].squeeze(-1))
                    
                    # Residual regression
                    pred_res = pred_heading[..., 12:]
                    # only for the ground truth bin
                    idx = torch.arange(mask.sum(), device=pred_heading.device)
                    target_bin_idx = target_bin[mask.bool()].squeeze(-1)
                    loss_heading_res = F.l1_loss(pred_res[mask.bool()][idx, target_bin_idx], target_res[mask.bool()].squeeze(-1))
                    
                    losses["loss_heading"] = loss_heading_bin + loss_heading_res

        if self.use_2d and "yolo2d" in outputs and "boxes2d" in targets:
            yolo_losses = self._compute_yolo2d_loss(outputs["yolo2d"], targets)
            losses.update(yolo_losses)
        
        total = sum(losses.values())
        return total, losses

    def _transpose_and_gather(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), feat.size(2))
        feat = feat.gather(1, ind)
        return feat

    def _compute_yolo2d_loss(self, yolo_outputs: list[dict], targets: dict) -> dict:
        target_labels = targets["labels"]
        target_boxes = targets["boxes2d"]
        image_size = targets.get("image_size")
        device = target_boxes.device
        
        all_cls_preds = []
        all_box_preds = []
        all_box_dist = []
        all_grids = []
        
        strides = [8.0, 16.0, 32.0] # Default
        if image_size is not None:
            img_h, img_w = image_size[0]
            strides = [img_h / out["cls"].shape[2] for out in yolo_outputs]

        for lvl, out in enumerate(yolo_outputs):
            h, w = out["cls"].shape[2:]
            stride = strides[lvl]
            ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            grid = (torch.stack([xs, ys], dim=-1).to(device).float() + 0.5) * stride
            all_grids.append(grid.view(-1, 2))

            cls_p = out["cls"].flatten(2).transpose(1, 2)
            ltrb_p = out["box"].flatten(2).transpose(1, 2) * stride # Distances to L,T,R,B
            
            # Convert LTRB to x1y1x2y2
            # grid shape (H*W, 2) -> (1, H*W, 2)
            g = grid.view(1, -1, 2)
            x1y1 = g - ltrb_p[:, :, :2]
            x2y2 = g + ltrb_p[:, :, 2:]
            box_p = torch.cat([x1y1, x2y2], dim=-1)
            
            all_cls_preds.append(cls_p)
            all_box_preds.append(box_p)
            all_box_dist.append(out["box_dist"].flatten(2).transpose(1, 2))

        cls_preds = torch.cat(all_cls_preds, dim=1)
        box_preds = torch.cat(all_box_preds, dim=1)
        box_dist = torch.cat(all_box_dist, dim=1)
        grids = torch.cat(all_grids, dim=0)
        
        # Get strides for each anchor
        all_strides = []
        for lvl, out in enumerate(yolo_outputs):
            h, w = out["cls"].shape[2:]
            stride = strides[lvl]
            all_strides.append(torch.full((h * w, 1), stride, device=device))
        strides_flat = torch.cat(all_strides, dim=0).view(1, -1, 1)

        target_cls, target_box, target_mask = self._tal_assigner(
            cls_preds.detach(), box_preds.detach(), grids, target_labels, target_boxes
        )

        loss_cls = self.cls_loss(cls_preds, target_cls, target_mask)
        loss_box = torch.tensor(0.0, device=device)
        loss_dfl = torch.tensor(0.0, device=device)
        
        if target_mask.sum() > 0:
            mask_bool = target_mask.bool()
            loss_box = self.iou_loss(box_preds[mask_bool], target_box[mask_bool]).mean()
            
            if self.reg_max > 1:
                # Get LTRB target for DFL: (left, top, right, bottom)
                # target_box is (x1, y1, x2, y2)
                gt_ltrb = torch.cat([
                    grids.unsqueeze(0) - target_box[..., :2],
                    target_box[..., 2:] - grids.unsqueeze(0)
                ], dim=-1) / strides_flat # [B, N, 4]
                
                loss_dfl = self._compute_dfl_loss(
                    box_dist[mask_bool], 
                    gt_ltrb[mask_bool]
                )

        total_yolo_loss = (loss_cls + 1.5 * loss_box + 0.5 * loss_dfl) * self.yolo2d_weight
        return {
            "loss_yolo2d": total_yolo_loss,
            "loss_vfl": loss_cls,
            "loss_ciou": loss_box,
            "loss_dfl": loss_dfl,
        }

    def _compute_dfl_loss(self, pred_dist, target_ltrb):
        # pred_dist: [N, 4*reg_max]
        # target_ltrb: [N, 4]
        device = pred_dist.device
        pred_dist = pred_dist.view(-1, self.reg_max) # [N*4, reg_max]
        target_ltrb = target_ltrb.view(-1).clamp(0, self.reg_max - 1.01) # [N*4]
        
        idx_left = target_ltrb.long()
        idx_right = idx_left + 1
        
        weight_left = idx_right.float() - target_ltrb
        weight_right = target_ltrb - idx_left.float()
        
        loss_left = F.cross_entropy(pred_dist, idx_left, reduction="none") * weight_left
        loss_right = F.cross_entropy(pred_dist, idx_right, reduction="none") * weight_right
        
        return (loss_left + loss_right).mean()

    def _tal_assigner(self, cls_preds, box_preds, grids, gt_labels, gt_boxes):
        device = cls_preds.device
        b, n, c = cls_preds.shape
        target_cls = torch.zeros((b, n, c), device=device)
        target_box = torch.zeros((b, n, 4), device=device)
        target_mask = torch.zeros((b, n), device=device)
        for i in range(b):
            valid = gt_labels[i] >= 0
            if not valid.any(): continue
            objs = gt_boxes[i][valid]
            labels = gt_labels[i][valid]
            for obj, label in zip(objs, labels):
                dist = torch.norm(grids - ((obj[:2] + obj[2:]) / 2), dim=1)
                idx = dist.argmin()
                target_cls[i, idx, int(label)] = 0.9
                target_box[i, idx] = obj
                target_mask[i, idx] = 1.0
        return target_cls, target_box, target_mask
