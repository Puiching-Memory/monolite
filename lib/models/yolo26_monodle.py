from __future__ import annotations

import torch
import torch.nn as nn
from ultralytics import YOLO
from lib.engine.registry import MODELS

@MODELS.register("yolo26_monodle")
class Yolo26Monodle(nn.Module):
    """
    Yolo26-based model with Monodle 3D detection heads.
    The yolo26 base is frozen and used as a feature extractor.
    """
    def __init__(self, pretrained: str, num_classes: int = 3, freeze_base: bool = True):
        super().__init__()
        
        # Load the base YOLO model (e.g. yolo26s.pt or a trained best.pt)
        # Using ultralytics to load the model correctly
        try:
            yolo_wrapper = YOLO(pretrained)
            self.base_model = yolo_wrapper.model
            # Clean up the wrapper to avoid attribute conflicts (like .train())
            del yolo_wrapper
        except Exception as e:
            print(f"Error loading YOLO model via ultralytics: {e}")
            raise

        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # We use layer 16 (P3/8) for 3D heads as in yolo26.yaml
        self.c_p3 = self._get_p3_channels()
        
        # 3D Heads configuration
        self.heads_config = {
            'heatmap3d': num_classes,
            'offset_2d': 2,
            'size_2d': 2,
            'depth': 1,      # Monolite expects 1-channel depth
            'offset_3d': 2,
            'size_3d': 3,
            'heading': 24    # 12 bins + 12 residuals
        }
        
        for head, out_channels in self.heads_config.items():
            # Monodle style: Conv 3x3 -> ReLU -> Conv 1x1
            head_module = nn.Sequential(
                nn.Conv2d(self.c_p3, 256, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
            )
            
            # Specialized initialization
            if 'heatmap' in head:
                # CenterNet heatmap initialization
                head_module[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(head_module)
            
            self.add_module(head, head_module)

    def load_pretrained_backbone(self, pretrained_path: str, strict: bool = False):
        """
        Weights are already loaded in __init__ via ultralytics.
        This method is provided for compatibility with the training script.
        """
        print(f"Backbone already initialized from {pretrained_path} in __init__")
        return [], []

    def _get_p3_channels(self) -> int:
        """Helper to find the output channel count of the P3 layer (index 16)."""
        # Try to find it from attributes first
        layer16 = self.base_model.model[16]
        if hasattr(layer16, 'cv2') and hasattr(layer16.cv2, 'conv'):
            return layer16.cv2.conv.out_channels
        
        # Fallback: Dummy pass
        device = next(self.base_model.parameters()).device
        dummy_input = torch.zeros(1, 3, 128, 128).to(device)
        y = []
        x = dummy_input
        for i, m in enumerate(self.base_model.model):
            if m.f != -1:
                x_in = [y[j] for j in m.f] if isinstance(m.f, list) else y[m.f]
            else:
                x_in = x
            x = m(x_in)
            y.append(x)
            if i == 16:
                return x.shape[1]
        return 256 # Best guess fallback

    def fill_fc_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        # Forward through the base model until layer 16 (P3)
        y = []
        feat_p3 = None
        curr_x = x
        
        for i, m in enumerate(self.base_model.model):
            if m.f != -1:
                # Handle multiple inputs if necessary, although for layers 0-16 it's usually simple
                if isinstance(m.f, int):
                    x_in = y[m.f]
                else:
                    x_in = [curr_x if j == -1 else y[j] for j in m.f]
            else:
                x_in = curr_x
            
            curr_x = m(x_in)
            y.append(curr_x)
            
            if i == 16:
                feat_p3 = curr_x
                break
        
        outputs = {
             "features": {
                "p3": feat_p3,
                "neck": feat_p3,
            },
            "heatmap3d": self.heatmap3d(feat_p3),
            "depth": self.depth(feat_p3),
            "size_3d": self.size_3d(feat_p3),
            "heading": self.heading(feat_p3),
            "offset_3d": self.offset_3d(feat_p3),
            "offset_2d": self.offset_2d(feat_p3),
            "size_2d": self.size_2d(feat_p3),
        }
        
        return outputs
