import torch
import torch.nn as nn
import torch.nn.functional as F

class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
    
class HSigmoidv2(nn.Module):
    def __init__(self, inplace: bool = False):
        
        super().__init__()

        self.inplace = inplace

    def forward(self, x: torch.Tensor):
        out = F.relu6((x + 3.), inplace=self.inplace) / 6.
        return out

def autopad(k, p=None, d=1):
    """
    Pads kernel to 'same' output shape, adjusting for optional dilation; returns padding size.

    `k`: kernel, `p`: padding, `d`: dilation.
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class ASPP(nn.Module):
    def __init__(self, in_channel, depth):
        super(ASPP,self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
 
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
 
    def forward(self, x):
        size = x.shape[2:]
 		# mean.shape = torch.Size([8, 3, 1, 1])
        image_features = self.mean(x)
        # conv.shape = torch.Size([8, 3, 1, 1])
        image_features = self.conv(image_features)
        # upsample.shape = torch.Size([8, 3, 32, 32])
        image_features = F.interpolate(image_features, size=size, mode='bilinear')
 		
 		# block1.shape = torch.Size([8, 3, 32, 32])
        atrous_block1 = self.atrous_block1(x)
 		
 		# block6.shape = torch.Size([8, 3, 32, 32])
        atrous_block6 = self.atrous_block6(x)
 		
 		# block12.shape = torch.Size([8, 3, 32, 32])
        atrous_block12 = self.atrous_block12(x)
 		
 		# block18.shape = torch.Size([8, 3, 32, 32])
        atrous_block18 = self.atrous_block18(x)
 		
 		# torch.cat.shape = torch.Size([8, 15, 32, 32])
 		# conv_1x1.shape = torch.Size([8, 3, 32, 32])
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net