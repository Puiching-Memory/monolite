import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.helpers.decode_helper import _transpose_and_gather_feat
from lib.losses.focal_loss import focal_loss_cornernet,Poly1CrossEntropyLoss
from lib.losses.uncertainty_loss import laplacian_aleatoric_uncertainty_loss
from lib.losses.dim_aware_loss import dim_aware_l1_loss,piou
import time
import os
try:
    device = int(os.environ["LOCAL_RANK"])
except:
    device = 'cpu'

def compute_centernet3d_loss(input, target):
    start_time = time.time()
    stats_dict = {}
    
    seg_loss = compute_segmentation_loss(input, target) # heatmap
    offset2d_loss = compute_offset2d_loss(input, target)
    size2d_loss = compute_size2d_loss(input, target)
    offset3d_loss = compute_offset3d_loss(input, target)
    # box3d_loss = compute_rdiou_loss(input,target)

    depth_loss = compute_depth_loss(input, target)
    size3d_loss = compute_size3d_loss(input, target)
    heading_loss = compute_heading_loss(input, target)
    
    #bbox_loss = compute_Powerfuliou_loss(input,target)

    # statistics
    stats_dict['seg'] = seg_loss.item()
    stats_dict['offset2d'] = offset2d_loss.item()
    stats_dict['size2d'] = size2d_loss.item()
    stats_dict['offset3d'] = offset3d_loss.item()
    stats_dict['depth'] = depth_loss.item()
    stats_dict['size3d'] = size3d_loss.item()
    stats_dict['heading'] = heading_loss.item()
    stats_dict['losstime'] = time.time()-start_time
    total_loss = seg_loss + offset2d_loss + size2d_loss + offset3d_loss + \
                depth_loss + size3d_loss + heading_loss
    return total_loss, stats_dict
    
def compute_line3d_loss(input,target,anchor):
    start_time = time.time()
    stats_dict = {}
    
    lane_reg_loss = compute_lane_reg_loss(input,target,anchor)
    total_loss = lane_reg_loss
    
    stats_dict['seg'] = torch.tensor(0.0,requires_grad=True)
    stats_dict['offset2d'] = torch.tensor(0.0,requires_grad=True)
    stats_dict['size2d'] = torch.tensor(0.0,requires_grad=True)
    stats_dict['offset3d'] = torch.tensor(0.0,requires_grad=True)
    stats_dict['depth'] = torch.tensor(0.0,requires_grad=True)
    stats_dict['size3d'] = torch.tensor(0.0,requires_grad=True)
    stats_dict['heading'] = torch.tensor(0.0,requires_grad=True)
    stats_dict['losstime'] = time.time()-start_time
    return total_loss,stats_dict

#@torch.compile
def compute_lane_reg_loss(input,target,anchor):
    #huberloss = nn.SmoothL1Loss()
    #huberloss = nn.MSELoss()
    huberloss = nn.HuberLoss()
    # 计算锚点距离损失
    if (target['distance'] == 0).all():
        return torch.tensor(0.0,requires_grad=True)
    loss_dis = huberloss(input['line_reg'].view(-1),target['distance'].view(-1))
    loss_dis = loss_dis * 2000
    
    # 计算锚点间向量余弦相似度损失
    pred_point = (anchor + input['line_reg'].reshape(-1,3,76))
    coss = nn.CosineSimilarity()
    l_left_left = torch.tensor([i for i in range(0,76,4)],device=device)
    l_left = torch.tensor([i for i in range(1,76,4)],device=device)
    l_right = torch.tensor([i for i in range(2,76,4)],device=device)
    l_right_right = torch.tensor([i for i in range(3,76,4)],device=device)
    # B*3*76 依据车道线切片
    l_left_left = pred_point[:,:,l_left_left] # B*3*19
    l_left = pred_point[:,:,l_left] # B*3*19
    l_right = pred_point[:,:,l_right] # B*3*19
    l_right_right = pred_point[:,:,l_right_right] # B*3*19

    # 计算每个点之间的方向向量
    l_left_left = l_left_left[:,:,1:] - l_left_left[:,:,:-1] # B*3*18
    l_left = l_left[:,:,1:] - l_left[:,:,:-1] # B*3*18
    l_right = l_right[:,:,1:] - l_right[:,:,:-1] # B*3*18
    l_right_right = l_right_right[:,:,1:] - l_right_right[:,:,:-1] # B*3*18

    # 计算相邻两个向量之间的余弦相似度
    # B*3*17
    cos_left_left = coss(l_left_left[:,:,:-1],l_left_left[:,:,1:]) # B*17 每个向量的余弦相似度[-1~1]$
    cos_left = coss(l_left_left[:,:,:-1],l_left_left[:,:,1:]) 
    cos_right = coss(l_left_left[:,:,:-1],l_left_left[:,:,1:]) 
    cos_right_right = coss(l_left_left[:,:,:-1],l_left_left[:,:,1:])

    # 过滤掉>0.1的值
    mask_left_left = (cos_left_left<0.1).nonzero() # B*17
    mask_left = (cos_left<0.1).nonzero()
    mask_right = (cos_right<0.1).nonzero()
    mask_right_right = (cos_right_right<0.1).nonzero()

    cos_left_left = cos_left_left[mask_left_left[:,0],mask_left_left[:,1]]
    cos_left = cos_left[mask_left[:,0],mask_left[:,1]]
    cos_right = cos_right[mask_right[:,0],mask_right[:,1]]
    cos_right_right = cos_right_right[mask_right_right[:,0],mask_right_right[:,1]]

    # 依据余弦相似度求回归距离损失
    loss_left_left = huberloss(cos_left_left,torch.fill(cos_left_left,0.1))
    loss_left = huberloss(cos_left,torch.fill(cos_left,0.1))
    loss_right = huberloss(cos_right,torch.fill(cos_right,0.1))
    loss_right_right = huberloss(cos_right_right,torch.fill(cos_right_right,0.1))

    loss_deg = torch.mean(torch.tensor((loss_left_left,loss_left,loss_right,loss_right_right),device=device))
    loss_deg *= 10
    #return loss_dis + loss_deg
    return loss_dis

def compute_Powerfuliou_loss(input,target):
    heatmap = torch.clamp(input['heatmap'].sigmoid_(), min=1e-4, max=1 - 1e-4) # B*3*96*320
    
    size2d_input = extract_input_from_tensor(input['size_2d'], target['indices'], target['mask_2d']) # num*4
    size2d_target = extract_target_from_tensor(target['size_2d'], target['mask_2d']) # num*4 #东南西北
    
    offset2d_input = extract_input_from_tensor(input['offset_2d'], target['indices'], target['mask_2d']) # num*2
    offset2d_target = extract_target_from_tensor(target['offset_2d'], target['mask_2d']) # num*2
    
    K = offset2d_input.shape[0] # num
    batch, channel, height, width = heatmap.size() # get shape
    scores, inds, cls_ids, xs, ys = _topk(heatmap, K=K)
    
    offset2d_input = offset2d_input.view(batch, K, 2) # B*K*2
    xs2d = xs.view(batch, K, 1) + offset2d_input[:, :, 0:1] # B*K*1
    ys2d = ys.view(batch, K, 1) + offset2d_input[:, :, 1:2] # B*K*1
    
    print('heatmap',heatmap.shape,'size2d_input',size2d_input.shape,'size2d_target',size2d_target.shape,'offset2d_input',offset2d_input.shape,'offset2d_target',offset2d_target.shape)

@torch.compile
def compute_segmentation_loss(input, target):
    input['heatmap'] = torch.clamp(input['heatmap'].sigmoid_(), min=1e-4, max=1 - 1e-4)
    loss = focal_loss_cornernet(input['heatmap'], target['heatmap'])
    #loss = quality_focal_loss(input['heatmap'], target['heatmap'])
    return loss

@torch.compile
def compute_size2d_loss(input, target):
    # compute size2d loss
    size2d_input = extract_input_from_tensor(input['size_2d'], target['indices'], target['mask_2d']) # num*4
    size2d_target = extract_target_from_tensor(target['size_2d'], target['mask_2d']) # num * 4
    size2d_loss = F.l1_loss(size2d_input, size2d_target, reduction='mean')

    # zeros = torch.zeros_like(size2d_input[:,0])
    # input_box = torch.tensor((zeros,size2d_input[:,3]+size2d_input[:,1],size2d_input[:,0]+size2d_input[:,2],zeros)).T
    # target_box = torch.tensor((zeros,size2d_target[:,3]+size2d_target[:,1],size2d_target[:,0]+size2d_target[:,2],zeros)).T
    # print(input_box,target_box)
    # size2d_loss = piou(input_box,target_box,PIoU2=True)
    # print(size2d_loss)
    # size2d_loss = torch.mean(size2d_loss)
    # print(size2d_loss)

    return size2d_loss

@torch.compile
def compute_offset2d_loss(input, target):
    # compute offset2d loss
    offset2d_input = extract_input_from_tensor(input['offset_2d'], target['indices'], target['mask_2d'])
    offset2d_target = extract_target_from_tensor(target['offset_2d'], target['mask_2d'])
    offset2d_loss = F.l1_loss(offset2d_input, offset2d_target, reduction='mean')
    return offset2d_loss

@torch.compile
def compute_depth_loss(input, target):
    depth_input = extract_input_from_tensor(input['depth'], target['indices'], target['mask_3d'])
    depth_input, depth_log_variance = depth_input[:, 0:1], depth_input[:, 1:2]
    depth_input = 1. / (depth_input.sigmoid() + 1e-6) - 1.
    depth_target = extract_target_from_tensor(target['depth'], target['mask_3d'])
    depth_loss = laplacian_aleatoric_uncertainty_loss(depth_input, depth_target, depth_log_variance) # B * 1
    return depth_loss

@torch.compile
def compute_offset3d_loss(input, target):
    offset3d_input = extract_input_from_tensor(input['offset_3d'], target['indices'], target['mask_3d'])
    offset3d_target = extract_target_from_tensor(target['offset_3d'], target['mask_3d'])
    #offset3d_loss = F.l1_loss(offset3d_input, offset3d_target, reduction='mean')
    offset3d_loss = F.smooth_l1_loss(offset3d_input, offset3d_target, reduction='mean')
    return offset3d_loss

@torch.compile
def compute_size3d_loss(input, target):
    size3d_input = extract_input_from_tensor(input['size_3d'], target['indices'], target['mask_3d'])
    size3d_target = extract_target_from_tensor(target['size_3d'], target['mask_3d'])
    size3d_loss = dim_aware_l1_loss(size3d_input, size3d_target, size3d_target) # B * 3
    return size3d_loss

#@torch.compile
def compute_heading_loss(input, target):
    heading_input = _transpose_and_gather_feat(input['heading'], target['indices'])   # B * C * H * W ---> B * K * C
    heading_input = heading_input.view(-1, 24)
    heading_target_cls = target['heading_bin'].view(-1)
    heading_target_res = target['heading_res'].view(-1)
    mask = target['mask_2d'].view(-1)
    mask = mask.bool() # uint8 is deprecated

    # classification loss
    heading_input_cls = heading_input[:, 0:12]
    heading_input_cls, heading_target_cls = heading_input_cls[mask], heading_target_cls[mask]
    if mask.sum() > 0:
        polyloss = Poly1CrossEntropyLoss(12,reduction='mean')
        cls_loss = polyloss(heading_input_cls, heading_target_cls)
        #cls_loss = F.cross_entropy(heading_input_cls, heading_target_cls, reduction='mean')
    else:
        cls_loss = 0.0
    
    # regression loss
    heading_input_res = heading_input[:, 12:24]
    heading_input_res, heading_target_res = heading_input_res[mask], heading_target_res[mask]
    cls_onehot = torch.zeros(heading_target_cls.shape[0], 12).cuda().scatter_(dim=1, index=heading_target_cls.view(-1, 1), value=1)
    heading_input_res = torch.sum(heading_input_res * cls_onehot, 1)
    reg_loss = F.l1_loss(heading_input_res, heading_target_res, reduction='mean')
    return cls_loss + reg_loss


######################  auxiliary functions #########################
def _topk(heatmap, K=50):
    batch, cat, height, width = heatmap.size()

    # batch * cls_ids * 50
    topk_scores, topk_inds = torch.topk(heatmap.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    # batch * cls_ids * 50
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_cls_ids = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_cls_ids, topk_xs, topk_ys

def _gather_feat(feat, ind, mask=None):
    '''
    Args:
        feat: tensor shaped in B * (H*W) * C
        ind:  tensor shaped in B * K (default: 50)
        mask: tensor shaped in B * K (default: 50)

    Returns: tensor shaped in B * K or B * sum(mask)
    '''
    dim  = feat.size(2)  # get channel dim
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)  # B*len(ind) --> B*len(ind)*1 --> B*len(ind)*C
    feat = feat.gather(1, ind)  # B*(HW)*C ---> B*K*C
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)  # B*50 ---> B*K*1 --> B*K*C
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _nms(heatmap, kernel=3):
    padding = (kernel - 1) // 2
    heatmapmax = nn.functional.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=padding)
    keep = (heatmapmax == heatmap).float()
    return heatmap * keep

def extract_input_from_tensor(input, ind, mask):
    input = _transpose_and_gather_feat(input, ind)  # B*C*H*W --> B*K*C
    mask = mask.bool() # uint8 is deprecated
    return input[mask]  # B*K*C --> M * C

def extract_target_from_tensor(target, mask):
    mask = mask.bool() # uint8 is deprecated
    return target[mask]


if __name__ == '__main__':
    input_cls  = torch.zeros(2, 50, 12)  # B * 50 * 24
    input_reg  = torch.zeros(2, 50, 12)  # B * 50 * 24
    target_cls = torch.zeros(2, 50, 1, dtype=torch.int64)
    target_reg = torch.zeros(2, 50, 1)

    input_cls, target_cls = input_cls.view(-1, 12), target_cls.view(-1)
    cls_loss = F.cross_entropy(input_cls, target_cls, reduction='mean')

    a = torch.zeros(2, 24, 10, 10)
    b = torch.zeros(2, 10).long()
    c = torch.ones(2, 10).long()
    d = torch.zeros(2, 10, 1).long()
    e = torch.zeros(2, 10, 1)
    print(compute_heading_loss(a, b, c, d, e))

