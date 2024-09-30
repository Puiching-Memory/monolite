import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import torch
import yaml
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lib.datasets.openlane import OpenLane_dataset
from lib.helpers.model_helper import build_model
from lib.helpers.decode_helper import extract_dets_from_outputs,decode_detections
import cv2
import numpy as np

def main():
    cfg = yaml.load(open(r'/desay/file_warehouse/ids/upload/zk/monodle/kitti_val.yaml', 'r'), Loader=yaml.Loader)
    model_state = torch.load(r'/desay/file_warehouse/ids/upload/zk/monodle/checkpoints/checkpoint_epoch_140.pth',weights_only=True)
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg['model']).cuda(device=device)
    model = torch.nn.DataParallel(model,device_ids=[5])
    model.load_state_dict(model_state['model_state'],strict=True)
    model.eval()

    point_range = [90,6,3]
    test_dataset = OpenLane_dataset(r'/desay/file_warehouse/ids/upload/zk/3dlane_dataset/openlane/images',
                              r'/desay/file_warehouse/ids/upload/zk/3dlane_dataset/openlane/lane3d_1000/training',
                              point_range)
    point_range = torch.tensor([point_range])
    
    test_dataloader = DataLoader(test_dataset,batch_size=1,shuffle=True) 
    
    anchors = build_anchor([5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,45,60,75,90], [6,2,-2,-6]).T

    for index,(image,target,info) in enumerate(test_dataloader):
        with torch.no_grad():
            output = model(image)

        ouptput_p = output['line_reg'][0].reshape(3,-1).cpu().detach()      # 模型输出偏置
        target_p = target['distance'][0].cpu().detach()                  # 真正的偏置
        intrinsic = info['intrinsic'][0]
        dets = extract_dets_from_outputs(outputs=output, K=50)     # 模型其他头输出
        dets = dets.detach().cpu() # [Bathc,num,det] det:[cls,conf,x,y,w,h,depth,*12角度分区,*12角度回归,3D_j,3D_k,3D_l,3D_x,3D_y,?]
        dets = dets[0] # dets:[num,37]
        dets = decode_det(dets,intrinsic) # dets:[num,20] det:[cls,conf,score,box2d_x,box2d_y,d,n,x,b,depth,x3d,y3d,alpha,ry,dim*3,location*3]

        target_p = target_p * torch.tensor([[90,6,3]]).T
        ouptput_p = ouptput_p * torch.tensor([[90,6,3]]).T
        
        target_line = anchors + target_p
        ouptput_line = anchors + ouptput_p 
        fig = plt.figure()  
        ax = fig.add_subplot(111, projection='3d')  
        ax.scatter(target_line[0], target_line[1], target_line[2],c='green',marker='o')  # c=z 根据z值的不同来设置颜色 
        ax.scatter(anchors[0],anchors[1],anchors[2],c='black',marker='x') # 原始锚点
        ax.scatter(ouptput_line[0],ouptput_line[1],ouptput_line[2],c='red',marker='.') # 模型偏移后锚点

        for i in range(len(target_line[1])):  
            ax.plot([target_line[0, i], ouptput_line[0, i]],  
                    [target_line[1, i], ouptput_line[1, i]],  
                    [target_line[2, i], ouptput_line[2, i]],  
                    color='blue', linestyle='-')  # 使用蓝色实线连接每个对应的点  
            
        #ax.scatter(target_p[0],target_p[1],target_p[2],c='red',marker='x')
        ax.set_xlabel('X Label')  
        ax.set_ylabel('Y Label')  
        ax.set_zlabel('Z Label')  
        # 显示图形
        plt.savefig(r'/desay/file_warehouse/ids/upload/zk/monodle/point.jpg')
        
        #eval_point = info['raw_point'][0]
        x2d_t,y2d_t = ctn2image(target_line,intrinsic)
        x2d_org,y2d_org = ctn2image(anchors,intrinsic)
        x2d_p,y2d_p = ctn2image(ouptput_line,intrinsic)
        mean = torch.tensor((0.485, 0.456, 0.406)).view((1,-1,1,1))
        std = torch.tensor((0.229, 0.224, 0.225)).view((1,-1,1,1))
        image = torch.clip(image * std + mean ,0,None) * 255.0
        image = image[0].permute(1, 2, 0).numpy().copy()
        scale = (1920,1280) # 1280,384
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image,scale) 
        
        # 黑色锚点   
        for x,y in zip(x2d_org,y2d_org):
            x = int(x)
            y = int(y)
            cv2.circle(image, (x,y), 5, (0,0,0), -1)
            cv2.circle(image, (x,y), 5, (255,255,255), 1)
        
        # 蓝色真值
        x0,y0 = int(x2d_t[0]),int(y2d_t[0])
        x1,y1 = int(x2d_t[1]),int(y2d_t[1])
        x2,y2 = int(x2d_t[2]),int(y2d_t[2])
        x3,y3 = int(x2d_t[3]),int(y2d_t[3])
        for index,(x,y) in enumerate(zip(x2d_t,y2d_t)):
            if index % 4 == 0: # left left
                xp,yp = x0,y0
            elif index % 4 == 1: # left
                xp,yp = x1,y1
            elif index % 4 == 2: # right
                xp,yp = x2,y2
            elif index % 4 == 3: # right right
                xp,yp = x3,y3
            x = int(x)
            y = int(y)
            cv2.line(image,(xp,yp),(x,y),color=(255,255,0))
            cv2.circle(image, (x,y), 5, (255,0,0), -1)
            cv2.circle(image, (x,y), 5, (255,255,0), 1)
            if index % 4 == 0: # left left
                x0,y0 = x,y
            elif index % 4 == 1: # left
                x1,y1 = x,y
            elif index % 4 == 2: # right
                x2,y2 = x,y
            elif index % 4 == 3: # right right
                x3,y3 = x,y
        
        # 红色预测
        x0,y0 = int(x2d_p[0]),int(y2d_p[0])
        x1,y1 = int(x2d_p[1]),int(y2d_p[1])
        x2,y2 = int(x2d_p[2]),int(y2d_p[2])
        x3,y3 = int(x2d_p[3]),int(y2d_p[3])
        for index,(x,y) in enumerate(zip(x2d_p,y2d_p)):
            if index % 4 == 0: # left left
                xp,yp = x0,y0
            elif index % 4 == 1: # left
                xp,yp = x1,y1
            elif index % 4 == 2: # right
                xp,yp = x2,y2
            elif index % 4 == 3: # right right
                xp,yp = x3,y3
            x = int(x)
            y = int(y)
            cv2.line(image,(xp,yp),(x,y),color=(0,255,255))
            cv2.circle(image, (x,y), 5, (0,0,255), -1)
            cv2.circle(image, (x,y), 5, (0,255,255), 1)
            if index % 4 == 0: # left left
                x0,y0 = x,y
            elif index % 4 == 1: # left
                x1,y1 = x,y
            elif index % 4 == 2: # right
                x2,y2 = x,y
            elif index % 4 == 3: # right right
                x3,y3 = x,y
        
        # 绘制2D检测框
        for x,y,d,n,s,b in zip(dets[:,3],dets[:,4],dets[:,5],dets[:,6],dets[:,7],dets[:,8]):
            x1 = int(x-s)
            y1 = int(y-b)
            x2 = int(x+d)
            y2 = int(y+n)
            cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),2)
            
        # 绘制3D中心点
        for x,y in zip(dets[:,10],dets[:,11]):
            x = int(x)
            y = int(y)
            cv2.drawMarker(image,(x,y),(0,0,255),cv2.MARKER_TILTED_CROSS,10,5)

        # 绘制3D检测框
        intrinsic_one = torch.cat((intrinsic,torch.ones((3,1))),dim=1)
        for ry,l,w,h,t0,t1,t2 in dets[:,13:20]:
            box2d,box3d = compute_box_3d(ry,l,w,h,t0,t1,t2,intrinsic)
            #box2d = ctnyzx2imagexyz(box3d.T,intrinsic)
            if box2d == None:
                continue
            # 绘制顶点
            cv2.circle(image, (int(box2d[0][0]),int(box2d[1][0])), 5, (0,255,100), -1)
            cv2.circle(image, (int(box2d[0][1]),int(box2d[1][1])), 5, (0,255,100), -1)
            cv2.circle(image, (int(box2d[0][2]),int(box2d[1][2])), 5, (0,255,100), -1)
            cv2.circle(image, (int(box2d[0][3]),int(box2d[1][3])), 5, (0,255,100), -1)
            cv2.circle(image, (int(box2d[0][4]),int(box2d[1][4])), 5, (0,255,100), -1)
            cv2.circle(image, (int(box2d[0][5]),int(box2d[1][5])), 5, (0,255,100), -1)
            cv2.circle(image, (int(box2d[0][6]),int(box2d[1][6])), 5, (0,255,100), -1)
            cv2.circle(image, (int(box2d[0][7]),int(box2d[1][7])), 5, (0,255,100), -1)
            
            # 绘制地面线
            cv2.line(image,(int(box2d[0][0]),int(box2d[1][0])),(int(box2d[0][1]),int(box2d[1][1])),(200,255,100))
            cv2.line(image,(int(box2d[0][1]),int(box2d[1][1])),(int(box2d[0][2]),int(box2d[1][2])),(200,255,100))
            cv2.line(image,(int(box2d[0][2]),int(box2d[1][2])),(int(box2d[0][3]),int(box2d[1][3])),(200,255,100))
            cv2.line(image,(int(box2d[0][3]),int(box2d[1][3])),(int(box2d[0][0]),int(box2d[1][0])),(200,255,100))

            #绘制顶部线
            cv2.line(image,(int(box2d[0][4]),int(box2d[1][4])),(int(box2d[0][5]),int(box2d[1][5])),(200,255,100))
            cv2.line(image,(int(box2d[0][5]),int(box2d[1][5])),(int(box2d[0][6]),int(box2d[1][6])),(200,255,100))
            cv2.line(image,(int(box2d[0][6]),int(box2d[1][6])),(int(box2d[0][7]),int(box2d[1][7])),(200,255,100))
            cv2.line(image,(int(box2d[0][7]),int(box2d[1][7])),(int(box2d[0][4]),int(box2d[1][4])),(200,255,100))

            #绘制中格线
            cv2.line(image,(int(box2d[0][0]),int(box2d[1][0])),(int(box2d[0][4]),int(box2d[1][4])),(200,255,100))
            cv2.line(image,(int(box2d[0][1]),int(box2d[1][1])),(int(box2d[0][5]),int(box2d[1][5])),(200,255,100))
            cv2.line(image,(int(box2d[0][2]),int(box2d[1][2])),(int(box2d[0][6]),int(box2d[1][6])),(200,255,100))
            cv2.line(image,(int(box2d[0][3]),int(box2d[1][3])),(int(box2d[0][7]),int(box2d[1][7])),(200,255,100))

        cv2.imwrite(r'/desay/file_warehouse/ids/upload/zk/monodle/output.jpg', image)
        
        break

def decode_det(dets,intrinsic):
    conf_mask = (dets[:,1] > 0.2).nonzero().T[0]
    dets = dets[conf_mask]
    _dets = torch.empty((0,20))
    for det in dets:
        cls = det[0]
        conf = det[1]
        score = det[-1] * conf # TODO:这步做了什么?
        box2d_x = det[2] * 1.5 * 4
        box2d_y = det[3] * 3.333 * 4
        box2d_d = det[4] * 1.5 * 4
        box2d_n = det[5] * 3.333 * 4
        box2d_s = det[6] * 1.5 * 4
        box2d_b = det[7] * 3.333 * 4
        depth = det[8] # 深度
        dimensions = det[33:36] # 3D长宽高
        x3d = det[36] * 1.5 * 4 # 3D中心点,2D坐标
        y3d = det[37] * 3.333 * 4
        locations = img_to_rect(x3d,y3d,depth,intrinsic).reshape(-1) # 3D中心点,3D坐标
        locations[1] += dimensions[0] / 2 # TODO:这步做了什么?
        alpha = det[9:33] # 3D旋转角
        alpha = get_heading_angle(alpha)
        ry = alpha2ry(alpha,x3d,intrinsic)
        det_apart = torch.tensor([cls,conf,score,box2d_x,box2d_y,box2d_d,box2d_n,box2d_s,box2d_b,depth,x3d,y3d,alpha,ry])
        det_apart = torch.cat((det_apart,dimensions,locations)).unsqueeze(0)
        _dets = torch.cat((_dets,det_apart),dim=0)
    return _dets

def ctn2image(lane_xyz,intrinsic):
    imgae_lane = torch.vstack((lane_xyz, torch.ones((1, lane_xyz.shape[1]))))
    #print(imgae_lane)
    cam_representation = torch.tensor([[0, -1, 0, 0],
                            [0, 0, -1, 0],
                            [1, 0, 0, 0],
                            [0, 0, 0, 1]],dtype=torch.float32) # xyz->yzx

    imgae_lane = torch.matmul(cam_representation,imgae_lane)
    imgae_lane = imgae_lane[0:3,:]
    #print(imgae_lane)
    imgae_lane = torch.matmul(intrinsic,imgae_lane)
    x2d = imgae_lane[0,:] / imgae_lane[2,:]
    y2d = imgae_lane[1,:] / imgae_lane[2,:]
    #print(x2d,y2d)
    
    return x2d,y2d

def ctnyzx2imagexyz(poin_xyz,intrinsic):
    imgae_lane = torch.vstack((poin_xyz, torch.ones((1, poin_xyz.shape[1]))))
    #print(imgae_lane)
    # cam_representation = torch.tensor([[0, 1, 0, 0],
    #                         [0, 0, 1, 0],
    #                         [1, 0, 0, 0],
    #                         [0, 0, 0, 1]],dtype=torch.float32) # yzx -> zxy
    cam_representation = torch.tensor([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]],dtype=torch.float32) # yzx -> yzx

    imgae_lane = torch.matmul(cam_representation,imgae_lane)
    imgae_lane = imgae_lane[0:3,:]
    imgae_lane = torch.matmul(intrinsic,imgae_lane)
    x2d = imgae_lane[0,:] / imgae_lane[2,:]
    y2d = imgae_lane[1,:] / imgae_lane[2,:]
    #print(x2d,y2d)
    return x2d,y2d

def build_anchor(anchor_x,anchor_y,anchor_z=-2):
    points = []
    for x in anchor_x:
        for y in anchor_y:
            points.append((x,y,anchor_z))
    
    points = torch.tensor(points)
    return points

def get_heading_angle(heading):
    heading_bin, heading_res = heading[0:12], heading[12:24]
    cls = torch.argmax(heading_bin)
    res = heading_res[cls]
    return class2angle(cls, res, to_label_format=True)

def class2angle(cls, residual, to_label_format=False):
    ''' Inverse function to angle2class. '''
    num_heading_bin = 12  # hyper param
    angle_per_class = 2 * torch.pi / float(num_heading_bin)
    angle_center = cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle > torch.pi:
        angle = angle - 2 * torch.pi
    return angle

def alpha2ry(alpha, u, intrinsic):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    
    FIXME:out of date
    """
    cu = intrinsic[0,2]
    fu = intrinsic[0,0]
    ry = alpha + torch.arctan2(u - cu, fu)

    if ry > torch.pi:
        ry -= 2 * torch.pi
    if ry < -torch.pi:
        ry += 2 * torch.pi
        
    return ry

def img_to_rect(u, v, depth_rect,intrinsic):
    """
    :param u: (N)
    :param v: (N)
    :param depth_rect: (N)
    :return:
    FIXME:out of date
    """
    cu = intrinsic[0,2]
    cv = intrinsic[1,2]
    fu = intrinsic[0,0]
    fv = intrinsic[1,1]
    tx,ty = 0,0 # 相对基准相机偏移,如果为前视(主相机)则为0
    x = ((u - cu) * depth_rect) / fu + tx
    y = ((v - cv) * depth_rect) / fv + ty
    pts_rect = torch.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), dim=1)
    return pts_rect

def compute_box_3d(ry,w,h,l,t0,t1,t2,P):
    """ Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    """
    # compute rotational matrix around yaw axis
    R = roty(ry)

    # 3d bounding box corners
    x_corners = torch.tensor([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    y_corners = torch.tensor([0, 0, 0, 0, -h, -h, -h, -h])
    z_corners = torch.tensor([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])

    # rotate and translate 3d bounding box
    corners_3d = torch.matmul(R, torch.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + t0# Y
    corners_3d[1, :] = corners_3d[1, :] + t1# Z
    corners_3d[2, :] = corners_3d[2, :] + t2 # X
    # print 'cornsers_3d: ', corners_3d
    # only draw 3d bounding box for objs in front of the camera
    if torch.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return corners_2d, corners_3d.T

    # project the 3d bounding box into the image plane
    #corners_2d = project_to_image(corners_3d.T, P)
    corners_2d = ctnyzx2imagexyz(corners_3d,P)
    # print 'corners_2d: ', corners_2d
    #print(corners_2d,'\n',corners_3d)
    return corners_2d, corners_3d.T

def roty(t):
    """ Rotation about the y-axis. """
    c = torch.cos(t)
    s = torch.sin(t)
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def project_to_image(pts_3d, P):
    """ Project 3d points to image plane.

    Usage: pts_2d = projectToImage(pts_3d, P)
      input: pts_3d: nx3 matrix
             P:      3x4 projection matrix
      output: pts_2d: nx2 matrix

      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
      => normalize projected_pts_2d(2xn)

      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
    """
    n = pts_3d.shape[0]
    pts_3d_extend = torch.hstack((pts_3d, torch.ones((n, 1))))
    # print(('pts_3d_extend shape: ', pts_3d_extend.shape))
    pts_2d = torch.mm(pts_3d_extend, P.T)  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]

if __name__ == '__main__':
    main()