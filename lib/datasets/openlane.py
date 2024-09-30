import orjson
import os
import torch
import torchvision
from torch.utils.data import Dataset
import time
from tqdm import tqdm
from rich.progress import track
from concurrent import futures
# from numba import jit,njit
# from annoy import AnnoyIndex
import sys
import random
import pickle
# from sklearn.neighbors import NearestNeighbors
from annoy import AnnoyIndex


class OpenLane_dataset(Dataset):

    def __init__(self, image_path, 
                   label_path,
                   point_range,
                   use_pkl = True,
                   ):
        self.lane_type = {0: 'unkown',
                        1: 'white-dash',
                        2: 'white-solid',
                        3: 'double-white-dash',
                        4: 'double-white-solid',
                        5: 'white-ldash-rsolid',
                        6: 'white-lsolid-rdash',
                        7: 'yellow-dash',
                        8: 'yellow-solid',
                        9: 'double-yellow-dash',
                        10: 'double-yellow-solid',
                        11: 'yellow-ldash-rsolid',
                        12: 'yellow-lsolid-rdash',
                        20: 'left-curbside',
                        21: 'right-curbside'}
        torch.compiler.allow_in_graph(os.listdir)
        torch.compiler.allow_in_graph(sys.audit)
        torch.compiler.allow_in_graph(os.fspath)
        
        self.use_pkl = use_pkl
        self.image_path = image_path #路径
        self.label_path = label_path #路径
        self.point_range = torch.tensor([point_range]).T # [3,1]
        self.anchors = build_anchor([5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,45,60,75,90], [6,2,-2,-6]).T / self.point_range # [3,0]
        
        self.label_stash = list_subdirectories(label_path) # 提取所有子文件夹

        # 提取所有json绝对路径
        temp_l = []
        for file_path in self.label_stash:
            for json_file in os.listdir(file_path): # xxx.json
                temp_l.append(f'{file_path}/{json_file}')
        self.label_stash = temp_l

        # NOTE: 并发生成PKL
        # tasks, results = [], []
        # with futures.ProcessPoolExecutor(max_workers=64) as executor:
        #     for n in range(len(self.label_stash)):
        #         tasks.append(executor.submit(self.build_pkl, n))
        #     task_pbar = tqdm(futures.as_completed(tasks), total=len(tasks))
        #     for index,task in enumerate(task_pbar):
        #         results.append(task.result())
        #         task_pbar.set_description(f'build pkl iter>>>{index}')
        #         task_pbar.update()
        # self.label_stash = results
        
        #self.find_best_anchor()
        #self.build_pkl()
        
    @torch.compile()
    def find_best_anchor(self):
        lane_max_all = torch.empty((3,0))
        lane_min_all = torch.empty((3,0))
        for i in tqdm(range(int(len(self.label_stash) * 1.0)),'find best anchor'):
            index = random.randint(0,len(self.label_stash)-1)
            label_file_path = self.label_stash[index]
            with open(label_file_path) as label:
                label = orjson.loads(label.read())
            lane_lines = label['lane_lines'] # many lanes
            if len(lane_lines) == 0: continue
            lane_xyz = torch.empty((3,0))
            for index,lane in enumerate(lane_lines):
                lane_attribute = lane['attribute']
                if lane_attribute == 0: continue
                lane_xyz = torch.cat((lane_xyz,torch.tensor(lane['xyz'])),dim=1)
            if lane_xyz.numel() == 0:continue
            lane_max = torch.amax(lane_xyz,dim=1).unsqueeze(1)
            lane_min = torch.amin(lane_xyz,dim=1).unsqueeze(1)
            lane_max_all = torch.cat((lane_max_all,lane_max),dim=1)
            lane_min_all = torch.cat((lane_min_all,lane_min),dim=1)
        
        lane_max = torch.amax(lane_max_all,dim=1)
        lane_min = torch.amin(lane_min_all,dim=1)
        lane_mean_max = torch.mean(lane_max_all,dim=1)
        lane_mean_min = torch.mean(lane_min_all,dim=1)
        lane_median_max = torch.median(lane_max_all,dim=1).values
        lane_median_min = torch.median(lane_min_all,dim=1).values
        
        range_x = torch.linspace(lane_median_max[0],lane_median_min[0],6)
        range_y = torch.linspace(lane_median_max[1],lane_median_min[1],4)
        suggest_anchor = [range_x,range_y,lane_median_min[2]-lane_median_max[2]]
        print(f'found max max {lane_max} min min {lane_min}')
        print(f'found mean max {lane_mean_max} min {lane_mean_min}')
        print(f'found median max {lane_median_max} min {lane_median_min}')
        print(f'max item {lane_max_all.shape} min item {lane_min_all.shape}')
        print(f'suggest anchor {suggest_anchor}')
        
    def build_pkl(self,index):
        cache_path = r'/desay/file_warehouse/ids/upload/zk/3dlane_dataset/openlane/cache'
        if os.path.exists(f'{cache_path}/{index}.pkl'):
            #os.remove(f'{cache_path}/{index}.pkl')
            #return
            pass
        data = self.__getitem__(index)
        with open(f'{cache_path}/{index}.pkl','wb') as file:
            pickle.dump(data,file,pickle.HIGHEST_PROTOCOL)

    def load_json(self,json_list):
        with open(json_list,'r') as label:
            decode_json = orjson.loads(label.read())
        return decode_json
    
    #@torch.compile()
    def __getitem__(self, index):
        start_time = time.time()
        #print('start data',start_time,index)
        if self.use_pkl == True:
            with open(f'/desay/file_warehouse/ids/upload/zk/3dlane_dataset/openlane/cache/{index}.pkl','rb') as pkl:
                #image,targets,info = torch.load(pkl,weights_only=False)
                image,targets,info = pickle.load(pkl)
                info['get_item_time'] = time.time()-start_time
                return image,targets,info
            
        label_file_path = self.label_stash[index]
        with open(label_file_path) as label:
            label = orjson.loads(label.read())

        intrinsic = torch.tensor(label['intrinsic'])
        extrinsic = torch.tensor(label['extrinsic'])
        lane_lines = label['lane_lines'] # many lanes
        file_path = label['file_path'].split('/',1)[1] #training/segment-xxx/xxx.jpg
        #pose = label['pose']

        image = torchvision.io.read_image(f'{self.image_path}/{file_path}').to(dtype=torch.float)
        img_resize = torchvision.transforms.Resize((384,1280))
        img_normalize = torchvision.transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
        image = img_resize(image) 
        image = image / 255.0
        image = img_normalize(image) 

        if len(lane_lines) == 0: 
            return image,{'distance':torch.zeros_like(self.anchors).float()},{'get_item_time':time.time()-start_time,'intrinsic':intrinsic,'extrinsic':extrinsic}
        
        lane_left_left =    torch.empty((3,0))
        lane_left =         torch.empty((3,0))
        lane_right_right =  torch.empty((3,0))
        lane_right =        torch.empty((3,0))
        lane_all =          torch.empty((3,0))
        # 提取车道线xyz
        for index,lane in enumerate(lane_lines):
            #lane_category = lane['category']
            lane_visibility = torch.tensor(lane['visibility']).nonzero().T[0]
            #lane_uv =None
            lane_xyz = torch.tensor(lane['xyz'])
            lane_xyz = lane_xyz[:,lane_visibility]
            lane_attribute = lane['attribute']
            #lane_track_id = None
            if lane_attribute == 1:
                lane_left_left = torch.cat((lane_left_left,lane_xyz),dim=1)
            elif lane_attribute == 2:
                lane_left = torch.cat((lane_left,lane_xyz),dim=1)
            elif lane_attribute == 3:
                lane_right = torch.cat((lane_right,lane_xyz),dim=1)
            elif lane_attribute == 4:
                lane_right_right = torch.cat((lane_right_right,lane_xyz),dim=1)
                
        lane_all = torch.cat((lane_all,lane_left_left,lane_left,lane_right,lane_right_right),dim=1)

        if lane_all.numel() == 0: 
            return image,{'distance':torch.zeros_like(self.anchors).float()},{'get_item_time':time.time()-start_time,'intrinsic':intrinsic,'extrinsic':extrinsic}

        #lane_small = range_flitter(lane_all,self.point_range) 
        #lane_left_left_s = range_flitter(lane_left_left,self.point_range) 
        #lane_left_s = range_flitter(lane_left,self.point_range)
        #lane_right_s = range_flitter(lane_right,self.point_range)
        #lane_right_right_s = range_flitter(lane_right_right,self.point_range)
        
        #lane_small = lane_small / self.point_range #FIXME:会导致坐标轴错误
        llls = lane_left_left / self.point_range # xyz归一化
        lls = lane_left / self.point_range
        lrrs = lane_right_right / self.point_range
        lrs = lane_right / self.point_range
                        
        # 依据锚点寻找最近点/最近点xyz偏置
        near_points = torch.empty((3,0))
        distance = torch.empty((3,0))
        for index,anchor_point in enumerate(self.anchors.T):
            new_find = anchor_point.unsqueeze(1)
            if index % 4 == 0:  # left_left
                if llls.numel() != 0:
                    kdt = AnnoyIndex(2, 'euclidean') #仅依据x,y距离寻找
                    for indexl,point in enumerate(llls.T):
                        kdt.add_item(indexl+1,point[0:2])
                    kdt.add_item(0,anchor_point[0:2])
                    kdt.build(10)
                    new_find = kdt.get_nns_by_item(0,2)[1] - 1
                    new_find = llls.T[new_find].unsqueeze(1)
                near_points = torch.cat((near_points,new_find),dim=1)
                distance = torch.cat((distance,new_find - anchor_point.unsqueeze(1)),dim=1)
            elif index % 4 == 1: # left
                if lls.numel() != 0:
                    kdt = AnnoyIndex(2, 'euclidean') #仅依据x,y距离寻找
                    for indexl,point in enumerate(lls.T):
                        kdt.add_item(indexl+1,point[0:2])
                    kdt.add_item(0,anchor_point[0:2])
                    kdt.build(10)
                    new_find = kdt.get_nns_by_item(0,2)[1] - 1
                    new_find = lls.T[new_find].unsqueeze(1)
                near_points = torch.cat((near_points,new_find),dim=1)
                distance = torch.cat((distance,new_find - anchor_point.unsqueeze(1)),dim=1)
            elif index % 4 == 2: # right
                if lrs.numel() != 0:
                    kdt = AnnoyIndex(2, 'euclidean') #仅依据x,y距离寻找
                    for indexl,point in enumerate(lrs.T):
                        kdt.add_item(indexl+1,point[0:2])
                    kdt.add_item(0,anchor_point[0:2])
                    kdt.build(10)
                    new_find = kdt.get_nns_by_item(0,2)[1] - 1
                    new_find = lrs.T[new_find].unsqueeze(1)
                near_points = torch.cat((near_points,new_find),dim=1)
                distance = torch.cat((distance,new_find - anchor_point.unsqueeze(1)),dim=1)
            elif index % 4 == 3: # right_right
                if lrrs.numel() != 0:
                    kdt = AnnoyIndex(2, 'euclidean') #仅依据x,y距离寻找
                    for indexl,point in enumerate(lrrs.T):
                        kdt.add_item(indexl+1,point[0:2])
                    kdt.add_item(0,anchor_point[0:2])
                    kdt.build(10)
                    new_find = kdt.get_nns_by_item(0,2)[1] - 1
                    new_find = lrrs.T[new_find].unsqueeze(1)
                near_points = torch.cat((near_points,new_find),dim=1)
                distance = torch.cat((distance,new_find - anchor_point.unsqueeze(1)),dim=1)

        targets = {'distance':distance}
        info = {'get_item_time':time.time()-start_time,
                'intrinsic':intrinsic,
                'extrinsic':extrinsic,
                #'raw_point':lane_all,
                }

        return image,targets,info

            
    def __len__(self):
        return len(self.label_stash)


def range_flitter(xyz,point_range):
    xyz = torch.abs(xyz)
    exceeds_threshold = (xyz < point_range).all(dim=0).nonzero().T[0]
    xyz = xyz[:,exceeds_threshold] 
    return xyz

def build_anchor(anchor_x,anchor_y,anchor_z=-2):
    points = []
    for x in anchor_x:
        for y in anchor_y:
            points.append((x,y,anchor_z))
    
    points = torch.tensor(points)
    return points

def find_nearest_point(query_point, kdt):  
    """  
    在点集points中寻找与query_point最近的点。  
  
    :param query_point: torch.Tensor，形状为(3,)，表示一个三维空间中的点。  
    :param points: torch.Tensor，形状为(N, 3)，表示N个三维空间中的点。  
    :return: torch.Tensor，形状为(3,)，表示找到的最近点。  
    """  
    if kdt == None: # 如果待查点集为None,则返回请求点自身
        return query_point
    query_point = query_point.unsqueeze(0)
    print(query_point)
    distances, indices = kdt.kneighbors(query_point)
    print(indices)
    
    return indices


def list_subdirectories(directory):  
    subdirs = []  
    for root, dirs, files in os.walk(directory):  
        for dir in dirs:  
            subdir = os.path.join(root, dir)  
            subdirs.append(subdir)  
    return subdirs

def ctn2image(lane_xyz,intrinsic):
    imgae_lane = torch.vstack((lane_xyz, torch.ones((1, lane_xyz.shape[1]))))
    cam_representation = torch.tensor([[0, -1, 0, 0],
                            [0, 0, -1, 0],
                            [1, 0, 0, 0],
                            [0, 0, 0, 1]],dtype=torch.float32)

    imgae_lane = torch.matmul(cam_representation,imgae_lane)
    imgae_lane = imgae_lane[0:3,:]
    imgae_lane = torch.matmul(intrinsic,imgae_lane)
    x2d = imgae_lane[0,:] / imgae_lane[2,:]
    y2d = imgae_lane[1,:] / imgae_lane[2,:]

    return x2d,y2d

if __name__ == "__main__":
    ''' parameter from config '''
    from torch.utils.data import DataLoader,DistributedSampler
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import cv2
    import numpy as np

    point_range = [90,6,3]

    test_dataset = OpenLane_dataset(r'/desay/file_warehouse/ids/upload/zk/3dlane_dataset/openlane/images',
                              r'/desay/file_warehouse/ids/upload/zk/3dlane_dataset/openlane/lane3d_1000/training',
                              point_range)
    
    point_range = torch.tensor([point_range])

    test_dataload = DataLoader(test_dataset,batch_size=1,shuffle=True)
    anchors = build_anchor([5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,45,60,75,90], [6,2,-2,-6]).T
    
    for index,(image,target,info) in enumerate(test_dataload):
        target_p = target['distance'][0].cpu().detach()                  # 真正的偏置
        ouptput_line = info['raw_point'][0].cpu().detach()

        target_p = target_p * torch.tensor([[90,6,3]]).T
        #ouptput_line = ouptput_line * torch.tensor([[90,6,3]]).T
        target_line = anchors + target_p

        fig = plt.figure()  
        ax = fig.add_subplot(111, projection='3d')  
        ax.scatter(target_line[0], target_line[1], target_line[2],c='green',marker='o')  # c=z 根据z值的不同来设置颜色 
        ax.scatter(anchors[0],anchors[1],anchors[2],c='black',marker='x') # 原始锚点
        ax.scatter(ouptput_line[0],ouptput_line[1],ouptput_line[2],c='red',marker='.') # 模型偏移后锚点

        # for i in range(len(target_line[1])):  
        #     ax.plot([target_line[0, i], ouptput_line[0, i]],  
        #             [target_line[1, i], ouptput_line[1, i]],  
        #             [target_line[2, i], ouptput_line[2, i]],  
        #             color='blue', linestyle='-')  # 使用蓝色实线连接每个对应的点  
            
        #ax.scatter(target_p[0],target_p[1],target_p[2],c='red',marker='x')
        ax.set_xlabel('X Label')  
        ax.set_ylabel('Y Label')  
        ax.set_zlabel('Z Label')  
        # 显示图形
        plt.savefig(r'/desay/file_warehouse/ids/upload/zk/monodle/point.jpg')
        
        #eval_point = info['raw_point'][0]
        intrinsic = info['intrinsic'][0]
        x2d_t,y2d_t = ctn2image(target_line,intrinsic)
        x2d_org,y2d_org = ctn2image(anchors,intrinsic)
        x2d_p,y2d_p = ctn2image(ouptput_line,intrinsic)
        image = image[0].permute(1, 2, 0).numpy().copy() * 255.0
        scale = (1920,1280)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image,scale) 
        # 黑色锚点   
        for x,y in zip(x2d_org,y2d_org):
            x = int(float(x.item()))
            y = int(float(y.item()))
            cv2.circle(image, (x,y), 5, (0,0,0), -1)
            cv2.circle(image, (x,y), 5, (255,255,255), 1)
                        
        # 红色预测
        for x,y in zip(x2d_p,y2d_p):
            x = int(float(x.item()))
            y = int(float(y.item()))
            cv2.circle(image, (x,y), 5, (0,0,255), -1)
            cv2.circle(image, (x,y), 5, (0,255,255), 1)    
            
        # 蓝色真值
        for x,y in zip(x2d_t,y2d_t):
            x = int(float(x.item()))
            y = int(float(y.item()))
            cv2.circle(image, (x,y), 7, (255,0,0), -1)
            cv2.circle(image, (x,y), 7, (255,255,0), 1)      
              
        cv2.imwrite(r'/desay/file_warehouse/ids/upload/zk/monodle/output.jpg', image)
        break