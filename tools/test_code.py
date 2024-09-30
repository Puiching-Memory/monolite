import torch

lane = torch.tensor(([1,2,3,4],[5,6,7,8],[9,10,11,12]))

def ctn2image(lane_xyz):
    imgae_lane = torch.vstack((lane_xyz, torch.ones((1, lane_xyz.shape[1]))))
    cam_representation = torch.tensor([[0, 0, 1, 0],
                            [-1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, 0, 1]],dtype=torch.float32)

    imgae_lane = torch.matmul(cam_representation,imgae_lane)
    # imgae_lane = imgae_lane[0:3,:]
    # imgae_lane = torch.matmul(intrinsic,imgae_lane)
    # x2d = imgae_lane[0,:] / imgae_lane[2,:]
    # y2d = imgae_lane[1,:] / imgae_lane[2,:]

    return imgae_lane

print(lane)
print(ctn2image(lane))
