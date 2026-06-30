import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
from lib.model.nets.generator.drr_projector_new import DRRProjector
from lib.dataset.data_augmentation_1030 import CT_XRAY_Data_Test
from PIL import Image
import torch
import os

def save_tensor_as_image(tensor, filepath, normalize=True):
    """
    Saves a 1x1xH×W tensor as a grayscale image.
    
    Args:
        tensor (torch.Tensor): shape (1, 1, H, W)
        filepath (str): Path to save the image (e.g., 'output.png')
        normalize (bool): Whether to normalize to [0, 255]
    """
    assert tensor.dim() == 4 and tensor.shape[0] == 1 and tensor.shape[1] == 1, \
        "Tensor shape must be (1, 1, H, W)"
    
    tensor = tensor.squeeze().detach().cpu()  # shape: H x W
    
    if normalize:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)  # [0,1]
        tensor = (tensor * 255).clamp(0, 255).byte()

    img = Image.fromarray(tensor.numpy(), mode='L')  # 'L' for grayscale
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    img.save(filepath)


if __name__ == '__main__':
    import torchvision
    import cv2
    import h5py

    test_file = '/home/work/Skull/Datasets/LIDC-HDF5-256/LIDC-HDF5-256/LIDC-IDRI-0001.20000101.3000566.1/ct_xray_data.h5'
    
    hdf = h5py.File(test_file, 'r')
    ct = np.asarray(hdf['ct'])
    xray = np.asarray(hdf['xray1'])
    xray = np.expand_dims(xray, 0)

    from lib.config.config import cfg, merge_dict_and_yaml
    opt = merge_dict_and_yaml(dict(), cfg)
    opt.fine_size, opt.ct_channel = 128, 128
    transform_normal = CT_XRAY_Data_Test(opt)
    ct_normal, xray_normal = transform_normal([ct, xray])

    print(ct_normal.shape, xray_normal.shape)  # Check shapes torch.Size([128, 128, 128]) torch.Size([1, 128, 128])

    projector = DRRProjector().cuda()
    directions = {
                'AP': (0.0, 0.0, 0.0),
                'PA': (0.0, 3.1416, 0.0),  # 180도 Y축 회전
                'RL': (0.0, -1.5708, 0.0),  # -90도 Y축 회전
                'LL': (0.0, 1.5708, 0.0),   # 90도 Y축 회전
                'SI': (-1.5708, 0.0, 0.0),  # -90도 X축 회전
                'IS': (1.5708, 0.0, 0.0),   # 90도 X축 회전
            }
    ct_normal = ct_normal.unsqueeze(0).unsqueeze(0).cuda()  # Add batch and channel dimensions
    device = ct_normal.device
    dtype = ct_normal.dtype

    for name, angles in directions.items():
        # 각 배치에 대해 동일한 회전 파라미터
        theta = torch.tensor([angles], device=device, dtype=dtype).repeat(1, 1)
        transform_param = torch.cat([theta, torch.zeros_like(theta)], dim=1)  # rotation + translation

        # DRR projection
        gt_proj = projector(ct_normal, transform_param=transform_param)

        print(gt_proj.shape)  # Output shape should be (batch_size, num_views, height, width) torch.Size([1, 1, 128, 128])
        save_tensor_as_image(gt_proj, f'demo/{name}_DRR.png')
    # projector = Backprojector().cuda()
    # output = projector(input1, input2)

    # volume = output.squeeze(0).cpu().numpy() # (1, 128, 128, 128) ll_feat.permute(0, 1, 3, 2)
    # volume = volume[:, ::-1, :, :]
    # # print(output.shape) 
    # save(volume.squeeze(), spacing=(1.0, 1.0, 1.0), origin=(0,0,0), path="demo1.mha")

