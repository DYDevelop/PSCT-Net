import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
from lib.model.nets.generator.drr_projector_new import DRRProjector

def get_6dofs_transformation_matrix(u, v):
    """ https://arxiv.org/pdf/1611.10336.pdf
    """
    x, y, z = u
    theta_x, theta_y, theta_z = v

    # rotate theta_z
    rotate_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0, 0],
        [np.sin(theta_z), np.cos(theta_z), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # rotate theta_y
    rotate_y = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y), 0],
        [0, 1, 0, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y), 0],
        [0, 0, 0, 1]
    ])

    # rotate theta_x and translate x, y, z
    rotate_x_translate_xyz = np.array([
        [1, 0, 0, x],
        [0, np.cos(theta_x), -np.sin(theta_x), y],
        [0, np.sin(theta_x), np.cos(theta_x), z],
        [0, 0, 0, 1]
    ])

    return rotate_x_translate_xyz.dot(rotate_y).dot(rotate_z)

class Backprojector(nn.Module):
    def __init__(self, interp="nearest", input_hw=128, output_hw=128):
        super().__init__()
        pixel = 1
        self.backproj = DRRProjector(
            mode="backward", volume_shape=(output_hw,output_hw,output_hw), detector_shape=(input_hw,input_hw),
            pixel_size=(pixel, pixel), interp=interp, source_to_detector_distance=1200)

    def get_T(self, inp):
        param =np.asarray(inp)
        param = param * np.pi
        T = get_6dofs_transformation_matrix(param[3:], param[:3])
        T = torch.FloatTensor(T[np.newaxis, ...])
        return torch.cat([T,T,T,T])
    
    def forward(self, xray_1, xray_2):
        self.backproj.to(xray_1.get_device())
        T_ap = self.get_T([1.5, 0, 0, 0, 0, 0])

        vol_in_1 = self.backproj(xray_1, T_ap.to(xray_1.get_device()))

        T_lat = self.get_T([-2, 0, 0, 0, 0, 0])
        vol_in_2 = self.backproj(xray_2, T_lat.to(xray_2.get_device()))

        bp_vol = (vol_in_1 + vol_in_2) / 2

        return bp_vol

class Backprojector_single(nn.Module):
    def __init__(self, interp="nearest", input_hw=128, output_hw=128):
        super().__init__()
        pixel = 1
        self.backproj = DRRProjector(
            mode="backward", volume_shape=(output_hw,output_hw,output_hw), detector_shape=(input_hw,input_hw),
            pixel_size=(pixel, pixel), interp=interp, source_to_detector_distance=1200).cuda()

    def get_T(self, inp):
        param =np.asarray(inp)
        param = param * np.pi
        T = get_6dofs_transformation_matrix(param[3:], param[:3])
        T = torch.FloatTensor(T[np.newaxis, ...])
        return torch.cat([T,T,T,T])
    
    def forward(self, feat, view):
        if view == 1:
            T_ap = self.get_T([1.5, 0, 0, 0, 0, 0])
            bp_vol = self.backproj(feat, T_ap.to(feat.get_device()))
        elif view == 2:
            T_lat = self.get_T([-2, 0, 0, 0, 0, 0])
            bp_vol = self.backproj(feat, T_lat.to(feat.get_device()))

        return bp_vol

import SimpleITK as sitk

def save(volume, spacing, origin, path):
    itkimage = sitk.GetImageFromArray(volume, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    sitk.WriteImage(itkimage, path, True)

if __name__ == '__main__':
    import torchvision
    import cv2
    # input1 = torch.randn(4, 1, 128, 128).cuda()  # View 1
    # input2 = torch.randn(4, 128, 64, 64).cuda()  # View 2

    input1 = cv2.imread('/home/work/Skull/Models/X2CT_TRCT/demo/demo_ap.png') # H, W
    input1 = cv2.resize(input1, (128, 128)) # 128, 128
    input1 = torch.from_numpy(input1).unsqueeze(0).unsqueeze(0).float().cuda()

    input2 = cv2.imread('/home/work/Skull/Models/X2CT_TRCT/demo/demo_la.png')
    input2 = cv2.resize(input2, (128, 128)) # 128, 128
    input2 = torch.from_numpy(input2).unsqueeze(0).unsqueeze(0).float().cuda()

    projector = Backprojector_single().cuda()
    output = projector(input2, view=1)

    # projector = Backprojector().cuda()
    # output = projector(input1, input2)

    volume = output.squeeze(0).cpu().numpy() # (1, 128, 128, 128) ll_feat.permute(0, 1, 3, 2)
    volume = volume[:, ::-1, :, :]
    # print(output.shape) 
    save(volume.squeeze(), spacing=(1.0, 1.0, 1.0), origin=(0,0,0), path="demo1.mha")

