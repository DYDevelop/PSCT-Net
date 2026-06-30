# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import lpips
from torchvision.models.video import r3d_18, s3d
from lib.model.loss.pytorch_i3d import InceptionI3d


class GANLoss(nn.Module):
  def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
    super(GANLoss, self).__init__()
    self.real_label = target_real_label
    self.fake_label = target_fake_label
    self.real_label_tensor = None
    self.fake_label_tensor = None
    if use_lsgan:
      self.loss = nn.MSELoss()
      print('GAN loss: {}'.format('LSGAN'))
    else:
      self.loss = nn.BCELoss()
      print('GAN loss: {}'.format('Normal'))

  def get_target_tensor(self, input, target_is_real):
    target_tensor = None
    if target_is_real:
      create_label = ((self.real_label_tensor is None) or
                      (self.real_label_tensor.numel() != input.numel()))
      if create_label:
        real_tensor = torch.ones(input.size(), dtype=torch.float).fill_(self.real_label)
        self.real_label_tensor = real_tensor.to(input)
      target_tensor = self.real_label_tensor
    else:
      create_label = ((self.fake_label_tensor is None) or
                      (self.fake_label_tensor.numel() != input.numel()))
      if create_label:
        fake_tensor = torch.ones(input.size(), dtype=torch.float).fill_(self.fake_label)
        self.fake_label_tensor = fake_tensor.to(input)
      target_tensor = self.fake_label_tensor
    return target_tensor

  def forward(self, input, target_is_real):
    # for multi_scale_discriminator
    if isinstance(input[0], list):
      loss = 0
      for input_i in input:
        pred = input_i[-1]
        target_tensor = self.get_target_tensor(pred, target_is_real)
        loss += self.loss(pred, target_tensor)
      return loss
    # for patch_discriminator
    else:
      target_tensor = self.get_target_tensor(input[-1], target_is_real)
      return self.loss(input[-1], target_tensor)


class WGANLoss(nn.Module):
  def __init__(self, grad_penalty=False):
    super(WGANLoss, self).__init__()
    self.grad_penalty = grad_penalty
    if grad_penalty:
      print('GAN loss: {}'.format('WGAN-GP'))
    else:
      print('GAN loss: {}'.format('WGAN'))

  def get_mean(self, input):
    input_mean = torch.mean(input)
    return input_mean

  def forward(self, input_fake, input_real=None, is_G=True):
    if is_G:
      assert input_real is None
    cost = 0.
    # for multi_scale_discriminator
    if isinstance(input_fake[0], list):
      for i in range(len(input_fake)):
        if is_G:
          disc_fake = input_fake[i][-1]
          cost += (-self.get_mean(disc_fake))
        else:
          disc_fake = input_fake[i][-1]
          disc_real = input_real[i][-1]
          cost += (self.get_mean(disc_fake) - self.get_mean(disc_real))
      return cost
    # for patch_discriminator
    else:
      if is_G:
        disc_fake = input_fake[-1]
        cost = (-self.get_mean(disc_fake))
      else:
        disc_fake = input_fake[-1]
        disc_real = input_real[-1]
        cost = (self.get_mean(disc_fake) - self.get_mean(disc_real))
      return cost

# Restruction Loss
class RestructionLoss(nn.Module):
  '''
  reduction: 'elementwise_mean' or 'none'
  '''
  def __init__(self, distance='l1', reduction='elementwise_mean'):
    super(RestructionLoss, self).__init__()
    if distance == 'l1':
      self.loss = nn.L1Loss(reduction=reduction)
    elif distance == 'mse':
      self.loss = nn.MSELoss(reduction=reduction)
    else:
      raise NotImplementedError()

  def forward(self, gt, pred):
    return self.loss(gt, pred)
  
class LpipLoss(nn.Module):
  def __init__(self):
    super(LpipLoss, self).__init__()
    self.lpips = lpips.LPIPS(net='vgg')  # 'alex' 

  def forward(self, gt, pred):  # NCHW
    return self.lpips(
        gt.repeat(1, 3, 1, 1),
        pred.repeat(1, 3, 1, 1),
        retPerLayer=False,
        normalize=True
    ).mean()

class LpipLoss_3D(nn.Module):
  def __init__(self):
    super(LpipLoss_3D, self).__init__()
    self.lpips = lpips.LPIPS(net='vgg')  # 'alex' 

  def forward(self, ten1, ten2, to_rgb=True): # BCDHW [B 1 128 128 128]
    vol_n, total = ten1.shape[2], 0.0
    for i in range(vol_n):

      if to_rgb:
        ten1_sub, ten2_sub = ten1[:, :, i, ...].repeat(1, 3, 1, 1), ten2[:, :, i, ...].repeat(1, 3, 1, 1)
      else:
        ten1_sub, ten2_sub = ten1[:, :, i, ...], ten2[:, :, i, ...] # ERROR

      total += self.lpips(ten1_sub, ten2_sub, retPerLayer=False, normalize=True).mean()

    return torch.squeeze(total/vol_n)

# class LpipLoss_3DNet(nn.Module):
#   def __init__(self, layers=None, resize_input=True):
#     super().__init__()
#     # Pretrained S3D model
#     self.s3d = s3d(weights='DEFAULT').eval()
#     for param in self.s3d.parameters():
#         param.requires_grad = False

#     # 선택적으로 사용할 중간 layer 이름 목록
#     self.selected_layers = layers if layers else ['features.2', 'features.5', 'features.8']
#     self.resize_input = resize_input

#     # Hook 저장용
#     self.outputs = {}

#     # Register forward hooks
#     for name, module in self.s3d.named_modules():
#         if name in self.selected_layers:
#             module.register_forward_hook(self._get_hook(name))

#   def _get_hook(self, name):
#     def hook(module, input, output):
#         self.outputs[name] = output
#     return hook

#   def forward(self, x, y):
#     if self.resize_input:
#       x = F.interpolate(x, size=(32, 224, 224), mode='trilinear', align_corners=False)
#       y = F.interpolate(y, size=(32, 224, 224), mode='trilinear', align_corners=False)

#     x, y = x.repeat(1, 3, 1, 1, 1), y.repeat(1, 3, 1, 1, 1)

#     self.outputs = {}
#     _ = self.s3d(x)
#     x_features = self.outputs.copy()

#     self.outputs = {}
#     _ = self.s3d(y)
#     y_features = self.outputs.copy()

#     loss = 0
#     for layer in self.selected_layers:
#       loss += F.l1_loss(x_features[layer], y_features[layer])
#     return loss

class LpipLoss_3DNet(nn.Module):
  def __init__(self, layers=None, input_size=(3, 64, 224, 224)):
    super().__init__()
    self.i3d = InceptionI3d(num_classes=400, in_channels=3)
    self.i3d.load_state_dict(torch.load('model-weights/rgb_imagenet.pt', weights_only=True))  # 사전학습된 모델 필요
    self.i3d.eval()
    for param in self.i3d.parameters():
        param.requires_grad = False

    # 사용할 레이어들 지정
    self.layers = layers if layers else ['Mixed_3c', 'Mixed_4f', 'Mixed_5c']
    self.output_hooks = {}
    self._register_hooks()

  def _get_hook(self, name):
    def hook(module, input, output):
      self.output_hooks[name] = output
    return hook

  def _register_hooks(self):
    for name, module in self.i3d.named_modules():
      if name in self.layers:
        module.register_forward_hook(self._get_hook(name))

  def forward(self, input, target):
    """
    input, target: [B, 3, T, H, W]
    """
    input = F.interpolate(input, size=(64, 224, 224), mode='trilinear', align_corners=False)
    target = F.interpolate(target, size=(64, 224, 224), mode='trilinear', align_corners=False)
    input, target = input.repeat(1, 3, 1, 1, 1), target.repeat(1, 3, 1, 1, 1)
    self.output_hooks = {}
    _ = self.i3d(input)
    input_feats = self.output_hooks.copy()

    self.output_hooks = {}
    _ = self.i3d(target)
    target_feats = self.output_hooks.copy()

    loss = 0
    for layer in self.layers:
      loss += F.l1_loss(input_feats[layer], target_feats[layer])
    return loss
    
from lib.model.nets.generator.drr_projector_new import DRRProjector

class Demo_loss(nn.Module):
    def __init__(self):
        super(Demo_loss, self).__init__()
        self.projector = DRRProjector()
        self.l1_loss = nn.L1Loss()

        # 6개 방향 회전각도 (radian 단위), (theta_x, theta_y, theta_z)
        # 회전 순서는 ZYX (코드 기준)
        self.directions = {
            'AP': (0.0, 0.0, 0.0),
            # 'PA': (0.0, 3.1416, 0.0),  # 180도 Y축 회전
            # 'RL': (0.0, -1.5708, 0.0),  # -90도 Y축 회전
            # 'LL': (0.0, 1.5708, 0.0),   # 90도 Y축 회전
            'SI': (-1.5708, 0.0, 0.0),  # -90도 X축 회전
            # 'IS': (1.5708, 0.0, 0.0),   # 90도 X축 회전
        }

    def forward(self, gt, pred):  # gt, pred: NCHW (N x 1 x D x H x W)
        device = gt.device
        dtype = gt.dtype
        N = gt.shape[0]
        total_loss = 0.0

        for name, angles in self.directions.items():
            # 각 배치에 대해 동일한 회전 파라미터
            theta = torch.tensor([angles], device=device, dtype=dtype).repeat(N, 1)
            transform_param = torch.cat([theta, torch.zeros_like(theta)], dim=1)  # rotation + translation

            # DRR projection
            gt_proj = self.projector(gt, transform_param=transform_param)
            pred_proj = self.projector(pred, transform_param=transform_param)

            # L1 loss
            total_loss += self.l1_loss(gt_proj, pred_proj)

        return total_loss / len(self.directions)
    
import torchvision.models as models
import pywt
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

class WaveletLoss2D(nn.Module):
    def __init__(self, wavelet='haar', levels=1, low_weight=0.1, high_weight=1.0):
        super(WaveletLoss2D, self).__init__()
        self.wavelet = wavelet
        self.levels = levels
        self.low_weight = low_weight
        self.high_weight = high_weight

    def wavelet_transform(self, img):
        """ 2D Wavelet Transform """
        coeffs = pywt.wavedec2(img.detach().cpu().numpy(), self.wavelet, level=self.levels)
        LL, (LH, HL, HH) = coeffs[0], coeffs[1]
        # Tensor로 변환 후 GPU로 이동
        LL = torch.tensor(LL).to(img.device)
        LH = torch.tensor(LH).to(img.device)
        HL = torch.tensor(HL).to(img.device)
        HH = torch.tensor(HH).to(img.device)
        return LL, LH, HL, HH
    
    def forward(self, input, target):
        # 동일한 디바이스로 텐서 이동
        device = input.device

        inp_LL, inp_LH, inp_HL, inp_HH = self.wavelet_transform(input)
        tgt_LL, tgt_LH, tgt_HL, tgt_HH = self.wavelet_transform(target)

        loss_low = torch.nn.functional.mse_loss(inp_LL, tgt_LL)
        loss_high = (
            torch.nn.functional.l1_loss(inp_LH, tgt_LH) +
            torch.nn.functional.l1_loss(inp_HL, tgt_HL) +
            torch.nn.functional.l1_loss(inp_HH, tgt_HH)
        )

        return self.low_weight * loss_low + self.high_weight * loss_high
    
class WaveletLoss3D(nn.Module):
    def __init__(self, wavelet='haar', levels=1, low_weight=1.0, high_weight=1.0):
        super(WaveletLoss3D, self).__init__()
        self.wavelet = wavelet
        self.levels = levels
        self.low_weight = low_weight
        self.high_weight = high_weight

    def wavelet_transform_3d(self, img):
        """ 3D Wavelet Transform을 위해 3D로 나누어 각각 2D Wavelet을 적용 """
        coeffs = []
        # 각 slice마다 2D Wavelet Transform을 적용
        for i in range(img.shape[2]):
            slice_img = img[:, :, i].detach().cpu().numpy()  # CPU로 이동하여 numpy 배열로 변환
            coeffs.append(pywt.dwt2(slice_img, self.wavelet))
        return coeffs

    def forward(self, input, target):
        # input과 target을 동일한 디바이스로 이동
        device = input.device
        
        input_coeffs = self.wavelet_transform_3d(input)
        target_coeffs = self.wavelet_transform_3d(target)

        # 각 주파수 대역에서 L1 Loss를 계산 (각각 LL, LH, HL, HH)
        loss_low = 0
        loss_high = 0
        for i in range(len(input_coeffs)):
            inp_LL, (inp_LH, inp_HL, inp_HH) = input_coeffs[i]
            tgt_LL, (tgt_LH, tgt_HL, tgt_HH) = target_coeffs[i]
            
            # Tensor를 GPU로 이동
            inp_LL = torch.tensor(inp_LL, device=device)
            tgt_LL = torch.tensor(tgt_LL, device=device)
            
            inp_LH = torch.tensor(inp_LH, device=device)
            tgt_LH = torch.tensor(tgt_LH, device=device)
            
            inp_HL = torch.tensor(inp_HL, device=device)
            tgt_HL = torch.tensor(tgt_HL, device=device)
            
            inp_HH = torch.tensor(inp_HH, device=device)
            tgt_HH = torch.tensor(tgt_HH, device=device)
            
            loss_low += torch.nn.functional.mse_loss(inp_LL, tgt_LL)
            loss_high += (
                torch.nn.functional.l1_loss(inp_LH, tgt_LH) +
                torch.nn.functional.l1_loss(inp_HL, tgt_HL) +
                torch.nn.functional.l1_loss(inp_HH, tgt_HH)
            )

        return self.low_weight * loss_low + self.high_weight * loss_high

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def _ssim_3D(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)
    
    
class SSIM3D(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)
    
def SSIM3D_loss(ten1, ten2):
    ssim3d = SSIM3D()
    return 1 - ssim3d(ten1, ten2)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def ssim3D(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim_3D(img1, img2, window, window_size, channel, size_average)

if __name__ == '__main__':
  input = torch.randn(1, 1, 128, 128, 128).cuda()
  target = torch.randn(1, 1, 128, 128, 128).cuda()
  loss = Demo_loss().cuda()
  print(loss(input, target))
