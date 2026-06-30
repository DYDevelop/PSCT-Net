# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

import numpy as np
import numpy.linalg as linalg
# from skimage.measure import compare_ssim as SSIM
from skimage.metrics import structural_similarity as SSIM

import lpips
loss_fn_alex = lpips.LPIPS(net='alex').cuda()
import torch

##############################################
'''
3D Metrics
'''
##############################################

# def lpips_3d(arr1, arr2, to_rgb=True):
#     # 1. Tensor 변환 및 GPU 이동 [B, D, H, W]
#     ten1 = torch.Tensor(arr1.copy()).cuda()
#     ten2 = torch.Tensor(arr2.copy()).cuda()
    
#     # 각 축의 인덱스 설정 (1: Depth, 2: Height, 3: Width)
#     axes = [1, 2, 3]
#     axis_totals = []

#     for axis in axes:
#         # 해당 축을 슬라이싱하기 좋게 차원 변경 (B, Slices, Dim1, Dim2)
#         dims = [0, 1, 2, 3]
#         dims[1], dims[axis] = dims[axis], dims[1]
#         t1 = ten1.permute(*dims)
#         t2 = ten2.permute(*dims)
        
#         vol_n = t1.shape[1]
#         total = 0
        
#         for i in range(vol_n):
#             # 슬라이스 추출 및 채널 차원 확보 [B, 1, Dim1, Dim2]
#             s1, s2 = t1[:, i:i+1, ...], t2[:, i:i+1, ...]
            
#             if to_rgb:
#                 # LPIPS는 기본적으로 3채널을 기대하므로 1채널을 3개로 복제
#                 s1, s2 = s1.repeat(1, 3, 1, 1), s2.repeat(1, 3, 1, 1)
#             else:
#                 # to_rgb가 False라도 최소 4차원 [B, C, H, W] 형태는 유지해야 함
#                 # (단, LPIPS 모델에 따라 1채널 입력 시 에러가 날 수 있으므로 주의)
#                 pass 

#             total += loss_fn_alex(s1, s2, normalize=True)
        
#         # 해당 축의 평균 LPIPS 저장
#         axis_totals.append(total / vol_n)

#     # 2. 3개 축의 평균 계산
#     final_mean = (axis_totals[0] + axis_totals[1] + axis_totals[2]) / 3
    
#     # 3. 기존 코드처럼 Squeeze 후 Numpy로 반환
#     return torch.squeeze(final_mean).cpu().detach().numpy()

def lpips_3d(arr1, arr2, to_rgb=True):
  ten1, ten2 = torch.Tensor(arr1.copy()).cuda(), torch.Tensor(arr2.copy()).cuda()
  vol_n, total = ten1.shape[1], 0

  for i in range(vol_n):

    if to_rgb:
      ten1_sub, ten2_sub = torch.unsqueeze(ten1[:, i, ...], 1).repeat(1, 3, 1, 1), torch.unsqueeze(ten2[:, i, ...], 1).repeat(1, 3, 1, 1)
    else:
      ten1_sub, ten2_sub = ten1[:, i, ...], ten2[:, i, ...] # ERROR

    total += loss_fn_alex(ten1_sub, ten2_sub, normalize=True)

  return torch.squeeze(total/vol_n).cpu().detach().numpy()

def MAE(arr1, arr2, size_average=True):
  '''
  :param arr1:
    Format-[NDHW], OriImage
  :param arr2:
    Format-[NDHW], ComparedImage
  :return:
    Format-None if size_average else [N]
  '''
  assert (isinstance(arr1, np.ndarray)) and (isinstance(arr2, np.ndarray))
  assert (arr1.ndim == 4) and (arr2.ndim == 4)
  arr1 = arr1.astype(np.float64)
  arr2 = arr2.astype(np.float64)
  if size_average:
    return np.abs(arr1 - arr2).mean()
  else:
    return np.abs(arr1 - arr2).mean(1).mean(1).mean(1)

def MSE(arr1, arr2, size_average=True):
  '''
  :param arr1:
    Format-[NDHW], OriImage
  :param arr2:
    Format-[NDHW], ComparedImage
  :return:
    Format-None if size_average else [N]
  '''
  assert (isinstance(arr1, np.ndarray)) and (isinstance(arr2, np.ndarray))
  assert (arr1.ndim == 4) and (arr2.ndim == 4)
  arr1 = arr1.astype(np.float64)
  arr2 = arr2.astype(np.float64)
  if size_average:
    return np.power(arr1 - arr2, 2).mean()
  else:
    return np.power(arr1 - arr2, 2).mean(1).mean(1).mean(1)

def Cosine_Similarity(arr1, arr2, size_average=True):
  '''
  :param arr1:
    Format-[NDHW], OriImage
  :param arr2:
    Format-[NDHW], ComparedImage
  :return:
    Format-None if size_average else [N]
  '''
  assert (isinstance(arr1, np.ndarray)) and (isinstance(arr2, np.ndarray))
  assert (arr1.ndim == 4) and (arr2.ndim == 4)
  arr1 = arr1.astype(np.float64)
  arr2 = arr2.astype(np.float64)
  arr1_squeeze = arr1.reshape((arr1.shape[0], -1))
  arr2_squeeze = arr2.reshape((arr2.shape[0], -1))
  cosineSimilarity = np.sum(arr1_squeeze*arr2_squeeze, 1) / (linalg.norm(arr2_squeeze, axis=1)*linalg.norm(arr1_squeeze, axis=1))
  if size_average:
    return cosineSimilarity.mean()
  else:
    return cosineSimilarity

def Peak_Signal_to_Noise_Rate_3D(arr1, arr2, size_average=True, PIXEL_MAX=1.0):
  '''
  :param arr1:
    Format-[NDHW], OriImage [0,1]
  :param arr2:
    Format-[NDHW], ComparedImage [0,1]
  :return:
    Format-None if size_average else [N]
  '''
  assert (isinstance(arr1, np.ndarray)) and (isinstance(arr2, np.ndarray))
  assert (arr1.ndim == 4) and (arr2.ndim == 4)
  arr1 = arr1.astype(np.float64)
  arr2 = arr2.astype(np.float64)
  eps = 1e-10
  se = np.power(arr1 - arr2, 2)
  mse = se.mean(axis=1).mean(axis=1).mean(axis=1)
  zero_mse = np.where(mse == 0)
  mse[zero_mse] = eps
  psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
  # #zero mse, return 100
  psnr[zero_mse] = 100

  if size_average:
    return psnr.mean()
  else:
    return psnr

##############################################
'''
2D Metrics
'''
##############################################
def Peak_Signal_to_Noise_Rate(arr1, arr2, size_average=True, PIXEL_MAX=1.0):
  '''
  :param arr1:
    Format-[NDHW], OriImage [0,1]
  :param arr2:
    Format-[NDHW], ComparedImage [0,1]
  :return:
    Format-None if size_average else [N]
  '''
  assert (isinstance(arr1, np.ndarray)) and (isinstance(arr2, np.ndarray))
  assert (arr1.ndim == 4) and (arr2.ndim == 4)
  arr1 = arr1.astype(np.float64)
  arr2 = arr2.astype(np.float64)
  eps = 1e-10
  se = np.power(arr1 - arr2, 2)
  # Depth
  mse_d = se.mean(axis=2, keepdims=True).mean(axis=3, keepdims=True).squeeze(3).squeeze(2)
  zero_mse = np.where(mse_d==0)
  mse_d[zero_mse] = eps
  psnr_d = 20 * np.log10(PIXEL_MAX / np.sqrt(mse_d))
  # #zero mse, return 100
  psnr_d[zero_mse] = 100
  psnr_d = psnr_d.mean(1)

  # Height
  mse_h = se.mean(axis=1, keepdims=True).mean(axis=3, keepdims=True).squeeze(3).squeeze(1)
  zero_mse = np.where(mse_h == 0)
  mse_h[zero_mse] = eps
  psnr_h = 20 * np.log10(PIXEL_MAX / np.sqrt(mse_h))
  # #zero mse, return 100
  psnr_h[zero_mse] = 100
  psnr_h = psnr_h.mean(1)

  # Width
  mse_w = se.mean(axis=1, keepdims=True).mean(axis=2, keepdims=True).squeeze(2).squeeze(1)
  zero_mse = np.where(mse_w == 0)
  mse_w[zero_mse] = eps
  psnr_w = 20 * np.log10(PIXEL_MAX / np.sqrt(mse_w))
  # #zero mse, return 100
  psnr_w[zero_mse] = 100
  psnr_w = psnr_w.mean(1)

  psnr_avg = (psnr_h + psnr_d + psnr_w) / 3
  if size_average:
    return [psnr_d.mean(), psnr_h.mean(), psnr_w.mean(), psnr_avg.mean()]
  else:
    return [psnr_d, psnr_h, psnr_w, psnr_avg]

def Structural_Similarity(arr1, arr2, size_average=True, PIXEL_MAX=1.0):
  '''
  :param arr1:
    Format-[NDHW], OriImage [0,1]
  :param arr2:
    Format-[NDHW], ComparedImage [0,1]
  :return:
    Format-None if size_average else [N]
  '''
  assert (isinstance(arr1, np.ndarray)) and (isinstance(arr2, np.ndarray))
  assert (arr1.ndim == 4) and (arr2.ndim == 4)
  arr1 = arr1.astype(np.float64)
  arr2 = arr2.astype(np.float64)

  N = arr1.shape[0]
  # Depth
  arr1_d = np.transpose(arr1, (0, 2, 3, 1))
  arr2_d = np.transpose(arr2, (0, 2, 3, 1))
  ssim_d = []
  for i in range(N):
    ssim = SSIM(arr1_d[i], arr2_d[i], data_range=PIXEL_MAX, channel_axis=True)
    ssim_d.append(ssim)
  ssim_d = np.asarray(ssim_d, dtype=np.float64)

  # Height
  arr1_h = np.transpose(arr1, (0, 1, 3, 2))
  arr2_h = np.transpose(arr2, (0, 1, 3, 2))
  ssim_h = []
  for i in range(N):
    ssim = SSIM(arr1_h[i], arr2_h[i], data_range=PIXEL_MAX, channel_axis=True)
    ssim_h.append(ssim)
  ssim_h = np.asarray(ssim_h, dtype=np.float64)

  # Width
  # arr1_w = np.transpose(arr1, (0, 1, 2, 3))
  # arr2_w = np.transpose(arr2, (0, 1, 2, 3))
  ssim_w = []
  for i in range(N):
    ssim = SSIM(arr1[i], arr2[i], data_range=PIXEL_MAX, channel_axis=True)
    ssim_w.append(ssim)
  ssim_w = np.asarray(ssim_w, dtype=np.float64)

  ssim_avg = (ssim_d + ssim_h + ssim_w) / 3

  if size_average:
    return [ssim_d.mean(), ssim_h.mean(), ssim_w.mean(), ssim_avg.mean()]
  else:
    return [ssim_d, ssim_h, ssim_w, ssim_avg]


#################
#Test
#################
def psnr(img_1, img_2, PIXEL_MAX = 255.0):
  mse = np.mean((img_1 - img_2) ** 2)
  # print(img_1, img_2)
  # print(img_1 - img_2, 'bb')
  # print(mse)
  if mse == 0:
    return 100

  return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

if __name__ == '__main__':
  img1 = np.random.randint(0, 256, size=(1,20,20,20), dtype=np.int64)
  img2 = np.random.randint(0, 256, size=(1,20,20,20), dtype=np.int64)
  # print(img1, img2, 'aa')
  img11 = img1 / 255.0
  img21 = img2 / 255.0
  print(img11.shape, type(img11))
  print(psnr(img1[0], img2[0], 255), psnr(img11[0], img21[0], 1.0))
  print(Peak_Signal_to_Noise_Rate_3D(img1, img2, PIXEL_MAX=255), Peak_Signal_to_Noise_Rate_3D(img11, img21, PIXEL_MAX=1.0))
  print(Peak_Signal_to_Noise_Rate(img1, img2, PIXEL_MAX=255), Peak_Signal_to_Noise_Rate(img11, img21, PIXEL_MAX=1.0))


# print(ssim(img1, img2, data_range=255, multichannel=True), ssim(img11, img21, data_range=1.0, multichannel=True))

# img1 = np.random.randint(0, 256, size=(20,20,10), dtype=np.int16)
# img2 = np.random.randint(0, 256, size=(20,20,10), dtype=np.int16)
# # print(img1, img2, 'aa')
# img11 = img1 / 255.0
# img21 = img2 / 255.0
# total = 0
# for i in range(10):
#   total += SSIM(img1[:, :, i], img2[:, :, i], data_range=255, multichannel=False)
# print(total / 10)
# print(SSIM(img1, img2, data_range=255, multichannel=True), SSIM(img11, img21, data_range=1.0, multichannel=True))

# img1 = np.random.randint(0, 256, size=(1,20,20,10), dtype=np.int16)
# img2 = np.random.randint(0, 256, size=(1,20,20,10), dtype=np.int16)
# print(img1, img2, 'aa')
# img11 = img1 / 255.0
# img21 = img2 / 255.0

# print(Structural_Similarity(img1, img2, PIXEL_MAX=255, size_average=False), Structural_Similarity(img11, img21, PIXEL_MAX=1.0, size_average=False))
# print(SSIM(img1[0], img2[0], data_range=255, multichannel=True), SSIM(img11[0], img21[0], data_range=1.0, multichannel=True))