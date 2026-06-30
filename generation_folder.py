# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

import argparse
from lib.config.config import cfg_from_yaml, cfg, merge_dict_and_yaml, print_easy_dict
from lib.dataset.factory import get_dataset
from lib.model.factory import get_model
from lib.utils import html
from lib.utils.visualizer import tensor_back_to_unnormalization, save_images, tensor_back_to_unMinMax
#from lib.utils.metrics_np import MAE, MSE, Peak_Signal_to_Noise_Rate, Structural_Similarity, Cosine_Similarity
from lib.utils import ct as CT
import copy
import tqdm
import torch
import numpy as np
import os
import time
import cv2, shutil
from tqdm import tqdm
from glob import glob
import pandas as pd

def parse_args():
  parse = argparse.ArgumentParser(description='CTGAN')
  parse.add_argument('--data', type=str, default='', dest='data',
                     help='input data ')
  parse.add_argument('--tag', type=str, default='', dest='tag',
                     help='distinct from other try')
  parse.add_argument('--dataroot', type=str, default='', dest='dataroot',
                     help='input data root')
  parse.add_argument('--dataset', type=str, default='', dest='dataset',
                     help='Train or test or valid')
  parse.add_argument('--datasetfile', type=str, default='', dest='datasetfile',
                     help='Train or test or valid file path')
  parse.add_argument('--ymlpath', type=str, default=None, dest='ymlpath',
                     help='config have been modified')
  parse.add_argument('--gpu', type=str, default='0,1', dest='gpuid',
                     help='gpu is split by ,')
  parse.add_argument('--dataset_class', type=str, default='unalign', dest='dataset_class',
                     help='Dataset class should select from unalign /')
  parse.add_argument('--model_class', type=str, default='cyclegan', dest='model_class',
                     help='Model class should select from cyclegan / ')
  parse.add_argument('--check_point', type=str, default=None, dest='check_point',
                     help='which epoch to load? ')
  parse.add_argument('--latest', action='store_true', dest='latest',
                     help='set to latest to use latest cached model')
  parse.add_argument('--verbose', action='store_true', dest='verbose',
                     help='if specified, print more debugging information')
  parse.add_argument('--load_path', type=str, default=None, dest='load_path',
                     help='if load_path is not None, model will load from load_path')
  parse.add_argument('--how_many', type=int, dest='how_many', default=50,
                     help='if specified, print more debugging information')
  parse.add_argument('--resultdir', type=str, default='', dest='resultdir',
                     help='dir to save result')
  args = parse.parse_args()
  return args

def crop_to_upper(im, ratio=0.7):
    h, _ = im.shape
    im = im[:int(h*ratio), :]
    return im

def single_gen(base_dir, idx, pa_path, la_path, addon='_rec_A'):
  # pa_xray, lat_xray = np.expand_dims(crop_to_upper(cv2.imread(f'demo/PA_ST/{id}{addon}.png', 0)), 0), \
  # np.expand_dims(crop_to_upper(cv2.imread(f'demo/LL_ST/{id}{addon}.png', 0)), 0)
  pa_xray, lat_xray = np.expand_dims(cv2.imread(f'{pa_path}', 0), 0), \
  np.expand_dims(cv2.imread(f'{la_path}', 0), 0)

  # print(lat_xray.shape)
  data_augmentation = opt.data_augmentation(opt)

  ct, pa_xray, lat_xray = data_augmentation([torch.rand(128, 128, 128), pa_xray, lat_xray])

  data = (torch.unsqueeze(ct.cuda(), dim=0)), (torch.unsqueeze(pa_xray.cuda(), dim=0), torch.unsqueeze(lat_xray.cuda(), dim=0)), ['testing']

  start_time = time.time()
  gan_model.set_input(data)
  gan_model.test()

  visuals = gan_model.get_current_visuals()
  img_path = gan_model.get_image_paths()
  end_time = time.time()

  elapsed = end_time - start_time
  print('Time taken for one image: {:.2f} seconds'.format(elapsed))

  # CT Source
  generate_CT = visuals['G_fake'].data.clone().cpu().numpy()
  real_CT = visuals['G_real'].data.clone().cpu().numpy()
  # To NDHW
  if 'std' in opt.dataset_class or 'baseline' in opt.dataset_class:
    generate_CT_transpose = generate_CT
    real_CT_transpose = real_CT
  else:
    generate_CT_transpose = np.transpose(generate_CT, (0, 2, 1, 3))
    real_CT_transpose = np.transpose(real_CT, (0, 2, 1, 3))
  # Inveser Deepth
  generate_CT_transpose = generate_CT_transpose[:, ::-1, :, :]
  real_CT_transpose = real_CT_transpose[:, ::-1, :, :]
  # To [0, 1]
  generate_CT_transpose = tensor_back_to_unnormalization(generate_CT_transpose, opt.CT_MEAN_STD[0],
                                                          opt.CT_MEAN_STD[1])
  real_CT_transpose = tensor_back_to_unnormalization(real_CT_transpose, opt.CT_MEAN_STD[0], opt.CT_MEAN_STD[1])
  # Clip generate_CT
  generate_CT_transpose = np.clip(generate_CT_transpose, 0, 1)

  # To HU coordinate
  generate_CT_transpose = tensor_back_to_unMinMax(generate_CT_transpose, opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]).astype(np.int32) - 1024
  real_CT_transpose = tensor_back_to_unMinMax(real_CT_transpose, opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]).astype(np.int32) - 1024
  # Save
  image_root = os.path.join(f'multiview/gen/{idx}', 'CT')
  if not os.path.exists(image_root):
    os.makedirs(image_root)
  save_path = os.path.join(image_root, 'fake_ct.mha')
  ctVisual.save(generate_CT_transpose.squeeze(0), spacing=(1.0, 1.0, 1.0), origin=(0,0,0), path=save_path)
  save_path = os.path.join(image_root, 'real_ct.mha')
  ctVisual.save(real_CT_transpose.squeeze(0), spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0), path=save_path)

  shutil.copyfile(pa_path, os.path.join(image_root, f"{idx}_xray1.png"))
  shutil.copyfile(la_path, os.path.join(image_root, f"{idx}_xray2.png"))

  del visuals, img_path

if __name__ == '__main__':
  args = parse_args()

  # check gpu
  if args.gpuid == '':
    args.gpu_ids = []
  else:
    if torch.cuda.is_available():
      split_gpu = str(args.gpuid).split(',')
      args.gpu_ids = [int(i) for i in split_gpu]
    else:
      print('There is no gpu!')
      exit(0)

  # check point
  if args.check_point is None:
    args.epoch_count = 1
  else:
    args.epoch_count = int(args.check_point)

  # merge config with yaml
  if args.ymlpath is not None:
    cfg_from_yaml(args.ymlpath)
  # merge config with argparse
  opt = copy.deepcopy(cfg)
  opt = merge_dict_and_yaml(args.__dict__, opt)
  print_easy_dict(opt)

  opt.serial_batches = True

  # add data_augmentation
  datasetClass, _, dataTestClass, collateClass = get_dataset(opt.dataset_class)
  opt.data_augmentation = dataTestClass

  # get model
  gan_model = get_model(opt.model_class)()
  print('Model --{}-- will be Used'.format(gan_model.name))

  # set to test
  gan_model.eval()

  gan_model.init_process(opt)
  total_steps, epoch_count = gan_model.setup(opt)

  # must set to test Mode again, due to  omission of assigning mode to network layers
  # model.training is test, but BN.training is training
  if opt.verbose:
    print('## Model Mode: {}'.format('Training' if gan_model.training else 'Testing'))
    for i, v in gan_model.named_modules():
      print(i, v.training)

  if 'batch' in opt.norm_G:
    gan_model.eval()
  elif 'instance' in opt.norm_G:
    gan_model.eval()
    # instance norm in training mode is better
    for name, m in gan_model.named_modules():
      if m.__class__.__name__.startswith('InstanceNorm'):
        m.train()
  else:
    raise NotImplementedError()

  if opt.verbose:
    print('## Change to Model Mode: {}'.format('Training' if gan_model.training else 'Testing'))
    for i, v in gan_model.named_modules():
      print(i, v.training)

  ctVisual = CT.CTVisual()
  df = pd.read_csv('demo/Real_ST/dataset_txt_medinfo.csv')
  mappings = {}
  for _, row in tqdm(df.iterrows()):
    idx, pa_path, la_path = row['patient_id'], row['AP_path'].split('/')[-1], row['Left_path'].split('/')[-1]
    mappings[pa_path] = [idx, la_path]

  # for _, row in tqdm(df.iterrows()):
  #   idx, pa_path, la_path = row['patient_id'], row['AP_path'].split('/')[-1].replace('.png', ''), \
  #   row['Left_path'].split('/')[-1].replace('.png', '')
  #   single_gen('demo/Real_ST', idx, pa_path, la_path, addon='')

  for pa_path in tqdm(glob("demo/PA_STB_UVC_READERS/*.png")):
    idx, la_imagename = mappings[pa_path.split('/')[-1]]
    la_path = pa_path.replace(f"{pa_path.split('/')[-1]}", la_imagename).replace("PA_STB_UVC_READERS", "LL_STB_UVC_READERS")
    single_gen('.', idx, pa_path, la_path, addon='')