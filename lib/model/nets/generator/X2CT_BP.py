# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import functools
from lib.model.nets.generator.encoder_decoder_utils import *
from lib.model.nets.generator.back_pro import Backprojector
from lib.model.nets.generator.feature_fuse import Resnet, SirenLinear1, creategrid, Linear
from lib.model.nets.generator.blocks import *

def UNetLike_DownStep5(input_shape, encoder_input_channels, decoder_output_channels, decoder_out_activation, encoder_norm_layer, decoder_norm_layer, upsample_mode, decoder_feature_out=False):
  # 64, 32, 16, 8, 4
  encoder_block_list = [6, 12, 24, 16, 6]
  decoder_block_list = [1, 2, 2, 2, 2, 0]
  growth_rate = 32
  encoder_channel_list = [64]
  decoder_channel_list = [16, 16, 32, 64, 128, 256]
  decoder_begin_size = input_shape // pow(2, len(encoder_block_list))
  return UNetLike_DenseDimensionNet(encoder_input_channels, decoder_output_channels, decoder_begin_size, encoder_block_list, decoder_block_list, growth_rate, encoder_channel_list, decoder_channel_list, decoder_out_activation, encoder_norm_layer, decoder_norm_layer, upsample_mode, decoder_feature_out)

def UNetLike_DownStep5_3(input_shape, encoder_input_channels, decoder_output_channels, decoder_out_activation, encoder_norm_layer, decoder_norm_layer, upsample_mode, decoder_feature_out=False):
  # 64, 32, 16, 8, 4
  encoder_block_list = [6, 12, 32, 32, 12]
  decoder_block_list = [3, 3, 3, 3, 3, 1]
  growth_rate = 32
  encoder_channel_list = [64]
  decoder_channel_list = [16, 32, 64, 64, 128, 256]
  decoder_begin_size = input_shape // pow(2, len(encoder_block_list))
  return UNetLike_DenseDimensionNet(encoder_input_channels, decoder_output_channels, decoder_begin_size, encoder_block_list, decoder_block_list, growth_rate, encoder_channel_list, decoder_channel_list, decoder_out_activation, encoder_norm_layer, decoder_norm_layer, upsample_mode, decoder_feature_out)

class UNetLike_DenseDimensionNet(nn.Module):
  def __init__(self, encoder_input_channels, decoder_output_channels, decoder_begin_size, encoder_block_list, decoder_block_list, growth_rate, encoder_channel_list, decoder_channel_list, decoder_out_activation, encoder_norm_layer=nn.BatchNorm2d, decoder_norm_layer=nn.BatchNorm3d, upsample_mode='nearest', decoder_feature_out=False):
    super(UNetLike_DenseDimensionNet, self).__init__()

    self.decoder_channel_list = decoder_channel_list
    self.decoder_block_list = decoder_block_list
    self.encoder_block_list = encoder_block_list
    self.n_downsampling = len(encoder_block_list)
    self.decoder_begin_size = decoder_begin_size
    self.decoder_feature_out = decoder_feature_out
    activation = nn.ReLU(True)
    bn_size = 4

    ##############
    # Encoder
    ##############
    if type(encoder_norm_layer) == functools.partial:
      use_bias = encoder_norm_layer.func == nn.InstanceNorm2d or decoder_norm_layer.func == nn.BatchNorm2d
    else:
      use_bias = encoder_norm_layer == nn.InstanceNorm2d or decoder_norm_layer == nn.BatchNorm2d

    encoder_layers0 = [
      nn.ReflectionPad2d(3),
      nn.Conv2d(encoder_input_channels, encoder_channel_list[0], kernel_size=7, padding=0, bias=use_bias),
      encoder_norm_layer(encoder_channel_list[0]),
      activation
    ]
    self.encoder_layer = nn.Sequential(*encoder_layers0)

    num_input_channels = encoder_channel_list[0]
    for index, channel in enumerate(encoder_block_list):
      # pooling
      down_layers = [
        encoder_norm_layer(num_input_channels),
        activation,
        nn.Conv2d(num_input_channels, num_input_channels, kernel_size=3, stride=2, padding=1, bias=use_bias),

      ]
      down_layers += [
        Dense_2DBlock(encoder_block_list[index], num_input_channels, bn_size, growth_rate, encoder_norm_layer, activation, use_bias),
      ]
      num_input_channels = num_input_channels + encoder_block_list[index] * growth_rate

      # feature maps are compressed into 1 after the lastest downsample layers
      # if index == (self.n_downsampling-1):
      #   down_layers += [
      #     nn.AdaptiveAvgPool2d(1)
      #   ]
      # else:
      if index != (self.n_downsampling-1):
        num_out_channels = num_input_channels // 2
        down_layers += [
          encoder_norm_layer(num_input_channels),
          activation,
          nn.Conv2d(num_input_channels, num_out_channels, kernel_size=1, stride=1, padding=0, bias=use_bias),
        ]
        num_input_channels = num_out_channels
      encoder_channel_list.append(num_input_channels)
      setattr(self, 'encoder_layer' + str(index), nn.Sequential(*down_layers))

    ##############
    # Linker
    ##############
    if type(decoder_norm_layer) == functools.partial:
      use_bias = decoder_norm_layer.func == nn.InstanceNorm3d or decoder_norm_layer.func == nn.BatchNorm3d
    else:
      use_bias = decoder_norm_layer == nn.InstanceNorm3d or decoder_norm_layer == nn.BatchNorm3d

    self.projection = Project2Dto3D()

    for index, channel in enumerate(encoder_channel_list[:-1]):
      in_channels = channel
      out_channels = decoder_channel_list[index]
      link_layers = [
        Dimension_UpsampleCutBlock(in_channels, out_channels, encoder_norm_layer, decoder_norm_layer, activation, use_bias)
      ]
      setattr(self, 'linker_layer' + str(index), nn.Sequential(*link_layers))

    ##############
    # Decoder
    ##############
    for index, channel in enumerate(decoder_channel_list[:-1]):
      out_channels = channel
      in_channels = decoder_channel_list[index+1]
      decoder_layers = []
      decoder_compress_layers = []
      if index != (len(decoder_channel_list) - 2):
        decoder_compress_layers += [
          nn.Conv3d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=use_bias),
          decoder_norm_layer(in_channels),
          activation
        ]
        for _ in range(decoder_block_list[index+1]):
          decoder_layers += [
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=use_bias),
            decoder_norm_layer(in_channels),
            activation
          ]
      decoder_layers += [
        Upsample_3DUnit(3, in_channels, out_channels, decoder_norm_layer, scale_factor=2, upsample_mode=upsample_mode, activation=activation, use_bias=use_bias)
      ]
      # If decoder_feature_out is True, compressed feature after upsampling and concatenation
      # can be obtained.
      if decoder_feature_out:
        setattr(self, 'decoder_compress_layer' + str(index), nn.Sequential(*decoder_compress_layers))
        setattr(self, 'decoder_layer' + str(index), nn.Sequential(*decoder_layers))
      else:
        setattr(self, 'decoder_layer' + str(index), nn.Sequential(*(decoder_compress_layers+decoder_layers)))
    # last decode
    decoder_layers = []
    decoder_compress_layers = [
      nn.Conv3d(decoder_channel_list[0] * 2, decoder_channel_list[0], kernel_size=3, padding=1, bias=use_bias),
      decoder_norm_layer(decoder_channel_list[0]),
      activation
    ]
    for _ in range(decoder_block_list[0]):
      decoder_layers += [
        nn.Conv3d(decoder_channel_list[0], decoder_channel_list[0], kernel_size=3, padding=1, bias=use_bias),
        decoder_norm_layer(decoder_channel_list[0]),
        activation
      ]
    if decoder_feature_out:
      setattr(self, 'decoder_compress_layer' + str(-1), nn.Sequential(*decoder_compress_layers))
      setattr(self, 'decoder_layer' + str(-1), nn.Sequential(*decoder_layers))
    else:
      setattr(self, 'decoder_layer' + str(-1), nn.Sequential(*(decoder_compress_layers + decoder_layers)))

    self.decoder_layer = nn.Sequential(*[
      nn.Conv3d(decoder_channel_list[0], decoder_output_channels, kernel_size=7, padding=3, bias=use_bias),
      decoder_out_activation()
    ])

  def forward(self, input):
    encoder_feature = self.encoder_layer(input)
    next_input = encoder_feature
    for i in range(self.n_downsampling):
      setattr(self, 'feature_linker' + str(i), getattr(self, 'linker_layer' + str(i))(next_input))
      next_input = getattr(self, 'encoder_layer'+str(i))(next_input)

    next_input = self.base_link(next_input.view(next_input.size(0), -1))
    next_input = next_input.view(next_input.size(0), self.decoder_channel_list[-1], self.decoder_begin_size, self.decoder_begin_size, self.decoder_begin_size)

    for i in range(self.n_downsampling - 1, -2, -1):
      if i == (self.n_downsampling - 1):
        if self.decoder_feature_out:
          next_input = getattr(self, 'decoder_layer' + str(i))(getattr(self, 'decoder_compress_layer' + str(i))(next_input))
        else:
          next_input = getattr(self, 'decoder_layer' + str(i))(next_input)

      else:
        if self.decoder_feature_out:
          next_input = getattr(self, 'decoder_layer' + str(i))(getattr(self, 'decoder_compress_layer' + str(i))(torch.cat((next_input, getattr(self, 'feature_linker'+str(i+1))), dim=1)))
        else:
          next_input = getattr(self, 'decoder_layer' + str(i))(torch.cat((next_input, getattr(self, 'feature_linker'+str(i+1))), dim=1))

    return self.decoder_layer(next_input)


class X2CT_BP(nn.Module):
  def __init__(self, view1Model, view2Model, view1Order, view2Order, backToSub, decoder_output_channels, decoder_out_activation, decoder_block_list=None, decoder_norm_layer=nn.BatchNorm3d, upsample_mode='nearest'):
    super(X2CT_BP, self).__init__()
    self.view1Model = view1Model
    self.view2Model = view2Model
    self.view1Order = view1Order
    self.view2Order = view2Order
    self.backToSub = backToSub
    self.n_downsampling = view2Model.n_downsampling
    self.decoder_channel_list = view2Model.decoder_channel_list
    if decoder_block_list is None:
      self.decoder_block_list = view2Model.decoder_block_list
    else:
      self.decoder_block_list = decoder_block_list

    activation = nn.ReLU(True)
    if type(decoder_norm_layer) == functools.partial:
      use_bias = decoder_norm_layer.func == nn.InstanceNorm3d
    else:
      use_bias = decoder_norm_layer == nn.InstanceNorm3d
    
    self.encoder_block_list = view1Model.encoder_block_list
    self.Backprojector = Backprojector(interp="nearest")

    ##############
    # Encoder
    ##############
    encoder_layers0 = [
      nn.ReflectionPad3d(3),
      nn.Conv3d(1, 16, kernel_size=7, padding=0, bias=use_bias),
      decoder_norm_layer(16),
      activation
    ]

    self.encoder_layer = nn.Sequential(*encoder_layers0)
    
    for index, channel, channel_2d in zip(range(5), self.decoder_channel_list[1:], [128, 128, 256, 512, 512]): # 0 ~ 5, 16 32 64 128 256
      out_channels = channel
      in_channels = self.decoder_channel_list[index + 1]
      down_layers = [
        decoder_norm_layer(in_channels),
        activation,
        nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=use_bias),
      ]
      down_layers += [ # [6, 12, 24, 16, 6]
        Dense_3DBlock(self.encoder_block_list[index], in_channels, False, 4, 32, decoder_norm_layer, activation, use_bias),
      ]
      in_channels = in_channels + self.encoder_block_list[index] * 32
      down_layers += [
        decoder_norm_layer(in_channels),
        activation,
        nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=use_bias),
      ]
      
      setattr(self, 'encoder_layer' + str(index), nn.Sequential(*down_layers))
      setattr(self, 'back_projection' + str(index), Backprojector(interp="nearest", input_hw=128//(2**index), output_hw=128//(2**index)))
      setattr(self, 'bp_avg_linker' + str(index), nn.Sequential(nn.Conv3d(channel_2d, out_channels//2, kernel_size=3, padding=1),
                                                                decoder_norm_layer(out_channels//2),
                                                                activation))
    # self.projector = Projector()
    ##############
    # Decoder
    ##############
    for index, channel in enumerate(self.decoder_channel_list[:-1]):
      out_channels = channel
      in_channels = self.decoder_channel_list[index + 1]
      decoder_layers = []
      decoder_compress_layers = []
      if index != (len(self.decoder_channel_list) - 2):
        decoder_compress_layers += [
          nn.Conv3d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=use_bias),
          decoder_norm_layer(in_channels),
          activation
        ]
        for _ in range(self.decoder_block_list[index+1]):
          decoder_layers += [
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=use_bias),
            decoder_norm_layer(in_channels),
            activation
          ]
      decoder_layers += [
        Upsample_3DUnit(3, in_channels, out_channels, decoder_norm_layer, scale_factor=2, upsample_mode=upsample_mode,
                        activation=activation, use_bias=use_bias)
      ]

      setattr(self, 'decoder_layer' + str(index), nn.Sequential(*(decoder_compress_layers + decoder_layers)))
    # last decode
    decoder_layers = []
    decoder_compress_layers = [
      nn.Conv3d(self.decoder_channel_list[0] * 2, self.decoder_channel_list[0], kernel_size=3, padding=1, bias=use_bias),
      decoder_norm_layer(self.decoder_channel_list[0]),
      activation
    ]
    for _ in range(self.decoder_block_list[0]):
      decoder_layers += [
        nn.Conv3d(self.decoder_channel_list[0], self.decoder_channel_list[0], kernel_size=3, padding=1, bias=use_bias),
        decoder_norm_layer(self.decoder_channel_list[0]),
        activation
      ]
    setattr(self, 'decoder_layer' + str(-1), nn.Sequential(*(decoder_compress_layers + decoder_layers)))
    self.decoder_layer = nn.Sequential(*[
      nn.Conv3d(self.decoder_channel_list[0], decoder_output_channels, kernel_size=7, padding=3, bias=use_bias),
      decoder_out_activation()
    ])

    self.transposed_layer = Transposed_And_Add(view1Order, view2Order)

    self.latent_block = MambaBlock3D(in_channels=256, drop_p=0.5)

  def forward(self, input):
    # only support two views
    assert len(input) == 2
    # View 1 and 2 encoding process
    view1_next_input = self.view1Model.encoder_layer(input[0])
    view2_next_input = self.view2Model.encoder_layer(input[1])

    # Back-Projected encoding process
    bp_volume = self.Backprojector(input[0], input[1]) # torch.Size([B, 1, 128, 128, 128])
    view_next_input = self.encoder_layer(bp_volume) # torch.Size([B, 16, 128, 128, 128])
    # skip_connections = [view_next_input]
    for i in range(self.view1Model.n_downsampling):
      
      if i >= 1 : # 1, 2, 3, 4
        view_2d_bp = getattr(self, 'back_projection' + str(i))(enc1_feature, enc2_feature)
        view_2d_bp = getattr(self, 'bp_avg_linker' + str(i))(view_2d_bp)
        view_next_input = getattr(self, 'encoder_layer' + str(i))(torch.cat([view_2d_bp, view_next_input], 1)) # torch.Size([4, 32, 64, 64, 64]) Condition injection
        # view_next_input = getattr(self, 'encoder_layer' + str(i))(torch.cat([view_next_input, view_next_input], 1)) # Without X-Ray BP Conditions
      else: # 0
        view_next_input = getattr(self, 'encoder_layer' + str(i))(view_next_input) # torch.Size([B, 128, 128, 128, 128])
      # skip_connections.append(view_next_input)
      setattr(self.view1Model, 'feature_linker' + str(i), getattr(self.view1Model, 'linker_layer' + str(i))(view1_next_input))
      view1_next_input = getattr(self.view1Model, 'encoder_layer'+str(i))(view1_next_input)
      setattr(self.view2Model, 'feature_linker' + str(i), getattr(self.view2Model, 'linker_layer' + str(i))(view2_next_input))
      view2_next_input = getattr(self.view2Model, 'encoder_layer' + str(i))(view2_next_input)
      enc1_feature, enc2_feature = view1_next_input, view2_next_input

    # print(view_next_input.shape) # torch.Size([4, 128, 8, 8, 8])
    # print(view1_next_input.shape) # torch.Size([4, 704, 4, 4])

    # View 1 decoding process Part1
    # view1_next_input = self.view1Model.base_link(view1_next_input.view(view1_next_input.size(0), -1))
    view1_next_input = self.view1Model.projection(view1_next_input) # apply attention
    # view1_next_input = view1_next_input.view(view1_next_input.size(0), self.view1Model.decoder_channel_list[-1], self.view1Model.decoder_begin_size,
    #                                          self.view1Model.decoder_begin_size, self.view1Model.decoder_begin_size)

    # print(view1_next_input.shape) # torch.Size([4, 256, 4, 4, 4])

    # View 2 decoding process Part1
    # view2_next_input = self.view2Model.base_link(view2_next_input.view(view2_next_input.size(0), -1))
    view2_next_input = self.view2Model.projection(view2_next_input)
    # view2_next_input = view2_next_input.view(view2_next_input.size(0), self.view2Model.decoder_channel_list[-1], self.view2Model.decoder_begin_size,
    #                                          self.view2Model.decoder_begin_size, self.view2Model.decoder_begin_size)

    # view1_next_input = view2_next_input = self.projector(view1_next_input, view2_next_input) # torch.Size([4, 704, 4, 4, 4])
    # print(view_next_input.shape) # torch.Size([4, 256, 4, 4, 4])
    view_next_input = self.latent_block(view_next_input)

    # View 1 and 2 decoding process Part2
    for skip, i in enumerate(range(self.n_downsampling - 1, -2, -1)):
      if i == (self.n_downsampling - 1):
        view1_next_input = getattr(self.view1Model, 'decoder_compress_layer' + str(i))(view1_next_input)
        view2_next_input = getattr(self.view2Model, 'decoder_compress_layer' + str(i))(view2_next_input)
        ########### MultiView Fusion
        # Method One: Fused feature back to sub-branch
        if self.backToSub:
          view_avg = self.transposed_layer(view1_next_input, view2_next_input) / 2
          view1_next_input = view_avg.permute(*self.view1Order)
          view2_next_input = view_avg.permute(*self.view2Order)
          # view_next_input = view_next_input + skip_connections.pop()
          # skip_connection = getattr(self, 'attention_linker' + str(skip))(g=view_next_input, x=skip_connections.pop())
          # view_next_input = getattr(self, 'skip_linker' + str(skip))(torch.cat((skip_connection, view_next_input), dim=1))
          view_next_input = getattr(self, 'decoder_layer' + str(i))(view_next_input)
        # Method Two: Fused feature only used in main-branch
        else:
          # view_avg = self.transposed_layer(view1_next_input, view2_next_input) / 2
          # skip_connection = getattr(self, 'attention_linker' + str(skip))(g=view_next_input, x=skip_connections.pop())
          # view_next_input = getattr(self, 'skip_linker' + str(skip))(torch.cat((skip_connection, view_next_input), dim=1))
          # view_next_input = view_next_input + skip_connections.pop()
          view_next_input = getattr(self, 'decoder_layer' + str(i))(view_next_input)
        ###########
        view1_next_input = getattr(self.view1Model, 'decoder_layer' + str(i))(view1_next_input)
        view2_next_input = getattr(self.view2Model, 'decoder_layer' + str(i))(view2_next_input)
      else:
        view1_next_input = getattr(self.view1Model, 'decoder_compress_layer' + str(i))(torch.cat((view1_next_input, getattr(self.view1Model, 'feature_linker' + str(i + 1))), dim=1))
        view2_next_input = getattr(self.view2Model, 'decoder_compress_layer' + str(i))(torch.cat((view2_next_input, getattr(self.view2Model, 'feature_linker' + str(i + 1))), dim=1))
        ########### MultiView Fusion
        # Method One: Fused feature back to sub-branch
        if self.backToSub:
          view_avg = self.transposed_layer(view1_next_input, view2_next_input) / 2
          view1_next_input = view_avg.permute(*self.view1Order)
          view2_next_input = view_avg.permute(*self.view2Order)
          # view_next_input = view_next_input + skip_connections.pop()
          # skip_connection = getattr(self, 'attention_linker' + str(skip))(g=view_next_input, x=skip_connections.pop())
          # view_next_input = getattr(self, 'skip_linker' + str(skip))(torch.cat((skip_connection, view_next_input), dim=1))
          view_next_input = getattr(self, 'decoder_layer' + str(i))(torch.cat((view_avg, view_next_input), dim=1))
        # Method Two: Fused feature only used in main-branch
        else:
          view_avg = self.transposed_layer(view1_next_input, view2_next_input) / 2
          # view_next_input = view_next_input + skip_connections.pop()
          # skip_connection = getattr(self, 'attention_linker' + str(skip))(g=view_next_input, x=skip_connections.pop())
          # view_next_input = getattr(self, 'skip_linker' + str(skip))(torch.cat((skip_connection, view_next_input), dim=1))
          view_next_input = getattr(self, 'decoder_layer' + str(i))(torch.cat((view_avg, view_next_input), dim=1))
        ###########
        view1_next_input = getattr(self.view1Model, 'decoder_layer' + str(i))(view1_next_input)
        view2_next_input = getattr(self.view2Model, 'decoder_layer' + str(i))(view2_next_input)


    return self.view1Model.decoder_layer(view1_next_input), self.view2Model.decoder_layer(view2_next_input), self.decoder_layer(view_next_input)

if __name__ == "__main__":
  
  input1 = torch.randn(4, 1, 128, 128).cuda()  # View 1
  input2 = torch.randn(4, 1, 128, 128).cuda()  # View 2

  encoder_input_shape = [128, 128]
  encoder_input_nc = 1
  activation_layer = nn.ReLU
  encoder_norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
  decoder_norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
  output_nc = 1
  CTOrder_Xray1 = [0, 1, 3, 2, 4] # B C D H W -> B C H D W
  CTOrder_Xray2 = [0, 1, 4, 2, 3] # B C D H W -> B C W D H -> B C H W D

  netG = X2CT_BP(view1Model=UNetLike_DownStep5(input_shape=encoder_input_shape[0], encoder_input_channels=encoder_input_nc, decoder_output_channels=output_nc, decoder_out_activation=activation_layer, encoder_norm_layer=encoder_norm_layer, decoder_norm_layer=decoder_norm_layer, upsample_mode='transposed', decoder_feature_out=True), view2Model=UNetLike_DownStep5(input_shape=encoder_input_shape[0], encoder_input_channels=encoder_input_nc, decoder_output_channels=output_nc, decoder_out_activation=activation_layer, encoder_norm_layer=encoder_norm_layer, decoder_norm_layer=decoder_norm_layer, upsample_mode='transposed', decoder_feature_out=True), view1Order=CTOrder_Xray1, view2Order=CTOrder_Xray2, backToSub=True, decoder_output_channels=output_nc, decoder_out_activation=activation_layer, decoder_block_list=[1, 1, 1, 1, 1, 0], decoder_norm_layer=decoder_norm_layer, upsample_mode='transposed').cuda()

  output = netG([input1, input2])
  print(output[0].shape, output[1].shape, output[2].shape) # torch.Size([4, 1, 128, 128, 128])

  # Projector = Projector().cuda()
  # input1 = torch.randn(4, 704, 4, 4).cuda()  # View 1
  # input2 = torch.randn(4, 704, 4, 4).cuda()  # View 2
  # output = Projector(input1, input2)
  # print(output.shape) # torch.Size([4, 8, 4, 4, 4])
  
  """ 
  BP input added --> [26.03 -> 26.40]

  Connection-C 이상함 -> 순서 변경 필요 : PA 는 그대로 두고 LL에서만 LL.permute(0, 1, 3, 2) 수행해야함. --> Not good

  Connection-A base linears to cross-attention layer --> [26.40 -> 26.91]

  Connection-A Nerf 기반 or BP 기반으로 변경 --> Not good

  U-Net Encoder -> Decoder skip connection (Attention) 추가 --> Not good

  2D 이미지 넣을때 WT 적용한 3 channel 넣어보기 --> Not good

  U-Net Encoder -> Decoder skip connection (Addition) 추가 --> Not good

  Using WT --> Not good
  - 입력/출력 모두 WT만 사용 후 IWT 복원
  - 입력만 [image + WT], 출력은 image
  - 출력 분리 (image head + WT head) Multi decoder

  InstanceNorm -> BatchNorm 변경 --> [26.91 -> 26.96]

  Latent Block 추가 --> TinyTransformerBlock3D [26.85 -> 26.95]

  Demo loss (Front Projection) 추가 --> Running

  loss 방법론 추가구성 (LPIPS_3D, LPIPS_Projection, Wavelet_2D, Wavelet_3D, SSIM_3D)
  """