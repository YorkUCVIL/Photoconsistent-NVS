# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file

from . import utils, layers, layerspp, normalization
import torch.nn as nn
import functools
import torch
import numpy as np

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


@utils.register_model(name='ncsnpp')
class NCSNpp(nn.Module):
  """NCSN++ model"""

  def __init__(self, config):
    super().__init__()
    self.config = config
    self.act = act = get_act(config)
    self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))

    self.nf = nf = config.model.nf
    ch_mult = config.model.ch_mult
    self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
    self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
    dropout = config.model.dropout
    resamp_with_conv = config.model.resamp_with_conv
    self.num_resolutions = num_resolutions = len(ch_mult)
    self.all_resolutions = all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolutions)]

    self.conditional = conditional = config.model.conditional  # noise-conditional
    fir = config.model.fir
    fir_kernel = config.model.fir_kernel
    self.skip_rescale = skip_rescale = config.model.skip_rescale
    self.resblock_type = resblock_type = config.model.resblock_type.lower()
    self.progressive = progressive = config.model.progressive.lower()
    self.progressive_input = progressive_input = config.model.progressive_input.lower()
    self.embedding_type = embedding_type = config.model.embedding_type.lower()
    init_scale = config.model.init_scale
    assert progressive in ['none', 'output_skip', 'residual']
    assert progressive_input in ['none', 'input_skip', 'residual']
    assert embedding_type in ['fourier', 'positional']
    combine_method = config.model.progressive_combine.lower()
    combiner = functools.partial(Combine, method=combine_method)

    AttnBlock = functools.partial(layerspp.AttnBlockpp,
                                  init_scale=init_scale,
                                  skip_rescale=skip_rescale)

    Upsample = functools.partial(layerspp.Upsample,
                                 with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)
    # Downsampling block
    Downsample = functools.partial(layerspp.Downsample,
                                   with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)


    self.pyramid_upsample = layerspp.Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
    self.pyramid_downsample = layerspp.Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)

    ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                    act=act,
                                    dropout=dropout,
                                    fir=fir,
                                    fir_kernel=fir_kernel,
                                    init_scale=init_scale,
                                    skip_rescale=skip_rescale,
                                    temb_dim=nf * 4)


    # ============================================== input layer for embedding the time/noise

    # timestep/noise_level embedding; only for continuous training
    # Gaussian Fourier features embeddings.
    assert config.training.continuous, "Fourier features are only used for continuous training."
    self.temb_modules = torch.nn.ModuleList()
    self.temb_modules.append(layerspp.GaussianFourierProjection(embedding_size=nf, scale=config.model.fourier_scale))
    embed_dim = 2 * nf

    # ============================================== time embedding pre processing
    if conditional:
      self.temb_modules.append(nn.Linear(embed_dim, nf * 4))
      self.temb_modules[-1].weight.data = default_initializer()(self.temb_modules[-1].weight.shape)
      nn.init.zeros_(self.temb_modules[-1].bias)
      self.temb_modules.append(nn.Linear(nf * 4, nf * 4))
      self.temb_modules[-1].weight.data = default_initializer()(self.temb_modules[-1].weight.shape)
      nn.init.zeros_(self.temb_modules[-1].bias)

    channels = config.data.num_channels + 4 # also a hack
    if progressive_input != 'none':
      input_pyramid_ch = channels

    # ============================================== first layer for input
    self.input_conv = conv3x3(8+128, nf) # expanded input for conditioning
    hs_c = [nf]

    # ============================================== lookup table for conditioning dimensions
    conditioning_dim_table = [128,128,256,256]

    encoder_modules = torch.nn.ModuleList()

    in_ch = nf
    for i_level in range(num_resolutions):
      level_modules = torch.nn.ModuleDict({
        'resblocks': torch.nn.ModuleList(),
        'attn': None,
        'downres': None,
        'combiner': None
      })
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        out_ch = nf * ch_mult[i_level]
        cond_dim = conditioning_dim_table[i_level]
        level_modules['resblocks'].append(ResnetBlock(in_ch=in_ch, out_ch=out_ch, conditioning_dim=cond_dim))
        in_ch = out_ch

        if all_resolutions[i_level] in attn_resolutions:
          level_modules['attn'] = AttnBlock(channels=in_ch)
        hs_c.append(in_ch)

      # if not last level of downsampling
      if i_level != num_resolutions - 1:
        cond_dim = conditioning_dim_table[i_level+1]
        level_modules['downres'] = ResnetBlock(down=True, in_ch=in_ch, conditioning_dim=cond_dim)

        level_modules['combiner'] = combiner(dim1=input_pyramid_ch, dim2=in_ch)
        if combine_method == 'cat':
          in_ch *= 2
        hs_c.append(in_ch)

      encoder_modules.append(level_modules)
    self.encoder_modules = encoder_modules

    # CENTRAL BLOCK
    in_ch = hs_c[-1]
    cond_dim = conditioning_dim_table[-1]
    central_block = torch.nn.ModuleList()
    central_block.append(ResnetBlock(in_ch=in_ch, conditioning_dim=cond_dim))
    central_block.append(AttnBlock(channels=in_ch))
    central_block.append(ResnetBlock(in_ch=in_ch, conditioning_dim=cond_dim))
    self.central_block = central_block

    pyramid_ch = 0
    # Upsampling block
    decoder_modules = torch.nn.ModuleList([None]*num_resolutions)
    for i_level in reversed(range(num_resolutions)):
      layer_modules = torch.nn.ModuleDict({
        'resblocks': torch.nn.ModuleList(),
        'attn': None,
        'group_norm': None,
        'conv3x3': None,
        'final_res': None,
      })
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        layer_modules['resblocks'].append(ResnetBlock(in_ch=in_ch + hs_c.pop(),out_ch=out_ch))
        in_ch = out_ch

      if all_resolutions[i_level] in attn_resolutions:
        layer_modules['attn'] = AttnBlock(channels=in_ch)

      layer_modules['group_norm'] = nn.GroupNorm(num_groups=min(in_ch // 4, 32),num_channels=in_ch, eps=1e-6)
      if i_level == num_resolutions - 1:
        layer_modules['conv3x3'] = conv3x3(in_ch, channels-4, init_scale=init_scale)
      else:
        layer_modules['conv3x3'] = conv3x3(in_ch, channels-4, bias=True, init_scale=init_scale)
      # HACK
      pyramid_ch = channels

      if i_level != 0:
        layer_modules['final_res'] = ResnetBlock(in_ch=in_ch, up=True)

      decoder_modules[i_level] = layer_modules
    self.decoder_modules = decoder_modules

    assert not hs_c

    # ================== my modules
    self.conditioning_conv_0 = nn.Conv2d(128,128, stride=1, bias=True, padding=1, kernel_size=3)
    self.conditioning_conv_1 = nn.Conv2d(128,128, stride=2, bias=True, padding=1, kernel_size=3)
    self.conditioning_conv_2 = nn.Conv2d(128, 256, stride=2, bias=True, padding=1, kernel_size=3)
    self.conditioning_conv_3 = nn.Conv2d(256, 256, stride=2, bias=True, padding=1, kernel_size=3)

  def forward(self, x, cond_im, time_cond, conditioning):
    # in this model, time cond is just the std of noise
    # adjust std of cond_im to match statistics of current x
    cond_im = (cond_im+0.14)*time_cond[:,None,None,None]*8.8

    # timestep/noise_level embedding; only for continuous training
    # Gaussian Fourier features embeddings.
    used_sigmas = time_cond
    temb = self.temb_modules[0](torch.log(used_sigmas))

    # time embedding preprocessing
    temb = self.temb_modules[1](temb)
    temb = self.temb_modules[2](self.act(temb))

    if not self.config.data.centered:
      # If input data is in [0, 1]
      x = 2 * x - 1.
      x = torch.cat([x,cond_im],1)

    # ========== create conditioning pyramid ==========
    cond_0 = self.conditioning_conv_0(self.act(conditioning))
    cond_1 = self.conditioning_conv_1(self.act(cond_0))
    cond_2 = self.conditioning_conv_2(self.act(cond_1))
    cond_3 = self.conditioning_conv_3(self.act(cond_2))
    conditioning_stack = {
      32: cond_0,
      16: cond_1,
      8: cond_2,
      4: cond_3,
    }
    # =============================================

    # Downsampling block
    input_pyramid = None
    if self.progressive_input != 'none':
      input_pyramid = x

    # ============================================
    x = torch.cat([x,conditioning],1)
    # ============================================

    hs = [self.input_conv(x)]; print(f'input layer output: {hs[-1].shape}')
    for i_level in range(self.num_resolutions):
      level_modules = self.encoder_modules[i_level]; print(f'\n======== Encoder level: {i_level}')
      # Residual blocks for this resolution
      for resblock in level_modules['resblocks']:
        layer_cond = conditioning_stack[hs[-1].shape[2]]
        h = resblock(hs[-1], temb, layer_cond); print(f'Resblock output: {h.shape}')
        if level_modules['attn'] is not None:
          h = level_modules['attn'](h); print(f'Attn output: {h.shape}')
        hs.append(h); print('Push Skip')

      if level_modules['downres'] is not None:
        print(f'===== level post processing')
        layer_cond = conditioning_stack[hs[-1].shape[2]//2]
        h = level_modules['downres'](hs[-1], temb, layer_cond); print(f'Resnet output: {h.shape}')

        input_pyramid = self.pyramid_downsample(input_pyramid)
        h = level_modules['combiner'](input_pyramid, h); print(f'combiner output: {h.shape}')

        hs.append(h); print('Push Skip')

    print(f'\n======== central block')
    h = hs[-1]
    layer_cond = conditioning_stack[h.shape[2]]
    h = self.central_block[0](h, temb, layer_cond)
    h = self.central_block[1](h)
    h = self.central_block[2](h, temb, layer_cond)
    print(f'Resnet output: {h.shape}')

    pyramid = None

    # Upsampling block
    for i_level in reversed(range(self.num_resolutions)):
      level_modules = self.decoder_modules[i_level]; print(f'\n======== Decoder level: {i_level}')
      for resblock in level_modules['resblocks']:
        h = resblock(torch.cat([h, hs.pop()], dim=1), temb); print(f'Resblock (cat skip) output: {h.shape}')

      if level_modules['attn'] is not None:
        h = level_modules['attn'](h); print(f'Attn output: {h.shape}')

      print(f'===== level post processing')
      pyramid_h = self.act(level_modules['group_norm'](h))
      print(f'GroupNorm output: {pyramid_h.shape}')
      pyramid_h = level_modules['conv3x3'](pyramid_h)
      print(f'Conv3x3 output: {pyramid_h.shape}')

      if i_level == self.num_resolutions - 1:
        pyramid = pyramid_h; print(f'Init pyramid')
      else:
        pyramid = self.pyramid_upsample(pyramid)
        pyramid = pyramid + pyramid_h; print(f'Sum pyramid')

      if level_modules['final_res'] is not None:
        h = level_modules['final_res'](h, temb); print(f'Final resnet output: {h.shape}')

    assert not hs # ensure we used all the skip connections

    h = pyramid

    if self.config.model.scale_by_sigma:
      print('Scale Sigma')
      used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
      h = h / used_sigmas

    return h
