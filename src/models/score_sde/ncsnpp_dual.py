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
from . import CrossAttnBlockpp
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

class NCSNpp_dual(nn.Module):
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

    CrossAttnBlock = functools.partial(CrossAttnBlockpp.CrossAttnBlockpp,
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

    channels = config.data.num_channels # also a hack
    if progressive_input != 'none':
      input_pyramid_ch = channels

    # ============================================== first layer for input
    self.input_conv = conv3x3(4, nf) # expanded input for conditioning
    hs_c = [nf]

    # ============================================== lookup table for conditioning dimensions
    conditioning_dim_table = [128,128,256,256]

    encoder_modules = torch.nn.ModuleList()

    in_ch = nf
    for i_level in range(num_resolutions):
      level_modules = torch.nn.ModuleDict({
        'resblocks': torch.nn.ModuleList(),
        'attn': None,
        'cross_attn': None,
        'downres': None,
        'combiner': None
      })
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        out_ch = nf * ch_mult[i_level]
        level_modules['resblocks'].append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
        in_ch = out_ch

        if all_resolutions[i_level] in attn_resolutions:
          cond_dim = conditioning_dim_table[i_level]
          level_modules['attn'] = AttnBlock(channels=in_ch)
          level_modules['cross_attn'] = CrossAttnBlock(channels=in_ch,cond_chans=cond_dim)
        hs_c.append(in_ch)

      # if not last level of downsampling
      if i_level != num_resolutions - 1:
        level_modules['downres'] = ResnetBlock(down=True, in_ch=in_ch)

        level_modules['combiner'] = combiner(dim1=input_pyramid_ch, dim2=in_ch)
        if combine_method == 'cat':
          in_ch *= 2
        hs_c.append(in_ch)

      encoder_modules.append(level_modules)
    self.encoder_modules = encoder_modules

    # CENTRAL BLOCK
    in_ch = hs_c[-1]
    central_block = torch.nn.ModuleList()
    central_block.append(ResnetBlock(in_ch=in_ch))
    central_block.append(AttnBlock(channels=in_ch))
    central_block.append(ResnetBlock(in_ch=in_ch))
    self.central_block = central_block

    pyramid_ch = 0
    # Upsampling block
    decoder_modules = torch.nn.ModuleList([None]*num_resolutions)
    for i_level in reversed(range(num_resolutions)):
      level_modules = torch.nn.ModuleDict({
        'resblocks': torch.nn.ModuleList(),
        'attn': None,
        'cross_attn': None,
        'group_norm': None,
        'conv3x3': None,
        'final_res': None,
      })
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        level_modules['resblocks'].append(ResnetBlock(in_ch=in_ch + hs_c.pop(),out_ch=out_ch))
        in_ch = out_ch

      if all_resolutions[i_level] in attn_resolutions:
        cond_dim = conditioning_dim_table[i_level]
        level_modules['attn'] = AttnBlock(channels=in_ch)
        level_modules['cross_attn'] = CrossAttnBlock(channels=in_ch,cond_chans=cond_dim)

      level_modules['group_norm'] = nn.GroupNorm(num_groups=min(in_ch // 4, 32),num_channels=in_ch, eps=1e-6)
      if i_level == num_resolutions - 1:
        level_modules['conv3x3'] = conv3x3(in_ch, channels, init_scale=init_scale)
      else:
        level_modules['conv3x3'] = conv3x3(in_ch, channels, bias=True, init_scale=init_scale)
      # HACK
      pyramid_ch = channels

      if i_level != 0:
        level_modules['final_res'] = ResnetBlock(in_ch=in_ch, up=True)

      decoder_modules[i_level] = level_modules
    self.decoder_modules = decoder_modules

    assert not hs_c

    # ================== my modules
    self.conditioning_conv_0 = nn.Conv2d(128,128, stride=1, bias=True, padding=1, kernel_size=3)
    self.conditioning_conv_1 = nn.Conv2d(128,128, stride=2, bias=True, padding=1, kernel_size=3)
    self.conditioning_conv_2 = nn.Conv2d(128, 256, stride=2, bias=True, padding=1, kernel_size=3)
    self.conditioning_conv_3 = nn.Conv2d(256, 256, stride=2, bias=True, padding=1, kernel_size=3)

  def forward(self, x_a, x_b, time_cond_a, time_cond_b, rays_ref, rays_a, rays_b):
    # both frames need to be processed symmetrically, so we stack them and treat them as a superbatch
    assert x_a.shape == x_b.shape
    b,c,h,w = x_a.shape
    x = torch.cat([x_a[:,None,:,:,:],x_b[:,None,:,:,:]],1).reshape(b*2,c,h,w)
    time_cond = torch.cat([time_cond_a[:,None],time_cond_b[:,None]],1).reshape(b*2)

    # in this model, time cond is just the std of noise

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

    # ========== create conditioning pyramid ==========
    cond_0 = self.conditioning_conv_0(self.act(torch.cat([rays_ref,rays_a,rays_b],0)))
    cond_1 = self.conditioning_conv_1(self.act(cond_0))
    cond_2 = self.conditioning_conv_2(self.act(cond_1))
    cond_3 = self.conditioning_conv_3(self.act(cond_2))
    conditioning_stack = {
      32: [cond_0[0:b,...],cond_0[b:b*2,...],cond_0[b*2:b*3,...]],
      16: [cond_1[0:b,...],cond_1[b:b*2,...],cond_1[b*2:b*3,...]],
      8: [cond_2[0:b,...],cond_2[b:b*2,...],cond_2[b*2:b*3,...]],
      4: [cond_3[0:b,...],cond_3[b:b*2,...],cond_3[b*2:b*3,...]],
    }
    # =============================================

    # Downsampling block
    input_pyramid = None
    if self.progressive_input != 'none':
      input_pyramid = x

    # ============================================
    # x = torch.cat([x,conditioning],1)
    # ============================================

    hs = [self.input_conv(x)]
    for i_level in range(self.num_resolutions):
      level_modules = self.encoder_modules[i_level]
      # Residual blocks for this resolution
      for resblock in level_modules['resblocks']:
        h = resblock(hs[-1], temb)
        if level_modules['attn'] is not None:
          layer_cond = conditioning_stack[h.shape[2]]
          h = level_modules['attn'](h)
          h = level_modules['cross_attn'](h,q_cond=layer_cond[0],k_a_cond=layer_cond[1],k_b_cond=layer_cond[2])
        hs.append(h)

      if level_modules['downres'] is not None:
        h = level_modules['downres'](hs[-1], temb)

        input_pyramid = self.pyramid_downsample(input_pyramid)
        h = level_modules['combiner'](input_pyramid, h)

        hs.append(h)

    h = hs[-1]
    h = self.central_block[0](h, temb)
    h = self.central_block[1](h)
    h = self.central_block[2](h, temb)

    pyramid = None

    # Upsampling block
    for i_level in reversed(range(self.num_resolutions)):
      level_modules = self.decoder_modules[i_level]
      for resblock in level_modules['resblocks']:
        h = resblock(torch.cat([h, hs.pop()], dim=1), temb)

      if level_modules['attn'] is not None:
        layer_cond = conditioning_stack[h.shape[2]]
        h = level_modules['attn'](h)
        h = level_modules['cross_attn'](h,q_cond=layer_cond[0],k_a_cond=layer_cond[1],k_b_cond=layer_cond[2])

      pyramid_h = self.act(level_modules['group_norm'](h))
      pyramid_h = level_modules['conv3x3'](pyramid_h)

      if i_level == self.num_resolutions - 1:
        pyramid = pyramid_h
      else:
        pyramid = self.pyramid_upsample(pyramid)
        pyramid = pyramid + pyramid_h

      if level_modules['final_res'] is not None:
        h = level_modules['final_res'](h, temb)

    assert not hs # ensure we used all the skip connections

    h = pyramid

    if self.config.model.scale_by_sigma:
      used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
      h = h / used_sigmas

    # unstack super batch
    b,c,hi,wi = x_a.shape
    h = h.reshape(b,2,c,hi,wi)
    h_a = h[:,0,:,:,:]
    h_b = h[:,1,:,:,:]

    return h_a,h_b
