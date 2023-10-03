from utils import *
from os.path import join as pjoin
import time
import datetime
import torch
from torchvision import datasets, transforms
from datasets import *
from torch.utils.data import DataLoader
import torch.optim as optim
from models.score_sde.ncsnpp_dual import NCSNpp_dual
import models.score_sde.sde_lib as sde_lib
from .Base_trainer import Base_trainer
from models.score_sde.configs.LDM import LDM_config
import numpy as np
from models.vqgan.vqgan import VQModel
from models.vqgan.configs.vqgan_32_4 import vqgan_32_4_config
from models.score_sde.ema import ExponentialMovingAverage
import functools
from models.score_sde.layerspp import ResnetBlockBigGANpp
import torch.nn.functional as F

def loss_fn(score_sde, batch, cond_im, ff_ref, ff_a, ff_b):
	score_sde.train()
	eps=1e-5

	t = score_sde.t_uniform(batch.shape[0],batch.device)
	perturbed_data, z, std = score_sde.forward_diffusion(batch,t)
	score = score_sde.score(perturbed_data, cond_im, t, ff_ref, ff_a, ff_b)

	losses = torch.square(score * std[:, None, None, None] + z)
	losses = 0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1)

	loss = torch.mean(losses)
	return loss

class Score_sde_trainer(Base_trainer):
	def __init__(self,local_rank,node_rank,n_gpus_per_node,n_nodes):
		# distributed helpers
		self.rank = node_rank*n_gpus_per_node + local_rank
		self.world_size = n_nodes*n_gpus_per_node
		self.gpu = local_rank
		compute_only = self.rank != 0
		super().__init__(compute_only)

		# configure checkpoint behaviour
		self.n_kept_checkpoints = 5
		self.checkpoint_interval = 33000 # 55 minutes
		self.max_epoch = 2000
		self.checkpoint_retention_interval = 20

		# vqgan
		self.vqgan = VQModel(**vqgan_32_4_config).to(f'cuda:{self.gpu}')
		self.vqgan.eval()

		# score sde configs
		self.config = LDM_config()
		config = self.config

		# main components, dataloaders, model, optimizer
		batch_size = 64//self.world_size
		self.dataset = Realestate_dataset_video('train')
		self.sampler = DistributedSaveableSampler(self.dataset,num_replicas=self.world_size,rank=self.rank,shuffle=True)
		self.train_loader = torch.utils.data.DataLoader(self.dataset,batch_size=batch_size,sampler=self.sampler,drop_last=True,num_workers=8,persistent_workers=True)

		score_model = NCSNpp_dual(config)
		score_model = score_model.to(f'cuda:{self.gpu}')
		score_model = torch.nn.parallel.DistributedDataParallel(score_model,device_ids=[self.gpu])
		sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)

		ResnetBlock = functools.partial(ResnetBlockBigGANpp,
										act=torch.nn.SiLU(),
										dropout=False,
										fir=True,
										fir_kernel=[1,3,3,1],
										init_scale=0,
										skip_rescale=True,
										temb_dim=None)
		ray_downsampler = torch.nn.Sequential(
			ResnetBlock(in_ch=56,out_ch=128,down=True).to(f'cuda:{self.gpu}'),
			ResnetBlock(in_ch=128,out_ch=128,down=True).to(f'cuda:{self.gpu}'))
		ray_downsampler = torch.nn.parallel.DistributedDataParallel(ray_downsampler,device_ids=[self.gpu])

		self.score_sde = Score_sde_model(score_model,sde,ray_downsampler)

		# only 1 worker needs to keep ema
		if self.rank == 0:
			self.ema = ExponentialMovingAverage(self.score_sde.parameters(),decay=0.999)

		self.lr = 2e-4
		self.optimizer = optim.Adam(self.score_sde.parameters(), lr=self.lr)
		self.warmup = 5000
		self.grad_clip = 1.

		self.tlog('Setup complete','note')

	def train_epoch(self):
		iteration_start = time.time()
		self.sampler.set_epoch(self.epoch) # important! or else split will be the same every epoch
		dataloader_iter = iter(self.train_loader) # need this to save state
		for epoch_it, batch_data in enumerate(dataloader_iter,start=1):
			self.total_iterations += 1

			# unpack data
			im_a = batch_data['im_a']
			im_b = batch_data['im_b']
			camera_enc_ref = batch_data['camera_enc_ref']
			camera_enc_a = batch_data['camera_enc_a']
			camera_enc_b = batch_data['camera_enc_b']

			# move data to gpu
			im_a = im_a.to(f'cuda:{self.gpu}')
			im_b = im_b.to(f'cuda:{self.gpu}')
			camera_enc_ref = camera_enc_ref.to(f'cuda:{self.gpu}')
			camera_enc_a = camera_enc_a.to(f'cuda:{self.gpu}')
			camera_enc_b = camera_enc_b.to(f'cuda:{self.gpu}')

			# encode with vqgan
			with torch.no_grad():
				encoded_a = self.vqgan.encode(im_a)
				encoded_b = self.vqgan.encode(im_b)

			# train
			self.optimizer.zero_grad()
			ff_ref = F.pad(freq_enc(camera_enc_ref),[0,0,0,0,1,1,0,0]) # pad, must be %4==0 for group norm
			ff_a = F.pad(freq_enc(camera_enc_a),[0,0,0,0,1,1,0,0])
			ff_b = F.pad(freq_enc(camera_enc_b),[0,0,0,0,1,1,0,0])
			loss = loss_fn(self.score_sde, encoded_a, encoded_b, ff_ref, ff_a, ff_b)
			loss.backward()

			# warmup and gradient clip
			if self.warmup > 0:
				for g in self.optimizer.param_groups:
					g['lr'] = self.lr * np.minimum(self.total_iterations / self.warmup, 1.0)
			if self.grad_clip >= 0:
				torch.nn.utils.clip_grad_norm_(self.score_sde.parameters(), max_norm=self.grad_clip)
			self.optimizer.step()

			# update ema
			if self.rank == 0:
				self.ema.update(self.score_sde.parameters())

			# print iteration details
			if not self.compute_only:
				# calc eta
				iteration_duration = time.time() - iteration_start
				its_per_sec = 1/iteration_duration
				remaining_its = self.max_epoch*len(self.train_loader) - self.total_iterations
				eta_sec = remaining_its * iteration_duration
				eta_min = eta_sec//60
				eta = str(datetime.timedelta(minutes=eta_min))

				self.tlog(f'{self.total_iterations} | loss: {loss.item()} | it/s: {its_per_sec} | ETA: {eta}','iter')
				self.tb_writer.add_scalar('training/loss', loss.item(), self.total_iterations)

			# checkpoint if haven't checkpointed in a while
			self.maybe_save_checkpoint(epoch_it,dataloader_iter)

			# check if termination requested
			self.check_termination_request(epoch_it,dataloader_iter)

			# start time here to include data fetching
			iteration_start = time.time()

	def validate(self):
		pass

	def state_dict(self,dataloader_iter=None):
		state_dict = {
			'epoch': self.epoch,
			'total_iterations': self.total_iterations,
			'optimizer': self.optimizer.state_dict(),
			'score_sde_model':self.score_sde.state_dict(),
			'training_data_sampler':self.sampler.state_dict(dataloader_iter),
			'ema': self.ema.state_dict()
		}
		return state_dict

	def load_state_dict(self,state_dict):
		self.epoch = state_dict['epoch']
		self.total_iterations = state_dict['total_iterations']
		self.score_sde.load_state_dict(state_dict['score_sde_model'])
		self.optimizer.load_state_dict(state_dict['optimizer'])
		self.sampler.load_state_dict(state_dict['training_data_sampler'])
		if self.rank == 0:
			self.ema.load_state_dict(state_dict['ema'])

	def load_checkpoint(self,checkpoint_path):
		super().load_checkpoint(checkpoint_path,map_location=f'cuda:{self.gpu}')
