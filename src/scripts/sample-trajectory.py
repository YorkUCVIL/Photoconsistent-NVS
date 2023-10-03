
import os
import sys
from os.path import join as pjoin

# add parent dir to path
script_path = os.path.dirname(os.path.realpath(sys.argv[0]))
src_path = os.path.abspath(os.path.join(script_path,'..'))
sys.path.append(src_path)
os.chdir(src_path)

import argparse
argParser = argparse.ArgumentParser(description="start/resume training")
argParser.add_argument("-g","--gpu",dest="gpu",action="store",default=0,type=int)
argParser.add_argument("-o",dest="output_dir",action="store",default='default-output',type=str)
argParser.add_argument("-s",dest="steps",action="store",default=2000,type=int)
argParser.add_argument("-e",dest="epoch",action="store",default=2000,type=int)
cli_args = argParser.parse_args()

from utils import *
# set cuda visible if not already defined
if "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = str(cli_args.gpu)
	tlog('CUDA_VISIBLE_DEVICES set to: {}'.format(cli_args.gpu), 'note')
else:
	tlog('CUDA_VISIBLE_DEVICES already set, ignoring manual gpu selection', 'note')

from models.score_sde import ncsnpp
import models.score_sde.sde_lib as sde_lib
from models.score_sde.ncsnpp_dual import NCSNpp_dual
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.optim as optim
import json
from models.score_sde.configs.LDM import LDM_config
import shutil
from models.vqgan.vqgan import VQModel
from models.vqgan.configs.vqgan_32_4 import vqgan_32_4_config
from models.score_sde.ema import ExponentialMovingAverage
import functools
from models.score_sde.layerspp import ResnetBlockBigGANpp
import torch.nn.functional as F
import pudb
from filelock import FileLock
import time

globals.instance_data_path = '../instance-data'
init_instance_dirs([cli_args.output_dir])
outdir = pjoin(globals.instance_data_path,cli_args.output_dir)

class Sampling_job:
	def __init__(self,scene_name,scene_root):
		self.scene_name = scene_name
		self.scene_root = scene_root
		with open(pjoin(self.scene_root,'sampling-spec.json')) as f:
			self.gen_spec = json.load(f)
		self.all_poses = [np.asarray(x) for x in self.gen_spec['poses']]
		self.generation_order = self.gen_spec['generation_order']
		self.dependencies = self.gen_spec['dependencies']
		self.n_generations = len(self.generation_order)
		self.cur_gen_run = None
		self.latent_path = None
		self.image_path = None
		self.next_gen_idx = None
	def get_runs(self):
		gen_runs = [int(x) for x in os.listdir(pjoin(self.scene_root,'samples')) if x != '.gitignore']
		gen_runs.sort()
		return gen_runs
	def create_new_run(self):
		existing_runs = self.get_runs()
		self.cur_gen_run = existing_runs[-1]+1 if len(existing_runs) > 0 else 0
		self.set_run(existing_runs[-1]+1)
	def set_run(self,run_idx):
		self.cur_gen_run = run_idx
		self.latent_path = pjoin(self.scene_root,'samples',f'{run_idx:08d}','latents')
		self.image_path = pjoin(self.scene_root,'samples',f'{run_idx:08d}','images')
		existing_runs = self.get_runs()
		if run_idx in existing_runs: # resume
			existing_latents = [int(x[:-4]) for x in os.listdir(self.latent_path) if x.endswith('.npy')]
			latest_generation = -1
			for latent_idx in existing_latents:
				# find which one is the largest in the generation order
				for gen_idx, pose_idx in enumerate(self.generation_order):
					if latent_idx == pose_idx and gen_idx > latest_generation:
						latest_generation = gen_idx
						break
			self.next_gen_idx = latest_generation + 1
		else: # create
			os.makedirs(self.latent_path)
			os.makedirs(self.image_path,exist_ok=True)
			self.next_gen_idx = 0
	def save_generation(self,pose_idx,decoded_image,latent):
		im_path = pjoin(job.image_path,f'{pose_idx:04d}.png')
		latent_path = pjoin(self.latent_path,f'{pose_idx:04d}.npy')
		im_out = torch.clip((decoded_image/2+0.5).permute(1,2,0).cpu()* 255., 0, 255).type(torch.uint8).numpy()
		im_out = Image.fromarray(im_out)
		im_out.save(im_path)
		with open(latent_path+'.part','wb') as f:
			np.save(f,latent.unsqueeze(0).cpu().detach().numpy())
		os.replace(latent_path+'.part',latent_path) # rename for atomic save

if __name__ == '__main__':
	# set up coordination
	sampling_jobs = []

	# vqgan
	vqgan = VQModel(**vqgan_32_4_config).cuda()
	vqgan.eval()

	# ========================= build model =========================
	config = LDM_config()
	score_model = NCSNpp_dual(config)
	score_model.cuda()
	sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=cli_args.steps)

	# ray downsampler
	ResnetBlock = functools.partial(ResnetBlockBigGANpp,
									act=torch.nn.SiLU(),
									dropout=False,
									fir=True,
									fir_kernel=[1,3,3,1],
									init_scale=0,
									skip_rescale=True,
									temb_dim=None)
	ray_downsampler = torch.nn.Sequential(
		ResnetBlock(in_ch=56,out_ch=128,down=True),
		ResnetBlock(in_ch=128,out_ch=128,down=True)).cuda()
	score_sde = Score_sde_model(score_model,sde,ray_downsampler,rays_require_downsample=False,rays_as_list=True)

	# load model, need to adapt, DDP changes key names
	checkpoint_epoch = cli_args.epoch
	checkpoint = torch.load(f'../instance-data/checkpoints/{checkpoint_epoch:04d}-00000000.pth')
	adapted_state = {}
	for k,v in checkpoint['score_sde_model'].items():
		key_parts = k.split('.')
		if key_parts[1] == 'module':
			key_parts.pop(1)
		new_key  = '.'.join(key_parts)
		adapted_state[new_key] = v
	score_sde.load_state_dict(adapted_state)

	ema = ExponentialMovingAverage(score_sde.parameters(),decay=0.999)
	ema.load_state_dict(checkpoint['ema'])
	ema.copy_to(score_sde.parameters())

	# substitute model with score modifier, used to make bulk sampling easier
	modifier = Score_modifier(score_sde.score_model,max_batch_size=1)
	score_sde.score_model = modifier

	tlog('Setup complete','note')

	# ========================= start generation =========================
	with torch.no_grad():
		sampling_jobs = [Sampling_job(outdir,outdir)]
		sampling_jobs[0].set_run(0)
		while True:
			# exit if we are done
			if sampling_jobs[0].next_gen_idx >= len(sampling_jobs[0].generation_order):
				print('Sampling complete')
				exit()

			n_sampling_jobs = len(sampling_jobs)
			# ========================= construct conditioning data
			conditioning_ims = []
			ff_refs = []
			ff_as = []
			ff_bs = []
			for job in sampling_jobs:
				job_cond_ims = []
				job_ff_refs = []
				job_ff_as = []
				job_ff_bs = []
				job_gen_idx = job.generation_order[job.next_gen_idx]
				for dep in job.dependencies[job_gen_idx]:
					# load conditioning images
					dep_latent_path = pjoin(job.latent_path,f'{dep:04d}.npy')
					dep_init_image_path = pjoin(job.scene_root,'init-ims',f'{dep:04d}.png')
					if os.path.exists(dep_latent_path):
						encoded_im = torch.Tensor(np.load(dep_latent_path)).cuda()
						job_cond_ims.append(encoded_im)
					else:
						assert os.path.exists(dep_init_image_path), 'Latent failed to save, or init image not provided'
						im = Image.open(dep_init_image_path)
						im = np.asarray(im)[:,:,:3].astype(np.float32).transpose(2,0,1)/127.5 - 1
						im = torch.Tensor(im).unsqueeze(0).cuda()
						encoded_im = vqgan.encode(im)
						job_cond_ims.append(encoded_im)

						# also save reconstruction
						decoded = vqgan.decode(encoded_im)
						job.save_generation(dep,decoded[0,...],encoded_im[0,...])

					# load encoded rays
					focal_y = job.gen_spec['focal_y']
					tform_a = np.array(job.all_poses[job_gen_idx])
					tform_b = np.array(job.all_poses[dep])
					tform_ref = np.eye(4)
					tform_a_inv = np.linalg.inv(tform_a)
					tform_b_inv = np.linalg.inv(tform_b)
					tform_a_relative = np.matmul(tform_b_inv,tform_a)
					tform_b_relative = np.matmul(tform_a_inv,tform_b)
					camera_enc_ref = rel_camera_ray_encoding(tform_ref,128,focal_y)
					camera_enc_a = rel_camera_ray_encoding(tform_a_relative,128,focal_y)
					camera_enc_b = rel_camera_ray_encoding(tform_b_relative,128,focal_y)
					camera_enc_ref = torch.Tensor(camera_enc_ref).unsqueeze(0)
					camera_enc_a = torch.Tensor(camera_enc_a).unsqueeze(0)
					camera_enc_b = torch.Tensor(camera_enc_b).unsqueeze(0)
					ff_ref = F.pad(freq_enc(camera_enc_ref),[0,0,0,0,1,1,0,0]) # pad, must be %4==0 for group norm
					ff_a = F.pad(freq_enc(camera_enc_a),[0,0,0,0,1,1,0,0])
					ff_b = F.pad(freq_enc(camera_enc_b),[0,0,0,0,1,1,0,0])
					job_ff_refs.append(ray_downsampler(ff_ref.cuda()))
					job_ff_as.append(ray_downsampler(ff_a.cuda()))
					job_ff_bs.append(ray_downsampler(ff_b.cuda()))

				conditioning_ims.append(job_cond_ims)
				ff_refs.append(job_ff_refs)
				ff_as.append(job_ff_as)
				ff_bs.append(job_ff_bs)

			# sampling loop
			sampling_shape = (n_sampling_jobs, 4, 32, 32)
			sampling_eps = 1e-5
			device = 'cuda:0'
			x = score_sde.sde.prior_sampling(sampling_shape).cuda()

			timesteps = torch.linspace(sde.T, sampling_eps, sde.N, device=device)
			for i in tqdm(range(0,sde.N)):
				t = timesteps[i]
				vec_t = torch.ones(sampling_shape[0], device=t.device) * t
				_, std = sde.marginal_prob(x, vec_t)
				x, x_mean = score_sde.reverse_diffusion_predictor(x,conditioning_ims,vec_t,ff_refs,ff_as,ff_bs)
				x, x_mean = score_sde.langevin_corrector(x,conditioning_ims,vec_t,ff_refs,ff_as,ff_bs)

			# decode latent rep
			decoded = vqgan.decode(x_mean)
			intermediate_sample = (decoded/2+0.5)
			intermediate_sample = torch.clip(intermediate_sample.permute(0,2,3,1).cpu()* 255., 0, 255).type(torch.uint8).numpy()

			for job_idx,job in enumerate(sampling_jobs):
				pose_idx = job.generation_order[job.next_gen_idx]
				job.save_generation(pose_idx,decoded[job_idx,...],x_mean[job_idx,...])
				job.next_gen_idx += 1



