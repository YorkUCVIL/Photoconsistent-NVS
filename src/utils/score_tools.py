import torch
import random
import pudb

class Score_sde_model(torch.nn.Module):
	def __init__(self,score_model,sde,ray_downsampler,rays_require_downsample=True):
		super().__init__()
		self.score_model = score_model
		self.sde = sde
		self.rsde = sde.reverse(self.score,probability_flow=False)

		self.ray_downsampler = ray_downsampler
		self.rays_require_downsample = rays_require_downsample

	def score(self,x,cond_im,t, ff_ref, ff_a, ff_b):
		# d_ff_ref = self.ray_downsampler(ff_ref) if self.rays_require_downsample else ff_ref
		# d_ff_a = self.ray_downsampler(ff_a) if self.rays_require_downsample else ff_a
		# d_ff_b = self.ray_downsampler(ff_b) if self.rays_require_downsample else ff_b
		d_ff_ref = [self.ray_downsampler(rays) for rays in ff_ref] if self.rays_require_downsample else ff_ref
		d_ff_a = [self.ray_downsampler(rays) for rays in ff_a] if self.rays_require_downsample else ff_a
		d_ff_b = [self.ray_downsampler(rays) for rays in ff_b] if self.rays_require_downsample else ff_b
		_, std = self.sde.marginal_prob(torch.zeros_like(x), t)
		cond_std = torch.ones_like(std)*0.01 # assume conditioning image has minimal noise
		score_a, score_b = self.score_model(x, cond_im, std, cond_std, d_ff_ref, d_ff_a, d_ff_b) # ignore second score
		return score_a

	def forward_diffusion(self,x,t):
		z = torch.randn_like(x)
		mean, std = self.sde.marginal_prob(x, t)
		perturbed_data = mean + std[:, None, None, None] * z
		return perturbed_data, z, std

	def t_uniform(self,batch_size,device=None,eps=1e-5):
		# eps prevents sampling exactly 0
		t = torch.rand(batch_size, device=device) * (self.sde.T - eps) + eps
		return t

	def reverse_diffusion_predictor(self, x, cond_im, t, ff_ref, ff_a, ff_b):
		f, G = self.rsde.discretize(x, cond_im, t, ff_ref, ff_a, ff_b)
		z = torch.randn_like(x)
		x_mean = x - f
		x = x_mean + G[:, None, None, None] * z
		return x, x_mean

	def langevin_corrector(self, x, cond_im, t, ff_ref, ff_a, ff_b):
		sde = self.sde
		n_steps = 1
		target_snr = 0.075

		# specific to VESDE
		alpha = torch.ones_like(t)

		for i in range(n_steps):
			grad = self.score(x, cond_im, t, ff_ref, ff_a, ff_b)
			noise = torch.randn_like(x)
			grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
			noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
			step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
			x_mean = x + step_size[:, None, None, None] * grad
			x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

		return x, x_mean

class Score_sde_monocular_model(torch.nn.Module):
	def __init__(self,score_model,sde):
		super().__init__()
		self.score_model = score_model
		self.sde = sde
		self.rsde = sde.reverse(self.score,probability_flow=False)

	def score(self,x,t):
		_, std = self.sde.marginal_prob(torch.zeros_like(x), t)
		score = self.score_model(x, std)
		return score

	def forward_diffusion(self,x,t):
		z = torch.randn_like(x)
		mean, std = self.sde.marginal_prob(x, t)
		perturbed_data = mean + std[:, None, None, None] * z
		return perturbed_data, z, std

	def t_uniform(self,batch_size,device=None,eps=1e-5):
		# eps prevents sampling exactly 0
		t = torch.rand(batch_size, device=device) * (self.sde.T - eps) + eps
		return t

	def reverse_diffusion_predictor(self, x, t):
		f, G = self.rsde.discretize(x, t)
		z = torch.randn_like(x)
		x_mean = x - f
		x = x_mean + G[:, None, None, None] * z
		return x, x_mean

	def langevin_corrector(self, x, t):
		sde = self.sde
		n_steps = 1
		target_snr = 0.075

		# specific to VESDE
		alpha = torch.ones_like(t)

		for i in range(n_steps):
			grad = self.score(x, t)
			noise = torch.randn_like(x)
			grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
			noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
			step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
			x_mean = x + step_size[:, None, None, None] * grad
			x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

		return x, x_mean

class Score_modifier(torch.nn.Module):
	def __init__(self,model,max_batch_size):
		super().__init__()
		self.model = model 
		self.max_batch_size = max_batch_size

	def __call__(self,x,cond_ims,std,cond_std,ff_refs,ff_as,ff_bs):
		# assert len(ff_refs) == len(ff_as) == len(ff_bs) == len(cond_ims)

		# generate indices to batch data
		indices = []
		for b in range(len(ff_refs)):
			for d in range(len(ff_refs[b])):
				indices.append([b,d])

		# compute scores in batches
		independent_scores_a = [[] for _ in ff_refs]
		independent_scores_b = [[] for _ in ff_refs]
		for start in range(0,len(indices),self.max_batch_size):
			batch_indices = indices[start:start+self.max_batch_size]
			batch_x = torch.stack([x[b,...] for [b,_] in batch_indices],0)
			batch_cond_ims = torch.cat([cond_ims[b][n] for [b,n] in batch_indices],0)
			batch_ff_refs = torch.cat([ff_refs[b][n] for [b,n] in batch_indices],0)
			batch_ff_as = torch.cat([ff_as[b][n] for [b,n] in batch_indices],0)
			batch_ff_bs = torch.cat([ff_bs[b][n] for [b,n] in batch_indices],0)
			batch_std = torch.stack([std[b] for [b,_] in batch_indices],0)
			batch_cond_std = torch.stack([cond_std[b] for [b,_] in batch_indices],0)
			batch_score_a,batch_score_b = self.model(batch_x,batch_cond_ims,batch_std,batch_cond_std,batch_ff_refs,batch_ff_as,batch_ff_bs)
			for idx,[b,n] in enumerate(batch_indices): # unpack scores
				independent_scores_a[b].append(batch_score_a[idx,...])
				independent_scores_b[b].append(batch_score_b[idx,...])

		aggregated_score_a = [torch.stack(sss,0).mean(0) for sss in independent_scores_a]
		aggregated_score_b = [torch.stack(sss,0).mean(0) for sss in independent_scores_b]

		aggregated_score_a = torch.stack(aggregated_score_a,0)
		aggregated_score_b = torch.stack(aggregated_score_b,0)

		return aggregated_score_a,aggregated_score_b

	def train(self): self.model.train()
	def eval(self): self.model.eval()

class Score_modifier_stochastic(torch.nn.Module):
	def __init__(self,model):
		super().__init__()
		self.model = model 

	def __call__(self,x,cond_ims,std,cond_std,ff_refs,ff_as,ff_bs):
		assert len(ff_refs) == len(ff_as) == len(ff_bs) == len(cond_ims)
		independent_scores_a = []
		independent_scores_b = []
		n_conditioning_views = len(ff_refs)

		n = random.choice(list(range(n_conditioning_views)))
		cond_im = cond_ims[n]
		score_a,score_b = self.model(x,cond_im,std,cond_std,ff_refs[n],ff_as[n],ff_bs[n])

		return score_a,score_b

	def train(self): self.model.train()
	def eval(self): self.model.eval()

class Score_modifier_stochastic_sanity(torch.nn.Module):
	def __init__(self,model):
		super().__init__()
		self.model = model 

	def __call__(self,x,cond_ims,std,cond_std,ff_refs,ff_as,ff_bs):
		assert len(ff_refs) == len(ff_as) == len(ff_bs) == len(cond_ims)
		independent_scores_a = []
		independent_scores_b = []
		n_conditioning_views = len(ff_refs)

		n = random.choice(list(range(n_conditioning_views)))
		# if n_conditioning_views == 1:
		#     n = 0
		# else:
		#     n = 1
		# n = 0
		cond_im = cond_ims[n]
		score_a,score_b = self.model(x,cond_im,std,cond_std,ff_refs[n],ff_as[n],ff_bs[n])

		return score_a,score_b

	def train(self): self.model.train()
	def eval(self): self.model.eval()
