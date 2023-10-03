
class Conditional_config:
	def __init__(self):
		self.model = lambda: None
		self.seed = lambda: None
		self.data = lambda: None
		self.eval = lambda: None
		self.sampling = lambda: None
		self.training = lambda: None

		self.training.continuous = True

		self.data.dataset = 'LSUN'
		self.data.image_size = 128
		self.data.random_flip = True
		self.data.uniform_dequantization = False
		self.data.centered = False
		self.data.num_channels = 3

		self.sampling.n_steps_each = 1
		self.sampling.noise_removal = True
		self.sampling.probability_flow = False
		self.sampling.snr = 0.075
		self.sampling.method = 'pc'
		self.sampling.predictor = 'reverse_diffusion'
		self.sampling.corrector = 'langevin'

		self.model.sigma_max = 378
		self.model.sigma_min = 0.01
		self.model.num_scales = 2000
		self.model.beta_min = 0.1
		self.model.beta_max = 20.
		self.model.dropout = 0.
		self.model.embedding_type = 'fourier'
		self.model.name = 'ncsnpp'
		self.model.scale_by_sigma = True
		self.model.ema_rate = 0.999
		self.model.normalization = 'GroupNorm'
		self.model.nonlinearity = 'swish'
		self.model.nf = 128
		# self.model.ch_mult = (1, 1, 2, 2, 2, 2, 2)
		self.model.ch_mult = (1, 1, 2, 2, 2, 2)
		self.model.num_res_blocks = 2
		self.model.attn_resolutions = (16,)
		self.model.resamp_with_conv = True
		self.model.conditional = True
		self.model.fir = True
		self.model.fir_kernel = [1, 3, 3, 1]
		self.model.skip_rescale = True
		self.model.resblock_type = 'biggan'
		self.model.progressive = 'output_skip'
		self.model.progressive_input = 'input_skip'
		self.model.progressive_combine = 'sum'
		self.model.attention_type = 'ddpm'
		self.model.init_scale = 0.
		self.model.fourier_scale = 16
		self.model.conv_size = 3

