

vqgan_32_256_config = {
	'ckpt_path': "../instance-data/vqgan-32-256.ckpt",
	'embed_dim': 256,
	# 'n_embed': 16384,
	'n_embed': 1024,
	'ddconfig': {
		'double_z': False,
		'z_channels': 256,
		'resolution': 128,
		'in_channels': 3,
		'out_ch': 3,
		'ch': 128,
		'ch_mult': [ 1,2,4 ],  # num_down = len(ch_mult)-1
		'num_res_blocks': 2,
		'attn_resolutions': [ 16 ],
		'dropout': 0.0,
	}
}

