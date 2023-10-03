

vqgan_32_4_config = {
	'ckpt_path': "../instance-data/taming-32-4-realestate-256.ckpt",
	'embed_dim': 4,
	# 'n_embed': 16384,
	'n_embed': 16384,
	'ddconfig': {
		'double_z': False,
		'z_channels': 4,
		'resolution': 256,
		'in_channels': 3,
		'out_ch': 3,
		'ch': 128,
		'ch_mult': [ 1,2,2,4],  # num_down = len(ch_mult)-1
		'num_res_blocks': 2,
		'attn_resolutions': [ 16 ],
		'dropout': 0.0,
	}
}

