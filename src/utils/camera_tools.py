import numpy as np
import torch

def rel_camera_ray_encoding(tform,im_size,focal):
	# create camera ray encoding
	# assumes square images with same focal length on all axes, assume principle = 0.5
	cam_center = tform[:3,-1]
	cam_rot = tform[:3,:3]

	# find max limit from focal length
	max_pos = 1/(2*focal) # our images are normalizedto -1,1

	# find rays for all pixels
	pix_size = 2/im_size
	x,y = np.meshgrid(range(im_size),range(im_size),indexing='xy')
	x = 2*x/(im_size-1) - 1
	y = 2*y/(im_size-1) - 1
	x *= max_pos # scale to match focal length
	y *= max_pos
	pix_grid = np.stack([x,-y],0)
	ray_grid = np.concatenate([pix_grid,-np.ones(shape=[1,im_size,im_size])],0)
	ray_grid_flat = ray_grid.reshape(3,-1)
	
	rays = np.matmul(cam_rot,ray_grid_flat)
	rays = rays/np.linalg.norm(rays,2,0)
	rays = rays.reshape(3,im_size,im_size)

	camera_center = np.tile(cam_center[:,None,None],(1,im_size,im_size))
	camera_data = np.concatenate([rays,camera_center],0)

	camera_data = camera_data.astype(np.float32)
	return camera_data

def abs_cameras_freq_encoding(pose_a,pose_b,focal_y):
	new_rel = np.matmul(np.linalg.inv(pose_a),pose_b)
	camera_data_a = rel_camera_ray_encoding(np.eye(4),128,focal_y)
	camera_data_b = rel_camera_ray_encoding(new_rel,128,focal_y)
	camera_data_a = torch.Tensor(camera_data_a).unsqueeze(0).cuda()
	camera_data_b = torch.Tensor(camera_data_b).unsqueeze(0).cuda()
	fourier_feats = torch.cat([freq_enc(camera_data_a),freq_enc(camera_data_b)],1)
	return fourier_feats

def freq_enc(camera_data,n_frequencies=4,half_period=6):
	# encodeing does not repeat until [-half_period, half_period]
	n_in_channels = camera_data.shape[1]
	frequency_exponent = torch.arange(n_frequencies)
	frequency_multiplier = (2.0**frequency_exponent)/half_period
	frequency_multiplier = frequency_multiplier.tile(n_in_channels,1).T.reshape(-1)[None,:,None,None]*torch.pi
	camera_data_tiled = camera_data.tile(1,n_frequencies,1,1)*frequency_multiplier.to(camera_data.device)
	camera_data_sin = torch.sin(camera_data_tiled)
	camera_data_cos = torch.cos(camera_data_tiled)
	fourier_feats = torch.cat([camera_data,camera_data_sin,camera_data_cos],1)
	return fourier_feats
