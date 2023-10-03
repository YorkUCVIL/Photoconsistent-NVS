import torch
from torch.utils import data
from os.path import join as pjoin
import os
from PIL import Image
import numpy as np
import json
import sys
from utils import *
import torchvision
from .sequence_blacklist import sequence_blacklist

class Realestate_dataset_video(data.Dataset):
	def __init__(self,split):
		super().__init__()
		self.root = f'../dataset-data/data/{split}/'
		self.video_root = pjoin(self.root,'videos')
		self.poses = np.load(pjoin(self.root,'poses.npy'),allow_pickle=True).item()

		# remove bad seequences
		for code in sequence_blacklist[split]:
			del self.poses[code]

		self.sequence_codes = list(self.poses.keys())
		self.im_size = 128


	def __len__(self):
		return len(self.sequence_codes)

	def __getitem__(self,idx):
		# idx indicates sequence idx, we will pick 2 random frames from there
		poses = self.poses[self.sequence_codes[idx]]
		sequence_max_len = len(poses['timestamp'])
		available_idxs = list(range(len(poses['timestamp'])-1)) # exclude last frame, might have fade between shots

		# choose frames
		chosen_idxs = np.random.choice(available_idxs,2,replace=False)

		# extract frames from video
		video_code = poses['video_code']
		video_path = pjoin(self.video_root,f'{video_code}.mp4')
		video_reader = torchvision.io.VideoReader(video_path)

		timestamp = poses['timestamp'][chosen_idxs[0]]
		seconds = timestamp/1000000
		for frame in video_reader.seek(seconds):
			frame_a = frame['data'].numpy().transpose(1,2,0)
			break

		timestamp = poses['timestamp'][chosen_idxs[1]]
		seconds = timestamp/1000000
		for frame in video_reader.seek(seconds):
			frame_b = frame['data'].numpy().transpose(1,2,0)
			break

		# ensure images have 360 height
		if frame_a.shape[0] != 360:
			new_w = round(frame_a.shape[1] * (360/frame_a.shape[0]))
			frame_a = np.asarray(Image.fromarray(frame_a).resize((new_w,360)))
			frame_b = np.asarray(Image.fromarray(frame_b).resize((new_w,360)))

		# crop and downsample
		left_pos = (frame_a.shape[1]-360)//2
		frame_a_cropped = frame_a[:,left_pos:left_pos+360,:]
		frame_b_cropped = frame_b[:,left_pos:left_pos+360,:]
		im_a = Image.fromarray(frame_a_cropped).resize((256,256))
		im_b = Image.fromarray(frame_b_cropped).resize((256,256))
		im_a = np.asarray(im_a).transpose(2,0,1)/127.5 - 1
		im_b = np.asarray(im_b).transpose(2,0,1)/127.5 - 1

		# scale image values
		frame_a_cropped = frame_a_cropped/127.5 - 1
		frame_b_cropped = frame_b_cropped/127.5 - 1

		# load and process transforms, 3x4 -> 4x4
		tform_a = np.concatenate([poses['pose'][chosen_idxs[0]],[[0,0,0,1]]],0)
		tform_b = np.concatenate([poses['pose'][chosen_idxs[1]],[[0,0,0,1]]],0)
		
		tform_a_inv = np.linalg.inv(tform_a)
		tform_b_inv = np.linalg.inv(tform_b)
		tform_ref = np.eye(4)
		tform_a_relative = np.matmul(tform_b_inv,tform_a)
		tform_b_relative = np.matmul(tform_a_inv,tform_b)
		
		# get focal length, create camera ray encoding
		focal_y_a = poses['focal_y'][chosen_idxs[0]] # these should be the same
		focal_y_b = poses['focal_y'][chosen_idxs[1]]
		camera_enc_ref = rel_camera_ray_encoding(tform_ref,self.im_size,focal_y_a)
		camera_enc_a = rel_camera_ray_encoding(tform_a_relative,self.im_size,focal_y_a)
		camera_enc_b = rel_camera_ray_encoding(tform_b_relative,self.im_size,focal_y_b)
		
		out_dict = {
			'sequence_code':self.sequence_codes[idx],
			'im_a': im_a.astype(np.float32),
			'im_b': im_b.astype(np.float32),
			'im_a_full': frame_a_cropped.transpose(2,0,1).astype(np.float32),
			'im_b_full': frame_b_cropped.transpose(2,0,1).astype(np.float32),
			'camera_enc_ref': camera_enc_ref,
			'camera_enc_a': camera_enc_a,
			'camera_enc_b': camera_enc_b,
			'tform_ref': tform_ref,
			'tform_a_relative': tform_a_relative,
			'tform_b_relative': tform_b_relative,
			'tform_ref': tform_ref,
			'tform_a': tform_a,
			'tform_b': tform_b,
			'focal_a': focal_y_a,
			'focal_b': focal_y_b,
		}
		return out_dict

