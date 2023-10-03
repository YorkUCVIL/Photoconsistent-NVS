import os
from os.path import join as pjoin
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
import sys

split = sys.argv[1]
sequence_root = f'data/RealEstate10K-original/{split}/'
sequence_files = os.listdir(sequence_root)
sequence_files.sort()
sequence_files = [pjoin(sequence_root,x) for x in sequence_files]

def orb_to_blender(orb_t):
	# blender start with camera looking down, -z forward
	# orb starts with camera looking forward +z forward
	pre_conversion = np.array([ # orb starts with +z forward, +y down
		[1,0,0,0],
		[0,-1,0,0],
		[0,0,-1,0],
		[0,0,0,1],
	])
	conversion = np.array([ # converts +y down world to z+ up world
		[1,0,0,0],
		[0,0,1,0],
		[0,-1,0,0],
		[0,0,0,1],
	])
	camera_local = np.linalg.inv(orb_t)
	orb_world = np.matmul(camera_local,pre_conversion)
	blender_world = np.matmul(conversion,orb_world)
	return blender_world

def extract_poses(seq_fn):
	seq_code = seq_fn[seq_fn.rindex('/')+1:-4]
	# my_seq_path = f'../data/{split}/poses/{seq_code}.npy'
	with open(seq_fn) as f:
		url = f.readline()
		video_code = url[url.index('=')+1:-1]

		# check if video dir exists
		frame_dir = f'data/{split}/videos/{video_code}.mp4'
		if not os.path.exists(frame_dir):
			print(f'Video does not exist: {video_code}')
			return

		# read all lines
		lines = []
		for line in f:
			lines.append(line)
		n_poses = len(lines)

		# reject sequences with less than 2 poses
		if n_poses < 2:
			print('too view frames, skipping')
			return

		out_dict = {
			'video_code': video_code,
			'timestamp':np.zeros(shape=[n_poses],dtype=np.int32),
			'focal_x':np.zeros(shape=[n_poses],dtype=np.float32),
			'focal_y':np.zeros(shape=[n_poses],dtype=np.float32),
			'princ_x':np.zeros(shape=[n_poses],dtype=np.float32),
			'princ_y':np.zeros(shape=[n_poses],dtype=np.float32),
			'pose':np.zeros(shape=[n_poses,3,4],dtype=np.float32),
		}

		# extract poses
		for n,line in enumerate(lines):
			fields = line.split(' ')
			timestamp_micro = int(fields[0])
			intrin = fields[1:5]
			extrin = fields[7:]
			extrin = [float(x) for x in extrin]+[0,0,0,1]
			extrin = np.array(extrin).reshape(4,4)
			extrin = orb_to_blender(extrin)
			out_dict['timestamp'][n] = timestamp_micro
			out_dict['focal_x'][n] = intrin[0]
			out_dict['focal_y'][n] = intrin[1]
			out_dict['princ_x'][n] = intrin[2]
			out_dict['princ_y'][n] = intrin[3]
			out_dict['pose'][n] = extrin[:-1,:]
		# np.save(my_seq_path,out_dict)
		return out_dict


with Pool(4) as p:
	result = list(tqdm(p.imap(extract_poses,sequence_files),total=len(sequence_files)))
	all_dict = {}
	for sequence_file, data in zip(sequence_files,result):
		if data is None: continue
		seq_code = sequence_file[sequence_file.rindex('/')+1:-4]
		all_dict[seq_code] = data

	np.save(f'data/{split}/poses.npy',all_dict)
