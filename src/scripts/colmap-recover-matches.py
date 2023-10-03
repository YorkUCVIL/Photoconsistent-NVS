
import os
import sys
from os.path import join as pjoin

# add parent dir to path
script_path = os.path.dirname(os.path.realpath(sys.argv[0]))
src_path = os.path.abspath(os.path.join(script_path,'..'))
sys.path.append(src_path)
os.chdir(src_path)

# args
import argparse
argParser = argparse.ArgumentParser(description="start/resume training")
argParser.add_argument("-o",dest="output_dir",action="store",default='default-output',type=str)
cli_args = argParser.parse_args()

import subprocess
from colmap_utils import *

outdir = pjoin('../instance-data',cli_args.output_dir)

if __name__ == '__main__':
	scene_root = outdir
	gen_runs = os.listdir(pjoin( scene_root,'samples' ))
	gen_runs.sort()
	newest_gen_run = gen_runs[-1]

	# create colmap dir
	colmap_root = pjoin(scene_root,'colmap',newest_gen_run)
	os.makedirs(colmap_root,exist_ok=False)
	im_path = pjoin(scene_root,'samples',newest_gen_run,'images')
	database_path = pjoin(colmap_root,'colmap.db')

	# feature extraction
	cmd = f'colmap feature_extractor --SiftExtraction.use_gpu 1 --SiftExtraction.edge_threshold 30 --ImageReader.camera_model PINHOLE --ImageReader.single_camera 1 --database_path {database_path} --image_path {im_path}'
	subprocess.run(cmd,shell=True,capture_output=True)

	# feature matcher
	cmd = f'colmap exhaustive_matcher --database_path {database_path}'
	subprocess.run(cmd,shell=True,capture_output=True)
