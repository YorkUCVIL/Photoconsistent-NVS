
import os
import sys
from os.path import join as pjoin
import argparse

# add parent dir to path
script_path = os.path.dirname(os.path.realpath(sys.argv[0]))
src_path = os.path.abspath(os.path.join(script_path,'..'))
sys.path.append(src_path)
os.chdir(src_path)

argParser = argparse.ArgumentParser(description="")
argParser.add_argument("-o",dest="output_dir",action="store",default='gt',type=str)
argParser.add_argument("--te",dest="t_e",action="store",default='2',type=float)
argParser.add_argument("--tm",dest="t_m",action="store",default='10',type=float)
cli_args = argParser.parse_args()

import json
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pudb

def compute_consistency(outdir,intra_scene_sed_threshold,intra_scene_min_consistent):
	scene_root = os.path.join('../instance-data',outdir)
	gen_runs = os.listdir(pjoin( scene_root,'samples' ))
	gen_runs.sort()
	newest_gen_run = gen_runs[-1]

	sed_path = pjoin(scene_root,'evaluation',newest_gen_run,'sed','neighbors.json')
	with open(sed_path) as f:
		sed_data = json.load(f)
	n_pairs = len(sed_data)

	n_consistent_pairs = 0
	for raw_data in sed_data:
		median = raw_data['median']
		n_matches = raw_data['n_matches']
		if n_matches >= intra_scene_min_consistent and median < intra_scene_sed_threshold:
			n_consistent_pairs += 1

	return n_consistent_pairs, n_pairs

min_matches = cli_args.t_m
threshold_error = cli_args.t_e
n_consistent_pairs, n_pairs = compute_consistency(cli_args.output_dir,threshold_error,min_matches)
print(f'{n_consistent_pairs}/{n_pairs} frame pairs are consistent')
