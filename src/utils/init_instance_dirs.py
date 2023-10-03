import os
from utils import globals
from os.path import join as pjoin

def init_instance_dirs(required_dirs=[]):
	root = globals.instance_data_path
	for sub_path in required_dirs:
		rel_path = pjoin(root,sub_path)
		if not os.path.isdir(rel_path):
			os.mkdir(rel_path)
