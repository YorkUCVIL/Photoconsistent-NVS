from utils import *
from os.path import join as pjoin
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import time

class Base_trainer:
	def __init__(self,compute_only=True):
		self.epoch = 0 # future: rename to epochs_completed
		self.total_iterations = 0
		self.max_epoch = 200
		self.n_kept_checkpoints = 0
		self.compute_only = compute_only
		self.last_checkpoint_time = time.time()
		self.checkpoint_interval = 0
		self.checkpoint_retention_interval = 0
		self.termination_requested = False

		if not self.compute_only:
			log_dir = os.path.join(globals.instance_data_path,'logs')
			self.tb_writer = SummaryWriter(log_dir=log_dir)

	def train(self):
		for epochs_completed in range(self.epoch, self.max_epoch):
			self.train_epoch()
			self.epoch += 1 # epoch done, +1 for correct saving

			# validate
			self.validate()

			# checkpoint
			if not self.compute_only:
				self.tlog(f'Epoch {self.epoch} complete, checkpointing','note')
				self.save_checkpoint(f'{self.epoch:04d}-{0:08d}.pth')
			else: # load checkpoint to sync params
				expected_checkpoint = f'../instance-data/checkpoints/{self.epoch:04d}-{0:08d}.pth'
				while not os.path.exists(expected_checkpoint):
					time.sleep(5)
				self.load_checkpoint(expected_checkpoint)

	def train_epoch(self):
		raise NotImplementedError

	def validate(self):
		raise NotImplementedError

	def state_dict(self):
		raise NotImplementedError

	def load_state_dict(self,state_dict):
		raise NotImplementedError

	def tlog(self,text,mode='debug'):
		if not self.compute_only:
			tlog(text,mode)

	def check_termination_request(self,epoch_it,dataloader_iter=None):
		if self.termination_requested:
			if self.compute_only: exit()
			self.tlog('Termination requested, checkpointing','note')
			self.save_checkpoint(f'{self.epoch:04d}-{epoch_it:08d}.pth',dataloader_iter)
			exit()

	def maybe_save_checkpoint(self,epoch_it,dataloader_iter=None):
		# saves checkpoint if haven't saved in more than checkpoint_interval
		if self.compute_only: return
		if self.checkpoint_interval == 0: return
		if (time.time() - self.last_checkpoint_time) < self.checkpoint_interval: return
		self.tlog(f'Haven\'t checkpointed in over {self.checkpoint_interval} seconds, checkpointing','note')
		self.save_checkpoint(f'{self.epoch:04d}-{epoch_it:08d}.pth',dataloader_iter)

	def save_checkpoint(self,checkpoint_filename,dataloader_iter=None):
		checkpoint_root = '../instance-data/checkpoints'
		# save model, atomic, saves as .part, then swaps when done
		tmp_out_path = pjoin(checkpoint_root,checkpoint_filename+'.part')
		out_path = pjoin(checkpoint_root,checkpoint_filename)
		checkpoint = self.state_dict(dataloader_iter)
		torch.save(checkpoint,tmp_out_path)
		os.replace(tmp_out_path,out_path)
		self.tlog(f'Checkpoint saved to: {out_path}','note')

		self.last_checkpoint_time = time.time()

		# don't clear checkpoints if n_kept_checkpoints == 0
		if self.n_kept_checkpoints == 0: return

		# delete excess checkpoints
		checkpoints = os.listdir(checkpoint_root)
		checkpoints = [x for x in checkpoints if x.endswith('.pth')]
		if self.checkpoint_retention_interval > 0: # ignore any  matching retention criterea
			checkpoints = [x for x in checkpoints if int(x[:4]) % self.checkpoint_retention_interval > 0 or int(x[5:-4]) > 0]
		checkpoints.sort()
		if len(checkpoints) > self.n_kept_checkpoints:
			n_extra = len(checkpoints) - self.n_kept_checkpoints 
			for i in range(n_extra):
				os.remove(pjoin(checkpoint_root,checkpoints[i]))
				self.tlog(f'Cleared old checkpoint: {checkpoints[i]}','note')

	def load_checkpoint(self,checkpoint_path,map_location=None):
		self.tlog(f'Checkpoint loaded from: {checkpoint_path}','note')
		checkpoint = torch.load(checkpoint_path,map_location=map_location)

		self.load_state_dict(checkpoint)
