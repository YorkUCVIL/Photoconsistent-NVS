import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

class DistributedSaveableSampler(DistributedSampler):
    """Just like with the case with
       torch.utils.data.distributed.DistributedSampler you *MUST* call
       self.set_epoch(epoch:int) to ensure all replicates use the same
       random shuffling within each epoch if shuffle is True
    """

    def __init__(self, *args, force_synchronization=False, **kwargs):
        """
        Arguments:
            force_synchronization (boolean, optional): If it's true then after
                each yield we will force a synchronization so each process'
                _curr_idx will be the same, this guarantees correctness of the
                save in case there is no synchronization during training, but
                comes at a performance cost
            For the rest of the arguments please see:
                https://pytorch.org/docs/1.7.1/data.html?highlight=distributed%20sampler#torch.utils.data.distributed.DistributedSampler
        """
        super().__init__(*args, **kwargs)
        self._curr_idx = 0
        self.force_synchronization = force_synchronization

    def __iter__(self):
        """Logic modified from
            https://pytorch.org/docs/1.7.1/_modules/torch/utils/data/distributed.html#DistributedSampler
        """
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset),
                                     generator=g).tolist()  # type: ignore
        else:
            indices = list(range(len(self.dataset)))  # type: ignore

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            indices += indices[:(self.total_size - len(indices))]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        while self._curr_idx + self.rank < self.total_size:
            to_yield = self.rank + self._curr_idx

            # we need to increment this before the yield because
            # there might be a save or preemption while we are yielding
            # so we must increment it before to save the right index
            self._curr_idx += self.num_replicas

            yield indices[to_yield]

            if self.force_synchronization:
                dist.barrier()
        self._curr_idx = 0

    def state_dict(self, dataloader_iter=None):
        prefetched_num = 0
        # in the case of multiworker dataloader, the helper worker could be
        # pre-fetching the data that is not consumed by the main dataloader.
        # we need to subtract the unconsumed part .
        if dataloader_iter is not None:
            if dataloader_iter._num_workers > 0:
                batch_size = dataloader_iter._index_sampler.batch_size
                prefetched_num = (
                    (dataloader_iter._send_idx - dataloader_iter._rcvd_idx) *
                    batch_size)

        return {
            "index": self._curr_idx - (prefetched_num * self.num_replicas),
            "epoch": self.epoch,
        }

    def load_state_dict(self, state_dict):
        self._curr_idx = state_dict["index"]
        self.epoch = state_dict["epoch"]
