# For DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

# Set up the process group for Distributed Data Parallel
def setup(rank, world_size):

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

# Define distributed data loader
# May not be necessary
def prepare(rank, world_size, batch_size, dataset):
    sampler = DistributedSampler(dataset, num_replicas=world_size, 
                                rank=rank, shuffle=False, drop_last=True)
    # dataloader = DataLoader(dataset, )
    return 

# Wrap the model with DDP
def wrap(model, data, rank, world_size):
    setup(rank, world_size)
    # Move data to designated device ID
    data = data.to(rank)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank
                find_unused_parameters=True)
    return 
