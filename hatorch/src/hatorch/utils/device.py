import torch
import os
from .logger import logger
from torch.distributed import init_process_group, destroy_process_group

def get_device(print_device: bool) -> torch.device:
    if print_device:
        if torch.cuda.is_available():
            logger.info(f"{torch.cuda.get_device_name()} with {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB memory")
        else:
            logger.info("No Cuda enabled device found, falling back to CPU")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def ddp_setup(rank: int, world_size: int, master_addr: str, master_port: str) -> None:
   """
   Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
        master_addr: Address of the master node (e.g., "localhost" or IP address)
        master_port: Port number for communication (e.g., "12355")
   """
   os.environ["MASTER_ADDR"] = master_addr
   os.environ["MASTER_PORT"] = master_port
   torch.cuda.set_device(rank)
   init_process_group(backend="nccl", rank=rank, world_size=world_size) # "nccl" for nvidia gpus
