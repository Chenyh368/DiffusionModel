import blobfile as bf
import torch
import torch.distributed as dist
import io
import socket

def load_state_dict(manager, path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    # if manager.get_rank() == 0:
    #     with bf.BlobFile(path, "rb") as f:
    #         data = f.read()
    #     dist.broadcast(data, 0)
    # else:
    #     data = None
    # data = MPI.COMM_WORLD.bcast(data)
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return torch.load(io.BytesIO(data), **kwargs)

def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with torch.no_grad():
            dist.broadcast(p, 0)

def dev(rank):
    """
    Get the device to use for torch.distributed.
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{rank}")
    return torch.device("cpu")

def find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()