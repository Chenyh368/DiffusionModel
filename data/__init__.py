import importlib
from data.base_dataset import BaseDataset
import torch
import torch.distributed as dist

def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset

def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options

def create_dataset(opt, manager):
    dataset_class = find_dataset_using_name(opt.dataset_mode)
    dataset = dataset_class(opt, manager)
    logger = manager.get_logger()
    logger.info("======> dataset [%s] was created" % type(dataset).__name__)
    dist.barrier()
    logger.warn(f"======> length of dataset in rank {manager.get_rank()}: {len(dataset)}")
    return dataset

def create_dataloader(dataset, opt):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.train.batch_size,
        shuffle=not opt.train.deterministic if opt.mode == "train" else not opt.eval.deterministic,
        num_workers=1,
        drop_last=True)

    return dataloader


