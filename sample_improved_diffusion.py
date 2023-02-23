"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import numpy as np
from options import Options, HyperParams
from utils.experiman import ExperiMan
import os
import torch
import torch.multiprocessing as mp
import logging
from utils import dist_util
import torch.distributed as dist
from models import create_model
from models.improved_diffusion_model import Trainer
from data import create_dataset

NUM_CLASSES = 1000

def main(opt, manager):
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."
    n_gpus = torch.cuda.device_count()
    # TODO: Run train.py simultaneously with different port
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '80000'
    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, opt, manager))

def run(rank, n_gpus, opt, manager):
    torch.cuda.set_device(rank)
    # Set up seed in every process
    manager.set_rank(rank)
    manager._setup_seed()
    manager._setup_logger(logging.DEBUG if rank in [-1, 0] else logging.WARN)
    if manager.is_master():
        manager._third_party_tools = ('tensorboard',)
        # Set up tensorboard in master (rank == 0)
        manager._setup_third_party_tools()

    logger = manager.get_logger()
    logger.info(f"======> N gpus / World size: {n_gpus}")
    logger.info(f"======> Log in Process: {rank}")

    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)

    # Model
    diffusion_model = create_model(opt, manager)
    diffusion_model.eval()

    all_images = []
    all_labels = []
    logger.info("=============== Sampling ===============")

    log_steps = 0
    while len(all_images) * opt.test.batch_size < opt.test.num_samples:
        model_kwargs = {}
        if opt.model.class_cond:
            classes = torch.randint(
                low=0, high=NUM_CLASSES, size=(opt.test.batch_size,), device=dist_util.dev(manager.get_rank())
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion_model.diffusion.p_sample_loop if not opt.test.use_ddim else diffusion_model.diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            diffusion_model.U_net,
            (opt.test.batch_size, 3, opt.model.image_size, opt.model.image_size),
            clip_denoised=opt.test.clip_denoised,
            model_kwargs=model_kwargs,
        )
        if manager.get_rank() == 0 and log_steps <= 10:
            manager.log_image(opt.test.model_path.split('/')[-1], sample, log_steps)
            log_steps += 1

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [torch.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if opt.model.class_cond:
            gathered_labels = [
                torch.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.info(f"======> created {len(all_images) * opt.test.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: opt.test.num_samples]
    if opt.model.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: opt.model.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(manager.get_image_dir(), f"samples_{shape_str}.npz")
        logger.info(f"======> saving to {out_path}")
        if opt.model.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.info("======> sampling complete")

if __name__ == "__main__":
    manager = ExperiMan(name='default')
    parser = manager.get_basic_arg_parser()
    opt = Options(parser).parse()  # get training options

    manager.setup(opt)
    opt = HyperParams(**vars(opt))
    main(opt, manager)

