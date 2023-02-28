import sys
sys.path.append("./")

import os
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
from data import create_dataset, create_dataloader_generator



def main(opt, manager):
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."
    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(dist_util.find_free_port())
    print(f"======> Using Address: {os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')}")
    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, opt, manager))

def run(rank, n_gpus, opt, manager):
    torch.cuda.set_device(rank)
    # Set up seed in every process
    manager.set_rank(rank)
    manager._setup_seed()
    manager._setup_logger(logging.DEBUG if rank in [-1, 0] else logging.WARN)


    logger = manager.get_logger()
    logger.info(f"======> N gpus / World size: {n_gpus}")
    logger.info(f"======> Log in Process: {rank}")

    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)

    # Model
    diffusion_model = create_model(opt, manager)
    logger.info(f"======> Load model from {opt.sample.model_path}")
    diffusion_model.U_net.load_state_dict(
        dist_util.load_state_dict(manager, opt.sample.model_path, map_location="cpu")
    )
    diffusion_model.eval()

    # Data
    dataset = create_dataset(opt, manager)
    data = create_dataloader_generator(dataset, opt)

    if manager.is_master():
        manager._third_party_tools = ('tensorboard',)
        # Set up tensorboard in master (rank == 0)
        manager._setup_third_party_tools()
    logger.info("=============== Eval ===============")
    run_bpd_evaluation(manager, diffusion_model.U_net, diffusion_model.diffusion, data, opt.eval.num_samples, opt.eval.clip_denoised)

def run_bpd_evaluation(manager, model, diffusion, data, num_samples, clip_denoised):
    logger = manager.get_logger()
    all_bpd = []
    all_metrics = {"vb": [], "mse": [], "xstart_mse": []}
    num_complete = 0
    while num_complete < num_samples:
        batch, model_kwargs = next(data)
        batch = batch.to(dist_util.dev(manager.get_rank()))
        model_kwargs = {k: v.to(dist_util.dev(manager.get_rank())) for k, v in model_kwargs.items()}
        minibatch_metrics = diffusion.calc_bpd_loop(
            model, batch, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )

        for key, term_list in all_metrics.items():
            terms = minibatch_metrics[key].mean(dim=0) / dist.get_world_size()
            dist.all_reduce(terms)
            term_list.append(terms.detach().cpu().numpy())

        total_bpd = minibatch_metrics["total_bpd"]
        total_bpd = total_bpd.mean() / dist.get_world_size()
        dist.all_reduce(total_bpd)
        all_bpd.append(total_bpd.item())
        num_complete += dist.get_world_size() * batch.shape[0]

        logger.info(f"======> done {num_complete} samples: bpd={np.mean(all_bpd)}")

    if dist.get_rank() == 0:
        for name, terms in all_metrics.items():
            out_path = os.path.join(manager.get_run_dir(), f"{name}_terms.npz")
            logger.info(f"======> saving {name} terms to {out_path}")
            np.savez(out_path, np.mean(np.stack(terms), axis=0))

    dist.barrier()
    logger.info("======> evaluation complete")


if __name__ == "__main__":
    manager = ExperiMan(name='default')
    parser = manager.get_basic_arg_parser()
    opt = Options(parser).parse()  # get training options

    manager.setup(opt)
    opt = HyperParams(**vars(opt))
    main(opt, manager)

