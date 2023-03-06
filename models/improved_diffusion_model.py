import numpy as np

from .base_model import BaseModel
from networks.unet import UNet
import diffusion.gaussian_diffusion as gd
from utils.fp16_util import *
import blobfile as bf
from networks.nn import update_ema
import functools
from diffusion.spaced_diffusion import space_timesteps, SpacedDiffusion
from utils.resample import create_named_schedule_sampler
from utils.resample import LossAwareSampler, UniformSampler
import torch
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import copy
from utils.train_util import *
from utils import dist_util
from utils.meter import *

NUM_CLASSES = 1000

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

class ImprovedDiffusionModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, mode):
        return parser

    def __init__(self, opt, manager):
        BaseModel.__init__(self, opt, manager)
        self.local_rank = manager.get_rank()
        self.U_net = self._create_unet().to(self.local_rank)
        print_networks(self.U_net, manager.get_logger())
        self.diffusion = self._create_diffusion()

    def _create_unet(self):
        if self.opt.model.image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif self.opt.model.image_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif self.opt.model.image_size == 32:
            channel_mult = (1, 2, 2, 2)
        else:
            raise ValueError(f"unsupported image size: {self.opt.model.image_size}")

        attention_ds = []
        for res in self.opt.model.attention_resolutions.split(","):
            attention_ds.append(self.opt.model.image_size // int(res))

        return UNet(
            in_channels=3,
            model_channels=self.opt.model.num_channels,
            out_channels=(3 if not self.opt.model.learn_sigma else 6),
            num_res_blocks=self.opt.model.num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=self.opt.model.dropout,
            channel_mult=channel_mult,
            num_classes=(NUM_CLASSES if self.opt.model.class_cond else None),
            use_checkpoint=self.opt.model.use_checkpoint,
            num_heads=self.opt.model.num_heads,
            num_heads_upsample=self.opt.model.num_heads_upsample,
            use_scale_shift_norm=self.opt.model.use_scale_shift_norm,
        )


    def _create_diffusion(self):
        self.steps = self.opt.model.diffusion_steps
        self.learn_sigma = self.opt.model.learn_sigma
        self.sigma_small = self.opt.model.sigma_small
        self.noise_scheduler = self.opt.model.noise_scheduler
        self.use_kl = self.opt.model.use_kl
        self.predict_xstart = self.opt.model.predict_xstart
        self.rescale_timesteps = self.opt.model.rescale_timesteps
        self.rescale_learned_sigmas = self.opt.model.rescale_learned_sigmas
        self.timestep_respacing = self.opt.model.timestep_respacing

        self.betas = gd.get_named_beta_schedule(self.noise_scheduler, self.steps)
        if self.use_kl:
            self.loss_type = gd.LossType.RESCALED_KL
        elif self.rescale_learned_sigmas:
            self.loss_type = gd.LossType.RESCALED_MSE
        else:
            self.loss_type = gd.LossType.MSE

        if not self.timestep_respacing:
            self.timestep_respacing = [self.steps]

        return SpacedDiffusion(
            use_timesteps=space_timesteps(self.steps, self.timestep_respacing),
            betas=self.betas,
            model_mean_type=(
                gd.ModelMeanType.EPSILON if not self.predict_xstart else gd.ModelMeanType.START_X
            ),
            model_var_type=(
                (
                    gd.ModelVarType.FIXED_LARGE
                    if not self.sigma_small
                    else gd.ModelVarType.FIXED_SMALL
                )
                if not self.learn_sigma
                else gd.ModelVarType.LEARNED_RANGE
            ),
            loss_type=self.loss_type,
            rescale_timesteps=self.rescale_timesteps,
        )

    def eval(self):
        self.U_net.eval()


class Trainer():
    def __init__(self, opt, manager, improved_diffusion_model, data):
        self.opt = opt
        self.manager = manager

        self.model = improved_diffusion_model.U_net
        self.diffusion = improved_diffusion_model.diffusion
        self.data = data

        self.schedule_sampler = create_named_schedule_sampler(opt.train.schedule_sampler,
                                                              self.diffusion) or UniformSampler(self.diffusion)

        self.batch_size = opt.train.batch_size
        self.microbatch = opt.train.microbatch if opt.train.microbatch > 0 else opt.train.batch_size
        self.lr = opt.train.lr
        self.ema_rate = (
            [opt.train.ema_rate]
            if isinstance(opt.train.ema_rate, float)
            else [float(x) for x in opt.train.ema_rate.split(",")]
        )
        self.log_interval = opt.train.log_interval
        self.save_interval = opt.train.save_interval
        self.resume_checkpoint = opt.train.resume_checkpoint
        self.use_fp16 = opt.train.use_fp16
        self.fp16_scale_growth = opt.train.fp16_scale_growth
        self.weight_decay = opt.train.weight_decay
        self.lr_anneal_steps = opt.train.lr_anneal_steps
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * manager.get_n_gpus()
        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = torch.cuda.is_available()
        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        self.optimizer = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]
        self.logger = manager.get_logger()

        if torch.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[manager.get_rank()],
                output_device=dist_util.dev(manager.get_rank()),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                self.logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                self.logger.info(f"======> loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        self.manager, resume_checkpoint, map_location=dist_util.dev(self.manager.get_rank())
                    )
                )

        dist_util.sync_params(self.model.parameters())

        self.meter = {
            "grad_norm": AverageMeter(),
        }

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                self.logger.info(f"======> loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    self.manager, ema_checkpoint, map_location=dist_util.dev(self.manager.get_rank())
                )
                ema_params = self._state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            self.logger.info(f"======> loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                self.manager, opt_checkpoint, map_location=dist_util.dev(self.manager.get_rank())
            )
            self.optimizer.load_state_dict(state_dict)

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                # self.logger.dumpkvs()
                self.meter["grad_norm"].reset()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def forward_backward(self, batch, cond):
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev(self.manager.get_rank()))
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev(self.manager.get_rank()))
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev(self.manager.get_rank()))

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    # Not syn buffer ?
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            if self.manager.is_master():
                for key, values in losses.items():
                    self.manager.log_metric(key, values.mean().item(), self.step, split="Train")

            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

    def optimize_fp16(self):
        if any(not torch.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            self.logger.info(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.optimizer.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.optimizer.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        self.meter["grad_norm"].update(np.sqrt(sqsum))
        self.grad_norm_msg = f"| grad_norm: {self.meter['grad_norm'].get_value()} |"

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        self.logger.info(f"|step: {self.step + self.resume_step} | samples: {(self.step + self.resume_step + 1) * self.global_batch} "+ self.grad_norm_msg)
        if self.use_fp16:
            self.logger.info(f" |lg_loss_scale: {self.lg_loss_scale}| ")

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                self.logger.info(f"======> saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(self.manager), filename), "wb") as f:
                    torch.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(self.manager), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                torch.save(self.optimizer.state_dict(), f)

        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                self.model.parameters(), master_params
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params
