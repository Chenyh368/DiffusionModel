from .base_model import BaseModel
from networks.autoencoder_and_latent_dm import Encoder, Decoder
from utils.ldm_losses import LPIPSWithDiscriminator
from utils.ldm_losses.distributions import DiagonalGaussianDistribution
import torch
from utils import dist_util
class AutoencoderKLModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, mode):
        return parser

    def __init__(self, opt, manager):
        BaseModel.__init__(self, opt, manager)
        self.local_rank = manager.get_rank()
        self.image_key = opt.addition.image_key
        self.encoder = Encoder(**opt.model.ddconfig)
        self.decoder = Decoder(**opt.model.ddconfig)
        if opt.model.params.lossconfig.target == "LPIPSWithDiscriminator":
            self.loss = LPIPSWithDiscriminator(**opt.model.params.lossconfig.params)
        else:
            raise NotImplementedError
        assert opt.model.ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * opt.model.ddconfig["z_channels"], 2 * opt.model.params.embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(opt.model.params.embed_dim, opt.model.ddconfig["z_channels"], 1)
        # TODO: Prepare DDP
        self.embed_dim = opt.model.params.embed_dim
        # TODO: Add colorize_nlabels buffer
        if opt.addition.colorize_nlabels is not None:
            assert type(opt.addition.colorize_nlabels) == int
            raise NotImplementedError
        # TODO: ADD Loading networks
        if opt.addition.ckpt_path is not None:
            raise NotImplementedError

        # Change in Step
        self.global_step = 0
        self.curr_lr = opt.model.base_learning_rate

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k].to(dist_util.dev(self.manager.get_rank()))
        if len(x.shape) == 3:
            x = x[..., None]
        # Chennel is not in shpape[1]
        # TODO: Check dataset x shape
        if x.shape[1] >= 3:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self.forward(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            if self.manager.is_master():
                for key, values in log_dict_ae.items():
                    self.manager.log_metric(key, values, self.global_step, split="Train")
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            if self.manager.is_master():
                for key, values in log_dict_disc.items():
                    self.manager.log_metric(key, values, self.global_step, split="Train")
            return discloss

    def validation_step(self, batch):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self.forward(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        if self.manager.is_master():
            for key, values in log_dict_ae.items():
                self.manager.log_metric(key, values, self.global_step, split="Val")
        if self.manager.is_master():
            for key, values in log_dict_disc.items():
                self.manager.log_metric(key, values, self.global_step, split="Val")

    def configure_optimizers(self):
        lr = self.curr_lr
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch,only_inputs=False):
        x = self.get_input(batch, self.image_key)
        if not only_inputs:
            xrec, posterior = self.forward(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                raise NotImplementedError
                # x = self.to_rgb(x)
                # xrec = self.to_rgb(xrec)

            self.manager.log_image("samples", (self.decode(torch.randn_like(posterior.sample())) + 1) / 2,
                                   self.global_step)
            self.manager.log_image("reconstructions", (xrec + 1) / 2,
                                   self.global_step)
        self.manager.log_image("inputs", (x + 1) / 2, self.global_step)

