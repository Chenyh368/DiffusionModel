"""
Experiment manager, a helper aimed for deep learning code.
"""

import argparse
from datetime import datetime
import base64
import os
import json
import shutil
import logging
import torch
import sys
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def _generate_short_uid(length):
    assert length < 43
    return base64.urlsafe_b64encode(os.urandom(32)).decode()[:length]


# The main class
class ExperiMan(object):
    """
    E.g
    from experiman import manager

    def add_parser_argument(parser):
    parser.add_argument('--other_opt', type=str)


    parser = manager.get_basic_arg_parser()
    # add_parserAdd your own parser here
    add_parser_argument(parser)
    opt = parser.parse_args()
    manager.setup(opt)
    logger = manager.get_logger()
    device = 'cuda'

    ...

    What we should pass to parser by shell
        (Default)
    - gpu='0'
    - code_dir=None
    - data_dir=None
    - log_dir=None
    - exp_name=None
    - run_name=None
    - run_number='0'
    - seed=None
    - option_for_existing_dir=None
    - [other_opt=None]

    Then by calling : manager.setup(opt)
        the manager will make the dir.
    by calling : logger = manager.get_logger()
        logger can record the output
            both in the screen and the .log file.

    Main methods:
    - get_basic_arg_parser()
    - setup(opt, third_party_tools=None, no_log=False, setup_logging=True)
        - opt : initialized / must specify third_party_tools = ["tensorboard"] if you want
    - get_logging() /
        - After getting logger: logger.info() / logger.warning
    - get_run_dir(run_name=None, run_number=None)
    - get_run_dir(run_name=None, run_number=None)
        - In these two methods, you can use another dir produced by another run
    - get_opt() : return a namespace
    - get_tensorboard() : return the tensorboard writer (log_dir=manager._run_dir)


    Note:
        1. Just calling logger.info() instead of print.
        2. The dir path is like
    log_dir(opt.log_dir)/exp_dir(opt.exp_name)/run_root_dir(opt.run_name)/run_number/
    """


    def __init__(self, name):
        """
        Args :
            name : Exepriment name
        """
        self._name = name
        self._rank = 0
        self._logger = None
        self._opt = None
        self._uid = None
        self._exp_dir = None
        self._run_dir = None
        self._checkpoint_dir = None
        self._third_party_tools = []


    def _get_run_number_str(self, run_root_dir, opt_run_number):
        if opt_run_number in ('new', 'last'):
            if os.path.exists(run_root_dir):
                if os.path.exists(run_root_dir):
                    current_numbers = [int(x) for x in os.listdir(run_root_dir) if x.isdigit()]
                    if current_numbers:
                        if opt_run_number == 'new':
                            run_number = max(current_numbers) + 1
                        else:
                            run_number = max(current_numbers)
                    else:
                        if opt_run_number == 'new':
                            run_number = 0
                        else:
                            raise OSError(f"{run_root_dir} is empty!")
                else:       # run_root_dir does not exist
                    if opt_run_number == 'new':
                        run_number = 0
                    else:
                        raise OSError(f"{run_root_dir} does not exist!")
                run_number_str = str(run_number)
        else:       # manual number
            assert opt_run_number.isdigit(), "`run_number` is not a valid number"
            run_number_str = opt_run_number
        return run_number_str

    def _setup_dirs(self):
        """
        Setting self._run_dir / self._checkpoint_dir
        """
        opt = self._opt
        # exp_dir: direcotry for the experiment
        exp_dir = os.path.join(opt.log_dir, opt.exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        self._exp_dir = exp_dir
        # run_dir: directory for the run
        run_root_dir = os.path.join(exp_dir, opt.run_name)
        opt.run_number = self._get_run_number_str(run_root_dir, opt.run_number)
        print("run_number : " + opt.run_number)
        run_dir = os.path.join(run_root_dir, opt.run_number)
        if os.path.exists(run_dir):
            if opt.option_for_existing_dir:
                op = opt.option_for_existing_dir
            else:
                print(f"Directory {run_dir} exists, please choose an option:")
                op = input("b (backup) / k (keep) / d (delete) / n (new) / q (quit): ")
            if op == 'b':
                with open(os.path.join(run_dir, 'args.json'), 'r') as fp:
                    old_opt = json.load(fp)
                d_backup = run_dir + f"-backup-({old_opt['uid']})"
                shutil.move(run_dir, d_backup)
                print(f"Old files backuped to {d_backup}.")
            elif op == 'k':
                print("Old files kept unchanged.")
                raise NotImplementedError
            elif op == 'd':
                shutil.rmtree(run_dir, ignore_errors=True)
                print("Old files deleted.")
            elif op == 'n':
                opt.run_number = self._get_run_number_str(run_root_dir, 'new')
                print(f"New run number: {opt.run_number}")
                run_dir = os.path.join(run_root_dir, opt.run_number)
            else:
                raise OSError("Quit without changes.")
        os.makedirs(run_dir, exist_ok=True)
        print(f"======> Directory for this run: {run_dir}")
        self._run_dir = run_dir
        # checkpoint_dir: directory for the checkpoints of the run
        checkpoint_dir = self.get_checkpoint_dir()
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._checkpoint_dir = checkpoint_dir
        # image_dir : directory for the images of the run
        image_dir = self.get_image_dir()
        os.makedirs(image_dir, exist_ok=True)
        self._image_dir = image_dir


    def _setup_uid(self):
        self._uid = '-'.join([datetime.now().strftime('%y%m%d-%H%M%S'),
                              _generate_short_uid(length=6)])
        self._opt.uid = self._uid
        print(f"======> UID of this run: {self._uid}")


    def _setup_logger(self, level= logging.DEBUG):
        """
        Setting self._logger
        """
        self._logger = logging.getLogger(name=self._name)
        self._logger.propagate = False
        self._logger.setLevel(level)
        # Stdout handler
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        self._logger.addHandler(ch)
        # Log file handler
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        filename = "log.log"
        path = os.path.join(self._run_dir, filename)
        fh = logging.FileHandler(path, encoding='utf-8')
        fh.setLevel(level)
        fh.setFormatter(formatter)
        self._logger.addHandler(fh)

    def _backup_code(self):
        dst = os.path.join(self._run_dir, "code")
        shutil.copytree(self._opt.code_dir, dst,
                        ignore=shutil.ignore_patterns('__pycache__', '.git', 'pymp-*'))

    def _setup_seed(self):
        np.random.seed(self._opt.seed)
        torch.manual_seed(self._opt.seed)
        torch.cuda.manual_seed_all(self._opt.seed)

    def _setup_torch(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self._opt.gpu
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    def _setup_third_party_tools(self):
        if 'tensorboard' in self._third_party_tools:
            self._tensorboard_writer = SummaryWriter(
                log_dir=self._run_dir,
                max_queue=100,
                flush_secs=60,
                purge_step=0,
            )

    def _export_arguments(self):
        """
        Save self._opt
        """
        opt = self._opt
        self._logger.info(f"Opts: {opt}")
        with open(os.path.join(self._run_dir, 'argv.txt'), 'w') as f:
            print(sys.argv, file=f)
        with open(os.path.join(self._run_dir, 'args.json'), 'w') as f:
            json.dump(vars(opt), fp=f)

    def _log_hparams(self):
        pass

    # Setting in every project
    def get_basic_arg_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--gpu', type=str, default="0", help="gpu device id")
        parser.add_argument('--code_dir', type=str, default="/home/yhchenmath/Code/DiffusionModel", help="code dir (for backup)")
        parser.add_argument('--data_dir', type=str, default="/home/yhchenmath/Dataset/CV", help="data dir")
        # ======> Setting run_dir
        parser.add_argument('--log_dir', type=str, default="/home/yhchenmath/Log", help="root dir for logging")
        parser.add_argument('--exp_name', type=str, default="diffusion", help="name of the experiment")
        parser.add_argument('--run_name', type=str, default="raw", help="name of this run")
        parser.add_argument('--run_number', type=str, default='0',
                            help="Number of this run. Choices: {new, last, MANUAL_NUMBER}")
        parser.add_argument('--seed', type=int, default="0", help="random seed")
        parser.add_argument('--option_for_existing_dir', '-O', type=str,
                            help="Specify the option for existing run_dir:" +
                                 " b (backup) / k (keep) / d (delete) / n (new) / q (quit)")
        return parser

    def setup(self, opt, third_party_tools=None, no_log=False, setup_logging=True):
        self._opt = opt
        if third_party_tools:
            self._third_party_tools = third_party_tools
        if not no_log:
            self._setup_uid()
            self._setup_dirs()
        else:
            self._exp_dir = os.path.join(opt.log_dir, opt.exp_name)
        self._setup_torch()
        if opt.seed is not None:
            self._setup_seed()
        if not no_log and setup_logging:
            self.setup_logging()
        self._n_gpus = torch.cuda.device_count()

    def setup_logging(self):
        self._setup_logger()
        self._export_arguments()
        if self._opt.code_dir is not None:
            self._backup_code()
        else:
            self._logger.warning(
                "Argument `code_dir` unspecified, code will not be backuped!")
        self._setup_third_party_tools()
        self._log_hparams()

    def get_opt(self):
        return self._opt

    def get_run_dir(self, run_name=None, run_number=None):
        """
        If run_name is None, return the directory for this run.
        Otherwise, return the directory of the specified run.
        (run_number defaults to 0)
        """
        if run_name is None:
            run_dir = self._run_dir
        else:
            if run_number is None:
                run_number = '0'
            run_dir = os.path.join(self._exp_dir, run_name, str(run_number))
        return run_dir

    def get_checkpoint_dir(self, run_name=None, run_number=None):
        """
        If run_name is None, return the checkpoint directory for this run.
        Otherwise, return the checkpoint directory of the specified run.
        (run_number defaults to 0)
        """
        run_dir = self.get_run_dir(run_name, run_number)
        return os.path.join(run_dir, 'checkpoints')

    def get_image_dir(self, run_name=None, run_number=None):
        run_dir = self.get_run_dir(run_name, run_number)
        return os.path.join(run_dir, 'images')

    def get_tensorboard(self):
        if 'tensorboard' in self._third_party_tools:
            return self._tensorboard_writer
        else:
            raise AttributeError

    def get_logger(self, name=None):
        if name is None:
            logger = self._logger
            # logger = logging.getLogger(name=self._name)
        else:
            logger_name = self._logger.name + '.' + name
            logger = logging.getLogger(name=logger_name)
        return logger

    def log_metric(self, name, value, global_step, split=None):
        if 'tensorboard' in self._third_party_tools:
            writer = self._tensorboard_writer
            if split is None:
                scaler_name = name
            else:
                scaler_name = '/'.join((split, name))
            writer.add_scalar(scaler_name, value, global_step)
        else:
            raise NotImplementedError

    def log_image(self, name, image, global_step, split=None):
        if 'tensorboard' in self._third_party_tools:
            writer = self._tensorboard_writer
            if split is None:
                image_name = name
            else:
                image_name = '/'.join((split, name))
            writer.add_images(image_name, image, global_step)
        else:
            raise NotImplementedError

    # =============== DDP Part ================
    def set_rank(self, rank):
        self._rank = rank

    def is_master(self):
        return self._rank == 0

    def get_rank(self):
        return self._rank

    def get_n_gpus(self):
        return self._n_gpus
