import argparse
import models
import data
import json

class Options():
    """This class defines options used during both training and test time.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """
    def __init__(self, parser):
        self.parser = parser

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # ============== JSON =====================
        parser.add_argument('--load_json', action='store_false')
        parser.add_argument('--json_path', type=str, default='./config/idm_cifar10.json')
        # ============== Dataset =====================
        parser.add_argument('--dataset_mode', type=str, default='image',
                            help='chooses which datasets are loaded. [image |]')
        parser.add_argument('--dataset_dir', type=str, default='cifar10/')
        # ============== Model =====================
        parser.add_argument('--model_name', type=str, default='improved_diffusion',
                            help='chooses which model to use. [improved_gaussian_diffusion |]')
        # ============== Additional =====================
        parser.add_argument('--mode', default="train", type=str, choices=['train', 'eval'])

        return parser

    def gather_options(self):
        """
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        parser = self.initialize(self.parser)
        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model_name
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, opt.mode)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, opt.mode)

        # save and return the parser
        self.parser = parser

        # Get json config
        opt, _ = parser.parse_known_args()
        if opt.load_json:
            with open(opt.json_path, "r") as f:
                js = f.read()
            config = json.loads(js)
            config.update(vars(parser.parse_known_args()[0]))
            opt = argparse.Namespace(**config)
        return opt

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        self.opt = opt
        return self.opt


class HyperParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HyperParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()