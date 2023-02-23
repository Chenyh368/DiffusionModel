from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, opt, manager):
        self.opt = opt
        self.manager = manager
        self.logger = manager.get_logger()

    @staticmethod
    def modify_commandline_options(parser, mode):
        """Add new model-specific options, and rewrite default values for existing options."""
        return parser