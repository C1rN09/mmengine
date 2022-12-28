# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
from abc import ABC
from typing import Any, Dict

import torch
import torch.nn as nn

from mmengine.config import Config
from mmengine.logging import print_log
from mmengine.optim import OptimWrapper, OptimWrapperDict
from mmengine.runner import Runner


def rgetattr(obj, attr):
    pass


def rsetattr(obj, attr, val):
    pass


class Strategy(ABC):

    components: Dict[str, str] = {}

    def __init__(self, full_cfg: Config):
        # deepcopy to prevent the original one from being modified
        self.cfg = copy.deepcopy(full_cfg)

        if 'optim_wrapper' in self.cfg:
            optimizer_cfg = self.cfg.optim_wrapper.pop('optimizer')
            if optimizer_cfg and self.cfg.get('optimizer'):
                print_log(
                    'an optimizer has been defined in config_file, so another'
                    'one defined in optim_wrapper will be ignored',
                    level=logging.WARN)
            self.cfg.setdefault('optimizer', optimizer_cfg)

    def setup(self, runner: Runner, entry: str):
        self.components.setdefault('model', 'model')
        if entry == 'train':
            self.components.setdefault('optim_wrapper', 'optim_wrapper')
            self.components.setdefault('param_schedulers', 'param_schedulers')
        self._pre_setup(runner)
        self._setup_model_and_optimizer(runner)
        self._setup_param_schedulers(runner)

    def load_checkpoint(self):
        pass

    def save_checkpoint(self):
        pass

    @property
    def resume_before_wrap_model(self):
        return True

    def _pre_setup(self, runner: Runner):
        self.model = runner.model
        self.optim_wrapper = runner.optim_wrapper
        self.param_schedulers = runner.param_schedulers

    def _setup_model_and_optimizer(self, runner: Runner):
        self._setup_model(runner)
        self._setup_optimizer(runner)

    def _setup_model(self, runner: Runner):
        pass

    def _setup_optimizer(self, runner: Runner):
        pass

    def _setup_param_schedulers(self, runner: Runner):
        pass

    @staticmethod
    def is_model_built(model: Any):
        return isinstance(model, nn.Module)

    @staticmethod
    def is_model_setup(model: Any):
        return isinstance(model, nn.Module)

    @staticmethod
    def is_optimizer_built(optimizer: Any):
        return isinstance(optimizer, torch.optim.Optimizer)

    @staticmethod
    def is_optim_wrapper_built(optim_wrapper: Any):
        return isinstance(optim_wrapper, (OptimWrapper, OptimWrapperDict))

    @staticmethod
    def is_param_schedulers_built(param_schedulers: Any):
        pass
