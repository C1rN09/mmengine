# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List

import torch.nn as nn

from mmengine.registry import (OPTIM_WRAPPER_CONSTRUCTORS, OPTIM_WRAPPERS,
                               OPTIMIZERS)
from .default_constructor import DefaultOptimWrapperConstructor


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class ColossalOptimConstructor(DefaultOptimWrapperConstructor):

    def __call__(self, model: nn.Module) -> Any:
        from mmengine.model.wrappers import ColossalZeroDDP
        assert isinstance(model, ColossalZeroDDP)
        optim_wrapper_cfg = self.optim_wrapper_cfg.copy()
        assert optim_wrapper_cfg.get('type') == 'ColossalZeroOptimizer'
        optimizer_cfg = self.optimizer_cfg.copy()
        # if no paramwise option is specified, just use the global setting
        if not self.paramwise_cfg:
            optimizer_cfg['params'] = model.parameters()
            optimizer = OPTIMIZERS.build(optimizer_cfg)
        else:
            # set param-wise lr and weight decay recursively
            params: List = []
            self.add_params(params, model)
            optimizer_cfg['params'] = params
            optimizer = OPTIMIZERS.build(optimizer_cfg)
        # ColossalZeroOptimizer requires ZeroDDP `model` as an argument.
        optim_wrapper = OPTIM_WRAPPERS.build(
            optim_wrapper_cfg,
            default_args=dict(optimizer=optimizer, model=model))
        return optim_wrapper
