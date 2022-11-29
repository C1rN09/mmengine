# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, Union

import torch

from mmengine.optim import ColossalZeroOptimizer
from mmengine.registry import MODEL_WRAPPERS

try:
    from colossalai.nn.parallel import ZeroDDP
except ImportError:
    ZeroDDP = object


@MODEL_WRAPPERS.register_module()
class ColossalZeroDDP(ZeroDDP):
    # TODO(C1rN09): Add docstring
    def __init__(self,
                 module,
                 placement_policy: str = 'cuda',
                 force_outputs_fp32: bool = False,
                 pin_memory: bool = False):
        from colossalai.gemini import GeminiManager
        from colossalai.gemini.chunk import init_chunk_manager
        from colossalai.utils import get_current_device
        from colossalai.utils.model.colo_init_context import ColoInitContext

        # Some modules in OpenMMLab are not direct subclasses of nn.Module, so
        # the context manager will not take effect, i.e turn module's
        # parameters to `ColoParameter`. We manually call its
        # `_post_init_method` on the built model, since it applies recursively
        with ColoInitContext(device=get_current_device()) as ctx:
            for m in module.modules():
                ctx._post_init_method(m)

        # TODO(C1rN09): Make chunk params configurable
        chunk_manager = init_chunk_manager(
            model=module, init_device=get_current_device(), search_range_mb=32)
        # TODO(C1rN09): 'auto' policy may require other arguments
        gemini_manager = GeminiManager(placement_policy, chunk_manager)
        super().__init__(
            module,
            gemini_manager,
            force_outputs_fp32=force_outputs_fp32,
            pin_memory=pin_memory)

    def train_step(
            self, data: Union[dict, tuple, list],
            optim_wrapper: ColossalZeroOptimizer) -> Dict[str, torch.Tensor]:
        """Interface for model forward, backward and parameters updating during
        training process.

        :meth:`train_step` will perform the following steps in order:

        - If :attr:`module` defines the preprocess method,
          call ``module.preprocess`` to pre-processing data.
        - Call ``module.forward(**data)`` and get losses.
        - Parse losses.
        - Call ``optim_wrapper.optimizer_step`` to update parameters.
        - Return log messages of losses.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (ColossalZeroOptimizer): Must be a
                ColossalZeroOptimizer

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        # Enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            data = self.module.data_preprocessor(data, training=True)
            losses = self._run_forward(data, mode='loss')
        parsed_loss, log_vars = self.module.parse_losses(losses)
        optim_wrapper.update_params(parsed_loss)
        return log_vars

    def val_step(self, data: Union[dict, tuple, list]) -> list:
        """Gets the prediction of module during validation process.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        return self.module.val_step(data)

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        """Gets the predictions of module during testing process.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        return self.module.test_step(data)

    def _run_forward(self, data: Union[dict, tuple, list], mode: str) -> Any:
        """Unpacks data for :meth:`forward`

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            mode (str): Mode of forward.

        Returns:
            dict or list: Results of training or testing mode.
        """
        if isinstance(data, dict):
            results = self(**data, mode=mode)
        elif isinstance(data, (list, tuple)):
            results = self(*data, mode=mode)
        else:
            raise TypeError('Output of `data_preprocessor` should be '
                            f'list, tuple or dict, but got {type(data)}')
        return results
