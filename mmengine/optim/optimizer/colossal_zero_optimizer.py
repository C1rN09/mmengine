# Copyright (c) OpenMMLab. All rights reserved.

import logging
from contextlib import contextmanager
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer

from mmengine.logging import MessageHub, print_log
from mmengine.registry import OPTIM_WRAPPERS
from mmengine.utils.dl_utils import has_batch_norm

try:
    from colossalai.zero import ZeroOptimizer as _ZeroOptimizer
except ImportError:
    _ZeroOptimizer = object


@OPTIM_WRAPPERS.register_module()
class ColossalZeroOptimizer(_ZeroOptimizer):

    def __init__(self,
                 optimizer: Optimizer,
                 model: nn.Module = None,
                 accumulative_counts: int = 1,
                 clip_grad: Optional[dict] = None,
                 **kwargs):
        super().__init__(optimizer, model, **kwargs)
        assert accumulative_counts > 0, (
            '_accumulative_counts at least greater than or equal to 1')
        self._accumulative_counts = accumulative_counts

        assert isinstance(optimizer, Optimizer), (
            'optimizer must be a `torch.optim.Optimizer` instance, but got '
            f'{type(optimizer)}')
        self.optimizer = optimizer

        # TODO(C1rN09): finish clip_grad logic
        if clip_grad is not None:
            # clip_grad_kwargs should not be non-empty dict.
            assert isinstance(clip_grad, dict) and clip_grad, (
                'If `clip_grad` is not None, it should be a `dict` '
                'which is the arguments of `ZeroOptimizer.clip_grad_norm`')
            clip_type = clip_grad.pop('type', 'norm')
            assert clip_type == 'norm', (
                'Only `clip_grad_norm is supported for `ZeroOptimizer`')
            self.clip_func = self.clip_grad_norm
            assert clip_grad, ('`clip_grad` should contain other arguments '
                               'besides `type`. The arguments should match '
                               'with the `ZeroOptimizer.clip_grad_norm`')
        self.clip_grad_kwargs = clip_grad
        self.message_hub = MessageHub.get_current_instance()
        self._inner_count = 0
        self._max_counts = -1
        self._remainder_counts = -1

    def update_params(self, loss: torch.Tensor) -> None:
        """Update parameters in :attr:`optimizer`.

        Args:
            loss (torch.Tensor): A tensor for back propagation.
        """
        self.backward(loss)
        # Update parameters only if `self._inner_count` is divisible by
        # `self._accumulative_counts` or `self._inner_count` equals to
        # `self._max_counts`
        if self.should_update():
            self.step()
            self.zero_grad()

    @property
    def param_groups(self) -> List[dict]:
        """A wrapper of ``Optimizer.param_groups``.

        Make OptimizeWrapper compatible with :class:`_ParamScheduler`.

        Returns:
             dict: the ``param_groups`` of :attr:`optimizer`.
        """
        return self.optimizer.param_groups

    @property
    def defaults(self) -> dict:
        """A wrapper of ``Optimizer.defaults``.

        Make OptimizeWrapper compatible with :class:`_ParamScheduler`.

        Returns:
             dict: the ``param_groups`` of :attr:`optimizer`.
        """
        return self.optimizer.defaults

    def get_lr(self) -> Dict[str, List[float]]:
        """Get the learning rate of the optimizer.

        Provide unified interface to get learning rate of optimizer.

        Returns:
            Dict[str, List[float]]: Learning rate of the optimizer.
        """
        lr = [group['lr'] for group in self.param_groups]
        return dict(lr=lr)

    def get_momentum(self) -> Dict[str, List[float]]:
        """Get the momentum of the optimizer.

        Provide unified interface to get momentum of optimizer.

        Returns:
            Dict[str, List[float]]: Momentum of the optimizer.
        """
        momentum = []
        for group in self.param_groups:
            # Get momentum of SGD.
            if 'momentum' in group.keys():
                momentum.append(group['momentum'])
            # Get momentum of Adam.
            elif 'betas' in group.keys():
                momentum.append(group['betas'][0])
            else:
                momentum.append(0)
        return dict(momentum=momentum)

    @contextmanager
    def optim_context(self, model: nn.Module):
        """A Context for gradient accumulation and automatic mix precision
        training.

        If subclasses need to enable the context for mix precision training,
        e.g., ``:class:`AmpOptimWrapper``,  the corresponding context should be
        enabled in `optim_context`. Since ``OptimWrapper`` uses default fp32
        training, ``optim_context`` will only enable the context for
        blocking the unnecessary gradient synchronization during gradient
        accumulation

        If model is an instance with ``no_sync`` method (which means
        blocking the gradient synchronization) and
        ``self._accumulative_counts != 1``. The model will not automatically
        synchronize gradients if ``cur_iter`` is divisible by
        ``self._accumulative_counts``. Otherwise, this method will enable an
        empty context.

        Args:
            model (nn.Module): The training model.
        """
        # During gradient accumulation process, the gradient synchronize
        # should only happen before updating parameters.
        if not self.should_sync() and hasattr(model, 'no_sync'):
            with model.no_sync():
                yield
        else:
            yield

    def initialize_count_status(self, model: nn.Module, init_counts: int,
                                max_counts: int) -> None:
        """Initialize gradient accumulation related attributes.

        ``OptimWrapper`` can be used without calling
        ``initialize_iter_status``. However, Consider the case of  ``len(
        dataloader) == 10``, and the ``accumulative_iter == 3``. Since 10 is
        not divisible by 3, the last iteration will not trigger
        ``optimizer.step()``, resulting in one less parameter updating.

        Args:
            model (nn.Module): Training model
            init_counts (int): The initial value of the inner count.
            max_counts (int): The maximum value of the inner count.
        """
        self._inner_count = init_counts
        self._max_counts = max_counts
        if self._inner_count % self._accumulative_counts != 0:
            print_log(
                'Resumed iteration number is not divisible by '
                '`_accumulative_counts` in `GradientCumulativeOptimizerHook`, '
                'which means the gradient of some iterations is lost and the '
                'result may be influenced slightly.',
                logger='current',
                level=logging.WARNING)

        if has_batch_norm(model) and self._accumulative_counts > 1:
            print_log(
                'Gradient accumulative may slightly decrease '
                'performance because the model has BatchNorm layers.',
                logger='current',
                level=logging.WARNING)
        # Remainder of `_max_counts` divided by `_accumulative_counts`
        self._remainder_counts = self._max_counts % self._accumulative_counts

    def should_update(self) -> bool:
        """Decide whether the parameters should be updated at the current
        iteration.

        Called by :meth:`update_params` and check whether the optimizer
        wrapper should update parameters at current iteration.

        Returns:
            bool: Whether to update parameters.
        """
        return (self._inner_count % self._accumulative_counts == 0
                or self._inner_count == self._max_counts)

    def should_sync(self) -> bool:
        """Decide whether the automatic gradient synchronization should be
        allowed at the current iteration.

        It takes effect when gradient accumulation is used to skip
        synchronization at the iterations where the parameter is not updated.

        Since ``should_sync`` is called by :meth:`optim_context`, and it is
        called before :meth:`backward` which means ``self._inner_count += 1``
        has not happened yet. Therefore, ``self._inner_count += 1`` should be
        performed manually here.

        Returns:
            bool: Whether to block the automatic gradient synchronization.
        """
        return ((self._inner_count + 1) % self._accumulative_counts == 0
                or (self._inner_count + 1) == self._max_counts)

    @property
    def inner_count(self):
        """Get the number of updating parameters of optimizer wrapper."""
        return self._inner_count

    def __repr__(self):
        wrapper_info = (f'Type: {type(self).__name__}\n'
                        f'_accumulative_counts: {self._accumulative_counts}\n'
                        'optimizer: \n')
        optimizer_str = repr(self.optimizer) + '\n'
        return wrapper_info + optimizer_str

    def state_dict(self):
        """Refer to:

        colossalai.utils.checkpoint.module_checkpoint.save_checkpoint
        """
        import copy

        import torch.distributed as torch_dist
        from colossalai.tensor import ColoTensor
        from colossalai.utils.checkpoint.utils import gather_tensor
        mapping = dict()
        optim_state = self.optim.state_dict()
        for k, v in optim_state['state'].items():
            for n, t in v.items():
                if isinstance(t, ColoTensor):
                    mapping[(k, n)] = t.dist_spec
                    gather_tensor(t)

        state_dict = {}
        if torch_dist.get_rank() == 0:
            state_dict = copy.deepcopy(optim_state)
            # recover colo tensors in rank0
            for k, v in self.optim.state_dict()['state'].items():
                for n, t in v.items():
                    if isinstance(t, ColoTensor):
                        assert hasattr(t, 'save_ready')
                        t.set_dist_spec(mapping[(k, n)])
                        delattr(t, 'save_ready')

        del optim_state
        del mapping
        torch_dist.barrier()
        return state_dict
