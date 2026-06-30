import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    base_lr: float = 1e-4,
    end_lr: float = 1e-5,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a learning-rate scheduler with two stages:

    1. Linear warm-up:
       The learning rate increases linearly from 0 to ``base_lr``.

    2. Cosine annealing:
       After warm-up, the learning rate follows a cosine decay from
       ``base_lr`` to ``end_lr``.

    This scheduler is implemented using ``torch.optim.lr_scheduler.LambdaLR``.
    The lambda function returns a multiplicative factor applied to the
    optimizer's initial learning rate.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer whose learning rate will be scheduled.

    num_warmup_steps : int
        Number of optimizer steps used for the linear warm-up phase.

    num_training_steps : int
        Total number of optimizer steps during training.

    num_cycles : float, optional
        Number of cosine cycles after warm-up.

        Typical values:
            0.5 -> half cosine (standard decay)
            1.0 -> one complete cosine wave
            2.0 -> two cosine waves

    base_lr : float, optional
        Maximum learning rate reached after warm-up.

    end_lr : float, optional
        Final learning rate at the end of training.

    last_epoch : int, optional
        Used when resuming training from a checkpoint.

    Returns
    -------
    LambdaLR
        Learning-rate scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        """
        Compute the learning-rate multiplier for the current step.

        Returns
        -------
        float
            Multiplicative factor applied to the optimizer's initial
            learning rate.
        """

        # -------------------------------------------------------------
        # Stage 1: Linear warm-up
        # -------------------------------------------------------------
        if current_step < num_warmup_steps:
            warmup_factor = current_step / max(1, num_warmup_steps)
            return warmup_factor

        # -------------------------------------------------------------
        # Stage 2: Cosine annealing
        # -------------------------------------------------------------
        # Normalize progress into the range [0, 1]
        progress = ( current_step - num_warmup_steps ) / max(1, num_training_steps - num_warmup_steps)
        # Cosine interpolation between 1 and 0
        cosine_factor = 0.5 * ( 1.0 + math.cos(2.0 * math.pi * num_cycles * progress) )
        # Convert cosine factor into an actual learning rate
        current_lr = end_lr + (base_lr - end_lr) * cosine_factor
        # LambdaLR expects a multiplicative factor
        return current_lr / base_lr

    return LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
        last_epoch=last_epoch,
    )