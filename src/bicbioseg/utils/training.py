from contextlib import nullcontext
import math

import torch


def autocast_context(device, precision):
    if precision == "fp32":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[precision])


def validate_training_options(device, options):
    precision = options["precision"]
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be 'fp32', 'fp16', or 'bf16'.")
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 training currently requires CUDA; use fp32 on MPS or bf16 on CPU.")
    if precision == "bf16" and device.type == "mps":
        raise ValueError("MPS training currently supports precision='fp32' only.")
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support bf16.")
    steps = options["accumulation_steps"]
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("accumulation_steps must be a positive integer.")
    clip = options["max_grad_norm"]
    if clip is not None and (not math.isfinite(clip) or clip <= 0):
        raise ValueError("max_grad_norm must be finite and positive.")
    weights = options["aux_loss_weights"]
    if weights is not None and any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("aux_loss_weights must be finite and nonnegative.")


def make_scheduler(optimizer, name, kwargs, epochs, mode):
    kwargs = dict(kwargs or {})
    if name is None:
        if kwargs:
            raise ValueError("scheduler_kwargs requires a scheduler.")
        return None
    if name == "step":
        kwargs.setdefault("step_size", 10)
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    if name == "cosine":
        kwargs.setdefault("T_max", epochs)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    if name == "plateau":
        kwargs.setdefault("mode", mode)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    raise ValueError("scheduler must be None, 'step', 'cosine', or 'plateau'.")
