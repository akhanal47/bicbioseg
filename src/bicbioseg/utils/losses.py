import torch
from torch import nn
from torch.nn import functional as F


def _probabilities_and_targets(inputs, targets, mode):
    if mode == "multiclass":
        probabilities = inputs.softmax(dim=1)
        targets = F.one_hot(targets.long(), inputs.shape[1]).movedim(-1, 1)
    elif mode == "binary":
        probabilities = inputs.sigmoid()
    else:
        raise ValueError("mode must be 'binary' or 'multiclass'.")
    return probabilities, targets.to(inputs.dtype)


def _overlap(inputs, targets, mode, smooth, alpha=0.5, beta=0.5):
    probabilities, targets = _probabilities_and_targets(inputs, targets, mode)
    dims = (0,) + tuple(range(2, inputs.ndim))
    tp = (probabilities * targets).sum(dim=dims)
    fp = (probabilities * (1 - targets)).sum(dim=dims)
    fn = ((1 - probabilities) * targets).sum(dim=dims)
    return ((tp + smooth) / (tp + alpha * fp + beta * fn + smooth)).mean()


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, smooth=1e-6, mode="binary"):
        super().__init__()
        self.smooth = smooth
        self.mode = mode

    def forward(self, inputs, targets):
        return 1 - _overlap(inputs, targets, self.mode, self.smooth)


class BCELoss(nn.Module):
    def forward(self, inputs, targets):
        return F.binary_cross_entropy_with_logits(inputs, targets)


class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, inputs, targets):
        return self.dice(inputs, targets) + F.binary_cross_entropy_with_logits(inputs, targets)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce)
        alpha = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha * (1 - pt).pow(self.gamma) * bce).mean()


class LogCoshDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, inputs, targets):
        return torch.log(torch.cosh(self.dice(inputs, targets)))


class JaccardLoss(DiceLoss):
    def forward(self, inputs, targets):
        return 1 - _overlap(inputs, targets, self.mode, self.smooth, alpha=1, beta=1)


class TverskyLoss(nn.Module):
    # alpha weights false positives; beta weights false negatives.

    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth

    def forward(self, inputs, targets):
        return 1 - _overlap(inputs, targets, "binary", self.smooth, self.alpha, self.beta)


class FocalTverskyLoss(TverskyLoss):
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.25, smooth=1e-6):
        super().__init__(alpha, beta, smooth)
        self.gamma = gamma

    def forward(self, inputs, targets):
        return super().forward(inputs, targets).pow(self.gamma)


class SensitivitySpecificityLoss(nn.Module):
    # weighted squared error, normalized separately over foreground/background.

    def __init__(self, alpha=0.3, smooth=1e-6):
        super().__init__()
        self.alpha, self.smooth = alpha, smooth

    def forward(self, inputs, targets):
        error = (inputs.sigmoid() - targets).square()
        foreground = (error * targets).sum() / (targets.sum() + self.smooth)
        background = (error * (1 - targets)).sum() / ((1 - targets).sum() + self.smooth)
        return self.alpha * foreground + (1 - self.alpha) * background


class UnifiedFocalLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Unified focal is not supported: Use focal or focal_tversky until a validated implementation is available."
        )


class ExponentialLogarithmicLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Exponential logarithmic loss has no validated implementation yet.")


class ShapeAwareLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Shape-aware loss has no implementation yet.")


class IgnoreLabelsLoss(nn.Module):
    def __init__(self, loss, ignore_index):
        super().__init__()
        self.loss, self.ignore_index = loss, ignore_index

    def forward(self, inputs, targets):
        labels = targets[:, 0] if targets.ndim == 4 else targets
        valid = labels != self.ignore_index
        if not valid.any():
            return inputs.sum() * 0
        selected = inputs.movedim(1, -1)[valid].T[None, :, None, :]
        truth = labels[valid][None, None, :]
        if inputs.shape[1] == 1:
            truth = truth[:, None].to(inputs.dtype)
        return self.loss(selected, truth)
