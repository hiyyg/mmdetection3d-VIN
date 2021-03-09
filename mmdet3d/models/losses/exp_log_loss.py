import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.losses.utils import weight_reduce_loss

from ..builder import LOSSES

def exp_log_dice_loss(log_probs, labels, gamma=1, class_weight=None):
    log_dices = []
    selected_weights = []
    lg2 = 0.6931471805599453
    for c in range(log_probs.size(1)):
        loc = labels == c
        ngt = loc.sum()
        if ngt == 0:
            continue

        ni = torch.logsumexp(log_probs[loc, c], dim=0)
        npr = torch.logsumexp(log_probs[:, c], dim=0)
        log_dice = lg2 + ni - torch.logaddexp(ngt.to(npr.dtype).log(), npr)
        log_dices.append(log_dice)
        if class_weight is not None:
            selected_weights.append(class_weight[c])

    losses = torch.stack(log_dices).neg().pow(gamma)
    if selected_weights:
        selected_weights = log_probs.new_tensor(selected_weights)
        losses = losses * selected_weights
    return losses.mean()

@LOSSES.register_module()
class ExpLogDiceLoss(nn.Module):
    def __init__(self,
                 gamma=0.3,
                 use_sigmoid=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0
    ):
        """
        Exponential Logarithm Dice Loss.

        `3D Segmentation with Exponential Logarithmic Loss for Highly
        Unbalanced Object Sizes <https://arxiv.org/abs/1809.00076>`
        """
        super().__init__()
        self.gamma = gamma
        self.use_sigmoid = use_sigmoid
        assert not self.use_sigmoid
        self.reduction = reduction
        self.class_weight = class_weight
        self.loss_weight = loss_weight

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):

        log_probs = cls_score.log_softmax(dim=1)

        return self.loss_weight * exp_log_dice_loss(log_probs, label,
            self.gamma, class_weight=self.class_weight)

@LOSSES.register_module()
class ExpLogCrossEntropyLoss(nn.Module):

    def __init__(self,
                 gamma=0.3,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        """Exponential Logarithm CrossEntropyLoss.

        `3D Segmentation with Exponential Logarithmic Loss for Highly
        Unbalanced Object Sizes <https://arxiv.org/abs/1809.00076>`

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(ExpLogCrossEntropyLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        assert not self.use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.gamma = gamma

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction (str, optional): The method used to reduce the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        logits = F.log_softmax(cls_score, 1).gather(1, label.unsqueeze(1)).squeeze()
        logits = logits.neg().clamp_min(1e-9).pow(self.gamma)
        
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
            logits = logits * class_weight[label]

        loss = self.loss_weight * weight_reduce_loss(logits,
            weight, reduction=reduction, avg_factor=avg_factor)

        return loss
