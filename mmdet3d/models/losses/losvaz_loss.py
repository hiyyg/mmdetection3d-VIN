# adapted from mmsegmentation
# https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/losses/lovasz_loss.py

import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.losses.utils import weight_reduce_loss

from ..builder import LOSSES

def lovasz_grad(gt_sorted):
    """Computes gradient of the Lovasz extension w.r.t sorted errors.
    See Alg. 1 in paper.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probs, labels, classes='present', class_weight=None):
    """Multi-class Lovasz-Softmax loss.
    Args:
        probs (torch.Tensor): [P, C], class probabilities at each prediction
            (between 0 and 1).
        labels (torch.Tensor): [P], ground truth labels (between 0 and C - 1).
        classes (str | list[int], optional): Classes choosed to calculate loss.
            'all' for all classes, 'present' for classes present in labels, or
            a list of classes to average. Default: 'present'.
        class_weight (list[float], optional): The weight for each class.
            Default: None.
    Returns:
        torch.Tensor: The calculated loss.
    """
    if probs.numel() == 0:
        # only void pixels, the gradients should be 0
        return probs.sum() * 0.
    C = probs.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probs[:, 0]
        else:
            class_pred = probs[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        loss = torch.dot(errors_sorted, lovasz_grad(fg_sorted))
        if class_weight is not None:
            loss *= class_weight[c]
        losses.append(loss)

    return torch.stack(losses).mean()


@LOSSES.register_module()
class LovaszLoss(nn.Module):
    """LovaszLoss.
    This loss is proposed in `The Lovasz-Softmax loss: A tractable surrogate
    for the optimization of the intersection-over-union measure in neural
    networks <https://arxiv.org/abs/1705.08790>`_.
    Args:
        classes (str | list[int], optional): Classes choosed to calculate loss.
            'all' for all classes, 'present' for classes present in labels, or
            a list of classes to average. Default: 'present'.\
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_image is True. Default: 'mean'.
        class_weight (list[float], optional): The weight for each class.
            Default: None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 use_sigmoid=False,
                 classes='present',
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        super(LovaszLoss, self).__init__()

        assert classes in ('all', 'present') or mmcv.is_list_of(classes, int)
        assert not use_sigmoid, "Losvaz has to be based on softmax"

        self.classes = classes
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None

        # transform logits to probs
        cls_score = F.softmax(cls_score, dim=1)

        loss_cls = self.loss_weight * lovasz_softmax_flat(
            cls_score,
            label,
            self.classes,
            class_weight=class_weight)

        return loss_cls
