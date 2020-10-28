import torch
from torch import nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weight_reduce_loss

from d3d.math import i0e

def vonmises_logpdf(x, mu, s):
    '''
    Log PDF of von-Mises distribution p(x|mu,k) with s=log(sigma^2)=log(1/k)
    '''
    k = torch.exp(-s)
    return k * torch.cos(x - mu) - torch.log(i0e(k)) - k

def gaussian_nll_loss(pred,
                      target,
                      logvar,
                      weight=None,
                      reduction='mean',
                      avg_factor=None,
                      clamp=False,
                      lambda_g=1.0):

    reg_shape = [1] * len(pred.shape)
    reg_shape[-1] = pred.shape[-1]
    regularize = torch.full(reg_shape, lambda_g, device=pred.device)

    diff = torch.pow(pred - target, 2)
    loss = (diff * torch.exp(-logvar) + regularize * logvar) / 2

    if clamp: # prevent very large loss
        loss = loss.clamp_max(1000.0)

    if torch.any(torch.isnan(loss)):
        print("DUMP: max var =", logvar.abs().max())
        print("DUMP: num of nans =", torch.sum(torch.isnan(loss)))
        raise ValueError("Detected nan in var loss")
    return weight_reduce_loss(loss, weight, reduction, avg_factor)

def gaussian_von_mises_nll_loss(pred,
                                target,
                                logvar,
                                weight=None,
                                reduction='mean',
                                avg_factor=None,
                                clamp=False,
                                angular_index=6,
                                lambda_g=1.0,
                                lambda_v=1.0):
    # losses of ordinary linear dimensions
    gauss_idx = [i for i in range(pred.shape[-1]) if i != angular_index]
    gauss_diff = torch.pow(pred[..., gauss_idx] - target[..., gauss_idx], 2)
    gauss_var = logvar[..., gauss_idx]
    gauss_loss = (gauss_diff * torch.exp(-gauss_var) + lambda_g * gauss_var) / 2

    # loss of angular dimension with regularization
    if not isinstance(angular_index, (list, tuple)):
        angular_index = [angular_index]
    vonmise_var = logvar[..., angular_index]
    vonmise_nll = -vonmises_logpdf(pred[..., angular_index],
                                 target[..., angular_index],
                                 vonmise_var)
     # introduce gradient when logvar is large
    vonmise_loss = vonmise_nll + lambda_v * F.elu(vonmise_var)
    loss = torch.cat([gauss_loss, vonmise_loss], dim=-1)

    if clamp: # prevent very large loss
        loss = loss.clamp_max(1000.0)

    if torch.any(torch.isnan(loss)):
        print("DUMP: max var =", logvar.abs().max())
        print("DUMP: num of nans =", torch.sum(torch.isnan(loss)))
        raise ValueError("Detected nan in var loss")
    return weight_reduce_loss(loss, weight, reduction, avg_factor)

@LOSSES.register_module()
class GaussianNLLLoss(nn.Module):
    """Uncertainty regression loss using NLL with Gaussian.

    arXiv: TO BE ADDED

    Args:
        lambda_g (float, optional): The regularization coefficient
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """
    def __init__(self,
                 clamp=False,
                 lambda_g=1.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(GaussianNLLLoss, self).__init__()
        self.clamp = clamp
        self.lambda_g = lambda_g
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                logvar,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_nll = self.loss_weight * gaussian_nll_loss(
            pred,
            target,
            logvar,
            weight,
            clamp=self.clamp,
            lambda_g=self.lambda_g,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_nll

@LOSSES.register_module()
class GaussianVonMisesNLLLoss(nn.Module):
    """Uncertainty regression loss using NLL with Gaussian and von-Mises Loss.

    arXiv: TO BE ADDED

    Args:
        lambda_g (float, optional): The regularization coefficient
            of the Guassian part
        lambda_v (float, optional): The regularization coefficient
            of the von-Mises part
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self,
                 clamp=False,
                 lambda_g=1.0,
                 lambda_v=1.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(GaussianVonMisesNLLLoss, self).__init__()
        self.clamp = clamp
        self.lambda_g = lambda_g
        self.lambda_v = lambda_v
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                logvar,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_nll = self.loss_weight * gaussian_von_mises_nll_loss(
            pred,
            target,
            logvar,
            weight,
            clamp=self.clamp,
            angular_index=6, # currently we fix angular term at 7th position
            lambda_v=self.lambda_v,
            lambda_g=self.lambda_g,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_nll
