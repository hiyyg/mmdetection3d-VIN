import mmcv
import torch.nn as nn

from ..builder import LOSSES, build_loss

@LOSSES.register_module()
class EnsembleLoss(nn.Module):
    '''
    Combine several losses together
    '''
    def __init__(self, losses, **kwargs):
        super(EnsembleLoss, self).__init__()

        self.losses = []
        for loss in losses:
            loss.update(kwargs)
            self.losses.append(build_loss(loss))

    def forward(self, *args, **kwargs):
        result = self.losses[0](*args, **kwargs)
        for loss in self.losses[1:]:
            result = result + loss(*args, **kwargs)
        return result
