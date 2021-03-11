from torch import nn

class FreezeMixin(nn.Module):
    def train(self, mode=True):
        super().train(mode)

        # unset batchnorm training mode
        if mode and self.freeze:
            for param in self.parameters():
                param.requires_grad = False
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.eval()
