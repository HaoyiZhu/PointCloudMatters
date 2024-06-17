import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDivergence(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar):
        if mu is None:
            return 0

        batch_size = mu.size(0)
        assert batch_size != 0
        if mu.data.ndimension() == 4:
            mu = mu.view(mu.size(0), mu.size(1))
        if logvar.data.ndimension() == 4:
            logvar = logvar.view(logvar.size(0), logvar.size(1))

        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        total_kld = klds.sum(1).mean(0, True)
        dimension_wise_kld = klds.mean(0)
        mean_kld = klds.mean(1).mean(0, True)

        return total_kld[0]  # , dimension_wise_kld, mean_kld
