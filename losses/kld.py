import torch


class KLDLoss(object):
    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
    
    def __call__(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        loss = 0.5 * (mu**2 + sigma**2 - torch.log(self.eps + sigma**2) - 1)
        return torch.mean(loss, dim=0)  # over batch