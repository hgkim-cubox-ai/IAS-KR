from .kld import KLDLoss
from torch.nn import MSELoss, BCELoss


LOSS_FN_DICT = {
    'mse': MSELoss,
    'kld': KLDLoss,
    'bce': BCELoss
}