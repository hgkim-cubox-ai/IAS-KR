from torch.nn import MSELoss, CrossEntropyLoss


LOSS_FN_DICT = {
    'mse': MSELoss,
    'crossentropy': CrossEntropyLoss
}