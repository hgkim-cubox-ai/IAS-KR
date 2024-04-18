from .ias_model import IASModel
from .fc_model import FCModel
from .lbp_model import LBPModel


MODEL_DICT = {
    'ias_model': IASModel,
    'fc_model': FCModel,
    'lbp_model': LBPModel,
}