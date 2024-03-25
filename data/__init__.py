from .ias_dataset import IASDataset


DATASET_DICT = {
    'cubox_4k_2211': IASDataset,
    'IAS_cubox_train_230102_renew': IASDataset,
    'IAS_cubox_train_230117_extra': IASDataset,
    'real_driver': IASDataset,
    'real_id': IASDataset,
    'real_passport': IASDataset,
    'shinhan': IASDataset
}