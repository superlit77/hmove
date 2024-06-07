from typing import Optional

from dhg.datapipe import *
from torch.utils.data import Dataset
from dhg.data import BaseData


class Hdataset(BaseData):
    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__('hdataset', data_root)
        self._content = {
            "num_classes": 2,
            'features': {
                'upon': [{ 'filename': 'features.pkl', 'md5': '05b45e9c38cc95f4fc44b3668cc9ddc9' }],
                'loader': load_from_pickle,
                'preprocess': [to_tensor],
            },
            'g_edge_list': {
                'upon': [{'filename': 'g_edge_list.pkl', 'md5': 'f488389c1edd0d898ce273fbd27822b3'}],
                'loader': load_from_pickle,
            },
            'hg_edge_list': {
                'upon': [{'filename': 'hg_edge_list.pkl', 'md5': 'f488389c1edd0d898ce273fbd27822b3'}],
                'loader': load_from_pickle,
            },
            "labels": {
                "upon": [
                    {
                        "filename": "labels.pkl",
                        "md5": "f1f3c0399c9c28547088f44e0bfd5c81",
                    }
                ],
                "loader": load_from_pickle,
                "preprocess": [to_long_tensor],
            },
            "train_mask": {
                "upon": [
                    {
                        "filename": "train_mask.pkl",
                        "md5": "66ea36bae024aaaed289e1998fe894bd",
                    }
                ],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
            "val_mask": {
                "upon": [
                    {
                        "filename": "val_mask.pkl",
                        "md5": "6c0d3d8b752e3955c64788cc65dcd018",
                    }
                ],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
            "test_mask": {
                "upon": [
                    {
                        "filename": "test_mask.pkl",
                        "md5": "0e1564904551ba493e1f8a09d103461e",
                    }
                ],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },

        }