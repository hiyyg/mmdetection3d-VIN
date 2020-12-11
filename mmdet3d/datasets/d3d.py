'''
This file provides dataset interfaces with d3d as backend
'''
from mmdet.datasets import DATASETS
from mmdet3d.datasets.custom_3d import Custom3DDataset
from d3d.dataset.base import DetectionDatasetBase

@DATASETS.register_module()
class D3DDetectionDataset(Custom3DDataset):
    def __init__(self,
                 ds_type: DetectionDatasetBase,
                 base_path,
                 inzip=False,
                 phase="training",
                 trainval_split=1,
                 trainval_random=False,
                 pipeline=None,
                 filter_empty_gt=True):
        super().__init__(
            data_root=base_path,
            ann_file=None,
            pipeline=pipeline,
            classes=ds_type.VALID_OBJ_CLASSES,
            modality=None,
            box_type_3d='LiDAR',
            filter_empty_gt=filter_empty_gt,
            test_mode=phase is not "training"
        )
        self._loader = ds_type(base_path, inzip=inzip,
            phase=phase, trainval_split=trainval_split,
            trainval_random=trainval_random)
    
    def load_annotations(self, ann_file):
        pass

    def get_data_info(self, index):
        pass

    def pre_pipeline(self, index):
        pass

    def prepare_train_data(self, index):
        pass

    def prepare_test_data(self, index):
        pass

    # def format_results(self):
    #     pass

    # def evaluate(self):
    #     pass


