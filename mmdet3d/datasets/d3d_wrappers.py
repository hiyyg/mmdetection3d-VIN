'''
This file provides dataset interfaces with d3d as backend
'''
import d3d
import mmcv
import torch
import tqdm
import numpy as np
from pathlib import Path
from d3d.box import crop_3dr
from d3d.dataset.base import DetectionDatasetBase
from d3d.dataset.kitti import KittiObjectLoader
from d3d.dataset.kitti360 import KITTI360Loader
from d3d.dataset.nuscenes import NuscenesLoader
from d3d.dataset.waymo import WaymoLoader
from random import random

from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet.datasets import DATASETS


def resolve_dataset_type(ds_type) -> DetectionDatasetBase:
    if isinstance(ds_type, DetectionDatasetBase):
        return ds_type

    ds_type = ds_type.lower()
    if ds_type == "kitti":
        return KittiObjectLoader
    elif ds_type == "kitti360":
        return KITTI360Loader
    elif ds_type == "nuscenes":
        return NuscenesLoader
    elif ds_type == "waymo":
        return WaymoLoader
    else:
        raise ValueError("Dataset name not recognized!")

def collect_ann_file(loader: DetectionDatasetBase, lidar_name: str):
    metalist = []

    if loader.phase == "testing":
        # currently no much metadata for testing samples
        for i in tqdm.trange(len(loader), desc="creating annotation"):
            metadata = dict(uidx=loader.identity(i))
            metalist.append(metadata)
    else: # traininig or validation phase
        for i in tqdm.trange(len(loader), desc="creating annotation"):
            metadata = dict(uidx=loader.identity(i))
            annos = dict(arr=[], num_lidar_pts=[])

            # add array of objects
            objects = loader.annotation_3dobject(i)
            box_arr = objects.to_numpy()
            box_arr[:, 2] -= box_arr[:, 5] / 2
            annos['arr'] = box_arr[:, :8].tolist()
            
            # calculate number of points in the boxes
            cloud = loader.lidar_data(i, names=lidar_name)
            box_arr = torch.tensor(box_arr[:, :7], dtype=torch.float32)
            mask = crop_3dr(torch.tensor(cloud), box_arr)
            npts = mask.sum(dim=1)
            annos['num_lidar_pts'] = npts.tolist()

            metadata['annos'] = annos
            metalist.append(metadata)

    return metalist

def d3d_data_prep(ds_name, root_path, info_prefix, out_dir,
    trainval_split=0.8, inzip=False, lidar_name=0):
    ds_type = resolve_dataset_type(ds_name)
    seed = int(str(random())[2:])
    if isinstance(lidar_name, int):
        lidar_name = ds_type.VALID_LIDAR_NAMES[lidar_name]

    # Creating info for training set
    train_loader = ds_type(root_path, inzip=inzip, phase="training",
        trainval_split=trainval_split, trainval_random=seed)
    info_path = Path(root_path, f'{info_prefix}_infos_train.pkl')
    mmcv.dump(collect_ann_file(train_loader, lidar_name), info_path)

    # Creating info to validation set
    val_loader = ds_type(root_path, inzip=inzip, phase="validation",
        trainval_split=trainval_split, trainval_random=seed)
    info_path = Path(root_path, f'{info_prefix}_infos_val.pkl')
    mmcv.dump(collect_ann_file(val_loader, lidar_name), info_path)

    # Creating info to test set
    test_loader = ds_type(root_path, inzip=inzip, phase="testing")
    info_path = Path(root_path, f'{info_prefix}_infos_test.pkl')
    mmcv.dump(collect_ann_file(test_loader, lidar_name), info_path)

@DATASETS.register_module()
class D3DDataset(Custom3DDataset):
    def __init__(self,
                 ds_type: DetectionDatasetBase,
                 data_root,
                 ann_file,

                 inzip=False,
                 phase="training",
                 trainval_split=1,
                 trainval_random=False,

                 pipeline=None,
                 modality=None,
                 classes=None,
                 filter_empty_gt=True,
                 test_mode=False,

                 lidar_name=0,
                 camera_name=0):

        # create loader and annotation file if needed
        ds_type = resolve_dataset_type(ds_type)
        self._loader = ds_type(data_root, inzip=inzip,
            phase=phase, trainval_split=trainval_split,
            trainval_random=trainval_random)

        # parse lidar and camera name selection
        if isinstance(lidar_name, int):
            self.lidar_name = self._loader.VALID_LIDAR_NAMES[lidar_name]
        else:
            self.lidar_name = str(lidar_name)
        if isinstance(camera_name, int):
            self.camera_name = self._loader.VALID_CAM_NAMES[lidar_name]
        else:
            self.camera_name = str(camera_name)

        # create info file if needed
        if not Path(ann_file).exists():
            mmcv.dump(collect_ann_file(self._loader, self.lidar_name), ann_file)

        # then call base constructor
        if classes:
            classes = [ds_type.VALID_OBJ_CLASSES[c] for c in classes]
        else:
            classes = [c for c in ds_type.VALID_OBJ_CLASSES]
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d='LiDAR',
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode
        )

    def get_data_info(self, index):
        sample_idx = self.data_infos[index]["uidx"]
        with self._loader.return_path():
            pts_filename = self._loader.lidar_data(sample_idx, self.lidar_name)
            img_filename = self._loader.camera_data(sample_idx, self.camera_name)
        calib = self._loader.calibration_data(sample_idx)
        lidar2img = calib.get_extrinsic(self.camera_name, self.lidar_name)
            
        input_dict = dict(
            sample_idx=sample_idx,
            pts_filename=pts_filename,
            img_prefix=None,
            img_info=dict(filename=img_filename),
            lidar2img=lidar2img
        )

        if not self.test_mode:
            annos = self.data_infos[index]['annos']
            box_arr = np.array(annos['arr'])
            gt_bboxes_3d = LiDARInstance3DBoxes(box_arr[:, :7])
            gt_labels_3d, gt_names = [], []
            for cid in box_arr[:, 7].astype(int):
                cenum = self._loader.VALID_OBJ_CLASSES(cid)
                gt_names.append(cenum.name)
                if cenum.name in self.CLASSES:
                    gt_labels_3d.append(self.CLASSES.index(cenum))
                else:
                    gt_labels_3d.append(-1)
            gt_labels_3d = np.asarray(gt_labels_3d)
            gt_names = np.asanyarray(gt_names)

            input_dict['ann_info'] = dict(
                gt_bboxes_3d=gt_bboxes_3d,
                gt_labels_3d=gt_labels_3d,
                gt_names=gt_names
            )

        return input_dict

    # def format_results(self):
    #     pass

    # def evaluate(self):
    #     pass

if __name__ == "__main__":
    ds = D3DDataset("kitti", "/mnt/storage8t/jacobz/kitti_object_extracted",
        "/mnt/storage8t/jacobz/kitti_object_extracted/temp_info_trainval.pkl",
        trainval_split=0.8)
    print(ds.get_data_info(2))
