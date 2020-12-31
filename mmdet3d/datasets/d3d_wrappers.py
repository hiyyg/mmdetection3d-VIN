'''
This file provides dataset interfaces with d3d as backend
'''
import d3d
import mmcv
import msgpack
import numpy as np
import os.path as osp
import torch
import tqdm
from d3d.abstraction import ObjectTag, ObjectTarget3D, Target3DArray
from d3d.benchmarks import DetectionEvaluator
from d3d.box import crop_3dr
from d3d.dataset.base import DetectionDatasetBase
from d3d.dataset.kitti import KittiObjectLoader
from d3d.dataset.kitti360 import KITTI360Loader
from d3d.dataset.nuscenes import NuscenesLoader
from d3d.dataset.waymo import WaymoLoader
from mmcv.utils import print_log
from pathlib import Path
from random import random
from scipy.spatial.transform import Rotation

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

def collect_ann_file(loader: DetectionDatasetBase, lidar_name: str, debug: bool = False):
    metalist = []

    loader_size = 3 if debug else len(loader) 
    if loader.phase == "testing":
        # currently no much metadata for testing samples
        for i in tqdm.trange(loader_size, desc="creating annotation"):
            metadata = dict(uidx=loader.identity(i))
            metalist.append(metadata)
    else: # traininig or validation phase
        for i in tqdm.trange(loader_size, desc="creating annotation"):
            metadata = dict(uidx=loader.identity(i))
            annos = dict(arr=[], num_lidar_pts=[])

            # parse array of objects
            objects = loader.annotation_3dobject(i)
            metadata['anno_frame'] = objects.frame
            box_arr = objects.to_numpy()

            # adapt to mmdet3d coordinate
            box_arr[:, 4] -= box_arr[:, 7] / 2 # move center to box bottom
            box_arr[:, [5,6]] = box_arr[:, [6,7]].copy() # swap w and h
            box_arr[:, 8] += np.pi / 2 # change yaw angle zero direction
            annos['arr'] = box_arr.tolist()
            
            # calculate number of points in the boxes
            cloud = loader.lidar_data(i, names=lidar_name)
            box_arr = torch.tensor(box_arr[:, 2:9], dtype=torch.float32)
            mask = crop_3dr(torch.tensor(cloud), box_arr)
            npts = mask.sum(dim=1)
            annos['num_lidar_pts'] = npts.tolist()

            metadata['annos'] = annos
            metalist.append(metadata)

    return metalist

def d3d_data_prep(ds_name, root_path, info_prefix, out_dir=None,
    trainval_split=0.8, inzip=False, lidar_name=0, debug=False):
    ds_type = resolve_dataset_type(ds_name)
    seed = int(str(random())[2:])
    if isinstance(lidar_name, int):
        lidar_name = ds_type.VALID_LIDAR_NAMES[lidar_name]
    if out_dir is None:
        out_dir = root_path

    # Creating info for training set
    train_loader = ds_type(root_path, inzip=inzip, phase="training",
        trainval_split=trainval_split, trainval_random=seed)
    info_path = Path(root_path, f'{info_prefix}_infos_train.pkl')
    mmcv.dump(collect_ann_file(train_loader, lidar_name, debug=debug), info_path)

    # Creating info to validation set
    val_loader = ds_type(root_path, inzip=inzip, phase="validation",
        trainval_split=trainval_split, trainval_random=seed)
    info_path = Path(root_path, f'{info_prefix}_infos_val.pkl')
    mmcv.dump(collect_ann_file(val_loader, lidar_name, debug=debug), info_path)

    # Creating info to test set
    test_loader = ds_type(root_path, inzip=inzip, phase="testing")
    info_path = Path(root_path, f'{info_prefix}_infos_test.pkl')
    mmcv.dump(collect_ann_file(test_loader, lidar_name, debug=debug), info_path)

    # Creating dataset ground-truth sampler
    # TODO: create pipeline for each possible dataset and call create_groundtruth_database


@DATASETS.register_module()
class D3DDataset(Custom3DDataset):
    def __init__(self,
                 ds_type: DetectionDatasetBase,
                 data_root,
                 ann_file,

                 inzip=False,
                 phase="training",
                 trainval_split=0.8,
                 trainval_random=False,

                 pipeline=None,
                 modality=None,
                 classes=None,
                 filter_empty_gt=True,
                 test_mode=False,

                 lidar_name=0,
                 camera_name=0):

        # create loader
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

        # create annotation file if needed
        if not Path(ann_file).exists():
            mmcv.dump(collect_ann_file(self._loader, self.lidar_name), ann_file)

        # then call base constructor
        classes = classes or [c.name for c in ds_type.VALID_OBJ_CLASSES]
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

        # store box dimension in case it's not full
        self._box_dimension = None

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
            if annos['arr']:
                box_arr = np.array(annos['arr'])
            else:
                box_arr = np.empty((0, self._box_dimension))
            if box_arr.shape[1] == 12: # with velocity
                self._box_dimension = 12
                gt_bboxes_3d = LiDARInstance3DBoxes(box_arr[:, 2:11], box_dim=9)
            else:
                self._box_dimension = 9
                gt_bboxes_3d = LiDARInstance3DBoxes(box_arr[:, 2:9], box_dim=7)
            gt_labels_3d, gt_names = [], []
            for cid in box_arr[:, 0].astype(int):
                cenum = self._loader.VALID_OBJ_CLASSES(cid)
                gt_names.append(cenum.name)
                if cenum.name in self.CLASSES:
                    gt_labels_3d.append(self.CLASSES.index(cenum.name))
                else:
                    gt_labels_3d.append(-1)
            gt_labels_3d = np.asarray(gt_labels_3d)
            gt_names = np.asarray(gt_names)

            input_dict['ann_info'] = dict(
                gt_bboxes_3d=gt_bboxes_3d,
                gt_labels_3d=gt_labels_3d,
                gt_names=gt_names
            )

        # mmcv.dump(input_dict, f"./.dev_scripts/temp/%s_d3d.pkl" % (sample_idx[0] + str(sample_idx[1])))
        # raise ValueError("Break")

        return input_dict

    def format_results(self, outputs, msgfile_prefix=None):
        """Convert 3D detection results to d3d TargetArray format
        
        Args:
            msgfile_prefix (str | None): The prefix of msg file.
        """
        assert isinstance(outputs, list), 'outputs must be a list (current type: %s)' % str(type(outputs))

        parsed_outputs = []

        for idx, result in enumerate(outputs):
            detections = Target3DArray(frame=self.data_infos[idx]['anno_frame'])
            for box, score, label in zip(result['boxes_3d'].tensor.tolist(),
                                         result['scores_3d'].tolist(),
                                         result['labels_3d'].tolist()):
                position = box[:3]
                dimension = box[3:6]

                # adapt back from mmdet3d format
                position[2] += dimension[2] / 2
                dimension[0], dimension[1] = dimension[1], dimension[0]
                rotation = Rotation.from_euler("Z", box[6] - np.pi / 2)

                tag = ObjectTag(self.CLASSES[label], self._loader.VALID_OBJ_CLASSES, score)
                detections.append(ObjectTarget3D(position, rotation, dimension, tag))
            parsed_outputs.append(detections)

        if msgfile_prefix is not None:
            mmcv.mkdir_or_exist(msgfile_prefix)
            with open(osp.join(msgfile_prefix, "results.msg"), "wb") as fout:
                fout.write(msgpack.packb([arr.serialize() for arr in parsed_outputs], use_single_float=True))

        return parsed_outputs

    def evaluate(self,
                 results, 
                 logger=None,
                 msgfile_prefix=None):
        classes = [self._loader.VALID_OBJ_CLASSES[name] for name in self.CLASSES]
        evaluator = DetectionEvaluator(classes, 0.7)

        anno_dt_list = self.format_results(results, msgfile_prefix=msgfile_prefix)

        print_log('Calculating eval metrics', logger=logger)
        for i, info in enumerate(mmcv.track_iter_progress(self.data_infos)):
            anno_gt = self._loader.annotation_3dobject(info['uidx'])
            anno_dt = anno_dt_list[i]
            stats = evaluator.calc_stats(anno_gt, anno_dt)
            evaluator.add_stats(stats)

        results_dict = dict()
        for k, v in evaluator.ap().items():
            results_dict["AP_" + k.name] = v
        return results_dict

    def __str__(self):
        return "D3DDataset(%s), phase: %s" % (self._loader.__class__.__name__, self._loader.phase)

if __name__ == "__main__":
    d3d_data_prep("kitti", "/mnt/storage8t/jacobz/kitti_object_extracted", "d3d_kitti_debug", debug=True)
