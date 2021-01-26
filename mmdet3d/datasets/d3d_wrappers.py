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
import pickle
from d3d.abstraction import ObjectTag, ObjectTarget3D, Target3DArray, EgoPose
from d3d.benchmarks import DetectionEvaluator
from d3d.box import crop_3dr
from d3d.dataset.base import DetectionDatasetBase, TrackingDatasetBase
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

if __name__ == "__main__":
    def register_module():
        def pass_through(cls):
            return cls
        return pass_through
else:
    register_module = DATASETS.register_module

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

def collect_ann_file(loader: DetectionDatasetBase, lidar_name: str, debug: bool = False, ninter_frames: int = 6):
    metalist = []
    default_frame = loader.VALID_LIDAR_NAMES[0]

    loader_size = 3 if debug else len(loader) 
    if loader.phase == "testing":
        # currently no much metadata for testing samples
        for i in tqdm.trange(loader_size, desc="creating annotation"):
            metadata = dict(uidx=loader.identity(i))

            # add intermediate lidar frames
            sweeps = []
            calib = loader.calibration_data(i)
            with loader.return_path():
                inter_lidar = loader.intermediate_data(i, names=default_frame, ninter_frames=ninter_frames)
            pose = loader.pose(i)
            for frame in inter_lidar:
                lidar_ego_rt = calib.get_extrinsic(frame_from=default_frame)
                rt = np.linalg.inv(lidar_ego_rt).dot(np.linalg.inv(pose.homo())).dot(frame.pose.homo()).dot(lidar_ego_rt)
                sweep = dict(data_path=frame.file.resolve(), timestamp=frame.timestamp,
                             sensor2lidar_translation=rt[:3,3], sensor2lidar_rotation=rt[:3,:3])
                sweeps.append(sweep)
            metadata['sweeps'] = sweeps

            metalist.append(metadata)
    else: # traininig or validation phase
        for i in tqdm.trange(loader_size, desc="creating annotation"):
            metadata = dict(uidx=loader.identity(i))
            annos = dict(arr=[], num_lidar_pts=[])

            # parse array of objects
            objects = loader.annotation_3dobject(i)
            metadata['anno_frame'] = default_frame
            if len(objects) == 0:
                continue
            calib = loader.calibration_data(i)
            objects = calib.transform_objects(objects, default_frame)
            box_arr = objects.to_numpy()

            # adapt to mmdet3d coordinate
            box_arr[:, 4] -= box_arr[:, 7] / 2 # move center to box bottom
            box_arr[:, [6,5]] = box_arr[:, [5,6]].copy() # swap l and w
            box_arr[:, 8] = -(box_arr[:, 8] + np.pi / 2) # change yaw angle zero direction
            box_arr[:, 8] = (box_arr[:, 8] + np.pi) % (2*np.pi) - np.pi # wrap angles
            annos['arr'] = box_arr.tolist()
            
            # calculate number of points in the boxes
            cloud = loader.lidar_data(i, names=lidar_name)
            box_arr = torch.tensor(box_arr[:, 2:9], dtype=torch.float32)
            mask = crop_3dr(torch.tensor(cloud), box_arr)
            npts = mask.sum(dim=1)
            annos['num_lidar_pts'] = npts.tolist()

            metadata['annos'] = annos

            # add intermediate lidar frames
            sweeps = []
            with loader.return_path():
                inter_lidar = loader.intermediate_data(i, names=default_frame, ninter_frames=ninter_frames)
            pose = loader.pose(i)
            for frame in inter_lidar:
                lidar_ego_rt = calib.get_extrinsic(frame_from=default_frame)
                rt = np.linalg.inv(lidar_ego_rt).dot(np.linalg.inv(pose.homo())).dot(frame.pose.homo()).dot(lidar_ego_rt)
                sweep = dict(data_path=frame.file.resolve(), timestamp=frame.timestamp,
                             sensor2lidar_translation=rt[:3,3], sensor2lidar_rotation=rt[:3,:3])
                sweeps.append(sweep)
            metadata['sweeps'] = sweeps

            metalist.append(metadata)

    return metalist

@register_module()
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
                 obj_classes=None,
                 pts_classes=None,
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
        obj_classes = obj_classes or [c.name for c in ds_type.VALID_OBJ_CLASSES]
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=obj_classes,
            modality=modality,
            box_type_3d='LiDAR',
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode
        )
        self.CLASSES_PTS = pts_classes or [c.value for c in ds_type.VALID_PTS_CLASSES]
        assert len(self.CLASSES_PTS) < 255
        # use 0 as background and unknown class
        self.CLASSES_PTS = np.array(self.CLASSES_PTS + [0], dtype='u1')

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
        if isinstance(self._loader, TrackingDatasetBase):
            input_dict['timestamp'] = self._loader.timestamp(sample_idx)

        if "sweeps" in self.data_infos[index]:
            input_dict["sweeps"] = self.data_infos[index]["sweeps"]

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

            # add semantic points if available
            if hasattr(self._loader, "annotation_3dpoints"):
                with self._loader.return_path():
                    semantic_path = self._loader.annotation_3dpoints(
                        sample_idx, names=self.lidar_name)['semantic']
                input_dict['ann_info']['pts_semantic_mask_path'] = semantic_path

        # mmcv.dump(input_dict, f"./.dev_scripts/temp_d3d/%s_nus.pkl" % (sample_idx[0] + '-' + str(sample_idx[1])))
        # mmcv.dump(input_dict, f"./.dev_scripts/temp_d3d/%d_kitti.pkl" % sample_idx)

        return input_dict

    def format_results(self, outputs, msgfile_prefix=None):
        """Convert 3D detection results to d3d TargetArray format
        
        Args:
            msgfile_prefix (str | None): The prefix of msg file.
        """
        assert isinstance(outputs, list), 'outputs must be a list (current type: %s)' % str(type(outputs))

        parsed_detections = []
        parsed_segmentation = []

        for idx, result in enumerate(outputs):
            if 'pts_bbox' in result: # this is the case with CenterNet output
                bbox_result = result.pop('pts_bbox')
            else:
                bbox_result = result

            # parse object detection result
            detections = Target3DArray(frame=self.data_infos[idx]['anno_frame'])
            for box, score, label in zip(bbox_result['boxes_3d'].tensor.tolist(),
                                         bbox_result['scores_3d'].tolist(),
                                         bbox_result['labels_3d'].tolist()):
                position = box[:3]
                dimension = box[3:6]

                # adapt back from mmdet3d format
                position[2] += dimension[2] / 2
                dimension[0], dimension[1] = dimension[1], dimension[0]
                rotation = Rotation.from_euler("Z", box[6] - np.pi / 2)

                tag = ObjectTag(self.CLASSES[label], self._loader.VALID_OBJ_CLASSES, score)
                detections.append(ObjectTarget3D(position, rotation, dimension, tag)) # TODO(zyxin): add velocity output
            parsed_detections.append(detections)

            # parse segmentation result
            if 'pts_pointwise' in result:
                semantic_result = result.pop('pts_pointwise')
                scores = semantic_result.pop('semantic_scores').numpy()
                label = semantic_result.pop('semantic_label').numpy()
                label = self.CLASSES_PTS[label]
                parsed_segmentation.append(dict(semantic_scores=scores, semantic_label=label))
            else:
                parsed_segmentation.append(None)

        if msgfile_prefix is not None:
            mmcv.mkdir_or_exist(msgfile_prefix)
            with open(osp.join(msgfile_prefix, "detection_results.msg"), "wb") as fout:
                fout.write(msgpack.packb([arr.serialize()
                                          for arr in parsed_detections
                ], use_single_float=True))
            if parsed_segmentation[0] is not None:
                with open(osp.join(msgfile_prefix, "segmentation_results.pkl"), "wb") as fout:
                    pickle.dump(parsed_segmentation, fout)

        return parsed_detections, parsed_segmentation

    def evaluate(self,
                 results, 
                 logger=None,
                 msgfile_prefix=None):
        classes = [self._loader.VALID_OBJ_CLASSES[name] for name in self.CLASSES]
        evaluator = DetectionEvaluator(classes, 0.7)

        anno_dt_list, anno_seg_list = self.format_results(results, msgfile_prefix=msgfile_prefix)

        print_log('Calculating eval metrics for object detection', logger=logger)
        for i, info in enumerate(mmcv.track_iter_progress(self.data_infos)):
            anno_gt = self._loader.annotation_3dobject(info['uidx'])
            calib = self._loader.calibration_data(info['uidx'])
            anno_dt = anno_dt_list[i]
            stats = evaluator.calc_stats(anno_gt, anno_dt, calib)
            evaluator.add_stats(stats)

        iou_list = [] # TODO(zyxin): calculate iou for each class
        valid_ids = np.zeros(128, dtype=bool) # 128 labels at most
        for id in self.CLASSES_PTS:
            valid_ids[id] = True

        print_log('Calculating eval metrics for semantic segmentation', logger=logger)
        for i, info in enumerate(mmcv.track_iter_progress(self.data_infos)):
            anno_seg = anno_seg_list[i] # estimated
            if anno_seg is not None:
                anno_seg_gt = self._loader.annotation_3dpoints(info['uidx'])['semantic']
                anno_seg_id = anno_seg.pop('semantic_label')
                assert len(anno_seg_id) == len(anno_seg_gt)

                # only evaluate trained id
                valid_mask = valid_ids[anno_seg_gt]
                anno_seg_gt = anno_seg_gt[valid_mask]
                anno_seg_id = anno_seg_id[valid_mask]
                tp = np.sum(anno_seg_id == anno_seg_gt)
                iou = tp / (len(anno_seg_gt)*2 - tp)
                iou_list.append(iou)

        results_dict = dict()
        for k, v in evaluator.ap().items():
            results_dict["AP/" + k.name] = v
        if iou_list:
            results_dict["mIoU"] = np.mean(iou_list)
        return results_dict

    def __str__(self):
        return "D3DDataset(%s), phase: %s" % (self._loader.__class__.__name__, self._loader.phase)


def d3d_data_prep(ds_name, root_path, info_prefix, out_dir=None,
    trainval_split=0.8, inzip=False, lidar_name=0, debug=False,
    database_save_path=None, db_info_save_path=None):
    ds_type = resolve_dataset_type(ds_name)
    seed = int(str(random())[2:])
    if isinstance(lidar_name, int):
        lidar_name = ds_type.VALID_LIDAR_NAMES[lidar_name]
    if out_dir is None:
        out_dir = root_path
    if debug and not info_prefix.endswith("debug"):
        info_prefix += "_debug"

    # Creating info for training set
    train_loader = ds_type(root_path, inzip=inzip, phase="training",
        trainval_split=trainval_split, trainval_random=seed, trainval_byseq=True)
    traininfo_path = info_path = Path(root_path, f'{info_prefix}_infos_train.pkl')
    mmcv.dump(collect_ann_file(train_loader, lidar_name, debug=debug), info_path)

    # Creating info to validation set
    val_loader = ds_type(root_path, inzip=inzip, phase="validation",
        trainval_split=trainval_split, trainval_random=seed, trainval_byseq=True)
    info_path = Path(root_path, f'{info_prefix}_infos_val.pkl')
    mmcv.dump(collect_ann_file(val_loader, lidar_name, debug=debug), info_path)

    # Creating info to test set
    test_loader = ds_type(root_path, inzip=inzip, phase="testing")
    info_path = Path(root_path, f'{info_prefix}_infos_test.pkl')
    mmcv.dump(collect_ann_file(test_loader, lidar_name, debug=debug), info_path)

    # Creating dataset ground-truth sampler
    try:
        from tools.data_converter.create_gt_database import create_groundtruth_database
    except ImportError:
        print("Cannot import tools. Groundtruth database won't be created!")
        return

    # set up pipelines
    file_client_args = dict(backend='disk')
    if ds_name == 'kitti':
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                file_client_args=file_client_args),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True)
        ]
    elif ds_name == 'nuscenes':
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=file_client_args),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=6,
                use_dim=5,
                pad_empty_sweeps=False,
                remove_close=True,
                file_client_args=file_client_args),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_seg_3d='u1')
        ]
    elif ds_name == 'waymo':
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=6,
                use_dim=5,
                file_client_args=file_client_args),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True)
        ]
    else:
        raise NotImplementedError("Dataset not supported")

    dataset = D3DDataset(ds_name, root_path, traininfo_path, inzip=inzip,
                         trainval_split=trainval_split, trainval_random=seed, pipeline=pipeline)
    create_groundtruth_database(dataset, root_path, info_prefix,
                                database_save_path=database_save_path,
                                db_info_save_path=db_info_save_path, with_mask_3d=True)

if __name__ == "__main__":
    d3d_data_prep("nuscenes", "/mnt/storage8t/jacobz/nuscenes_converted", "d3d_nuscenes", trainval_split=0.95)
