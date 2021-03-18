'''
This file provides dataset interfaces with d3d as backend
'''
import mmcv
import msgpack
import msgpack_numpy

msgpack_numpy.patch()
import numpy as np
import os.path as osp
import pickle
import torch
import tqdm
from d3d.abstraction import (EgoPose, ObjectTag, ObjectTarget3D, Target3DArray,
                             TrackingTarget3D)
from d3d.benchmarks import DetectionEvaluator, SegmentationEvaluator
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

def collect_ann_file(loader: DetectionDatasetBase, lidar_name: str, debug: bool = False, ninter_frames: int = 10, with_label=None):
    if with_label is None:
        with_label = loader.phase != "testing"
    
    metalist = []

    loader_size = 3 if debug else len(loader)
    for i in tqdm.trange(loader_size, desc="creating annotation"):
        metadata = dict(uidx=loader.identity(i), anno_frame=lidar_name)
        calib = loader.calibration_data(i)

        # add intermediate lidar frames
        sweeps = []
        with loader.return_path():
            inter_lidar = loader.intermediate_data(i, names=lidar_name, ninter_frames=ninter_frames)
        pose = loader.pose(i)
        for frame in inter_lidar:
            lidar_ego_rt = calib.get_extrinsic(frame_from=lidar_name)
            rt = np.linalg.inv(lidar_ego_rt).dot(np.linalg.inv(pose.homo())).dot(frame.pose.homo()).dot(lidar_ego_rt)
            sweep = dict(data_path=frame.file.resolve(), timestamp=frame.timestamp,
                        sensor2lidar_translation=rt[:3,3], sensor2lidar_rotation=rt[:3,:3])
            sweeps.append(sweep)
        metadata['sweeps'] = sweeps

        if with_label:
            annos = dict(arr=[], num_lidar_pts=[], num_radar_pts=[])

            # parse array of objects
            objects = loader.annotation_3dobject(i)
            if len(objects) != 0:
                objects = calib.transform_objects(objects, lidar_name)
                
                # calculate number of points in the boxes
                annos['num_lidar_pts'] = [box.aux['num_lidar_pts'] for box in objects]
                annos['num_radar_pts'] = [box.aux['num_radar_pts'] for box in objects]

                # adapt box params to mmdet3d coordinate
                box_arr = objects.to_numpy()
                box_arr[:, 4] -= box_arr[:, 7] / 2 # move center to box bottom
                box_arr[:, [6,5]] = box_arr[:, [5,6]].copy() # swap l and w
                box_arr[:, 8] = -(box_arr[:, 8] + np.pi / 2) # change yaw angle zero direction
                box_arr[:, 8] = (box_arr[:, 8] + np.pi) % (2*np.pi) - np.pi # wrap angles
                annos['arr'] = box_arr.tolist()

                metadata['annos'] = annos
            elif loader.phase == "training": # Skip empty sample in training split
                continue

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
                 filter_ignore=True, # skip boxes with classid = 0
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
        self.filter_ignore = filter_ignore
        self.CLASSES_PTS = pts_classes or []
        assert len(self.CLASSES_PTS) < 255
        # use 0 as background and unknown class
        self.CLASSES_PTS = np.array(self.CLASSES_PTS + [0], dtype='u1')

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
            input_dict['timestamp'] = self._loader.timestamp(sample_idx) / 1e6

        if "sweeps" in self.data_infos[index]:
            input_dict["sweeps"] = self.data_infos[index]["sweeps"]

        if not self.test_mode:
            if 'annos' not in self.data_infos[index]:
                return None
            annos = self.data_infos[index]['annos']
            if annos['arr']:
                box_arr = np.array(annos['arr'])
            else:
                return None
            # filter boxes
            if self.filter_empty_gt:
                box_arr = box_arr[np.array(annos['num_lidar_pts']) + np.array(annos['num_radar_pts']) > 0]
            if self.filter_ignore:
                box_arr = box_arr[box_arr[:, 0] > 0]
            if box_arr.shape[1] == 12: # with velocity
                gt_bboxes_3d = LiDARInstance3DBoxes(box_arr[:, 2:11], box_dim=9)
            else:
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

    def format_results(self, outputs, dump_prefix=None, dump_visual=False, dump_submission=False, score_threshold=0.4, iterative=False):
        """Convert 3D detection results to d3d TargetArray format
        
        Args:
            dump_prefix (str | None): The prefix of dumped output file.
            dump_visual (bool): Whether dump visualization files.
            dump_submission (bool): Whether dump submission files.
        """
        assert isinstance(outputs, list), 'outputs must be a list (current type: %s)' % str(type(outputs))

        packer = msgpack.Packer(use_single_float=True)
        
        if dump_prefix is not None:
            mmcv.mkdir_or_exist(dump_prefix)
            det_fout = open(osp.join(dump_prefix, "detection_results.msg"), "wb")
            det_fout.write(packer.pack_array_header(len(outputs)))
            seg_fout = open(osp.join(dump_prefix, "segmentation_results.msg"), "wb")
            seg_fout.write(packer.pack_array_header(len(outputs)))

        if dump_visual:
            vis_path = osp.join(dump_prefix, "visual")
            mmcv.mkdir_or_exist(vis_path)

        if dump_submission:
            sdet_path = osp.join(dump_prefix, "submission_detection")
            sseg_path = osp.join(dump_prefix, "submission_segmentation")
            mmcv.mkdir_or_exist(sdet_path)
            mmcv.mkdir_or_exist(sseg_path)

        def parse():
            avel = [float('nan')] * 3 # dummy angular velocity
            for idx, result in enumerate(outputs):
                if 'pts_bbox' in result: # this is the case with CenterNet output
                    bbox_result = result.pop('pts_bbox')
                else:
                    bbox_result = result

                # parse object detection result
                detections = Target3DArray(frame=self.data_infos[idx]['anno_frame'])
                boxes_3d = bbox_result.pop('boxes_3d').tensor.tolist()
                scores_3d = bbox_result.pop('scores_3d').tolist()
                labels_3d = bbox_result.pop('labels_3d').tolist()
                for box, score, label in zip(boxes_3d, scores_3d, labels_3d):
                    position = box[:3]
                    dimension = box[3:6]

                    # adapt back from mmdet3d format
                    position[2] += dimension[2] / 2
                    dimension[0], dimension[1] = dimension[1], dimension[0]
                    rotation = Rotation.from_euler("Z", -box[6] - np.pi / 2)

                    tag = ObjectTag(self.CLASSES[label], self._loader.VALID_OBJ_CLASSES, score)

                    if len(box) > 7: # with velocity output
                        vel = [box[7], box[8], 0.0]
                        detections.append(TrackingTarget3D(position, rotation, dimension, vel, avel, tag))
                    else:
                        detections.append(ObjectTarget3D(position, rotation, dimension, tag))

                # parse segmentation result
                if result.get('pts_pointwise', None) is not None:
                    semantic_result = result.pop('pts_pointwise')
                    scores = semantic_result.pop('semantic_scores').numpy()
                    label = semantic_result.pop('semantic_label').numpy()
                    label = self.CLASSES_PTS[label]
                    segmentations = dict(semantic_scores=scores, semantic_label=label)
                else:
                    segmentations = None

                if dump_prefix is not None:
                    det_fout.write(packer.pack(detections.serialize()))
                    seg_fout.write(packer.pack(segmentations))
                    uidx = self.data_infos[idx]['uidx']

                    # save visualization results
                    if dump_visual:
                        cloud = self._loader.lidar_data(uidx)[:, :4]
                        visdata = dict(cloud=cloud)

                        if detections is not None:
                            calib = self._loader.calibration_data(uidx)
                            visdata['anno_dt'] = calib.transform_objects(detections.filter_score(score_threshold), self._loader.VALID_LIDAR_NAMES[0])
                            if self._loader.phase != "testing":
                                visdata['anno_gt'] = calib.transform_objects(self._loader.annotation_3dobject(uidx),  self._loader.VALID_LIDAR_NAMES[0])
                        if segmentations is not None:
                            visdata['semantic_dt'] = segmentations['semantic_label']
                            visdata['semantic_scores'] = segmentations['semantic_scores']
                            if self._loader.phase != "testing":
                                visdata['semantic_gt'] = self._loader.annotation_3dpoints(uidx)['semantic']
                            visdata['semantic_colormap'] = [l.color for l in self._loader.VALID_PTS_CLASSES]

                        with open(osp.join(vis_path, "%06d.pkl" % idx), "wb") as fout:
                            pickle.dump(visdata, fout)

                    # save submission file for official eval
                    if dump_submission:
                        if detections is not None:
                            self._loader.dump_detection_output(uidx, detections, osp.join(sdet_path, "%06d.dump" % idx), ranges={})

                        if segmentations is not None:
                            self._loader.dump_segmentation_output(uidx, segmentations['semantic_label'], sseg_path, raw2seg=False)

                yield detections, segmentations

            if dump_prefix is not None:
                det_fout.close()
                seg_fout.close()

        if iterative:
            return parse()
        else:
            return list(parse())

    def evaluate(self,
                 results,
                 metric='bbox+segm', # list or string, box: box ap; segm: segm iou; raw: segm iou with tag not parsed
                 logger=None,
                 dump_prefix=None,
                 dump_visual=False,
                 dump_submission=False):

        if 'bbox' in metric:
            classes = [self._loader.VALID_OBJ_CLASSES[name] for name in self.CLASSES]
            deval = DetectionEvaluator(classes, 0.5, pr_sample_count=16)

        if 'segm' in metric:
            seval = SegmentationEvaluator([self._loader.VALID_PTS_CLASSES(i) for i in self.CLASSES_PTS])

        results_dict = dict()
        formatted = self.format_results(results, dump_prefix=dump_prefix, dump_visual=dump_visual, dump_submission=dump_submission, iterative=True)
        for i, (pred_det, pred_seg) in enumerate(tqdm.tqdm(formatted, total=len(results), desc="Evaluating")):
            uidx = self.data_infos[i]['uidx']

            if 'bbox' in metric:
                gt_det = self._loader.annotation_3dobject(uidx)
                calib = self._loader.calibration_data(uidx)
                stats = deval.calc_stats(gt_det, pred_det, calib)
                deval.add_stats(stats)

            if 'segm' in metric:
                gt_seg = self._loader.annotation_3dpoints(uidx)
                stats = seval.calc_stats(gt_seg.semantic, pred_seg['semantic_label'])
                seval.add_stats(stats)

        message = "Eval completed"
        if 'bbox' in metric:
            aps = []
            for k, v in deval.ap().items():
                if np.isnan(v):
                    continue
                aps.append(v)
                results_dict["AP/" + k.name] = v
            message += ", mAP: %.2f%%" % (np.mean(aps) * 100)

        if 'segm' in metric:
            ious = []
            for k, v in seval.iou().items():
                if not np.isnan(v):
                    ious.append(v)
                    results_dict["IoU/" + k.name] = v
            message += ", mIoU: %.2f%%" % (np.mean(ious) * 100)

        print(message)
        return results_dict

    def __str__(self):
        return "D3DDataset(%s), phase: %s" % (self._loader.__class__.__name__, self._loader.phase)

    def get_cat_ids(self, idx):
        if 'annos' not in self.data_infos[idx]:
            return []

        annos = self.data_infos[idx]['annos']

        cat_ids = []
        for i, box in enumerate(annos['arr']):
            if annos['num_lidar_pts'][i] == 0: # skip invalid boxes
                continue
            cenum = self._loader.VALID_OBJ_CLASSES(int(box[0]))
            if cenum.name in self.CLASSES:
                cat_ids.append(self.cat2id[cenum.name])

        return list(set(cat_ids))


def d3d_data_prep(ds_name, root_path, info_prefix, out_dir=None,
    trainval_split=0.8, inzip=False, lidar_name=0, debug=False, ninter_frames=10,
    database_save_path=None, db_generate=False, db_info_save_path=None):
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
    mmcv.dump(collect_ann_file(train_loader, lidar_name, ninter_frames=ninter_frames, debug=debug), info_path)

    # Creating info for validation set
    val_loader = ds_type(root_path, inzip=inzip, phase="validation",
        trainval_split=trainval_split, trainval_random=seed, trainval_byseq=True)
    info_path = Path(root_path, f'{info_prefix}_infos_val.pkl')
    mmcv.dump(collect_ann_file(val_loader, lidar_name, ninter_frames=ninter_frames, debug=debug), info_path)

    # Creating info for test set
    test_loader = ds_type(root_path, inzip=inzip, phase="testing")
    info_path = Path(root_path, f'{info_prefix}_infos_test.pkl')
    mmcv.dump(collect_ann_file(test_loader, lidar_name, ninter_frames=ninter_frames, debug=debug), info_path)

    # Creating dataset ground-truth sampler
    try:
        from tools.data_converter.create_gt_database import \
            create_groundtruth_database
    except ImportError:
        print("Cannot import tools. Groundtruth database won't be created!")
        return

    if not debug and db_generate:
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
                    sweeps_num=9,
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
                             trainval_split=trainval_split, trainval_random=seed, pipeline=pipeline,
                             filter_empty_gt=True, filter_ignore=True)
        create_groundtruth_database(dataset, root_path, info_prefix,
                                    database_save_path=database_save_path,
                                    db_info_save_path=db_info_save_path, with_mask_3d=True)

if __name__ == "__main__":
    d3d_data_prep("nuscenes", "data/nuscenes_d3d", "d3d_nuscenes", trainval_split="official", db_generate=True)
