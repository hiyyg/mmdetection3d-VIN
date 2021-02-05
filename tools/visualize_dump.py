import pickle
import msgpack
import os.path as osp
import numpy as np
import argparse
from d3d.abstraction import Target3DArray
from d3d.benchmarks import DetectionEvaluator
from d3d.dataset.kitti import KittiObjectClass, KittiObjectLoader
from d3d.dataset.nuscenes import NuscenesLoader, NuscenesDetectionClass
from d3d.vis.pcl import visualize_detections

parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="Dataset type", type=str)
parser.add_argument("dataset_path", help="Dataset root path", type=str)
parser.add_argument("info_path", help="Path to the annotation info", type=str)
parser.add_argument("output_path", help="Output directory for detection.msg and segmentation.pkl")
parser.add_argument("-z", "--inzip", help="Whether dataset in zip", action="store_true")
parser.add_argument("-i", "--index", help="Specify the index of frame to be visualized", type=int, default=0)
parser.add_argument("-s", "--score", help="Score threshold of the objects to be visualized", type=float, default=0.4)
args = parser.parse_args()

if args.dataset == "kitti":
    loader = KittiObjectLoader(args.dataset_path, inzip=args.inzip)
if args.dataset == "nuscenes":
    loader = NuscenesLoader(args.dataset_path, inzip=args.inzip)
else:
    raise NotImplementedError("Unsupported d3d dataset type!")

with open(osp.join(args.output_path, "detection_results.msg"), "rb") as fin:
    det_results = msgpack.unpackb(fin.read())
with open(osp.join(args.output_path, "segmentation_results.pkl"), "rb") as fin:
    sem_results = pickle.load(fin)
with open(args.info_path, "rb") as fin:
    anno_info = pickle.load(fin)

lidar_frame = loader.VALID_LIDAR_NAMES[0]
uidx = anno_info[args.index]["uidx"]
anno_dt = Target3DArray.deserialize(det_results[args.index]).filter_score(args.score)
anno_gt = loader.annotation_3dobject(uidx)
calib = loader.calibration_data(uidx)
anno_dt = calib.transform_objects(anno_dt, lidar_frame)
anno_gt = calib.transform_objects(anno_gt, lidar_frame)
cloud = loader.lidar_data(uidx)[:, :4]

with open(osp.join(args.output_path, "vis.pkl"), "wb") as fout:
    pickle.dump(dict(
        cloud=cloud,
        semantic_dt=sem_results[args.index]['semantic_label'],
        semantic_gt=loader.annotation_3dpoints(anno_info[args.index]["uidx"])['semantic'],
        semantic_scores=sem_results[args.index]['semantic_scores'],
        anno_gt=anno_gt,
        anno_dt=anno_dt
    ), fout)
print("Visualization context saved! Use visualize_show.py to show the results.")
