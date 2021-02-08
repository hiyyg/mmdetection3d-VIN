import argparse
import numpy as np
import pcl
import pickle
from addict import Dict as edict
from d3d.vis.pcl import visualize_detections

parser = argparse.ArgumentParser()
parser.add_argument("dump_file", help="Path to the file created by visualize_dump", type=str)
args = parser.parse_args()

with open(args.dump_file, "rb") as fin:
    data = edict(pickle.load(fin))

vis = pcl.Visualizer()
vp1 = vis.createViewPort(0, 0, 0.5, 1)
vp2 = vis.createViewPort(0.5, 0, 1, 1)
vis.addPointCloud(pcl.create_xyzl(np.hstack([data.cloud[:,:3], data.semantic_gt.reshape(-1,1)])), viewport=vp1, id="cloud1")
vis.addPointCloud(pcl.create_xyzl(np.hstack([data.cloud[:,:3], data.semantic_dt.reshape(-1,1)])), viewport=vp2, id="cloud2")
visualize_detections(vis, data.anno_gt.frame, data.anno_gt, None, id_prefix="gt_left", viewport=vp1)
visualize_detections(vis, data.anno_gt.frame, data.anno_gt, None, text_scale=0, id_prefix="gt_right", viewport=vp2)
visualize_detections(vis, data.anno_dt.frame, data.anno_dt, None, box_color=(1,1,0), text_color=(1,0.8,0), id_prefix="dt", viewport=vp2)
vis.setRepresentationToWireframeForAllActors()
vis.spin()
