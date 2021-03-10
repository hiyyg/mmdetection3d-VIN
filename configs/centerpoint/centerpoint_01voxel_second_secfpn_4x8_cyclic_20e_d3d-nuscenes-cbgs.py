_base_ = [
    '../_base_/datasets/d3d-nuscenes-cbgs.py',
    '../_base_/models/centerpoint_01voxel_second_secfpn_nus.py',
    '../_base_/schedules/cyclic_20e.py', '../_base_/default_runtime.py'
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

model = dict(
    pts_voxel_layer=dict(point_cloud_range=point_cloud_range),
    pts_bbox_head=dict(bbox_coder=dict(pc_range=point_cloud_range[:2])))
# model training and testing settings
train_cfg = dict(pts=dict(point_cloud_range=point_cloud_range))
test_cfg = dict(pts=dict(pc_range=point_cloud_range[:2]))

evaluation = dict(interval=1)
