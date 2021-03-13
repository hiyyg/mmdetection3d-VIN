_base_ = './centerpoint_01voxel_second_secfpn_4x8_cyclic_30e_d3d-nuscenes_semantic.py'

model=dict(
    pts_middle_encoder=dict(freeze=True),
    pts_backbone=dict(freeze=True),
    pts_neck=dict(freeze=True),
    pts_bbox_head=dict(freeze_bbox=True)
)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8)
