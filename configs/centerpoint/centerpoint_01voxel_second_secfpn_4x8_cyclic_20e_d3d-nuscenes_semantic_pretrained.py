_base_ = './centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_d3d-nuscenes_semantic.py'

model=dict(
    pts_backbone=dict(freeze=True)
)
