_base_ = './centerpoint_01voxel_second_secfpn_4x8_cyclic_30e_d3d-nuscenes_semantic.py'

model = dict(
    pts_middle_encoder=dict(
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 64, 128), (128, 128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, 0, [0, 1, 1]), (0, 0, 0)),
    ),
    pts_bbox_head=dict(
        semantic_head=dict(
            mlp_channels=[256, 128, 64, 32, 32, 32]
        )
    )
)

evaluation = dict(interval=1)
