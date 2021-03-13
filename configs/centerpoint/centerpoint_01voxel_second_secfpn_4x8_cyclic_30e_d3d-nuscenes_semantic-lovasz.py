_base_ = './centerpoint_01voxel_second_secfpn_4x8_cyclic_30e_d3d-nuscenes_semantic.py'

model = dict(
    pts_bbox_head=dict(
        loss_semantic=dict(type="EnsembleLoss", losses=[
            # dict(type="ExpLogCrossEntropyLoss", gamma=0.3, loss_weight=0.5),
            # dict(type="ExpLogDiceLoss", gamma=0.3, loss_weight=0.5),
            dict(type="CrossEntropyLoss", loss_weight=0.5),
            dict(type="LovaszLoss", loss_weight=1)
        ], use_sigmoid=False, class_weight=seg_weights)))

evaluation = dict(interval=1)
