_base_ = [
    '../_base_/datasets/d3d-nuscenes-cbgs.py',
    '../_base_/models/centerpoint_01voxel_second_secfpn_nus.py',
    '../_base_/schedules/cyclic_20e.py', '../_base_/default_runtime.py'
]

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
# directly specify class mapping. 
seg_mapping = [ 0,  0,  7,  7,  7,  0,  7,  0,  0,  1,  0,  0,  8,  0,  2,  3,  3,
        4,  5,  0,  0,  6,  9, 10, 11, 12, 13, 14, 15,  0, 16,  0]
seg_nclasses = max(seg_mapping)+1 # including background
seg_mapping = [i-1 for i in seg_mapping] # valid label are all subtracted by 1 to prevent 0 as background
seg_class_ids = list(range(1, seg_nclasses)) # used to reverse mapping
# weights of class frequencies [w_i = -log(n_i/n_sum), normalized to sum=nclasses], note that last weight is for background class
seg_weights = [1.0708557589401888, 1.9958194216109202 , 1.2262594995681615 ,
               0.7594261376431927, 1.4719244483895535 , 1.751393522540997  ,
               1.3782007208888054, 1.631250726947591  , 1.2121927978851932 ,
               0.9533181555783328, 0.29144125484285666, 1.0893137743893344 ,
               0.6244533354975337, 0.6241631582586307 , 0.41868257555378247,
               0.5013047114649253, 0.0]

dataset_type = 'nuscenes'
data_root = 'data/nuscenes_d3d/'
file_client_args = dict(backend='disk')

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'd3d_nuscenes_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args),
    dict(type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_seg_3d='u1'),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        remove_close=True,
        file_client_args=file_client_args),
    dict(type='ObjectSample', db_sampler=db_sampler, sample_semantics=False),
    dict(type='PointSegClassMapping',
        valid_cat_ids=seg_mapping,
        remove_invalid=False,
        as_mapping=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(
        type='RandomFlip3D',
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d',
        'pts_semantic_mask', 'pts_of_interest_idx'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                # This step makes semantic estimation impossible for points out of range
                # TODO(zyxin): consider also taking in the whole point cloud and estimate the points
                #              out of range by the nearest voxel?
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'pts_of_interest_idx', 'pts_of_interest_revidx'])
        ])
]

model = dict(
    pts_voxel_layer=dict(point_cloud_range=point_cloud_range),
    pts_bbox_head=dict(
        bbox_coder=dict(pc_range=point_cloud_range[:2]),
        semantic_head=dict(
            type='SemanticHead',
            num_classes=seg_nclasses,
            point_cloud_range=point_cloud_range,
            mlp_channels=[256, 128, 64, 32],
            in_pts_channels=5),
        loss_semantic=dict(type="EnsembleLoss", losses=[
            dict(type="ExpLogCrossEntropyLoss", gamma=0.3, loss_weight=1),
            dict(type="ExpLogDiceLoss", gamma=0.3, loss_weight=1),
            dict(type="LovaszLoss", loss_weight=1.5)
        ], use_sigmoid=False, class_weight=seg_weights)))

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(dataset=dict(pts_classes=seg_class_ids, pipeline=train_pipeline)),
    val=dict(pts_classes=seg_class_ids, pipeline=test_pipeline),
    test=dict(pts_classes=seg_class_ids, pipeline=test_pipeline))

# model training and testing settings
train_cfg = dict(pts=dict(point_cloud_range=point_cloud_range))
test_cfg = dict(pts=dict(pc_range=point_cloud_range[:2]))

evaluation = dict(interval=1, dump_prefix='work_dirs/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_d3d-nuscenes_semantic')
