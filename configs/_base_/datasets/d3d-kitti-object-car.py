# dataset settings
dataset_type = 'kitti'
data_root = 'data/kitti/'
class_names = ['Car']
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
input_modality = dict(use_lidar=True, use_camera=False)
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(filter_by_difficulty=[-1], filter_by_min_points=dict(Car=5)),
    classes=class_names,
    sample_groups=dict(Car=15))

file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel', path_mapping=dict(data='s3://kitti_data/'))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        file_client_args=file_client_args),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0.5],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.78539816, 0.78539816]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
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
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

data = dict(
    samples_per_gpu=5,
    workers_per_gpu=4,
    train=dict(
        type='D3DDataset',
        ds_type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'd3d_kitti_infos_train.pkl',
        phase='training',
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        obj_classes=class_names),
    val=dict(
        type='D3DDataset',
        ds_type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'd3d_kitti_infos_val.pkl',
        phase='validation',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        obj_classes=class_names),
    test=dict(
        type='D3DDataset',
        ds_type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'd3d_kitti_infos_test.pkl',
        phase='testing',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        obj_classes=class_names)
)

evaluation = dict(interval=1)
