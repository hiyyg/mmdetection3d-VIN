_base_ = [
    '../_base_/datasets/d3d-nuscenes.py',
    '../_base_/models/centerpoint_01voxel_second_secfpn_semantic_nus.py',
    '../_base_/schedules/cyclic_20e.py', '../_base_/default_runtime.py'
]

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
seg_class_ids = [1,2,3,4,5,6,7,9,12,13,14,20,21,24,25,26,29]
dataset_type = 'nuscenes'
data_root = 'data/nuscenes_d3d/'
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)
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
    sample_groups=dict( # TODO: there might be problem with this since d3d has different id order
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
        use_dim=5,
        file_client_args=file_client_args))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_seg_3d='u1'),
    dict(type='PointSegClassMapping',
        valid_cat_ids=seg_class_ids,
        remove_invalid=True),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=3,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='ObjectSample', db_sampler=db_sampler),
        # TODO: append seg label to sampled points
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d',
        'pts_semantic_mask', 'pts_semantic_idx'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=3,
        use_dim=5,
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
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type="D3DDataset",
        ds_type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'd3d_nuscenes_infos_train.pkl',
        phase='training',
        pipeline=train_pipeline,
        obj_classes=class_names,
        pts_classes=seg_class_ids,
        modality=input_modality,
        test_mode=False),
    val=dict(
        type="D3DDataset",
        ds_type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'd3d_nuscenes_infos_val.pkl',
        phase='validation',
        pipeline=test_pipeline,
        obj_classes=class_names,
        pts_classes=seg_class_ids,
        modality=input_modality,
        test_mode=True),
    test=dict(
        type="D3DDataset",
        ds_type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'd3d_nuscenes_infos_test.pkl',
        phase='testing',
        pipeline=test_pipeline,
        obj_classes=class_names,
        pts_classes=seg_class_ids,
        modality=input_modality,
        test_mode=True))

model = dict(
    pts_voxel_layer=dict(point_cloud_range=point_cloud_range),
    pts_bbox_head=dict(bbox_coder=dict(pc_range=point_cloud_range[:2])))
# model training and testing settings
train_cfg = dict(pts=dict(point_cloud_range=point_cloud_range))
test_cfg = dict(pts=dict(pc_range=point_cloud_range[:2]))

evaluation = dict(interval=1, msgfile_prefix='/home/jacobz/PointCloud/mmdetection3d/.dev_scripts')