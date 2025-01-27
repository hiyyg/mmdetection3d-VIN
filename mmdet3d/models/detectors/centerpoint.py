import torch

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d, merge_aug_ptlabels_3d
from mmdet.models import DETECTORS
from .mvx_two_stage import MVXTwoStageDetector


@DETECTORS.register_module()
class CenterPoint(MVXTwoStageDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(CenterPoint,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def forward_pts_train(self,
                          points,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          out_of_range_points=None,
                          pts_semantic_mask=None,
                          pts_of_interest_idx=None,
                          pts_instance_mask=None,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(points, pts_feats, pts_of_interest_idx=pts_of_interest_idx, out_of_range_points=out_of_range_points)

        if out_of_range_points is not None:
            out_of_range_splits = [torch.sum(poi < len(p)) for p, poi in zip(points, pts_of_interest_idx)]
        else:
            out_of_range_splits = None

        losses = self.pts_bbox_head.loss(
            gt_bboxes_3d, gt_labels_3d, outs,
            out_of_range_splits=out_of_range_splits,
            pts_of_interest_idx=pts_of_interest_idx,
            pts_semantic_mask=pts_semantic_mask)
        return losses

    def simple_test_pts(self, points, pts_feats, img_metas,
                        rescale=False, out_of_range_points=None, pts_of_interest_idx=None,
                        pts_of_interest_revidx=None):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(points, pts_feats, pts_of_interest_idx=pts_of_interest_idx, out_of_range_points=out_of_range_points)

        if getattr(self.pts_bbox_head, 'semantic_head', None) is not None:
            bbox_feat = outs[:-1]
            semantic_feat = outs[-1]
            if out_of_range_points is not None:
                points = [torch.cat([cloud, oor_cloud], dim=0) for cloud, oor_cloud in zip(points, out_of_range_points)]
            pts_results = self.pts_bbox_head.get_semantic(points, semantic_feat, pts_of_interest_revidx)
        else:
            bbox_feat = outs
            pts_results = [None] * len(img_metas)

        bbox_list = self.pts_bbox_head.get_bboxes(
            bbox_feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results, pts_results

    def aug_test_pts(self, points, pts_feats, img_metas, rescale=False, out_of_range_points=None,
                     pts_of_interest_idx=None, pts_of_interest_revidx=None):
        """Test function of point cloud branch with augmentaiton.

        The function implementation process is as follows:

            - step 1: map features back for double-flip augmentation.
            - step 2: merge all features and generate boxes.
            - step 3: map boxes back for scale augmentation.
            - step 4: merge box results.
            - step 5: merge point semantic results

        Args:
            pts_feats (list[torch.Tensor]): Feature of point cloud.
            img_metas (list[dict]): Meta information of samples.
            rescale (bool): Whether to rescale bboxes. Default: False.

        Returns:
            dict: Returned bboxes consists of the following keys:

                - boxes_3d (:obj:`LiDARInstance3DBoxes`): Predicted bboxes.
                - scores_3d (torch.Tensor): Scores of predicted boxes.
                - labels_3d (torch.Tensor): Labels of predicted boxes.
        """
        # only support aug_test for one sample
        # step 1
        outs_list = []
        pts_outs_list = [] # for semantic result

        for i, (x, img_meta) in enumerate(zip(pts_feats, img_metas)):
            outs = self.pts_bbox_head(points[i], x, pts_of_interest_idx=pts_of_interest_idx[i], out_of_range_points=out_of_range_points[i])

            if getattr(self.pts_bbox_head, 'semantic_head', None) is not None:
                semantic_feat = outs[-1]
                outs = outs[:-1]

                if out_of_range_points is not None:
                    sem_points = torch.cat([points[i], out_of_range_points[i]], dim=0)
                else:
                    sem_points = points[i]
                pts_outs_list.append(self.pts_bbox_head.get_semantic(
                    sem_points, semantic_feat, pts_of_interest_revidx[i])[0])

            # merge augmented outputs before decoding bboxes
            for task_id, out in enumerate(outs):
                for key in out[0].keys():
                    if img_meta[0]['pcd_horizontal_flip']:
                        outs[task_id][0][key] = torch.flip(
                            outs[task_id][0][key], dims=[2])
                        if key == 'reg':
                            outs[task_id][0][key][:, 1, ...] = 1 - outs[
                                task_id][0][key][:, 1, ...]
                        elif key == 'rot':
                            outs[task_id][0][
                                key][:, 1,
                                     ...] = -outs[task_id][0][key][:, 1, ...]
                        elif key == 'vel':
                            outs[task_id][0][
                                key][:, 1,
                                     ...] = -outs[task_id][0][key][:, 1, ...]
                    if img_meta[0]['pcd_vertical_flip']:
                        outs[task_id][0][key] = torch.flip(
                            outs[task_id][0][key], dims=[3])
                        if key == 'reg':
                            outs[task_id][0][key][:, 0, ...] = 1 - outs[
                                task_id][0][key][:, 0, ...]
                        elif key == 'rot':
                            outs[task_id][0][
                                key][:, 0,
                                     ...] = -outs[task_id][0][key][:, 0, ...]
                        elif key == 'vel':
                            outs[task_id][0][
                                key][:, 0,
                                     ...] = -outs[task_id][0][key][:, 0, ...]

            outs_list.append(outs)

        # step 2
        preds_dicts = dict()
        scale_img_metas = []

        # concat outputs sharing the same pcd_scale_factor
        for i, (img_meta, outs) in enumerate(zip(img_metas, outs_list)):
            pcd_scale_factor = img_meta[0]['pcd_scale_factor']
            if pcd_scale_factor not in preds_dicts.keys():
                preds_dicts[pcd_scale_factor] = outs
                scale_img_metas.append(img_meta)
            else:
                for task_id, out in enumerate(outs):
                    for key in out[0].keys():
                        preds_dicts[pcd_scale_factor][task_id][0][key] += out[
                            0][key]

        # step 3
        aug_bboxes = []

        for pcd_scale_factor, preds_dict in preds_dicts.items():
            for task_id, pred_dict in enumerate(preds_dict):
                # merge outputs with different flips before decoding bboxes
                for key in pred_dict[0].keys():
                    preds_dict[task_id][0][key] /= len(outs_list) / len(
                        preds_dicts.keys())
            bbox_list = self.pts_bbox_head.get_bboxes(
                preds_dict, img_metas[0], rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # step 4
        if len(preds_dicts.keys()) > 1:
            # merge outputs with different scales after decoding bboxes
            bbox_results = merge_aug_bboxes_3d(aug_bboxes, scale_img_metas,
                                                self.pts_bbox_head.test_cfg)
        else:
            for key in bbox_list[0].keys():
                bbox_list[0][key] = bbox_list[0][key].to('cpu')
            bbox_results = bbox_list[0]

        # step 5
        if pts_outs_list:
            pts_results = merge_aug_ptlabels_3d(pts_outs_list, img_metas, self.pts_bbox_head.test_cfg)
        else:
            pts_results = None

        return bbox_results, pts_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False, out_of_range_points=None,
                 pts_of_interest_idx=None, pts_of_interest_revidx=None):
        """Test function with augmentaiton."""
        img_feats, pts_feats = self.extract_feats(points, img_metas, imgs)
        bbox_list = dict()
        if pts_feats and self.with_pts_bbox:
            pts_bbox, pts_labels = self.aug_test_pts(points, pts_feats, img_metas, rescale, out_of_range_points,
                pts_of_interest_idx, pts_of_interest_revidx)
            bbox_list.update(pts_bbox=pts_bbox)
            bbox_list.update(pts_pointwise=pts_labels)
        return [bbox_list]
