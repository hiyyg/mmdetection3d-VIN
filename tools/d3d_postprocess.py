import pickle
import msgpack
import numpy as np
from itertools import repeat
from multiprocessing import Pool
from tqdm import tqdm
from d3d.box import box3dp_crop
from d3d.benchmarks import SegmentationEvaluator, DetectionEvaluator
from d3d.dataset.nuscenes import NuscenesLoader, NuscenesDetectionClass, NuscenesSegmentationClass
from d3d.abstraction import Target3DArray


def eval_frame_det(inputs):
    uidx, pred_boxes, loader, evaluator = inputs

    # load boxes
    calib = loader.calibration_data(uidx)
    gt_boxes = loader.annotation_3dobject(uidx)
    pred_boxes = Target3DArray.deserialize(pred_boxes)

    # calculate
    stats = evaluator.calc_stats(gt_boxes, pred_boxes, calib)
    return stats

def eval_detection(debug=False,
                   anno_info_path="data/nuscenes_d3d/d3d_nuscenes_infos_val.pkl",
                   result_path="work_dirs/temp/detection_results.msg",
                   dataset_path = "/mnt/cache2t/jacobz/nuscenes_converted"):

    with open(anno_info_path, "rb") as fin:
        annos = pickle.load(fin)
    with open(result_path, "rb") as fin:
        dets = msgpack.unpack(fin)
    print("results loaded.")

    loader = NuscenesLoader(dataset_path)
    evaluator = DetectionEvaluator(list(NuscenesDetectionClass)[1:], 0.5)

    if debug:
        mapping = map(eval_frame_det, zip((item['uidx'] for item in annos), dets, repeat(loader), repeat(evaluator)))
    else:
        pool = Pool(processes=15)
        mapping = pool.imap_unordered(eval_frame_det, zip((item['uidx'] for item in annos), dets))

    for i, stats in enumerate(tqdm(mapping, total=len(annos))):
        evaluator.add_stats(stats)
        if debug and i > 10:
            break

    print(evaluator.summary())

def eval_frame_seg(inputs):
    uidx, pred_boxes, pred_labels, loader, evaluator, score_threshold = inputs

    # load gt
    calib = loader.calibration_data(uidx)
    cloud = loader.lidar_data(uidx)
    gt_boxes = loader.annotation_3dobject(uidx)
    gt_boxes = calib.transform_objects(gt_boxes, "lidar_top")
    gt_labels = loader.annotation_3dpoints(uidx).semantic
    gt_ids = gt_boxes.paint_label(cloud, gt_labels)

    # load pred
    pred_boxes = Target3DArray.deserialize(pred_boxes).filter_score(score_threshold)
    pred_boxes = calib.transform_objects(pred_boxes, "lidar_top")
    pred_labels = pred_labels['semantic_label']
    pred_ids = pred_boxes.paint_label(cloud, pred_labels)

    # calculate
    assert gt_labels.shape == pred_labels.shape
    return evaluator.calc_stats(gt_labels, pred_labels, gt_ids, pred_ids)

def eval_segmentation(debug=False,
                      anno_info_path="data/nuscenes_d3d/d3d_nuscenes_infos_val.pkl",
                      det_result_path="work_dirs/temp/detection_results.msg",
                      seg_result_path="work_dirs/temp/segmentation_results.msg",
                      dataset_path = "/mnt/cache2t/jacobz/nuscenes_converted",
                      score_threshold = 0.4):
                      
    with open(anno_info_path, "rb") as fin:
        annos = pickle.load(fin)
    with open(det_result_path, "rb") as fin:
        dets = msgpack.unpack(fin)
    with open(seg_result_path, "rb") as fin:
        segs = msgpack.unpack(fin)
    print("results loaded.")

    loader = NuscenesLoader(dataset_path)
    evaluator = SegmentationEvaluator(list(NuscenesSegmentationClass), min_points=5)

    # run evaluator
    if debug:
        mapping = map(eval_frame_seg, zip((item['uidx'] for item in annos), dets, segs,
                                          repeat(loader), repeat(evaluator), repeat(score_threshold)))
    else:
        pool = Pool(processes=15)
        mapping = pool.imap_unordered(eval_frame_seg, zip((item['uidx'] for item in annos), dets, segs,
                                                          repeat(loader), repeat(evaluator), repeat(score_threshold)))

    for stats in tqdm(mapping, total=len(annos)):
        evaluator.add_stats(stats)
    print(evaluator.summary())

    # report category-wise results
    thing_cls = [NuscenesSegmentationClass(i) for i in range(1, 11)]
    stuff_cls = [NuscenesSegmentationClass(i) for i in range(11, 17)]

    msq = [evaluator.sq()[k] for k in thing_cls]
    msq = sum(msq) / len(thing_cls)
    mrq = [evaluator.rq()[k] for k in thing_cls]
    mrq = sum(mrq) / len(thing_cls)
    mpq = [evaluator.pq()[k] for k in thing_cls]
    mpq = sum(mpq) / len(thing_cls)
    print("Things SQ=%.3f, RQ=%.3f, PQ=%.3f" % (msq, mrq, mpq))

    msq = [evaluator.sq()[k] for k in stuff_cls]
    msq = sum(msq) / len(stuff_cls)
    mrq = [evaluator.rq()[k] for k in stuff_cls]
    mrq = sum(mrq) / len(stuff_cls)
    mpq = [evaluator.pq()[k] for k in stuff_cls]
    mpq = sum(mpq) / len(stuff_cls)
    print("Stuff SQ=%.3f, RQ=%.3f, PQ=%.3f" % (msq, mrq, mpq))

    mpq = [evaluator.pq()[k] for k in thing_cls] + [evaluator.iou()[k] for k in stuff_cls]
    mpq = sum(mpq) / len(thing_cls)
    print("Replaced PQ=%.3f" % mpq)

def detseg_crossfix(debug=False,
                    anno_info_path="data/nuscenes_d3d/d3d_nuscenes_infos_val.pkl",
                    det_result_path="work_dirs/temp/detection_results.msg",
                    seg_result_path="work_dirs/temp/segmentation_results.msg"):
    # TODO: output file be detection_results.fix.msg or segmentation_results.fix.msg
    pass

if __name__ == "__main__":
    # TODO: use fire?
    eval_detection()
    eval_segmentation()
    detseg_crossfix()
