import pickle
from re import L
import msgpack
import msgpack_numpy
from numpy.lib.arraysetops import isin, unique
msgpack_numpy.patch()

import numpy as np
import torch
from collections import namedtuple, defaultdict
from itertools import count, repeat, islice, groupby
from multiprocessing import Pool
from tqdm import tqdm, trange
from d3d.benchmarks import SegmentationEvaluator, DetectionEvaluator
from d3d.dataset.base import TrackingDatasetBase
from d3d.dataset.nuscenes import NuscenesLoader, NuscenesDetectionClass, NuscenesSegmentationClass
from d3d.abstraction import Target3DArray, TransformSet


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
                   dataset_path="data/nuscenes_d3d/",
                   min_overlap=0.5,
                   pr_sample_count=20,
                   ratio=1):

    with open(anno_info_path, "rb") as fin:
        annos = pickle.load(fin)
    with open(result_path, "rb") as fin:
        dets = msgpack.unpack(fin)
    N = len(annos)
    uidx_list = [item['uidx'] for item in annos]
    print("results loaded.")

    loader = NuscenesLoader(dataset_path)
    evaluator = DetectionEvaluator(list(NuscenesDetectionClass)[1:], min_overlap, pr_sample_count=pr_sample_count)

    if debug:
        mapping = map(eval_frame_det, zip(uidx_list, dets, repeat(loader), repeat(evaluator)))
    else:
        pool = Pool(processes=15)
        mapping = pool.imap_unordered(eval_frame_det, zip(uidx_list, dets, repeat(loader), repeat(evaluator)))

    for i, stats in enumerate(tqdm(islice(mapping, int(N*ratio)), total=int(N*ratio))):
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
    stats = evaluator.calc_stats(gt_labels, pred_labels, gt_ids, pred_ids)
    return stats

def eval_segmentation(debug=False,
                      anno_info_path="data/nuscenes_d3d/d3d_nuscenes_infos_val.pkl",
                      det_result_path="work_dirs/temp/detection_results.msg",
                      seg_result_path="work_dirs/temp/segmentation_results.msg",
                      dataset_path = "data/nuscenes_d3d",
                      score_threshold = 0.4,
                      ratio = 1):

    with open(anno_info_path, "rb") as fin:
        annos = pickle.load(fin)
    with open(det_result_path, "rb") as fin:
        dets = msgpack.unpack(fin)
    with open(seg_result_path, "rb") as fin:
        segs = msgpack.unpack(fin)
    N = len(annos)
    uidx_list = [item['uidx'] for item in annos]
    print("results loaded.")

    loader = NuscenesLoader(dataset_path)
    evaluator = SegmentationEvaluator(list(NuscenesSegmentationClass), min_points=5)

    # run evaluator
    if debug:
        mapping = map(eval_frame_seg, zip(uidx_list, dets, segs, repeat(loader), repeat(evaluator), repeat(score_threshold)))
    else:
        mapping = Pool(processes=8).imap_unordered(eval_frame_seg, zip(uidx_list, dets, segs, repeat(loader), repeat(evaluator), repeat(score_threshold)))

    for i, stats in tqdm(islice(enumerate(mapping), int(N*ratio)), total=int(N*ratio)):
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
    mpq = sum(mpq) / len(mpq)
    print("Replaced PQ=%.3f" % mpq)

def detseg_crossfix(dataset_path="data/nuscenes_d3d/",
                    anno_info_path="data/nuscenes_d3d/d3d_nuscenes_infos_val.pkl",
                    det_result_path="work_dirs/temp/detection_results.msg",
                    seg_result_path="work_dirs/temp/segmentation_results.msg",
                    debug=False, index=None,
                    point_to_box_label=True,
                    box_to_point_label=True,
                    point_to_box_scale=True,
                    score_margin=0.4,
                    count_threshold=10,
                    thing_labels=list(range(1,11)),
                    count_coeff=1, consistent_coeff=1):
    '''
    Crossfix on detection and segmentation results
    :param index: If None, then all frames are converted, if int, then only specific frame will be converted
    :param score_margin: when two box overlaps, if the one box has a higher score by this margin than another,
        then the box with lower score will be ignored when calculating
    :param count_threshold: Minimum number of points in a box for it to be considered
    :param thing_labels: Labels of thing categories
    :param count_coeff: Power index for point count of a category before normalization
    :param consistent_coeff: Power index for box score when calculate consistency score
    :param point_to_box_label: override box label from point label
    :param box_to_point_label: override point label from box label
    :param point_to_box_scale: change box dimension or generate new box according to point label
    '''
    thing_labels = set(thing_labels)
    loader = NuscenesLoader(dataset_path)

    # load results
    with open(anno_info_path, "rb") as fin:
        annos = pickle.load(fin)
        N = len(annos)
        uidx_list = [item['uidx'] for item in annos]
    with open(det_result_path, "rb") as fin:
        dets = msgpack.unpack(fin)
    with open(seg_result_path, "rb") as fin:
        segs = msgpack.unpack(fin)

    if index is None:
        idlist = list(range(N))
    elif isinstance(index, int):
        idlist = [index]
    elif isinstance(index, slice):
        idlist = list(range(N)[index])
    else:
        raise ValueError("Unexpected index type!")
    if debug:
        idlist = idlist[:2]

    # open output
    packer = msgpack.Packer(use_single_float=True)
    det_fout = open(det_result_path.replace('.msg', '.fix.msg'), 'wb')
    det_fout.write(packer.pack_array_header(len(idlist)))
    seg_fout = open(seg_result_path.replace('.msg', '.fix.msg'), 'wb')
    seg_fout.write(packer.pack_array_header(len(idlist)))

    for i in tqdm(idlist):
        uidx = uidx_list[i]
        calib = loader.calibration_data(uidx)
        cloud = loader.lidar_data(uidx)
        boxes = Target3DArray.deserialize(dets[i])
        if len(boxes) == 0:
            continue
        if debug:
            for j, b in enumerate(boxes): # tag each box for debugging
                b.tid = j+1
        boxes.sort_by_score() # score from lowest to highest
        boxes = calib.transform_objects(boxes, loader.VALID_LIDAR_NAMES[0])
        ptlabel = segs[i]['semantic_label'].copy()
        ptscore = segs[i]['semantic_scores'].copy()

        mask = boxes.crop_points(cloud)

        # pre-caculate score jump table
        score_lp = -1
        score_hp = 0
        score_map_low = [] # score_map_low[i] is the biggest index that its box score < boxes[i].score - score_margin
                           # score_map_low[i]=-1 means no box satisfies the inequality
        score_map_high = [] # score_map_high[i] is the smallest index that its box score > boxes[i].score + score_margin
                            # score_map_high[i]=len(boxes) means no box satisfies the inequality
        for box in boxes:
            score_low = box.tag_top_score - score_margin
            while boxes[score_lp+1].tag_top_score < score_low:
                score_lp += 1
            score_map_low.append(score_lp)

            score_high = box.tag_top_score + score_margin
            while score_hp < len(boxes) and boxes[score_hp].tag_top_score <= score_high:
                score_hp += 1
            score_map_high.append(score_hp)

        # mark consistent box-point label
        consistency = np.full(len(cloud), -1, dtype='i2') # index of box that correctly contains a point,
                                                          # can represent 32767 boxes at most
        for j, box in enumerate(boxes):
            imask = np.where(mask[j])[0]
            consistent_mask = ptlabel[imask] == box.tag_top.value
            consistency[imask[consistent_mask]] = j

        # try to override box label using point label
        if point_to_box_label:
            for j, box in reversed(list(enumerate(boxes))): # from highest score to lowest
                # excluding points correctly classified by other boxes (with score high enough)
                inconsistent_mask = mask[j] & ((consistency <= score_map_low[j]) | (consistency == j))
                if np.sum(inconsistent_mask) < count_threshold: # skip boxes with too few points
                    continue
                box_ptlabel = ptlabel[inconsistent_mask]
                box_ptscore = ptscore[inconsistent_mask]

                # calculate a metric
                ulabels, ucounts = np.unique(box_ptlabel, return_counts=True)
                thing_mask = np.array([l in thing_labels for l in ulabels])
                if np.sum(thing_mask) == 0: # no points of things
                    continue

                s_count = ucounts ** count_coeff # score for point count
                s_count = s_count / np.sum(s_count)
                s_prob = [np.mean(box_ptscore[box_ptlabel == l]) for l in ulabels] # score for label probabilities
                s_consistent = [(1 + box.tag_top_score**consistent_coeff) if l == box.tag_top.value else 1 for l in ulabels] # score for consistant label
                s = s_count * s_prob * s_consistent # TODO: sum or multiply?

                if debug:
                    print(f"frame{i} obj{box.tid-1}({box.tag_top_score:.3f}), original:{box.tag_top.value}, candidates:{ulabels}, scores:{s}")

                proposed = ulabels[thing_mask][np.argmax(s[thing_mask])]
                # TODO: do not change if there is a box with higher score and overlap with this box (with enough IoU)
                #       or more strategically remove label of box with higher score from possible proposals (before argmax)
                if proposed != box.tag_top.value:
                    # update consistency matrix
                    consistency[np.where(inconsistent_mask & (ptlabel == proposed))[0]] = j
                    # FIXME: old incorrect labels are preserved, it's hard to fix

                    # overwrite
                    tqdm.write(f"Fix frame{i} object with score {box.tag_top_score:.3f}: {box.tag_top.value} -> {proposed}")
                    box.tag_top = proposed

                    # TODO: increase box score if label is correct, and decrease if wrong?
                    #       suppose box score is s,
                    #           if correct then new score is 1-(1-s)*(1-a)^b
                    #           if incorrect then new score is 1-(1-s)*(1+a)^b
                    #       a is a parameter, b is how correct is the box (for example use s_count)
                    #       this score update need to be applied after all proposals to prevent reordering

        # override point label from box label
        # TODO: only override points with low score (lower than some threshold)
        #       box with lower score should not overwrite the label of points with higher score
        if box_to_point_label:
            for j, box in enumerate(boxes): # from lower score to higher score
                override_mask = mask[j].copy()
                if np.sum(override_mask) < count_threshold: # skip boxes with too few points
                    continue

                # skip points that is close to box boundary
                ipoints = np.where(override_mask)[0]
                overlaps = np.any(mask[:,ipoints], axis=1)
                for k in np.where(overlaps)[0]: # loop over overlap box
                    if k <= score_map_low[j]: # ignore boxes with low score
                        continue

                    if k == j:
                        # for this box
                        pdist = box.points_distance(cloud[override_mask])
                    else:
                        # for other box (also ignore points inside other boxes)
                        pdist = -boxes[k].points_distance(cloud[override_mask])
                    # TODO: use min dimension or z dimension? use Z we can explicitly prevent ground points to be overwrite
                    pdist_thres = max(np.min(boxes[k].dimension) * (1-boxes[k].tag_top_score), 0.1) # at least 0.1 m
                    override_mask[np.where(pdist < pdist_thres)] = False

                ptlabel[override_mask] = box.tag_top.value

        # rescale or create box from point labels
        if point_to_box_scale:
            pass # TODO: to be implemented

        det_fout.write(packer.pack(boxes.serialize()))
        seg_fout.write(packer.pack(dict(semantic_scores=ptscore, semantic_label=ptlabel)))

    det_fout.close()
    seg_fout.close()
    # TODO: add stats for what has been fixed


Tstat = namedtuple("Tstat", ['frame_id', 'obj_id', 'obj_label', 'total_count', 'lstats'])
Lstat = namedtuple("Lstat", ['label', 'count', 'mean_dist', 'std_dist', 'max_dist'])

def crossfix_in_dataset(phase='training'):
    loader = NuscenesLoader("data/nuscenes_d3d", phase=phase)
    stats = []

    for i in trange(len(loader)):
        label_box = loader.annotation_3dobject(i)
        calib = loader.calibration_data(i)
        points = loader.lidar_data(i)
        label_point = loader.annotation_3dpoints(i)

        label_box = calib.transform_objects(label_box, loader.VALID_LIDAR_NAMES[0])
        for j in range(len(label_box)):
            tag = label_box[j].tag_top.value
            mask = label_box[j].crop_points(points)
            pdist = label_box[j].points_distance(points[mask])
            check = label_point.semantic[mask] == tag
            if not np.all(check):
                err_labels = label_point.semantic[mask][~check]
                err_pdist = pdist[~check]
                lstats = []
                for l, count in zip(*np.unique(err_labels, return_counts=True)):
                    ldist = err_pdist[err_labels == l]
                    lstats.append(Lstat(l, count, np.mean(ldist), np.std(ldist), np.max(ldist)))
                stats.append(Tstat(i, j, tag, np.sum(mask), lstats))

    with open("stats.pkl", "wb") as fout:
        pickle.dump(stats, fout)

def crossfix_in_dataset_analysis(stats_path="stats.pkl", find=None, csv=False):
    stats: Tstat
    with open(stats_path, "rb") as fin:
        stats = pickle.load(fin)

    if find is None:
        dist_stat = defaultdict(list)
        ratio_stat = defaultdict(list)

        for stat in stats:
            for lstat in stat.lstats:
                lpair = stat.obj_label, lstat.label
                dist_stat[lpair].append((lstat.count, lstat.mean_dist, lstat.std_dist, lstat.max_dist))
                ratio_stat[lpair].append((lstat.count, stat.total_count))

        if csv:
            print("label, err_label, err_count, total_count, err_ratio(%), dist_mean(cm), dist_stddev(cm), dist_max1, dist_max2, dist_max3, dist_max4, dist_max5")

        for k, v in ratio_stat.items():
            collect = np.sum(np.array(v), axis=0)
            dist_arr = np.array(dist_stat[k])
            dmean = np.average(dist_arr[:,1], weights=dist_arr[:,0])
            dstd = np.sqrt(np.average(np.square(dist_arr[:,2]), weights=dist_arr[:,0]))
            dmax5 = -np.partition(-dist_arr[:,3], 5)[:5] if len(dist_arr) > 5 else dist_arr[:,3]
            if find is None:
                if csv:
                    dmax5 = (dmax5*100).tolist()
                    dmax5_str = ','.join(str(i) for i in dmax5) + ','*(5-len(dmax5))
                    print(f"{k[0]},{k[1]},{collect[0]},{collect[1]},{(collect[0] / collect[1]) * 100},{dmean*100},{dstd*100},{dmax5_str}")
                else:
                    print(f"{k[0]:2} -> {k[1]:2}: {collect[0]:7}/{collect[1]:<8} ({(collect[0] / collect[1]) * 100:6.2f}%), dist mean: {dmean*100:8.4f}, dist stddev: {dstd*100:8.4f}, dist max 5: {dmax5*100}")

        if not csv:
            print("(stats in cm)")

    else:
        collects = []
        for stat in stats:
            if find[0] != stat.obj_label:
                continue
            
            for lstat in stat.lstats:
                if find[1] != lstat.label:
                    continue
                collects.append((stat.frame_id, stat.obj_id, lstat.count))

        # clustering by scenes
        loader = NuscenesLoader("data/nuscenes_d3d")
        for k, g in groupby(collects, lambda x: loader.identity(x[0])[0]):
            carr = np.array(list(g))
            most = np.argmax(carr[:,2])
            frame_id, obj_id, count = carr[most]
            print(f"frame: {frame_id:5}, obj_id: {obj_id:3}, count: {count:4}")

if __name__ == "__main__":
    # TODO: use fire?
    # detseg_crossfix(index=slice(2000))
    # detseg_crossfix(index=2764, debug=True)

    # eval_detection(ratio=0.3, result_path="work_dirs/temp/detection_results.fix.msg")
    eval_segmentation(ratio=0.3, 
        det_result_path="work_dirs/temp/detection_results.fix.msg",
        seg_result_path="work_dirs/temp/segmentation_results.fix.msg")

    # crossfix_in_dataset(phase="validation")
    # crossfix_in_dataset_analysis(csv=True)
    # crossfix_in_dataset_analysis(find=(10, 9))