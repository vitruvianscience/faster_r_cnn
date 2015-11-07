# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# modified for Theano by Markus Beissinger
# --------------------------------------------------------

import theano.tensor as T
from theano import scan
from theano.scan_module import until

def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = T.log(gt_widths / ex_widths)
    targets_dh = T.log(gt_heights / ex_heights)

    targets = T.stack((targets_dx, targets_dy, targets_dw, targets_dh)).T
    return targets

def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return T.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths.dimshuffle(0,'x') + ctr_x.dimshuffle(0,'x')
    pred_ctr_y = dy * heights.dimshuffle(0,'x') + ctr_y.dimshuffle(0,'x')
    pred_w = T.exp(dw) * widths.dimshuffle(0,'x')
    pred_h = T.exp(dh) * heights.dimshuffle(0,'x')

    pred_boxes = T.zeros_like(deltas, dtype=deltas.dtype)
    # x1
    pred_boxes = T.set_subtensor(pred_boxes[:, 0::4], pred_ctr_x - 0.5 * pred_w)
    # y1
    pred_boxes = T.set_subtensor(pred_boxes[:, 1::4], pred_ctr_y - 0.5 * pred_h)
    # x2
    pred_boxes = T.set_subtensor(pred_boxes[:, 2::4], pred_ctr_x + 0.5 * pred_w)
    # y2
    pred_boxes = T.set_subtensor(pred_boxes[:, 3::4], pred_ctr_y + 0.5 * pred_h)

    return pred_boxes

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    # x1 >= 0
    boxes = T.set_subtensor(boxes[:, 0::4], T.maximum(T.minimum(boxes[:, 0::4], im_shape[1] - 1), 0))
    # y1 >= 0
    boxes = T.set_subtensor(boxes[:, 1::4], T.maximum(T.minimum(boxes[:, 1::4], im_shape[0] - 1), 0))
    # x2 < im_shape[1]
    boxes = T.set_subtensor(boxes[:, 2::4], T.maximum(T.minimum(boxes[:, 2::4], im_shape[1] - 1), 0))
    # y2 < im_shape[0]
    boxes = T.set_subtensor(boxes[:, 3::4], T.maximum(T.minimum(boxes[:, 3::4], im_shape[0] - 1), 0))
    return boxes

def filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    # keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    keep = (T.ge(ws, min_size) & T.ge(hs, min_size)).nonzero()[0]
    return keep

def nms(dets, thresh):
    """NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    def step(ord):
        i = ord[0]

        xx1 = T.maximum(x1[i], x1[ord[1:]])
        yy1 = T.maximum(y1[i], y1[ord[1:]])
        xx2 = T.minimum(x2[i], x2[ord[1:]])
        yy2 = T.minimum(y2[i], y2[ord[1:]])

        w = T.maximum(0.0, xx2 - xx1 + 1)
        h = T.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[ord[1:]] - inter)

        inds = T.le(ovr, thresh).nonzero()[0]
        ord = ord[inds + 1]

        return (i, ord), until(order.size > 0)

    (keep, _), updates = scan(fn=step, outputs_info=[None, order], n_steps=500000)

    return keep, updates