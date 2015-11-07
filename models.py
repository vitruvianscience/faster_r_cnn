"""
Define some convolutional nets! (we are implementing here from low-level opendeep code,
but they will also be provided as multilayer opendeep models).
"""
import numpy as np
import theano
import theano.tensor as T
from opendeep.models import Model, Conv2D, Dense, Softmax
from opendeep.models.utils import ModifyLayer, Pool2D, Noise

from anchors import generate_anchors
from bbox import bbox_transform_inv, clip_boxes, filter_boxes, nms

#######################################
# helper for constructing param dicts #
#######################################
def p_dict(prefix, model):
    return {prefix + name: param for name, param in model.get_params().items()}

##########
# VGG-19 #
##########
class VGG(Model):
    def __init__(self, inputs):
        super(VGG, self).__init__(inputs)
        # vgg conv layers
        self.conv1_1 = Conv2D(inputs=inputs, n_filters=64, filter_size=(3, 3), activation='relu', border_mode='full')
        self.conv1_2 = Conv2D(inputs=self.conv1_1, n_filters=64, filter_size=(3, 3), activation='relu', border_mode='full')
        self.pool1 = Pool2D(inputs=self.conv1_2, size=(2, 2), stride=(2, 2))

        self.conv2_1 = Conv2D(inputs=self.pool1, n_filters=128, filter_size=(3, 3), activation='relu', border_mode='full')
        self.conv2_2 = Conv2D(inputs=self.conv2_1, n_filters=128, filter_size=(3, 3), activation='relu', border_mode='full')
        self.pool2 = Pool2D(inputs=self.conv2_2, size=(2, 2), stride=(2, 2))

        self.conv3_1 = Conv2D(inputs=self.pool2, n_filters=256, filter_size=(3, 3), activation='relu', border_mode='full')
        self.conv3_2 = Conv2D(inputs=self.conv3_1, n_filters=256, filter_size=(3, 3), activation='relu', border_mode='full')
        self.conv3_3 = Conv2D(inputs=self.conv3_2, n_filters=256, filter_size=(3, 3), activation='relu', border_mode='full')
        self.pool3 = Pool2D(inputs=self.conv3_3, size=(2, 2), stride=(2, 2))

        self.conv4_1 = Conv2D(inputs=self.pool3, n_filters=512, filter_size=(3, 3), activation='relu', border_mode='full')
        self.conv4_2 = Conv2D(inputs=self.conv4_1, n_filters=512, filter_size=(3, 3), activation='relu', border_mode='full')
        self.conv4_3 = Conv2D(inputs=self.conv4_2, n_filters=512, filter_size=(3, 3), activation='relu', border_mode='full')
        self.pool4 = Pool2D(inputs=self.conv4_3, size=(2, 2), stride=(2, 2))

        self.conv5_1 = Conv2D(inputs=self.pool4, n_filters=512, filter_size=(3, 3), activation='relu', border_mode='full')
        self.conv5_2 = Conv2D(inputs=self.conv5_1, n_filters=512, filter_size=(3, 3), activation='relu', border_mode='full')
        self.conv5_3 = Conv2D(inputs=self.conv5_2, n_filters=512, filter_size=(3, 3), activation='relu', border_mode='full')
        self.pool5 = Pool2D(inputs=self.conv4_3, size=(2, 2), stride=(2, 2))

        fc6_in = self.pool5.get_outputs().flatten(2)
        dims_prod = None if any([size is None for size in self.pool5.output_size[1:]]) else np.prod(self.pool5.output_size[1:])
        fc6_in_shape = (self.pool5.output_size[0], dims_prod)
        self.fc6 = Dense(inputs=(fc6_in_shape, fc6_in), outputs=4096, activation='relu')
        fc6_drop = Noise(inputs=self.fc6, noise='dropout', noise_level=0.5)
        self.fc7 = Dense(inputs=fc6_drop, outputs=406, activation='relu')
        fc7_drop = Noise(inputs=self.fc7, noise='dropout', noise_level=0.5)
        self.fc8 = Softmax(inputs=fc7_drop, outputs=1000)

        self.output_size = self.fc8.output_size
        self.outputs = self.fc8.get_outputs()

        self.switches = fc6_drop.get_switches() + fc7_drop.get_switches()

        self.params = {}
        self.params.update(p_dict("conv1_1_", self.conv1_1))
        self.params.update(p_dict("conv1_2_", self.conv1_2))
        self.params.update(p_dict("conv2_1_", self.conv2_1))
        self.params.update(p_dict("conv2_2_", self.conv2_2))
        self.params.update(p_dict("conv3_1_", self.conv3_1))
        self.params.update(p_dict("conv3_2_", self.conv3_2))
        self.params.update(p_dict("conv3_3_", self.conv3_3))
        self.params.update(p_dict("conv4_1_", self.conv4_1))
        self.params.update(p_dict("conv4_2_", self.conv4_2))
        self.params.update(p_dict("conv4_3_", self.conv4_3))
        self.params.update(p_dict("conv5_1_", self.conv5_1))
        self.params.update(p_dict("conv5_2_", self.conv5_2))
        self.params.update(p_dict("conv5_3_", self.conv5_3))
        self.params.update(p_dict("fc6_", self.fc6))
        self.params.update(p_dict("fc7_", self.fc7))
        self.params.update(p_dict("fc8_", self.fc8))

    def get_inputs(self):
        return self.inputs[0][1]

    def get_outputs(self):
        return self.outputs

    def get_params(self):
        return self.params

    def get_switches(self):
        return self.switches

#######################
# Region proposal net #
#######################
class RPN(Model):
    def __init__(self, conv_in, im_info):
        ## inputs is a convolutional net (i.e. VGG or ZFNet) before the fully-connected layers.
        super(RPN, self).__init__(inputs=conv_in)
        in_filters = conv_in.output_size[1] # 512
        # RPN conv layers
        classes = 2
        n_anchors = 9
        min_size = 16
        anchor_size = 16
        nms_thresh = 0.7
        topN = 2000

        self.conv = Conv2D(inputs=self.inputs,
                           n_filters=in_filters, filter_size=(3, 3), stride=(1, 1), activation='relu', border_mode='full')

        self.cls_score = Conv2D(inputs=self.conv,
                                n_filters=classes*n_anchors, filter_size=(1, 1), stride=(1, 1), activation='linear', border_mode='valid')

        # need to dimshuffle/flatten it down to get the softmax class probabilities for each class of `classes`
        cls_shape = self.cls_score.get_outputs().shape
        cls_score = self.cls_score.get_outputs().reshape((cls_shape[0], classes, -1, cls_shape[3]))
        # shuffle to (classes, batch, row, col)
        cls_shuffle = cls_score.dimshuffle((1, 0, 2, 3))
        # flatten to (classes, batch*row*col)
        cls_flat = cls_shuffle.flatten(2)
        # shuffle to (batch*row*col, classes)
        cls_flat = cls_flat.dimshuffle((1, 0))
        # softmax for probability!
        cls_probs_flat = T.nnet.softmax(cls_flat)
        # now shuffle back up to 4D output from cls_score (undo what we did)
        cls_probs = cls_probs_flat.dimshuffle((1, 0)).reshape(cls_shuffle.shape)
        cls_probs = cls_probs.dimshuffle((1, 0, 2, 3))
        self.cls_probs = cls_probs.reshape(cls_shape)

        self.bbox_pred = Conv2D(inputs=self.conv,
                                n_filters=4*n_anchors, filter_size=(1, 1), stride=(1, 1), activation='linear', border_mode='valid')

        ###############
        #  1. Generate proposals from bbox deltas and shifted anchors
        ###############
        anchors = theano.shared(generate_anchors(anchor_size))
        object_probs = self.cls_probs[:, n_anchors:, :, :]
        bbox_deltas = self.bbox_pred.get_outputs()
        H, W = object_probs.shape[-2:]
        # essentially do numpy's meshgrid by tiling
        shift_x = (T.arange(0, W) * anchor_size).reshape((1, W))
        shift_y = (T.arange(0, H) * anchor_size).reshape((1, H))
        shift_x = T.tile(shift_x, (H, 1))
        shift_y = T.tile(shift_y.T, (1, W))
        shifts = T.stack([shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel()]).T
        # Enumerate all shifted anchors:
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = n_anchors
        K = shifts.shape[0]
        anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4))
        anchors = anchors.dimshuffle((1, 0, 2)).reshape((K*A, 4))
        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        bbox_deltas = bbox_deltas.dimshuffle((0, 2, 3, 1)).reshape((-1, 4))

        # Same story for the object scores:
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        scores = object_probs.dimshuffle((0, 2, 3, 1)).reshape((-1, 1))

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info[:2])

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = filter_boxes(proposals, min_size * im_info[2])
        proposals = proposals[keep, :]
        scores = scores[keep]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        order = scores.ravel().argsort()[::-1]

        proposals = proposals[order, :]
        scores = scores[order]

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 2000)
        # 8. return the top proposals (-> RoIs top)
        keep, self.updates = nms(T.concatenate([proposals, scores], axis=1), nms_thresh)
        keep = keep[:topN]
        self.proposals = proposals[keep, :]
        self.scores = scores[keep]

        self.outputs = [self.proposals, self.scores]
        # self.output_size = [self.cls_score.output_size, self.bbox_pred.output_size]

        self.params = {}
        self.params.update(p_dict("rpn_conv/3x3_", self.conv))
        self.params.update(p_dict("rpn_cls_score_", self.cls_score))
        self.params.update(p_dict("rpn_bbox_pred_", self.bbox_pred))

    def get_outputs(self):
        return self.outputs
    def get_params(self):
        return self.params
    def get_inputs(self):
        return self.inputs[1]
    def get_updates(self):
        return self.updates

###############
# ROI Pooling #
###############
class ROIPool(ModifyLayer):
    def __init__(self, conv_features, rois, size, spatial_scale):
        super(ROIPool, self).__init__(inputs=[conv_features, rois])
        pooled_height = size[0]
        pooled_width = size[1]

        def step(roi):
            roi_start_w = T.round(roi[0] * spatial_scale)
            roi_start_h = T.round(roi[1] * spatial_scale)
            roi_end_w = T.round(roi[2] * spatial_scale)
            roi_end_h = T.round(roi[3] * spatial_scale)

            roi_height = T.max(roi_end_h - roi_start_h + 1, 1)
            roi_width = T.max(roi_end_w - roi_start_w + 1, 1)

            bin_size_h = roi_height / pooled_height
            bin_size_w = roi_width / pooled_width

if __name__ == '__main__':
    images_input = T.tensor4("images")
    images_shape = (None, 3, 224, 224)
    v = VGG((images_shape, images_input))

    r = RPN(v.conv5_3, (1,3,5))
