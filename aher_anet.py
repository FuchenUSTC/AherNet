import math 
from collections import namedtuple

import numpy as np 
import tensorflow as tf

import tf_extended as tfe
import custom_layers
import aher_common
import tf_utils


AHERParams = namedtuple('AHERParameters', ['temporal_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_size_bounds',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'normalizations',
                                         'prior_scaling'
                                         ])

class AHERNet(object):
    """Implementation of the AHER network for action detection
    """
    default_params = AHERParams(
        temporal_shape=512,
        num_classes=200,
        no_annotation_label=21,
        feat_layers=['conv_a1', 'conv_a2', 'conv_a3', 'conv_a4', 'conv_a5', 'conv_a6', 'conv_a7', 'conv_a8'],
        feat_shapes=[(128,1), (64,1), (32,1), (16,1), (8,1), (4,1), (2,1), (1,1)],
        anchor_size_bounds=[0.15, 0.90],
        anchor_sizes=[(4),
                      (8),
                      (16),
                      (32),
                      (64),
                      (128),
                      (256),
                      (512)],
        anchor_ratios=[[1.260, 1.587],
                       [1.260, 1.587],
                       [1.260, 1.587],
                       [1.260, 1.587],
                       [1.260, 1.587],
                       [1.260, 1.587],
                       [1.260, 1.587],
                       [1.260, 1.587]],
        anchor_steps=[4, 8, 16, 32, 64, 128, 256, 512],
        anchor_offset=0.5,
        normalizations=[-1, -1, -1, -1, -1, -1, -1, -1],
        prior_scaling=[0.1, 0.1]
        )

    def __init__(self, params=None):
        """Init the AHER net with some parameters. Use the default ones
        if none provided.
        """
        if isinstance(params, AHERParams):
            self.params = params
        else:
            self.params = AHERNet.default_params

    # ======================================================================= #
    def net(self, inputs,
            is_training=True,
            update_feat_shapes=True,
            dropout_keep_prob=0.5,
            prediction_fn=tf.contrib.layers.softmax,
            reuse=None,
            scope='aher'):
        """AHER 1D network definition.
        """
        r = aher_net(inputs,
                    num_classes=self.params.num_classes,
                    feat_layers=self.params.feat_layers,
                    anchor_sizes=self.params.anchor_sizes,
                    anchor_ratios=self.params.anchor_ratios,
                    normalizations=self.params.normalizations,
                    is_training=is_training,
                    dropout_keep_prob=dropout_keep_prob,
                    prediction_fn=prediction_fn,
                    reuse=reuse,
                    scope=scope)
        # Update feature shapes (try at least!)
        if update_feat_shapes:
            shapes = aher_feat_shapes_from_net(r[0], self.params.feat_shapes)
            self.params = self.params._replace(feat_shapes=shapes)
        return r

    # only predict the recognition
    def net_cls(self, inputs,
            is_training=True,
            num_classes=87,
            update_feat_shapes=True,
            dropout_keep_prob=0.5,
            prediction_fn=tf.contrib.layers.softmax,
            reuse=None,
            scope='aher'):
        """AHER 1D network definition.
        """
        r = aher_net_cls(inputs,
                    num_classes=num_classes,
                    feat_layers=self.params.feat_layers,
                    anchor_sizes=self.params.anchor_sizes,
                    anchor_ratios=self.params.anchor_ratios,
                    normalizations=self.params.normalizations,
                    is_training=is_training,
                    dropout_keep_prob=dropout_keep_prob,
                    prediction_fn=prediction_fn,
                    reuse=reuse,
                    scope=scope)
        # Update feature shapes (try at least!)
        if update_feat_shapes:
            shapes = aher_feat_shapes_from_net(r[0], self.params.feat_shapes)
            self.params = self.params._replace(feat_shapes=shapes)
        return r

    # only predict the recognition with anchor pooling
    def net_pool_cls(self, inputs,
            is_training=True,
            num_classes=200,
            untrim_num = 87,
            start_label_id = 0,
            update_feat_shapes=True,
            dropout_keep_prob=0.5,
            prediction_fn=tf.contrib.layers.softmax,
            reuse=None,
            cls_suffix = '_anchor',
            scope='aher'):
        """AHER network definition.
        """
        r = aher_net_pool_cls(inputs,
                    num_classes=num_classes,
                    untrim_num=untrim_num,
                    start_label_id = start_label_id,
                    feat_layers=self.params.feat_layers,
                    anchor_sizes=self.params.anchor_sizes,
                    anchor_ratios=self.params.anchor_ratios,
                    normalizations=self.params.normalizations,
                    is_training=is_training,
                    dropout_keep_prob=dropout_keep_prob,
                    prediction_fn=prediction_fn,
                    reuse=reuse,
                    cls_suffix=cls_suffix,
                    scope=scope)
        # Update feature shapes (try at least!)
        if update_feat_shapes:
            shapes = aher_feat_shapes_from_net(r[0], self.params.feat_shapes)
            self.params = self.params._replace(feat_shapes=shapes)
        return r    

    # predict proposal
    def net_prop(self, inputs,
            clsweights,
            is_training=True,
            num_classes=87,
            update_feat_shapes=True,
            dropout_keep_prob=0.5,
            prediction_fn=tf.contrib.layers.softmax,
            reuse=None,
            scope='aher'):
        """AHER network definition.
        """
        r = aher_net_prop(inputs,clsweights,
                    num_classes=num_classes,
                    feat_layers=self.params.feat_layers,
                    anchor_sizes=self.params.anchor_sizes,
                    anchor_ratios=self.params.anchor_ratios,
                    normalizations=self.params.normalizations,
                    is_training=is_training,
                    dropout_keep_prob=dropout_keep_prob,
                    prediction_fn=prediction_fn,
                    reuse=reuse,
                    scope=scope)
        # Update feature shapes (try at least!)
        if update_feat_shapes:
            shapes = aher_feat_shapes_from_net(r[0], self.params.feat_shapes)
            self.params = self.params._replace(feat_shapes=shapes)
        return r

    # predict proposal with iou
    def net_prop_iou(self, inputs,
            clsweights,
            clsbias,
            is_training=True,
            num_classes=87,
            update_feat_shapes=True,
            dropout_keep_prob=0.5,
            prediction_fn=tf.contrib.layers.softmax,
            reuse=None,
            cls_suffix='_anchor',
            scope='aher'):
        """AHER network definition.
        """
        r = aher_net_prop_iou(inputs,clsweights,clsbias,
                    num_classes=num_classes,
                    feat_layers=self.params.feat_layers,
                    anchor_sizes=self.params.anchor_sizes,
                    anchor_ratios=self.params.anchor_ratios,
                    normalizations=self.params.normalizations,
                    is_training=is_training,
                    dropout_keep_prob=dropout_keep_prob,
                    prediction_fn=prediction_fn,
                    reuse=reuse,
                    cls_suffix=cls_suffix,
                    scope=scope)
        # Update feature shapes (try at least!)
        if update_feat_shapes:
            shapes = aher_feat_shapes_from_net(r[0], self.params.feat_shapes)
            self.params = self.params._replace(feat_shapes=shapes)
        return r

    # predict proposal with iou
    def net_prop_iou_th(self, inputs,
            clsweights,
            clsbias,
            is_training=True,
            num_classes=20,
            update_feat_shapes=True,
            dropout_keep_prob=0.5,
            prediction_fn=tf.contrib.layers.softmax,
            reuse=None,
            scope='aher'):
        """AHER network definition.
        """
        r = aher_net_prop_iou_th(inputs,clsweights,clsbias,
                    num_classes=num_classes,
                    feat_layers=self.params.feat_layers,
                    anchor_sizes=self.params.anchor_sizes,
                    anchor_ratios=self.params.anchor_ratios,
                    normalizations=self.params.normalizations,
                    is_training=is_training,
                    dropout_keep_prob=dropout_keep_prob,
                    prediction_fn=prediction_fn,
                    reuse=reuse,
                    scope=scope)
        # Update feature shapes (try at least!)
        if update_feat_shapes:
            shapes = aher_feat_shapes_from_net(r[0], self.params.feat_shapes)
            self.params = self.params._replace(feat_shapes=shapes)
        return r

    # predict proposal with iou without transfer weight/bias version (input dimension 2048)
    def net_prop_iou_pure(self, inputs,
            is_training=True,
            num_classes=200,
            update_feat_shapes=True,
            dropout_keep_prob=0.5,
            prediction_fn=tf.contrib.layers.softmax,
            reuse=None,
            cls_suffix='anet',
            scope='aher'):
        """AHER network definition.
        """
        r = aher_net_prop_iou_pure(inputs,
                    num_classes=num_classes,
                    feat_layers=self.params.feat_layers,
                    anchor_sizes=self.params.anchor_sizes,
                    anchor_ratios=self.params.anchor_ratios,
                    normalizations=self.params.normalizations,
                    is_training=is_training,
                    dropout_keep_prob=dropout_keep_prob,
                    prediction_fn=prediction_fn,
                    reuse=reuse,
                    cls_suffix = cls_suffix,
                    scope=scope)
        # Update feature shapes (try at least!)
        if update_feat_shapes:
            shapes = aher_feat_shapes_from_net(r[0], self.params.feat_shapes)
            self.params = self.params._replace(feat_shapes=shapes)
        return r

    # detection net with single class
    def net_detect(self, inputs,
            is_training=True,
            num_classes=200,
            update_feat_shapes=True,
            dropout_keep_prob=0.5,
            prediction_fn=tf.contrib.layers.softmax,
            reuse=None,
            scope='aher'):
        """AHER network definition.
        """
        r = aher_net_detect(inputs,
                    num_classes=num_classes,
                    feat_layers=self.params.feat_layers,
                    anchor_sizes=self.params.anchor_sizes,
                    anchor_ratios=self.params.anchor_ratios,
                    normalizations=self.params.normalizations,
                    is_training=is_training,
                    dropout_keep_prob=dropout_keep_prob,
                    prediction_fn=prediction_fn,
                    reuse=reuse,
                    scope=scope)
        # Update feature shapes (try at least!)
        if update_feat_shapes:
            shapes = aher_feat_shapes_from_net(r[0], self.params.feat_shapes)
            self.params = self.params._replace(feat_shapes=shapes)
        return r

    # predict all in depth-wise 1d-conv backbone
    def net_depth(self, inputs,
            is_training=True,
            update_feat_shapes=True,
            dropout_keep_prob=0.5,
            prediction_fn=tf.contrib.layers.softmax,
            reuse=None,
            scope='aher'):
        """AHER network definition.
        """
        r = aher_depthconv1d_net(inputs,
                    num_classes=self.params.num_classes,
                    feat_layers=self.params.feat_layers,
                    anchor_sizes=self.params.anchor_sizes,
                    anchor_ratios=self.params.anchor_ratios,
                    normalizations=self.params.normalizations,
                    is_training=is_training,
                    dropout_keep_prob=dropout_keep_prob,
                    prediction_fn=prediction_fn,
                    reuse=reuse,
                    scope=scope)
        # Update feature shapes (try at least!)
        if update_feat_shapes:
            shapes = aher_feat_shapes_from_net(r[0], self.params.feat_shapes)
            self.params = self.params._replace(feat_shapes=shapes)
        return r

    # predict all in original 1d-conv
    def net_whole(self, inputs, 
            num_class = 200,
            is_training=True,
            update_feat_shapes=True,
            dropout_keep_prob=0.5,
            prediction_fn=tf.contrib.layers.softmax,
            reuse=None,
            scope='aher'):
        """AHER network definition.
        """
        r = aher_net_whole(inputs,
                    num_classes=num_class,
                    feat_layers=self.params.feat_layers,
                    anchor_sizes=self.params.anchor_sizes,
                    anchor_ratios=self.params.anchor_ratios,
                    normalizations=self.params.normalizations,
                    is_training=is_training,
                    dropout_keep_prob=dropout_keep_prob,
                    prediction_fn=prediction_fn,
                    reuse=reuse,
                    scope=scope)
        # Update feature shapes (try at least!)
        if update_feat_shapes:
            shapes = aher_feat_shapes_from_net(r[0], self.params.feat_shapes)
            self.params = self.params._replace(feat_shapes=shapes)
        return r 

    # ======================================================================= #
    def update_feature_shapes(self, predictions):
        """Update feature shapes from predictions collection (Tensor or Numpy
        array).
        """
        shapes = aher_feat_shapes_from_net(predictions, self.params.feat_shapes)
        self.params = self.params._replace(feat_shapes=shapes)

    def anchors(self, temporal_shape, dtype=np.float32):
        """Compute the default anchor boxes, given an image shape.
        """
        return aher_anchors_all_layers(temporal_shape,
                                      self.params.feat_shapes,
                                      self.params.anchor_sizes,
                                      self.params.anchor_ratios,
                                      self.params.anchor_steps,
                                      self.params.anchor_offset,
                                      dtype)

    def anchors_classwise(self, temporal_shape, dtype=np.float32):
        """Compute the default anchor boxes, given an image shape.
        """
        return aher_anchors_all_layers(temporal_shape,
                                      self.params.feat_shapes,
                                      self.params.anchor_sizes,
                                      [[],[],[],[],[],[],[],[]],
                                      self.params.anchor_steps,
                                      self.params.anchor_offset,
                                      dtype)

    def bboxes_encode(self, labels, bboxes, anchors,
                      scope=None):
        """Encode labels and bounding boxes.
        """
        #bboxes_n = tf.reshape(tf.boolean_mask(bboxes,tf.not_equal([-1.0,-1.0],bboxes)),[-1,2])
        #bboxes_n = tf.reshape(bboxes,[-1,None,2])
        bboxes_n = bboxes
        #bboxes_n = bboxes_n / self.default_params.temporal_shape

        #labels = tf.Print(labels, [labels], 'The input labels: ')
        #bboxes_n = tf.Print(bboxes_n, [bboxes_n], 'The bounding box: ')
        #anchors =  tf.Print(anchors, [anchors[0]], 'The anchors box: ')

        return aher_common.tf_aher_bboxes_encode(
            labels, bboxes_n, anchors,
            self.params.num_classes,
            self.params.temporal_shape,
            matching_threshold=0.70,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    def bboxes_decode(self, feat_localizations, duration, anchors,
                      scope='aher_bboxes_decode'):
        """Encode labels and bounding boxes.
        """
        return aher_common.tf_aher_bboxes_decode(
            feat_localizations, duration, anchors,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    def detected_bboxes(self, predictions, localisations,
                        select_threshold=None, nms_threshold=0.5,
                        clipping_bbox=None, top_k=400, keep_top_k=200, iou_flag=False):
        """Get the detected bounding boxes from the AHER network output.
        """
        binary_num_classes = 2
        # Select top_k bboxes from predictions, and clip

        rscores, rbboxes = \
            aher_common.tf_aher_bboxes_select(predictions, localisations,
                                            select_threshold=select_threshold,
                                            #num_classes=self.params.num_classes)
                                            num_classes = binary_num_classes,
                                            IoU_flag = iou_flag)
        rscores, rbboxes = \
            tfe.bboxes_sort(rscores, rbboxes, top_k=top_k)
        # Apply NMS algorithm.
        rscores, rbboxes = \
            tfe.bboxes_nms_batch(rscores, rbboxes,
                                 nms_threshold=nms_threshold,
                                 keep_top_k=keep_top_k)
        #if clipping_bbox is not None:
        #    rbboxes = tfe.bboxes_clip(clipping_bbox, rbboxes)
        return rscores, rbboxes

    def detected_bboxes_classwise(self, predictions, localisations,
                        select_threshold=None, nms_threshold=0.5,
                        clipping_bbox=None, top_k=400, keep_top_k=200, iou_flag=False):
        """Get the detected bounding boxes from the AHER network output.
        """
        anet_num_classes = 2
        # Select top_k bboxes from predictions, and clip

        rscores, rbboxes = \
            aher_common.tf_aher_bboxes_select(predictions, localisations,
                                            select_threshold=select_threshold,
                                            num_classes = anet_num_classes,
                                            IoU_flag = iou_flag)
        rscores, rbboxes = \
            tfe.bboxes_sort(rscores, rbboxes, top_k=top_k)
        # Apply NMS algorithm.
        rscores, rbboxes = \
            tfe.bboxes_nms_batch(rscores, rbboxes,
                                 nms_threshold=nms_threshold,
                                 keep_top_k=keep_top_k)
        #if clipping_bbox is not None:
        #    rbboxes = tfe.bboxes_clip(clipping_bbox, rbboxes)
        return rscores, rbboxes

    def losses(self, logits, localisations,iouprediction,
               gclasses, glocalisations, gscores, giou,
               match_threshold=0.3,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               cls_weights=1.0,
               iou_weights=1.0,
               scope='aher_losses'):
        """Define the AHER network losses.
        """
        #return aher_losses(logits, localisations,iouprediction,
        return aher_losses_classwise(
                          logits, localisations,iouprediction,
                          gclasses, glocalisations, gscores, giou,
                          match_threshold=match_threshold,
                          negative_ratio=negative_ratio,
                          alpha=alpha,
                          label_smoothing=label_smoothing,
                          cls_weights=cls_weights,
                          iou_weights=iou_weights,
                          scope=scope)

    def losses_complete(self, logits, localisations, 
               proplogits, iouprediction,
               gclasses, glocalisations, gscores, giou,
               match_threshold=0.3,
               neg_match_threshold=0.3,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               cls_weights=1.0,
               iou_weights=1.0,
               scope='aher_losses',
               anchor_position_wise=False):

        return aher_losses_classwise_iou(
                          logits, localisations, proplogits,iouprediction,
                          gclasses, glocalisations, gscores, giou,
                          match_threshold=match_threshold,
                          negative_match_threshold = neg_match_threshold,
                          negative_ratio=negative_ratio,
                          alpha=alpha,
                          label_smoothing=label_smoothing,
                          cls_weights=cls_weights,
                          iou_weights=iou_weights,
                          anchor_position_wise=anchor_position_wise,
                          scope=scope)

    def losses_complete_mome(self, logits, localisations, 
               proplogits, iouprediction,
               gclasses, glocalisations, gscores, giou,
               match_threshold=0.3,
			   neg_match_threshold=0.3,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               cls_weights=1.0,
               iou_weights=1.0,
               scope='aher_losses',
               anchor_position_wise=False):

        return aher_losses_mome_iou(
                          logits, localisations, proplogits,iouprediction,
                          gclasses, glocalisations, gscores, giou,
                          match_threshold=match_threshold,
						  negative_match_threshold = neg_match_threshold,
                          negative_ratio=negative_ratio,
                          alpha=alpha,
                          label_smoothing=label_smoothing,
                          cls_weights=cls_weights,
                          iou_weights=iou_weights,
                          anchor_position_wise=anchor_position_wise,
                          scope=scope)

    def losses_detect_s(self, localisations, 
               proplogits, iouprediction,
               gclasses, glocalisations, gscores, giou,
               match_threshold=0.3,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               cls_weights=1.0,
               iou_weights=1.0,
               scope='aher_losses',
               anchor_position_wise=False):

        return aher_losses_detect_s(
                          localisations, proplogits,iouprediction,
                          gclasses, glocalisations, gscores, giou,
                          match_threshold=match_threshold,
                          negative_ratio=negative_ratio,
                          alpha=alpha,
                          label_smoothing=label_smoothing,
                          cls_weights=cls_weights,
                          iou_weights=iou_weights,
                          anchor_position_wise=anchor_position_wise,
                          scope=scope)

    def cls_losses(self, logits,gclasses,
                gscores,
                match_threshold=0.3,
                negative_ratio=3.,
                alpha=1.,
                label_smoothing=0.,
                cls_weights=1.0,
                iou_weights=1.0,
                scope='cls_losses'):

        return aher_losses_cls(
                          logits,
                          gclasses,
                          gscores,
                          match_threshold=match_threshold,
                          negative_ratio=negative_ratio,
                          alpha=alpha,
                          label_smoothing=label_smoothing,
                          cls_weights=cls_weights,
                          iou_weights=iou_weights,
                          scope=scope)                

    def reg_losses(self, logits,gclasses,
                match_threshold=0.3,
                negative_ratio=3.,
                alpha=1.,
                label_smoothing=0.,
                cls_weights=1.0,
                iou_weights=1.0,
                scope='reg_losses'):

        return aher_losses_reg(
                          logits,
                          gclasses,
                          match_threshold=match_threshold,
                          negative_ratio=negative_ratio,
                          alpha=alpha,
                          label_smoothing=label_smoothing,
                          cls_weights=cls_weights,
                          iou_weights=iou_weights,
                          scope=scope)

    def bboxes_decode_logits(self, feat_localizations, duration, anchors, logitsprediction,
                      scope='aher_bboxes_decode'):
        """Encode labels and bounding boxes.
        """
        return aher_common.tf_aher_bboxes_decode_logits(
            feat_localizations, duration, anchors,logitsprediction,
            prior_scaling=self.params.prior_scaling,
            scope=scope)
    
    def bboxes_decode_detection(self, feat_localizations, duration, anchors, propprediction,
                      scope='aher_bboxes_decode'):
        """Encode labels and bounding boxes.
        """
        return aher_common.tf_aher_bboxes_decode_detect(
            feat_localizations, duration, anchors,propprediction,
            prior_scaling=self.params.prior_scaling,
            scope=scope)


# =========================================================================== #
# AHER tools...
# =========================================================================== #
def aher_size_bounds_to_values(size_bounds,
                               n_feat_layers,
                               temporal_size=512):
    """Compute the reference sizes of the anchor boxes from relative bounds.
    The absolute values are measured in pixels, based on the network
    default size (512 feature length).

    Return:
      list of list containing the absolute sizes at each scale. For each scale,
      the ratios only apply to the first value.
    """

    min_ratio = int(size_bounds[0] * 100)
    max_ratio = int(size_bounds[1] * 100)
    step = int(math.floor((max_ratio - min_ratio) / (n_feat_layers - 2)))
    # Start with the following smallest sizes.
    sizes = [[img_size * size_bounds[0] / 2, img_size * size_bounds[0]]]
    for ratio in range(min_ratio, max_ratio + 1, step):
        sizes.append((img_size * ratio / 100.,
                      img_size * (ratio + step) / 100.))
    return sizes

def aher_feat_shapes_from_net(predictions, default_shapes=None):
    """Try to obtain the feature shapes from the prediction layers. The latter
    can be either a Tensor or Numpy ndarray.

    Return:
      list of feature shapes. Default values if predictions shape not fully
      determined.
    """
    feat_shapes = []
    for l in predictions:
        # Get the shape, from either a np array or a tensor.
        if isinstance(l, np.ndarray):
            shape = l.shape
        else:
            shape = l.get_shape().as_list()
        shape = shape[1:4]
        # Problem: undetermined shape...
        if None in shape:
            return default_shapes
        else:
            feat_shapes.append(shape)
    return feat_shapes

def aher_anchor_one_layer(temporal_shape,
                         feat_shape,
                         sizes,
                         ratios,
                         step,
                         offset=0.5,
                         dtype=np.float32):
    """Computer AHER default anchor boxes for one feature layer.
    Determine the relative position grid of the centers, and the relative
    width and height.

    Arguments:
      feat_shape: Feature shape, used for computing relative position grids;
      size: Absolute reference sizes;
      ratios: Ratios to use on these features;
      temporal_shape: Temporal length
      offset: Grid offset.

    Return:
       y, t: the center location y and temporal length t
    """
    y = np.mgrid[0:feat_shape[0]]
    y = (y.astype(dtype) + offset) * step
    y = np.expand_dims(y, axis=-1)

    num_anchors = 1 + len(ratios)
    t = np.zeros((num_anchors, ), dtype=dtype)
    # Add first anchor boxes with ratio=1.
    t[0] = sizes
    di = 1
    for i, r in enumerate(ratios):
        t[i+di] = r * sizes 
    return y, t

def aher_anchors_all_layers(temporal_shape,
                           layers_shape,
                           anchor_sizes,
                           anchor_ratios,
                           anchor_steps,
                           offset=0.5,
                           dtype=np.float32):
    """Compute anchor boxes for all feature layers.
    """
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = aher_anchor_one_layer(temporal_shape, s,
                                             anchor_sizes[i],
                                             anchor_ratios[i],
                                             anchor_steps[i],
                                             offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors

# =========================================================================== #
# Functional definition of AHER Layer 
# =========================================================================== #

def tensor_shape(x, rank=3):
    """Returns the dimensions of a tensor.
    Args:
      image: A N-D Tensor of shape.
    Returns:
      A list of dimensions. Dimensions that are statically known are python
        integers,otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]

def aher_multibox_layer(inputs,
                       num_classes,
                       sizes,
                       ratios=[1],
                       normalization=-1,
                       bn_normalization=False):
    """Construct a multibox layer, return a class and localization predictions.
    """

    num_classes = 2 # (binary)

    net = inputs
    #if normalization > 0:
    #    net = custom_layers.l2_normalization(net, scaling=True)
    # Number of anchors.
    
    num_anchors = 1 + len(ratios)

    # Class prediction.
    num_cls_pred = num_anchors * num_classes
    cls_pred = tf.contrib.layers.conv1d(net, num_cls_pred, 3, activation_fn=None,
                           scope='conv_cls')
    cls_pred = custom_layers.channel_to_last(cls_pred)
    cls_pred = tf.reshape(cls_pred,
                          tensor_shape(cls_pred, 4)[:-1]+[num_anchors, num_classes])

    # Location.
    num_loc_pred = num_anchors * 2
    loc_pred = tf.contrib.layers.conv1d(net, num_loc_pred, 3, activation_fn=None,
                           scope='conv_loc')
    loc_pred = custom_layers.channel_to_last(loc_pred)
    loc_pred = tf.reshape(loc_pred,
                          tensor_shape(loc_pred, 4)[:-1]+[num_anchors, 2])

    # IoU prediction
    num_iou_pred = num_anchors
    iou_pred = tf.contrib.layers.conv1d(net, num_iou_pred, 3, activation_fn=None,
                           scope='conv_iou')
    iou_pred = custom_layers.channel_to_last(iou_pred)
    iou_pred = tf.reshape(iou_pred,
                          tensor_shape(iou_pred, 4)[:-1]+[num_anchors, 1])


    return cls_pred, loc_pred, iou_pred

def aher_multibox_classwise_layer(
                       inputs,
                       num_classes,
                       sizes,
                       ratios=[1],
                       normalization=-1,
                       bn_normalization=False):
    """Construct a multibox layer, return a class and localization predictions.
    """

    num_classes = 200 # (binary)

    net = inputs
    #if normalization > 0:
    #    net = custom_layers.l2_normalization(net, scaling=True)
    # Number of anchors.
    
    num_anchors = 1 + len(ratios)

    # Class prediction.
    num_cls_pred = 1 * num_classes
    cls_pred = tf.contrib.layers.conv1d(net, num_cls_pred, 3, activation_fn=None,
                           scope='conv_action')
    cls_pred = custom_layers.channel_to_last(cls_pred)
    cls_pred = tf.reshape(cls_pred,
                          tensor_shape(cls_pred, 4)[:-1]+[1, num_classes])

    # Location.
    num_loc_pred = num_anchors * 2
    loc_pred = tf.contrib.layers.conv1d(net, num_loc_pred, 3, activation_fn=None,
                           scope='conv_loc')
    loc_pred = custom_layers.channel_to_last(loc_pred)
    loc_pred = tf.reshape(loc_pred,
                          tensor_shape(loc_pred, 4)[:-1]+[num_anchors, 2])

    # Proposal prediction
    num_prop_pred = num_anchors * 2
    prop_pred = tf.contrib.layers.conv1d(net, num_prop_pred, 3, activation_fn=None,
                           scope='conv_cls')
    prop_pred = custom_layers.channel_to_last(prop_pred)
    prop_pred = tf.reshape(prop_pred,
                          tensor_shape(prop_pred, 4)[:-1]+[num_anchors, 2])


    return cls_pred, loc_pred, prop_pred

def aher_multibox_classwise_iou_layer(
                       inputs,
                       num_classes,
                       sizes,
                       ratios=[1],
                       normalization=-1,
                       bn_normalization=False,
                       cls_suffix=''):
    """Construct a multibox layer, return a class and localization predictions.
    """

    net = inputs

    # Number of anchors.
    num_anchors = 1 + len(ratios)

    cls_scope_name = 'conv_action' + cls_suffix
    # Class prediction.
    num_cls_pred = 1 * num_classes
    cls_pred = tf.contrib.layers.conv1d(net, num_cls_pred, 3, activation_fn=None,
                           scope=cls_scope_name)
    cls_pred = custom_layers.channel_to_last(cls_pred)
    cls_pred = tf.reshape(cls_pred,
                          tensor_shape(cls_pred, 4)[:-1]+[1, num_classes])

    # Location.
    num_loc_pred = num_anchors * 2
    loc_pred = tf.contrib.layers.conv1d(net, num_loc_pred, 3, activation_fn=None,
                           scope='conv_loc')
    loc_pred = custom_layers.channel_to_last(loc_pred)
    loc_pred = tf.reshape(loc_pred,
                          tensor_shape(loc_pred, 4)[:-1]+[num_anchors, 2])

    # Prop prediction
    num_prop_pred = num_anchors * 2
    prop_pred = tf.contrib.layers.conv1d(net, num_prop_pred, 3, activation_fn=None,
                           scope='conv_cls')
    prop_pred = custom_layers.channel_to_last(prop_pred)
    prop_pred = tf.reshape(prop_pred,
                          tensor_shape(prop_pred, 4)[:-1]+[num_anchors, 2])

    # IoU prediction
    num_iou_pred = num_anchors * 1
    iou_pred = tf.contrib.layers.conv1d(net, num_iou_pred, 3, activation_fn=None,
                           scope='conv_iou')
    iou_pred = custom_layers.channel_to_last(iou_pred)
    iou_pred = tf.reshape(iou_pred,
                          tensor_shape(iou_pred, 4)[:-1]+[num_anchors, 1])

    return cls_pred, loc_pred, prop_pred, iou_pred

def aher_mutibox_single_classification_layer(inputs,
                       num_classes,
                       sizes,
                       ratios=[1],
                       normalization=-1,
                       bn_normalization=False):
    """Construct a multibox layer, return a class and localization predictions.
    """
    net = inputs

    # Class prediction.
    num_cls_pred = 1 * num_classes
    cls_pred = tf.contrib.layers.conv1d(net, num_cls_pred, 1, activation_fn=None,
                           scope='conv_action')
    cls_pred = custom_layers.channel_to_last(cls_pred)
    cls_pred = tf.reshape(cls_pred,
                          tensor_shape(cls_pred, 4)[:-1]+[1, num_classes])
    cls_weights = tf.get_variable('conv_action/weights')
    cls_weights = tf.transpose(cls_weights,perm=(2,1,0))
    cls_weights = tf.reshape(cls_weights,shape=(num_classes,-1))
    return cls_pred,cls_weights

def aher_mutibox_single_classification_layer_with_pool(
                       inputs,
                       num_classes,
                       untrim_num,
                       sizes,
                       ratios=[1],
                       normalization=-1,
                       bn_normalization=False,
                       start_label_id=0,
                       cls_suffix='_anchor'):
    """Construct a multibox layer, return a class and localization predictions.
    """
    net = inputs
    
    lshape = tfe.get_shape(inputs, 3) # NWC
    num_anchors = 1 + len(ratios)

    # Class prediction.
    num_cls_pred = 1 * num_classes
    cls_name = 'conv_action' + cls_suffix
    netp = tf.layers.average_pooling1d(net,lshape[1],1,padding='valid',name='ave_pool')
    cls_pred = tf.contrib.layers.conv1d(netp, num_cls_pred, 1, activation_fn=None,
                           scope=cls_name)
    cls_pred = custom_layers.channel_to_last(cls_pred)
    cls_pred = tf.reshape(cls_pred,
                          tensor_shape(cls_pred, 4)[:-1]+[1, num_classes])
    cls_weights = tf.get_variable('%s/weights'%(cls_name))
    cls_bias = tf.get_variable('%s/biases'%(cls_name))
    cls_weights = tf.transpose(cls_weights,perm=(2,1,0))
    cls_weights = tf.reshape(cls_weights,shape=(num_classes,-1))
    cls_bias = tf.reshape(cls_bias,shape=[num_classes,-1])
    cls_weights = cls_weights[start_label_id:start_label_id+untrim_num,:]
    cls_bias = cls_bias[start_label_id:start_label_id+untrim_num,:]

    # proposal prediction.
    num_prop_pred = num_anchors * 2
    prop_pred = tf.contrib.layers.conv1d(net, num_prop_pred, 3, activation_fn=None,
                           scope='conv_cls')
    prop_pred = custom_layers.channel_to_last(prop_pred)
    prop_pred = tf.reshape(prop_pred,
                          tensor_shape(prop_pred, 4)[:-1]+[num_anchors, 2])

    # Location.
    num_loc_pred = num_anchors * 2
    loc_pred = tf.contrib.layers.conv1d(net, num_loc_pred, 3, activation_fn=None,
                           scope='conv_loc')
    loc_pred = custom_layers.channel_to_last(loc_pred)
    loc_pred = tf.reshape(loc_pred,
                          tensor_shape(loc_pred, 4)[:-1]+[num_anchors, 2])

    # IoU prediction
    num_iou_pred = num_anchors * 1
    iou_pred = tf.contrib.layers.conv1d(net, num_iou_pred, 3, activation_fn=None,
                           scope='conv_iou')
    iou_pred = custom_layers.channel_to_last(iou_pred)
    iou_pred = tf.reshape(iou_pred,
                          tensor_shape(iou_pred, 4)[:-1]+[num_anchors, 1]) 

    return cls_pred, cls_weights, cls_bias, loc_pred, prop_pred, iou_pred

def aher_mutibox_prediction_proposal_layer(inputs,
                       clsweights,
                       num_classes,
                       sizes,
                       ratios=[1],
                       normalization=-1,
                       bn_normalization=False):
    """Construct a multibox layer, return a class and localization predictions.
    """
    net = inputs

    num_anchors = 1 + len(ratios)

    cshape = tfe.get_shape(clsweights, 2)

    # weight prediction
    action_weight = tf.contrib.layers.fully_connected(clsweights,cshape[1])
    action_weight = tf.transpose(action_weight,perm=(1,0))
    action_weight = tf.reshape(action_weight,shape=(1,cshape[1],num_classes))

    # Class prediction.
    #num_cls_pred = 1 * num_classes
    cls_pred = tf.nn.conv1d(net, action_weight, 1, 'SAME', name='conv_action')
    cls_pred = custom_layers.channel_to_last(cls_pred)
    cls_pred = tf.reshape(cls_pred,
                          tensor_shape(cls_pred, 4)[:-1]+[1, num_classes])
    # proposal prediction.
    num_prop_pred = num_anchors * 2
    prop_pred = tf.contrib.layers.conv1d(net, num_prop_pred, 3, activation_fn=None,
                           scope='conv_cls')
    prop_pred = custom_layers.channel_to_last(prop_pred)
    prop_pred = tf.reshape(prop_pred,
                          tensor_shape(prop_pred, 4)[:-1]+[num_anchors, 2])

    # Location.
    num_loc_pred = num_anchors * 2
    loc_pred = tf.contrib.layers.conv1d(net, num_loc_pred, 3, activation_fn=None,
                           scope='conv_loc')
    loc_pred = custom_layers.channel_to_last(loc_pred)
    loc_pred = tf.reshape(loc_pred,
                          tensor_shape(loc_pred, 4)[:-1]+[num_anchors, 2])

    return cls_pred, loc_pred, prop_pred

def aher_mutibox_prediction_proposal_iou_layer(inputs,
                       clsweights,clsbias,
                       num_classes,
                       sizes,
                       ratios=[1],
                       normalization=-1,
                       bn_normalization=False):
    """Construct a multibox layer, return a class and localization predictions.
    """
    net = inputs

    num_anchors = 1 + len(ratios)

    cshape = tfe.get_shape(clsweights, 2)
    
    # pure iou classification for activityNet
    # Class prediction.
    num_cls_pred = 1 * num_classes
    cls_pred = tf.contrib.layers.conv1d(net, num_cls_pred, 3, activation_fn=None,
                           scope='conv_action_anet')
    cls_pred = custom_layers.channel_to_last(cls_pred)
    cls_pred = tf.reshape(cls_pred,
                          tensor_shape(cls_pred, 4)[:-1]+[1, num_classes])


    # proposal prediction.
    num_prop_pred = num_anchors * 2
    prop_pred = tf.contrib.layers.conv1d(net, num_prop_pred, 3, activation_fn=None,
                           scope='conv_cls')
    prop_pred = custom_layers.channel_to_last(prop_pred)
    prop_pred = tf.reshape(prop_pred,
                          tensor_shape(prop_pred, 4)[:-1]+[num_anchors, 2])

    # Location.
    num_loc_pred = num_anchors * 2
    loc_pred = tf.contrib.layers.conv1d(net, num_loc_pred, 3, activation_fn=None,
                           scope='conv_loc')
    loc_pred = custom_layers.channel_to_last(loc_pred)
    loc_pred = tf.reshape(loc_pred,
                          tensor_shape(loc_pred, 4)[:-1]+[num_anchors, 2])

    # IoU prediction
    num_iou_pred = num_anchors * 1
    iou_pred = tf.contrib.layers.conv1d(net, num_iou_pred, 3, activation_fn=None,
                           scope='conv_iou')
    iou_pred = custom_layers.channel_to_last(iou_pred)
    iou_pred = tf.reshape(iou_pred,
                          tensor_shape(iou_pred, 4)[:-1]+[num_anchors, 1])

    return cls_pred, loc_pred, prop_pred, iou_pred

def aher_mutibox_prediction_proposal_iou_layer_anet(inputs,
                       clsweights,clsbias,
                       num_classes,
                       sizes,
                       ratios=[1],
                       normalization=-1,
                       bn_normalization=False,
                       cls_suffix='_anchor'):
    """Construct a multibox layer, return a class and localization predictions.
    """
    net = inputs

    num_anchors = 1 + len(ratios)

    cshape = tfe.get_shape(clsweights, 2)

    # Identity weight prediction
    action_weight_1 = tf.contrib.layers.fully_connected(clsweights,cshape[1],weights_initializer=tf.initializers.identity(),activation_fn=tf.nn.elu,scope='fully1')
    action_weight = tf.contrib.layers.fully_connected(action_weight_1,cshape[1],weights_initializer=tf.initializers.identity(),activation_fn=tf.nn.elu,scope='fully2')
    action_weight = tf.transpose(action_weight,perm=(1,0))
    action_weight = tf.reshape(action_weight,shape=(1,cshape[1],num_classes))

    # Indentity bias prediction
    action_bias_1 = tf.contrib.layers.fully_connected(clsbias,1,weights_initializer=tf.initializers.identity(),activation_fn=tf.nn.elu,scope='fully3')
    action_bias = tf.contrib.layers.fully_connected(action_bias_1,1,weights_initializer=tf.initializers.identity(),activation_fn=tf.nn.elu,scope='fully4')
    action_bias = tf.reshape(action_bias,[cshape[0]])

    # Class prediction.

    cls_name = 'conv_action' + cls_suffix
    num_cls_pred = 1 * num_classes
    cls_pred = tf.nn.conv1d(net, action_weight, 1, 'SAME', name=cls_name)
    cls_pred = tf.nn.bias_add(cls_pred, action_bias)
    cls_pred = custom_layers.channel_to_last(cls_pred)
    cls_pred = tf.reshape(cls_pred,
                          tensor_shape(cls_pred, 4)[:-1]+[1, num_classes])
    
    # proposal prediction.
    num_prop_pred = num_anchors * 2
    prop_pred = tf.contrib.layers.conv1d(net, num_prop_pred, 3, activation_fn=None,
                           scope='conv_cls')
    prop_pred = custom_layers.channel_to_last(prop_pred)
    prop_pred = tf.reshape(prop_pred,
                          tensor_shape(prop_pred, 4)[:-1]+[num_anchors, 2])

    # Location.
    num_loc_pred = num_anchors * 2
    loc_pred = tf.contrib.layers.conv1d(net, num_loc_pred, 3, activation_fn=None,
                           scope='conv_loc')
    loc_pred = custom_layers.channel_to_last(loc_pred)
    loc_pred = tf.reshape(loc_pred,
                          tensor_shape(loc_pred, 4)[:-1]+[num_anchors, 2])

    # IoU prediction
    num_iou_pred = num_anchors * 1
    iou_pred = tf.contrib.layers.conv1d(net, num_iou_pred, 3, activation_fn=None,
                           scope='conv_iou')
    iou_pred = custom_layers.channel_to_last(iou_pred)
    iou_pred = tf.reshape(iou_pred,
                          tensor_shape(iou_pred, 4)[:-1]+[num_anchors, 1])

    return cls_pred, loc_pred, prop_pred, iou_pred

def aher_mutibox_detection_layer(inputs,
                       num_classes,
                       sizes,
                       ratios=[1],
                       normalization=-1,
                       bn_normalization=False):
    """Construct a multibox layer, return a class and localization predictions.
    """
    net = inputs

    num_anchors = 1 + len(ratios)

    # proposal prediction.
    num_prop_pred = num_anchors * 200
    prop_pred = tf.contrib.layers.conv1d(net, num_prop_pred, 3, activation_fn=None,
                           scope='conv_cls')
    prop_pred = custom_layers.channel_to_last(prop_pred)
    prop_pred = tf.reshape(prop_pred,
                          tensor_shape(prop_pred, 4)[:-1]+[num_anchors, 200])

    # Location.
    num_loc_pred = num_anchors * 2
    loc_pred = tf.contrib.layers.conv1d(net, num_loc_pred, 3, activation_fn=None,
                           scope='conv_loc')
    loc_pred = custom_layers.channel_to_last(loc_pred)
    loc_pred = tf.reshape(loc_pred,
                          tensor_shape(loc_pred, 4)[:-1]+[num_anchors, 2])

    # IoU prediction
    num_iou_pred = num_anchors * 1
    iou_pred = tf.contrib.layers.conv1d(net, num_iou_pred, 3, activation_fn=None,
                           scope='conv_iou')
    iou_pred = custom_layers.channel_to_last(iou_pred)
    iou_pred = tf.reshape(iou_pred,
                          tensor_shape(iou_pred, 4)[:-1]+[num_anchors, 1])

    return loc_pred, prop_pred, iou_pred

def aher_multibox_adv_layer(inputs,
                       num_classes,
                       sizes,
                       ratios=[1],
                       normalization=-1,
                       bn_normalization=False):
    """Construct a multibox layer, return a class and localization predictions.
    """

    num_classes = 2 # (binary)
    net = inputs
    num_anchors = 1 + len(ratios)

    # IoU prediction
    num_adv_pred = num_anchors
    adv_pred = tf.contrib.layers.conv1d(net, num_adv_pred, 3, activation_fn=None,
                           scope='d_conv_adv')
    adv_pred = custom_layers.channel_to_last(adv_pred)
    adv_pred = tf.reshape(adv_pred,
                          tensor_shape(adv_pred, 4)[:-1]+[num_anchors, 1])


    return adv_pred

def aher_net(inputs,
            num_classes=AHERNet.default_params.num_classes,
            feat_layers=AHERNet.default_params.feat_layers,
            anchor_sizes=AHERNet.default_params.anchor_sizes,
            anchor_ratios=AHERNet.default_params.anchor_ratios,
            normalizations=AHERNet.default_params.normalizations,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=tf.contrib.layers.softmax,
            reuse=None,
            #scope='aher'
            scope='aher_anet'):
    """AHER net definition.
    """
    # End_points collect relevant activations for external use.
    end_points = {}

    with tf.variable_scope(scope, 'aher_anet', [inputs], reuse=reuse):
        net = tf.contrib.layers.conv1d(inputs, 4096, 3, scope = 'conv1')
        net = tf.contrib.layers.conv1d(net, 2048, 3, scope = 'conv2')
        net = tf.layers.max_pooling1d(net,3,2,name='pool2')
    
        net = tf.contrib.layers.conv1d(net, 512, 3, 2, scope='conv_a1')
        end_points['conv_a1'] = net
        net = tf.contrib.layers.conv1d(net, 512, 3, 2, scope='conv_a2')
        end_points['conv_a2'] = net
        net = tf.contrib.layers.conv1d(net, 1024, 3, 2, scope='conv_a3')
        end_points['conv_a3'] = net
        net = tf.contrib.layers.conv1d(net, 1024, 3, 2, scope='conv_a4')
        end_points['conv_a4'] = net
        net = tf.contrib.layers.conv1d(net, 2048, 3, 2, scope='conv_a5')
        end_points['conv_a5'] = net
        net = tf.contrib.layers.conv1d(net, 2048, 3, 2, scope='conv_a6')
        end_points['conv_a6'] = net
        net = tf.contrib.layers.conv1d(net, 4096, 3, 2, scope='conv_a7')
        end_points['conv_a7'] = net
        net = tf.contrib.layers.conv1d(net, 4096, 3, 2, scope='conv_a8')
        end_points['conv_a8'] = net        

        # Prediction and localisations layers.
        predictions = []
        logits = []
        localisations = []
        proplogits = []
        proppredictions=[]
        for i, layer in enumerate(feat_layers):
            with tf.variable_scope(layer + '_box',reuse=reuse):
                p, l, prop = aher_multibox_classwise_layer(
                                          end_points[layer],
                                          num_classes,
                                          anchor_sizes[i],
                                          anchor_ratios[i],
                                          normalizations[i])
            predictions.append(prediction_fn(p))
            logits.append(p)
            localisations.append(l)
            proplogits.append(prop)
            proppredictions.append(prediction_fn(prop))
        return predictions, localisations, logits, proplogits, proppredictions, end_points

# predict recognition, localization, iou, proposal
def aher_net_whole(inputs,
            num_classes=200,
            feat_layers=AHERNet.default_params.feat_layers,
            anchor_sizes=AHERNet.default_params.anchor_sizes,
            anchor_ratios=AHERNet.default_params.anchor_ratios,
            normalizations=AHERNet.default_params.normalizations,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=tf.contrib.layers.softmax,
            reuse=None,
            scope='aher'):
    """AHER net definition.
    """
    # End_points collect relevant activations for external use.
    end_points = {}

    with tf.variable_scope(scope, 'aher', [inputs], reuse=reuse):
        net = tf.contrib.layers.conv1d(inputs, 4096, 3, scope = 'g_conv1')
        net = tf.contrib.layers.conv1d(net, 2048, 3, scope = 'g_conv2')
        net = tf.layers.max_pooling1d(net,3,2,name='pool2')
    
        net = tf.contrib.layers.conv1d(net, 512, 3, 2, scope='g_conv_a1')
        end_points['conv_a1'] = net
        net = tf.contrib.layers.conv1d(net, 512, 3, 2, scope='g_conv_a2')
        end_points['conv_a2'] = net
        net = tf.contrib.layers.conv1d(net, 1024, 3, 2, scope='g_conv_a3')
        end_points['conv_a3'] = net
        net = tf.contrib.layers.conv1d(net, 1024, 3, 2, scope='g_conv_a4')
        end_points['conv_a4'] = net
        net = tf.contrib.layers.conv1d(net, 2048, 3, 2, scope='g_conv_a5')
        end_points['conv_a5'] = net
        net = tf.contrib.layers.conv1d(net, 2048, 3, 2, scope='g_conv_a6')
        end_points['conv_a6'] = net
        net = tf.contrib.layers.conv1d(net, 4096, 3, 2, scope='g_conv_a7')
        end_points['conv_a7'] = net
        net = tf.contrib.layers.conv1d(net, 4096, 3, 2, scope='g_conv_a8')
        end_points['conv_a8'] = net        

        # Prediction and localisations layers.
        predictions = []
        logits = []
        localisations = []
        proplogits = []
        proppredictions=[]
        iouprediction = []
        for i, layer in enumerate(feat_layers):
            with tf.variable_scope(layer + '_box',reuse=reuse):
                p, l, prop, iou = aher_multibox_classwise_iou_layer(
                                          end_points[layer],
                                          num_classes,
                                          anchor_sizes[i],
                                          anchor_ratios[i],
                                          normalizations[i])
            predictions.append(prediction_fn(p))
            logits.append(p)
            localisations.append(l)
            proplogits.append(prop)
            proppredictions.append(prediction_fn(prop))
            iouprediction.append(tf.math.sigmoid(iou))
        return predictions, localisations, logits, proplogits, proppredictions, iouprediction, end_points

def aher_net_v2(inputs,
            num_classes=AHERNet.default_params.num_classes,
            feat_layers=AHERNet.default_params.feat_layers,
            anchor_sizes=AHERNet.default_params.anchor_sizes,
            anchor_ratios=AHERNet.default_params.anchor_ratios,
            normalizations=AHERNet.default_params.normalizations,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=tf.contrib.layers.softmax,
            reuse=None,
            scope='aher'):
    """AHER net definition.
    """
    # End_points collect relevant activations for external use.
    end_points = {}

    with tf.variable_scope(scope, 'aher', [inputs], reuse=reuse):
        # base feature network
        net = tf.contrib.layers.conv1d(inputs, 4096, 3, scope = 'g_conv1')
        net = tf.contrib.layers.conv1d(net, 2048, 3, scope = 'g_conv2')
        net = tf.layers.max_pooling1d(net,3,2,name='pool2')
    
        net = tf.contrib.layers.conv1d(net, 512, 3, 2, scope='g_conv_a1')
        end_points['conv_a1'] = net
        net = tf.contrib.layers.conv1d(net, 512, 3, 2, scope='g_conv_a2')
        end_points['conv_a2'] = net
        net = tf.contrib.layers.conv1d(net, 1024, 3, 2, scope='g_conv_a3')
        end_points['conv_a3'] = net
        net = tf.contrib.layers.conv1d(net, 1024, 3, 2, scope='g_conv_a4')
        end_points['conv_a4'] = net
        net = tf.contrib.layers.conv1d(net, 2048, 3, 2, scope='g_conv_a5')
        end_points['conv_a5'] = net
        net = tf.contrib.layers.conv1d(net, 2048, 3, 2, scope='g_conv_a6')
        end_points['conv_a6'] = net
        net = tf.contrib.layers.conv1d(net, 4096, 3, 2, scope='g_conv_a7')
        end_points['conv_a7'] = net
        net = tf.contrib.layers.conv1d(net, 4096, 3, 2, scope='g_conv_a8')
        end_points['conv_a8'] = net


        # Prediction and localisations layers.
        predictions = []
        logits = []
        localisations = []
        iouprediction = []
        for i, layer in enumerate(feat_layers):
            with tf.variable_scope(layer + '_box',reuse=reuse):
                p, l, iou = aher_multibox_layer(
                                          end_points[layer],
                                          num_classes,
                                          anchor_sizes[i],
                                          anchor_ratios[i],
                                          normalizations[i])
            predictions.append(prediction_fn(p))
            logits.append(p)
            localisations.append(l)
            iouprediction.append(tf.math.sigmoid(iou))
        return predictions, localisations, logits, iouprediction, end_points

def aher_net_cls(inputs,
            num_classes=AHERNet.default_params.num_classes,
            feat_layers=AHERNet.default_params.feat_layers,
            anchor_sizes=AHERNet.default_params.anchor_sizes,
            anchor_ratios=AHERNet.default_params.anchor_ratios,
            normalizations=AHERNet.default_params.normalizations,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=tf.contrib.layers.softmax,
            reuse=None,
            scope='aher'):
    """AHER net definition.
    """
    # End_points collect relevant activations for external use.
    end_points = {}

    with tf.variable_scope(scope, 'aher', [inputs], reuse=reuse):
        # base feature network
        net = tf.contrib.layers.conv1d(inputs, 4096, 3, scope = 'g_conv1')
        net = tf.contrib.layers.conv1d(net, 2048, 3, scope = 'g_conv2')
        net = tf.layers.max_pooling1d(net,3,2,name='pool2')
    
        net = tf.contrib.layers.conv1d(net, 512, 3, 2, scope='g_conv_a1')
        end_points['conv_a1'] = net
        net = tf.contrib.layers.conv1d(net, 512, 3, 2, scope='g_conv_a2')
        end_points['conv_a2'] = net
        net = tf.contrib.layers.conv1d(net, 1024, 3, 2, scope='g_conv_a3')
        end_points['conv_a3'] = net
        net = tf.contrib.layers.conv1d(net, 1024, 3, 2, scope='g_conv_a4')
        end_points['conv_a4'] = net
        net = tf.contrib.layers.conv1d(net, 2048, 3, 2, scope='g_conv_a5')
        end_points['conv_a5'] = net
        net = tf.contrib.layers.conv1d(net, 2048, 3, 2, scope='g_conv_a6')
        end_points['conv_a6'] = net
        net = tf.contrib.layers.conv1d(net, 4096, 3, 2, scope='g_conv_a7')
        end_points['conv_a7'] = net
        net = tf.contrib.layers.conv1d(net, 4096, 3, 2, scope='g_conv_a8')
        end_points['conv_a8'] = net


        # Prediction and localisations layers.
        predictions = []
        logits = []
        clsweights = []
        for i, layer in enumerate(feat_layers):
            with tf.variable_scope(layer + '_box',reuse=tf.AUTO_REUSE):
                p, w = aher_mutibox_single_classification_layer(
                                          end_points[layer],
                                          num_classes,
                                          anchor_sizes[i],
                                          anchor_ratios[i],
                                          normalizations[i])
            predictions.append(prediction_fn(p))
            logits.append(p)
            clsweights.append(w)
        return predictions, logits, clsweights, end_points

def aher_net_pool_cls(inputs,
            num_classes=AHERNet.default_params.num_classes,
            untrim_num=87,
            start_label_id=0,
            feat_layers=AHERNet.default_params.feat_layers,
            anchor_sizes=AHERNet.default_params.anchor_sizes,
            anchor_ratios=AHERNet.default_params.anchor_ratios,
            normalizations=AHERNet.default_params.normalizations,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=tf.contrib.layers.softmax,
            reuse=None,
            cls_suffix='_anchor',
            scope='aher'):
    """AHER net definition.
    """
    # End_points collect relevant activations for external use.
    end_points = {}

    with tf.variable_scope(scope, 'aher', [inputs], reuse=reuse):
        # base feature network
        net = tf.contrib.layers.conv1d(inputs, 2048, 3, scope = 'g_conv1')
        net = tf.contrib.layers.conv1d(net, 1024, 3, scope = 'g_conv2')
        net = tf.layers.max_pooling1d(net,3,2,name='pool2')
    
        net = tf.contrib.layers.conv1d(net, 256, 3, 2, scope='g_conv_a1')
        end_points['conv_a1'] = net
        net = tf.contrib.layers.conv1d(net, 256, 3, 2, scope='g_conv_a2')
        end_points['conv_a2'] = net
        net = tf.contrib.layers.conv1d(net, 512, 3, 2, scope='g_conv_a3')
        end_points['conv_a3'] = net
        net = tf.contrib.layers.conv1d(net, 512, 3, 2, scope='g_conv_a4')
        end_points['conv_a4'] = net
        net = tf.contrib.layers.conv1d(net, 1024, 3, 2, scope='g_conv_a5')
        end_points['conv_a5'] = net
        net = tf.contrib.layers.conv1d(net, 1024, 3, 2, scope='g_conv_a6')
        end_points['conv_a6'] = net
        net = tf.contrib.layers.conv1d(net, 2048, 3, 2, scope='g_conv_a7')
        end_points['conv_a7'] = net
        net = tf.contrib.layers.conv1d(net, 2048, 3, 2, scope='g_conv_a8')
        end_points['conv_a8'] = net

        # Prediction and localisations layers.
        predictions = []
        logits = []
        clsweights = []
        clsbias = []
        localisations = []
        proplogits = []
        proppredictions = []
        iouprediction = []        
        for i, layer in enumerate(feat_layers):
            with tf.variable_scope(layer + '_box',reuse=tf.AUTO_REUSE):
                p, w, wb, l, prop, iou  = aher_mutibox_single_classification_layer_with_pool(
                                          end_points[layer],
                                          num_classes,
                                          untrim_num,
                                          anchor_sizes[i],
                                          anchor_ratios[i],
                                          normalizations[i],
                                          start_label_id=start_label_id,
                                          cls_suffix=cls_suffix)
            predictions.append(prediction_fn(p))
            logits.append(p)
            clsweights.append(w)
            clsbias.append(wb)
            localisations.append(l)
            proplogits.append(prop)
            proppredictions.append(prediction_fn(prop))
            iouprediction.append(tf.math.sigmoid(iou))
        return predictions, localisations, logits, proplogits, proppredictions, iouprediction, clsweights, clsbias, end_points

def aher_net_prop(inputs,clsweights,
            num_classes=AHERNet.default_params.num_classes,
            feat_layers=AHERNet.default_params.feat_layers,
            anchor_sizes=AHERNet.default_params.anchor_sizes,
            anchor_ratios=AHERNet.default_params.anchor_ratios,
            normalizations=AHERNet.default_params.normalizations,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=tf.contrib.layers.softmax,
            reuse=None,
            scope='aher'):
    """AHER net definition.
    """
    # End_points collect relevant activations for external use.
    end_points = {}

    with tf.variable_scope(scope, 'aher', [inputs], reuse=reuse):
        # base feature network
        net = tf.contrib.layers.conv1d(inputs, 4096, 3, scope = 'g_conv1')
        net = tf.contrib.layers.conv1d(net, 2048, 3, scope = 'g_conv2')
        net = tf.layers.max_pooling1d(net,3,2,name='pool2')
    
        net = tf.contrib.layers.conv1d(net, 512, 3, 2, scope='g_conv_a1')
        end_points['conv_a1'] = net
        net = tf.contrib.layers.conv1d(net, 512, 3, 2, scope='g_conv_a2')
        end_points['conv_a2'] = net
        net = tf.contrib.layers.conv1d(net, 1024, 3, 2, scope='g_conv_a3')
        end_points['conv_a3'] = net
        net = tf.contrib.layers.conv1d(net, 1024, 3, 2, scope='g_conv_a4')
        end_points['conv_a4'] = net
        net = tf.contrib.layers.conv1d(net, 2048, 3, 2, scope='g_conv_a5')
        end_points['conv_a5'] = net
        net = tf.contrib.layers.conv1d(net, 2048, 3, 2, scope='g_conv_a6')
        end_points['conv_a6'] = net
        net = tf.contrib.layers.conv1d(net, 4096, 3, 2, scope='g_conv_a7')
        end_points['conv_a7'] = net
        net = tf.contrib.layers.conv1d(net, 4096, 3, 2, scope='g_conv_a8')
        end_points['conv_a8'] = net

        # Prediction and localisations layers.
        predictions = []
        logits = []
        localisations = []
        proplogits = []
        proppredictions = []
        for i, layer in enumerate(feat_layers):
            with tf.variable_scope(layer + '_box',reuse=tf.AUTO_REUSE):
                p, l, prop = aher_mutibox_prediction_proposal_layer(
                                          end_points[layer],
                                          clsweights[i],
                                          num_classes,
                                          anchor_sizes[i],
                                          anchor_ratios[i],
                                          normalizations[i])
            predictions.append(prediction_fn(p))
            logits.append(p)
            localisations.append(l)
            proplogits.append(prop)
            proppredictions.append(prediction_fn(prop))
        return predictions, localisations, logits, proplogits, proppredictions, end_points

# predict recognition, localization, iou, proposal
def aher_net_prop_iou(inputs,clsweights,clsbias,
            num_classes=AHERNet.default_params.num_classes,
            feat_layers=AHERNet.default_params.feat_layers,
            anchor_sizes=AHERNet.default_params.anchor_sizes,
            anchor_ratios=AHERNet.default_params.anchor_ratios,
            normalizations=AHERNet.default_params.normalizations,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=tf.contrib.layers.softmax,
            reuse=None,
            cls_suffix='_anchor',
            scope='aher'):
    """AHER net definition.
    """
    # End_points collect relevant activations for external use.
    end_points = {}

    with tf.variable_scope(scope, 'aher', [inputs], reuse=reuse):
        # base feature network
        net = tf.contrib.layers.conv1d(inputs, 2048, 3, scope = 'g_conv1')
        net = tf.contrib.layers.conv1d(net, 1024, 3, scope = 'g_conv2')
        net = tf.layers.max_pooling1d(net,3,2,name='pool2')
    
        net = tf.contrib.layers.conv1d(net, 256, 3, 2, scope='g_conv_a1')
        end_points['conv_a1'] = net
        net = tf.contrib.layers.conv1d(net, 256, 3, 2, scope='g_conv_a2')
        end_points['conv_a2'] = net
        net = tf.contrib.layers.conv1d(net, 512, 3, 2, scope='g_conv_a3')
        end_points['conv_a3'] = net
        net = tf.contrib.layers.conv1d(net, 512, 3, 2, scope='g_conv_a4')
        end_points['conv_a4'] = net
        net = tf.contrib.layers.conv1d(net, 1024, 3, 2, scope='g_conv_a5')
        end_points['conv_a5'] = net
        net = tf.contrib.layers.conv1d(net, 1024, 3, 2, scope='g_conv_a6')
        end_points['conv_a6'] = net
        net = tf.contrib.layers.conv1d(net, 2048, 3, 2, scope='g_conv_a7')
        end_points['conv_a7'] = net
        net = tf.contrib.layers.conv1d(net, 2048, 3, 2, scope='g_conv_a8')
        end_points['conv_a8'] = net

        # Prediction and localisations layers.
        predictions = []
        logits = []
        localisations = []
        proplogits = []
        proppredictions = []
        iouprediction = []
        for i, layer in enumerate(feat_layers):
            with tf.variable_scope(layer + '_box',reuse=tf.AUTO_REUSE):
                p, l, prop, iou = aher_mutibox_prediction_proposal_iou_layer_anet(
                                          end_points[layer],
                                          clsweights[i],
                                          clsbias[i],
                                          num_classes,
                                          anchor_sizes[i],
                                          anchor_ratios[i],
                                          normalizations[i],
                                          cls_suffix=cls_suffix)
            predictions.append(prediction_fn(p))
            logits.append(p)
            localisations.append(l)
            proplogits.append(prop)
            proppredictions.append(prediction_fn(prop))
            iouprediction.append(tf.math.sigmoid(iou))
        return predictions, localisations, logits, proplogits, proppredictions, iouprediction, end_points

# predict recognition, localization, iou, proposal (without cls weight and bias)
def aher_net_prop_iou_pure(inputs,
            num_classes=AHERNet.default_params.num_classes,
            feat_layers=AHERNet.default_params.feat_layers,
            anchor_sizes=AHERNet.default_params.anchor_sizes,
            anchor_ratios=AHERNet.default_params.anchor_ratios,
            normalizations=AHERNet.default_params.normalizations,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=tf.contrib.layers.softmax,
            reuse=None,
            cls_suffix='anet',
            scope='aher'):
    """AHER net definition.
    """
    # End_points collect relevant activations for external use.
    end_points = {}

    with tf.variable_scope(scope, 'aher', [inputs], reuse=reuse):
        # base feature network
        net = tf.contrib.layers.conv1d(inputs, 2048, 3, scope = 'g_conv1')
        net = tf.contrib.layers.conv1d(net, 1024, 3, scope = 'g_conv2')
        net = tf.layers.max_pooling1d(net,3,2,name='pool2')
    
        net = tf.contrib.layers.conv1d(net, 256, 3, 2, scope='g_conv_a1')
        end_points['conv_a1'] = net
        net = tf.contrib.layers.conv1d(net, 256, 3, 2, scope='g_conv_a2')
        end_points['conv_a2'] = net
        net = tf.contrib.layers.conv1d(net, 512, 3, 2, scope='g_conv_a3')
        end_points['conv_a3'] = net
        net = tf.contrib.layers.conv1d(net, 512, 3, 2, scope='g_conv_a4')
        end_points['conv_a4'] = net
        net = tf.contrib.layers.conv1d(net, 1024, 3, 2, scope='g_conv_a5')
        end_points['conv_a5'] = net
        net = tf.contrib.layers.conv1d(net, 1024, 3, 2, scope='g_conv_a6')
        end_points['conv_a6'] = net
        net = tf.contrib.layers.conv1d(net, 2048, 3, 2, scope='g_conv_a7')
        end_points['conv_a7'] = net
        net = tf.contrib.layers.conv1d(net, 2048, 3, 2, scope='g_conv_a8')
        end_points['conv_a8'] = net

        # Prediction and localisations layers.
        predictions = []
        logits = []
        localisations = []
        proplogits = []
        proppredictions = []
        iouprediction = []
        for i, layer in enumerate(feat_layers):
            with tf.variable_scope(layer + '_box',reuse=tf.AUTO_REUSE):
                p, l, prop, iou = aher_multibox_classwise_iou_layer(
                                          end_points[layer],
                                          num_classes,
                                          anchor_sizes[i],
                                          anchor_ratios[i],
                                          normalizations[i],
                                          cls_suffix=cls_suffix)
            predictions.append(prediction_fn(p))
            logits.append(p)
            localisations.append(l)
            proplogits.append(prop)
            proppredictions.append(prediction_fn(prop))
            iouprediction.append(tf.math.sigmoid(iou))
        return predictions, localisations, logits, proplogits, proppredictions, iouprediction, end_points

def aher_net_detect(inputs,
            num_classes=AHERNet.default_params.num_classes,
            feat_layers=AHERNet.default_params.feat_layers,
            anchor_sizes=AHERNet.default_params.anchor_sizes,
            anchor_ratios=AHERNet.default_params.anchor_ratios,
            normalizations=AHERNet.default_params.normalizations,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=tf.nn.tanh,
            reuse=None,
            scope='aher'):
    """AHER net definition.
    """
    # End_points collect relevant activations for external use.
    end_points = {}

    with tf.variable_scope(scope, 'aher', [inputs], reuse=reuse):
        # base feature network
        net = tf.contrib.layers.conv1d(inputs, 4096, 3, scope = 'g_conv1')
        net = tf.contrib.layers.conv1d(net, 2048, 3, scope = 'g_conv2')
        net = tf.layers.max_pooling1d(net,3,2,name='pool2')
    
        net = tf.contrib.layers.conv1d(net, 512, 3, 2, scope='g_conv_a1')
        end_points['conv_a1'] = net
        net = tf.contrib.layers.conv1d(net, 512, 3, 2, scope='g_conv_a2')
        end_points['conv_a2'] = net
        net = tf.contrib.layers.conv1d(net, 1024, 3, 2, scope='g_conv_a3')
        end_points['conv_a3'] = net
        net = tf.contrib.layers.conv1d(net, 1024, 3, 2, scope='g_conv_a4')
        end_points['conv_a4'] = net
        net = tf.contrib.layers.conv1d(net, 2048, 3, 2, scope='g_conv_a5')
        end_points['conv_a5'] = net
        net = tf.contrib.layers.conv1d(net, 2048, 3, 2, scope='g_conv_a6')
        end_points['conv_a6'] = net
        net = tf.contrib.layers.conv1d(net, 4096, 3, 2, scope='g_conv_a7')
        end_points['conv_a7'] = net
        net = tf.contrib.layers.conv1d(net, 4096, 3, 2, scope='g_conv_a8')
        end_points['conv_a8'] = net

        # Prediction and localisations layers.
        localisations = []
        proplogits = []
        proppredictions = []
        iouprediction = []
        for i, layer in enumerate(feat_layers):
            with tf.variable_scope(layer + '_box',reuse=tf.AUTO_REUSE):
                l, prop, iou = aher_mutibox_detection_layer(
                                          end_points[layer],
                                          num_classes,
                                          anchor_sizes[i],
                                          anchor_ratios[i],
                                          normalizations[i])
            localisations.append(l)
            proplogits.append(prop)
            proppredictions.append(prediction_fn(prop))
            iouprediction.append(tf.math.sigmoid(iou))
        return localisations, proplogits, proppredictions, iouprediction, end_points

def aher_depthconv1d_net(inputs,
            num_classes=AHERNet.default_params.num_classes,
            feat_layers=AHERNet.default_params.feat_layers,
            anchor_sizes=AHERNet.default_params.anchor_sizes,
            anchor_ratios=AHERNet.default_params.anchor_ratios,
            normalizations=AHERNet.default_params.normalizations,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=tf.contrib.layers.softmax,
            reuse=None,
            scope='aher'):
    """AHER net definition.
    """
    # End_points collect relevant activations for external use.
    end_points = {}

    with tf.variable_scope(scope, 'aher', [inputs], reuse=reuse):
        # original 1d conv layer
        net = tf.contrib.layers.conv1d(inputs, 4096, 3, scope = 'g_conv1')
        net = tf.contrib.layers.conv1d(net, 2048, 3, scope = 'g_conv2')
        net = tf.layers.max_pooling1d(net,3,2,padding='same',name='pool2')

        # depth separate block
        net = tf_utils.depthwise_conv1d(net, k_w=3, strides=2, name='g_depconv_a1')
        net = tf.contrib.layers.conv1d(net, 512, 1, scope = 'g_sep_conv_a1')
        end_points['conv_a1'] = net
        net = tf_utils.depthwise_conv1d(net, k_w=3, strides=2, name='g_depconv_a2')
        net = tf.contrib.layers.conv1d(net, 512, 1, scope = 'g_sep_conv_a2')
        end_points['conv_a2'] = net      
        net = tf_utils.depthwise_conv1d(net, k_w=3, strides=2, name='g_depconv_a3')
        net = tf.contrib.layers.conv1d(net, 1024, 1, scope = 'g_sep_conv_a3')
        end_points['conv_a3'] = net  
        net = tf_utils.depthwise_conv1d(net, k_w=3, strides=2, name='g_depconv_a4')
        net = tf.contrib.layers.conv1d(net, 1024, 1, scope = 'g_sep_conv_a4')
        end_points['conv_a4'] = net
        net = tf_utils.depthwise_conv1d(net, k_w=3, strides=2, name='g_depconv_a5')
        net = tf.contrib.layers.conv1d(net, 2048, 1, scope = 'g_sep_conv_a5')
        end_points['conv_a5'] = net    
        net = tf_utils.depthwise_conv1d(net, k_w=3, strides=2, name='g_depconv_a6')
        net = tf.contrib.layers.conv1d(net, 2048, 1, scope = 'g_sep_conv_a6')
        end_points['conv_a6'] = net    
        net = tf_utils.depthwise_conv1d(net, k_w=3, strides=2, name='g_depconv_a7')
        net = tf.contrib.layers.conv1d(net, 4096, 1, scope = 'g_sep_conv_a7')
        end_points['conv_a7'] = net 
        net = tf_utils.depthwise_conv1d(net, k_w=3, strides=2, name='g_depconv_a8')
        net = tf.contrib.layers.conv1d(net, 4096, 1, scope = 'g_sep_conv_a8')
        end_points['conv_a8'] = net  

        # Prediction and localisations layers.
        predictions = []
        logits = []
        localisations = []
        proplogits = []
        proppredictions=[]
        iouprediction = []
        for i, layer in enumerate(feat_layers):
            with tf.variable_scope(layer + '_box',reuse=reuse):
                p, l, prop, iou = aher_multibox_classwise_iou_layer(
                                          end_points[layer],
                                          num_classes,
                                          anchor_sizes[i],
                                          anchor_ratios[i],
                                          normalizations[i])
            predictions.append(prediction_fn(p))
            logits.append(p)
            localisations.append(l)
            proplogits.append(prop)
            proppredictions.append(prediction_fn(prop))
            iouprediction.append(tf.math.sigmoid(iou))
        return predictions, localisations, logits, proplogits, proppredictions, iouprediction, end_points

# =========================================================================== #
# AHER loss function.
# =========================================================================== #
def aher_losses(logits, localisations, iouprediction,
               gclasses, glocalisations, gscores, giou,
               match_threshold=0.3,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               cls_weights = 1.0,
               iou_weights = 1.0,
               device='/cpu:0',
               scope=None):
    with tf.name_scope(scope, 'aher_losses'):
        lshape = tfe.get_shape(logits[0], 8)
        num_classes = lshape[-1]
        batch_size = lshape[0]

        # Flatten out all vectors!
        flogits = []
        fgclasses = []
        fgscores = []
        fgiou = []
        flocalisations = []
        fglocalisations = []
        fiou = []
        for i in range(len(logits)):
            flogits.append(tf.reshape(logits[i], [-1, num_classes]))
            fgclasses.append(tf.reshape(gclasses[i], [-1]))
            fgscores.append(tf.reshape(gscores[i], [-1]))
            fgiou.append(tf.reshape(giou[i],[-1]))
            flocalisations.append(tf.reshape(localisations[i], [-1, 2]))
            fglocalisations.append(tf.reshape(glocalisations[i], [-1, 2]))
            fiou.append(tf.reshape(iouprediction[i],[-1]))
        # And concat the crap!
        logits = tf.concat(flogits, axis=0)
        gclasses = tf.concat(fgclasses, axis=0)
        gscores = tf.concat(fgscores, axis=0)
        giou = tf.concat(fgiou, axis=0)
        localisations = tf.concat(flocalisations, axis=0)
        glocalisations = tf.concat(fglocalisations, axis=0)
        iouprediction = tf.concat(fiou, axis=0)
        dtype = logits.dtype

        # Compute positive matching mask...
        pmask = gscores > match_threshold
        fpmask = tf.cast(pmask, dtype)
        fpmask_num = tf.reduce_sum(fpmask)
        print('The localization iou threshold is %f'%(match_threshold))
		
        pos_mask = gscores > 0.70
        f_pos_mask = tf.cast(pmask, dtype)
        n_pos = tf.reduce_sum(f_pos_mask)
        pos_classes = tf.cast(pos_mask, tf.int32)

        # Hard negative mining...
        neg_classes = tf.cast(pos_mask, tf.int32)
        predictions = tf.contrib.layers.softmax(logits)
        nmask = tf.logical_and(tf.logical_not(pos_mask),gscores < 0.3)
        fnmask = tf.cast(nmask, dtype)
        nvalues = tf.where(nmask,
                           predictions[:, 0],
                           1. - fnmask)
        nvalues_flat = tf.reshape(nvalues, [-1])
        # Number of negative entries to select.
        max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
        n_neg = tf.cast(negative_ratio * n_pos, tf.int32) + batch_size
        n_neg = tf.minimum(n_neg, max_neg_entries)

        val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
        max_hard_pred = -val[-1]
        # Final negative mask.
        nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
        fnmask = tf.cast(nmask, dtype)

        # combine the pos mask and neg mask
        final_mask = tf.logical_or(pos_mask,nmask)
        f_all_mask = tf.cast(final_mask, dtype)

        #gclasses = tf.one_hot(gclasses, depth=200, axis=1, dtype=tf.float32)
        #no_classes = tf.one_hot(no_classes, depth=200, axis=1, dtype=tf.float32)

        gclass = tf.one_hot(pos_classes, depth=2, axis=1, dtype=tf.float32)

        #iou maks 
        iou_mask = tf.not_equal(giou,-1.0)
        f_iou_mask = tf.cast(iou_mask,dtype)
        iou_num = tf.reduce_sum(f_iou_mask)

        # Add cross-entropy loss.
        #with tf.name_scope('cross_entropy_pos'):
        #    loss = tf.contrib.losses.softmax_cross_entropy(logits=logits,
        #                                                   onehot_labels=gclasses)
        #    loss = tf.div(tf.reduce_sum(loss * fpmask), batch_size, name='value')
        #    tf.losses.add_loss(loss)
        #    #cross_entropy_pos = loss

        #with tf.name_scope('cross_entropy_neg'):
        #    loss = tf.contrib.losses.softmax_cross_entropy(logits=logits,
        #                                                   onehot_labels=no_classes)
        #    loss = tf.div(tf.reduce_sum(loss * fnmask), batch_size, name='value')
        #    tf.losses.add_loss(loss)
            #cross_entropy_neg = loss
    
        with tf.name_scope('classification'):
            loss = tf.losses.softmax_cross_entropy(logits=logits,
                                                   onehot_labels=gclass,
                                                   weights=cls_weights,
                                                   reduction=tf.losses.Reduction.NONE)
            loss = tf.div(tf.reduce_sum(loss*f_all_mask), batch_size, name='value')
            tf.losses.add_loss(loss)
            #cross_entropy_pos = loss
    
        # Add localization loss: smooth L1, L2, ...
        with tf.name_scope('localization'):
            # Weights Tensor: positive mask + random negative.
            weights = tf.expand_dims(alpha * fpmask, axis=-1)
            loss = custom_layers.abs_smooth(localisations - glocalisations)
            #loss = tf.div(tf.reduce_sum(loss * weights), fpmask_num, name='value')
            loss = tf.div(tf.reduce_sum(loss * weights), batch_size, name='value')
            tf.losses.add_loss(loss)
            #localization = loss
        

        with tf.name_scope('iou_prediction'):
            loss = tf.losses.mean_squared_error(predictions = iouprediction,
                                                labels = giou,
                                                weights =iou_weights,
                                                reduction=tf.losses.Reduction.NONE)
            loss = tf.div(tf.reduce_sum(loss*f_iou_mask), iou_num, name='value')
            tf.losses.add_loss(loss)

def aher_losses_classwise(
               logits, localisations, proplogits,
               gclasses, glocalisations, gscores, giou,
               match_threshold=0.3,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               cls_weights = 1.0,
               iou_weights = 1.0,
               device='/cpu:0',
               scope=None):
    with tf.name_scope(scope, 'aher_losses'):
        lshape = tfe.get_shape(logits[0], 8)
        num_classes = lshape[-1]
        batch_size = lshape[0]

        # Flatten out all vectors!
        flogits = []
        fgclasses = []
        fgscores = []
        fgiou = []
        flocalisations = []
        fglocalisations = []
        fprop = []
        for i in range(len(logits)):
            flogits.append(tf.reshape(logits[i], [-1, num_classes]))
            fgclasses.append(tf.reshape(gclasses[i], [-1]))
            fgscores.append(tf.reshape(gscores[i], [-1]))
            fgiou.append(tf.reshape(giou[i],[-1]))
            flocalisations.append(tf.reshape(localisations[i], [-1, 2]))
            fglocalisations.append(tf.reshape(glocalisations[i], [-1, 2]))
            fprop.append(tf.reshape(proplogits[i],[-1, 2]))
        # And concat the crap!
        logits = tf.concat(flogits, axis=0)
        gclasses = tf.concat(fgclasses, axis=0)
        gscores = tf.concat(fgscores, axis=0)
        giou = tf.concat(fgiou, axis=0)
        localisations = tf.concat(flocalisations, axis=0)
        glocalisations = tf.concat(fglocalisations, axis=0)
        proplogits = tf.concat(fprop, axis=0)
        dtype = logits.dtype

        classpredictions = tf.contrib.layers.softmax(logits)

        # Compute positive matching mask...
        pmask = gscores > match_threshold
        fpmask = tf.cast(pmask, dtype)
        fpmask_num = tf.reduce_sum(fpmask)
        print('The localization iou threshold is %f'%(match_threshold))
		
        pos_mask = gscores > 0.70
        f_pos_mask = tf.cast(pos_mask, dtype)
        n_pos = tf.reduce_sum(f_pos_mask)
        pos_classes = tf.cast(pos_mask, tf.int32)

        # Hard negative mining...
        neg_classes = tf.cast(pos_mask, tf.int32)
        predictions = tf.contrib.layers.softmax(proplogits)
        nmask = tf.logical_and(tf.logical_not(pos_mask),gscores < 0.3)
        fnmask = tf.cast(nmask, dtype)
        nvalues = tf.where(nmask,
                           predictions[:, 0],
                           1. - fnmask)
        nvalues_flat = tf.reshape(nvalues, [-1])
        # Number of negative entries to select.
        max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
        n_neg = tf.cast(negative_ratio * n_pos, tf.int32) + batch_size
        n_neg = tf.minimum(n_neg, max_neg_entries)

        val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
        max_hard_pred = -val[-1]
        # Final negative mask.
        nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
        fnmask = tf.cast(nmask, dtype)

        # combine the pos mask and neg mask
        final_mask = tf.logical_or(pos_mask,nmask)
        f_all_mask = tf.cast(final_mask, dtype)


        gclass = tf.one_hot(pos_classes, depth=2, axis=1, dtype=tf.float32)
        gclasses_gt = tf.one_hot(gclasses, depth=num_classes, axis=1, dtype=tf.float32)

        #iou maks 
        #iou_mask = tf.not_equal(giou,-1.0)
        #f_iou_mask = tf.cast(iou_mask,dtype)
        #iou_num = tf.reduce_sum(f_iou_mask)

        # IoU threshold 0.70
        with tf.name_scope('proposal_cls'):
            loss = tf.losses.softmax_cross_entropy(logits=proplogits,
                                                   onehot_labels=gclass,
                                                   weights=cls_weights,
                                                   reduction=tf.losses.Reduction.NONE)
            prop_loss = tf.div(tf.reduce_sum(loss*f_all_mask), batch_size, name='value')


        with tf.name_scope('classification'):
            loss = tf.losses.softmax_cross_entropy(logits=logits,
                                                   onehot_labels=gclasses_gt,
                                                   weights= cls_weights)
            class_loss = tf.div(tf.reduce_sum(loss), batch_size, name='value')
            correct_prediction = tf.equal(tf.argmax(gclasses_gt, 1), tf.argmax(classpredictions, 1))
            class_acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # IoU threshold 0.65
        # Add localization loss: smooth L1, L2, ...
        with tf.name_scope('localization'):
            # Weights Tensor: positive mask + random negative.
            weights = tf.expand_dims(alpha * fpmask, axis=-1)
            loss = custom_layers.abs_smooth(localisations - glocalisations)
            #loss = tf.div(tf.reduce_sum(loss * weights), fpmask_num, name='value')
            localization_loss = tf.div(tf.reduce_sum(loss * weights), batch_size, name='value')
        

        #with tf.name_scope('iou_prediction'):
        #    loss = tf.losses.mean_squared_error(predictions = iouprediction,
        #                                        labels = giou,
        #                                        weights =iou_weights)
        #    iou_loss = tf.div(tf.reduce_sum(loss*f_iou_mask), iou_num, name='value')
            #tf.losses.add_loss(loss)


        return prop_loss,localization_loss,class_loss,class_acc

def aher_losses_cls(
               logits, 
               gclasses,
               gscores,
               match_threshold=0.3,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               cls_weights = 1.0,
               iou_weights = 1.0,
               device='/cpu:0',
               scope=None):
    with tf.name_scope(scope, 'aher_losses'):
        lshape = tfe.get_shape(logits[0], 8)
        num_classes = lshape[-1]
        batch_size = lshape[0]

        # Flatten out all vectors!
        flogits = []
        fgclasses = []
        fgscores = []

        for i in range(len(logits)):
            flogits.append(tf.reshape(logits[i], [-1, num_classes]))
            fgclasses.append(tf.reshape(gclasses[i], [-1]))
            fgscores.append(tf.reshape(gscores[i], [-1]))

        # And concat the crap!
        logits = tf.concat(flogits, axis=0)
        gclasses = tf.concat(fgclasses, axis=0)
        gscores = tf.concat(fgscores, axis=0)
        dtype = logits.dtype

        classpredictions = tf.contrib.layers.softmax(logits)

        gclasses_gt = tf.one_hot(gclasses, depth=num_classes, axis=1, dtype=tf.float32)

        pos_mask = gscores > 0.015
        fp_mask = tf.cast(pos_mask, dtype)
        fpmask_num = tf.reduce_sum(fp_mask)

        with tf.name_scope('classification'):
            loss = tf.losses.softmax_cross_entropy(logits=logits,
                                                   onehot_labels=gclasses_gt,
                                                   weights= cls_weights,
                                                   reduction=tf.losses.Reduction.NONE)
            class_loss = tf.div(tf.reduce_sum(loss*fp_mask), fpmask_num, name='value') #fp_mask

            max_gt = tf.argmax(gclasses_gt, 1)
            max_pre = tf.argmax(classpredictions, 1)

            mask_max_gt = tf.boolean_mask(max_gt,pos_mask)
            mask_max_pre = tf.boolean_mask(max_pre,pos_mask)

            correct_prediction = tf.equal(mask_max_gt, mask_max_pre)
            class_acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))
   
        return class_loss,class_acc

def aher_losses_reg(
               logits, 
               gclasses,
               match_threshold=0.3,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               cls_weights = 1.0,
               iou_weights = 1.0,
               device='/cpu:0',
               scope=None):
    with tf.name_scope(scope, 'aher_losses'):
        lshape = tfe.get_shape(logits[0], 8)
        num_classes = lshape[-1]
        batch_size = lshape[0]

        # Flatten out all vectors!
        flogits = []
        fgclasses = []

        for i in range(len(logits)):
            flogits.append(tf.reshape(logits[i], [-1, num_classes]))
            fgclasses.append(tf.reshape(tf.reduce_mean(gclasses[i],1), [-1]))

        # And concat the crap!
        logits = tf.concat(flogits, axis=0)
        gclasses = tf.concat(fgclasses, axis=0)
        dtype = logits.dtype

        classpredictions = tf.contrib.layers.softmax(logits)

        gclasses_gt = tf.one_hot(gclasses, depth=num_classes, axis=1, dtype=tf.float32)


        with tf.name_scope('classification'):
            loss = tf.losses.softmax_cross_entropy(logits=logits,
                                                   onehot_labels=gclasses_gt,
                                                   weights= cls_weights)
            class_loss = tf.div(loss, batch_size, name='value') 

            max_gt = tf.argmax(gclasses_gt, 1)
            max_pre = tf.argmax(classpredictions, 1)
            correct_prediction = tf.equal(max_gt, max_pre)
            class_acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))
   
        return class_loss,class_acc

def aher_losses_classwise_iou(
               logits, localisations, proplogits, iouprediction,
               gclasses, glocalisations, gscores, giou,
               match_threshold=0.3,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               cls_weights = 1.0,
               iou_weights = 1.0,
               device='/cpu:0',
               anchor_position_wise=False,
               negative_match_threshold=0.3,
               scope=None):
    with tf.name_scope(scope, 'aher_losses'):
        lshape = tfe.get_shape(logits[0], 8)
        num_classes = lshape[-1]
        batch_size = lshape[0]

        # Flatten out all vectors!
        flogits = []
        fgclasses = []
        fgscores = []
        fgiou = []
        flocalisations = []
        fglocalisations = []
        fprop = []
        fiou = []
        fgscores_unit = []
        for i in range(len(logits)):
            flogits.append(tf.reshape(logits[i], [-1, num_classes]))
            fgclasses.append(tf.reshape(gclasses[i], [-1]))
            fgscores.append(tf.reshape(gscores[i], [-1]))
            fgiou.append(tf.reshape(giou[i],[-1]))
            flocalisations.append(tf.reshape(localisations[i], [-1, 2]))
            fglocalisations.append(tf.reshape(glocalisations[i], [-1, 2]))
            fprop.append(tf.reshape(proplogits[i],[-1, 2]))
            fiou.append(tf.reshape(iouprediction[i],[-1]))
            fgscores_unit.append(tf.reshape(gscores[i][:,:,0],[-1]))

        # And concat the crap!
        logits = tf.concat(flogits, axis=0)
        gclasses = tf.concat(fgclasses, axis=0)
        gscores = tf.concat(fgscores, axis=0)
        giou = tf.concat(fgiou, axis=0)
        localisations = tf.concat(flocalisations, axis=0)
        glocalisations = tf.concat(fglocalisations, axis=0)
        proplogits = tf.concat(fprop, axis=0)
        iouprediction = tf.concat(fiou, axis=0)
        gscores_u = tf.concat(fgscores_unit,axis=0)
        dtype = logits.dtype

        classpredictions = tf.contrib.layers.softmax(logits)

        gsshape = tfe.get_shape(gscores, 8)
        gs_sample_num = gsshape[0]

        # Compute positive matching mask...
        pmask = gscores > match_threshold
        fpmask = tf.cast(pmask, dtype)
        fpmask_num = tf.reduce_sum(fpmask)
        #print('The localization iou threshold is %f'%(match_threshold))
		
        pos_mask = gscores > match_threshold
        f_pos_mask = tf.cast(pos_mask, dtype)
        n_pos = tf.reduce_sum(f_pos_mask)
        pos_classes = tf.cast(pos_mask, tf.int32)
        print('The localization pos iou threshold is %f, neg iou threshold is %f'%(match_threshold,negative_match_threshold))
        print('The localization weight alpha is %f'%(alpha))

        # Hard negative mining...
        neg_classes = tf.cast(pos_mask, tf.int32)
        predictions = tf.contrib.layers.softmax(proplogits)
        nmask = tf.logical_and(tf.logical_not(pos_mask),gscores < negative_match_threshold) # refine the negratio
        fnmask = tf.cast(nmask, dtype)
        nvalues = tf.where(nmask,
                           predictions[:, 0],
                           1. - fnmask)
        nvalues_flat = tf.reshape(nvalues, [-1])
        # Number of negative entries to select.
        max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
        n_neg = tf.cast(negative_ratio * n_pos, tf.int32) + batch_size
        n_neg = tf.minimum(n_neg, max_neg_entries)

        val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
        max_hard_pred = -val[-1]
        # Final negative mask.
        nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
        fnmask = tf.cast(nmask, dtype)

        # combine the pos mask and neg mask
        final_mask = tf.logical_or(pos_mask,nmask)
        f_all_mask = tf.cast(final_mask, dtype)
        final_mask_num = tf.reduce_sum(f_all_mask)


        gclass = tf.one_hot(pos_classes, depth=2, axis=1, dtype=tf.float32)
        gclasses_gt = tf.one_hot(gclasses, depth=num_classes, axis=1, dtype=tf.float32)

        #iou maks 
        iou_mask = tf.not_equal(giou,-1.0)
        f_iou_mask = tf.cast(iou_mask,dtype)
        iou_num = tf.reduce_sum(f_iou_mask)

        # anchor position wise
        unit_pmask = gscores_u > match_threshold
        unit_f_pos_mask = tf.cast(unit_pmask, dtype)
        unit_pos = tf.reduce_sum(unit_f_pos_mask)
        if anchor_position_wise:
            print('The anchor position flag is True.')
        else: print('The anchor position flag is False.')

        print('The negative vs. positive ratio is: %0.1f'%(negative_ratio))
        print('The gscore sample num is %d'%(gs_sample_num))

        # IoU threshold 0.70
        with tf.name_scope('proposal_cls'):
            loss = tf.losses.softmax_cross_entropy(logits=proplogits,
                                                   onehot_labels=gclass,
                                                   weights=cls_weights,
                                                   reduction=tf.losses.Reduction.NONE)
            prop_loss = tf.div(tf.reduce_sum(loss*f_all_mask), final_mask_num, name='value') # batch_size


        with tf.name_scope('classification'):
            loss = tf.losses.softmax_cross_entropy(logits=logits,
                                                   onehot_labels=gclasses_gt,
                                                   weights= cls_weights,
                                                   reduction=tf.losses.Reduction.NONE)
            if anchor_position_wise:
                class_loss = tf.div(tf.reduce_sum(loss*unit_f_pos_mask), unit_pos, name='value')
                max_gt = tf.argmax(gclasses_gt, 1)
                max_pre = tf.argmax(classpredictions, 1)
                mask_max_gt = tf.boolean_mask(max_gt,unit_pmask)
                mask_max_pre = tf.boolean_mask(max_pre,unit_pmask)
                correct_prediction = tf.equal(mask_max_gt, mask_max_pre)
                class_acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            else:
                class_loss = tf.div(tf.reduce_sum(loss), batch_size, name='value')
                correct_prediction = tf.equal(tf.argmax(gclasses_gt, 1), tf.argmax(classpredictions, 1))
                class_acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # IoU threshold 0.65
        # Add localization loss: smooth L1, L2, .. 
        # Need positive and negative weights
        with tf.name_scope('localization'):
            # Weights Tensor: positive mask + random negative.
            weights = tf.expand_dims(alpha * fpmask, axis=-1)
            loss = custom_layers.abs_smooth(localisations - glocalisations)
            localization_loss = tf.div(tf.reduce_sum(loss * weights), final_mask_num, name='value') # fpmask_num final_mask_num
        

        with tf.name_scope('iou_prediction'):
            loss = tf.losses.mean_squared_error(predictions = iouprediction,
                                                labels = giou,
                                                weights =iou_weights,
                                                reduction=tf.losses.Reduction.NONE)
            iou_loss = tf.div(tf.reduce_sum(loss*f_iou_mask), iou_num, name='value')
            #tf.losses.add_loss(loss)


        return prop_loss,localization_loss,class_loss,class_acc,iou_loss

def aher_losses_mome_iou(
               logits, localisations, proplogits, iouprediction,
               gclasses, glocalisations, gscores, giou,
               match_threshold=0.3,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               cls_weights = 1.0,
               iou_weights = 1.0,
               device='/cpu:0',
               anchor_position_wise=False,
			   negative_match_threshold=0.3,	
               scope=None):
    with tf.name_scope(scope, 'aher_losses'):
        lshape = tfe.get_shape(logits[0], 8)
        num_classes = lshape[-1]
        batch_size = lshape[0]

        # Flatten out all vectors!
        flogits = []
        fgclasses = []
        fgscores = []
        fgiou = []
        flocalisations = []
        fglocalisations = []
        fprop = []
        fiou = []
        fgscores_unit = []
        for i in range(len(logits)):
            flogits.append(tf.reshape(logits[i], [-1, num_classes]))
            fgclasses.append(tf.reshape(tf.reduce_mean(gclasses[i],1), [-1])) # reduce the anchor based classification to video level
            fgscores.append(tf.reshape(gscores[i], [-1]))
            fgiou.append(tf.reshape(giou[i],[-1]))
            flocalisations.append(tf.reshape(localisations[i], [-1, 2]))
            fglocalisations.append(tf.reshape(glocalisations[i], [-1, 2]))
            fprop.append(tf.reshape(proplogits[i],[-1, 2]))
            fiou.append(tf.reshape(iouprediction[i],[-1]))
            fgscores_unit.append(tf.reshape(gscores[i][:,:,0],[-1]))

        # And concat the crap!
        logits = tf.concat(flogits, axis=0)
        gclasses = tf.concat(fgclasses, axis=0)
        gscores = tf.concat(fgscores, axis=0)
        giou = tf.concat(fgiou, axis=0)
        localisations = tf.concat(flocalisations, axis=0)
        glocalisations = tf.concat(fglocalisations, axis=0)
        proplogits = tf.concat(fprop, axis=0)
        iouprediction = tf.concat(fiou, axis=0)
        gscores_u = tf.concat(fgscores_unit,axis=0)
        dtype = logits.dtype

        classpredictions = tf.contrib.layers.softmax(logits)

        gsshape = tfe.get_shape(gscores, 8)
        gs_sample_num = gsshape[0]

        # Compute positive matching mask...
        pmask = gscores > match_threshold
        fpmask = tf.cast(pmask, dtype)
        fpmask_num = tf.reduce_sum(fpmask)
        #print('The localization iou threshold is %f'%(match_threshold))
		
        pos_mask = gscores > match_threshold
        f_pos_mask = tf.cast(pos_mask, dtype)
        n_pos = tf.reduce_sum(f_pos_mask)
        pos_classes = tf.cast(pos_mask, tf.int32)
        print('The localization pos iou threshold is %f, neg iou threshold is %f'%(match_threshold,negative_match_threshold))
        print('The localization weight alpha is %f'%(alpha))

        # Hard negative mining...
        neg_classes = tf.cast(pos_mask, tf.int32)
        predictions = tf.contrib.layers.softmax(proplogits)
        nmask = tf.logical_and(tf.logical_not(pos_mask),gscores < negative_match_threshold)
        fnmask = tf.cast(nmask, dtype)
        nvalues = tf.where(nmask,
                           predictions[:, 0],
                           1. - fnmask)
        nvalues_flat = tf.reshape(nvalues, [-1])
        # Number of negative entries to select.
        max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
        n_neg = tf.cast(negative_ratio * n_pos, tf.int32) + batch_size
        n_neg = tf.minimum(n_neg, max_neg_entries)

        val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
        max_hard_pred = -val[-1]
        # Final negative mask.
        nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
        fnmask = tf.cast(nmask, dtype)

        # combine the pos mask and neg mask
        final_mask = tf.logical_or(pos_mask,nmask)
        f_all_mask = tf.cast(final_mask, dtype)
        final_mask_num = tf.reduce_sum(f_all_mask)


        gclass = tf.one_hot(pos_classes, depth=2, axis=1, dtype=tf.float32)
        gclasses_gt = tf.one_hot(gclasses, depth=num_classes, axis=1, dtype=tf.float32)

        #iou maks 
        iou_mask = tf.not_equal(giou,-1.0)
        f_iou_mask = tf.cast(iou_mask,dtype)
        iou_num = tf.reduce_sum(f_iou_mask)

        # anchor position wise
        unit_pmask = gscores_u > match_threshold
        unit_f_pos_mask = tf.cast(unit_pmask, dtype)
        unit_pos = tf.reduce_sum(unit_f_pos_mask)
        if anchor_position_wise:
            print('The anchor position flag is True.')
        else: print('The anchor position flag is False.')

        print('The negative vs. positive ratio is: %0.1f'%(negative_ratio))
        print('The gscore sample num is %d'%(gs_sample_num))

        # IoU threshold 0.70
        with tf.name_scope('proposal_cls'):
            loss = tf.losses.softmax_cross_entropy(logits=proplogits,
                                                   onehot_labels=gclass,
                                                   weights=cls_weights,
                                                   reduction=tf.losses.Reduction.NONE)
            prop_loss = tf.div(tf.reduce_sum(loss*f_all_mask), final_mask_num, name='value') # batch_size


        with tf.name_scope('classification'):
            loss = tf.losses.softmax_cross_entropy(logits=logits,
                                                   onehot_labels=gclasses_gt,
                                                   weights= cls_weights)
            class_loss = tf.div(loss, batch_size, name='value') 

            max_gt = tf.argmax(gclasses_gt, 1)
            max_pre = tf.argmax(classpredictions, 1)
            correct_prediction = tf.equal(max_gt, max_pre)
            class_acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # IoU threshold 0.65
        # Add localization loss: smooth L1, L2, .. 
        # Need positive and negative weights
        with tf.name_scope('localization'):
            # Weights Tensor: positive mask + random negative.
            weights = tf.expand_dims(alpha * fpmask, axis=-1)
            loss = custom_layers.abs_smooth(localisations - glocalisations)
            localization_loss = tf.div(tf.reduce_sum(loss * weights), final_mask_num, name='value')
        

        with tf.name_scope('iou_prediction'):
            loss = tf.losses.mean_squared_error(predictions = iouprediction,
                                                labels = giou,
                                                weights =iou_weights,
                                                reduction=tf.losses.Reduction.NONE)
            iou_loss = tf.div(tf.reduce_sum(loss*f_iou_mask), iou_num, name='value')
            #tf.losses.add_loss(loss)


        return prop_loss,localization_loss,class_loss,class_acc,iou_loss

def aher_losses_detect_s(
               localisations, proplogits, iouprediction,
               gclasses, glocalisations, gscores, giou,
               match_threshold=0.3,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               cls_weights = 1.0,
               iou_weights = 1.0,
               device='/cpu:0',
               anchor_position_wise=False,
               scope=None):
    with tf.name_scope(scope, 'aher_losses'):
        lshape = tfe.get_shape(proplogits[0], 8)
        num_classes = lshape[-1]
        batch_size = lshape[0]

        # Flatten out all vectors!
        fgclasses = []
        fgscores = []
        fgiou = []
        flocalisations = []
        fglocalisations = []
        fprop = []
        fiou = []
        fgscores_unit = []
        for i in range(len(proplogits)):
            fgclasses.append(tf.reshape(tf.tile(gclasses[i],(1,1,3)),[-1]))
            fgscores.append(tf.reshape(gscores[i], [-1]))
            fgiou.append(tf.reshape(giou[i],[-1]))
            flocalisations.append(tf.reshape(localisations[i], [-1, 2]))
            fglocalisations.append(tf.reshape(glocalisations[i], [-1, 2]))
            fprop.append(tf.reshape(proplogits[i],[-1, num_classes]))
            fiou.append(tf.reshape(iouprediction[i],[-1]))
            fgscores_unit.append(tf.reshape(gscores[i][:,:,0],[-1]))

        # And concat the crap!
        gclasses = tf.concat(fgclasses, axis=0)
        gscores = tf.concat(fgscores, axis=0)
        giou = tf.concat(fgiou, axis=0)
        localisations = tf.concat(flocalisations, axis=0)
        glocalisations = tf.concat(fglocalisations, axis=0)
        proplogits = tf.concat(fprop, axis=0)
        iouprediction = tf.concat(fiou, axis=0)
        gscores_u = tf.concat(fgscores_unit,axis=0)
        dtype = proplogits.dtype

        kshape = tfe.get_shape(proplogits, 8)
        allanchor_num = kshape[0]

        proppredictions = tf.nn.tanh(proplogits)

        gsshape = tfe.get_shape(gscores, 8)
        gs_sample_num = gsshape[0]

        # Compute positive matching mask...
        pmask = gscores > match_threshold
        fpmask = tf.cast(pmask, dtype)
        fpmask_num = tf.reduce_sum(fpmask)
        
        # Compute the hinge label and tile the tensor
        fnegmask = tf.cast(tf.logical_not(pmask),dtype)
        hingel = fpmask - fnegmask
        hingel = tf.reshape(hingel,[-1,1])
        hingecls = tf.tile(hingel,(1,num_classes))
        gclasses_gt_value = tf.one_hot(gclasses, depth=num_classes, axis=1, dtype=tf.float32)
        gclasses_bool_mask = tf.cast(gclasses_gt_value,tf.bool)

		
        pos_mask = gscores > match_threshold
        f_pos_mask = tf.cast(pos_mask, dtype)
        n_pos = tf.reduce_sum(f_pos_mask)
        pos_classes = tf.cast(pos_mask, tf.int32)
        print('The localization iou threshold is %f'%(match_threshold))

        # Hard negative mining...
        neg_classes = tf.cast(pos_mask, tf.int32)
        nmask = tf.logical_and(tf.logical_not(pos_mask),gscores < 0.3)
        fnmask = tf.cast(nmask, dtype)
        prediction_bool = tf.boolean_mask(proppredictions,gclasses_bool_mask)
        nvalues = tf.where(nmask,
                           tf.reshape(prediction_bool,[allanchor_num]),
                           1. - fnmask)
        nvalues_flat = tf.reshape(nvalues, [-1])
        # Number of negative entries to select.
        max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
        if negative_ratio < 0: 
            n_neg = max_neg_entries
        else:
            n_neg = tf.cast(negative_ratio * n_pos, tf.int32) + batch_size
            n_neg = tf.minimum(n_neg, max_neg_entries)

        val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
        max_hard_pred = -val[-1]
        # Final negative mask.
        nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
        fnmask = tf.cast(nmask, dtype)

        # combine the pos mask and neg mask
        final_mask = tf.logical_or(pos_mask,nmask)
        f_all_mask = tf.cast(final_mask, dtype)
        final_mask_num = tf.reduce_sum(f_all_mask)
        
        f_all_mask_t = tf.reshape(f_all_mask,[-1,1])
        f_all_mask_tile = tf.tile(f_all_mask_t,(1,num_classes))

        #gclass = tf.one_hot(pos_classes, depth=2, axis=1, dtype=tf.float32)
        

        #iou maks 
        iou_mask = tf.not_equal(giou,-1.0)
        f_iou_mask = tf.cast(iou_mask,dtype)
        iou_num = tf.reduce_sum(f_iou_mask)

        # anchor position wise
        unit_pmask = gscores_u > match_threshold
        unit_f_pos_mask = tf.cast(unit_pmask, dtype)
        unit_pos = tf.reduce_sum(unit_f_pos_mask)
        if anchor_position_wise:
            print('The anchor position flag is True.')
        else: print('The anchor position flag is False.')

        print('The negative vs. positive ratio is: %0.1f'%(negative_ratio))
        print('The gscore sample num is %d'%(gs_sample_num))

        # IoU threshold 0.70
        with tf.name_scope('proposal_cls'):
            #loss = custom_layers.single_hinge_loss(proppredictions,hingecls)
            loss = tf.losses.hinge_loss(hingecls,proppredictions,reduction=tf.losses.Reduction.NONE)
            loss_mask_mining = loss * f_all_mask_tile
            loss_mask_cls = loss_mask_mining * gclasses_gt_value
            prop_loss = tf.div(tf.reduce_sum(loss_mask_cls),final_mask_num,name='value')

            #loss = tf.losses.softmax_cross_entropy(logits=proplogits,
            #                                       onehot_labels=gclass,
            #                                       weights=cls_weights,
            #                                       reduction=tf.losses.Reduction.NONE)
            #prop_loss = tf.div(tf.reduce_sum(loss*f_all_mask), final_mask_num, name='value') # batch_size


        #with tf.name_scope('classification'):
        #    loss = tf.losses.softmax_cross_entropy(logits=logits,
        #                                           onehot_labels=gclasses_gt,
        #                                           weights= cls_weights,
        #                                           reduction=tf.losses.Reduction.NONE)
        #    if anchor_position_wise:
        #        class_loss = tf.div(tf.reduce_sum(loss*unit_f_pos_mask), unit_pos, name='value')
        #        max_gt = tf.argmax(gclasses_gt, 1)
        #        max_pre = tf.argmax(classpredictions, 1)
        #        mask_max_gt = tf.boolean_mask(max_gt,unit_pmask)
        #        mask_max_pre = tf.boolean_mask(max_pre,unit_pmask)
        #        correct_prediction = tf.equal(mask_max_gt, mask_max_pre)
        #        class_acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        #    else:
        #        class_loss = tf.div(tf.reduce_sum(loss), batch_size, name='value')
        #        correct_prediction = tf.equal(tf.argmax(gclasses_gt, 1), tf.argmax(classpredictions, 1))
        #        class_acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # IoU threshold 0.65
        # Add localization loss: smooth L1, L2, .. 
        # Need positive and negative weights
        with tf.name_scope('localization'):
            # Weights Tensor: positive mask + random negative.
            weights = tf.expand_dims(alpha * fpmask, axis=-1)
            #weights = tf.div(weights,final_mask_num)
            loss = custom_layers.abs_smooth(localisations - glocalisations)
            #loss = tf.div(tf.reduce_sum(loss * weights), fpmask_num, name='value')
            localization_loss = tf.div(tf.reduce_sum(loss * weights), 2.0, name='value')
        

        with tf.name_scope('iou_prediction'):
            loss = tf.losses.mean_squared_error(predictions = iouprediction,
                                                labels = giou,
                                                weights =iou_weights,
                                                reduction=tf.losses.Reduction.NONE)
            iou_loss = tf.div(tf.reduce_sum(loss*f_iou_mask), iou_num, name='value')
            #tf.losses.add_loss(loss)


        return prop_loss,localization_loss,iou_loss