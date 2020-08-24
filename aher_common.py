"""Shared function between different AherNet implementations.
"""
import numpy as np
import tensorflow as tf
import tf_extended as tfe

batch_size_fix=16

# =========================================================================== #
# TensorFlow implementation of boxes Aher encoding / decoding.
# =========================================================================== #
def tf_aher_bboxes_encode_layer(labels,
                               bboxes,
                               anchors_layer,
                               num_classes,
                               temporal_shape,
                               matching_threshold=0.70,
                               prior_scaling=[1.0,1.0],
                               dtype=tf.float32):
    """Encode groundtruth labels and bounding boxes using 1D anchors from
    one layer.

    Arguments:
      labels: 1D Tensor(int32) containing groundtruth labels;
      bboxes: batch_size x N x 2 Tensor(float) with bboxes relative coordinates;
      anchors_layer: Numpy array with layer anchors;
      matching_threshold: Threshold for positive match with groundtruth bboxes;
      prior_scaling: Scaling of encoded coordinates.

    Return:
      (target_labels, target_localizations, target_scores): Target Tensors.
    """
    # Anchors coordinates and volume.
    yref, tref= anchors_layer
    ymin = yref - tref / 2.
    ymax = yref + tref / 2.
    
    ymin = tf.maximum(0.0, ymin)
    ymax = tf.minimum(ymax, temporal_shape - 1.0)
  
    vol_anchors = ymax - ymin

    if batch_size_fix == 1: 
        bboxes = tf.reshape(bboxes,[1,bboxes.shape[0],bboxes.shape[1]])
        labels = tf.reshape(labels,[1])

    # Initialize tensors...
    shape = (bboxes.shape[0], yref.shape[0], tref.size)
    s_shape = (yref.shape[0], tref.size)
    feat_scores = tf.zeros(shape, dtype=dtype)
    feat_max_iou = tf.zeros(shape, dtype=dtype)

    feat_ymin = tf.zeros(shape, dtype=dtype)
    feat_ymax = tf.ones(shape, dtype=dtype)
    mask_minus = feat_ymin - feat_ymax
    s_max_one = tf.ones(s_shape, dtype=tf.int32)
    s_max_one_f = tf.ones(s_shape, dtype=dtype)

    label_shape = (bboxes.shape[0], yref.shape[0], 1) 
    s_label_shape = (yref.shape[0], 1) 
    s_label_max_one = tf.ones(s_label_shape, dtype=tf.int32)
    feat_labels = tf.zeros(label_shape, dtype=tf.int32)
    

    def jaccard_with_anchors(bbox):
        """Compute jaccard score between a box and the anchors.
        """
        int_ymin = tf.maximum(ymin, bbox[0])
        int_ymax = tf.minimum(ymax, bbox[1])
        t = tf.maximum(int_ymax - int_ymin, 0.)

        # Volumes.
        inter_vol = t
        union_vol = vol_anchors - inter_vol \
            + (bbox[1] - bbox[0])
        jaccard = tf.div(inter_vol, union_vol)
        return jaccard

    def intersection_with_anchors(bbox):
        """Compute intersection between score a box and the anchors.
        """
        int_ymin = tf.maximum(ymin, bbox[0])
        int_ymax = tf.minimum(ymax, bbox[1])
        t = tf.maximum(int_ymax - int_ymin, 0.)
        inter_vol = t
        scores = tf.div(inter_vol, vol_anchors)
        return scores

    def condition(i,ik, feat_labels, feat_scores,
                  feat_ymin, feat_ymax):
        """Condition: check label index.
        """
        # remove unusable gt
        bbox = bboxes[i]
        bbox = tf.reshape(tf.boolean_mask(bbox,tf.not_equal([-1.0,-1.0],bbox)),[-1,2])
        r = tf.less(ik, tf.shape(bbox)[0])
        return r

    def body(batch_id, i, feat_labels, feat_scores,
             feat_ymin, feat_ymax):
        """Body: update feature labels, scores and bboxes.
          - assign values when jaccard > 0.5;
          - only update if beat the score of other bboxes.
        """
        # Jaccard score.
        label = labels[batch_id]
        bbox = bboxes[batch_id]

        bbox = tf.reshape(tf.boolean_mask(bbox,tf.not_equal([-1.0,-1.0],bbox)),[-1,2])[i]

        jaccard = jaccard_with_anchors(bbox)
        # Mask: check threshold + scores + no annotations + num_classes.
        mask = tf.greater(jaccard, feat_scores)
        imask = tf.cast(mask, tf.int32)
        fmask = tf.cast(mask, dtype)
        # Update values using mask.
        
        #feat_labels = imask * label + (1 - imask) * feat_labels
        feat_labels = s_label_max_one * label
        
        feat_scores = tf.where(mask, jaccard, feat_scores)

        feat_ymin = fmask * bbox[0] + (1 - fmask) * feat_ymin
        feat_ymax = fmask * bbox[1] + (1 - fmask) * feat_ymax

        # Check no annotation label: ignore these anchors...
        # interscts = intersection_with_anchors(bbox)
        # mask = tf.logical_and(interscts > ignore_threshold,
        #                       label == no_annotation_label)
        # # Replace scores by -1.
        # feat_scores = tf.where(mask, -tf.cast(mask, dtype), feat_scores)

        return [batch_id, i+1, feat_labels, feat_scores,feat_ymin, feat_ymax]
    
    def batch_condition(i, feat_labels, feat_scores,
                  feat_ymin, feat_ymax):
        r = tf.less(i, tf.shape(bboxes)[0])
        return r     

    def batch_body(i, feat_labels, feat_scores, 
        feat_ymin,feat_ymax):
        ik = 0
        s_feat_labels = tf.zeros(s_label_shape, dtype=tf.int32)
        s_feat_scores  = tf.zeros(s_shape, dtype=dtype)
        s_feat_ymin = tf.zeros(s_shape, dtype=dtype)
        s_feat_ymax = tf.ones(s_shape, dtype=dtype)
        [i, ik, s_feat_labels, s_feat_scores, s_feat_ymin,s_feat_ymax] = tf.while_loop(condition, body,
                                        [i, ik, s_feat_labels, s_feat_scores, s_feat_ymin, s_feat_ymax])

        s_feat_labels = tf.reshape(s_feat_labels,[1,s_label_shape[0],s_label_shape[1]])
        s_feat_scores = tf.reshape(s_feat_scores,[1,s_shape[0],s_shape[1]])
        s_feat_ymin = tf.reshape(s_feat_ymin,[1,s_shape[0],s_shape[1]])
        
       

        # labels
        if batch_size_fix != 1:
            s_feat_labels_1 = tf.zeros([i,s_label_shape[0],s_label_shape[1]], dtype=tf.int32)
            s_feat_labels_2 = tf.zeros([batch_size_fix-i-1,s_label_shape[0],s_label_shape[1]], dtype=tf.int32)
            s_feat_labels = tf.concat([s_feat_labels_1,s_feat_labels,s_feat_labels_2],0)
            feat_labels = feat_labels + s_feat_labels
        else:
            feat_labels = s_feat_labels

        # scores
        if batch_size_fix != 1:
            s_feat_scores_1 = tf.zeros([i,s_shape[0],s_shape[1]], dtype=dtype)
            s_feat_scores_2 = tf.zeros([batch_size_fix-i-1,s_shape[0],s_shape[1]], dtype=dtype)
            s_feat_scores = tf.concat([s_feat_scores_1,s_feat_scores,s_feat_scores_2],0)
            feat_scores = feat_scores + s_feat_scores
        else:
            feat_scores = s_feat_scores

        # ymin
        if batch_size_fix != 1:
            s_feat_ymin_1 = tf.zeros([i,s_shape[0],s_shape[1]], dtype=dtype)
            s_feat_ymin_2 = tf.zeros([batch_size_fix-i-1,s_shape[0],s_shape[1]], dtype=dtype)
            s_feat_ymin = tf.concat([s_feat_ymin_1,s_feat_ymin,s_feat_ymin_2],0)
            feat_ymin = feat_ymin + s_feat_ymin
        else:
            feat_ymin = s_feat_ymin

        # ymax
        if batch_size_fix != 1:
            s_feat_ymax = s_feat_ymax - s_max_one_f
            s_feat_ymax = tf.reshape(s_feat_ymax,[1,s_shape[0],s_shape[1]])
            s_feat_ymax_1 = tf.zeros([i,s_shape[0],s_shape[1]], dtype=dtype)
            s_feat_ymax_2 = tf.zeros([batch_size_fix-i-1,s_shape[0],s_shape[1]], dtype=dtype)
            s_feat_ymax = tf.concat([s_feat_ymax_1,s_feat_ymax,s_feat_ymax_2],0)
            feat_ymax = feat_ymax + s_feat_ymax
        else:
            s_feat_ymax = tf.reshape(s_feat_ymax,[1,s_shape[0],s_shape[1]])
            feat_ymax = s_feat_ymax

        #feat_labels = tf.concat([feat_labels,s_feat_labels],0)
        #feat_scores = tf.concat([feat_scores,s_feat_scores],0)
        #feat_ymin = tf.concat([feat_ymin,s_feat_ymin],0)
        #feat_ymax = tf.concat([feat_ymax,s_feat_ymax],0)
        #feat_labels[i] = s_feat_labels
        #feat_scores[i] = s_feat_scores
        #feat_ymin[i] = s_feat_ymin
        #feat_ymax[i] = s_feat_ymax
        return [i+1,feat_labels,feat_scores,feat_ymin,feat_ymax]

    # Main loop definition.
    i = 0
    #[i, feat_labels, feat_scores,feat_ymin, feat_ymax] = tf.while_loop(condition, body,
    #                                       [i, feat_labels, feat_scores, feat_ymin,feat_ymax])
    [i, feat_labels, feat_scores,feat_ymin, feat_ymax] = tf.while_loop(batch_condition, batch_body,
                                           [i, feat_labels, feat_scores, feat_ymin,feat_ymax])

    #feat_labels = labels
    mask_zero = tf.equal(feat_scores,0.0)
    fmask_zero = tf.cast(mask_zero,dtype)
    feat_max_iou = fmask_zero * mask_minus + (1-fmask_zero)*feat_scores

    # Transform to center / size.
    feat_cy = (feat_ymax + feat_ymin) / 2.
    feat_t = feat_ymax - feat_ymin
    # Encode features.
    feat_cy = (feat_cy - yref) / tref / prior_scaling[0]
    feat_t = tf.log(feat_t / tref) / prior_scaling[1]
    # Use ordering: x / y / w / h instead of ours.
    feat_localizations = tf.stack([feat_cy, feat_t], axis=-1)
    
    return feat_labels, feat_localizations, feat_scores, feat_max_iou

def tf_aher_bboxes_encode(labels,
                         bboxes,
                         anchors,
                         num_classes,
                         temporal_shape,
                         matching_threshold=0.70,
                         prior_scaling=[1.0,1.0],
                         dtype=tf.float32,
                         scope='aher_bboxes_encode'):
    """Encode groundtruth labels and bounding boxes using 1D net anchors.
    Encoding boxes for all feature layers.

    Arguments:
      labels: 1D Tensor(int32) containing groundtruth labels;
      bboxes: Nx2 Tensor(float) with bboxes relative coordinates;
      anchors: List of Numpy array with layer anchors;
      matching_threshold: Threshold for positive match with groundtruth bboxes;
      prior_scaling: Scaling of encoded coordinates.

    Return:
      (target_labels, target_localizations, target_scores):
        Each element is a list of target Tensors.
    """
    with tf.name_scope(scope):
        target_labels = []
        target_localizations = []
        target_scores = []
        target_iou = []
        for i, anchors_layer in enumerate(anchors):
            with tf.name_scope('bboxes_encode_block_%i' % i):
                t_labels, t_loc, t_scores, t_iou = \
                    tf_aher_bboxes_encode_layer(labels, bboxes, anchors_layer,
                                               num_classes, temporal_shape,
                                               matching_threshold,
                                               prior_scaling, dtype)
                target_labels.append(t_labels)
                target_localizations.append(t_loc)
                target_scores.append(t_scores)
                target_iou.append(t_iou)
        return target_labels, target_localizations, target_scores, target_iou

def tf_aher_bboxes_decode_layer(feat_localizations,
                               duration,
                               anchors_layer,
                               prior_scaling=[1.0, 1.0]):
    """Compute the relative bounding boxes from the layer features and
    reference anchor bounding boxes.

    Arguments:
      feat_localizations: Tensor containing localization features.
      anchors: List of numpy array containing anchor boxes.

    Return:
      Tensor Nx2: ymin, ymax
    """
    yref, tref = anchors_layer

    # Compute center, height and width
    cy = feat_localizations[:, :, :, 0] * tref * prior_scaling[0]  + yref # tf.expand_dims(yref,-1)
    t = tref * tf.exp(feat_localizations[:, :, :, 1] * prior_scaling[1])
    # Boxes coordinates.
    ymin = cy - t / 2.
    ymax = cy + t / 2.
    ymin = tf.maximum(ymin, 0.0) / 512.0 * duration
    ymax = tf.minimum(ymax, 512.0) / 512.0 * duration
    bboxes = tf.stack([ymin, ymax], axis=-1)
    return bboxes

def tf_aher_bboxes_decode_logits_layer(feat_localizations,
                               duration,
                               anchors_layer,
                               logitsprediction,
                               prior_scaling=[1.0, 1.0]):
    """Compute the relative bounding boxes from the layer features and
    reference anchor bounding boxes.

    Arguments:
      feat_localizations: Tensor containing localization features.
      anchors: List of numpy array containing anchor boxes.

    Return:
      Tensor Nx2: ymin, ymax
    """
    yref, tref = anchors_layer

    # Compute center, height and width
    cy = feat_localizations[:, :, :, 0] * tref * prior_scaling[0]  + yref # tf.expand_dims(yref,-1)
    t = tref * tf.exp(feat_localizations[:, :, :, 1] * prior_scaling[1])
    # Boxes coordinates.
    ymin = cy - t / 2.
    ymax = cy + t / 2.
    ymin = tf.maximum(ymin, 0.0) / 512.0 * duration
    ymax = tf.minimum(ymax, 512.0) / 512.0 * duration
    
    maxid = tf.argmax(logitsprediction,axis=-1)
    maxidtile = tf.tile(maxid,(1,1,3))
    maxidtilef = tf.cast(maxidtile,tf.float32)

    maxscore = tf.reduce_max(logitsprediction,axis=-1)
    maxscoretile = tf.tile(maxscore,(1,1,3))

    bboxes = tf.stack([ymin, ymax, maxidtilef, maxscoretile], axis=-1)

    return bboxes

def tf_aher_bboxes_decode_detect_layer(feat_localizations,
                               duration,
                               anchors_layer,
                               propredictions,
                               prior_scaling=[1.0, 1.0]):
    """Compute the relative bounding boxes from the layer features and
    reference anchor bounding boxes.

    Arguments:
      feat_localizations: Tensor containing localization features.
      anchors: List of numpy array containing anchor boxes.
      

    Return:
      Tensor Nx2: ymin, ymax
    """
    yref, tref = anchors_layer

    # Compute center, height and width
    cy = feat_localizations[:, :, :, 0] * tref * prior_scaling[0]  + yref # tf.expand_dims(yref,-1)
    t = tref * tf.exp(feat_localizations[:, :, :, 1] * prior_scaling[1])
    # Boxes coordinates.
    ymin = cy - t / 2.
    ymax = cy + t / 2.
    ymin = tf.maximum(ymin, 0.0) / 512.0 * duration
    ymax = tf.minimum(ymax, 512.0) / 512.0 * duration
    
    #maxid = tf.argmax(propredictions,axis=-1)
    #maxidtile = tf.tile(maxid,(1,1,3))
    #maxidf = tf.cast(maxid,tf.float32)

    #maxscore = tf.reduce_max(propredictions,axis=-1)
    #maxscoretile = tf.tile(maxscore,(1,1,3))

    bboxes = tf.stack([ymin, ymax], axis=-1)
    bboxes = tf.concat([bboxes, propredictions],axis=-1)

    return bboxes                               

def tf_aher_bboxes_decode(feat_localizations,
                         duration,
                         anchors,
                         prior_scaling=[1.0,1.0],
                         scope='aher_bboxes_decode'):
    """Compute the relative bounding boxes from the net features and
    reference anchors bounding boxes.

    Arguments:
      feat_localizations: List of Tensors containing localization features.
      anchors: List of numpy array containing anchor boxes.

    Return:
      List of Tensors Nx2: ymin, ymax
    """
    with tf.name_scope(scope):
        bboxes = []
        for i, anchors_layer in enumerate(anchors):
            bboxes.append(
                tf_aher_bboxes_decode_layer(feat_localizations[i],
                                           duration,
                                           anchors_layer,
                                           prior_scaling))
        return bboxes

def tf_aher_bboxes_decode_logits(feat_localizations,
                         duration,
                         anchors,
                         logitsprediction,
                         prior_scaling=[1.0,1.0],
                         scope='aher_bboxes_decode'):
    """Compute the relative bounding boxes from the net features and
    reference anchors bounding boxes.

    Arguments:
      feat_localizations: List of Tensors containing localization features.
      anchors: List of numpy array containing anchor boxes.

    Return:
      List of Tensors Nx2: ymin, ymax
    """
    with tf.name_scope(scope):
        bboxes = []
        for i, anchors_layer in enumerate(anchors):
            bboxes.append(
                tf_aher_bboxes_decode_logits_layer(feat_localizations[i],
                                           duration,
                                           anchors_layer,
                                           logitsprediction[i],
                                           prior_scaling))
        return bboxes

def tf_aher_bboxes_decode_detect(feat_localizations,
                         duration,
                         anchors,
                         propprediction,
                         prior_scaling=[1.0,1.0],
                         scope='aher_bboxes_decode'):
    """Compute the relative bounding boxes from the net features and
    reference anchors bounding boxes.

    Arguments:
      feat_localizations: List of Tensors containing localization features.
      anchors: List of numpy array containing anchor boxes.

    Return:
      List of Tensors Nx2: ymin, ymax
    """
    with tf.name_scope(scope):
        bboxes = []
        for i, anchors_layer in enumerate(anchors):
            bboxes.append(
                tf_aher_bboxes_decode_detect_layer(feat_localizations[i],
                                           duration,
                                           anchors_layer,
                                           propprediction[i],
                                           prior_scaling))
        return bboxes

# =========================================================================== #
# temporal boxes selection.
# =========================================================================== #
def tf_aher_bboxes_select_layer(predictions_layer, localizations_layer,
                               select_threshold=None,
                               num_classes=21,
                               ignore_class=0,
                               scope=None,
                               IoU_flag=False):
    """Extract classes, scores and bounding boxes from features in one layer.
    Batch-compatible: inputs are supposed to have batch-type shapes.

    Args:
      predictions_layer: A prediction layer;
      localizations_layer: A localization layer;
      select_threshold: Classification threshold for selecting a box. All boxes
        under the threshold are set to 'zero'. If None, no threshold applied.
    Return:
      d_scores, d_bboxes: Dictionary of scores and bboxes Tensors of
        size Batches X N x 1 | 2. Each key corresponding to a class.
    """
    select_threshold = 0.0 if select_threshold is None else select_threshold
    with tf.name_scope(scope, 'aher_bboxes_select_layer',
                       [predictions_layer, localizations_layer]):
        # Reshape features: Batches x N x N_labels | 4
        p_shape = tfe.get_shape(predictions_layer)
        predictions_layer = tf.reshape(predictions_layer,
                                       tf.stack([p_shape[0], -1, p_shape[-1]]))
        if IoU_flag:
            zeros_m = tf.zeros([predictions_layer.shape[1],1])
            predictions_layer = tf.reshape(tf.stack([zeros_m, 
            tf.reshape(predictions_layer,[predictions_layer.shape[1],1])],axis=1),
            [predictions_layer.shape[0],predictions_layer.shape[1],2])
        l_shape = tfe.get_shape(localizations_layer)
        localizations_layer = tf.reshape(localizations_layer,
                              tf.stack([l_shape[0], -1, l_shape[-1]]))

        d_scores = {}
        d_bboxes = {}
        for c in range(0, num_classes):
            if c != ignore_class:
                # Remove boxes under the threshold.
                scores = predictions_layer[:, :, c]
                fmask = tf.cast(tf.greater_equal(scores, select_threshold), scores.dtype)
                scores = scores * fmask
                bboxes = localizations_layer * tf.expand_dims(fmask, axis=-1)
                # Append to dictionary.
                d_scores[c] = scores
                d_bboxes[c] = bboxes

        return d_scores, d_bboxes

def tf_aher_bboxes_select(predictions_net, localizations_net,
                         select_threshold=None,
                         num_classes=21,
                         ignore_class=0,
                         scope=None,
                         IoU_flag = False):
    """Extract classes, scores and bounding boxes from network output layers.
    Batch-compatible: inputs are supposed to have batch-type shapes.

    Args:
      predictions_net: List of prediction layers;
      localizations_net: List of localization layers;
      select_threshold: Classification threshold for selecting a box. All boxes
        under the threshold are set to 'zero'. If None, no threshold applied.
    Return:
      d_scores, d_bboxes: Dictionary of scores and bboxes Tensors of
        size Batches X N x 1 | 4. Each key corresponding to a class.
    """
    with tf.name_scope(scope, 'aher_bboxes_select',
                       [predictions_net, localizations_net]):
        l_scores = []
        l_bboxes = []
        for i in range(len(predictions_net)):
            scores, bboxes = tf_aher_bboxes_select_layer(predictions_net[i],
                                                        localizations_net[i],
                                                        select_threshold,
                                                        num_classes,
                                                        ignore_class,
                                                        IoU_flag = IoU_flag)
            l_scores.append(scores)
            l_bboxes.append(bboxes)
        # Concat results.
        d_scores = {}
        d_bboxes = {}
        for c in l_scores[0].keys():
            ls = [s[c] for s in l_scores]
            lb = [b[c] for b in l_bboxes]
            d_scores[c] = tf.concat(ls, axis=1)
            d_bboxes[c] = tf.concat(lb, axis=1)
        return d_scores, d_bboxes

def tf_aher_bboxes_select_layer_all_classes(predictions_layer, localizations_layer,
                                           select_threshold=None):
    """Extract classes, scores and bounding boxes from features in one layer.
     Batch-compatible: inputs are supposed to have batch-type shapes.

     Args:
       predictions_layer: A prediction layer;
       localizations_layer: A localization layer;
      select_threshold: Classification threshold for selecting a box. If None,
        select boxes whose classification score is higher than 'no class'.
     Return:
      classes, scores, bboxes: Input Tensors.
     """
    # Reshape features: Batches x N x N_labels | 4
    p_shape = tfe.get_shape(predictions_layer)
    predictions_layer = tf.reshape(predictions_layer,
                                   tf.stack([p_shape[0], -1, p_shape[-1]]))
    l_shape = tfe.get_shape(localizations_layer)
    localizations_layer = tf.reshape(localizations_layer,
                                     tf.stack([l_shape[0], -1, l_shape[-1]]))
    # Boxes selection: use threshold or score > no-label criteria.
    if select_threshold is None or select_threshold == 0:
        # Class prediction and scores: assign 0. to 0-class
        classes = tf.argmax(predictions_layer, axis=2)
        scores = tf.reduce_max(predictions_layer, axis=2)
        scores = scores * tf.cast(classes > 0, scores.dtype)
    else:
        sub_predictions = predictions_layer[:, :, 1:]
        classes = tf.argmax(sub_predictions, axis=2) + 1
        scores = tf.reduce_max(sub_predictions, axis=2)
        # Only keep predictions higher than threshold.
        mask = tf.greater(scores, select_threshold)
        classes = classes * tf.cast(mask, classes.dtype)
        scores = scores * tf.cast(mask, scores.dtype)
    # Assume localization layer already decoded.
    bboxes = localizations_layer
    return classes, scores, bboxes

def tf_aher_bboxes_select_all_classes(predictions_net, localizations_net,
                                     select_threshold=None,
                                     scope=None):
    """Extract classes, scores and bounding boxes from network output layers.
    Batch-compatible: inputs are supposed to have batch-type shapes.

    Args:
      predictions_net: List of prediction layers;
      localizations_net: List of localization layers;
      select_threshold: Classification threshold for selecting a box. If None,
        select boxes whose classification score is higher than 'no class'.
    Return:
      classes, scores, bboxes: Tensors.
    """
    with tf.name_scope(scope, 'aher_bboxes_select',
                       [predictions_net, localizations_net]):
        l_classes = []
        l_scores = []
        l_bboxes = []
        for i in range(len(predictions_net)):
            classes, scores, bboxes = \
                tf_aher_bboxes_select_layer_all_classes(predictions_net[i],
                                                       localizations_net[i],
                                                       select_threshold)
            l_classes.append(classes)
            l_scores.append(scores)
            l_bboxes.append(bboxes)

        classes = tf.concat(l_classes, axis=1)
        scores = tf.concat(l_scores, axis=1)
        bboxes = tf.concat(l_bboxes, axis=1)
        return classes, scores, bboxes

