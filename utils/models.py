import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from utils.SegDataset import *
from utils.herramientas import *

lbls= pd.read_csv("utils/labels.txt")

def set_input_tensor(interpreter, input, Mobilenet=False):
  input_details = interpreter.get_input_details()[0]
  scale, zero_point = input_details['quantization']
  if Mobilenet:
    interpreter.set_tensor(input_details['index'], np.uint8(input / scale + zero_point))
  else:
    interpreter.set_tensor(input_details['index'], input)


def features_image(interpreter, input, Mobilenet=False):
  set_input_tensor(interpreter, input, Mobilenet)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = interpreter.get_tensor(output_details['index'])
  output_details1 = interpreter.get_output_details()[1]
  output1 = interpreter.get_tensor(output_details1['index'])
  return output, output1

def segment_image(interpreter, input):
  set_input_tensor(interpreter, input)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = interpreter.get_tensor(output_details['index'])
  top_1 = np.argmax(output, axis=-1)
  return top_1

def detect_image(interpreter, input):
  input_details = interpreter.get_input_details()[0]
  interpreter.set_tensor(input_details['index'], input[0])

  input_details1 = interpreter.get_input_details()[1]
  interpreter.set_tensor(input_details1['index'], input[1])

  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = interpreter.get_tensor(output_details['index'])
  output_details1 = interpreter.get_output_details()[1]
  output1 = interpreter.get_tensor(output_details1['index'])
  return output,output1


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3] # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),[1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),[grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs

def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape):
    '''Process Conv layer output'''
    # Activaciones capa Convolucional
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,anchors, num_classes, input_shape)
    # Escale los cuadros de nuevo a la forma original de la imagen (x_min,y_min,x_max,y_max)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores

def yolo_correct_boxes(box_xy, box_wh, input_shape):
    # Indica que para (...,) todas las dimensiones - (::-1) recorriendo todas las posciciones con un paso de (-1) orden inverso
    box_yx = box_xy[..., ::-1] 
    box_hw = box_wh[..., ::-1]

    input_shape = K.cast(input_shape, K.dtype(box_yx))
    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])
    # Escale los cuadros de nuevo a la forma original de la imagen.
    boxes *= K.concatenate([input_shape, input_shape])
    return boxes

def yolo_eval(yolo_outputs,anchors,num_classes,max_boxes=20,score_threshold=.3,iou_threshold=.45):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = 2
    anchor_mask = [0,1]
    input_shape = K.shape(yolo_outputs[0])[1:3] * 16
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],np.array(anchors[anchor_mask[l]]), num_classes, input_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

        nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_

def bbox_draw(image,classesBBox,out_boxes,out_classes,colors,out_scores=None,etq=False,linewidth=None,text_size=4):
  if linewidth == None:  
    thickness = (image.shape[0] + image.shape[1]) // 600
  else:
    thickness = linewidth
  fontScale=1
  ObjectsList = []
  image2 = image.copy()
  for i, c in reversed(list(enumerate(out_classes))):
      predicted_class = classesBBox[c]
      box = out_boxes[i]
      if not etq:
        score = out_scores[i]
        label = '{} {:.2f}'.format(str(i)+": "+predicted_class, score)
        scores = '{:.2f}'.format(score)
        top, left, bottom, right = box
      else:
        label = str(i)+": "+predicted_class
        left, top, right, bottom = box
      top = max(0, np.floor(top + 0.5).astype('int32'))
      left = max(0, np.floor(left + 0.5).astype('int32'))
      bottom = min(image2.shape[0], np.floor(bottom + 0.5).astype('int32'))
      right = min(image2.shape[1], np.floor(right + 0.5).astype('int32'))

      mid_h = (bottom-top)/2+top
      mid_v = (right-left)/2+left

      # put object rectangle
      cv2.rectangle(image2, (left, top), (right, bottom), colors[c], thickness)
      # get text size
      (test_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, thickness/text_size, 1)
      # put text rectangle
      cv2.rectangle(image2, (left, top), (left + test_width, top - text_height - baseline), colors[c], thickness=cv2.FILLED)
      # put text above rectangle
      cv2.putText(image2, label, (left, top-2), cv2.FONT_HERSHEY_SIMPLEX, thickness/text_size, (0, 0, 0), 1)
  return image2