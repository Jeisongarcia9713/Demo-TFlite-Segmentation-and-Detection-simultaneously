import numpy as np


size=(448,448) #formato del tamaÃƒÂ±o es width and height
YOLO_ANCHORS= [[[11, 11],[8,16], [18,8]],
               [[17,15], [11, 25], [28,21]],       
               [[19,44], [48,35],  [79,79]]]

YOLO_TINY_ANCHORS=[[[19,  15], [12,  33],[22,   31]],
                   [[51,  23], [54,  40],[107,  93]],
                   [[0,    0], [0,    0],[0,     0]]]
                   
YOLO_STRIDES = [16, 32, 64]
strides = np.array(YOLO_STRIDES)
train_output_sizes = size[0] // strides
anchor_per_scale = 3
max_bbox_per_scale = 100
classesBBox=['traffic sign','bicycle','person','car','traffic light', 'truck', 'rider','motorcycle', 'bus']
cambio=[]
cambio.append(['static','dynamic','parking','trailer','rail track','train','ground','cargroup','persongroup'])
cambio.append(['obstacle','obstacle','road','truck','sydewalk','bus','road','car','person'])
num_classes=len(classesBBox)
classes_det =len(classesBBox)
anchors = (np.array(YOLO_TINY_ANCHORS).T/strides).T
#classes =classes_det
