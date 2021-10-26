import cv2
import threading
import tflite_runtime.interpreter as tflite
from utils.models import *
from glob import glob
from skimage import io
import time
import platform
from utils.herramientas import *
from utils.Dataset import *


EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

model_file="models/features_quant.tflite"
model_file, *device = model_file.split('@')
interpreter_features = tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])
interpreter_features.allocate_tensors()
interpreter_segmentation = tflite.Interpreter('models/segment_quant.tflite')
interpreter_segmentation.allocate_tensors()
interpreter_detection = tf.lite.Interpreter('models/detection_quant.tflite')
interpreter_detection.allocate_tensors()

path="video1/10000.png"
img=     io.imread(path)
image_ = np.expand_dims(img, 0)
features = features_image(interpreter_features,image_)
image_ = segment_image(interpreter_segmentation,features[0])
maskf=np.zeros(img.shape,dtype=np.uint8)
for i in zip(lbls.labelID,lbls.labels):
    ids=np.where(image_==i[0])[1:3]
    color=info_dataset[info_dataset.name==i[1]]["color"]
    maskf[ids[0],ids[1],:]=np.array(color.values[0])
overlay = img.copy()
alpha=0.3
cv2.addWeighted(maskf, alpha, img, 1 - alpha,0, overlay)
io.imsave("result_test/seg_"+path.split("/")[-1],overlay)

colors = colores()
image_ = detect_image(interpreter_detection,features)
image_=[K.constant(image_[0]),K.constant(image_[1])]
boxes, scores, clases=yolo_eval(image_,anchors,classes_det)
image_detection=np.zeros(img.shape,dtype=np.uint8)
image_detection=bbox_draw(img,det_class,boxes,clases,colors,scores)
io.imsave("result_test/det_"+path.split("/")[-1], image_detection)


