from flask import Flask, render_template, Response
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

app = Flask(__name__)
type_=0
num_video=0
paths_video=["video1/","video2/"]
videos=[]
start=False
start_features=False
start_segm=False
start_det=False


frame_video=0
image =None 
image_segmentation =None 
mask_seg=None
image_detection =None 
image_fusion=None
features= None
time_features=0

model_file="models/features_quant_edgetpu.tflite"
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

@app.route('/video_feed/<type>')
def video_type(type=None):
    global type_
    if "+" in type:
        if type_<2:
            type_=type_+1
        else:
            type_=0
    else:
        if type_>1:
            type_=type_-1
        else:
            type_=2
    return Response("ok")

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video/<type>')
def video_type_(type=None):
    global num_video
    global frame_video
    if "+" in type:
        if num_video==0:
            num_video=1
        else:
            num_video=0
    else:
        if num_video==0:
            num_video=1
        else:
            num_video=0
    frame_video=0;
    return Response("ok")

@app.route('/video')
def video():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def flask_server():
    app.run(host="0.0.0.0")
    
    
def ReadVideos():
    for url in paths_video:
        video=[]
        print(url+"*.png")
        lista=glob(url+"*.png")

        lista.sort()
        for img in lista:
            image=io.imread(img)
            video.append(image)
        videos.append(video)

def Features():
    global image
    global image_segmentation
    global features
    global start
    global start_features
    global time_features
    while True:
        if(start):
            
            img= image.copy()
            image_ = np.expand_dims(img, 0)
            time1=time.perf_counter()
            features = features_image(interpreter_features,image_)
            time_features=time1-time.perf_counter()
            start_features=True
            
            

def Segmentation():
    global image
    global image_segmentation
    global features
    global start_features
    global start_segm
    global mask_seg
    global time_features
    while True:
        if(start_features):
            img= image.copy()
            time1=time.perf_counter()
            image_ = segment_image(interpreter_segmentation,features[0])
            timeseg=time_features+time.perf_counter()-time1
            mask_seg=np.zeros(image.shape,dtype=np.uint8)
            for i in zip(lbls.labelID,lbls.labels):
                ids=np.where(image_==i[0])[1:3]
                color=info_dataset[info_dataset.name==i[1]]["color"]
                mask_seg[ids[0],ids[1],:]=np.array(color.values[0])
            alpha=0.3
            image_segmentation=np.zeros(img.shape,dtype=np.uint8)
            cv2.addWeighted(mask_seg, alpha, img, 1 - alpha,0, image_segmentation) 
            cv2.putText(image_segmentation,"FPS :"+str(1/timeseg),(150,150),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
            start_segm=True  

def Detection():
    global image
    global image_detection
    global features
    global start_features
    global start_det
    global image_fusion
    colors = colores()
    while True:
        if(start_features):
            img= image.copy()
            time1=time.perf_counter()
            image_ = detect_image(interpreter_detection,features)
            timedet=time_features+time.perf_counter()-time1
            image_=[K.constant(image_[0]),K.constant(image_[1])]
            boxes, scores, clases=yolo_eval(image_,anchors,classes_det)
            image_detection=np.zeros(img.shape,dtype=np.uint8)
            image_detection=bbox_draw(img,det_class,boxes,clases,colors,scores)
            image_fusion=np.zeros(image.shape,dtype=np.uint8)
            alpha=0.7
            cv2.addWeighted(image_detection, alpha,mask_seg, 1 - alpha,0, image_fusion)
            cv2.putText(image_detection,"FPS :"+str(1/timedet),(150,150),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)   
            start_det=True                

def gen_frames():  # generate frame by frame from camera
    global image_segmentation
    global image_fusion
    global image_detection
    global image
    global type_
    global start
    global start_segm
    global start_det
    image_choose=None
    new=0
    while True:
        retu=True
        try:
            if(start):
                if(start_segm and type_ ==1):
                        image_choose=image_segmentation
                        start_segm=False
                        new=1
                elif(start_det and type_==0):
                        new=1
                        image_choose=image_detection
                        start_det= False
                elif(start_det and type_==2):
                        image_choose=image_fusion
                        new=1
                else:
                    if new==0:
                        image_choose=image

                ret, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_choose, cv2.COLOR_RGB2BGR))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 
        except:
            pass

def gen_frames_video():  # generate frame by frame from camera
    global num_video
    global frame_video
    global image
    global start
    while True:
        start_ = time.perf_counter()
        if(num_video==0):
            if(frame_video==345*2):
                frame_video=0
            image=videos[0][frame_video//2];
        else:
            if(frame_video==295*2):
                frame_video=0
            image=videos[1][frame_video//2];
        start=True
        frame_video=frame_video+1;        
        image1=image.copy()
        cv2.putText(image1,"frame :"+str(frame_video),(150,150),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        ret, buffer = cv2.imencode('.jpg', cv2.cvtColor(image1, cv2.COLOR_RGB2BGR))
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 
        end = time.perf_counter()
        try:
            time.sleep(0.05-(end - start_))
        except:
            pass


if __name__ == '__main__':
    ReadVideos()

    thread1 = threading.Thread(target=Features)
    thread2 = threading.Thread(target=Detection)
    thread3 = threading.Thread(target=Segmentation)
    thread4 = threading.Thread(target=flask_server)
    
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    
    
    
