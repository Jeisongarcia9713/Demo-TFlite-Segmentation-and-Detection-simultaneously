from flask import Flask, render_template, Response
import cv2
import threading
from glob import glob
from skimage import io
import time

app = Flask(__name__)
type_=0
num_video=0
paths_video=["/home/jeison/Videos/Tesis/video1/","/home/jeison/Videos/Tesis/video2/"]
videos=[]
frame_video=0
image =None 

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

def Segmentation():
    pass

def Detection():
    pass


def gen_frames():  # generate frame by frame from camera
    global image
    while True:
        try:
            ret, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  
        except:
            pass

def gen_frames_video():  # generate frame by frame from camera
    global num_video
    global frame_video
    global image
    while True:
        start = time.time()
        if(num_video==0):
            if(frame_video==345):
                frame_video=0
            image=videos[0][frame_video];
        else:
            if(frame_video==295):
                frame_video=0
            image=videos[1][frame_video];
        
        frame_video=frame_video+1;        
        image1=image.copy()
        cv2.putText(image1,"frame :"+str(frame_video),(150,150),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        ret, buffer = cv2.imencode('.jpg', cv2.cvtColor(image1, cv2.COLOR_RGB2BGR))
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 
        end = time.time()
        try:
            time.sleep(0.05-(end - start))
        except:
            pass

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


if __name__ == '__main__':
    ReadVideos()
    thread1 = threading.Thread(target=Segmentation)
    thread2 = threading.Thread(target=Detection)
    thread1.start()
    thread2.start()
    app.run()
    
