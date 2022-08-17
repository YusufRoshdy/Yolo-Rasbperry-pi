
from multiprocessing.connection import Listener
from flask import Flask, flash, redirect, render_template, \
     request, url_for, Response, session

import time
from datetime import datetime
import cv2
import numpy as np

from pathlib import Path


custom_models = []

app = Flask(__name__)
first_run_flag = True

@app.route('/')
def index():
    global first_run_flag, custom_models
    if first_run_flag:
        first_run_flag = False
        # session['manager_dict'] = manager_dict
    
    data = [
            {'name': 'Yolov5n'},
            {'name': 'Yolov5s'},
            {'name': 'Yolov5m'},
            {'name': 'Yolov5l'},
            {'name': 'Yolov5x'},
            {'name': 'Yolov-Fastestv2'},
            # {'name':'Yolov7n'},
            {'name': 'yolov7_256x320.onnx'},
            {'name': 'yolov7_256x480.onnx'},
            {'name': 'yolov7_256x640.onnx'},
            {'name': 'yolov7_384x640.onnx'},
            {'name': 'yolov7_480x640.onnx'},
            {'name': 'yolov7_640x640.onnx'},
            {'name': 'yolov7_736x1280.onnx'},
            {'name': 'yolov7-tiny_256x320.onnx'},
            {'name': 'yolov7-tiny_256x480.onnx'},
            {'name': 'yolov7-tiny_256x640.onnx'},
            {'name': 'yolov7-tiny_384x640.onnx'},
            {'name': 'yolov7-tiny_480x640.onnx'},
            {'name': 'yolov7-tiny_640x640.onnx'},
            {'name': 'yolov7-tiny_736x1280.onnx'}
    ]
    for model_name in custom_models:
        data.append({'name': model_name})
    
    return render_template(
        'index.html',
        data=data,
    )

@app.route("/stream" , methods=['GET', 'POST'])
def stream():
    global stream_model, first_run_flag
    
    if first_run_flag:
        first_run_flag = False
        # session['manager_dict'] = manager_dict
        
    stream_model = request.form.get('comp_select', default='yolov7_256x320.onnx')
    
    conn.send(stream_model + ' 0')
    
    return render_template('stream.html')

def gen():
    """Video streaming generator function."""
    global stream_model

    font = cv2.FONT_HERSHEY_SIMPLEX
    frame = np.ones((512, 1024), dtype=np.int8)*254
    cv2.putText(frame, "Trying to load the camera", (7,70), font, 2, (0, 255, 0), 2)
    frame = cv2.imencode('.jpg', frame)[1].tobytes()

    conn.send(stream_model + ' 1')
    while True:
        received = False
        while conn.poll():
            frame = conn.recv_bytes()
            received = True
        if received:
            conn.send(stream_model + ' 1')
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    address = ('localhost', 6000)     # family is deduced to be 'AF_INET'
    listener = Listener(address, authkey=b'secret password')
    conn = listener.accept()
    print('connection accepted from', listener.last_accepted)

    for child in Path('custom').iterdir():
        if child.is_file():
            print(child.name, type(child.name))
            custom_models.append('custom/' + child.name)


    app.run('0.0.0.0', 5000, debug=False)