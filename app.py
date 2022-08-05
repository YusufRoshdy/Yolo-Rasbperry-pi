#!/usr/bin/env python
from flask import Flask, flash, redirect, render_template, \
     request, url_for, Response
import time
from datetime import datetime
import cv2
import numpy as np

import yolov5

# fastestv2
import torch
import model.detector as model_fastest
import utils.utils
'''
# yolov7
import numpy as np
import torch
from numpy import random
import torch.nn as nn
from models.experimental import attempt_load as v7_attempt_load
from utils.plots import plot_one_box
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
'''

# onnx
from YOLOv7onnx import YOLOv7
from YOLOv7onnx.utils import class_names as yolov7_names


app = Flask(__name__)


stream_model = ''
models = {}
cap = None


from os.path import exists

if not exists('log.txt'):
    with open("log.txt", "w") as f:
        L = 'Time, Xmin, Ymin, Xmax, Ymax, Confidence, Class ID, Class Name\n'
        f.write(L)

# Append-adds at last
log_file = open("log.txt", "a")  # append mode
log_file.write("Today \n")
log_file.flush()
#file1.close()



@app.route('/')
def index():
    return render_template(
        'index.html',
        data=[
            {'name':'Yolov5n'},
            {'name':'Yolov5s'},
            {'name':'Yolov5m'},
            {'name':'Yolov5l'},
            {'name':'Yolov5x'},
            {'name':'Yolov-Fastestv2'},
            # {'name':'Yolov7n'},
            {'name':'yolov7_256x320.onnx'},
            {'name':'yolov7_256x480.onnx'},
            {'name':'yolov7_256x640.onnx'},
            {'name':'yolov7_384x640.onnx'},
            {'name':'yolov7_480x640.onnx'},
            {'name':'yolov7_640x640.onnx'},
            {'name':'yolov7_736x1280.onnx'},
            {'name':'yolov7-tiny_256x320.onnx'},
            {'name':'yolov7-tiny_256x480.onnx'},
            {'name':'yolov7-tiny_256x640.onnx'},
            {'name':'yolov7-tiny_384x640.onnx'},
            {'name':'yolov7-tiny_480x640.onnx'},
            {'name':'yolov7-tiny_640x640.onnx'},
            {'name':'yolov7-tiny_736x1280.onnx'},
        ]
    )

@app.route("/stream" , methods=['GET', 'POST'])
def stream():
	global stream_model
	stream_model = request.form.get('comp_select', default='yolov7_256x320.onnx')
	return render_template('stream.html')

def gen():
    """Video streaming generator function."""
    global stream_model, cap
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        try:
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(0)
            old_time = time.time()

            # Read until video is completed
            while(cap.isOpened()):
                # Capture frame-by-frame
                ret, frame = cap.read()
                if ret != True:
                    break
                new_time = time.time()
                fps = 1/(new_time-old_time)
                old_time = new_time

                if 'Yolov5' in stream_model:
                    frame = yolov5_process(frame, size=stream_model[-1])
                elif 'Yolov7n' in stream_model:
                    frame = yolov7_process(frame, size=stream_model[-1])
                elif 'Yolov-Fastestv2' == stream_model:
                    frame = yolo_fastest_process(frame, frame.shape[1], frame.shape[0])
                else:
                    frame = yolov7_onnx_process(frame)

                log_file.flush()

                fps = f'FPS:{fps: 0.2f}'
                cv2.putText(frame, fps, (7,70), font, 2, (0, 255, 0), 2)
                
                frame = cv2.imencode('.jpg', frame)[1].tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                rame = np.ones((512, 512))
                cv2.putText(frame, "Camera is not available", (7,70), font, 2, (0, 255, 0), 2)
                
                frame = cv2.imencode('.jpg', frame)[1].tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                cap.release()
                break
        except:
            frame = np.ones((512, 1024), dtype=np.int8)*254
            cv2.putText(frame, "Camera isn't available", (7,70), font, 2, (0, 255, 0), 2)
            
            frame = cv2.imencode('.jpg', frame)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            cap.release()
            break
        

def yolov5_process(frame, size='n'):
	global models
	
	model_name = f'yolov5{size}.pt'
	# load pretrained model
	if model_name in models:
		model = models[model_name]
	else:
		import yolov5
		model = yolov5.load(model_name)
		# set model parameters
		model.conf = 0.25  # NMS confidence threshold
		model.iou = 0.45  # NMS IoU threshold
		model.agnostic = False  # NMS class-agnostic
		model.multi_label = False  # NMS multiple labels per box
		model.max_det = 3  # maximum number of detections per image

		models[model_name] = model
		
	results = model(frame)
	if len(results.pandas().xyxy[0]) >0:
		now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
		for idx, row in results.pandas().xyxy[0].iterrows():
			line = f'{now}, {int(row.xmin):3d}, {int(row.ymin):3d}, {int(row.xmax):3d}, {int(row.ymax):3d}, {row.confidence:0.2f}, {int(row["class"]):2d}, {row["name"]}\n' 
			# print(line)
			log_file.write(line)
	new_frame = results.render()[0]
	return new_frame
	
	
def yolo_fastest_process(frame, w, h):
	global models
	
	model_name = f'yolovfastest.pt'
	# load pretrained model
	if model_name in models:
		(model, cfg, LABEL_NAMES, device, scale_h, scale_w) = models[model_name]
	else:
		weights = 'modelzoo/coco2017-0.241078ap-model.pth'
		weights = 'modelzoo/fastest1.pth'
		data = 'data/coco.data'
		cfg = utils.utils.load_datafile('data/coco.data')
		
		
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model = model_fastest.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
		model.load_state_dict(torch.load(weights, map_location=device))

		#sets the module in eval node
		model.eval()
		
		
		LABEL_NAMES = []
		with open(cfg["names"], 'r') as f:
			for line in f.readlines():
				LABEL_NAMES.append(line.strip())
				
		scale_h, scale_w = h / cfg["height"], w / cfg["width"]


		models[model_name] = (model, cfg, LABEL_NAMES, device, scale_h, scale_w)
		
	
	res_frame = cv2.resize(frame, (cfg["width"], cfg["height"]), interpolation = cv2.INTER_LINEAR) 
	temp_frame = res_frame.reshape(1, cfg["height"], cfg["width"], 3)
	temp_frame = torch.from_numpy(temp_frame.transpose(0,3, 1, 2))
	temp_frame = temp_frame.to(device).float() / 255.0
	
	preds = model(temp_frame)


	output = utils.utils.handel_preds(preds, cfg, device)
	output_boxes = utils.utils.non_max_suppression(output, conf_thres = 0.3, iou_thres = 0.4)

	

	now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
	for box in output_boxes[0]:
		box = box.tolist()
	   
		obj_score = box[4]
		category = LABEL_NAMES[int(box[5])]

		x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
		x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)

		cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
		cv2.putText(frame, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)	
		cv2.putText(frame, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)

		line = f'{now}, {x1:3d}, {y1:3d}, {x2:3d}, {y2:3d}, {obj_score:0.2f}, {int(box[5]):2d}, {category}\n'
		# print(line)
		log_file.write(line)
	return frame





def yolov7_onnx_process(frame, size='tiny_256x320'):
	global models

	model_name = f'yolov7-{size}.onnx'
	# load pretrained model
	if model_name in models:
		model = models[model_name]
	else:
		model_path = f"models/{model_name}"
		model = YOLOv7(model_path, conf_thres=0.5, iou_thres=0.5)

		models[model_name] = model

	# Update object localizer
	boxes, scores, class_ids = model(frame)
	combined_img = model.draw_detections(frame)

	now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
	for i in range(len(class_ids)):
		x1,y1,x2,y2 = boxes[i]
		score = scores[i]
		class_id = class_ids[i]
		line = f'{now}, {int(x1):3d}, {int(y1):3d}, {int(x2):3d}, {int(y2):3d}, {score:0.2f}, {class_id:2d}, {yolov7_names[class_id]}\n'
		# print(line)
		log_file.write(line)
	return combined_img










def yolov7_process(frame, size='n'):
	global models
	
	model_name = f'yolov7{size}.pt'
	# load pretrained model
	if model_name in models:
		(model, names, colors, img_size, stride, device) = models[model_name]
	else:
		# source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
		weights = 'yolov7/yolov7-tiny.pt'
		# Initialize
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# Load model
		model = v7_attempt_load(weights, map_location=device)  # load FP32 model
		stride = int(model.stride.max())  # model stride

		print('webcam')

		# Get names and colors
		names = model.module.names if hasattr(model, 'module') else model.names
		colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
		img_size = 640
		stride = 32
		models[model_name] = (model, names, colors, img_size, stride, device)
		
		
		
	img = letterbox(frame, img_size, stride=stride)[0]
	# Convert
	img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
	img = np.ascontiguousarray(img)

	img = torch.from_numpy(img).to(device)
	img = img.float()  # uint8 to fp16/32
	img /= 255.0  # 0 - 255 to 0.0 - 1.0
	if img.ndimension() == 3:
		img = img.unsqueeze(0)
	
	augment = 'store_true'
	pred = model(img, augment=augment)[0]

	# Apply NMS
	conf_thres = 0.25
	iou_thres = 0.45
	classes = None
	agnostic_nms = 'store_true'
	pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
	
	# Process detections
	for i, det in enumerate(pred):  # detections per image
		frame = frame.copy()
	
		gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]  # normalization gain whwh
		if len(det):
			# Rescale boxes from img_size to frame size
			det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

			# Write results
			for *xyxy, conf, cls in reversed(det):
					
				label = f'{names[int(cls)]} {conf:.2f}'
				plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
	return frame	







@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

    
if __name__ == '__main__':
	app.run(host='0.0.0.0', debug=False)
