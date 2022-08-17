#!/usr/bin/env python

from multiprocessing import Process, Manager
from multiprocessing.connection import Client

import time
from datetime import datetime
import cv2
import numpy as np

import yolov5

# fastestv2
import torch
import model.detector as model_fastest
import utils.utils

# onnx
from YOLOv7onnx import YOLOv7
from YOLOv7onnx.utils import class_names as yolov7_names

from os.path import exists


mode = 'a'
if not exists('log.txt'):
    mode = 'w'
log_file = open("log.txt", mode)

log_file.write(f'\nstarting connection at {datetime.now()}')
try:
    address = ('localhost', 6000)
    connection = Client(address, authkey=b'secret password')
except Exception as e:
    log_file.write('\tConnection failed:', e)
    exit()
log_file.write('\tConnected')
log_file.close()
    
if not exists('detection log.txt'):
    with open("detection log.txt", "w") as f:
        L = 'Time, Xmin, Ymin, Xmax, Ymax, Confidence, Class ID, Class Name\n'
        f.write(L)

# Append-adds at last
detection_log_file = open("detection log.txt", "a")  # append mode
#file1.close()


font = cv2.FONT_HERSHEY_SIMPLEX
cap = None
models = {}
stream_model = "yolov7-tiny_256x320.onnx"

def send_frame(frame):
    frame = cv2.imencode('.jpg', frame)[1].tobytes()
    connection.send_bytes(frame)

def camera_process():
    global cap, stream_model
    while True:
        try:
            print('opening cap')
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture('How To See Germs Spread Experiment (Coronavirus).mp4')
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
            old_time = time.time()

            # Read until video is completed
            send_flag = '0'
            while (cap.isOpened()):
                # Capture frame-by-frame
                ret, frame = cap.read()
                if ret != True:
                    print('no frame')
                    break
                
                new_time = time.time()
                fps = 1 / (new_time - old_time)
                old_time = new_time

                # check if a new model is chosen
                # print('polling connection')
                
                
                if connection.poll():
                    stream_model, send_flag = connection.recv().split()
                    # print(f'\tReceived data: {stream_model=}, {send_flag=}')
                
                time_begin = time.time()
                if 'custom' in stream_model.lower():
                    # print(stream_model)

                    if 'yolov5' in stream_model:
                        frame = yolov5_process(frame, model_name=stream_model)
                    
                    elif 'yolov-fastestv2' == stream_model.lower():
                        frame = yolo_fastest_process(frame, frame.shape[1], frame.shape[0], weights=stream_model)
                        
                    elif 'yolov7' in stream_model.lower():
                        frame = yolov7_onnx_process(frame, model_name=stream_model)
                                        
                elif 'Yolov5' in stream_model:
                    frame = yolov5_process(frame, model_name=f'yolov5{stream_model[-1]}.pt')
                    
                elif 'Yolov-Fastestv2' == stream_model:
                    frame = yolo_fastest_process(frame, frame.shape[1], frame.shape[0])
                    
                elif 'yolov7' in stream_model:
                    frame = yolov7_onnx_process(frame, model_name=f"models/{stream_model}")

                # detection_log_file.flush()

                fps = f'FPS: {fps: 0.2f}'
                cv2.putText(frame, fps, (7, 70), font, 2, (0, 255, 0), 2)
                time_end = time.time()
                # print('Processing time:', time_end - time_begin)

                # send the frame
                if send_flag == '1':
                    # print('sending..')
                    send_frame(frame)
                    # print('\tsent', fps)
                    send_flag = '0'
                else:
                    print('No need to send', fps)
                # frame = cv2.imencode('.jpg', frame)[1].tobytes()
                # yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                frame = np.ones((512, 1024))
                cv2.putText(frame, "Camera is not available", (7, 70), font, 2, (0, 255, 0), 2)
                send_frame(frame)
        except Exception as e:
            print(e)
            frame = np.ones((512, 1024), dtype=np.int8) * 254
            cv2.putText(frame, "Camera is not available", (7, 70), font, 2, (0, 255, 0), 2)
            send_frame(frame)

            # frame = cv2.imencode('.jpg', frame)[1].tobytes()
            # yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def yolov5_process(frame, model_name=f'yolov5n.pt'):
    global models

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
    if len(results.pandas().xyxy[0]) > 0:
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        for idx, row in results.pandas().xyxy[0].iterrows():
            line = f'{now}, {int(row.xmin):3d}, {int(row.ymin):3d}, {int(row.xmax):3d}, {int(row.ymax):3d}, {row.confidence:0.2f}, {int(row["class"]):2d}, {row["name"]}\n'
            # print(line)
            detection_log_file.write(line)
    new_frame = results.render()[0]
    return new_frame


def yolo_fastest_process(frame, w, h, weights='modelzoo/fastest1.pth', datafile='data/coco.data'):
    global models

    model_name = f'yolovfastest.pt'
    # load pretrained model
    if model_name in models:
        (model, cfg, LABEL_NAMES, device, scale_h, scale_w) = models[model_name]
    else:
        # weights = 'modelzoo/coco2017-0.241078ap-model.pth'
        # weights = 'modelzoo/fastest1.pth'
        data = 'data/coco.data'
        cfg = utils.utils.load_datafile(datafile)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model_fastest.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
        model.load_state_dict(torch.load(weights, map_location=device))

        # sets the module in eval node
        model.eval()

        LABEL_NAMES = []
        with open(cfg["names"], 'r') as f:
            for line in f.readlines():
                LABEL_NAMES.append(line.strip())

        scale_h, scale_w = h / cfg["height"], w / cfg["width"]

        models[model_name] = (model, cfg, LABEL_NAMES, device, scale_h, scale_w)

    res_frame = cv2.resize(frame, (cfg["width"], cfg["height"]), interpolation=cv2.INTER_LINEAR)
    temp_frame = res_frame.reshape(1, cfg["height"], cfg["width"], 3)
    temp_frame = torch.from_numpy(temp_frame.transpose(0, 3, 1, 2))
    temp_frame = temp_frame.to(device).float() / 255.0

    preds = model(temp_frame)

    output = utils.utils.handel_preds(preds, cfg, device)
    output_boxes = utils.utils.non_max_suppression(output, conf_thres=0.3, iou_thres=0.4)

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
        detection_log_file.write(line)
    return frame


def yolov7_onnx_process(frame, model_name="models/yolov7-tiny_256x320.onnx"):
    global models

    # load pretrained model
    if model_name in models:
        model = models[model_name]
    else:
        model = YOLOv7(model_name, conf_thres=0.5, iou_thres=0.5)

        models[model_name] = model

    # Update object localizer
    boxes, scores, class_ids = model(frame)
    combined_img = model.draw_detections(frame)

    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    for i in range(len(class_ids)):
        x1, y1, x2, y2 = boxes[i]
        score = scores[i]
        class_id = class_ids[i]
        line = f'{now}, {int(x1):3d}, {int(y1):3d}, {int(x2):3d}, {int(y2):3d}, {score:0.2f}, {class_id:2d}, {yolov7_names[class_id]}\n'
        # print(line)
        detection_log_file.write(line)
    return combined_img


if __name__ == '__main__':
    try:
        camera_process()
    except Exception as e:
        with open("log.txt", "a") as f:
            f.write('Crash: ' + str(e))