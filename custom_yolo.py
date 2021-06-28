"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""


import os
import time
import argparse
import numpy as np

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO



def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    parser.add_argument(
        '-n', '--skip_frame', type=int, default=1,
        help="skip frame number"
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    
    cls_dict = get_cls_dict(args.category_num)
    vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)

    N_frame = args.skip_frame
    
    cap = cv2.VideoCapture(args.video)
    start = time.time()
    while(cap.isOpened()):
        ret, img = cap.read()
        if ret:
            if N_frame == args.skip_frame:
                boxes, confs, clss = trt_yolo.detect(img, 0.3)
                img = vis.draw_bboxes(img, boxes, confs, clss)
                cv2.imshow('video', img)
                N_frame = 0

            else:
                img = vis.draw_bboxes(img, boxes, confs, clss)
                cv2.imshow('video', img)
                
                N_frame += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break
    print('frame skip yolo time: {:.3f}sec'.format(time.time()-start))
    cap.release()
    cv2.destroyAllWindows()
    