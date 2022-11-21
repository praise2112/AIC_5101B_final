#!/bin/bash
cd yolov5_train_test/yolov5 || exit
python3 prepare_yolo_data.py
python3 train.py --img 512 --batch 8 --epochs 1 --data person_detection.yaml --weights yolov5s.pt
