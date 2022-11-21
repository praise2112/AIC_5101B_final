import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from copy import copy
from enum import Enum
import urllib.request
import time
import argparse


def dataset_generator(dir: str = 'dataset'):
    PERSON_LABEL = 0
    annotations = os.path.join(dir,
                               'annotation_CROSS_X-F1-B1_P880043_20200625111459_225/CROSS_X-F1-B1_P880043_20200625111459_225/boxes_2d.csv')
    depth_dir = os.path.join(dir,
                             'depth_CROSS_X-F1-B1_P880043_20200625111459_225/CROSS_X-F1-B1_P880043_20200625111459_225')
    ir_dir = os.path.join(dir, 'ir_CROSS_X-F1-B1_P880043_20200625111459_225/CROSS_X-F1-B1_P880043_20200625111459_225')

    annotations = pd.read_csv(annotations, sep=';')
    depth_images = [f for f in os.listdir(depth_dir) if f.endswith('.png')]
    ir_images = [f for f in os.listdir(ir_dir) if f.endswith('.png')]

    print('Number of depth images:', len(depth_images))
    print('Number of infra red images:', len(ir_images))
    print('Number of annotations:', len(annotations))

    for depth_image_path, ir_image_path in zip(depth_images, ir_images):
        depth_image = cv2.imread(os.path.join(depth_dir, depth_image_path), cv2.IMREAD_UNCHANGED)
        # depth_image = cv2.imread(os.path.join(depth_dir, depth_image_path), cv2.IMREAD_UNCHANGED)
        ir_image = cv2.imread(os.path.join(ir_dir, ir_image_path), cv2.IMREAD_UNCHANGED)
        frame_number = int(os.path.basename(depth_image_path).split('.')[0][-5:])
        annotation = annotations[annotations['frame'] == frame_number]
        yield depth_image, ir_image, annotation, frame_number


def plot_image_with_annotation(depth_image, ir_image, annotation, frame_num, combined=False):
    if combined:
        # interpolate the depth and infra red image into one image using opencv
        combined_image = cv2.addWeighted(depth_image, 0.9, ir_image, 0.1, 0)
        fig, ax = plt.subplots(1)
        fig.suptitle(f'Frame {frame_num}')
        ax.set_title('Depth and Infra Red Image')
        ax.imshow(combined_image, cmap='gray')
        for _, row in annotation.iterrows():
            rect = Rectangle((row['x_min'], row['y_min']), row['x_max'] - row['x_min'], row['y_max'] - row['y_min'],
                             linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    else:
        fig, ax = plt.subplots(1, 2)
        fig.suptitle(f'Frame {frame_num}')

        ax[0].imshow(depth_image, cmap='gray')
        ax[0].set_title('Depth Image')

        ax[1].imshow(ir_image, cmap='gray')
        ax[1].set_title('IR Image')
        for _, row in annotation.iterrows():
            rect = Rectangle((row['x_min'], row['y_min']), row['x_max'] - row['x_min'], row['y_max'] - row['y_min'],
                             linewidth=1, edgecolor='r', facecolor='none')
            ax[0].add_patch(copy(rect))
            ax[1].add_patch(copy(rect))

    plt.show()


def person_detection_basic(grayscale_image, frame_num):
    # detects people in the image by converting to binary image, using morphological operations to remove noise and then
    # using contours to detect the people
    # Apply a gaussian blur to the image to reduce noise
    image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
    # Apply a threshold to the image to segment the foreground from the background
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # Apply dilation and erosion to the image to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    # apply morphology to the image to find the contours of the people in the image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    # convert to uint8
    image = np.uint8(image)
    # find contours
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    fig, ax = plt.subplots(1)
    fig.suptitle(f'Frame {frame_num}')
    ax.set_title('Depth and Infra Red Image (Basic Person Detection)')
    ax.imshow(grayscale_image, cmap='gray')
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        rect = Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()


def person_detection_HOG(image, frame_num):
    # detects people in the image using Histogram of Oriented Gradients (HOG) and Support Vector Machine (SVM)
    # use HOG and SVM to detect people
    _image = np.uint8(image)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    boxes, weights = hog.detectMultiScale(_image, winStride=(8, 8), padding=(32, 32), scale=1.05,
                                          useMeanshiftGrouping=False)
    fig, ax = plt.subplots(1)
    fig.suptitle(f'Frame {frame_num}')
    ax.set_title('Depth and Infra Red Image (HOG Person Detection)')
    ax.imshow(image, cmap='gray')
    for (x, y, w, h) in boxes:
        rect = Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()


LABELS = None


def person_detection_YOLO(image, frame_num, cfg_path, weights_path, labels_path, confidence_threshold=0.5,
                          nms_threshold=0.3):
    global LABELS
    if not LABELS:
        LABELS = open(labels_path).read().strip().split('\n')
    n_class = len(LABELS)
    COLORS = np.random.randint(0, 255, size=(n_class, 3), dtype='uint8')
    _image = np.uint8(image)
    (H, W) = _image.shape[:2]

    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(_image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layer_outputs = net.forward(ln)
    end = time.time()
    print('YOLO model casr {:.2f} second to predict'.format(end - start))

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[class_ids[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[class_ids[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    fig, ax = plt.subplots(1)
    fig.suptitle(f'Frame {frame_num}')
    ax.set_title('Depth and Infra Red Image (YOLO Person Detection)')
    ax.imshow(image, cmap='gray')
    plt.show()
    return len(idxs)


class PersonDetectionMethod(Enum):
    BASIC = 1
    HOG = 2
    YOLO = 3


# cfg cfg/yolov3.cfg
def person_detection(depth_image, ir_image, frame_num, cfg_path, weights_path, labels_path,
                     detection_method=PersonDetectionMethod.BASIC):
    image = cv2.addWeighted(depth_image, 0.9, ir_image, 0.1, 0)
    # convert to rgb
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if detection_method == PersonDetectionMethod.BASIC:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        person_detection_basic(image, frame_num)
    elif detection_method == PersonDetectionMethod.HOG:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        person_detection_HOG(image, frame_num)
    elif detection_method == PersonDetectionMethod.YOLO:
        person_detection_YOLO(image, frame_num, cfg_path, weights_path, labels_path)
    else:
        raise ValueError('Invalid detection method')


def main(method, show_combined):
    # data = [x for x in list(dataset_generator()) if x[3] in [115, 223, 160, 235]]
    # fig, ax = plt.subplots(2, 2)

    # for i, (depth_image, ir_image, annotation, frame_num) in enumerate(data):
    #     image = cv2.addWeighted(depth_image, 0.9, ir_image, 0.1, 0)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     ax[i // 2, i % 2].imshow(image, cmap='gray')
    #     ax[i // 2, i % 2].set_title(f'Frame {frame_num} ')
    #     for _, row in annotation.iterrows():
    #         rect = Rectangle((row['x_min'], row['y_min']), row['x_max'] - row['x_min'], row['y_max'] - row['y_min'],
    #                          linewidth=1, edgecolor='r', facecolor='none')
    #         ax[i // 2, i % 2].add_patch(rect)
    # plt.show()
    #

    for depth_image, ir_image, annotation, frame_num in dataset_generator():
        print(depth_image.shape, ir_image.shape, annotation)
        # if frame_num != 166:
        # if frame_num != 115:
        # if frame_num != 136:
        # if frame_num != 238:
        # if frame_num != 211:
        # continue
        plot_image_with_annotation(depth_image, ir_image, annotation, frame_num, combined=show_combined)
        person_detection(depth_image, ir_image, frame_num, detection_method=method,
                         cfg_path='yolov3.cfg', weights_path='yolov3.weights',
                         labels_path='coco.names')


def download_yolo_req():
    # check if file exists
    if not os.path.exists('yolov3.weights'):
        print('Downloading YOLO weights...')
        url = 'https://pjreddie.com/media/files/yolov3.weights'
        urllib.request.urlretrieve(url, 'yolov3.weights')
    if not os.path.exists('coco.names'):
        print('Downloading coco names...')
        url = 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
        urllib.request.urlretrieve(url, 'coco.names')
    if not os.path.exists('yolov3.cfg'):
        print('Downloading yolov3.cfg...')
        url = 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg'
        urllib.request.urlretrieve(url, 'yolov3.cfg')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Person Detection')
    parser.add_argument('-m', '--method', type=str, default='yolo', help='Person detection method')
    parser.add_argument('-c', '--combined', type=bool, default=True, help='Show combined image')
    args = parser.parse_args()
    download_yolo_req()
    main(method=PersonDetectionMethod[args.method.upper()], show_combined=args.combined)
