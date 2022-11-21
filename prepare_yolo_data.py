import os
import time

import cv2
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
from sklearn import preprocessing



def annotation_to_yolo_format(annotation, image):
    # convert annotation to yolo format
    # annotation: x_min y_min x_max y_max area
    # new_annotation: x_center y_center width height
    # Normalize x_center, y_center, width, height by image width and height

    x_center = (annotation['x_min'] + annotation['x_max']) / 2
    y_center = (annotation['y_min'] + annotation['y_max']) / 2
    width = annotation['x_max'] - annotation['x_min']
    height = annotation['y_max'] - annotation['y_min']
    # normalize between 0 and 1
    x_center = x_center / image.shape[1]
    y_center = y_center / image.shape[0]
    width = width / image.shape[1]
    height = height / image.shape[0]

    # set negative width and height to 0
    width[width < 0] = 0
    height[height < 0] = 0
    # normalize x_center, y_center, width, height between 0 and 1
    min_max_scaler = preprocessing.MinMaxScaler()
    # x_center = min_max_scaler.fit_transform(x_center.values.reshape(-1, 1))
    # y_center = min_max_scaler.fit_transform(y_center.values.reshape(-1, 1))
    # width = min_max_scaler.fit_transform(width.values.reshape(-1, 1))
    # height = min_max_scaler.fit_transform(height.values.reshape(-1, 1))


    # plot the bounding boxes and center
    # rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # for i in range(len(width)):
    #     cv2.rectangle(rgb_image, (int(x_center.values[i] - width.values[i] / 2),
    #                             int(y_center.values[i] - height.values[i] / 2)),
    #                     (int(x_center.values[i] + width.values[i] / 2),
    #                      int(y_center.values[i] + height.values[i] / 2)), (0, 255, 0), 2)
    #     print((int(x_center.values[i]), int(y_center.values[i])))
    #     cv2.circle(rgb_image, (int(x_center.values[i]), int(y_center.values[i])), 2, (0, 0, 255), 2)
    # plt.imshow(rgb_image)
    # plt.show()
    PERSON_CLASS = 0
    person_class_col = [PERSON_CLASS] * len(x_center)
    # df = pd.DataFrame({'x_center': x_center, 'y_center': y_center, 'width': width, 'height': height})
    # df_scaled = pd.DataFrame(min_max_scaler.fit_transform(df), columns=df.columns)
    # df_scaled['class'] = person_class_col
    # df_scaled.insert(0, 'class', df_scaled.pop('class'))


    # normalized_df = (df - df.mean()) / df.std()
    # normalized_df['class'] = person_class_col
    # normalized_df.insert(0, 'class', normalized_df.pop('class'))
    return pd.DataFrame({'class': person_class_col, 'x_center': x_center, 'y_center': y_center, 'width': width, 'height': height, })
    # return df_scalded


def prepare_yolo_data(overwrite=False):
    dataset_name = 'person_detection'
    input_dir = 'dataset'
    output_dir = f'yolov5_train_test/datasets/{dataset_name}'
    yolo_repo_dir = 'yolov5_train_test/yolov5'
    if overwrite and os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'test'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)

    annotations = os.path.join(input_dir,
                               'annotation_CROSS_X-F1-B1_P880043_20200625111459_225/CROSS_X-F1-B1_P880043_20200625111459_225/boxes_2d.csv')
    depth_dir = os.path.join(input_dir,
                             'depth_CROSS_X-F1-B1_P880043_20200625111459_225/CROSS_X-F1-B1_P880043_20200625111459_225')
    ir_dir = os.path.join(input_dir,
                          'ir_CROSS_X-F1-B1_P880043_20200625111459_225/CROSS_X-F1-B1_P880043_20200625111459_225')
    annotations = pd.read_csv(annotations, sep=';')

    depth_images = [f for f in os.listdir(depth_dir) if f.endswith('.png')]
    ir_images = [f for f in os.listdir(ir_dir) if f.endswith('.png')]
    print(f'Found {len(depth_images)} depth images and {len(ir_images)} ir images')

    dataset = []

    for depth_image_path, ir_image_path in zip(depth_images, ir_images):
        depth_image = cv2.imread(os.path.join(depth_dir, depth_image_path), cv2.IMREAD_UNCHANGED)
        ir_image = cv2.imread(os.path.join(ir_dir, ir_image_path), cv2.IMREAD_UNCHANGED)
        combined_image = cv2.addWeighted(depth_image, 0.9, ir_image, 0.1, 0)
        # combined_image = cv2.cvtColor(combined_image, cv2.COLOR_GRAY2RGB)
        # ret, combined_image = cv2.threshold(combined_image, 0, 255, cv2.THRESH_BINARY)
        # plt.imshow(combined_image)
        # plt.show()
        # plt.imshow(cv2.addWeighted(depth_image, 0.9, ir_image, 0.1, 0), cmap='gray')
        # plt.show()
        frame_number = int(os.path.basename(depth_image_path).split('.')[0][-5:])
        annotation = annotations[annotations['frame'] == frame_number]
        annotation = annotation_to_yolo_format(annotation, combined_image)
        dataset.append([combined_image, annotation, frame_number])
    # split to train, test, val
    train, test = train_test_split(dataset, test_size=0.15, random_state=42)
    train, val = train_test_split(train, test_size=0.1, random_state=42)

    # save train dataset
    for image, annotation, frame_number in train:
        # save image as grayscale
        plt.imsave(os.path.join(output_dir, 'images', 'train', f'{frame_number}.png'), image, cmap='gray')
        # cv2.imwrite(os.path.join(output_dir, 'images', 'train', f'{frame_number}.png'), image)
        annotation.to_csv(os.path.join(output_dir, 'labels', 'train', f'{frame_number}.txt'), index=False, header=False,
                          sep=' ')

    # save test dataset
    for image, annotation, frame_number in test:
        plt.imsave(os.path.join(output_dir, 'images', 'test', f'{frame_number}.png'), image, cmap='gray')
        # cv2.imwrite(os.path.join(output_dir, 'images', 'test', f'{frame_number}.png'), image)
        annotation.to_csv(os.path.join(output_dir, 'labels', 'test', f'{frame_number}.txt'), index=False, header=False,
                          sep=' ')

    # save val dataset
    for image, annotation, frame_number in val:
        plt.imsave(os.path.join(output_dir, 'images', 'val', f'{frame_number}.png'), image, cmap='gray')
        # cv2.imwrite(os.path.join(output_dir, 'images', 'val', f'{frame_number}.png'), image)
        annotation.to_csv(os.path.join(output_dir, 'labels', 'val', f'{frame_number}.txt'), index=False, header=False,
                          sep=' ')

    # create training yaml file
    with open(os.path.join(yolo_repo_dir, 'data', f'{dataset_name}.yaml'), 'w') as f:
        f.write(f"""path: ../datasets/{dataset_name}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: person""")

    print(f'Yolo data prepared in {output_dir}')
    print(f'Train: {len(train)}')
    print(f'Val: {len(val)}')
    print(f'Test: {len(test)}')
    #  python3 train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov5s.pt
    #  python3 train.py --img 640 --batch 8 --epochs 3 --data person_detection.yaml --weights yolov5s.pt


def delete_ini_files(output_dir='yolov5_train_test'):
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file == 'desktop.ini':
                print(f'Deleting {os.path.join(root, file)}')
                os.remove(os.path.join(root, file))


if __name__ == "__main__":
    prepare_yolo_data(overwrite=True)
    # delete_ini_files()
