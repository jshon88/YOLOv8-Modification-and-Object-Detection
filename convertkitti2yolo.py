# ref: https://www.kaggle.com/code/ibrahimalobaid/convert-kitti-dataset-to-yolo-format

import numpy as np
import pandas as pd
import os
import shutil
import cv2
import yaml
from sklearn.model_selection import train_test_split

# Define paths
kitti_base_path = '/data/cmpe258-sp24/017553289/cmpe249/dataset/Kitti/training'
yolo_base_path = '/data/cmpe258-sp24/017553289/cmpe249/dataset/Kitti/'  # Use a writable directory
images_path = os.path.join(kitti_base_path, 'image_2')
labels_path = os.path.join(kitti_base_path, 'label_2')
yolo_images_path = os.path.join(yolo_base_path, 'image_2_yolo')
yolo_labels_path = os.path.join(yolo_base_path, 'label_2_yolo')

# Create YOLO directories
os.makedirs(yolo_images_path, exist_ok=True)
os.makedirs(yolo_labels_path, exist_ok=True)

kitti_classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']

# Define function to convert KITTI bbox to YOLO bbox
def convert_bbox(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

# Process each label file
for label_file in os.listdir(labels_path):
    if not label_file.endswith('.txt'):
        continue
    image_file = label_file.replace('.txt', '.png')

    image = cv2.imread(os.path.join(images_path, image_file))
    h, w, _ = image.shape

    with open(os.path.join(labels_path, label_file), 'r') as lf:
        lines = lf.readlines()
    yolo_labels = []
    for line in lines:
        elements = line.strip().split(' ')
        class_id = elements[0]
        if class_id in kitti_classes:
            class_id = kitti_classes.index(class_id)  # In KITTI, the first element is the class
            xmin, ymin, xmax, ymax = map(float, elements[4:8])
            bbox = convert_bbox((w, h), (xmin, xmax, ymin, ymax))
            yolo_labels.append(f"{class_id} {' '.join(map(str, bbox))}\n")

    with open(os.path.join(yolo_labels_path, label_file), 'w') as yf:
        yf.writelines(yolo_labels)
    # Copy image to YOLO directory
    shutil.copy(os.path.join(images_path, image_file), yolo_images_path)

# Split dataset into train, val, test sets
all_images = [f for f in os.listdir(yolo_images_path) if f.endswith('.png')]
train_images, val_images = train_test_split(all_images, test_size=0.1, random_state=42)
# val_images, test_images = train_test_split(test_images, test_size=0.5, random_state=42)

# Function to move files to appropriate directories
def move_files(file_list, dest_dir):
    os.makedirs(os.path.join(dest_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'labels'), exist_ok=True)
    for file_name in file_list:
        shutil.move(os.path.join(yolo_images_path, file_name), os.path.join(dest_dir, 'images', file_name))
        label_file = file_name.replace('.png', '.txt')
        shutil.move(os.path.join(yolo_labels_path, label_file), os.path.join(dest_dir, 'labels', label_file))

# Move files to train, val, test directories
move_files(train_images, os.path.join(yolo_base_path, 'train'))
move_files(val_images, os.path.join(yolo_base_path, 'val'))
# move_files(test_images, os.path.join(yolo_base_path, 'test'))

print("Conversion and splitting complete.")



# Define the data dictionary
data = {
    'train': 
        {'root': '/data/cmpe258-sp24/017553289/cmpe249/dataset/Kitti/',
         'split': 'train',
         'image_dir': 'image_2',
         'labels_dir': 'label_2'},
    'val': 
        {'root' : '/data/cmpe258-sp24/017553289/cmpe249/dataset/Kitti/',
         'split': 'val',
         'image_dir': 'image_2',
         'labels_dir': 'label_2'},
    'nc': 8,  # number of classes
    'names': ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
}

# Save to data.yaml
with open('/data/cmpe258-sp24/017553289/cmpe249/dataset/Kitti/kitti.yaml', 'w') as outfile:
    yaml.dump(data, outfile, default_flow_style=False)
