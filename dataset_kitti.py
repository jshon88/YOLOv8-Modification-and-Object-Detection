import torch
import torchvision
import numpy as np
from torchvision import datasets, transforms
#from DeepDataMiningLearning.detection.coco_utils import get_coco
# from DeepDataMiningLearning.detection import trainutils
import os
from typing import Any, Callable, List, Optional, Tuple, Dict
from PIL import Image
import cv2
import csv
from pathlib import Path

WrapNewDict = False

class KittiDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 root: str,
                 train: bool = True,
                 split: str = 'train', #'val' 'test'
                 transform: Optional[Callable] = None,
                 image_dir: str = "image_2", 
                 labels_dir: str = "label_2"):
        self.images = []
        self.targets = []
        self.root = root
        self.train = train
        self.transform = transform
        # self._location = "train" if self.train else "val"
        self._location = split
        self.image_dir_name = image_dir
        self.labels_dir_name = labels_dir
        # load all image files, sorting them to
        # ensure that they are aligned
        self.split = split
        split_dir = Path(self.root) / (self.split + '.txt') #select Kitti/val.txt
        #sample_id_list: str list
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None
        self.root_split_path = os.path.join(self.root, self._location)
        
        # image_dir = os.path.join(self.root, "raw", self._location, self.image_dir_name)
        # if self.train:
        #     labels_dir = os.path.join(self.root, "raw", self._location, self.labels_dir_name)
        # for img_file in os.listdir(image_dir):
        #     self.images.append(os.path.join(image_dir, img_file))
        #     if self.train:
        #         self.targets.append(os.path.join(labels_dir, f"{img_file.split('.')[0]}.txt"))
        #self.imgs = list(sorted(os.listdir(os.path.join(self.root, "PNGImages"))))
        self.INSTANCE_CATEGORY_NAMES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']
        self.INSTANCE2id = {'Car': 0, 'Van': 1, 'Truck': 2, 'Pedestrian':3, 'Person_sitting':4, 'Cyclist':5, 'Tram':6, 'Misc':7, 'DontCare':8} 
        self.id2INSTANCE = {v: k for k, v in self.INSTANCE2id.items()}
        self.numclass = 8 # excluding the 'DontCare'

    def get_image(self, idx):
        img_file = Path(self.root_split_path) / self.image_dir_name / ('%s.png' % idx)
        assert img_file.exists()
        # image = Image.open(img_file)
        image = Image.open(img_file)
        return image
    
    def get_label(self, idx):
        label_file = Path(self.root_split_path) / self.labels_dir_name / ('%s.txt' % idx)
        assert label_file.exists()
        target = []
        with open(label_file) as inp:
            content = csv.reader(inp, delimiter=" ")
            for line in content:
                target.append(
                    {
                        "image_id": idx, #new added to ref the filename
                        "type": line[0], #one of the following: 'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', or 'DontCare'. 'DontCare' is used for objects that are present but ignored for evaluation.
                        # "truncated": float(line[1]), #A value of 0.0 means the object is fully visible, and 1.0 means the object is completely outside the image frame.
                        # "occluded": int(line[2]), #integer value indicating the degree of occlusion, where 0 means fully visible, and higher values indicate increasing levels of occlusion.
                        # "alpha": float(line[3]), #The observation angle of the object in radians, relative to the camera. It is the angle between the object's heading direction and the positive x-axis of the camera.
                        "bbox": [float(x) for x in line[1:5]], #represent the pixel locations of the top-left and bottom-right corners of the bounding box
                        # "dimensions": [float(x) for x in line[8:11]], #3D dimensions of the object (height, width, and length) in meters
                        # "location": [float(x) for x in line[11:14]], #3D location of the object's centroid in the camera coordinate system (in meters)
                        # "rotation_y": float(line[14]), #The rotation of the object around the y-axis in camera coordinates, in radians.
                    }
                )
        return target #dict list

    def convert_target(self, image_id, target):
        num_objs = len(target)
        boxes = []
        labels = []
        for i in range(num_objs):
            bbox = target[i]['bbox'] ##represent the pixel locations of the top-left and bottom-right corners of the bounding box
            x_center, y_center, width, height = bbox
            objecttype=target[i]['type']
            if objecttype != '8':
                labelid = int(objecttype)
                # labelid = self.INSTANCE2id[objecttype]
                labels.append(labelid)
                boxes.append([x_center, y_center, width, height]) #required for Torch [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H
        num_objs = len(labels) #update num_objs
        newtarget = {}
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = int(image_id)
        #image_id = torch.tensor([image_id])
        #Important!!! do not make image_id a tensor, otherwise the coco evaluation will send error.
        #image_id = torch.tensor(image_id)
        area = boxes[:, 2] * boxes[:, 3]
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        if num_objs >0:
            newtarget["boxes"] = boxes
            newtarget["labels"] = labels
            # newtarget["masks"] = masks
            newtarget["image_id"] = image_id
            newtarget["area"] = area
            newtarget["iscrowd"] = iscrowd
        else:
            #negative example, ref: https://github.com/pytorch/vision/issues/2144
            newtarget['boxes'] = torch.zeros((0, 4), dtype=torch.float32)#not empty
            target['labels'] = labels #torch.as_tensor(np.array(labels), dtype=torch.int64)#empty
            target['image_id'] =image_id
            target["area"] = area #torch.as_tensor(np.array(target_areas), dtype=torch.float32)#empty
            target["iscrowd"] = iscrowd #torch.as_tensor(np.array(target_crowds), dtype=torch.int64)#empty
        return newtarget

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Get item at a given index.

        Args:
            index (int): Index
        Returns:
            tuple: (image, target), where
            target is a list of dictionaries with the following keys:

            - type: str
            - truncated: float
            - occluded: int
            - alpha: float
            - bbox: float[4]
            - dimensions: float[3]
            - locations: float[3]
            - rotation_y: float

        """
        
        if index>len(self.sample_id_list):
            print("Index out-of-range")
            image = None
        else:
            imageidx=self.sample_id_list[index]
            image = self.get_image(imageidx)
            if self.train:
                target = self.get_label(imageidx) #list of dicts
                target = self.convert_target(imageidx, target)
                target['img'] = image

            #target, image_id = self._parse_target(index) if self.train else None

        if WrapNewDict:
            target = dict(image_id=imageidx, annotations=target) #new changes, not used now
        if self.transform:
            image, target = self.transform(image, target)
        # return target
        return image, target

    def __len__(self) -> int:
        return len(self.sample_id_list)#(self.images)

