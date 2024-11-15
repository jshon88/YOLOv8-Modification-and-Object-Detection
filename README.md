# Object Detection on Kitti Dataset with Cusotmized YOLO v8
YOLO v8 Object Detection model on Kitti Dataset, enhanced with Vision Trasnformer Encoder block added after each C2f block

## Usage
Download Kitti dataset and use convertkitti2yolo.py to convert Kitti to YOLO format
To train the model, use the following command:
You can modify arguments in default.yaml to test different hyperparameters

'''bash
python train.py --model-config path-to-model-config-file --data-config path-to-data-config-file --default-config path-to-default-config-file --epochs 30 --batch-size 16

## Features
- Kitti to YOLO conversion
- Custom ViT Encoder module added in the backbone

## Performance Comparison
![YOLOv8 vs YOLOv8 with ViT](/data/cmpe258-sp24/017553289/cmpe249/ObjectDetection2D/results/yolo/experiment2/coco_evaluation_metrics_comparsion.png)