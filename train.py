# train.py

import sys
sys.path.append('/data/cmpe258-sp24/017553289/cmpe249/ObjectDetection2D')
import argparse
import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as utils
import re
from torchvision.transforms.functional import InterpolationMode
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
# from torch.utils.data.distributed import DistributedSampler
from dataset_kitti import KittiDataset
from modules.yolomodels import YoloDetectionModel
from modules.lossv8 import myv8DetectionLoss
from transforms import Compose, RandomHorizontalFlip, PILToTensor, ToDtype, RandomPhotometricDistort, RandomIoUCrop, RandomZoomOut, ToNumpy, Squeeze, PILResize
from modules.yolotransform import YoloTransform
from utils import (
    MetricLogger,
    save_on_master,
    setup_for_distributed,
    init_distributed_mode,
    mkdir,
    mycollate_fn  # Import mycollate_fn
)
# from trainutils import GroupedBatchSampler, create_aspect_ratio_groups
from modules.metrics import DetMetrics
from modules.utils import yolov8_non_max_suppression, box_iou
from modules.tal import dist2bbox, make_anchors
import csv
from myevaluator import CocoEvaluator, convert_to_coco_api
import logging
import io
from contextlib import redirect_stdout


def parse_args():
    parser = argparse.ArgumentParser(description="Train a customized YOLOv8 model on KITTI dataset")
    parser.add_argument('--model-config', type=str, default='./modules/yolov8n.yaml', help='Path to YOLOv8 model config file')
    parser.add_argument('--data-config', type=str, default='/data/cmpe258-sp24/017553289/cmpe249/dataset/Kitti/kitti.yaml', help='Path to dataset config file')
    parser.add_argument('--default-config', type=str, default='./modules/default.yaml', help='Path to default config file')
    parser.add_argument('--weights', type=str, default=None, help='Path to pretrained weights file (e.g., yolov8n.pt)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=None, help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=None, help='Weight decay')
    parser.add_argument('--workers', type=int, default=None, help='Number of data loading workers')
    parser.add_argument('--device', type=str, default=None, help='Device to use for training (e.g., "cuda:0")')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    parser.add_argument('--save-dir', type=str, default=None, help='Directory to save checkpoints and logs')
    parser.add_argument(
    "--lr-steps",
    default=[16, 22],
    nargs="+",
    type=int,
    help="decrease lr every step-size epochs (multisteplr scheduler only)",
)
    parser.add_argument(
        "--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)"
    )
    # parser.add_argument('--world-size', type=int, default=1, help='Number of distributed processes')
    # parser.add_argument('--dist-url', type=str, default='env://', help='URL used to set up distributed training')
    args = parser.parse_args()
    return args


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(default, override):
    """
    Recursively merge two dictionaries. The 'override' dictionary takes precedence.
    """
    for key, value in override.items():
        if key in default and isinstance(default[key], dict) and isinstance(value, dict):
            merge_configs(default[key], value)
        else:
            default[key] = value
    return default


def create_dataloaders(data_config, transforms, batch_size, workers, distributed):
    train_data = KittiDataset(**data_config['train'], transform=transforms['train'])
    val_data = KittiDataset(**data_config['val'], transform=transforms['val'])

    # Aspect ratio grouping for efficient batching
    # aspect_ratios = create_aspect_ratio_groups(train_data, k=2)
    # train_sampler = DistributedSampler(train_data) if distributed else torch.utils.data.RandomSampler(train_data)
    # train_batch_sampler = GroupedBatchSampler(
    #     sampler=train_sampler,
    #     group_ids=aspect_ratios,
    #     batch_size=batch_size
    # )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        collate_fn=mycollate_fn  # Add collate_fn here
    )

    # val_sampler = DistributedSampler(val_data) if distributed else torch.utils.data.SequentialSampler(val_data)
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=mycollate_fn  # Add collate_fn for validation as well
    )

    return train_loader, val_loader


def build_model(model_config_path, num_classes, device, resume_path=None, pretrained_weights=None):
    model = YoloDetectionModel(cfg=model_config_path, nc=num_classes, verbose=True)
    
    if resume_path:
        # Resume training from a checkpoint
        checkpoint = torch.load(resume_path, map_location=device)
        model_state_dict = checkpoint['model_state_dict']
        
        # Load the filtered state_dict
        missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
        print(f"Resumed training from checkpoint: {resume_path}")
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")
    elif pretrained_weights:
        # Load pretrained weights from yolov8n.pt
        if os.path.exists(pretrained_weights):
            checkpoint = torch.load(pretrained_weights, map_location=device)
            if 'model' in checkpoint:
                # Extract state_dict from the 'model' key
                model_state_dict = checkpoint['model'].state_dict()
            else:
                model_state_dict = checkpoint

            # Create the mapping
            mapping = create_key_mapping(model_state_dict, model)

            # Adjust the keys in the state dict
            adjusted_state_dict = adjust_state_dict_keys(model_state_dict, mapping)
            
            # Load the filtered state_dict
            missing, unexpected = model.load_state_dict(adjusted_state_dict, strict=False)
            print(f"Loaded pretrained weights from {pretrained_weights} (excluding ViTEncoder)")
            if missing:
                print(f"Missing keys: {missing}")
            if unexpected:
                print(f"Unexpected keys: {unexpected}")
        else:
            print(f"Pretrained weights file not found at {pretrained_weights}. Proceeding without loading pretrained weights.")
    else:
        print("No pretrained weights specified. Proceeding with random initialization.")
    
    model = model.to(device)
    return model

def create_key_mapping(pretrained_state_dict, modified_model):
    mapping = {}

    # Identify the insertion points based on your model definition
    # For example, if ViTEncoders are inserted after layers 5, 7, 9 in the pretrained model
    vi_encoder_insert_indices = [5, 7, 9]

    for key in pretrained_state_dict.keys():
        # Extract the layer index from the key
        match = re.match(r'model\.(\d+)\.(.*)', key)
        if match:
            idx = int(match.group(1))
            rest = match.group(2)

            # Calculate the new index
            # shift = sum(1 for insert_idx in vi_encoder_insert_indices if idx >= insert_idx)
            # new_idx = idx + shift

            # Create the new key
            # new_key = f'model.{new_idx}.{rest}'
            # mapping[key] = new_key
            if idx != 22: # excluding detection header
                shift = sum(1 for insert_idx in vi_encoder_insert_indices if idx >= insert_idx)
                new_idx = idx + shift

                # Create the new key
                new_key = f'model.{new_idx}.{rest}'
                mapping[key] = new_key
        else:
            # If the key does not match the expected pattern, keep it as is
            mapping[key] = key

    return mapping

def adjust_state_dict_keys(pretrained_state_dict, mapping):
    adjusted_state_dict = {}
    for old_key, value in pretrained_state_dict.items():
        new_key = mapping.get(old_key, None)
        if new_key is not None:
            adjusted_state_dict[new_key] = value
        else:
            # Handle keys that don't need to be changed or can be ignored
            pass
    return adjusted_state_dict

def validate_batch(batch):
    bboxes = batch["bboxes"]
    cls = batch["cls"]
    
    # Check for valid bounding boxes
    valid_bboxes = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])
    if not valid_bboxes.all():
        return False
    
    # Check for valid class labels (assuming classes start from 0)
    if cls.min() < 0 or cls.max() >= model.nc:  # Replace 'model.nc' with the number of classes
        return False
    
    return True

def decode_bboxes(anchor_points, pred_dist, reg_max=16):
    '''
    Decode predicted bounding box distribution into [xmin, ymin, xmax, ymax] format
    '''
    if reg_max > 1:
        pred_dist = pred_dist.view(pred_dist.shape[0], pred_dist.shape[1], 4, reg_max) # [b, n_anchors, 4, n_bins]
        proj = torch.arange(reg_max, dtype=torch.float, device=pred_dist.device) # 16
        pred_bboxes = pred_dist.softmax(3).matmul(proj.type(pred_dist.dtype))
        # pred_dist = torch.softmax(pred_dist, dim=-1) # softmax applied over bins
    else:
        pred_bboxes = pred_dist  # No decoding needed if reg_max == 1

    # Assuming bbox_decode_fn converts normalized [xmin, ymin, xmax, ymax] to absolute scale
    decoded_boxes = dist2bbox(pred_bboxes, anchor_points, xywh=False)  # [B, num_anchors, 4]

    return decoded_boxes

    


def main():
    logging.basicConfig(level=logging.INFO)

    args = parse_args()

    # Initialize distributed training if applicable
    # init_distributed_mode(args)

    # Load default configuration
    default_config = load_config(args.default_config)

    # Load model and data configurations
    model_config = load_config(args.model_config) if args.model_config else {}
    data_config = load_config(args.data_config) if args.data_config else {}

    # Merge configurations: model_config and data_config override default_config
    full_config = merge_configs(default_config, model_config)
    full_config = merge_configs(full_config, data_config)

    # Override configurations with command-line arguments if provided
    if args.epochs is not None:
        full_config['epochs'] = args.epochs
    if args.batch_size is not None:
        full_config['batch'] = args.batch_size
    if args.lr is not None:
        full_config['lr0'] = args.lr
    if args.momentum is not None:
        full_config['momentum'] = args.momentum
    if args.weight_decay is not None:
        full_config['weight_decay'] = args.weight_decay
    if args.workers is not None:
        full_config['workers'] = args.workers
    if args.device is not None:
        full_config['device'] = args.device
    if args.save_dir is not None:
        full_config['save_dir'] = args.save_dir
    if args.resume:
        full_config['resume'] = args.resume

    # Create directories for saving checkpoints and logs
    save_dir = full_config.get('save_dir', 'runs/train')
    mkdir(save_dir)

    # Device configuration
    device = torch.device(full_config['device'] if torch.cuda.is_available() else 'cpu')

    # Data transformations
    # transforms = {
    # 'train': Compose([
    #     PILToTensor(),
    #     ToDtype(torch.float, scale=False),
    #     ToNumpy(),
    #     YoloTransform(
    #         min_size=full_config['imgsz'][1],
    #         max_size=full_config['imgsz'][0],
    #         image_mean=full_config.get('image_mean', None),
    #         image_std=full_config.get('image_std', None),
    #         # device=device,
    #         fp16=full_config.get('half', False),
    #         cfgs=full_config,
    #         size_divisible=32
    #     ),
    #     Squeeze(),
    #     ]),
    #     'val': Compose([
    #         PILToTensor(),
    #         ToDtype(torch.float, scale=False),
    #         ToNumpy(),
    #         YoloTransform(
    #             min_size=full_config['imgsz'][1],
    #             max_size=full_config['imgsz'][0],
    #             image_mean=full_config.get('image_mean', None),
    #             image_std=full_config.get('image_std', None),
    #             # device=device,
    #             fp16=full_config.get('half', False),
    #             cfgs=full_config,
    #             size_divisible=32
    #         ),
    #         Squeeze(),
            
    #     ])
    # }

    transforms = {
    'train': Compose([
        PILResize(),
        PILToTensor(),
        ToDtype(torch.float, scale=True),
        ]),

    'val': Compose([
        PILResize(),
        PILToTensor(),
        ToDtype(torch.float, scale=True),
    ])
    }

    # Create DataLoaders
    train_loader, val_loader = create_dataloaders(
        data_config=full_config,
        transforms=transforms,
        batch_size=full_config['batch'],
        workers=full_config['workers'],
        distributed=full_config.get('world_size', 1) > 1
    )

    # Determine the path to pretrained weights
    pretrained_weights = full_config.get('model_path', None)
    if args.weights:
        pretrained_weights = args.weights  # Override if --weights is provided

    # Build the model
    num_classes = full_config['nc']
    model = build_model(
        model_config_path=args.model_config,
        num_classes=num_classes,
        device=device,
        resume_path=full_config['resume'] if full_config.get('resume') else None,
        pretrained_weights=pretrained_weights
    )

    # Define loss function
    criterion = myv8DetectionLoss(model)

    # Define optimizer and scheduler
    optimizer = optim.SGD(
        model.parameters(),
        lr=full_config['lr0'],
        momentum=full_config['momentum'],
        weight_decay=full_config['weight_decay']
    )
    # optimizer = optim.Adam(
    #     model.parameters(),
    #     lr=full_config['lr0'],
    #     # momentum=full_config['momentum'],
    #     weight_decay=full_config['weight_decay']
    # )

    # COCO Format conversion from xmin, ymin, xmax, ymax to xmin, ymin, w, h after gt bbox being adjusted when resizing dataset
    coco_gt = convert_to_coco_api(val_loader.dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco_gt, iou_types)

    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=full_config['epochs'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    class_names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']

    best_mAP50 = 0.0
    best_mAP50_95 = 0.0

    from torch.utils.tensorboard import SummaryWriter
    wr = SummaryWriter(log_dir='runs/experiment2')

    metrics = DetMetrics(save_dir=save_dir, plot=False, on_plot=None, names=class_names)
    csv_file = os.path.join(save_dir, 'training_metrics.csv')
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Epoch",
            "Train_Box_Loss",
            "Train_Cls_Loss",
            "Train_DFL_Loss",
            "Val_Box_Loss",
            "Val_Cls_Loss",
            "Val_DFL_Loss",
            "Precision",
            "Recall",
            "mAP50",
            "mAP50-95"
        ])

    # Initialize MetricLogger
    logger = MetricLogger(delimiter="  ")

    iou_threshold = 0.5  # IoU threshold for determining true positives

    # Training loop
    for epoch in range(full_config['epochs']):
        model.train()
        if full_config.get('world_size', 1) > 1:
            train_loader.sampler.set_epoch(epoch)

        logger.update(lr=optimizer.param_groups[0]["lr"])
        train_box_loss = 0.0
        train_cls_loss = 0.0
        train_dfl_loss = 0.0

        for batch_idx, batch in enumerate(
            logger.log_every(
                train_loader,
                print_freq=10,
                header=f"Epoch [{epoch+1}/{full_config['epochs']}]"
            )
        ):

            batch['img'] = batch['img'].to(device)
            batch['bboxes'] = batch['bboxes'].to(device)
            # batch['bboxes'] = xyxy2xywh(batch['bboxes']).to(device)
            batch['cls'] = batch['cls'].to(device)
            batch['batch_idx'] = batch['batch_idx'].to(device)
            batch['img_id'] = batch['img_id'].to(device)
            images = batch['img']
            bboxes = batch['bboxes']
            cls = batch['cls']
            batch_idx_tensor = batch['batch_idx']


            # images = batch['img'].to(device)
            # # print(images.shape)
            # bboxes = batch['bboxes'].to(device)
            # cls = batch['cls'].to(device)
            # batch_idx_tensor = batch['batch_idx'].to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss, loss_items = criterion(outputs, batch)
            # print(loss_items)

            loss.backward()

            utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            total_norm = 0
            for name, param in model.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"Gradient Norm: {total_norm}")

            optimizer.step()

            box_loss = loss_items[0].item()
            cls_loss = loss_items[1].item()
            dfl_loss = loss_items[2].item()
            
            train_box_loss += box_loss
            train_cls_loss += cls_loss
            train_dfl_loss += dfl_loss

            # Create a local dictionary for logging
            loss_dict = {
                'box_loss': box_loss,
                'cls_loss': cls_loss,
                'dfl_loss': dfl_loss
            }


            if batch_idx % 50 == 0 or batch_idx == len(train_loader):
                wr.add_scalar('Loss/Box', box_loss, epoch * len(train_loader) + batch_idx)
                wr.add_scalar('Loss/Cls', cls_loss, epoch * len(train_loader) + batch_idx)
                wr.add_scalar('Loss/DFL', dfl_loss, epoch * len(train_loader) + batch_idx)
                wr.add_scalar('Gradient Norm', total_norm, epoch * len(train_loader) + batch_idx)

            # Log loss_items
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{full_config['epochs']}], "
                        f"Batch [{batch_idx}], "
                        f"Box Loss: {loss_items[0].item():.4f}, "
                        f"Cls Loss: {loss_items[1].item():.4f}, "
                        f"DFL Loss: {loss_items[2].item():.4f}, "
                        f"Total Loss: {loss.item():.4f}")

            # Update metrics
            logger.update(loss=loss.item(), **loss_dict)
        
        wr.flush()

        # Average training losses
        avg_train_box_loss = train_box_loss / len(train_loader)
        avg_train_cls_loss = train_cls_loss / len(train_loader)
        avg_train_dfl_loss = train_dfl_loss / len(train_loader)

        scheduler.step()

        # Validation
        model.eval()
        val_box_loss = 0.0
        val_cls_loss = 0.0
        val_dfl_loss = 0.0

        # all_tp = []
        # all_conf = []
        # all_pred_cls = []
        # all_target_cls = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch['img'].to(device)
                bboxes = batch['bboxes'].to(device)
                cls = batch['cls'].to(device)
                batch_idx_tensor = batch['batch_idx'].to(device)

                outputs = model(images)
                loss, loss_items = criterion(outputs, batch)
                val_box_loss += loss_items[0].item()
                val_cls_loss += loss_items[1].item()
                val_dfl_loss += loss_items[2].item()

                # Create correct input for yolov8_non_max_suppression function
                feats = outputs[1] if isinstance(outputs, tuple) else outputs #preds=feats=[[16, 72, 48, 156], [16, 72, 24, 78], [16, 72, 12, 39]]
                pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], criterion.no, -1) for xi in feats], 2).split(
                        (criterion.reg_max * 4, criterion.nc), 1)  # [b, nc + 64, h * w]
                pred_scores = pred_scores.permute(0, 2, 1).contiguous() #->[b, num_boxes, nc]
                pred_distri = pred_distri.permute(0, 2, 1).contiguous() #->[16, num_boxes, 64]
                anchor_points, stride_tensor = make_anchors(feats, criterion.stride, 0.5)
                decode_boxes = decode_bboxes(anchor_points, pred_distri) # [b, num_boxes, 4]

                nms_inputs = torch.cat([pred_scores, decode_boxes], dim=2)
                num_inputs = nms_inputs.permute(0, 2, 1).contiguous() # [b, nc + 4, num_boxes]


                # Apply Non-Maximum Suppression using yolov8_non_max_suppression
                nms_outputs = yolov8_non_max_suppression(
                    prediction=num_inputs,
                    conf_thres=0.25,  # Confidence threshold, adjust as needed
                    # iou_thres=0.45,   # IoU threshold, adjust as needed
                    iou_thres=full_config['iou'],
                    classes=None,     # Filter classes if needed
                    agnostic=False,
                    multi_label=False,
                    labels=[],        # No additional labels
                    max_det=300,
                    nc=num_classes,   # Number of classes
                )

                # Prepare predictions for CocoEvaluator
                predictions = {}
                image_ids = batch['img_id'].unique()
                prediction_boxes = torch.Tensor()
                prediction_scores = torch.Tensor()
                prediction_labels = torch.Tensor()
                for i, detections in enumerate(nms_outputs):
                    if detections is None or detections.shape[0] == 0:
                        continue
                    # Extract boxes, scores, and labels
                    prediction_boxes = detections[:, :4]  # [num_predictions, 4]
                    prediction_scores = detections[:, 4]  # [num_predictions]
                    prediction_labels = detections[:, 5].long()  # [num_predictions]
                    results = {
                        'boxes' : prediction_boxes,
                        'scores' : prediction_scores,
                        'labels' : prediction_labels
                    }

                    predictions[int(image_ids[i])] = results

                coco_evaluator.update(predictions)

        coco_evaluator.accumulate()

        f = io.StringIO()
        with redirect_stdout(f):
            coco_evaluator.summarize()
        summary_str = f.getvalue()

        # Save to a text file
        summary_file_path = '/data/cmpe258-sp24/017553289/cmpe249/ObjectDetection2D/runs/experiment2/validation_metrics_summary.txt'
        with open(summary_file_path, 'a') as file:
            file.write(f"Epoch {epoch+1} Summary:\n")
            file.write(summary_str)
            file.write("\n" + "-"*80 + "\n")  # Add a separator for readability
        print(f"Validation metrics for Epoch {epoch+1} have been appended to {summary_file_path}")


                # # Collect predictions and ground truths for metric computation
                # for i in range(len(nms_outputs)):
                #     detections = nms_outputs[i]
                #     if detections is None or detections.shape[0] == 0:
                #         continue

                #     # Extract boxes, scores, and labels
                #     pred_boxes = detections[:, :4]  # Shape: (num_predictions, 4)
                #     pred_scores = detections[:, 4]  # Shape: (num_predictions,)
                #     pred_labels = detections[:, 5].long()  # Shape: (num_predictions,)

                #     # Ground truth data
                #     gt_boxes = bboxes[i]
                #     gt_labels = cls[i]

                #     # Compute IoU matrix
                #     iou_matrix = box_iou(pred_boxes, gt_boxes)

                #     # Initialize match status
                #     num_preds = pred_boxes.shape[0]
                #     num_gts = gt_boxes.shape[0]
                #     matched_gt = torch.zeros(num_gts, dtype=torch.bool, device=device)
                #     tp = torch.zeros(num_preds, dtype=torch.uint8, device=device)

                #     # Sort predictions by confidence
                #     sorted_indices = torch.argsort(-pred_scores)

                #     for pred_idx in sorted_indices:
                #         pred_box = pred_boxes[pred_idx]
                #         pred_label = pred_labels[pred_idx]
                #         pred_score = pred_scores[pred_idx]

                #         # Get IoUs between this prediction and all ground truths
                #         ious = iou_matrix[pred_idx]

                #         # Find the best matching ground truth
                #         max_iou, gt_idx = ious.max(dim=0)
                #         gt_label = gt_labels[gt_idx]

                #         # Check if IoU exceeds threshold, classes match, and ground truth is not already matched
                #         if max_iou >= iou_threshold and pred_label == gt_label and not matched_gt[gt_idx]:
                #             tp[pred_idx] = 1  # True Positive
                #             matched_gt[gt_idx] = True  # Mark ground truth as matched
                #         else:
                #             tp[pred_idx] = 0  # False Positive

                #     # Collect results
                #     all_tp.append(tp.cpu().numpy())
                #     all_conf.append(pred_scores.cpu().numpy())
                #     all_pred_cls.append(pred_labels.cpu().numpy())
                #     all_target_cls.append(gt_labels.cpu().numpy())


                    
        # # Concatenate all results
        # tp_array = np.concatenate(all_tp)
        # conf_array = np.concatenate(all_conf)
        # pred_cls_array = np.concatenate(all_pred_cls)
        # target_cls_array = np.concatenate(all_target_cls)

        # # Process metrics
        # metrics.process(
        #     tp=tp_array,
        #     conf=conf_array,
        #     pred_cls=pred_cls_array,
        #     target_cls=target_cls_array
        # )


        # Retrieve computed metrics
        # precision, recall, mAP50, mAP50_95 = metrics.mean_results()

        avg_val_box_loss = val_box_loss / len(val_loader)
        avg_val_cls_loss = val_cls_loss / len(val_loader)
        avg_val_dfl_loss = val_dfl_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{full_config['epochs']}], "
            f"Train Losses: Box={avg_train_box_loss:.4f}, Cls={avg_train_cls_loss:.4f}, DFL={avg_train_dfl_loss:.4f}, "
            f"Val Losses: Box={avg_val_box_loss:.4f}, Cls={avg_val_cls_loss:.4f}, DFL={avg_val_dfl_loss:.4f}, ")

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch + 1,
                avg_train_box_loss,
                avg_train_cls_loss,
                avg_train_dfl_loss,
                avg_val_box_loss,
                avg_val_cls_loss,
                avg_val_dfl_loss
            ])

        save_checkpoint(
            epoch=epoch,
            val_loss=avg_val_box_loss + avg_val_cls_loss + avg_val_dfl_loss,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            save_dir=save_dir
        )

        # # Save best_mAP50.pt
        # if mAP50 > best_mAP50:
        #     best_mAP50 = mAP50
        #     save_checkpoint(
        #         epoch=epoch,
        #         val_loss=avg_val_box_loss + avg_val_cls_loss + avg_val_dfl_loss,
        #         model=model,
        #         optimizer=optimizer,
        #         scheduler=scheduler,
        #         save_dir=save_dir,
        #         filename="best_mAP50.pt"
        #     )
        #     print(f"New best mAP50: {best_mAP50:.4f}. Saved best checkpoint.")

        # # Save best_mAP50_95.pt
        # if mAP50_95 > best_mAP50_95:
        #     best_mAP50_95 = mAP50_95
        #     save_checkpoint(
        #         epoch=epoch,
        #         val_loss=avg_val_box_loss + avg_val_cls_loss + avg_val_dfl_loss,
        #         model=model,
        #         optimizer=optimizer,
        #         scheduler=scheduler,
        #         save_dir=save_dir,
        #         filename="best_mAP50_95.pt"
        #     )
        #     print(f"New best mAP50-95: {best_mAP50_95:.4f}. Saved best checkpoint.")

        print(f"End of Epoch {epoch+1}\n")


    print("Training completed successfully!")


def save_checkpoint(epoch, val_loss, model, optimizer, scheduler, save_dir, filename=None):
    """Save the model checkpoint."""
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': val_loss,
    }
    if filename is None:
        filename = f"checkpoint_epoch_{epoch+1}.pth"
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")


# def main():
#     # Hyperparameters
#     batch_size = 4
#     num_workers = 2
#     learning_rate = 1e-3
#     epochs = 5  # Set to a small number for testing
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     seed = 42

#     # Set random seed for reproducibility
#     torch.manual_seed(seed)
#     random.seed(seed)

#     # Define transformations with corrected order (excluding SimpleCopyPaste)
#     transform = Compose([
#         RandomPhotometricDistort(),
#         RandomHorizontalFlip(p=0.5),
#         RandomIoUCrop(),
#         RandomZoomOut(),
#         PILToTensor(),        # Convert PIL Image to Tensor first
#         ToDtype(torch.float32, scale=True),  # Then change dtype
#         # SimpleCopyPaste()    # Removed from per-sample transforms
#     ])

#     # Initialize the dataset
#     is_train=True
#     root_dir = '/data/cmpe258-sp24/017553289/cmpe249/dataset/Kitti'  # Replace with your KITTI dataset path
#     train_dataset = KittiDataset(
#         root=root_dir,
#         train=is_train,
#         split='train',
#         transform=transform
#     )

#     # Create DataLoader
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         collate_fn=mycollate_fn
#     )

    


if __name__ == "__main__":
    main()