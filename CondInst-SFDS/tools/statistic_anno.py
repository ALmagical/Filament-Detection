#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import numpy as np
import os
from itertools import chain
import cv2
import tqdm
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data.build import get_detection_dataset_dicts, build_detection_train_loader, print_instances_class_histogram
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

from gxl.units.visualizer import ColorMode
from demo.predictor import VisualizationDemo
from gxl.units.visualizer import VisualizerNoneLabel
from adet.config import get_cfg
from adet.data.dataset_mapper import DatasetMapperWithBasis
# @gxl
from detectron2.data.datasets import register_coco_instances
# see https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=PIbAM2pv-urF
# for detial
# @gxl

register_coco_instances(
    "coco_filament_train", {}, 
    "datasets/annotations/train.json", 
    "datasets/annotations/train")
register_coco_instances(
    "coco_filament_val", {},
    "datasets/annotations/val.json", 
    "datasets/annotations/val")


if __name__ == "__main__":
    config_file = 'configs\CondInst\RESD_50_1x_baseline_cls_2.yaml'
    # Set config file
    cfg = get_cfg()
    if config_file:
        cfg.merge_from_file(config_file)
    cfg.freeze()
    logger = setup_logger()
    metadata_train = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    metadata_val = MetadataCatalog.get(cfg.DATASETS.TEST[0])

    train_name = cfg.DATASETS.TRAIN
    val_name = cfg.DATASETS.TEST

    datasets_train_dict = get_detection_dataset_dicts(
        train_name,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS, 
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
    )

    #print_instances_class_histogram(datasets_train_dict, ["Filaments", "Filament_alone"])
    
    datasets_val_dict = get_detection_dataset_dicts(
        val_name,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS, 
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
    )

    #print_instances_class_histogram(datasets_val_dict, ["Filaments", "Filament_alone"])