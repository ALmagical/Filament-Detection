# coding=UTF-8
'''
'''
import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog, build_detection_train_loader

from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.utils.events import EventStorage
from detectron2.evaluation import (
    COCOEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.logger import setup_logger
from adet.data.dataset_mapper import DatasetMapperWithBasis
from adet.config import get_cfg
from adet.checkpoint import AdetCheckpointer
from adet.evaluation import TextEvaluator
from detectron2.data.datasets import register_coco_instances

register_coco_instances(
    "coco_filament_train", {}, 
    "/home/gxl/Data/20210405/annotations/train.json", 
    "/home/gxl/Data/20210405/train")
register_coco_instances(
    "coco_filament_val", {},
    "/home/gxl/Data/20210405/annotations/val.json", 
    "/home/gxl/Data/20210405/val")

class Trainer(DefaultTrainer):