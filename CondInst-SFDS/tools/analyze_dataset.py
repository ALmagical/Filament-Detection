import numpy as np
import pandas as pd
import os
import json
import tqdm
from PIL import Image
from matplotlib import pyplot as plt
import datetime
import collections
import pycocotools
import random

from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data.build import filter_images_with_few_keypoints, get_detection_dataset_dicts
from detectron2.utils.logger import setup_logger
from detectron2.structures import BoxMode
from adet.config import get_cfg
from adet.data.dataset_mapper import DatasetMapperWithBasis
from adet.data.detection_utils import (annotations_to_instances, build_augmentation, transform_instance_annotations)

from detectron2.data.datasets import register_coco_instances

register_coco_instances(
    "coco_filament_train", {},
    "/home/gxl/Data/20210528/annotations/train.json",
    "/home/gxl/Data/20210528/train")
register_coco_instances(
    "coco_filament_val", {},
    "/home/gxl/Data/20210528/annotations/val.json",
    "/home/gxl/Data/20210528/val")
def get_annotations(file_path, annos):
    with open(file_path, 'r') as json_file:
        data_dict = json.load(json_file)
    data = []
    if "annotations" in data_dict:
        for anno in annos:
            anno_list = []
            for annotation in data_dict["annotations"]:
                anno_list.append(annotation[anno])
            data.append(anno_list)
        #anno_list = np.array(anno_list).reshape(-1, len(annos))
        anno_list = pd.DataFrame(data, index=annos)
        # Switch the rows and columns
        anno_list = pd.DataFrame(anno_list.values.T, index=anno_list.columns, columns=anno_list.index)
        return anno_list
    else:
        return None


if __name__ == "__main__":
    file_val = "/home/gxl/Data/20210528/annotations/val.json"
    file_train = "/home/gxl/Data/20210528/annotations/train.json"
    annos = ["category_id", "area"]
    anno_val = get_annotations(file_val, annos)
    anno_train =get_annotations(file_train, annos)
    anno_all = np.concatenate((anno_train, anno_val), axis=0)

    a=0
    #file_json_train = json.loads("")