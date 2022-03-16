import numpy as np
import copy
import os
from itertools import chain
import cv2
import tqdm
from PIL import Image
import json

from detectron2.data import transforms as T
from detectron2.data.transforms.augmentation_impl import RandomRotation, RandomBrightness, RandomContrast
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data.build import filter_images_with_few_keypoints
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

from adet.config import get_cfg
from adet.data.dataset_mapper import DatasetMapperWithBasis
from adet.data.dataset_mapper_aug import DatasetMapper_AUG
from detectron2.data.datasets import register_coco_instances
register_coco_instances(
    "coco_filament_train", {}, 
    # test dataloader
    "/home/gxl/code/AdelaiDet/gxl/json/annotations.json", 
    "/home/gxl/code/AdelaiDet/gxl/json/JPEGImage/")
register_coco_instances(
    "coco_filament_val", {},
    "/home/gxl/Data/20210423/annotations/val.json", 
    "/home/gxl/Data/20210423/val")

if __name__ == "__main__":
    config_file = '/home/gxl/code/AdelaiDet/configs/CondInst/MS_R_50_1x_gxl_test.yaml'
    output_dir = ''
    logger = setup_logger()
    # set config file
    cfg = get_cfg()

    if config_file:
        cfg.merge_from_file(config_file)
    cfg.freeze()

    if output_dir:
        dirname = output_dir
        os.makedirs(dirname, exist_ok=True)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    '''
    augs =[T.AugmentationList([
        RandomRotation(angle = 15, expand = False)
    ])]
    '''
    def output(vis, fname):
        filepath = os.path.join('/home/gxl/code/AdelaiDet/gxl/test_rotate', fname)
        print("Saving to {} ...".format(filepath))
        vis.save(filepath)

    i=0
    for angle in range(35, 360, 35):
        print("Rotate angele is" + str(angle))
        augs =[
            RandomRotation(angle = angle, expand = False)
        ]
        mapper = DatasetMapper_AUG(cfg, True, augmentationlist = augs, recompute_boxes = True)
        datas = build_detection_train_loader(cfg, mapper=mapper)

        scale = 1.0
        for batch in datas:
            for per_img in batch:
                i += 1
                img=per_img["image"].permute(1, 2, 0)
                if cfg.INPUT.FORMAT == "BGR":
                    img = img[:, :, [2, 1, 0]]
                else:
                    img = np.asarray(Image.fromarray(img, mode=cfg.INPUT.FORMAT).convert("RGB"))
                visualizer = Visualizer(img, metadata=metadata, scale=scale)
                target_fields = per_img["instances"].get_fields()
                labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
                vis = visualizer.overlay_instances(
                            labels=labels,
                            boxes=target_fields.get("gt_boxes", None),
                            masks=target_fields.get("gt_masks", None),
                            keypoints=target_fields.get("gt_keypoints", None),
                        )
                output(vis, str(i) + ".jpg")