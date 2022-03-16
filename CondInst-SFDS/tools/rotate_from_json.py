
#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# @gxl based on Detectron2 tools/visualize_data.py
# To Verify the correctness of the augmentation, you should use the script in tools/visualize_data.py
# This code is used to transforming the 'Fragment' to 'Non-isolated Filament'.
# At same time, the samples will be rotated to generate more samples.
#
# Using the dataloader in Detectron2 to load data with annotations
# Using the augmentation functions in Detectron2 to augment data offline
# 
from tools.test.thresh_seg import thresh_segmentation
import numpy
import copy
import os, sys
import csv
import codecs
from itertools import chain
import cv2
import tqdm
from PIL import Image
from matplotlib import pyplot as plt
import json
import datetime
import collections
import pycocotools
import random
import pandas
from multiprocessing.dummy import Pool as ThreadPool

import multiprocessing as mp
from functools import  partial # pool.map needs more than one parameter
import threading
from itertools import islice
import pp
from parallelize import parallelize
import gc

from detectron2.data import transforms as T
from detectron2.data.transforms.augmentation_impl import RandomRotation, RandomBrightness, RandomContrast
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data.build import filter_images_with_few_keypoints, get_detection_dataset_dicts
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BoxMode, BitMasks
from adet.config import get_cfg
from adet.data.detection_utils import (annotations_to_instances, build_augmentation, transform_instance_annotations)

from gxl.test.thresh_seg import thresh_segmentation, shape_to_mask

from detectron2.data.datasets import register_coco_instances
register_coco_instances(
    "coco_filament_train", {"thing_classes": ["Non-isolated filament", "Isolate filament"],
    "thing_colors":[[252, 220, 65], [0, 154, 73]]}, 
    "datasets/annotations/train.json", 
    "datasets/annotations/train")
register_coco_instances(
    "coco_filament_train", {"thing_classes": ["Non-isolated filament", "Isolate filament"],
    "thing_colors":[[252, 220, 65], [0, 154, 73]]}, 
    "datasets/annotations/val.json", 
    "datasets/annotations/val")
# JSON encoder
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        return json.JSONEncoder.default(self, obj)


def get_centroid(points):
    points = points.reshape(-1, 2)
    area = 0.0
    Gx = 0.0
    Gy = 0.0
    for i, point in enumerate(points):
        if (i+1) >= len(points):
            point_nex = points[0]
        else:
            point_nex = points[i+1]
        temp = (point[1]*point_nex[0] - point[0]*point_nex[1]) / 2.0
        area += temp
        Gx += temp * (point[0] + point_nex[0]) / 3.0
        Gy += temp * (point[1] + point_nex[1]) / 3.0
    Gx = int(Gx / area)
    Gy = int(Gy / area)
    # Setting 3th row with 1. This format is the annotations of keypoints in COCO.
    return [Gx, Gy, 1]


def get_inter_mask(k, segms, class_ids, main_segm, visit_mat, img , classes):
    sub_segm = segms[k]
    class_id = class_ids[k]
    mask = None
    if visit_mat[k] == False:
        class_id = int(class_id)
        class_name = classes[class_id]
        if class_name != 'Filament':
            return None
        sub_segm = shape_to_mask(img.shape[:2], numpy.asarray(sub_segm).reshape(-1, 2))
        sub_segm = numpy.asfortranarray(sub_segm)
        sub_segm = pycocotools.mask.encode(sub_segm)
        iou = pycocotools.mask.iou([sub_segm], [main_segm], [0])
        if iou > 0:
            # Recompute mask by using local thresh
            new_mask, class_name, area = thresh_segmentation(img, sub_segm, class_name, erosion=False, k_shape=cv2.MORPH_ELLIPSE, k_size=(3,3))
            visit_mat[k] = True
            if area == 0:
                return None
            mask = new_mask
    return mask

# TODO: Add function to change the label in original annotations
# TODO: Change the input to argument
if __name__ == "__main__":
    train_or_val = 'val'
    config_file = 'configs/CondInst/MS_R_50_1x_gxl_test.yaml'
    output_dir = 'output' + train_or_val
    output_json_dir = 'output'
    labels = "Data/labels.txt" # Label categorys to get class id 
    ignore_label = ['Fragment']
    paralle_num = 8
    merg_label = []
    isvisualize = False
    if not labels:
        print("Pleases offer label file!")
        sys.exit(1)

    logger = setup_logger()
    # Set config file
    cfg = get_cfg()
    if config_file:
        cfg.merge_from_file(config_file)
    cfg.freeze()

    if output_dir:
        dirname = output_dir
        os.makedirs(dirname, exist_ok = True)

    if train_or_val == 'train':
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    elif train_or_val == 'val':
        metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

    if output_json_dir:
        dirname = output_json_dir
        os.makedirs(dirname, exist_ok = True)
    else:
        dirname = os.path.join(output_dir, "JSON")
        os.makedirs(dirname, exist_ok = True)
        output_json_dir = dirname
    output_img_dir=os.path.join(output_json_dir, train_or_val)
    os.makedirs(output_img_dir, exist_ok = True) 
    print("Creating dataset:", output_json_dir)
    
    # Save image to file
    def output(vis, fname):
        filepath = os.path.join(output_dir, fname)
        vis.save(filepath)


    # Prepare Json file
    now = datetime.datetime.now()
    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[dict(url=None, id=0, name=None,)],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type="instances",
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    # Map class name to id
    class_name_to_id = {}
    j = 0
    for i, line in enumerate(open(labels).readlines()):
        class_id = i - 1 - j  # starts with -1
        class_name = line.strip()
        if class_name in merg_label:
            if merg_label.find(class_name) != 0:
                continue
        if class_name in ignore_label:
            j += 1
            continue
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        class_name_to_id[class_name] = class_id
        data["categories"].append(
            dict(supercategory=None, id=class_id, name=class_name,)
        )

    os.makedirs(os.path.join(output_json_dir, "annotations"), exist_ok = True)

    # Get the dict of Dataset
    if train_or_val == 'train':
        datasets_name = cfg.DATASETS.TRAIN
        out_ann_file = os.path.join(output_json_dir, "annotations/train.json")
    elif train_or_val == 'val':
        datasets_name = cfg.DATASETS.TEST
        out_ann_file = os.path.join(output_json_dir, "annotations/val.json")
    datasets_dict = get_detection_dataset_dicts(
        datasets_name,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS, 
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
    )

    i = 0
    means = []
    stds = []
    areas = []
    keypoints = []
    # Rotate the image pre rotate_angle.
    rotate_angle = 360
    tot_img_num = (len(datasets_dict) * (int(360 / rotate_angle) + 1))
    # Generating shuffle index for image 
    inds_img = list(range(1, tot_img_num + 1))
    #random.shuffle(inds_img)
    classes = metadata.get('thing_classes', None)
    if paralle_num > 1:
        #pool = ThreadPool(paralle_num)
        pool = mp.Pool(paralle_num)
    '''
    # pp
    ppservers = ()
    if len(sys.argv) > 1:
        ncpus = int(sys.argv[1])
    # Creates jobserver with ncpus workers
        job_server = pp.Server(ncpus, ppservers=ppservers)
    else:
    # Creates jobserver with automatically detected number of workers
        job_server = pp.Server(ppservers=ppservers)
    print("Starting pp with", job_server.get_ncpus(), "workers")
    '''
    #for k, data_ori in enumerate(datasets_dict):
    for data_ori in tqdm.tqdm(datasets_dict, desc='Data in datasets'):
        # data_ori = datasets_dict[len(datasets_dict)-1]
        # Read image and bounding box
        image_input = cv2.imread(data_ori["file_name"])
        image_name = os.path.split(data_ori["file_name"])[-1]

        # Geting bounding boxes
        boxes = numpy.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in data_ori["annotations"]
            ]
        )
        for angle in range(0, 360, rotate_angle):

            if image_input.mean() < 60 and angle != 0:
                continue
            # Get image's shape 
            image_shape=image_input.shape[:2]

            # Random allocate index for image
            image_id = inds_img[i]
            i += 1

            aug = T.AugmentationList([T.RotationTransform(image_shape[0],image_shape[1],angle=angle, expand=False),])
            img_aug = T.AugInput(image_input, boxes=boxes,sem_seg=None)
            transforms = aug(img_aug)
            img = img_aug.image

            # compute mean and std
            mean = img.mean()
            std = img.std()
            means.append(mean)
            stds.append(std)
            file_name = image_name[:-4] + "_" + str(int(angle/rotate_angle)) + ".jpg"
            # Write image info to dict
            data["images"].append(
                dict(
                    license=0,
                    url=None,
                    #file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
                    file_name=file_name,
                    height=img.shape[0],
                    width=img.shape[1],
                    date_captured=None,
                    id=i,
                )
            )
            keypoint = []
            # Deep copy, this will not change the origin annotations
            data_anno = copy.deepcopy(data_ori)
            if "annotations" in data_anno:
                # USER: Implement additional transformations if you have other types of data
                annos = [
                    transform_instance_annotations(
                        obj,
                        transforms,
                        image_shape,
                        keypoint_hflip_indices=None,
                    )
                    for obj in data_anno["annotations"]
                    if obj.get("iscrowd", 0) == 0
                ]
                instances = annotations_to_instances(
                    annos, image_shape, mask_format="polygon"
                )
                # Recompute bounding boxes
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                data_anno["instances"] = utils.filter_empty_instances(instances)

                target_fields = data_anno["instances"].get_fields()
                bboxes = target_fields.get("gt_boxes", None)
                bboxes_xyxy = bboxes
                # Change bbox mode from XYXY_ABS to XYWH_ABS
                # Coco save boxes annotations as XYWH_ABS
                bboxes = BoxMode.convert(
                            bboxes.tensor.numpy(), BoxMode.XYXY_ABS, BoxMode.XYWH_ABS
                        )
                segms = target_fields.get("gt_masks", None)
                class_ids = target_fields.get("gt_classes", None)
                assert len(bboxes) == len(segms)
                visit_mat = numpy.zeros(len(bboxes), dtype=bool)
                for j, [bbox, segm, class_id] in enumerate(zip(bboxes, segms, class_ids)):
                    class_id = int(class_id)
                    class_name = classes[class_id]
                    if visit_mat[j] == True:
                        continue
                    if class_name == 'Filament_alone':
                        # Recompute mask by using local thresh
                        new_mask, class_name, area = thresh_segmentation(img, segm, class_name, erosion=False, k_shape=cv2.MORPH_ELLIPSE, k_size=(3,3))
                        visit_mat[j] = True
                        if area < 20:
                            continue
                        areas.append(area)
                        class_id = class_name_to_id[class_name]
                        
                    elif class_name == 'Filaments':
                        '''
                        main_segm = segm
                        main_segm = numpy.asfortranarray(numpy.uint8(main_segm))
                        main_segm = pycocotools.mask.encode(main_segm)
                        '''
                        mask_all = []
                        main_segm = shape_to_mask(img.shape[:2], numpy.asarray(segm).reshape(-1, 2))
                        main_segm = numpy.asfortranarray(main_segm)
                        main_segm = pycocotools.mask.encode(main_segm)
                        # Version parallelize
                        '''
                        def get_inter_mask(k, segms, class_ids, main_segm, visit_mat, img , classes):
                            sub_segm = segms[k]
                            class_id = class_ids[k]
                            mask = None
                            if visit_mat[k] == False:
                                class_id = int(class_id)
                                class_name = classes[class_id]
                                if class_name != 'Filament':
                                    return None
                                sub_segm = shape_to_mask(img.shape[:2], numpy.asarray(sub_segm).reshape(-1, 2))
                                sub_segm = numpy.asfortranarray(sub_segm)
                                sub_segm = pycocotools.mask.encode(sub_segm)
                                iou = pycocotools.mask.iou([sub_segm], [main_segm], [0])
                                if iou > 0:
                                    # Recompute mask by using local thresh
                                    new_mask, class_name, area = thresh_segmentation(img, sub_segm, class_name)
                                    visit_mat[k] = True
                                    if area == 0:
                                        return None
                                    mask = new_mask
                            return mask
                            '''
                        if paralle_num > 1:
                            iter_range = [x for x in range(len(segms))]
                            #pool = ThreadPool(16)
                            mask_all = pool.map(partial(get_inter_mask, segms=segms, class_ids=class_ids, main_segm=main_segm, visit_mat=visit_mat, img=img, classes=classes), iter_range)
                            #pool.close()
                            #pool.join()
                            mask_all = [mask for mask in mask_all if mask != None]
                        else:
                        # Single process
                            for k, [sub_segm, class_id] in enumerate(zip(segms, class_ids)):
                                if visit_mat[k] == False:
                                    class_id = int(class_id)
                                    class_name = classes[class_id]
                                    if class_name != 'Filament':
                                        continue
                                    
                                    #sub_segm = numpy.asfortranarray(numpy.uint8(sub_segm))
                                    #sub_segm = pycocotools.mask.encode(sub_segm)
                                    
                                    sub_segm = shape_to_mask(img.shape[:2], numpy.asarray(sub_segm).reshape(-1, 2))
                                    sub_segm = numpy.asfortranarray(sub_segm)
                                    sub_segm = pycocotools.mask.encode(sub_segm)
                                    iou = pycocotools.mask.iou([sub_segm], [main_segm], [0])
                                    if iou > 0:
                                        # Recompute mask by using local thresh
                                        new_mask, class_name, area = thresh_segmentation(img, sub_segm, class_name, erosion=False, k_shape=cv2.MORPH_ELLIPSE, k_size=(3,3))
                                        visit_mat[k] = True
                                        if area == 0:
                                            continue
                                        mask_all.append(new_mask)
                                    
                        # Mearge masks
                        visit_mat[j] = True
                        if len(mask_all) > 0:
                            new_mask = numpy.zeros_like(pycocotools.mask.decode(mask_all[0]))
                            for mask_s in mask_all:
                                new_mask = new_mask | pycocotools.mask.decode(mask_s)
                            new_mask = pycocotools.mask.encode(new_mask)
                            area = float(pycocotools.mask.area(new_mask))
                            if area < 20:
                                continue
                            areas.append(area)
                            if len(mask_all) == 1:
                                class_name = 'Filament_alone'
                            else:
                                class_name = 'Filaments'
                            class_id = class_name_to_id[class_name]
                        else:
                            continue
                    else:
                        continue
                    # Get bounding boxes
                    bbox = pycocotools.mask.toBbox(new_mask).tolist()
                    # Write annotations to dict
                    data["annotations"].append(
                        dict(
                            id=len(data["annotations"]),
                            image_id=i,
                            category_id=class_id,
                            segmentation=new_mask, # Coco formate
                            area=area,
                            bbox=bbox,
                            #keypoints=centroid,
                            iscrowd=0,
                        )
                    )
                
            #keypoints.append(keypoint)
            img_name = file_name
            filepath = os.path.join(output_img_dir, img_name)
            img_save=Image.fromarray(numpy.uint8(img)).convert("RGB")
            img_save.save(filepath)
            print("Saving to {} ...".format(filepath))
            if isvisualize == True:
                # Visualize image and annotations
                if cfg.INPUT.FORMAT == "BGR":
                    img = img[:, :, [2, 1, 0]]
                else:
                    img = numpy.asarray(Image.fromarray(img, mode=cfg.INPUT.FORMAT).convert("RGB"))
                #target_fields = data["instances"].get_fields()
                visualizer = Visualizer(img, metadata=metadata, scale=1)
                keypoint = numpy.array(keypoint).reshape(-1, 1, 3)
                labels = [metadata.thing_classes[cla] for cla in target_fields["gt_classes"]]
                vis = visualizer.overlay_instances(
                            labels=labels,
                            boxes=bboxes_xyxy,
                            masks=None,
                            #keypoints=keypoint,
                            keypoints=target_fields.get("gt_keypoints", None),
                        )
                # Save in different place
                output(vis, img_name)
            gc.get_threshold()
            # print("mean: ", str(mean), " std: ", str(std))
    # assert i == 0
    if paralle_num > 1:
        pool.close()
        pool.join()
    mean = sum(means) / i
    std = sum(stds) / i
    data_static=dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        data=dict(
            means=means,
            stds=stds,
            areas=areas,
        ),
    )

    filepath_static = os.path.join(output_json_dir, 'static.json')
    with open(filepath_static, "w") as f:
        json.dump(data_static, f)
    #filepath_static = os.path.join(output_json_dir, 'static.csv')
    print("mean: ", str(mean))
    print("std: ", str(std))
    with open(out_ann_file, "w") as f:
        json.dump(data, f, cls=MyEncoder,)