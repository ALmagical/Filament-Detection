from detectron2.data.datasets import register_coco_instances
# see https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=PIbAM2pv-urF
# for detial
# @gxl
register_coco_instances(
    "coco_filament_train", {}, 
    "/home/gxl/Data/20210405/annotations/train.json", 
    "/home/gxl/Data/20210405/train")
register_coco_instances(
    "coco_filament_val", {},
    "/home/gxl/Data/20210405/annotations/val.json", 
    "/home/gxl/Data/20210405/val")