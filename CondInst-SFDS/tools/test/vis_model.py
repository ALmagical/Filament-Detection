import torch
import tensorwatch as tw
from gxl.train_net import Trainer, setup
from detectron2.engine import default_argument_parser, launch
from torchviz import make_dot
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.data.build import get_detection_dataset_dicts
from adet.data.dataset_mapper_gxl import DatasetMapper_GXL
from adet.checkpoint import AdetCheckpointer

def main(args):
    with torch.no_grad():
        cfg = setup(args)
        datasets_name = cfg.DATASETS.TRAIN
        model = Trainer.build_model(cfg)
        AdetCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                    cfg.MODEL.WEIGHTS, resume=args.resume
                )
        input=next(iter(Trainer.build_test_loader(cfg,datasets_name[0])))
        model.eval()
        y = model(input)
        print(y)
        g = make_dot(y)
        g.render('output/20210701/condinst_model', view= False)

if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args(args=['--config-file', '/home/gxl/code/AdelaiDet/output/20210701/config.yaml', '--num-gpus', '2', '--eval-only'])
    launch(
            main,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,),
        )
