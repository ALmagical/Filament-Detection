# CondInst-SFDS

CondInst-SFDS (CondInst-Solar Filaments Detection and Segmentation) is used for solar filaments detection and segmentation, which is based on [CondInst](https://github.com/aim-uofa/AdelaiDet/blob/master/configs/CondInst/README.md).

<div align=center><img src=".\examples\results\BBSO\bbso_halph_fl_20140310_180912.jpg " width = "512"/>

**Detecting example of CondInst-SFDS**

<div align=left>


## Introduction

In this project, we provide different versions of CondInst-SFDS. You can find all config files in [here](.\configs\CondInst).

We implement many different backbones for CondInst-SFDS, includes the combine of ResNet-C, ResNet-D, and ResNet v2, the combine of ResNet-D, ResNet v2, and YOLOX, etc.

The attention model, such as SE-attention, CBMA-attention, PSA-attention, etc, can be inserted into the different parts of CondInst-SFDS, which only need a little changes in the codes or config files. Thanks for the work of [FightingCV Codebase](https://github.com/xmu-xiaoma666/External-Attention-pytorch) that implement some core code of self-attention models.

We provide some useful tools in this [folder](CondInst-SFDS\tools).


## Usage

Please See **[command_used.md](command_used.md)** for how to use.

