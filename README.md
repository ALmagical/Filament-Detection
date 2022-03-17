# CondInst-Filament-Detection

We proposed a new solar filaments detection and classification method based on CondInst. The results show that the method performs well in detecting and classifying isolated and non-isolated filaments,especially in solving the fragments problem.

There are two folders, **CondInst-SFDS** and **detectron2**, in this repository.


## CondInst-SFDS

CondInst-SFDS (CondInst-Solar Filaments Detection and Segmentation) is used for solar filaments detection and segmentation, which is based on [CondInst](https://github.com/aim-uofa/AdelaiDet/blob/master/configs/CondInst/README.md).

The code of CondInst-SFDS can be found in this folder.

See the [README](CondInst-SFDS\README.md) in this folder for more detials.

## detectron2

Some codes in detectron2 has been changed by us.

The modifications include adding the categories of solar filaments, supporting Adam optimazer and so on.

You can access the original [Detectron2](https://github.com/facebookresearch/detectron2) by this link.


## Installation

Before you use CondInst-SFDS, make sure the **detectron2** has been installed.

More detail of detectron2 you can see [Readme of detectron2](detectron2\README.md).

Build CondInst_SFDS with:

```
git clone https://github.com/ALmagical/Filament-Detection.git

python -m pip install -e detectron2

cd Filament-Detection

python setup.py build develop
```

