# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import (
    CNNBlockBase,
    Conv2d,
    ShapeSpec,
    get_norm,
)

from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from attention.SEAttention import SEAttention

from adet.modeling.backbone.resnet_d import BasicStem
from .yolox_block import (
    BaseConv, Focus, DWConv, CSPLayer, Bottleneck, ResLayer, SPPBottleneck
)
from adet.layers.activation import get_activation

__all__ = [
    "BaseConv",
    "DWConv",
    "BottleneckBlock",
    "SPPBottleneck",
    "CSPBlock",
    "StemBlock",
    "CSPNet",
    "build_cspnet_backbone",
]


class BaseConv(CNNBlockBase):
    """
        Batchnorm -> relu/mish block -> A Conv2d
        Style of Resnet-V2
    """

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, norm="BN", activation="Mish"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2

        self.bn = get_norm(norm, in_channels)
        self.activation = get_activation(activation, inplace=True)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):

        out = self.bn(x)
        out = self.activation(out)
        out = self.conv(out)

        return out

    def fuseforward(self, x):

        out = self.activation(x)
        out = self.conv(out)

        return out

class DWConv(CNNBlockBase):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, activation="Mish"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            activation=activation,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, activation=activation
        )

    def forward(self, x):

        out = self.dconv(x)
        out = self.pconv(out)

        return out

class BottleneckBlock(CNNBlockBase):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        activation="Mish",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, activation=activation)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, activation=activation)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):

        out = self.conv1(x)
        out = self.conv2(out)
        if self.use_add:
            out = out + x

        return out

class ResLayer(CNNBlockBase):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, activation="Mish"
        )
        self.conv2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, activation="Mish"
        )

    def forward(self, x):

        out = self.conv1(x)
        out = self.conv2(out)
        out = out + x

        return out

class SPPBottleneck(CNNBlockBase):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), num_groups=1, norm="BN", activation="Mish"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, group=num_groups, norm=norm, activation=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, group=num_groups, norm=norm, activation=activation)

    def forward(self, x):

        out = self.conv1(x)
        out = torch.cat([out] + [m(out) for m in self.m], dim=1)
        out = self.conv2(out)

        return out

class CSPBlock(CNNBlockBase):

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expension=0.5,
        depthwise=False,
        stride=1,
        num_groups=1,
        norm="BN",
        activation="Msih",
        ):
        super().__init__(in_channels, out_channels, stride)

        hidden_channels = int(out_channels * expension)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, group=num_groups, norm=norm, activation=activation)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, group=num_groups, norm=norm, activation=activation)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, group=num_groups, norm=norm, activation=activation)
        module_list = [
            BottleneckBlock (
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, activation=activation
            )
            for _ in range(n)
        ]
        self.res = nn.Sequential(*module_list)

        def forward(self, x):

            part1 = self.conv1(x)
            part2 = self.conv2(x)
            part1 = self.res(part1)

            out = torch.cat((part1, part2), dim=1)
            out = self.conv3(out)

            return out

class StemBlock(CNNBlockBase):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, activation="Mish"):
        super().__init__()

        self.conv1 = BaseConv(in_channels * 4, out_channels, ksize, stride, activation=activation)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)

        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]

        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        out = self.conv1(x)

        return out


class CSPNet(Backbone):

    def __init__(self, stem, stages, num_classes=None, out_features=None):
        
        super().__init__()
        
        self.stem = stem
        self.num_classes = num_classes

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        self.stage_names, self.stages = [], []

        for i, blocks in enumerate(stages):
            assert len(blocks) > 0, len(blocks)
            for block in blocks:
                assert isinstance(block, CNNBlockBase), block

            name = 'csp' + str(i + 2)

            stage = nn.Sequential(*blocks)

            self.add_module(name, stage)
            self.stage_names.append(name)
            self.stages.append(stage)

            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in blocks])
            )
            self._out_feature_channels[name] = curr_channels = blocks[-1].out_channels
        self.stage_names = tuple(self.stage_names)  # Make it static for scripting

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)

            # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "The 1000-way fully-connected layer is initialized by
            # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
            nn.init.normal_(self.linear.weight, std=0.01)
            name = "linear"

        if out_features is None:
            out_features = [name]

        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert x.dim() == 4, f"Taking an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        x = self.stem(x)
        #print('stem: ', x.shape)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for name, stage in zip(self.stage_names, self.stages):
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
            #print(name, ': ', x.shape)
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def freeze(self, freeze_at=0):
        """
        Freeze the first several stages of the ResNet. Commonly used in
        fine-tuning.

        Layers that produce the same feature map spatial size are defined as one
        "stage" by :paper:`FPN`.

        Args:
            freeze_at (int): number of stages to freeze.
                `1` means freezing the stem. `2` means freezing the stem and
                one residual stage, etc.

        Returns:
            nn.Module: this ResNet itself
        """
        if freeze_at >= 1:
            self.stem.freeze()
        for idx, stage in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                for block in stage.children():
                    block.freeze()
        return self

    @staticmethod
    def make_stage(
        block_class, num_blocks, first_stride=None, *, in_channels, out_channels, **kwargs
    ):
        """
        Create a list of blocks of the same type that forms one ResNet stage.

        Args:
            block_class (type): a subclass of CNNBlockBase that's used to create all blocks in this
                stage. A module of this type must not change spatial resolution of inputs unless its
                stride != 1.
            num_blocks (int): number of blocks in this stage
            first_stride (int): deprecated
            in_channels (int): input channels of the entire stage.
            out_channels (int): output channels of **every block** in the stage.
            kwargs: other arguments passed to the constructor of
                `block_class`. If the argument name is "xx_per_block", the
                argument is a list of values to be passed to each block in the
                stage. Otherwise, the same argument is passed to every block
                in the stage.

        Returns:
            list[nn.Module]: a list of block module.

        Examples:
        ::
            stages = ResNet.make_stage(
                BottleneckBlock, 3, in_channels=16, out_channels=64,
                bottleneck_channels=16, num_groups=1,
                stride_per_block=[2, 1, 1],
                dilations_per_block=[1, 1, 2]
            )

        Usually, layers that produce the same feature map spatial size are defined as one
        "stage" (in :paper:`FPN`). Under such definition, ``stride_per_block[1:]`` should
        all be 1.
        """
        if first_stride is not None:
            assert "stride" not in kwargs and "stride_per_block" not in kwargs
            kwargs["stride_per_block"] = [first_stride] + [1] * (num_blocks - 1)
            logger = logging.getLogger(__name__)
            logger.warning(
                "CSP.make_stage(first_stride=) is deprecated!  "
                "Use 'stride_per_block' or 'stride' instead."
            )

        blocks = []
        for i in range(num_blocks):
            curr_kwargs = {}
            for k, v in kwargs.items():
                if k.endswith("_per_block"):
                    assert len(v) == num_blocks, (
                        f"Argument '{k}' of make_stage should have the "
                        f"same length as num_blocks={num_blocks}."
                    )
                    newk = k[: -len("_per_block")]
                    assert newk not in kwargs, f"Cannot call make_stage with both {k} and {newk}!"
                    curr_kwargs[newk] = v[i]
                else:
                    curr_kwargs[k] = v

            blocks.append(
                block_class(in_channels=in_channels, out_channels=out_channels, **curr_kwargs)
            )
            in_channels = out_channels
        return blocks

@BACKBONE_REGISTRY.register()
def build_cspnet_backbone(cfg, input_shape):
    """
    Create a CSPNet instance from config.

    Returns:
        CSPNet: a :class:`CSPNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.CSPNET.NORM
    activation = cfg.MODEL.CSPNET.ACTIVATION
    stem = StemBlock(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.CSPNET.STEM_OUT_CHANNELS,
        norm=norm,
        activation=activation,
    )

    # fmt: off
    freeze_at           = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features        = cfg.MODEL.CSPNET.OUT_FEATURES
    depth               = cfg.MODEL.CSPNET.DEPTH
    num_groups          = cfg.MODEL.CSPNET.NUM_GROUPS
    width_per_group     = cfg.MODEL.CSPNET.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels         = cfg.MODEL.CSPNET.STEM_OUT_CHANNELS
    out_channels        = cfg.MODEL.CSPNET.CSP2_OUT_CHANNELS
    stride_in_1x1       = cfg.MODEL.CSPNET.STRIDE_IN_1X1
    csp5_dilation       = cfg.MODEL.CSPNET.CSP5_DILATION
    # fmt: on
    assert csp5_dilation in {1, 2}, "csp5_dilation cannot be {}.".format(csp5_dilation)

    num_blocks_per_stage = {
        53: [2, 8, 8, 4],
    }[depth]

    stages = []

    # Avoid creating variables without gradients
    # It consumes extra memory and may cause allreduce to fail
    out_stage_idx = [
        {"csp2": 2, "csp3": 3, "csp4": 4, "csp5": 5}[f] for f in out_features if f != "stem"
    ]
    max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        dilation = csp5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "stride_per_block": [first_stride] + [1] * (num_blocks_per_stage[idx] - 1),
            "in_channels": in_channels,
            "out_channels": out_channels,
            "norm": norm,
        }

        stage_kargs["bottleneck_channels"] = bottleneck_channels
        stage_kargs["stride_in_1x1"] = stride_in_1x1
        stage_kargs["dilation"] = dilation
        stage_kargs["num_groups"] = num_groups
        stage_kargs["block_class"] = CSPBlock

        blocks = CSPNet.make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        stages.append(blocks)
    return CSPNet(stem, stages, out_features=out_features).freeze(freeze_at)