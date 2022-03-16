import math
from typing import List, Dict
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, NaiveSyncBatchNorm, ASPP
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY

from adet.layers import DFConv2d, NaiveGroupNorm
from adet.utils.comm import compute_locations
from .fcos_outputs import FCOSOutputs
from attention.PSA import PSA
from attention.DANet import DAModule
from attention.CBAM import CBAMBlock
from attention.PolarizedSelfAttention import SequentialPolarizedSelfAttention,  ParallelPolarizedSelfAttention
from adet.layers.activation import get_activation

__all__ = ["FCOS"]

INF = 100000000


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class ModuleListDial(nn.ModuleList):
    def __init__(self, modules=None):
        super(ModuleListDial, self).__init__(modules)
        self.cur_position = 0

    def forward(self, x):
        result = self[self.cur_position](x)
        self.cur_position += 1
        if self.cur_position >= len(self):
            self.cur_position = 0
        return result


@PROPOSAL_GENERATOR_REGISTRY.register()
class FCOS(nn.Module):
    """
    Implement FCOS (https://arxiv.org/abs/1904.01355).
    """
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self.in_features = cfg.MODEL.FCOS.IN_FEATURES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.yield_proposal = cfg.MODEL.FCOS.YIELD_PROPOSAL # False
        
        self.fcos_head = FCOSHead(cfg, [input_shape[f] for f in self.in_features])
        self.in_channels_to_top_module = self.fcos_head.in_channels_to_top_module

        self.fcos_outputs = FCOSOutputs(cfg)
        
    def forward_head(self, features, top_module=None):
        features = [features[f] for f in self.in_features]
        pred_class_logits, pred_deltas, pred_centerness, top_feats, bbox_towers = self.fcos_head(
            features, top_module, self.yield_proposal)
        return pred_class_logits, pred_deltas, pred_centerness, top_feats, bbox_towers

    def forward(self, images, features, gt_instances=None, top_module=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        features = [features[f] for f in self.in_features]
        locations = self.compute_locations(features)
        logits_pred, reg_pred, ctrness_pred, top_feats, bbox_towers = self.fcos_head(
            features, locations, top_module, self.yield_proposal
        )

        results = {}
        if self.yield_proposal:
            results["features"] = {
                f: b for f, b in zip(self.in_features, bbox_towers)
            }

        if self.training:
            results, losses = self.fcos_outputs.losses(
                logits_pred, reg_pred, ctrness_pred,
                locations, gt_instances, top_feats
            )
            # @gxl
            # What's the function of this 'yield_proposal' 
            if self.yield_proposal:
                with torch.no_grad():
                    results["proposals"] = self.fcos_outputs.predict_proposals(
                        logits_pred, reg_pred, ctrness_pred,
                        locations, images.image_sizes, top_feats
                    )
            return results, losses
        else:
            results = self.fcos_outputs.predict_proposals(
                logits_pred, reg_pred, ctrness_pred,
                locations, images.image_sizes, top_feats
            )

            return results, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = compute_locations(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations


class FCOSHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super().__init__()
        # TODO: Implement the sigmoid version first.
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        head_configs = {"cls": (cfg.MODEL.FCOS.NUM_CLS_CONVS,
                                cfg.MODEL.FCOS.USE_DEFORMABLE),
                        "bbox": (cfg.MODEL.FCOS.NUM_BOX_CONVS,
                                 cfg.MODEL.FCOS.USE_DEFORMABLE),
                        "share": (cfg.MODEL.FCOS.NUM_SHARE_CONVS,
                                  False)}
        norm = None if cfg.MODEL.FCOS.NORM == "none" else cfg.MODEL.FCOS.NORM
        self.num_levels = len(input_shape)
        # @gxl
        self.top_with_coords = cfg.MODEL.CONDINST.TOP_WITH_COORDS
        self.use_attention = cfg.MODEL.FCOS.USE_ATTENTION
        self.attention_type = cfg.MODEL.FCOS.ATTENTION_MODEL
        self.activation = cfg.MODEL.FCOS.ACTIVATION
        
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]
        # @gxl
        # attention module
        self.attention_model = torch.nn.ModuleList([])
        if self.use_attention:
            if self.attention_type == 'PSA':
                for i in range(self.num_levels):
                    psa = PSA(channel=in_channels, reduction=4, S=4)
                    psa.init_weights()
                    self.attention_model.append(psa)                  
            elif self.attention_type == 'DA':
                for i in range(self.num_levels):
                    self.attention_model.append(DAModule(d_model=in_channels, kernel_size=3, H=7, W=7))
            elif self.attention_type == 'CBAM':
                for i in range(self.num_levels):
                    self.attention_model.append(CBAMBlock(channel=in_channels, reduction=16, kernel_size=7))
            elif self.attention_type == 'PPSA':
                for i in range(self.num_levels):
                    self.attention_model.append(ParallelPolarizedSelfAttention(channel=in_channels))
            elif self.attention_type == 'SPSA':
                for i in range(self.num_levels):
                    self.attention_model.append(SequentialPolarizedSelfAttention(channel=in_channels))

        self.in_channels_to_top_module = in_channels
        # @gxl
        self.box_with_coords = cfg.MODEL.CONDINST.BOX_WITH_COORDS
        self.cls_with_coords = cfg.MODEL.CONDINST.CLS_WITH_COORDS

        for head in head_configs:
            tower = []
            num_convs, use_deformable = head_configs[head]
            for i in range(num_convs):
                if use_deformable and i == num_convs - 1:
                    conv_func = DFConv2d
                else:
                    conv_func = nn.Conv2d
                # @gxl
                if self.box_with_coords and i == 0 and head == 'bbox':
                    tower.append(conv_func(
                    in_channels+2, in_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=True
                    ))
                elif self.cls_with_coords and i == 0 and head == 'cls':
                    tower.append(conv_func(
                    in_channels+2, in_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=True
                    ))
                else:
                    tower.append(conv_func(
                        in_channels, in_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=True
                    ))
                if norm == "GN":
                    tower.append(nn.GroupNorm(32, in_channels))
                elif norm == "NaiveGN":
                    tower.append(NaiveGroupNorm(32, in_channels))
                elif norm == "BN":
                    tower.append(ModuleListDial([
                        nn.BatchNorm2d(in_channels) for _ in range(self.num_levels)
                    ]))
                elif norm == "SyncBN":
                    tower.append(ModuleListDial([
                        NaiveSyncBatchNorm(in_channels) for _ in range(self.num_levels)
                    ]))
                tower.append(get_activation(self.activation, inplace = True))
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        self.cls_logits = nn.Conv2d(
            in_channels, self.num_classes,
            kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3,
            stride=1, padding=1
        )
        self.ctrness = nn.Conv2d(
            in_channels, 1, kernel_size=3,
            stride=1, padding=1
        )

        if cfg.MODEL.FCOS.USE_SCALE:
            self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(self.num_levels)])
        else:
            self.scales = None

        for modules in [
            self.cls_tower, self.bbox_tower,
            self.share_tower, self.cls_logits,
            self.bbox_pred, self.ctrness
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x, locations, top_module=None, yield_bbox_towers=False):
        logits = []
        bbox_reg = []
        ctrness = []
        top_feats = []
        bbox_towers = []
        # @gxl
        # x is in_features
        for l, feature in enumerate(x):
            N, C, H, W = feature.size()
            feature = self.share_tower(feature)

            if self.use_attention:
                feature_in = self.attention_model[l](feature)
            else:
                feature_in = feature
            # @gxl
            # Add abs_coords to fcos head
            if self.box_with_coords:
                location =locations[l].reshape(1, -1, 2).permute(0, 2, 1)
                box_in = torch.cat([feature_in.reshape(N, C, H*W), location], dim=1)
                box_in = box_in.reshape(N, -1, H, W)
            else:
                box_in = feature_in
            if self.cls_with_coords:
                location =locations[l].reshape(1, -1, 2).permute(0, 2, 1)
                cls_in = torch.cat([feature_in.reshape(N, C, H*W), location], dim=1)
                cls_in = cls_in.reshape(N, -1, H, W)
            else:
                cls_in = feature_in
            cls_tower = self.cls_tower(cls_in)
            bbox_tower = self.bbox_tower(box_in)
            
            if yield_bbox_towers:
                bbox_towers.append(bbox_tower)

            logits.append(self.cls_logits(cls_tower))
            ctrness.append(self.ctrness(bbox_tower))
            reg = self.bbox_pred(bbox_tower)
            if self.scales is not None:
                reg = self.scales[l](reg)
            # Note that we use relu, as in the improved FCOS, instead of exp.
            bbox_reg.append(F.relu(reg))
            if top_module is not None:
                if self.top_with_coords:
                    location =locations[l].reshape(1, -1, 2).permute(0, 2, 1)
                    top_in = torch.cat([bbox_tower.reshape(N, C, H*W), location], dim=1)
                    top_in = top_in.reshape(N, -1, H, W)
                else:
                    top_in = bbox_tower
                top_feats.append(top_module(top_in))
        return logits, bbox_reg, ctrness, top_feats, bbox_towers