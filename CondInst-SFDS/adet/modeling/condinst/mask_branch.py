# coding=UTF-8
from typing import Dict
import math

import torch
from torch import nn

from fvcore.nn import sigmoid_focal_loss_jit
from detectron2.layers import ShapeSpec, ASPP

from adet.layers import conv_with_kaiming_uniform
from adet.utils.comm import aligned_bilinear
from attention.DANet import DAModule 
from attention.PSA import PSA 
from attention.CBAM import CBAMBlock
from attention.PolarizedSelfAttention import SequentialPolarizedSelfAttention,  ParallelPolarizedSelfAttention
from adet.layers.activation import get_activation

INF = 100000000


def build_mask_branch(cfg, input_shape):
    return MaskBranch(cfg, input_shape)


class MaskBranch(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        '''
         Default value:
            MASK_BRANCH.IN_FEATURES=[P3,P4,P5]
            MASK_BRANCH.SEMANTIC_LOSS_ON=FALSE
            MASK_BRANCH.OUT_CHANNELS=8
            MASK_BRANCH.NORM=BN
            MASK_BRANCH.NUM_CONVS=4
            MASK_BRANCH.CHANNELS=128
        '''
        self.in_features = cfg.MODEL.CONDINST.MASK_BRANCH.IN_FEATURES
        self.sem_loss_on = cfg.MODEL.CONDINST.MASK_BRANCH.SEMANTIC_LOSS_ON
        self.num_outputs = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS
        norm = cfg.MODEL.CONDINST.MASK_BRANCH.NORM
        num_convs = cfg.MODEL.CONDINST.MASK_BRANCH.NUM_CONVS
        channels = cfg.MODEL.CONDINST.MASK_BRANCH.CHANNELS
        self.out_stride = input_shape[self.in_features[0]].stride
        
        # @gxl
        self.fuse_type = cfg.MODEL.CONDINST.MASK_BRANCH.FUSE_TYPE
        self.use_attention = cfg.MODEL.CONDINST.MASK_BRANCH.USE_ATTENTION
        self.attention_type = cfg.MODEL.CONDINST.MASK_BRANCH.ATTENTION_MODEL
        
        self.activation = cfg.MODEL.CONDINST.MASK_BRANCH.ACTIVATION
        self.use_aspp = cfg.MODEL.CONDINST.MASK_BRANCH.ASPP
        self.use_decoder = cfg.MODEL.CONDINST.MASK_BRANCH.DECODER

        feature_channels = {k: v.channels for k, v in input_shape.items()}

        conv_block = conv_with_kaiming_uniform(norm, activation=self.activation)
   
        
        if self.use_aspp == True:
            self.aspp = ASPP(feature_channels[self.in_features[0]], channels, [6, 12, 18], norm=norm, activation=get_activation(self.activation))
        else:
            if self.fuse_type == 'sum':
                in_channels_fuse = channels
            elif self.fuse_type == 'cat':
                in_channels_fuse = len(self.in_features)*channels
            self.refine = nn.ModuleList()
            for in_feature in self.in_features:
                self.refine.append(conv_block(
                    feature_channels[in_feature],
                    channels, 3, 1
                ))
                self.fuse = nn.Conv2d(in_channels_fuse, channels, kernel_size=1, stride=1, bias=False)
                torch.nn.init.normal_(self.fuse.weight, std=0.01)

        tower = []
        for i in range(num_convs):
            tower.append(conv_block(
                channels, channels, 3, 1
            ))
        tower.append(nn.Conv2d(
            channels, max(self.num_outputs, 1), 1
        ))
        self.add_module('tower', nn.Sequential(*tower))



        if self.use_decoder == True:
            self.decoder = nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False)
            torch.nn.init.normal_(self.decoder.weight, std=0.01)

        # attention
        if self.use_attention:
            if self.attention_type == 'DA':
                #_, _, H, W = 
                self.attention_model = DAModule(d_model=channels, kernel_size=3, H=7, W=7)
            elif self.attention_type == 'PSA':
                self.attention_model = PSA(channel=channels, reduction=4, S=4)
            elif self.attention_type == 'CBAM':
                self.attention_model = CBAMBlock(channel=channels, reduction=16, kernel_size=7)
            elif self.attention_type == 'PPSA':
                self.attention_model = ParallelPolarizedSelfAttention(channel=channels)
            elif self.attention_type == 'SPSA':
                self.attention_model = SequentialPolarizedSelfAttention(channel=channels)
        

        
        if self.sem_loss_on:
            num_classes = cfg.MODEL.FCOS.NUM_CLASSES
            self.focal_loss_alpha = cfg.MODEL.FCOS.LOSS_ALPHA
            self.focal_loss_gamma = cfg.MODEL.FCOS.LOSS_GAMMA
            
            '''
            in_channels = feature_channels[self.in_features[0]]
            self.seg_head = nn.Sequential(
                conv_block(in_channels, channels, kernel_size=3, stride=1),
                conv_block(channels, channels, kernel_size=3, stride=1)
            )
            '''

            # self.logits = nn.Conv2d(channels, num_classes, kernel_size=1, stride=1)

            self.logits = nn.Conv2d(max(self.num_outputs, 1), num_classes, kernel_size=1, stride=1)

            prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.logits.bias, bias_value)

    def forward(self, features, gt_instances=None):

        if self.use_aspp == True:
            if self.use_decoder == True:
                aspp_feature = features[self.in_features[1]]
            else:
                aspp_feature = features[self.in_features[0]]
            x = self.aspp(aspp_feature)

            if self.use_decoder == True:
                x = aligned_bilinear(x, 2)
                x = x + features[self.in_features[0]]
                x = self.decoder(x) 
        else:
            for i, f in enumerate(self.in_features):
                if i == 0:
                    x = self.refine[i](features[f])
                else:
                    x_p = self.refine[i](features[f])

                    target_h, target_w = x.size()[-2:]
                    h, w = x_p.size()[-2:]
                    assert target_h % h == 0
                    assert target_w % w == 0
                    factor_h, factor_w = target_h // h, target_w // w
                    assert factor_h == factor_w
                    x_p = aligned_bilinear(x_p, factor_h)
                    if self.fuse_type == 'sum':
                        x = x + x_p
                    elif self.fuse_type == 'cat':    
                        x = torch.cat([x, x_p], dim=1)
            
        
            if self.fuse_type == 'cat':
                x = self.fuse(x)
        #print('mask: ', x.shape)

        if self.use_attention:
            x = self.attention_model(x)
        
        mask_feats = self.tower(x)

        # output is empty array
        if self.num_outputs == 0:
            mask_feats = mask_feats[:, :self.num_outputs]
        #print('mask_feats: ', mask_feats.shape)
        losses = {}
        # auxiliary thing semantic loss
        if self.training and self.sem_loss_on:
            '''
            logits_pred = self.logits(self.seg_head(
                features[self.in_features[0]]
            ))
            '''
            logits_pred = self.logits(mask_feats)
            # compute semantic targets
            semantic_targets = []
            for per_im_gt in gt_instances:
                h, w = per_im_gt.gt_bitmasks_full.size()[-2:]
                areas = per_im_gt.gt_bitmasks_full.sum(dim=-1).sum(dim=-1)
                areas = areas[:, None, None].repeat(1, h, w)
                areas[per_im_gt.gt_bitmasks_full == 0] = INF
                areas = areas.permute(1, 2, 0).reshape(h * w, -1)
                min_areas, inds = areas.min(dim=1)
                per_im_sematic_targets = per_im_gt.gt_classes[inds] + 1
                per_im_sematic_targets[min_areas == INF] = 0
                per_im_sematic_targets = per_im_sematic_targets.reshape(h, w)
                semantic_targets.append(per_im_sematic_targets)

            semantic_targets = torch.stack(semantic_targets, dim=0)

            # resize target to reduce memory
            semantic_targets = semantic_targets[
                               :, None, self.out_stride // 2::self.out_stride,
                               self.out_stride // 2::self.out_stride
                               ]

            # prepare one-hot targets
            num_classes = logits_pred.size(1)
            class_range = torch.arange(
                num_classes, dtype=logits_pred.dtype,
                device=logits_pred.device
            )[:, None, None]
            class_range = class_range + 1
            one_hot = (semantic_targets == class_range).float()
            num_pos = (one_hot > 0).sum().float().clamp(min=1.0)

            loss_sem = sigmoid_focal_loss_jit(
                logits_pred, one_hot,
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="sum",
            ) / num_pos
            losses['loss_sem'] = loss_sem

        return mask_feats, losses
