# -*- coding: utf-8 -*-
from cProfile import label
import logging

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from detectron2.structures import ImageList, Boxes
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures.instances import Instances
from detectron2.structures.masks import PolygonMasks, polygons_to_bitmask
from detectron2.layers import ASPP

from .dynamic_mask_head import build_dynamic_mask_head
from .mask_branch import build_mask_branch

from adet.utils.comm import aligned_bilinear
# @gxl Tensorboard
from adet.utils.puttotensorboard import put_image_to_tensorboard, vis_bbox
from detectron2.utils.events import EventStorage, get_event_storage
from adet.modeling.solov2.utils import matrix_nms, mask_nms

from gxl.units.unit import erosion, medium_filter, mean_filter

__all__ = ["CondInst"]


logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class CondInst(nn.Module):
    """
    Main class for CondInst architectures (see https://arxiv.org/abs/2003.05664).
    """
    '''
    PROPOSAL_GENERATOR='FCOS'
    '''
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.mask_head = build_dynamic_mask_head(cfg)
        self.mask_branch = build_mask_branch(cfg, self.backbone.output_shape())
        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE
        self.max_proposals = cfg.MODEL.CONDINST.MAX_PROPOSALS
        self.debug = cfg.SOLVER.DEBUG

        # NMS
        self.use_mask_nms = cfg.MODEL.CONDINST.MASK_HEAD.USE_MASK_NMS
        self.max_before_nms = cfg.MODEL.CONDINST.MASK_HEAD.NMS_PRE
        self.score_threshold = cfg.MODEL.CONDINST.MASK_HEAD.SCORE_THR
        self.update_threshold = cfg.MODEL.CONDINST.MASK_HEAD.UPDATE_THR
        self.mask_threshold = cfg.MODEL.CONDINST.MASK_HEAD.MASK_THR
        self.max_per_img = cfg.MODEL.CONDINST.MASK_HEAD.MAX_PRE_IMG
        self.nms_kernel = cfg.MODEL.CONDINST.MASK_HEAD.NMS_KERNEL
        self.nms_sigma = cfg.MODEL.CONDINST.MASK_HEAD.NMS_SIGMA
        self.fuse_mask = cfg.MODEL.CONDINST.MASK_HEAD.FUSE_MASK

        self.use_aspp = cfg.MODEL.CONDINST.ASPP

        
        # build top module
        in_channels = self.proposal_generator.in_channels_to_top_module
        self.top_with_coords = cfg.MODEL.CONDINST.TOP_WITH_COORDS
        
        #print(self.backbone.output_shape())
        #exit()
        # @gxl
        if self.top_with_coords:
            self.controller = nn.Sequential(
                nn.Conv2d(in_channels+2, in_channels,
                kernel_size=1, stride=1, padding=0
                ),
                nn.ReLU(),
                nn.Conv2d(
                in_channels, self.mask_head.num_gen_params,
                kernel_size=3, stride=1, padding=1
                ))
            #initialize
            for layer in self.controller:
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)
        else:
            # If here using nn.Sequential() will not fit original condinst
            # That means the model, which was trained for old version, will
            # not work.
            self.controller = nn.Conv2d(
                in_channels, self.mask_head.num_gen_params,
                kernel_size=3, stride=1, padding=1
            )
            torch.nn.init.normal_(self.controller.weight, std=0.01)
            torch.nn.init.constant_(self.controller.bias, 0)
        
        
        # torch.nn.init.normal_(self.controller.weight, std=0.01)
        # torch.nn.init.constant_(self.controller.bias, 0)
        # PIXEL_MEAN: [103.53,116.28,123.675]
        # PIXEL_STD: [1.0,1.0,1.0]
        # This Datas is coming from COCO
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        # lambda用于创建匿名函数
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        #print(type(images))
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)


        #print('image: ',images.tensor.shape)
        #exit()
        #print(features)
        #for feature in features:
            
           # print(feature, ': ',features[feature].shape)
        
          
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            self.add_bitmasks(gt_instances, images.tensor.size(-2), images.tensor.size(-1))
        else:
            gt_instances = None

        mask_feats, sem_losses = self.mask_branch(features, gt_instances)

        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances, self.controller
        )

        if self.training:
            # @gxl
            # Add image to tensorboard
            if self.debug:
                storage = get_event_storage()
                put_image_to_tensorboard(mask_feats, storage, 'mask_feat')
                input_image = torch.cat([image for image in images])
                input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
                storage.put_image('input_image', input_image)
                instance = proposals['instances']
                '''
                vis_bbox(input_image,
                        instance.get_fields().get("reg_pred", None),
                        instance.get_fields().get("reg_pred", None),
                        storage)
                        '''
                del input_image

            loss_mask = self._forward_mask_heads_train(proposals, mask_feats, gt_instances)

            losses = {}
            losses.update(sem_losses)
            losses.update(proposal_losses)
            losses.update({"loss_mask": loss_mask})
            return losses
        else:
            if self.use_mask_nms == True:
                max_before_nms = self.max_before_nms
                proposals_temp = proposals
                
                num_images = len(proposals_temp)

                results = []
                for i in range(num_images):
                    result = proposals_temp[i]
                    number_of_detections = len(result)
                    if number_of_detections > max_before_nms > 0:                
                        cls_scores = result.scores
                        image_thresh, _ = torch.kthvalue(
                            cls_scores.cpu(),
                            number_of_detections - max_before_nms + 1
                        )
                        keep = cls_scores >= image_thresh.item()
                        keep = torch.nonzero(keep).squeeze(1)
                        result = result[keep]
                    results.append(result)        
                proposals = results

            pred_instances_w_masks = self._forward_mask_heads_test(proposals, mask_feats)

            padded_im_h, padded_im_w = images.tensor.size()[-2:]
            processed_results = []
            for im_id, (input_per_image, image_size) in enumerate(zip(batched_inputs, images.image_sizes)):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])

                instances_per_im = pred_instances_w_masks[pred_instances_w_masks.im_inds == im_id]
                if self.use_mask_nms == False:
                    instances_per_im = self.postprocess(
                        instances_per_im, height, width,
                        padded_im_h, padded_im_w, self.mask_threshold
                    )
                else:
                    instances_per_im = self.postprocess_nms(
                        instances_per_im, height, width,
                        padded_im_h, padded_im_w, self.mask_threshold
                    )

                processed_results.append({
                    "instances": instances_per_im
                })
            #return pred_instances_w_masks
            return processed_results

    def _forward_mask_heads_train(self, proposals, mask_feats, gt_instances):
        # prepare the inputs for mask heads
        pred_instances = proposals["instances"]

        if 0 <= self.max_proposals < len(pred_instances):
            inds = torch.randperm(len(pred_instances), device=mask_feats.device).long()
            logger.info("clipping proposals from {} to {}".format(
                len(pred_instances), self.max_proposals
            ))
            pred_instances = pred_instances[inds[:self.max_proposals]]

        pred_instances.mask_head_params = pred_instances.top_feats

        # mask_branch.out_stride=4
        loss_mask = self.mask_head(
            mask_feats, self.mask_branch.out_stride,
            pred_instances, gt_instances
        )

        return loss_mask

    def _forward_mask_heads_test(self, proposals, mask_feats):
        # prepare the inputs for mask heads
        for im_id, per_im in enumerate(proposals):
            per_im.im_inds = per_im.locations.new_ones(len(per_im), dtype=torch.long) * im_id
        pred_instances = Instances.cat(proposals)
        pred_instances.mask_head_params = pred_instances.top_feat

        pred_instances_w_masks = self.mask_head(
            mask_feats, self.mask_branch.out_stride, pred_instances
        )

        return pred_instances_w_masks

    def add_bitmasks(self, instances, im_h, im_w):
        for per_im_gt_inst in instances:
            if not per_im_gt_inst.has("gt_masks"):
                continue
            start = int(self.mask_out_stride // 2)
            if isinstance(per_im_gt_inst.get("gt_masks"), PolygonMasks):
                polygons = per_im_gt_inst.get("gt_masks").polygons
                per_im_bitmasks = []
                per_im_bitmasks_full = []
                for per_polygons in polygons:
                    bitmask = polygons_to_bitmask(per_polygons, im_h, im_w)
                    bitmask = torch.from_numpy(bitmask).to(self.device).float()
                    start = int(self.mask_out_stride // 2)
                    bitmask_full = bitmask.clone()
                    bitmask = bitmask[start::self.mask_out_stride, start::self.mask_out_stride]

                    assert bitmask.size(0) * self.mask_out_stride == im_h
                    assert bitmask.size(1) * self.mask_out_stride == im_w

                    per_im_bitmasks.append(bitmask)
                    per_im_bitmasks_full.append(bitmask_full)

                per_im_gt_inst.gt_bitmasks = torch.stack(per_im_bitmasks, dim=0)
                per_im_gt_inst.gt_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0)
            else: # RLE format bitmask
                bitmasks = per_im_gt_inst.get("gt_masks").tensor
                h, w = bitmasks.size()[1:]
                # pad to new size
                bitmasks_full = F.pad(bitmasks, (0, im_w - w, 0, im_h - h), "constant", 0)
                bitmasks = bitmasks_full[:, start::self.mask_out_stride, start::self.mask_out_stride]
                per_im_gt_inst.gt_bitmasks = bitmasks
                per_im_gt_inst.gt_bitmasks_full = bitmasks_full
    
    def postprocess(self, results, output_height, output_width, padded_im_h, padded_im_w, mask_threshold=0.5):
        """
        Resize the output instances.
        The input images are often resized when entering an object detector.
        As a result, we often need the outputs of the detector in a different
        resolution from its inputs.
        This function will resize the raw outputs of an R-CNN detector
        to produce outputs according to the desired output resolution.
        Args:
            results (Instances): the raw outputs from the detector.
                `results.image_size` contains the input image resolution the detector sees.
                This object might be modified in-place.
            output_height, output_width: the desired output resolution.
        Returns:
            Instances: the resized output from the model, based on the output resolution
        """
        scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
        resized_im_h, resized_im_w = results.image_size
        results = Instances((output_height, output_width), **results.get_fields())
        
        if results.has("pred_boxes"):
            output_boxes = results.pred_boxes
        elif results.has("proposal_boxes"):
            output_boxes = results.proposal_boxes

        output_boxes.scale(scale_x, scale_y)
        output_boxes.clip(results.image_size)
        
        results = results[output_boxes.nonempty()]
        
        if results.has("pred_global_masks"):
            mask_h, mask_w = results.pred_global_masks.size()[-2:]
            factor_h = padded_im_h // mask_h
            factor_w = padded_im_w // mask_w
            assert factor_h == factor_w
            factor = factor_h
            pred_global_masks = aligned_bilinear(
                results.pred_global_masks.cpu(), factor
            )
            pred_global_masks = pred_global_masks[:, :, :resized_im_h, :resized_im_w]
            pred_global_masks = F.interpolate(
                pred_global_masks,
                size=(output_height, output_width),
                mode="bilinear", align_corners=False
            )
            pred_global_masks = pred_global_masks[:, 0, :, :]
            results.pred_masks = (pred_global_masks > mask_threshold).float()
        '''
        # recomput bboxes
        # @gxl
        # Get bbox from mask
        pred_boxes = torch.zeros(pred_global_masks.size(0), 4)
        for i in range(pred_global_masks.size(0)):
            mask = pred_global_masks[i].squeeze()
            ys, xs = torch.where(mask)
            if ys.numel() == 0 or xs.numel() == 0:
                continue
            pred_boxes[i] = torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()]).float()
        results.pred_boxes = Boxes(pred_boxes)
        '''
        return results
        
    def postprocess_nms(self, results, output_height, output_width, padded_im_h, padded_im_w, mask_threshold=0.5):
        """
        @gxl
        Using matrix-nms
        For instance segmentation, mask nms is more suitable. Especially for our project.
        The code is based on original condinst's postprocess and solov2's nms part.
        But do same edit to fit condinst's output feature.
        Resize the output instances.
        The input images are often resized when entering an object detector.
        As a result, we often need the outputs of the detector in a different
        resolution from its inputs.
        This function will resize the raw outputs of an R-CNN detector
        to produce outputs according to the desired output resolution.
        Args:
            results (Instances): the raw outputs from the detector.
                `results.image_size` contains the input image resolution the detector sees.
                This object might be modified in-place.
            output_height, output_width: the desired output resolution.
        Returns:
            Instances: the resized output from the model, based on the output resolution
        """

        resized_im_h, resized_im_w = results.image_size
        
        if not results.has("pred_global_masks"):
            results = Instances(results.image_size)
            results.scores = torch.tensor([])
            results.pred_classes = torch.tensor([])
            results.pred_masks = torch.tensor([])
            results.pred_boxes = Boxes(torch.tensor([]))
            return results

        seg_preds = results.pred_global_masks
        N, _, H, W = seg_preds.size()
        pred_classes = results.pred_classes
        pred_scores = results.scores

        # Remove nan
        inds_not_nan = torch.where(torch.isnan(pred_scores), False, True)
        pred_scores = pred_scores[inds_not_nan]
        seg_preds = seg_preds[inds_not_nan]
        pred_classes = pred_classes[inds_not_nan]

        '''
        if num_nan > 0:
            num_inst = len(pred_scores)
            sort_inds = torch.argsort(pred_scores, descending=True)
            sort_inds = sort_inds[:num_inst-num_nan]
           '''
        # New results
        results = Instances((output_height, output_width))
        if len(pred_classes) == 0:
            results = Instances(results.image_size)
            results.scores = torch.tensor([])
            results.pred_classes = torch.tensor([])
            results.pred_masks = torch.tensor([])
            results.pred_boxes = Boxes(torch.tensor([]))
            return results

        # Using mask_threshold to adjust the mask area of prediction.
        seg_masks = seg_preds > mask_threshold
        # Computing area of each mask.
        sum_masks = seg_masks.sum((2, 3)).float()

        # Remove masks which area are 0. Because it will result in nan for seg_scores.
        inds_not_zero = torch.where(sum_masks==0, False, True)
        inds_not_zero = inds_not_zero.reshape(inds_not_zero.shape[0])
        seg_masks = seg_masks[inds_not_zero]
        sum_masks = sum_masks[inds_not_zero]
        seg_preds = seg_preds[inds_not_zero]
        pred_classes = pred_classes[inds_not_zero]
        pred_scores = pred_scores[inds_not_zero]

        seg_scores = ((seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks).sum(1)

        seg_scores = torch.where(torch.isnan(seg_scores), torch.full_like(seg_scores, 0), seg_scores)

        pred_scores *= seg_scores

        # Do filter before matrix nms
        sort_inds = torch.argsort(pred_scores, descending=True)
        if len(sort_inds) > self.max_before_nms:
            sort_inds = sort_inds[:self.max_before_nms]
        seg_masks = seg_masks[sort_inds, :, :]
        seg_preds = seg_preds[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        pred_scores = pred_scores[sort_inds]
        pred_classes = pred_classes[sort_inds]

        # before NMS
        keep = pred_scores >= self.score_threshold
        if keep.sum() == 0:
            results = Instances(results.image_size)
            results.scores = torch.tensor([])
            results.pred_classes = torch.tensor([])
            results.pred_masks = torch.tensor([])
            results.pred_boxes = Boxes(torch.tensor([]))
            return results
            
        seg_masks = seg_masks[keep, :, :]
        seg_preds = seg_preds[keep, :, :]
        pred_classes = pred_classes[keep]
        pred_scores = pred_scores[keep]
        sum_masks =sum_masks[keep]

        fake_label = torch.zeros_like(pred_classes)
        # Matrix NMS
        pred_scores = matrix_nms(fake_label, seg_masks, sum_masks, pred_scores, sigma=self.nms_sigma, kernel=self.nms_kernel)
        # Filter masks which score is less than update threshold.
        # In my project, I find the update threshold is greater than that used in solov2.

        # '>=' not supported between instances of 'list' and 'float'
        if len(pred_scores) == 0:
            results = Instances(results.image_size)
            results.scores = torch.tensor([])
            results.pred_classes = torch.tensor([])
            results.pred_masks = torch.tensor([])
            results.pred_boxes = Boxes(torch.tensor([]))
            return results

        keep = pred_scores >= self.update_threshold
        if keep.sum() == 0:
            results = Instances(results.image_size)
            results.scores = torch.tensor([])
            results.pred_classes = torch.tensor([])
            results.pred_masks = torch.tensor([])
            results.pred_boxes = Boxes(torch.tensor([]))
            return results

        seg_preds = seg_preds[keep, :, :]
        pred_classes = pred_classes[keep]
        pred_scores = pred_scores[keep]
        sum_masks =sum_masks[keep]

        
        # Erosion
        # Resize mask.
        """
        resized_mask_h = 768
        resized_mask_w = 768
        factor = round(resized_mask_h / H)
        seg_preds = aligned_bilinear(
            seg_preds, factor
        )
        seg_preds = seg_preds[:, :, :resized_mask_h, :resized_mask_w]
        seg_preds = F.interpolate(
            seg_preds,
            size=(resized_mask_h, resized_mask_w),
            mode="bilinear", align_corners=False
        )
        
        # erosion
        #num_inst = len(pred_classes)

        #seg_preds = mean_filter(seg_preds, ksize=(3,3))
        #masks = medium_filter(seg_preds.cpu(), ksize=(7,7))
        #masks = masks.cuda()
        #masks = (masks > 0)
        
        masks = seg_preds > 0.2

        masks = erosion(masks, ksize=(7,7))
        #masks = erosion(masks, ksize=(3,3))

        #seg_preds = seg_preds * masks

        seg_preds = torch.nn.functional.softsign(seg_preds + masks)

        seg_preds = mean_filter(seg_preds, ksize=(3,3))
        """
        """
        # Using mask_threshold to adjust the mask area of prediction.
        seg_masks = seg_preds > mask_threshold
        # Computing area of each mask.
        sum_masks = seg_masks.sum((2, 3)).float()

        # Remove masks which area are 0. Because it will result in nan for seg_scores.
        inds_not_zero = torch.where(sum_masks==0, False, True)
        inds_not_zero = inds_not_zero.reshape(inds_not_zero.shape[0])
        seg_masks = seg_masks[inds_not_zero]
        sum_masks = sum_masks[inds_not_zero]
        seg_preds = seg_preds[inds_not_zero]
        pred_classes = pred_classes[inds_not_zero]
        pred_scores = pred_scores[inds_not_zero]

        if len(pred_classes) == 0:
            results = Instances(results.image_size)
            results.scores = torch.tensor([])
            results.pred_classes = torch.tensor([])
            results.pred_masks = torch.tensor([])
            results.pred_boxes = Boxes(torch.tensor([]))
            return results
        """
        # @gxl
        # Fusion mask
        if self.fuse_mask == True:
            num_inst = len(pred_classes)
            
            seg_masks = (seg_preds > mask_threshold).reshape(num_inst, -1).float()
            # inter.
            inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))
            sum_masks_x = sum_masks.expand(num_inst, num_inst)
            # iou.
            iou_matrix = (inter_matrix / (sum_masks_x + sum_masks_x.transpose(1, 0) - inter_matrix)).triu(diagonal=1)
            select_matrix = (iou_matrix > 0.15)
            _ , col_inds = select_matrix.max(0)
            for i, col_ind in enumerate(col_inds):
                col_ind = col_ind.item()
                if col_ind == 0 or col_ind == i:
                    continue
                seg_preds[i] = (seg_preds[i] + seg_preds[col_ind])
                pred_scores[i] = (pred_scores[i] + pred_scores[col_ind]) / 2.0
                pred_classes[i] = 0
                seg_preds[col_ind] = torch.zeros_like(seg_preds[col_ind])
                pred_scores[col_ind] = 0

            keep = pred_scores >= self.update_threshold

            seg_preds = seg_preds[keep, :, :]
            pred_classes = pred_classes[keep]
            pred_scores = pred_scores[keep]

            sort_inds = torch.argsort(pred_scores, descending=True)

            if len(sort_inds) > self.max_per_img:
                sort_inds = sort_inds[:self.max_per_img]

            seg_preds = seg_preds[sort_inds, :, :]
            pred_classes = pred_classes[sort_inds]
            pred_scores = pred_scores[sort_inds]
        

        
        # Resize mask.
        
        N, _, H, W = seg_preds.size()
        mask_h, mask_w = H, W
        factor_h = padded_im_h // mask_h
        factor_w = padded_im_w // mask_w
        assert factor_h == factor_w
        factor = factor_h
        pred_global_masks = aligned_bilinear(
            seg_preds.cpu(), factor
        )
        pred_global_masks = pred_global_masks[:, :, :resized_im_h, :resized_im_w]
        pred_global_masks = F.interpolate(
            pred_global_masks,
            size=(output_height, output_width),
            mode="bilinear", align_corners=False
        )
        pred_global_masks = pred_global_masks[:, 0, :, :]

        #pred_global_masks = seg_preds[:, 0, :, :]

        # Using mask threshold filter pred_mask again.
        pred_global_masks = pred_global_masks > mask_threshold
        results.pred_masks = pred_global_masks
        results.pred_classes = pred_classes
        results.scores = pred_scores

        # Get bbox from mask
        pred_boxes = torch.zeros(pred_global_masks.size(0), 4)
        for i in range(pred_global_masks.size(0)):
            mask = pred_global_masks[i].squeeze()
            ys, xs = torch.where(mask)
            if ys.numel() == 0 or xs.numel() == 0:
                continue
            pred_boxes[i] = torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()]).float()
        results.pred_boxes = Boxes(pred_boxes)

        return results