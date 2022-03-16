# coding=UTF-8
import torch
from torch.nn import functional as F
from torch import nn

from adet.utils.comm import compute_locations, aligned_bilinear
# @gxl Tensorboard
from adet.utils.puttotensorboard import put_image_to_tensorboard
from detectron2.utils.events import EventStorage, get_event_storage
from adet.layers.activation import mish

def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    # loss = 1. - (2 * intersection / union)
    # @gxl
    # Laplace smoothing. https://zhuanlan.zhihu.com/p/86704421
    loss = 1. - ((2 * intersection + 1.0) / (union + 1.0))
    return loss


def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(
        params, weight_nums + bias_nums, dim=1
    ))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts)

    return weight_splits, bias_splits


def build_dynamic_mask_head(cfg):
    return DynamicMaskHead(cfg)

# @gxl
'''
def loss_gxl(mask_scores, gt_bitmasks, gt_instances, gt_inds, special_classes):
    gt_labels = torch.cat([per_im.gt_classes for per_im in gt_instances])
    gt_labels = gt_labels[gt_inds]
    alpha = 0.2
    beta = 0.8
    # Every bitmap has it's own label
    inds_spc_labels = []
    inds_nom_labels = []
    for ind, label in enumerate(gt_labels):
        if label in special_classes:
            inds_spc_labels.append(ind)
        else:
            inds_nom_labels.append(ind)

    # compute loss
    loss_nom = dice_coefficient(mask_scores[inds_nom_labels], gt_bitmasks[inds_nom_labels])
    loss_spc = dice_coefficient(mask_scores[inds_spc_labels], gt_bitmasks[inds_spc_labels])
    loss_1 = dice_coefficient(mask_scores,gt_bitmasks)
    loss_spc_1 = loss_1[inds_spc_labels]
    loss_nom_1 = loss_1[inds_nom_labels]
    loss_1 = loss_nom_1.mean() * beta + loss_spc_1.mean() * alpha
    loss = loss_nom.mean() * beta + loss_spc.mean() * alpha
    return loss
'''

class DynamicMaskHead(nn.Module):
    def __init__(self, cfg):
        super(DynamicMaskHead, self).__init__()
        '''
         Default values:
            Mask_head.num_layers=3
            Mask_head.channels=8
            Mask_branch.out_channels=8
            Mask_out_stride=2
            Mask_head.disable_rel_coords=false
            Fcos.size_of_insterst=[64,128,256,512]

            weight_nums=[80,64,8]
            bias_nums=[8,8,1]
            num_gen_params=169
         ''' 
        self.num_layers = cfg.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS
        self.channels = cfg.MODEL.CONDINST.MASK_HEAD.CHANNELS
        self.in_channels = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS
        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE
        self.disable_rel_coords = cfg.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS

        # @gxl
        # For compute mask loss
        self.special_classes = cfg.MODEL.CONDINST.MASK_HEAD.SPECIAL_CLASSES
        self.special_classes_loss = cfg.MODEL.CONDINST.MASK_HEAD.SPECIAL_CLASSES_LOSS
        self.alpha = cfg.MODEL.CONDINST.MASK_HEAD.ALPHA
        self.beta = cfg.MODEL.CONDINST.MASK_HEAD.BETA
        self.debug = cfg.SOLVER.DEBUG

        soi = cfg.MODEL.FCOS.SIZES_OF_INTEREST
        self.register_buffer("sizes_of_interest", torch.tensor(soi + [soi[-1] * 2]))

        weight_nums, bias_nums = [], []
        for l in range(self.num_layers):
            if l == 0:
                if not self.disable_rel_coords:
                    weight_nums.append((self.in_channels + 2) * self.channels)
                else:
                    weight_nums.append(self.in_channels * self.channels)
                bias_nums.append(self.channels)
            elif l == self.num_layers - 1:
                weight_nums.append(self.channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.channels * self.channels)
                bias_nums.append(self.channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        '''
        torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
        input – input tensor of shape (minibatch,in_channels,iH,iW)
        weight – filters of shape (out_channels, in_channels/groups,kH,kW)
        bias – optional bias tensor of shape (out_channels) . Default: None
        stride – the stride of the convolving kernel. Can be a single number or a tuple (sH, sW). Default: 1
        padding – implicit paddings on both sides of the input. Can be a single number or a tuple (padH, padW). Default: 0
        dilation – the spacing between kernel elements. Can be a single number or a tuple (dH, dW). Default: 1
        groups – split input into groups, in_channels should be divisible by the number of groups. Default: 1
        '''
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                #x = F.relu(x)
                x = mish(x)
        return x

    def mask_heads_forward_with_coords(
            self, mask_feats, mask_feat_stride, instances
    ):
        # @gxl
        # locations.size = (w*h, 2)
        locations = compute_locations(
            mask_feats.size(2), mask_feats.size(3),
            stride=mask_feat_stride, device=mask_feats.device
        )
        n_inst = len(instances)

        im_inds = instances.im_inds
        mask_head_params = instances.mask_head_params
        #print(mask_feats.shape)
        N, _, H, W = mask_feats.size()

        # disable_rel_coords=False
        if not self.disable_rel_coords:
            # compute the relative coordinates of the input prediction and the original image
            instance_locations = instances.locations
            relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
            relative_coords = relative_coords.permute(0, 2, 1).float()
            soi = self.sizes_of_interest.float()[instances.fpn_levels]
            relative_coords = relative_coords / soi.reshape(-1, 1, 1)
            relative_coords = relative_coords.to(dtype=mask_feats.dtype)

            # attention the mask_feats will be reshaped to [instances_nums, in_channels, H*W]
            mask_head_inputs = torch.cat([
                relative_coords, mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)
            ], dim=1)
        else:
            mask_head_inputs = mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)

        # change the dim to 1, first dim means batch of input
        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)
        #print('mask_head_inputs: ', mask_head_inputs.shape)
        weights, biases = parse_dynamic_params(
            mask_head_params, self.channels,
            self.weight_nums, self.bias_nums
        )

        mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, n_inst)

        mask_logits = mask_logits.reshape(-1, 1, H, W)
        #print(mask_feat_stride)
        #print(self.mask_out_stride)
        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0
        mask_logits = aligned_bilinear(mask_logits, int(mask_feat_stride / self.mask_out_stride))
        #print('mask_logits: ', mask_logits.shape)
        #exit()
        return mask_logits.sigmoid()

    def __call__(self, mask_feats, mask_feat_stride, pred_instances, gt_instances=None):
        if self.training:
            gt_inds = pred_instances.gt_inds
            gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in gt_instances])
            gt_bitmasks = gt_bitmasks[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)

            if len(pred_instances) == 0:
                loss_mask = mask_feats.sum() * 0 + pred_instances.mask_head_params.sum() * 0
            else:
                mask_scores = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances
                )
                # @gxl
                # Add image to tensorboard
                if self.debug:
                    storage = get_event_storage()
                    put_image_to_tensorboard(mask_scores.permute(1, 0, 2, 3), storage, 'mask_score')

                #print(mask_scores.shape)
                #print(gt_bitmasks.shape)
                mask_losses = dice_coefficient(mask_scores, gt_bitmasks)
                # @gxl
                # Choosing the instances which are belong to the special classes.
                # Changing the ratio of different classes in the loss function.
                # That's because in our task, the mask of some classes is not as important as others.
                # mask_losses_spc = loss_gxl(mask_scores, gt_bitmasks, gt_instances, gt_inds, self.special_classes)
                if self.special_classes_loss:
                    # Get index of special and normal classes
                    gt_labels = torch.cat([per_im.gt_classes for per_im in gt_instances])
                    gt_labels = gt_labels[gt_inds]
                    alpha = self.alpha
                    beta = self.beta
                    # Every bitmap has it's own label
                    inds_spc_labels = []
                    inds_nom_labels = []
                    for ind, label in enumerate(gt_labels):
                        if label in self.special_classes:
                            inds_spc_labels.append(ind)
                        else:
                            inds_nom_labels.append(ind)
                    len_nom = len(inds_nom_labels)
                    len_spc = len(inds_spc_labels)
                    len_inst = len_nom + len_spc
                    # Only one index list may be empty
                    #print('len_spc: ', len_spc, ' len_nom: ', len_nom) 
                    '''
                    else:
                    if len_nom == 0 and len_spc == 0:
                        loss_mask = mask_losses.mean()
                    '''
                    if len_nom == 0:
                        loss_mask = mask_losses.mean() * beta
                    elif len_spc == 0:
                        loss_mask = mask_losses.mean() * alpha
                    else:
                        loss_spc = mask_losses[inds_spc_labels]
                        loss_nom = mask_losses[inds_nom_labels]
                        loss_mask = (loss_spc.mean() * len_spc / len_inst) * alpha +\
                                    (loss_nom.mean() * len_nom / len_inst) * beta
                        #print('loss_spc: ', loss_spc.mean(), ' loss_nom: ', loss_nom.mean())
                else:
                    loss_mask = mask_losses.mean()
            #print('mask_loss: ', loss_mask)
            return loss_mask.float()
        else:
            if len(pred_instances) > 0:
                mask_scores = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances
                )
                pred_instances.pred_global_masks = mask_scores.float()

            return pred_instances
