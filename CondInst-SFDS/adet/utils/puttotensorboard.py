# -*- coding: utf-8 -*-
# @gxl
# Put input tensor(image) to tensorboard

import torch
import math
from detectron2.utils.visualizer import Visualizer

def put_image_to_tensorboard(input, storage, name, concat=False):
    '''
        input: Tensor. I am not sure, perhaps list can also be used. (batch, channel, h, w)
        storage: A handle of Event storage. You can got this by using get_event_storage() function.
        name: String. The title of the tensor you want show in tensorboard.
    '''
    if torch.is_tensor(input):
        input = input.cpu()
    if concat:
        for i, input_feat in enumerate(input):
            c, _, _ = input_feat.shape
            input_feat = (input_feat - input_feat.min()) / (input_feat.max() - input_feat.min())
            for j in range(c):
                storage.put_image((name + '_' + str(i + j + 1)), input_feat)
    else:
        for i, input_feat in enumerate(input):
            c, h, w = input_feat.shape
            input_feat = (input_feat - input_feat.min()) / (input_feat.max() - input_feat.min())
            # Compute the number of rows
            rows = c ** 0.5
            rows = math.floor(rows)
            cols = int(c / rows)
            # The tensor will be reshape
            for j in range(rows):
                feat = input_feat[j * cols]
                for k in range(1, cols):
                    feat = torch.cat((feat, input_feat[j * cols + k]), dim=1)
                if j == 0:
                    feats = feat
                else:
                    feats = torch.cat((feats, feat), dim=0)
                if j == rows - 1:
                    storage.put_image((name + '_' + str(i + 1)), feats.reshape(1, rows * h, -1))

def vis_bbox(images, instances, storage):

    pass