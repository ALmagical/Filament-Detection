import copy
from typing import List

import torch
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.modeling.anchor_generator import ANCHOR_GENERATOR_REGISTRY 



class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


def _create_grid_offsets(size, stride, offset, device):
    grid_height, grid_width = size
    shifts_start = offset * stride
    shifts_x = torch.arange(
        shifts_start, grid_width * stride + shifts_start, step=stride,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        shifts_start, grid_height * stride + shifts_start, step=stride,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    return shift_x, shift_y

@ANCHOR_GENERATOR_REGISTRY.register()   
class ShiftGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set of shifts.
    """
    # TODO: unused_arguments: cfg
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        self.num_shifts = cfg.MODEL.SHIFT_GENERATOR.NUM_SHIFTS
        self.strides    = [x.stride for x in input_shape]
        self.offset     = cfg.MODEL.SHIFT_GENERATOR.OFFSET
        # fmt: on

        self.num_features = len(self.strides)

    @property
    def num_cell_shifts(self):
        return [self.num_shifts for _ in self.strides]

    def grid_shifts(self, grid_sizes, device):
        shifts_over_all = []
        for size, stride in zip(grid_sizes, self.strides):
            shift_x, shift_y = _create_grid_offsets(size, stride, self.offset, device)
            shifts = torch.stack((shift_x, shift_y), dim=1)

            shifts_over_all.append(shifts.repeat_interleave(self.num_shifts, dim=0))

        return shifts_over_all

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): list of backbone feature maps on which to generate shifts.
        Returns:
            list[list[Tensor]]: a list of #image elements. Each is a list of #feature level tensors.
                The tensors contains shifts of this image on the specific feature level.
        """
        num_images = len(features[0])
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        shifts_over_all = self.grid_shifts(grid_sizes, features[0].device)

        shifts = [copy.deepcopy(shifts_over_all) for _ in range(num_images)]
        return shifts