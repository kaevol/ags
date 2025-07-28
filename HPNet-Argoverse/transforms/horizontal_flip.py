import math
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

from utils import wrap_angle


class HorizontalFlip(BaseTransform):
    def __init__(self,
                 flip_p=0.5):
        super(HorizontalFlip, self).__init__()
        self.flip_p = flip_p

    def flip_position_and_heading(self, position, heading):
        position[..., 0] = -position[..., 0]
        angle = wrap_angle(math.pi - heading)
        return position, angle

    def __call__(self, data: HeteroData) -> HeteroData:
        if torch.rand(1).item() < self.flip_p:
            # Flip agent data
            data['agent']['position'], data['agent']['heading'] = self.flip_position_and_heading(
                data['agent']['position'], data['agent']['heading'])
            # Flip velocity x-component
            data['agent']['velocity'][..., 0] = -data['agent']['velocity'][..., 0]

            # Flip lane data
            data['lane']['position'], data['lane']['heading'] = self.flip_position_and_heading(data['lane']['position'],
                                                                                               data['lane']['heading'])

            # Flip polyline data
            data['polyline']['position'], data['polyline']['heading'] = self.flip_position_and_heading(
                data['polyline']['position'], data['polyline']['heading'])
            # Swap left and right polyline sides
            data['polyline']['side'] = 2 - data['polyline']['side']

            # Swap left and right neighbor edges
            data['lane', 'lane']['left_neighbor_edge_index'], data['lane', 'lane']['right_neighbor_edge_index'] = \
            data['lane', 'lane']['right_neighbor_edge_index'], data['lane', 'lane']['left_neighbor_edge_index']
        return data