from collections import namedtuple
from enum import Enum

SamModelItem = namedtuple('SamModelItem', ['name', 'checkpoint'])


class SamModel(SamModelItem, Enum):
    """
    SAM available models.
    """

    ViT_B = SamModelItem(
        name='vit_b',
        checkpoint='model_checkpoints/sam_vit_b_01ec64.pth'
    )
    ViT_L = SamModelItem(
        name='vit_l',
        checkpoint='model_checkpoints/sam_vit_l_0b3195.pth'
    )
    ViT_H = SamModelItem(
        name='vit_h',
        checkpoint='model_checkpoints/sam_vit_h_4b8939.pth'
    )
