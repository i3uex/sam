"""
Enumeration to simplify SAM usage.

Each item contains the name SAM understands
and corresponding path to its weights. This way, given an instance of the
enumeration the rest of needed values are available. Besides, there is no
possible misspelling as the enumeration values are provided to the developer.

Items are sorted from the smallest (ViT_B) to the biggest (ViT_H). The biggest
is the more capable version of the model, but also the one that needs more
resources.
"""

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
