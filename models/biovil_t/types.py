from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import List

import torch

@dataclass
class ImageModelOutput:
    """
    Output structure for the image encoder.
    """
    img_embedding: torch.Tensor
    patch_embeddings: torch.Tensor
    projected_global_embedding: torch.Tensor
    class_logits: torch.Tensor
    projected_patch_embeddings: torch.Tensor

@unique
class ImageEncoderType(str, Enum):
    BIOVIL_T_SINGLE = "biovil_t_single_image"
    BIOVIL_T_MULTI = "biovil_t_multi_image"

    @classmethod
    def list_all(cls, multi_image_only: bool = False) -> List[ImageEncoderType]:
        """
        Return available encoder types. If multi_image_only, only multi-image variant.
        """
        if multi_image_only:
            return [cls.BIOVIL_T_MULTI]
        return [cls.BIOVIL_T_SINGLE, cls.BIOVIL_T_MULTI]