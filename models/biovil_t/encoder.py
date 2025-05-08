import torch
import torch.nn as nn
from pathlib import Path

from .pretrained import get_biovil_t_image_encoder
from .types import ImageEncoderType


def get_encoder_from_type(encoder_type: ImageEncoderType):
    """
    Factory: return an encoder module matching the requested type.
    - Single-image BioViL-T
    - Multi-image BioViL-T (fuse two views)
    """
    if encoder_type == ImageEncoderType.BIOVIL_T_SINGLE:
        return SingleImageEncoder()
    elif encoder_type == ImageEncoderType.BIOVIL_T_MULTI:
        return MultiImageEncoder()
    else:
        raise ValueError(f"Unsupported encoder type: {encoder_type}")


def get_encoder_output_dim(encoder: nn.Module) -> int:
    """
    Determine the dimension of the encoder's global feature vector.
    Runs a dummy forward pass with a 224×224 RGB image.
    """
    encoder.eval()
    with torch.no_grad():
        # dummy single image
        dummy = torch.zeros(1, 3, 224, 224)
        patch_feats, global_feats = encoder(dummy, return_patch_embeddings=True)
    return global_feats.shape[1]


class SingleImageEncoder(nn.Module):
    """
    Wraps BioViL-T to provide the same interface as the original ImageEncoder:
        forward(x, return_patch_embeddings) -> (patch_feats, global_feats)
    """
    def __init__(self):
        super().__init__()
        # load the full ImageModel and extract its encoder trunk
        image_model = get_biovil_t_image_encoder(pretrained=True)
        self.encoder = image_model.encoder

    def forward(self, x: torch.Tensor, return_patch_embeddings: bool = False):
        # delegate to the underlying encoder
        return self.encoder(x, return_patch_embeddings=return_patch_embeddings)


class MultiImageEncoder(nn.Module):
    """
    For two-image inputs: encode each separately and average their embeddings.
    Provides same interface: forward(current, previous, return_patch_embeddings)
    """
    def __init__(self):
        super().__init__()
        # single-image encoder instance
        self.single = SingleImageEncoder()

    def forward(
        self,
        current_image: torch.Tensor,
        previous_image: torch.Tensor = None,
        return_patch_embeddings: bool = False
    ):
        # encode current view
        patch_curr, global_curr = self.single(current_image, return_patch_embeddings=True)

        if previous_image is not None:
            patch_prev, global_prev = self.single(previous_image, return_patch_embeddings=True)
        else:
            # if no previous, use zeros of same shape
            patch_prev = torch.zeros_like(patch_curr)
            global_prev = torch.zeros_like(global_curr)

        # fuse by averaging
        patch_fused = (patch_curr + patch_prev) / 2
        global_fused = (global_curr + global_prev) / 2

        if return_patch_embeddings:
            return patch_fused, global_fused
        return global_fused