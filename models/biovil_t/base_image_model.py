import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
# from models.biovil_t.encoder import get_encoder_from_type, get_encoder_output_dim
from models.biovil_t.types import ImageModelOutput

class ImageModel(nn.Module):
    def __init__(
        self,
        img_encoder_type: str = "biovil_t_multi_image",
        lora_config: LoraConfig = LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=["query","key","value"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        ),
        joint_feature_size: int = 128,
        freeze_backbone: bool = False,
    ):
        from models.biovil_t.encoder import get_encoder_from_type, get_encoder_output_dim
        super().__init__()
        # 1) load the wrapper (single‑ or multi‑image) which has an internal .vit
        self.encoder = get_encoder_from_type(img_encoder_type)
        # 2) optionally freeze every weight in that wrapper
        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False
        # 3) inject LoRA only into the .vit attribute
        if hasattr(self.encoder, "vit"):
            self.encoder.vit = get_peft_model(self.encoder.vit, lora_config)
        else:
            raise RuntimeError("Cannot find .vit in encoder to apply LoRA")
        # 4) build the projector from encoder’s global‐feature dim → joint space
        feat_dim = get_encoder_output_dim(self.encoder)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, joint_feature_size),
            nn.ReLU(),
            nn.Linear(joint_feature_size, joint_feature_size),
        )


    def forward(self, x: torch.Tensor) -> ImageModelOutput:
        # 1. Extract patch-wise and global features from encoder
        patch_feats, global_feats = self.encoder(x, return_patch_embeddings=True)
        # 2. Flatten patches and project to joint space
        B, D, H, W = patch_feats.shape
        patches = patch_feats.view(B, D, -1).transpose(1, 2)  # (B, N, D)
        projected_patches = self.projector(patches)          # (B, N, joint_dim)
        # 3. Compute global joint embedding by averaging patch embeddings
        joint_global = projected_patches.mean(dim=1)         # (B, joint_dim)
        # 4. Reshape projected patches back to spatial map
        projected_patch_map = (projected_patches
                              .view(B, H, W, -1)
                              .permute(0, 3, 1, 2))      # (B, joint_dim, H, W)
        # 5. Return structured output
        return ImageModelOutput(
            img_embedding=global_feats,
            patch_embeddings=patch_feats,
            projected_patch_embeddings=projected_patch_map,
            projected_global_embedding=joint_global,
            class_logits=None
        )
