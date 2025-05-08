import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from health_multimodal.image.model.encoder import get_encoder_from_type, get_encoder_output_dim
from health_multimodal.image.model.types import ImageModelOutput

class BreastCancerImageModel(nn.Module):
    """
    Image encoder for breast cancer report generation.
    - Uses BioViL-T as backbone.
    - Injects LoRA adapters into the ViT attention layers.
    - Projects features to joint vision-language space.
    - Optionally freeze backbone except LoRA.
    """
    def __init__(
        self,
        img_encoder_type: str = "biovil_t",
        loraconfig: LoraConfig = LoraConfig(
            r=4,    
            lora_alpha=16,
            target_modules=["qkv"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        ),
        joint_feature_size: int = 128,  
    ):
        super().__init__()
        # 1. Load BioViL-T encoder
        self.encoder = get_encoder_from_type(img_encoder_type)
        # 2. Optionally freeze full backbone
        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False
        # 3. Inject LoRA into the ViT (assumes encoder.vit attribute)
        #    Adapt query, key, value projections to new domain
        if hasattr(self.encoder, 'vit'):
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["query", "key", "value"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
            )
            self.encoder.vit = get_peft_model(self.encoder.vit, lora_config)
        # 4. Build projection head: patch -> joint space
        feat_dim = get_encoder_output_dim(self.encoder)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, joint_feature_size),
            nn.ReLU(),
            nn.Linear(joint_feature_size, joint_feature_size)
        )

    def forward(self, x: torch.Tensor) -> ImageModelOutput:
        # 1. Extract patch and global features
        patch_feats, global_feats = self.encoder(x, return_patch_embeddings=True)
        # 2. Project to joint space
        B, D, H, W = patch_feats.shape
        patches = patch_feats.view(B, D, -1).transpose(1, 2) 
        projected = self.projector(patches)  
        joint_global = projected.mean(dim=1)  
        return ImageModelOutput(
            img_embedding=global_feats,
            patch_embeddings=patch_feats,
            projected_patch_embeddings=projected.view(B, H, W, -1).permute(0, 3, 1, 2),
            projected_global_embedding=joint_global,
            class_logits=None
        )