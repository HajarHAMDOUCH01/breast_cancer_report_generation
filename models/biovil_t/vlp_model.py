import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
from models.biovil_t.image_model import BreastCancerImageModel

class BreastCancerVLP(nn.Module):
    """
    Multimodal Vision-Language model for breast cancer report generation.
    Combines a fine-tuned BioViL-T image encoder with a T5 text decoder.

    Args:
        image_model: instance of BreastCancerImageModel producing joint image embeddings.
        text_decoder_name: Hugging Face model name for the text decoder (e.g., 't5-small').
    """
    def __init__(self,
                 image_model: BreastCancerImageModel,
                 text_decoder_name: str = "google/t5-efficient-small"):
        super().__init__()
        # vision backbone + LoRA + projector
        self.image_model = image_model
        # text generation model
        self.text_decoder = AutoModelForSeq2SeqLM.from_pretrained("google/t5-efficient-small")


        # tokenizer = AutoTokenizer.from_pretrained("google/t5-efficient-small")
        # model = AutoModelForSeq2SeqLM.from_pretrained("google/t5-efficient-small")

        # tokenizer for inference convenience
        self.tokenizer = AutoTokenizer.from_pretrained(text_decoder_name)
        # linear map from image joint_feature_size to text encoder hidden size
        img_dim = image_model.projector[-1].out_features
        txt_dim = self.text_decoder.config.d_model
        self.img2txt = nn.Linear(img_dim, txt_dim)

    def forward(self,
                images: tuple[torch.Tensor, ...],
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  
        """
        Forward pass for training.

        Args:
            images: tuple of one or more image tensors, each [B,C,H,W]
            input_ids: token ids for target report [B, T]
            attention_mask: attention mask for decoder [B, T]
        Returns:
            loss: cross-entropy loss against input_ids
            logits: raw decoder output logits [B, T, vocab_size]
        """
        # 1) obtain joint global embedding from image model
        # image_model returns ImageModelOutput
        out = self.image_model(images if len(images)>1 else images[0])
        img_feat = out.projected_global_embedding  # [B, joint_dim]
        # 2) map to text encoder dimension and create pseudo encoder outputs
        enc_embeds = self.img2txt(img_feat).unsqueeze(1)  # [B,1,txt_dim]
        # 3) call T5 with forced teacher forcing
        #    by passing encoder_outputs and decoder inputs
        decoder_outputs = self.text_decoder(
            encoder_outputs=(enc_embeds,),
            decoder_input_ids=input_ids,
            decoder_attention_mask=attention_mask, ### =>>
            labels=input_ids,
        )
        return decoder_outputs.loss, decoder_outputs.logits

    def generate_report(self,
                        image: torch.Tensor,
                        max_length: int = 256) -> str:
        """
        Inference: generate a text report for a single image or tuple of images.

        Args:
            image: a single tensor [C,H,W] or tuple of such tensors
            max_length: maximum length of generated report
        Returns:
            Generated report string
        """
        self.eval()
        # prepare batch of size 1
        if isinstance(image, tuple):
            imgs = tuple(img.unsqueeze(0) for img in image)
        else:
            imgs = (image.unsqueeze(0),)
        # image forward
        with torch.no_grad():
            out = self.image_model(imgs if len(imgs)>1 else imgs[0])
            img_feat = out.projected_global_embedding  # [1, joint_dim]
            enc_embeds = self.img2txt(img_feat).unsqueeze(1)
            # generate tokens
            generated_ids = self.text_decoder.generate(
                encoder_outputs=(enc_embeds,),
                max_length=max_length,
            )
        # decode to text
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
