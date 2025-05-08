import torch.nn as nn
from transformers import T5ForConditionalGeneration
from models.biovil_t.image_model.py thon import BreastCancerImageModel

class BreastCancerVLP(nn.Module):
    def __init__(self, image_model: BreastCancerImageModel, text_model_name="google/t5-efficient-small"):
        super().__init__()
        self.image_model = image_model
        self.text_model = T5ForConditionalGeneration.from_pretrained(text_model_name)
        # project image joint embedding → text_model encoder hidden size
        d_img = image_model.projector[-1].out_features
        d_txt = self.text_model.config.d_model
        self.img_to_txt = nn.Linear(d_img, d_txt)

    def forward(self, images, input_ids, attention_mask):
        if len(images)==2:
            patch, global_emb = self.image_model(images[0], images[1])
        else:
            patch, global_emb = self.image_model(images[0])
        encoder_embeds = self.img_to_txt(global_emb).unsqueeze(1)  
        outputs = self.text_model(
          inputs_embeds=encoder_embeds,
          attention_mask=None,
          decoder_input_ids=input_ids,
          decoder_attention_mask=attention_mask,
          labels=input_ids
        )
        return outputs.loss, outputs.logits
