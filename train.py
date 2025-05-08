import torch
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig
from models.biovil_t.dataset import BreastReportDataset
from models.biovil_t.image_model import BreastCancerImageModel
from models.biovil_t.vlp_model import BreastCancerVLP
from peft import TaskType

################################################
import os
import pandas as pd

root_dir = "C:\\Users\\LENONVO\\Downloads\\dataset_RO\\dataset"

data = []

for patient_folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, patient_folder)
    
    if not os.path.isdir(folder_path):
        continue

    # Collect all images and the text file
    images = [f for f in os.listdir(folder_path) if f.endswith(('.dcm'))]
    txts = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    if len(images) < 2 or len(txts) == 0:
        print(f"Skipping {patient_folder} — not enough images or no report.")
        continue

    # Sort images to ensure consistent order
    images = sorted(images)
    image_path_1 = os.path.join(folder_path, images[0])
    image_path_2 = os.path.join(folder_path, images[1])
    
    # Read the report text
    txt_path = os.path.join(folder_path, txts[0])
    with open(txt_path, 'r', encoding='utf-8') as f:
        report = f.read().strip()

    data.append({
        'image_path_1': image_path_1,
        'image_path_2': image_path_2,
        'report_text': report
    })

# Create the DataFrame
train_df = pd.DataFrame(data)
print(f"Found {len(train_df)} samples.")

########################################################

train_ds = BreastReportDataset(train_df)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)


lora_conf = LoraConfig(r=4, lora_alpha=16, target_modules=["qkv"], task_type=TaskType.SEQ_2_SEQ_LM)
img_model = BreastCancerImageModel(freeze_backbone=True, lora_config=lora_conf)
vlp = BreastCancerVLP(img_model).cuda()

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, vlp.parameters()), lr=5e-5)

for epoch in range(5):
    vlp.train()
    total_loss = 0
    for images, input_ids, attn_mask in train_loader:
        # move to GPU
        images = tuple(x.cuda() for x in images)
        input_ids = input_ids.cuda()
        attn_mask = attn_mask.cuda()

        optimizer.zero_grad()
        loss, _ = vlp(images, input_ids, attn_mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch} avg loss = {total_loss/len(train_loader)}")