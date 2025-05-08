import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoTokenizer

from models.biovil_t.image_transform import image_transform  


class BreastReportDataset(Dataset):
    """
    PyTorch Dataset for (1 or 2) breast images → report text.

    Expects a DataFrame with columns:
      - 'image_path_1' : str, path to first image
      - 'image_path_2' : str or NaN, path to second image (optional)
      - 'report_text'  : str, the ground‑truth report
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform=image_transform,
        tokenizer_name: str = "google/t5-efficient-small",
        max_len: int = 256,
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ——— load and transform first image ———
        img1_path = row["image_path_1"]
        img1 = Image.open(img1_path).convert("RGB")
        x1 = self.transform(img1)

        # ——— load and transform second image if present ———
        img2_path = row.get("image_path_2", None)
        if isinstance(img2_path, str) and os.path.isfile(img2_path):
            img2 = Image.open(img2_path).convert("RGB")
            x2 = self.transform(img2)
            images = (x1, x2)
        else:
            images = (x1,)

        # ——— tokenize the report text ———
        report = row["report_text"]
        tokens = self.tokenizer(
            report,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        # remove the extra batch‑dim
        input_ids = tokens.input_ids.squeeze(0)
        attention_mask = tokens.attention_mask.squeeze(0)

        return images, input_ids, attention_mask