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
