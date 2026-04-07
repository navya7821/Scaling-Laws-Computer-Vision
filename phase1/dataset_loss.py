import os
import torch
from torch import nn, optim
from torchvision import models, transforms
from PIL import Image
import csv

# ======================
# CONFIGURATION
# ======================
DATASETS = {
    100: r"C:\Users\taral\OneDrive\Desktop\Coe_internship\Images\100Images",
    200: r"C:\Users\taral\OneDrive\Desktop\Coe_internship\Images\200Images",
    300: r"C:\Users\taral\OneDrive\Desktop\Coe_internship\Images\300Images"
}
OUTPUT_CSV = r"C:\Users\taral\OneDrive\Desktop\Coe_internship\dataset_vs_loss.csv"
EPOCHS = 5 

# ======================
# TRANSFORM
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ======================
# MODEL
# ======================
resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
for param in resnet18.parameters():
    param.requires_grad = False
resnet18.fc = nn.Linear(resnet18.fc.in_features, 512)

criterion = nn.MSELoss()
optimizer = optim.SGD(resnet18.fc.parameters(), lr=0.001)

# ======================
# PREPARE CSV STRUCTURE
# ======================

max_images = max([len(os.listdir(path)) for path in DATASETS.values()])
csv_rows = []
for i in range(1, max_images + 1):
    csv_rows.append({
        'image_name': f"{i}.JPEG",
        'loss_100': 'NULL',
        'loss_200': 'NULL',
        'loss_300': 'NULL'
    })

# ======================
# FUNCTION TO PROCESS A DATASET
# ======================
def process_dataset(size, folder_path, column_name):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()  # numeric/alphabetical order

    images = []
    image_names = []
    for img_file in image_files:
        try:
            img_path = os.path.join(folder_path, img_file)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            images.append(img_tensor)
            image_names.append(img_file)
        except Exception as e:
            print(f"Error loading {img_file}: {e}")

    resnet18.train()
    for epoch in range(EPOCHS):
        for img_tensor, img_name in zip(images, image_names):
            optimizer.zero_grad()
            input_vec = img_tensor.unsqueeze(0)
            try:
                output_vec = resnet18(input_vec)
                loss_val = criterion(output_vec.view(-1), input_vec.view(-1)[:output_vec.numel()])
                loss_val.backward()
                optimizer.step()
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                loss_val = None
            
            # Find index in csv_rows
            index = int(''.join(filter(str.isdigit, img_name))) - 1
            if 0 <= index < len(csv_rows):
                csv_rows[index][column_name] = float(loss_val) if loss_val is not None else 'NULL'

# ======================
# PROCESS ALL DATASETS
# ======================
for size, path in DATASETS.items():
    print(f"Processing dataset of size {size}...")
    col_name = f"loss_{size}"
    process_dataset(size, path, col_name)

# ======================
# SAVE CSV
# ======================
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['image_name', 'loss_100', 'loss_200', 'loss_300'])
    writer.writeheader()
    for row in csv_rows:
        writer.writerow(row)

print("Dataset vs loss logging completed!")
