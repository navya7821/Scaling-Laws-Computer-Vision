import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import pandas as pd

# ------------------ SETTINGS ------------------
data_folder = r"C:\Users\taral\OneDrive\Desktop\Coe_internship\Images\200Images"  # folder with all images directly inside
num_epochs = 5           # for legible pre-loss values
batch_size = 1           # one image per forward pass
learning_rate = 0.001
models_to_run = ["resnet18", "resnet50", "resnet101"]

# ------------------ TRANSFORMS ------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ------------------ CUSTOM DATASET ------------------
class SingleFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = sorted(os.listdir(folder_path), key=lambda x: int(os.path.splitext(x)[0]))
        self.folder_path = folder_path
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.files[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0  # dummy label

dataset = SingleFolderDataset(data_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# ------------------ HELPER FUNCTIONS ------------------
def get_model(name):
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif name == "resnet101":
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    else:
        raise ValueError(f"Unknown model {name}")
    
    # Freeze all layers except FC
    for param in model.parameters():
        param.requires_grad = False
    
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_features)  # identity-like FC
    return model

# ------------------ CSV SETUP ------------------
all_image_names = [os.path.basename(path) for path in dataset.files]
loss_dict = {img: {} for img in all_image_names}

# ------------------ MAIN LOOP ------------------
for model_name in models_to_run:
    print(f"\nTraining {model_name}...")
    model = get_model(model_name)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training FC layer
    for epoch in range(num_epochs):
        for inputs, _ in dataloader:  # labels not used
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, outputs.detach())  # dummy backward to train FC
            loss.backward()
            optimizer.step()

    # Log "loss" per image using L2 norm of FC output
    for i, (inputs, _) in enumerate(dataloader):
        with torch.no_grad():
            outputs = model(inputs)
            loss_value = outputs.norm().item()  # L2 norm as pre-loss
        img_name = all_image_names[i]
        loss_dict[img_name][f"{model_name}_loss"] = loss_value

# ------------------ SAVE CSV ------------------
df = pd.DataFrame.from_dict(loss_dict, orient="index")
df.index.name = "image_name"
csv_path = r"C:\Users\taral\OneDrive\Desktop\Coe_internship\model_size_vs_loss.csv"
df.to_csv(csv_path)

print(f"\n✅ Model-size vs loss CSV created at {csv_path}")


