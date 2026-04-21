import torch
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from archs import UNet

# load model
model = UNet(num_classes=1, input_channels=3)

model_path = "models/chase_unet/model.pth"
model.load_state_dict(torch.load(model_path, map_location="cpu"))

model.eval()

# dataset path
image_path = "inputs/inputs/chase/test/images"

images = os.listdir(image_path)

for img_name in images[:3]:

    img_file = os.path.join(image_path, img_name)

    img = Image.open(img_file).convert("RGB")
    img = img.resize((256,256))

    img_np = np.array(img)/255.0

    img_tensor = torch.tensor(img_np).float()
    img_tensor = img_tensor.permute(2,0,1).unsqueeze(0)

    with torch.no_grad():
        pred = model(img_tensor)

    pred = pred.squeeze().numpy()

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.title("Original Image")
    plt.imshow(img_np)
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Predicted Vessel Mask")
    plt.imshow(pred, cmap="gray")
    plt.axis("off")

    plt.show()