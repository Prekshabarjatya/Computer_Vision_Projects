#PyTorch
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision
from PIL import Image
from torchvision import transforms
#Getting details about the data of animals 
csv_file = 'train_labels.csv'
df = pd.read_csv(csv_file)
category_totals = df.iloc[:, 1:].sum()
plt.figure(figsize=(10, 6))
category_totals.plot(kind='bar', color='skyblue')
plt.title('Total Count of Each Animal Category', fontsize=16)
plt.xlabel('Animal Categories', fontsize=14)
plt.ylabel('Total Count', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

hog_image_pil = Image.open("train_features/ZJ000005.jpg") 
hog_image_pil.show()
print(hog_image_pil.size)
print(hog_image_pil.mode)
hog_tensor = transforms.ToTensor()(hog_image_pil)
print(hog_image_pil.dtype)

#Antelop Image 
antelop_image_pil = Image.open("train_features/ZJ000007.jpg")
antelop_image_pil.show()
print(antelop_image_pil.size)
print(antelop_image_pil.mode)
antelop_tensor = transforms.ToTensor()(antelop_image_pil)
print(antelop_image_pil.dtype)


fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5))
# Plot red channel
red_channel = antelope_tensor[0, :, :]
ax0.imshow(red_channel, cmap="Reds")
ax0.set_title("Antelope, Red Channel")
ax0.axis("off")
# Plot green channel
green_channel = antelope_tensor[1,:,:]
ax1.imshow(green_channel, cmap="Greens")
ax1.set_title("Antelope, Green Channel")
ax1.axis("off")
# Plot blue channel
blue_channel = antelope_tensor[2,:,:]
ax2.imshow(blue_channel, cmap="Blues")
ax2.set_title("Antelope, Blue Channel")
ax2.axis("off")
plt.tight_layout();

