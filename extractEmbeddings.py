import torch
from torchvision import transforms
from  PIL import Image


# This seems to be working only for color images


dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2','dinov2_vitg14')

image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224, interpolation= transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.Normalize([0.5], [0.5])
])

# load image
dua_image = Image.open('dua_color.jpeg')
test_img = image_transforms(dua_image)[:3].unsqueeze(0)

dino_emb = dinov2_vitg14(test_img)