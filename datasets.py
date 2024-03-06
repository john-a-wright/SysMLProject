import torch
from torchvision import datasets, transforms

# Set up the data transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Set up the datasets
imagenet_dataset = datasets.ImageNet('imagenet', split='train', transform=transform)

# Set up the data loaders
imagenet_loader = torch.utils.data.DataLoader(imagenet_dataset, batch_size=64, shuffle=True)

# Example usage
for images, labels in imagenet_loader:
    # Do something with the ImageNet data
    pass
