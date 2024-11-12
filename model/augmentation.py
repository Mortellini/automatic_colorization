import torchvision.transforms as transforms

# Define a set of augmentations
transform_train = transforms.Compose([
    #transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally
    transforms.RandomResizedCrop(size=256, scale=(0.6, 1.0)),  # Randomly crop and resize
    transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.1, hue=0.1),  # Random brightness, contrast, etc.
    transforms.RandomRotation(degrees=30),  # Randomly rotate images by +/-10 degrees
    #transforms.Grayscale(num_output_channels=1),
    #transforms.ToTensor(),  # Convert image to tensor
])

