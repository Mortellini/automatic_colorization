import os

import torch
import torch.optim as optim
import torch.nn as nn
from eccv16 import eccv16
from torchvision import datasets
from torch.utils.data import DataLoader
from colorization_dataset import *
from perceptual_loss import *
from augmentation import *
from util import *


# Load pts_in_hull file (contains the 313 quantized ab values)
pts_in_hull = np.load('pts_in_hull.npy')

for i in range(3):
    print("Durchlauf: " + str(i))
    # Create your model
    model = eccv16(pretrained=True, use_finetune_layer=True, use_training=False)

    # Freeze all layers except the fine-tuning layer
    for param in model.parameters():
        param.requires_grad = True

    # Unfreeze model8 (layer before the fine-tuning layer)
    for param in model.model8.parameters():
        param.requires_grad = True

    for param in model.finetune_layer.parameters():
        param.requires_grad = True  # Unfreeze fine-tuning layer

    # Define current working directory (where the script is running)
    save_dir = os.getcwd()  # This gets the current working directory

    # Define your optimizer and loss function
    optimizer = optim.Adam(model.finetune_layer.parameters(), lr=0.0001)
    # Instantiate perceptual loss
    perceptual_loss = PerceptualLoss(vgg)

    # MSE Loss for pixel-wise color comparison
    mse_loss = nn.MSELoss()
    # Cross Entropy Function
    criterion = nn.CrossEntropyLoss()

    # Load your training data
    train_dataset = ColorizationDataset('images/', transform_train)
    print(len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Training loop
    num_epochs = 20
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            input_l, ground_truth_ab = data  # Get your data

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output_ab = model(input_l)
            # Expected output_ab shape: [batch_size, 313, 256, 256]

            # Quantize the ground truth ab channels to bin indices
            ground_truth_ab = ground_truth_ab.permute(0, 2, 3, 1)  # Reorder dims to (batch_size, H, W, 2)
            ground_truth_ab_quantized = torch.zeros(ground_truth_ab.shape[0], ground_truth_ab.shape[1],
                                                    ground_truth_ab.shape[2], dtype=torch.long)

            for idx in range(ground_truth_ab.shape[0]):
                ab_channels = ground_truth_ab[idx].cpu().numpy()
                ab_quantized = quantize_ab(ab_channels)  # Quantize to nearest bin index
                ground_truth_ab_quantized[idx] = torch.from_numpy(ab_quantized)

            # Ensure ground_truth_ab_quantized is already in 256x256 resolution
            ground_truth_ab_quantized = torch.clamp(ground_truth_ab_quantized, min=0, max=312)  # Assuming 313 bins
            ground_truth_ab_quantized = ground_truth_ab_quantized.to(output_ab.device)  # Move to the same device as model output

            # Compute cross-entropy loss without downsampling
            print(ground_truth_ab_quantized.shape)
            print(output_ab.shape)
            loss = F.cross_entropy(output_ab, ground_truth_ab_quantized)
            print(loss.item())

            # Backpropagation
            loss.backward()

            # Clip gradients' norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 10 == 0:  # Print every 10 batches
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 1:.8f}')
                running_loss = 0.0

    print("Training Finished")

    # Save the final model after training
    torch.save(model.state_dict(), os.path.join(save_dir, "colorization_model_all_new.pth"))
