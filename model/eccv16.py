import torch
import torch.nn as nn
import numpy as np
from IPython import embed

# Assuming 'BaseColor' is defined somewhere else

# enable this for main
from .base_color import *

# enable this for train
#from base_color import *



# Define the ECCVGenerator class, which inherits from BaseColor
class ECCVGenerator(BaseColor):
    # The constructor of the model
    def __init__(self, norm_layer=nn.BatchNorm2d, use_finetune_layer=False, use_training=False):
        # Initialize the parent class (BaseColor)
        super(ECCVGenerator, self).__init__()

        self.use_finetune_layer = use_finetune_layer  # Add a flag for the fine-tuning layer
        self.training = use_training

        # Define the first block of layers (Model 1)
        model1 = [
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(64),
        ]

        # Define the second block of layers (Model 2)
        model2 = [
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(128),
        ]

        # Define the third block of layers (Model 3)
        model3 = [
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(256),
        ]

        # Define the fourth block of layers (Model 4)
        model4 = [
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]

        # Define the fifth block of layers (Model 5) with dilation
        model5 = [
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]

        # Define the sixth block of layers (Model 6) with dilation
        model6 = [
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]

        # Define the seventh block of layers (Model 7)
        model7 = [
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        ]

        # Define the eighth block of layers (Model 8) for upsampling
        model8 = [
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True),
        ]

        # Sequentially store each model (block of layers) as a part of the generator's architecture
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)




        # Conditional fine-tuning layer (if enabled), applied after model8
        if self.use_finetune_layer:
            self.finetune_layer = nn.Sequential(
                nn.Conv2d(313, 313, kernel_size=3, stride=1, padding=1, bias=True),
                #nn.ReLU(True), no relu for finetune layer
                nn.BatchNorm2d(313)  # normalization

            )

        # Softmax layer to predict probability distribution over color bins
        self.softmax = nn.Softmax(dim=1)

        # Final layer to map predicted color bins to the actual A and B channels
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        # Upsample the output by a factor of 4 using bilinear interpolation
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')




    # Define the forward pass for the model
    def forward(self, input_l):
        # Pass the input through each block of layers in sequence
        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)



        conv8_3 = self.model8(conv7_3)

        # Apply fine-tuning layer (if enabled)
        if self.use_finetune_layer:
            conv8_3 = self.finetune_layer(conv8_3)



        # Logits over the 313 quantized color bins
        out_logits = self.upsample4(conv8_3)  # This should have shape (batch_size, 313, 256, 256)

        if not self.training:
            # Apply softmax only during inference to convert logits to probabilities
            out_probs = self.softmax(out_logits)  # (batch_size, 313, 256, 256)
            out_ab = self.model_out(out_probs)  # Map to AB channels (batch_size, 2, 256, 256)
            output = self.unnormalize_ab(self.upsample4(out_ab))  # Upsample to final size
        else:
            # During training, use raw logits (without softmax) for cross-entropy loss
            output = out_logits


        return output


# Function to create and optionally load a pretrained ECCV model
def eccv16(pretrained=True, use_finetune_layer=False, use_training=False):
    model = ECCVGenerator(use_finetune_layer=use_finetune_layer, use_training=use_training)

    if pretrained:
        state_dict = torch.load('model/colorization_model.pth', map_location='cpu')
        if use_finetune_layer:
            print("Loading pretrained model...")
            checkpoint = torch.load('model_checkpoint.pth', map_location='cpu')
            #state_dict = torch.load('model/colorization_model_finetune.pth', map_location='cpu')

            # Lade den gespeicherten Zustand der Modellparameter
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)

        # If fine-tuning layer is used but it's not in the pre-trained state, initialize it
        if use_finetune_layer:

            # Initialisiert eine fine tune layer, die den gleichen output wie die vorherige layer liefert
            missing_keys = [key for key in model.state_dict().keys() if key.startswith('finetune_layer')]
            if missing_keys:
                print(f"Initializing missing fine-tuning layer weights: {missing_keys}")
                model.load_state_dict(state_dict, strict=False)

                # Initialize fine-tune layer weights to identity-like behavior
                with torch.no_grad():
                    # Set all weights to 0 initially
                    nn.init.zeros_(model.finetune_layer[0].weight)

                    # Setz die zentrale Position im Filter für diagonale Kanäle (Input == Output)
                    for out_channel in range(313):  # Output-Kanäle
                        for in_channel in range(313):  # Input-Kanäle
                            if out_channel == in_channel:  # Nur auf der Diagonale setzen
                                model.finetune_layer[0].weight[out_channel, in_channel, 1, 1] = 1.0

                    # Set bias to 0
                    nn.init.zeros_(model.finetune_layer[0].bias)

                    # Initialize BatchNorm
                    nn.init.ones_(model.finetune_layer[1].weight)  # Gamma = 1
                    nn.init.zeros_(model.finetune_layer[1].bias)  # Beta = 0

                print("Gewichte der Fine-Tune-Layer:")
                for i in range(313):  # Gehe durch alle 313 Kanäle
                    print(f"Filter für Output-Kanal {i}, Input-Kanal {i}:")
                    print(model.finetune_layer[0].weight[i, i, :, :])
                    if i >= 2:  # Breche nach 3 Iterationen ab
                        break

                input_tensor = torch.randn(1, 313, 64, 64)  # Dummy Input mit 313 Kanälen
                output_tensor = model.finetune_layer(input_tensor)

                # Prüfe maximale Differenz zwischen Eingabe und Ausgabe
                print("Max Differenz zwischen Eingabe und Ausgabe:",
                      torch.max(torch.abs(input_tensor - output_tensor)).item())

            else:
                #model.load_state_dict(state_dict)
                print("1")
        else:
            model.load_state_dict(state_dict)
            print("2")

    return model
