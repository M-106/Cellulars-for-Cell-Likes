# -----------
# > Imports <
# -----------
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

from cfc.model.neural_cellular_automata import NeuralCellularAutomata



# ---------
# > Model <
# ---------
class EfficientNetClassifier(nn.Module):
    def __init__(self, input_channels=3, input_width=224, input_height=224, num_classes=10, dropout=0.5, **kwargs):
        super().__init__()

        self.model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)

        in_features = self.model.classifier[1].in_features

        # out: [batch, 1000]

        model_out = sum(self.model(torch.zeros(2, input_channels, input_height, input_width)).shape[1:])

        # --- Classification ---
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),  # inplace=True
            nn.Flatten(1),
            nn.Linear(model_out, 512),
            nn.SiLU(),

            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )


    def forward(self, x, classify=True):
        x = self.model(x)
        if classify:
            x = self.classifier(x)
        return x



class EfficientNetNCAClassifier(nn.Module):
    def __init__(self, 
        input_channels=3, 
        input_width=600, 
        input_height=450, 
        num_classes=10, 
        dropout=0.5,
        hidden_channels=64, 
        steps=8, 
        update_blocks=1, 
        update_blocks_activation_kernel_size=3, 
        update_blocks_activation_kernel_size_2=1,
        update_blocks_activation="relu",
        final_update_block_activation_kernel_size=1,
        final_update_block_activation="sigmoid",
        perception_filter="sobel", 
        **kwargs
    ):
        super().__init__()

        self.model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)

        features = self.model.extract_features(torch.zeros(2, input_channels, input_height, input_width))
        model_out_channels = features.shape[1] # 1792 for EfficientNet-B4

        # --- Classification ---
        self.classifier = NeuralCellularAutomata(
            input_channels=model_out_channels, 
            num_classes=num_classes, 
            hidden_channels=hidden_channels, 
            steps=steps, 
            update_blocks=update_blocks, 
            update_blocks_activation_kernel_size=update_blocks_activation_kernel_size, 
            update_blocks_activation_kernel_size_2=update_blocks_activation_kernel_size_2,
            update_blocks_activation=update_blocks_activation,
            final_update_block_activation_kernel_size=final_update_block_activation_kernel_size,
            final_update_block_activation=final_update_block_activation,
            perception_filter=perception_filter,
            dropout=dropout,
            classification_mode=True
        )
        


    def forward(self, x, classify=True):
        x = self.model.extract_features(x) # Shape: [B, 1792, H, W]
        if classify:
            x = self.classifier(x)
        return x


















