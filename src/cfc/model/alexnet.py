# -----------
# > Imports <
# -----------
import torch
import torch.nn as nn
import torch.nn.functional as F

from cfc.model.neural_cellular_automata import NeuralCellularAutomata



# ---------
# > Model <
# ---------
class AlexNet(nn.Module):
    def __init__(self, input_channels=3, input_width=600, input_height=450, num_classes=10, dropout=0.5, only_use_backbone=False, **kwargs):
        super().__init__()

        # --- Feature Extraction ---
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # if only_use_backbone:
        #     self.features = nn.Sequential(*list(self.features.children())[:-2])

        # we do not self compute the outpute size, but just give a dummy data through the net
        with torch.no_grad():
            x = torch.zeros(1, input_channels, input_height, input_width)
            x = self.features(x)
            # _, c, h, w = x.shape
            self.feature_width = x.size(3)
            self.feature_height = x.size(2)
            self.feature_channels = x.size(1)

        # --- Classification ---
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Flatten(1),
            nn.Linear(self.feature_channels*self.feature_height*self.feature_width, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )


    def forward(self, x, classify=True):
        x = self.features(x)
        if classify:
            x = self.classifier(x)
        return x



class AlexNetNCAFeatureBoosterNet(nn.Module):
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

        self.backbone = AlexNet(
            input_channels=input_channels, 
            input_width=input_width, 
            input_height=input_height, 
            num_classes=num_classes, 
            dropout=0.5,
            only_use_backbone=True
        )
        # self.encoder = self.backbone

        backbone_out = [self.backbone.feature_channels, 
                        self.backbone.feature_height, 
                        self.backbone.feature_height]

        self.class_head = NeuralCellularAutomata(
            input_channels=backbone_out[0], 
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
        self.nca = self.class_head

        self.train_target_state = 0
        self.freeze_via_train_target_state()


    def freeze_via_train_target_state(self):
        if self.train_target_state == 0:
            self.backbone.requires_grad_(True)
            self.class_head.requires_grad_(False)
        elif self.train_target_state == 1:
            self.backbone.requires_grad_(True)
            self.class_head.requires_grad_(True)
        else:
            raise ValueError(f"Unknown train-target-state: {self.train_target_state}")


    def forward(self, x):
        x = self.backbone(x, classify=False)
        x = self.class_head(x)
        return x



    def get_last_state(self, x):
        x = self.backbone(x, classify=False)
        return self.class_head.get_last_state(x)


    def epoch_update(self, epoch, total):
        epoch_progress = epoch/total
        if epoch_progress < 0.25:
            self.train_target_state = 0
        else:
            self.train_target_state = 1
        self.freeze_via_train_target_state()


















