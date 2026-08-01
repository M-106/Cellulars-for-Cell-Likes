# ----------
# > Import <
# ----------
import torch
import torch.nn as nn
import torch.nn.functional as F

from cfc.model.neural_cellular_automata import NeuralCellularAutomata



# ---------
# > Model <
# ---------
class ConvAE(nn.Module):
    def __init__(self, num_classes, latent_dim=128, input_width=600, input_height=450, dropout=0.2, vae_using_nca=False, **kwargs):
        super().__init__()

        # --- Encoder ---
        # [3, 224, 224]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),   # [32, 112, 112]
            nn.LeakyReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),   # [64, 56, 56]
            nn.LeakyReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # [128, 28, 28]
            nn.LeakyReLU(),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # [256, 14, 14]
            nn.LeakyReLU(),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # [512, 7, 7]
            nn.LeakyReLU(),
        )

        with torch.no_grad():
            x = torch.zeros(2, 3, input_height, input_width)
            x = self.encoder(x)
            _, c, h, w = x.shape
        self.encoder_out_width = w
        self.encoder_out_height = h
        self.encoder_out_channels = c

        # # get goal sizes
        # def calc_out_dim(dim, padding, kernel_size, stride):
        #     return ((dim + 2 * padding - kernel_size) // stride) + 1

        # goal_sizes = []
        # width = input_width
        # height = input_height
        # for layer in self.encoder:
        #     if isinstance(layer, nn.Conv2d):
        #         padding_ = layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding
        #         kernel_ = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
        #         stride_ = layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride

        #         # conv_dim_update = lambda x: int( ((x + 2 * padding_ - kernel_) // stride_) +1 )
            
        #         width = calc_out_dim(dim=width, padding=padding_, kernel_size=kernel_, stride=stride_)
        #         height = calc_out_dim(dim=height, padding=padding_, kernel_size=kernel_, stride=stride_)

        #         goal_sizes.append([width, height])

        # --- Latent-Space Projection ---
        # Deterministic bottleneck mapping for standard AE
        self.encoder_fc = nn.Linear(self.encoder_out_channels * self.encoder_out_width * self.encoder_out_height, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, self.encoder_out_channels * self.encoder_out_width * self.encoder_out_height)

        # --- Decoder ---
        self.decoder = nn.Sequential(
            # size=(14, 14),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            # nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1, output_padding=1),
            nn.LeakyReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # -> [0; 1]
        )

        # --- NCA Layer ---
        self.vae_using_nca = vae_using_nca
        if vae_using_nca:
            self.nca = NeuralCellularAutomata(
                input_channels=self.encoder_out_channels,  
                num_classes=num_classes, 
                hidden_channels=self.encoder_out_channels, 
                steps=8, 
                update_blocks=1, 
                update_blocks_activation_kernel_size=3, 
                update_blocks_activation_kernel_size_2=1,
                update_blocks_activation="relu",
                final_update_block_activation_kernel_size=1,
                final_update_block_activation="tanh",
                perception_filter="learnable",
                dropout=0.1,
                classification_mode=False
            )

        # --- Classification Head ---
        self.class_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2),

            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, num_classes)
        )

        # set weight learning state
        self.train_target_state = 0
        self.freeze_via_train_target_state()


    def freeze_via_train_target_state(self):
        if self.train_target_state == 0:
            self.encoder.requires_grad_(True)
            self.encoder_fc.requires_grad_(True)
            self.decoder_input.requires_grad_(True)
            self.decoder.requires_grad_(True)
            self.class_head.requires_grad_(False)
            if self.vae_using_nca:
                self.nca.requires_grad_(True)
        elif self.train_target_state == 1:
            self.encoder.requires_grad_(True)
            self.encoder_fc.requires_grad_(True)
            self.decoder_input.requires_grad_(True)
            self.decoder.requires_grad_(True)
            self.class_head.requires_grad_(True)
            if self.vae_using_nca:
                self.nca.requires_grad_(True)
        else:
            raise ValueError(f"Unknown train-target-state: {self.train_target_state}")


    def forward(self, x, classify=False):
        # encoding
        latent_space = self.encoder(x)  # torch.Size([6, 512, 3, 3])

        # backbone refinement NCA
        if self.vae_using_nca:
            latent_space = self.nca(latent_space)

        latent_space = torch.flatten(latent_space, start_dim=1)  # torch.Size([6, 4608])
        z = self.encoder_fc(latent_space) # Deterministic latent representation

        if classify or self.train_target_state == 1:
            # classify
            class_out = self.class_head(z)

            if classify:
                return class_out
        
        # decoding
        out = self.decoder_input(z)
        out = out.view(-1, self.encoder_out_channels, self.encoder_out_height, self.encoder_out_width)
        reconstructed_x = self.decoder(out)

        # interpolate if needed
        if reconstructed_x.shape[-2:] != (x.shape[-2], x.shape[-1]):
            reconstructed_x = F.interpolate(reconstructed_x, size=x.shape[-2:], mode='bilinear', align_corners=False)

        if self.train_target_state == 0:
            return reconstructed_x, z
        else:
            return reconstructed_x, z, class_out


    def epoch_update(self, epoch, total):
        epoch_progress = epoch / total
        if epoch_progress < 0.5:
            self.train_target_state = 0
        else:
            self.train_target_state = 1
        self.freeze_via_train_target_state()














