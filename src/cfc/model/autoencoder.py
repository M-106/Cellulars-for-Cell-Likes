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
class ConvVAE(nn.Module):
    def __init__(self, num_classes, latent_dim=128, input_width=600, input_height=450, vae_is_latent_training=True, dropout=0.2, vae_using_nca=False, **kwargs):
        super().__init__()

        # --- Encoder ---
        # [3, 600, 450]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),   # [32, 300, 225]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),   # [64, 150, 112]  # cut number after comma
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),   # [128, 75, 56]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # [256, 37, 28]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # [512, 18, 14]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            # nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # [512, 9, 7]
            # nn.BatchNorm2d(512),
            # nn.LeakyReLU(0.2),
        )


        # --- Latent-Space Projection ---

        # calc height & width for FCs
        def calc_out_dim(dim, padding, kernel_size, stride):
            return ((dim + 2 * padding - kernel_size) // stride) + 1
            # using // for floor operator, we want to round down, always
        
        width = input_width
        height = input_height
        last_out_channels = 3
        for layer in self.encoder:
            if isinstance(layer, nn.Conv2d):
                padding_ = layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding
                kernel_ = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
                stride_ = layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride

                # conv_dim_update = lambda x: int( ((x + 2 * padding_ - kernel_) // stride_) +1 )
            
                width = calc_out_dim(dim=width, padding=padding_, kernel_size=kernel_, stride=stride_)
                height = calc_out_dim(dim=height, padding=padding_, kernel_size=kernel_, stride=stride_)

                last_out_channels = layer.out_channels[0] if isinstance(layer.out_channels, tuple) else layer.out_channels
        self.width = width
        self.height = height
        self.last_out_channels = last_out_channels

        # -> mean value mu & variance logarithm logvar
        self.fc_mu = nn.Linear(last_out_channels * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(last_out_channels * 7 * 7, latent_dim)


        # --- Decoder ---
        self.decoder_input = nn.Linear(latent_dim, last_out_channels * width * height)

        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(512),
            # nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # -> [0; 1]
        )


        # --- NCA Layer ---
        self.vae_using_nca = vae_using_nca
        if vae_using_nca:
            self.nca = NeuralCellularAutomata(
                input_channels=512,  # latent_dim, 
                num_classes=num_classes, 
                hidden_channels=512, 
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
            nn.Linear(latent_dim, 4096),
            nn.LeakyReLU(0.2),

            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(0.2),

            nn.Linear(4096, num_classes)
        )

        # set weight learning state
        self.set_freeze_state(vae_is_latent_training)


    def set_freeze_state(self, vae_is_latent_training):
        if vae_is_latent_training:
            self.encoder.requires_grad_(True)
            self.fc_mu.requires_grad_(True)
            self.fc_logvar.requires_grad_(True)
            self.decoder_input.requires_grad_(True)
            self.decoder.requires_grad_(True)
            self.class_head.requires_grad_(False)
            if self.vae_using_nca:
                self.nca.requires_grad_(True)
        else:
            self.encoder.requires_grad_(False)
            self.fc_mu.requires_grad_(False)
            self.fc_logvar.requires_grad_(False)
            self.decoder_input.requires_grad_(False)
            self.decoder.requires_grad_(False)
            self.class_head.requires_grad_(True)
            if self.vae_using_nca:
                self.nca.requires_grad_(False)


    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std * eps

        Reference: https://www.geeksforgeeks.org/deep-learning/reparameterization-trick/
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return mu + eps * std


    def forward(self, x, classify=False):
        # encoding
        latent_space = self.encoder(x)  # torch.Size([6, 512, 3, 3])

        # backbone refinement NCA
        if self.vae_using_nca:
            latent_space = self.nca(latent_space)

        latent_space = torch.flatten(latent_space, start_dim=1)  # torch.Size([6, 4608])
        mu = self.fc_mu(latent_space)
        logvar = self.fc_logvar(latent_space)

        # sample from latent space
        z = self.reparameterize(mu, logvar)

        if classify:
            # classify
            class_out = self.class_head(z)
            # return F.softmax(class_out, dim=1)
            return class_out
        
        else:
            # decoding
            out = self.decoder_input(z)
            out = out.view(-1, self.last_out_channels, self.height, self.width)
            reconstructed_x = self.decoder(out)

            # interpolate if needed
            if reconstructed_x.shape[-2:] != (x.shape[-2], x.shape[-1]):
                reconstructed_x = F.interpolate(reconstructed_x, size=x.shape[-2:], mode='bilinear', align_corners=False)

            return reconstructed_x, mu, logvar



















