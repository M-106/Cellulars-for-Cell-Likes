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
    def __init__(self, num_classes, latent_dim=128, input_width=600, input_height=450, dropout=0.2, vae_using_nca=False, **kwargs):
        super().__init__()

        # --- Encoder ---
        # [3, 600, 450]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),   # [32, 300, 225]
            # nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),   # [64, 150, 112]  # cut number after comma
            # nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),   # [128, 75, 56]
            # nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # [256, 37, 28]
            # nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # [512, 18, 14]
            # nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # [512, 9, 7]
            # nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
        )


        with torch.no_grad():
            x = torch.zeros(2, 3, input_height, input_width)
            x = self.encoder(x)
            _, c, h, w = x.shape
        self.encoder_out_width = w
        self.encoder_out_height = h
        self.encoder_out_channels = c


        # --- Latent-Space Projection ---

        # -> mean value mu & variance logarithm logvar
        self.fc_mu = nn.Linear(self.encoder_out_channels * self.encoder_out_width * self.encoder_out_height, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_out_channels * self.encoder_out_width * self.encoder_out_height, latent_dim)


        # --- Decoder ---
        self.decoder_input = nn.Linear(latent_dim, self.encoder_out_channels * self.encoder_out_width * self.encoder_out_height)

        self.decoder = nn.Sequential(
            # output-padding tells the network to use the bigger option -> 3 could be 6 or 7, because downsampling 7 also resolves in 6
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, output_padding=1),
            # nn.InstanceNorm2d(512),  # InstanceNorm?
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            # nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            # nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            # nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            # nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # -> [0; 1]
        )


        # --- NCA Layer ---
        self.vae_using_nca = vae_using_nca
        if vae_using_nca:
            self.nca = NeuralCellularAutomata(
                input_channels=self.encoder_out_channels,  # latent_dim, 
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
            self.fc_mu.requires_grad_(True)
            self.fc_logvar.requires_grad_(True)
            self.decoder_input.requires_grad_(True)
            self.decoder.requires_grad_(True)
            self.class_head.requires_grad_(False)
            if self.vae_using_nca:
                self.nca.requires_grad_(True)
        elif self.train_target_state == 1:
            self.encoder.requires_grad_(False)
            self.fc_mu.requires_grad_(False)
            self.fc_logvar.requires_grad_(False)
            self.decoder_input.requires_grad_(False)
            self.decoder.requires_grad_(False)
            self.class_head.requires_grad_(True)
            if self.vae_using_nca:
                self.nca.requires_grad_(True)  # Should learn or not?
        else:
            raise ValueError(f"Unknown train-target-state: {self.train_target_state}")


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
            raise ValueError("DEBUGGING STOP, should not go here")
            latent_space = self.nca(latent_space)

        latent_space = torch.flatten(latent_space, start_dim=1)  # torch.Size([6, 4608])
        mu = self.fc_mu(latent_space)
        logvar = self.fc_logvar(latent_space)
        logvar = torch.clamp(logvar, min=-20.0, max=10.0)
        # else exploding grdients can occur

        # sample from latent space
        z = self.reparameterize(mu, logvar)

        if classify or self.train_target_state == 1:
            # classify
            class_out = self.class_head(z)

            if classify:
                # return F.softmax(class_out, dim=1)
                return class_out
        
        # decoding
        out = self.decoder_input(z)
        out = out.view(-1, self.encoder_out_channels, self.encoder_out_height, self.encoder_out_width)
        reconstructed_x = self.decoder(out)

        # interpolate if needed
        if reconstructed_x.shape[-2:] != (x.shape[-2], x.shape[-1]):
            reconstructed_x = F.interpolate(reconstructed_x, size=x.shape[-2:], mode='bilinear', align_corners=False)

        if self.train_target_state == 0:
            return reconstructed_x, mu, logvar
        else:
            return reconstructed_x, mu, logvar, class_out


    def epoch_update(self, epoch, total):
        epoch_progress = epoch/total
        if epoch_progress < 0.5:
            self.train_target_state = 0
        else:
            self.train_target_state = 1
        self.freeze_via_train_target_state()
































