# -----------
# > Imports <
# -----------
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision.models import resnet50, ResNet50_Weights

# matplotlib background mode without tkinter, default is TkAgg
# Agg = Anti-Grain Geometry: A purely file-based backend
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from cfc.model.utils import init_weights



# -------------------
# > Training Helper <
# -------------------
def measure_nca_stability(model, input_img):
    """
    Function to measure how quickly the nca 
    reaches its goal state. 
    """
    model.eval()
    # prev_dropout = model.dropout
    # model.dropout = 0.0
    with torch.no_grad():
        if not isinstance(model, NeuralCellularAutomata):
            if hasattr(model, "encoder"):
                x = model.encoder(input_img)
            else:
                input_img = model.backbone(input_img, classify=False)
            model = model.nca

        z, gamma, beta = model._preprocess_for_steps(x=input_img)
            
        x = model.input_projection_net(input_img)

        total_change = 0.0
        for t in range(model.steps):
            x_prev = x
            gamma, beta = model._get_time_film_params(timestep=t, latent_z=z, gamma=gamma, beta=beta, device=x.device)
            x = model.step(x, gamma=gamma, beta=beta, raw_img=input_img, z=z)

            change = torch.norm(x - x_prev, p=2) / x.numel()
            total_change += change.item()

    # model.dropout = prev_dropout

    return total_change / model.steps



# ----------
# > Helper <
# ----------
def get_activation(activation_name):
    if activation_name.lower() == "relu":
        return nn.ReLU()
    elif activation_name.lower() == "leaky_relu":
        return nn.LeakyReLU()
    elif activation_name.lower() == "tanh":
        return nn.Tanh()
    elif activation_name.lower() == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unknown activation '{activation_name}'")



def handle_optional_list_param(value, dtype, goal_len, apply_func=lambda x:x):
    if isinstance(value, dtype):
        final_value = [apply_func(value)]*goal_len 
    else:
        final_value = []
        for cur_value in value:
            final_value.append(apply_func(cur_value))

        if len(final_value) != goal_len:
            raise ValueError(f"Parameter List has {len(final_value)} elements but {goal_len} are needed.")
    
    return final_value




# --------------------
# > Global Attention <
# --------------------
class GlobalCrossAttention(nn.Module):
    def __init__(self, channels, latent_dim):
        super().__init__()
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Linear(latent_dim, channels)
        self.v = nn.Linear(latent_dim, channels)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        
        # Zero-initialize scale parameter so the block starts as identity (x + 0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, z):
        # x: (B, C, H, W), z: (B, latent_dim)
        B, C, H, W = x.shape
        
        # Project queries, keys, and values
        q = self.q(x).flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        k = self.k(z).unsqueeze(1)                # (B, 1, C)
        v = self.v(z).unsqueeze(1)                # (B, 1, C)

        # Dot-product attention scores
        # (B, H*W, C) x (B, C, 1) -> (B, H*W, 1)
        scores = torch.bmm(q, k.transpose(1, 2)) / (C ** 0.5)
        attn = F.softmax(scores, dim=-1)           # Softmax across key dimension

        # Aggregate values: (B, H*W, 1) x (B, 1, C) -> (B, H*W, C)
        out = torch.bmm(attn, v)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        out = self.out_proj(out)

        # Residual connection with zero-gated scale
        return x + self.gamma * out



# class ImageCrossAttention(nn.Module):
#     def __init__(self, channels, img_channels=3):
#         super().__init__()
#         self.q = nn.Conv2d(channels, channels, 1)
#         # project rgb image on channel-size of of NCA-state
#         self.k_proj = nn.Conv2d(img_channels, channels, 1)
#         self.v_proj = nn.Conv2d(img_channels, channels, 1)
        
#         self.out_proj = nn.Conv2d(channels, channels, 1)
#         self.gamma = nn.Parameter(torch.zeros(1))

#     def forward(self, x, raw_img):
#         # x: (B, C, H, W) -> NCA State
#         # raw_img: (B, 3, H, W) -> Original Image
#         B, C, H, W = x.shape
        
#         q = self.q(x).flatten(2).permute(0, 2, 1)              # (B, H*W, C)
#         k = self.k_proj(raw_img).flatten(2).permute(0, 2, 1)   # (B, H*W, C)
#         v = self.v_proj(raw_img).flatten(2).permute(0, 2, 1)   # (B, H*W, C)

#         # Full Spatial Attention: Every NCA-Cell looks at the same whole image
#         scores = torch.bmm(q, k.transpose(1, 2)) / (C ** 0.5)  # (B, H*W, H*W)
#         attn = F.softmax(scores, dim=-1)

#         out = torch.bmm(attn, v).permute(0, 2, 1).view(B, C, H, W)

#         return x + self.gamma * self.out_proj(out)

class ImageCrossAttention(nn.Module):
    def __init__(self, channels, img_channels=3, downsample_size=(56, 56)):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(downsample_size)
        
        self.q = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(img_channels, channels, 1)
        self.v_proj = nn.Conv2d(img_channels, channels, 1)
        
        self.out_proj = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, raw_img):
        B, C, H, W = x.shape
        
        img_small = self.pool(raw_img)
        
        # PyTorch SDPA expects Shape: (B, num_heads, sequence_length, head_dim)
        # We only use 1 head
        q = self.q(x).flatten(2).permute(0, 2, 1).unsqueeze(1)               # (B, 1, H*W, C)
        k = self.k_proj(img_small).flatten(2).permute(0, 2, 1).unsqueeze(1)  # (B, 1, K_len, C)
        v = self.v_proj(img_small).flatten(2).permute(0, 2, 1).unsqueeze(1)  # (B, 1, K_len, C)

        # PyTorch Fused SDPA (Extreme fast & memory efficient)
        out = F.scaled_dot_product_attention(q, k, v)                        # (B, 1, H*W, C)

        # Reshape back to (B, C, H, W)
        out = out.squeeze(1).permute(0, 2, 1).view(B, C, H, W)

        return x + self.gamma * self.out_proj(out)

    

# -------------
# > NCA Model <
# -------------
class StepAwareFiLM(nn.Module):
    def __init__(self, latent_dim, hidden_channels, max_steps=32):
        super().__init__()
        self.step_embed = nn.Embedding(max_steps, 64)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + 64, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 2 * hidden_channels)
        )

    def forward(self, z, step_idx):
        t_emb = self.step_embed(step_idx)
        t_emb = t_emb.unsqueeze(0).repeat(z.shape[0], 1)
        combined = torch.cat([z, t_emb], dim=1)
        params = self.mlp(combined)
        gamma, beta = torch.chunk(params, 2, dim=1)
        return gamma + 1.0, beta



class Perception(nn.Module):
    """
    Without Perception the update-net have to learn
    through only the training of the weights how to compute with neighbors.
    -> Difficult and unstable

    The Perception adds an channel, with the physical view of the cell 
    about the environment. -> The update-net just have to learn the update rules. 
    Not the perception by itself.

    Can be used as Sobel or as Laplacian.
    -> Explain...
    """
    def __init__(self, channels, filter="sobel"):
        super().__init__()

        self.filter = filter

        if self.filter.lower() == "sobel":
            self.register_buffer(
                "kernel", torch.tensor([
                    [-1.0, 0.0, 1.0],
                    [-2.0, 0.0, 2.0],
                    [-1.0, 0.0, 1.0],
                ]).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
            ) 
            self.size = 3
        elif self.filter.lower() == "laplacian":
            # self.register_buffer(
            # self.kernel = torch.tensor([
            self.register_buffer(
                "kernel", torch.Tensor([
                    [-1.0, -1.0, -1.0],
                    [-1.0,  8.0, -1.0],
                    [-1.0, -1.0, -1.0]
                ]).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
            )
            self.size = 2
        elif self.filter.lower() == "learnable":
            self.conv1 = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding="same")
            self.size = 1
        elif self.filter.lower() == "pretrained":
            resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
            pretrained_layer = resnet.conv1
            
            self.features = torch.nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=7,
                stride=1,
                padding="same",
                bias=False
            )

            with torch.no_grad():
                weight_repeat = (channels//3) + 1
                # 6
                expanded_weight = pretrained_layer.weight.repeat(1, weight_repeat, 1, 1)[:, :channels, :, :]
                # torch.Size([64, 16, 7, 7])
                # ResNet has 64 Out-Channels -> repeat until fitting to our needed size
                if channels <= 64:
                    self.features.weight.copy_(expanded_weight[:channels])
                else:
                    out_repeat = (channels // 64) + 1
                    self.features.weight.copy_(expanded_weight.repeat(out_repeat, 1, 1, 1)[:channels])
            
            # # freeze pretrained weights?
            # for param in self.features.parameters():
            #     param.requires_grad = False

            self.size = 1
        else:
            raise ValueError(f"Unknown Filter passed: '{self.filter}'.") 


    def forward(self, x):
        
        if self.filter.lower() == "sobel":
            dx = F.conv2d(x, self.kernel, padding=1, groups=x.shape[1])
            dy = F.conv2d(x, self.kernel.transpose(2, 3), padding=1, groups=x.shape[1])
            return torch.cat([x, dx, dy], dim=1)
        elif self.filter.lower() == "laplacian":
            laplacian = F.conv2d(x, self.kernel.to(x.device), padding=1, groups=x.shape[1])
            return torch.cat([x, laplacian], dim=1)
        elif self.filter.lower() == "learnable":
            return self.conv1(x)
        elif self.filter.lower() == "pretrained":
            return self.features(x)
        else:
            raise ValueError(f"Unknown Filter passed: '{self.filter}'.") 



class NCAUpdateBlock(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, kernel_size=3, kernel_size_2=1, activation=nn.ReLU(), is_final_block=False):
        super().__init__()
        self.is_final_block = is_final_block

        # padding = "same" should have the same effect but also works with different kernel sizes
        # if self.is_final_block:
        #     padding1 = 0
        # else:
        #     padding1 = 1

        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=kernel_size, padding="same")
        self.activation = activation
        if not self.is_final_block:
            self.conv2 = nn.Conv2d(hidden_channels, output_channels, kernel_size=kernel_size_2, padding="same")

        # formular: output_size = input_size + 2*padding - (kernel_size - 1)
        # we use padding 1, so that the whole image/grid is processed 
        # and no cell is skipped, else, the border pixels would be skipped


    def forward(self, x, gamma=None, beta=None):

        x = self.conv1(x)

        if gamma is not None and beta is not None:
            # Reshape for Broadcasting over Spatial Dimensions: (B, C) -> (B, C, 1, 1)
            g = gamma.unsqueeze(-1).unsqueeze(-1)
            b = beta.unsqueeze(-1).unsqueeze(-1)
            x = g * x + b

        x = self.activation(x)

        if not self.is_final_block:
            x = self.conv2(x)
        return x
    


class NeuralCellularAutomata(torch.nn.Module):
    """
    NCA model for image classification. 
    The model consists of an input projection layer, a series of update blocks, and a classification head. 
    The update blocks are responsible for iteratively updating the hidden state of the model based on learned rules. 
    The final update block uses a Sigmoid activation to ensure that the updates are in a stable range.

    Optionally with a FiLM adjsutment, where every Step gets multiplied and added a value (2 values) 
    computed by a MLP using the latent representation from a Autoencoder 
    to add global information (bottleneck of current approach).
    FiLM = Feature-wise Linear Modulations

    > Note: Weights must be low initialized to ensure stability of the NCA. 
    > The final update block uses a Sigmoid activation to ensure that the updates are in a stable range. 
    """
    def __init__(self, input_channels, num_classes, 
                 hidden_channels=64, 
                 steps=8, 
                 update_blocks=1, 
                 update_blocks_activation_kernel_size=3, 
                 update_blocks_activation_kernel_size_2=1,
                 update_blocks_activation="relu",
                 final_update_block_activation_kernel_size=1,
                 final_update_block_activation="sigmoid",
                 perception_filter="sobel",
                 dropout=0.1,
                 classification_mode=True,
                 latent_model=None,
                 input_width=224,
                 input_height=224,
                 latent_film_time_activated=False,
                 use_film=False,
                 use_global_attn_context=False,
                 use_img_global_attn=True,
                 **kwargs):
        super().__init__()

        if update_blocks < 1:
            raise ValueError(f"At least 1 update block is needed, but {update_blocks} are wanted.")
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.steps = steps
        self.dropout = dropout
        self.classification_mode = classification_mode
        self.latent_model = latent_model
        if self.latent_model is not None:
            self.latent_model.requires_grad_(False)
        self.use_film = use_film
        self.film_time_based = latent_film_time_activated
        self.use_global_attn_context = use_global_attn_context
        self.use_img_global_attn = use_img_global_attn

        if self.use_global_attn_context and not self.use_img_global_attn and self.latent_model is None:
            raise ValueError("If using global context with latent context, you must also providea latent model.")

        if self.use_film and self.latent_model is None:
            raise ValueError("If using FiLM (Feature-wise Linear Modulations) you have to provide a latent model!")

        # Perception -> already give every cell the information of their enviornment, via additional channels
        self.perception = Perception(hidden_channels, filter=perception_filter)
        input_size = hidden_channels * self.perception.size
        # update_block_input_channels = [input_size] + [hidden_channels]*(update_blocks-1)

        # Input-Projection -> 3 Channel image to hidden state
        self.input_projection_net = nn.Conv2d(input_channels, hidden_channels, 1)

        # State-Update-Network -> core of NCA -> learning rules for cell updates
        kernel_sizes = handle_optional_list_param(update_blocks_activation_kernel_size, int, update_blocks)
        kernel_sizes_2 = handle_optional_list_param(update_blocks_activation_kernel_size_2, int, update_blocks)
        activations = handle_optional_list_param(update_blocks_activation, str, update_blocks, get_activation)
        self.first_update_block = NCAUpdateBlock(
            input_channels=input_size, 
            hidden_channels=hidden_channels,
            output_channels=hidden_channels, 
            kernel_size=kernel_sizes[0], 
            kernel_size_2=kernel_sizes_2[0], 
            activation=activations[0], 
            is_final_block=False
        )
        update_blocks = [
            NCAUpdateBlock(
                input_channels=hidden_channels, 
                hidden_channels=hidden_channels,
                output_channels=hidden_channels, 
                kernel_size=k, 
                kernel_size_2=k2, 
                activation=a, 
                is_final_block=False
            ) 
            for _, k, k2, a in zip(range(update_blocks-1), kernel_sizes[1:], kernel_sizes_2[1:], activations[1:])
        ]
        
        self.final_update_block = NCAUpdateBlock(
            input_channels=hidden_channels, 
            hidden_channels=hidden_channels,
            output_channels=hidden_channels,
            kernel_size=final_update_block_activation_kernel_size, 
            activation=get_activation(final_update_block_activation), 
            is_final_block=True
        )
        
        # update_blocks.append(final_update_block)  
        # self.update_net = nn.Sequential(first_update_block, *update_blocks, final_update_block)
        self.middle_blocks = nn.ModuleList(update_blocks)
        # FiLM Generator Setup
        if self.latent_model is not None:
            device = next(latent_model.parameters()).device
            with torch.no_grad():
                dummy_in = torch.zeros(1, input_channels, input_height, input_width, device=device)
                dummy_out = self.latent_model(dummy_in, classify=False)[1] # if hasattr(self.latent_model, "encoder") else self.latent_model(dummy_in)
                latent_dim = dummy_out.view(1, -1).shape[1]

            # Creates gamma and beta (2 * hidden_channels)
            if self.use_film:
                if self.film_time_based:
                    self.film_generator = StepAwareFiLM(
                        latent_dim=latent_dim, 
                        hidden_channels=hidden_channels, 
                        max_steps=steps
                    )
                else:
                    self.film_generator = nn.Sequential(
                        nn.Linear(latent_dim, 256),
                        nn.LeakyReLU(0.2),
                        nn.Linear(256, 2 * hidden_channels)
                    )

        # Global Context Attention Net
        if self.use_global_attn_context:
            if self.use_img_global_attn:
                self.global_attn = ImageCrossAttention(hidden_channels, img_channels=input_channels)
            else:
                self.global_attn = GlobalCrossAttention(hidden_channels, latent_dim)

        # Classification Head -> hidden state to class prediction
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # global average pooling -> (B, C, H, W) -> (B, C, 1, 1)
            nn.Flatten(),  # flatten to (B, C)
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(hidden_channels, num_classes)
        )

        self.apply(init_weights)  # initialize weights of the model near 0

    def _preprocess_for_steps(self, x):
        if self.use_film:
            if self.film_time_based:  
                # call latent_model
                z = self.latent_model(x, classify=False)[1]
                gamma = beta = None
            else:
                gamma, beta = self._get_film_params(x)
                z = None
        else:
            z = None
            gamma = None
            beta = None

        if self.use_global_attn_context and not self.use_img_global_attn:
            z = self.latent_model(x, classify=False)[1]

        return z, gamma, beta

    def _get_film_params(self, raw_img):
        """
        Extracts the Latent-Vector and creates gamma/beta.
        """
        if self.latent_model is None or self.use_film == False:
            return None, None
            
        # call latent_model
        z = self.latent_model(raw_img, classify=False)[1]

        # creates gamma and beta
        film_params = self.film_generator(z)
        
        # Split in gamma and beta
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        
        # Gamma around 1.0 centrate for stabile initialization (Identity Operation at beginning)
        gamma = gamma + 1.0

        return gamma, beta

    def _get_time_film_params(self, timestep, latent_z, gamma, beta, device):
        if self.film_time_based and self.use_film:
            step_tensor = torch.tensor(timestep, device=device)
            gamma, beta = self.film_generator(latent_z, step_tensor)

        return gamma, beta

    def step(self, x, gamma=None, beta=None, raw_img=None, z=None):
        perception = self.perception(x)
        # update = self.update_net(perception)

        h = self.first_update_block(perception, gamma=gamma, beta=beta)
        for block in self.middle_blocks:
            h = block(h, gamma=gamma, beta=beta)

        if self.use_global_attn_context:
            if self.use_img_global_attn:
                h = self.global_attn(h, raw_img)
            else:
                h = self.global_attn(h, z)

        update = self.final_update_block(h, gamma=gamma, beta=beta)

        # return x + update

        # if update.shape[2:] != x.shape[2:]:
        #     update = F.interpolate(update, size=x.shape[2:], mode='bilinear', align_corners=False)
    
        # stochastic mask -> vor better generalization ability
        #    -> update dropout
        # if masking:
        #     mask = torch.rand(x.shape[0], 1, x.shape[2], x.shape[3]) > self.dropout
        #     return x + update * mask.to(x.device)
        # else:
        return x + update
    

    def get_last_state(self, x):
        """
        Like forward but returns the last hidden state instead of the logits.
        """

        z, gamma, beta = self._preprocess_for_steps(x=x)

        # project input image to hidden state
        h = self.input_projection_net(x)

        # iterative updates of the hidden state
        for t in range(self.steps):
            gamma, beta = self._get_time_film_params(timestep=t, latent_z=z, gamma=gamma, beta=beta, device=x.device)
            h = self.step(h, gamma=gamma, beta=beta, raw_img=x, z=z)

        return h  # return the 4D grid state: [B, C, H, W]
    

    def forward(self, x):
        # global modulation (if latent model exist) 
        z, gamma, beta = self._preprocess_for_steps(x=x)

        # project input image to hidden state
        # if self.classification_mode:
        h = self.input_projection_net(x)
        # else:
        #     x = x.view(-1, self.hidden_channels, 1, 1)

        # iterative updates of the hidden state
        for t in range(self.steps):
            gamma, beta = self._get_time_film_params(timestep=t, latent_z=z, gamma=gamma, beta=beta, device=x.device)
            h = self.step(h, gamma=gamma, beta=beta, raw_img=x, z=z)

        if self.classification_mode:
            # classify the final hidden state
            logits = self.classification_head(h)
            return logits
        else:
           return h


    def save_transition_sequence(self, x, save_path):
        """
        Saves the state of the NCA at each step as an image file.
        """
        x = x[0:1]

        z, gamma, beta = self._preprocess_for_steps(x=x)

        current_x = self.input_projection_net(x)

        history = []
        history.append(current_x.detach().cpu())

        for t in range(self.steps):
            gamma, beta = self._get_time_film_params(timestep=t, latent_z=z, gamma=gamma, beta=beta, device=x.device)
            current_x = self.step(current_x, gamma=gamma, beta=beta, raw_img=x, z=z)
            history.append(current_x.detach().cpu())

        # apply PCA
        history_tensor = torch.cat(history, dim=0)
        T, C, H, W = history_tensor.shape

        data = history_tensor.permute(0, 2, 3, 1).reshape(-1, C).numpy()

        pca = PCA(n_components=3)
        pca_data = pca.fit_transform(data)

        pca_images = pca_data.reshape(T, H, W, 3).transpose(0, 3, 1, 2)
        pca_images = torch.from_numpy(pca_images)
        pca_images = (pca_images - pca_images.min()) / (pca_images.max() - pca_images.min())

        num_images = len(pca_images)
        cols = int(math.ceil(math.sqrt(num_images)))
        rows = int(math.ceil(num_images / cols))

        # make a grid from the images
        grid_img = vutils.make_grid(pca_images, nrow=cols, padding=2, normalize=False)

        vutils.save_image(grid_img, save_path)

        
        # plt.figure(figsize=(5*cols, 5*rows))
        # plt.imshow(grid_img.permute(1, 2, 0))
        # plt.axis("off")
        # plt.title(f"NCA Transition Sequence (Steps: {self.steps})")
        # plt.tight_layout()
        # plt.savefig(save_path, dpi=300)
        # plt.close()





# XXXXXXXXXXXXXXXX
# ----------------
# ################
#     AlexNet
# ################
# ----------------
# XXXXXXXXXXXXXXXX

# # -----------
# # > Imports <
# # -----------
# import os
# import math

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.utils as vutils
# from torchvision.models import resnet50, ResNet50_Weights

# # matplotlib background mode without tkinter, default is TkAgg
# # Agg = Anti-Grain Geometry: A purely file-based backend
# # import matplotlib
# # matplotlib.use('Agg')
# # import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

# from cfc.model.utils import init_weights



# # -------------------
# # > Training Helper <
# # -------------------
# def measure_nca_stability(model, input_img):
#     """
#     Function to measure how quickly the nca 
#     reaches its goal state. 
#     """
#     model.eval()
#     # prev_dropout = model.dropout
#     # model.dropout = 0.0
#     with torch.no_grad():
#         if not isinstance(model, NeuralCellularAutomata):
#             if hasattr(model, "encoder"):
#                 input_img = model.encoder(input_img)
#             else:
#                 input_img = model.backbone(input_img, classify=False)
#             model = model.nca
#         x = model.input_projection_net(input_img)

#         total_change = 0.0
#         for _ in range(model.steps):
#             x_prev = x
#             x = model.step(x)

#             change = torch.norm(x - x_prev, p=2) / x.numel()
#             total_change += change.item()

#     # model.dropout = prev_dropout

#     return total_change / model.steps



# # ----------
# # > Helper <
# # ----------
# def get_activation(activation_name):
#     if activation_name.lower() == "relu":
#         return nn.ReLU()
#     elif activation_name.lower() == "leaky_relu":
#         return nn.LeakyReLU()
#     elif activation_name.lower() == "tanh":
#         return nn.Tanh()
#     elif activation_name.lower() == "sigmoid":
#         return nn.Sigmoid()
#     else:
#         raise ValueError(f"Unknown activation '{activation_name}'")



# def handle_optional_list_param(value, dtype, goal_len, apply_func=lambda x:x):
#     if isinstance(value, dtype):
#         final_value = [apply_func(value)]*goal_len 
#     else:
#         final_value = []
#         for cur_value in value:
#             final_value.append(apply_func(cur_value))

#         if len(final_value) != goal_len:
#             raise ValueError(f"Parameter List has {len(final_value)} elements but {goal_len} are needed.")
    
#     return final_value


# # -------------
# # > NCA Model <
# # -------------
# class Perception(nn.Module):
#     """
#     Without Perception the update-net have to learn
#     through only the training of the weights how to compute with neighbors.
#     -> Difficult and unstable

#     The Perception adds an channel, with the physical view of the cell 
#     about the environment. -> The update-net just have to learn the update rules. 
#     Not the perception by itself.

#     Can be used as Sobel or as Laplacian.
#     -> Explain...
#     """
#     def __init__(self, channels, filter="sobel"):
#         super().__init__()

#         self.filter = filter

#         if self.filter.lower() == "sobel":
#             self.register_buffer(
#                 "kernel", torch.tensor([
#                     [-1.0, 0.0, 1.0],
#                     [-2.0, 0.0, 2.0],
#                     [-1.0, 0.0, 1.0],
#                 ]).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
#             ) 
#             self.size = 3
#         elif self.filter.lower() == "laplacian":
#             # self.register_buffer(
#             # self.kernel = torch.tensor([
#             self.register_buffer(
#                 "kernel", torch.Tensor([
#                     [-1.0, -1.0, -1.0],
#                     [-1.0,  8.0, -1.0],
#                     [-1.0, -1.0, -1.0]
#                 ]).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
#             )
#             self.size = 2
#         elif self.filter.lower() == "learnable":
#             self.conv1 = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding="same")
#             self.size = 1
#         elif self.filter.lower() == "pretrained":
#             resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
#             pretrained_layer = resnet.conv1
            
#             self.features = torch.nn.Conv2d(
#                 in_channels=channels,
#                 out_channels=channels,
#                 kernel_size=7,
#                 stride=1,
#                 padding="same",
#                 bias=False
#             )

#             with torch.no_grad():
#                 weight_repeat = (channels//3) + 1
#                 # 6
#                 expanded_weight = pretrained_layer.weight.repeat(1, weight_repeat, 1, 1)[:, :channels, :, :]
#                 # torch.Size([64, 16, 7, 7])
#                 # ResNet has 64 Out-Channels -> repeat until fitting to our needed size
#                 if channels <= 64:
#                     self.features.weight.copy_(expanded_weight[:channels])
#                 else:
#                     out_repeat = (channels // 64) + 1
#                     self.features.weight.copy_(expanded_weight.repeat(out_repeat, 1, 1, 1)[:channels])
            
#             # # freeze pretrained weights?
#             # for param in self.features.parameters():
#             #     param.requires_grad = False

#             self.size = 1
#         else:
#             raise ValueError(f"Unknown Filter passed: '{self.filter}'.") 


#     def forward(self, x):
        
#         if self.filter.lower() == "sobel":
#             dx = F.conv2d(x, self.kernel, padding=1, groups=x.shape[1])
#             dy = F.conv2d(x, self.kernel.transpose(2, 3), padding=1, groups=x.shape[1])
#             return torch.cat([x, dx, dy], dim=1)
#         elif self.filter.lower() == "laplacian":
#             laplacian = F.conv2d(x, self.kernel.to(x.device), padding=1, groups=x.shape[1])
#             return torch.cat([x, laplacian], dim=1)
#         elif self.filter.lower() == "learnable":
#             return self.conv1(x)
#         elif self.filter.lower() == "pretrained":
#             return self.features(x)
#         else:
#             raise ValueError(f"Unknown Filter passed: '{self.filter}'.") 



# class NCAUpdateBlock(nn.Module):
#     def __init__(self, input_channels, hidden_channels, output_channels, kernel_size=3, kernel_size_2=1, activation=nn.ReLU(), is_final_block=False):
#         super().__init__()
#         self.is_final_block = is_final_block

#         if self.is_final_block:
#             padding1 = 0
#         else:
#             padding1 = 1

#         self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=kernel_size, padding=padding1)
#         self.activation = activation
#         if not self.is_final_block:
#             self.conv2 = nn.Conv2d(hidden_channels, output_channels, kernel_size=kernel_size_2, padding=0)

#         # formular: output_size = input_size + 2*padding - (kernel_size - 1)
#         # we use padding 1, so that the whole image/grid is processed 
#         # and no cell is skipped, else, the border pixels would be skipped


#     def forward(self, x):
#         if self.is_final_block:
#             x = self.conv1(x)
#             x = self.activation(x)
#         else:
#             x = self.conv1(x)
#             x = self.activation(x)
#             x = self.conv2(x)
#         return x
    


# class NeuralCellularAutomata(torch.nn.Module):
#     """
#     NCA model for image classification. 
#     The model consists of an input projection layer, a series of update blocks, and a classification head. 
#     The update blocks are responsible for iteratively updating the hidden state of the model based on learned rules. 
#     The final update block uses a Sigmoid activation to ensure that the updates are in a stable range.

#     > Note: Weights must be low initialized to ensure stability of the NCA. 
#     > The final update block uses a Sigmoid activation to ensure that the updates are in a stable range. 
#     """
#     def __init__(self, input_channels, num_classes, 
#                  hidden_channels=64, 
#                  steps=8, 
#                  update_blocks=1, 
#                  update_blocks_activation_kernel_size=3, 
#                  update_blocks_activation_kernel_size_2=1,
#                  update_blocks_activation="relu",
#                  final_update_block_activation_kernel_size=1,
#                  final_update_block_activation="sigmoid",
#                  perception_filter="sobel",
#                  dropout=0.1,
#                  classification_mode=True,
#                  **kwargs):
#         super().__init__()

#         if update_blocks < 1:
#             raise ValueError(f"At least 1 update block is needed, but {update_blocks} are wanted.")
        
#         self.input_channels = input_channels
#         self.hidden_channels = hidden_channels
#         self.num_classes = num_classes
#         self.steps = steps
#         self.dropout = dropout
#         self.classification_mode = classification_mode

#         # Perception -> already give every cell the information of their enviornment, via additional channels
#         self.perception = Perception(hidden_channels, filter=perception_filter)
#         input_size = hidden_channels * self.perception.size
#         # update_block_input_channels = [input_size] + [hidden_channels]*(update_blocks-1)

#         # Input-Projection -> 3 Channel image to hidden state
#         self.input_projection_net = nn.Conv2d(input_channels, hidden_channels, 1)

#         # State-Update-Network -> core of NCA -> learning rules for cell updates
#         kernel_sizes = handle_optional_list_param(update_blocks_activation_kernel_size, int, update_blocks)
#         kernel_sizes_2 = handle_optional_list_param(update_blocks_activation_kernel_size_2, int, update_blocks)
#         activations = handle_optional_list_param(update_blocks_activation, str, update_blocks, get_activation)
#         first_update_block = NCAUpdateBlock(
#             input_channels=input_size, 
#             hidden_channels=hidden_channels,
#             output_channels=hidden_channels, 
#             kernel_size=kernel_sizes[0], 
#             kernel_size_2=kernel_sizes_2[0], 
#             activation=activations[0], 
#             is_final_block=False
#         )
#         update_blocks = [
#             NCAUpdateBlock(
#                 input_channels=hidden_channels, 
#                 hidden_channels=hidden_channels,
#                 output_channels=hidden_channels, 
#                 kernel_size=k, 
#                 kernel_size_2=k2, 
#                 activation=a, 
#                 is_final_block=False
#             ) 
#             for _, k, k2, a in zip(range(update_blocks-1), kernel_sizes[1:], kernel_sizes_2[1:], activations[1:])
#         ]
        
#         final_update_block = NCAUpdateBlock(
#             input_channels=hidden_channels, 
#             hidden_channels=hidden_channels,
#             output_channels=hidden_channels,
#             kernel_size=final_update_block_activation_kernel_size, 
#             activation=get_activation(final_update_block_activation), 
#             is_final_block=True
#         )
        
#         # update_blocks.append(final_update_block)  
#         self.update_net = nn.Sequential(first_update_block, *update_blocks, final_update_block)

#         # Classification Head -> hidden state to class prediction
#         self.classification_head = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),  # global average pooling -> (B, C, H, W) -> (B, C, 1, 1)
#             nn.Flatten(),  # flatten to (B, C)
#             nn.Dropout(p=dropout, inplace=False),
#             nn.Linear(hidden_channels, num_classes)
#         )

#         self.apply(init_weights)  # initialize weights of the model near 0


#     def step(self, x):
#         perception = self.perception(x)
#         update = self.update_net(perception)
#         # return x + update

#         # if update.shape[2:] != x.shape[2:]:
#         #     update = F.interpolate(update, size=x.shape[2:], mode='bilinear', align_corners=False)
    
#         # stochastic mask -> vor better generalization ability
#         #    -> update dropout
#         # if masking:
#         #     mask = torch.rand(x.shape[0], 1, x.shape[2], x.shape[3]) > self.dropout
#         #     return x + update * mask.to(x.device)
#         # else:
#         return x + update
    

#     def get_last_state(self, x):
#         """
#         Like forward but returns the last hidden state instead of the logits.
#         """
#         # project input image to hidden state
#         x = self.input_projection_net(x)

#         # iterative updates of the hidden state
#         for _ in range(self.steps):
#             x = self.step(x)

#         return x  # return the 4D grid state: [B, C, H, W]
    

#     def forward(self, x):
#         # project input image to hidden state
#         # if self.classification_mode:
#         x = self.input_projection_net(x)
#         # else:
#         #     x = x.view(-1, self.hidden_channels, 1, 1)

#         # iterative updates of the hidden state
#         for _ in range(self.steps):
#             x = self.step(x)

#         if self.classification_mode:
#             # classify the final hidden state
#             logits = self.classification_head(x)
#             return logits
#         else:
#            return x


#     def save_transition_sequence(self, x, save_path):
#         """
#         Saves the state of the NCA at each step as an image file.
#         """
#         x = x[0:1]

#         current_x = self.input_projection_net(x)

#         history = []
#         history.append(current_x.detach().cpu())

#         for _ in range(self.steps):
#             current_x = self.step(current_x)
#             history.append(current_x.detach().cpu())

#         # apply PCA
#         history_tensor = torch.cat(history, dim=0)
#         T, C, H, W = history_tensor.shape

#         data = history_tensor.permute(0, 2, 3, 1).reshape(-1, C).numpy()

#         pca = PCA(n_components=3)
#         pca_data = pca.fit_transform(data)

#         pca_images = pca_data.reshape(T, H, W, 3).transpose(0, 3, 1, 2)
#         pca_images = torch.from_numpy(pca_images)
#         pca_images = (pca_images - pca_images.min()) / (pca_images.max() - pca_images.min())

#         num_images = len(pca_images)
#         cols = int(math.ceil(math.sqrt(num_images)))
#         rows = int(math.ceil(num_images / cols))

#         # make a grid from the images
#         grid_img = vutils.make_grid(pca_images, nrow=cols, padding=2, normalize=False)

#         vutils.save_image(grid_img, save_path)

        
#         # plt.figure(figsize=(5*cols, 5*rows))
#         # plt.imshow(grid_img.permute(1, 2, 0))
#         # plt.axis("off")
#         # plt.title(f"NCA Transition Sequence (Steps: {self.steps})")
#         # plt.tight_layout()
#         # plt.savefig(save_path, dpi=300)
#         # plt.close()



# XXXXXXXXXXXXXXXX
# ----------------
# ################
#     Origin
# ################
# ----------------
# XXXXXXXXXXXXXXXX

# # -----------
# # > Imports <
# # -----------
# import os
# import math

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.utils as vutils

# # matplotlib background mode without tkinter, default is TkAgg
# # Agg = Anti-Grain Geometry: A purely file-based backend
# # import matplotlib
# # matplotlib.use('Agg')
# # import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

# from cfc.model.utils import init_weights



# # -------------------
# # > Training Helper <
# # -------------------
# def measure_nca_stability(model, input_img):
#     """
#     Function to measure how quickly the nca 
#     reaches its goal state. 
#     """
#     model.eval()
#     prev_dropout = model.dropout
#     model.dropout = 0.0
#     with torch.no_grad():
#         x = model.input_projection_net(input_img)

#         total_change = 0.0
#         for _ in range(model.steps):
#             x_prev = x
#             x = model.step(x)

#             change = torch.norm(x - x_prev, p=2) / x.numel()
#             total_change += change.item()

#     model.dropout = prev_dropout

#     return total_change / model.steps



# # ----------
# # > Helper <
# # ----------
# def get_activation(activation_name):
#     if activation_name.lower() == "relu":
#         return nn.ReLU()
#     elif activation_name.lower() == "leaky_relu":
#         return nn.LeakyReLU()
#     elif activation_name.lower() == "tanh":
#         return nn.Tanh()
#     elif activation_name.lower() == "sigmoid":
#         return nn.Sigmoid()
#     else:
#         raise ValueError(f"Unknown activation '{activation_name}'")



# def handle_optional_list_param(value, dtype, goal_len, apply_func=lambda x:x):
#     if isinstance(value, dtype):
#         final_value = [apply_func(value)]*goal_len 
#     else:
#         final_value = []
#         for cur_value in value:
#             final_value.append(apply_func(cur_value))

#         if len(final_value) != goal_len:
#             raise ValueError(f"Parameter List has {len(final_value)} elements but {goal_len} are needed.")
    
#     return final_value


# # -------------
# # > NCA Model <
# # -------------
# class Perception(nn.Module):
#     """
#     Without Perception the update-net have to learn
#     through only the training of the weights how to compute with neighbors.
#     -> Difficult and unstable

#     The Perception adds an channel, with the physical view of the cell 
#     about the environment. -> The update-net just have to learn the update rules. 
#     Not the perception by itself.

#     Can be used as Sobel or as Laplacian.
#     -> Explain...
#     """
#     def __init__(self, channels, filter="sobel"):
#         super().__init__()

#         self.filter = filter

#         if self.filter.lower() == "sobel":
#             self.register_buffer(
#                 "kernel", torch.tensor([
#                     [-1.0, 0.0, 1.0],
#                     [-2.0, 0.0, 2.0],
#                     [-1.0, 0.0, 1.0],
#                 ]).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
#             ) 
#             self.size = 3
#         elif self.filter.lower() == "laplacian":
#             self.kernel = torch.tensor([
#                 [-1.0, -1.0, -1.0],
#                 [-1.0,  8.0, -1.0],
#                 [-1.0, -1.0, -1.0]
#             ]).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
#             self.size = 2
#         else:
#             raise ValueError(f"Unknown Filter passed: '{self.filter}'.") 


#     def forward(self, x):
        
#         if self.filter.lower() == "sobel":
#             dx = F.conv2d(x, self.kernel, padding=1, groups=x.shape[1])
#             dy = F.conv2d(x, self.kernel.transpose(2, 3), padding=1, groups=x.shape[1])
#             return torch.cat([x, dx, dy], dim=1)
#         elif self.filter.lower() == "laplacian":
#             laplacian = F.conv2d(x, self.kernel.to(x.device), padding=1, groups=x.shape[1])
#             return torch.cat([x, laplacian], dim=1)
#         else:
#             raise ValueError(f"Unknown Filter passed: '{self.filter}'.") 



# class NCAUpdateBlock(nn.Module):
#     def __init__(self, input_channels, hidden_channels, output_channels, kernel_size=3, kernel_size_2=1, activation=nn.ReLU(), is_final_block=False):
#         super().__init__()
#         self.is_final_block = is_final_block

#         if self.is_final_block:
#             padding1 = 0
#         else:
#             padding1 = 1

#         self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=kernel_size, padding=padding1)
#         self.activation = activation
#         if not self.is_final_block:
#             self.conv2 = nn.Conv2d(hidden_channels, output_channels, kernel_size=kernel_size_2, padding=0)

#         # formular: output_size = input_size + 2*padding - (kernel_size - 1)
#         # we use padding 1, so that the whole image/grid is processed 
#         # and no cell is skipped, else, the border pixels would be skipped


#     def forward(self, x):
#         if self.is_final_block:
#             x = self.conv1(x)
#             x = self.activation(x)
#         else:
#             x = self.conv1(x)
#             x = self.activation(x)
#             x = self.conv2(x)
#         return x
    


# class NeuralCellularAutomata(torch.nn.Module):
#     """
#     NCA model for image classification. 
#     The model consists of an input projection layer, a series of update blocks, and a classification head. 
#     The update blocks are responsible for iteratively updating the hidden state of the model based on learned rules. 
#     The final update block uses a Sigmoid activation to ensure that the updates are in a stable range.

#     > Note: Weights must be low initialized to ensure stability of the NCA. 
#     > The final update block uses a Sigmoid activation to ensure that the updates are in a stable range. 
#     """
#     def __init__(self, input_channels, num_classes, 
#                  hidden_channels=64, 
#                  steps=8, 
#                  update_blocks=1, 
#                  update_blocks_activation_kernel_size=3, 
#                  update_blocks_activation_kernel_size_2=1,
#                  update_blocks_activation="relu",
#                  final_update_block_activation_kernel_size=1,
#                  final_update_block_activation="sigmoid",
#                  perception_filter="sobel",
#                  dropout=0.1):
#         super().__init__()

#         if update_blocks < 1:
#             raise ValueError(f"At least 1 update block is needed, but {update_blocks} are wanted.")
        
#         self.input_channels = input_channels
#         self.hidden_channels = hidden_channels
#         self.num_classes = num_classes
#         self.steps = steps
#         self.dropout = dropout

#         # Perception -> already give every cell the information of their enviornment, via additional channels
#         self.perception = Perception(hidden_channels, filter=perception_filter)
#         input_size = hidden_channels * self.perception.size
#         # update_block_input_channels = [input_size] + [hidden_channels]*(update_blocks-1)

#         # Input-Projection -> 3 Channel image to hidden state
#         self.input_projection_net = nn.Conv2d(input_channels, hidden_channels, 1)

#         # State-Update-Network -> core of NCA -> learning rules for cell updates
#         kernel_sizes = handle_optional_list_param(update_blocks_activation_kernel_size, int, update_blocks)
#         kernel_sizes_2 = handle_optional_list_param(update_blocks_activation_kernel_size_2, int, update_blocks)
#         activations = handle_optional_list_param(update_blocks_activation, str, update_blocks, get_activation)
#         first_update_block = NCAUpdateBlock(
#             input_channels=input_size, 
#             hidden_channels=hidden_channels,
#             output_channels=hidden_channels, 
#             kernel_size=kernel_sizes[0], 
#             kernel_size_2=kernel_sizes_2[0], 
#             activation=activations[0], 
#             is_final_block=False
#         )
#         update_blocks = [
#             NCAUpdateBlock(
#                 input_channels=hidden_channels, 
#                 hidden_channels=hidden_channels,
#                 output_channels=hidden_channels, 
#                 kernel_size=k, 
#                 kernel_size_2=k2, 
#                 activation=a, 
#                 is_final_block=False
#             ) 
#             for _, k, k2, a in zip(range(update_blocks-1), kernel_sizes[1:], kernel_sizes_2[1:], activations[1:])
#         ]
        
#         final_update_block = NCAUpdateBlock(
#             input_channels=hidden_channels, 
#             hidden_channels=hidden_channels,
#             output_channels=hidden_channels,
#             kernel_size=final_update_block_activation_kernel_size, 
#             activation=get_activation(final_update_block_activation), 
#             is_final_block=True
#         )
        
#         # update_blocks.append(final_update_block)  
#         self.update_net = nn.Sequential(first_update_block, *update_blocks, final_update_block)

#         # Classification Head -> hidden state to class prediction
#         self.classification_head = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),  # global average pooling -> (B, C, H, W) -> (B, C, 1, 1)
#             nn.Flatten(),  # flatten to (B, C)
#             nn.Linear(hidden_channels, num_classes)
#         )

#         self.apply(init_weights)  # initialize weights of the model near 0


#     def step(self, x):
#         perception = self.perception(x)
#         update = self.update_net(perception)
#         # return x + update

#         # if update.shape[2:] != x.shape[2:]:
#         #     update = F.interpolate(update, size=x.shape[2:], mode='bilinear', align_corners=False)
    
#         # stochastic mask -> vor better generalization ability
#         #    -> update dropout
#         mask = torch.rand(x.shape[0], 1, x.shape[2], x.shape[3]) > self.dropout
#         return x + update * mask.to(x.device)
    

#     def get_last_state(self, x):
#         """
#         Like forward but returns the last hidden state instead of the logits.
#         """
#         # project input image to hidden state
#         x = self.input_projection_net(x)

#         # iterative updates of the hidden state
#         for _ in range(self.steps):
#             x = self.step(x)

#         return x  # return the 4D grid state: [B, C, H, W]
    

#     def forward(self, x):
#         # project input image to hidden state
#         x = self.input_projection_net(x)

#         # iterative updates of the hidden state
#         for _ in range(self.steps):
#             x = self.step(x)

#         # classify the final hidden state
#         logits = self.classification_head(x)
#         return logits


#     def save_transition_sequence(self, x, save_path):
#         """
#         Saves the state of the NCA at each step as an image file.
#         """
#         x = x[0:1]

#         current_x = self.input_projection_net(x)

#         history = []
#         history.append(current_x.detach().cpu())

#         for _ in range(self.steps):
#             current_x = self.step(current_x)
#             history.append(current_x.detach().cpu())

#         # apply PCA
#         history_tensor = torch.cat(history, dim=0)
#         T, C, H, W = history_tensor.shape

#         data = history_tensor.permute(0, 2, 3, 1).reshape(-1, C).numpy()

#         pca = PCA(n_components=3)
#         pca_data = pca.fit_transform(data)

#         pca_images = pca_data.reshape(T, H, W, 3).transpose(0, 3, 1, 2)
#         pca_images = torch.from_numpy(pca_images)
#         pca_images = (pca_images - pca_images.min()) / (pca_images.max() - pca_images.min())

#         num_images = len(pca_images)
#         cols = int(math.ceil(math.sqrt(num_images)))
#         rows = int(math.ceil(num_images / cols))

#         # make a grid from the images
#         grid_img = vutils.make_grid(pca_images, nrow=cols, padding=2, normalize=False)

#         vutils.save_image(grid_img, save_path)

        
#         # plt.figure(figsize=(5*cols, 5*rows))
#         # plt.imshow(grid_img.permute(1, 2, 0))
#         # plt.axis("off")
#         # plt.title(f"NCA Transition Sequence (Steps: {self.steps})")
#         # plt.tight_layout()
#         # plt.savefig(save_path, dpi=300)
#         # plt.close()























