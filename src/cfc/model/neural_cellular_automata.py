# -----------
# > Imports <
# -----------
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    prev_dropout = model.dropout
    model.dropout = 0.0
    with torch.no_grad():
        x = model.input_projection_net(input_img)

        total_change = 0.0
        for _ in range(model.steps):
            x_prev = x
            x = model.step(x)

            change = torch.norm(x - x_prev, p=2) / x.numel()
            total_change += change.item()

    model.dropout = prev_dropout

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


# -------------
# > NCA Model <
# -------------
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
            self.kernel = torch.tensor([
                [-1.0, -1.0, -1.0],
                [-1.0,  8.0, -1.0],
                [-1.0, -1.0, -1.0]
            ]).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
            self.size = 2
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
        else:
            raise ValueError(f"Unknown Filter passed: '{self.filter}'.") 



class NCAUpdateBlock(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, kernel_size=3, kernel_size_2=1, activation=nn.ReLU(), is_final_block=False):
        super().__init__()
        self.is_final_block = is_final_block

        if self.is_final_block:
            padding1 = 0
        else:
            padding1 = 1

        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=kernel_size, padding=padding1)
        self.activation = activation
        if not self.is_final_block:
            self.conv2 = nn.Conv2d(hidden_channels, output_channels, kernel_size=kernel_size_2, padding=0)

        # formular: output_size = input_size + 2*padding - (kernel_size - 1)
        # we use padding 1, so that the whole image/grid is processed 
        # and no cell is skipped, else, the border pixels would be skipped


    def forward(self, x):
        if self.is_final_block:
            x = self.conv1(x)
            x = self.activation(x)
        else:
            x = self.conv1(x)
            x = self.activation(x)
            x = self.conv2(x)
        return x
    


class NeuralCellularAutomata(torch.nn.Module):
    """
    NCA model for image classification. 
    The model consists of an input projection layer, a series of update blocks, and a classification head. 
    The update blocks are responsible for iteratively updating the hidden state of the model based on learned rules. 
    The final update block uses a Sigmoid activation to ensure that the updates are in a stable range.

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
                 dropout=0.1):
        super().__init__()

        if update_blocks < 1:
            raise ValueError(f"At least 1 update block is needed, but {update_blocks} are wanted.")
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.steps = steps
        self.dropout = dropout

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
        first_update_block = NCAUpdateBlock(
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
        
        final_update_block = NCAUpdateBlock(
            input_channels=hidden_channels, 
            hidden_channels=hidden_channels,
            output_channels=hidden_channels,
            kernel_size=final_update_block_activation_kernel_size, 
            activation=get_activation(final_update_block_activation), 
            is_final_block=True
        )
        
        # update_blocks.append(final_update_block)  
        self.update_net = nn.Sequential(first_update_block, *update_blocks, final_update_block)

        # Classification Head -> hidden state to class prediction
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # global average pooling -> (B, C, H, W) -> (B, C, 1, 1)
            nn.Flatten(),  # flatten to (B, C)
            nn.Linear(hidden_channels, num_classes)
        )

        self.apply(init_weights)  # initialize weights of the model near 0


    def step(self, x):
        perception = self.perception(x)
        update = self.update_net(perception)
        # return x + update

        # if update.shape[2:] != x.shape[2:]:
        #     update = F.interpolate(update, size=x.shape[2:], mode='bilinear', align_corners=False)
    
        # stochastic mask -> vor better generalization ability
        #    -> update dropout
        mask = torch.rand(x.shape[0], 1, x.shape[2], x.shape[3]) > self.dropout
        return x + update * mask.to(x.device)
    

    def get_last_state(self, x):
        """
        Like forward but returns the last hidden state instead of the logits.
        """
        # project input image to hidden state
        x = self.input_projection_net(x)

        # iterative updates of the hidden state
        for _ in range(self.steps):
            x = self.step(x)

        return x  # return the 4D grid state: [B, C, H, W]
    

    def forward(self, x):
        # project input image to hidden state
        x = self.input_projection_net(x)

        # iterative updates of the hidden state
        for _ in range(self.steps):
            x = self.step(x)

        # classify the final hidden state
        logits = self.classification_head(x)
        return logits

    










