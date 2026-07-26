# -----------
# > Imports <
# -----------
import torch
import torch.nn as nn
import torch.nn.functional as F



# ---------------------
# > Define Focal Loss <
# ---------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()



# ---------------------
# > Define VAE Loss <
# ---------------------
class VAELoss(nn.Module):
    def __init__(self, beta=1.0, beta_growing=True, beta_start_epoch=0.2):
        super().__init__()
        self.target_beta = beta
        self.beta = beta
        self.beta_growing = beta_growing
        self.beta_start_epoch = beta_start_epoch

        self.latest_reconstruction_loss = float("inf")
        self.latest_kl_loss = float("inf")

    def forward(self, inputs, targets):
        reconstructed_x, mu, logvar = inputs
        return self.loss_(reconstructed_x, targets, mu, logvar)

    def loss_(self, reconstructed_x, x, mu, logvar):
        # batch_size = x.size(0)

        # comparison to the original image
        reconstruction_loss = F.mse_loss(reconstructed_x, x, reduction='mean')
        self.latest_reconstruction_loss = reconstruction_loss.detach().cpu()
    
        # kl-divergence -> force latent space to follow standard-normal-distribution
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        self.latest_kl_loss = kl_loss.detach().cpu()

        total_loss = reconstruction_loss + (self.beta * kl_loss)

        return total_loss 

    def epoch_update(self, new_epoch, total_epochs):
        # Regulation against Posterior Collapse
        if self.beta_growing:
            epoch_progress = new_epoch / total_epochs

            if epoch_progress < self.beta_start_epoch:
                self.beta = 0.0
            else:
                norm_progress = (epoch_progress - self.beta_start_epoch) / (1.0 - self.beta_start_epoch + 1e-8)
                self.beta = min(self.target_beta, norm_progress * self.target_beta)
        else:
            self.beta = self.target_beta



class MixedVAEClassificationLoss(nn.Module):
    def __init__(self, beta=1.0, beta_growing=True, beta_start_epoch=0.2):
        super().__init__()
        self.vae_loss = VAELoss(beta=beta, beta_growing=beta_growing, beta_start_epoch=beta_start_epoch)
        self.class_loss = FocalLoss()

        self.loss_state = 0

    def forward(self, inputs, targets):
        if self.loss_state == 0:
            return self.vae_loss(inputs, targets)
        elif self.loss_state == 1:
            return self.class_loss(inputs, targets)
        else:
            raise ValueError(f"Have unknwon loss-state: {self.loss_state}")

    def epoch_update(self, new_epoch, total_epochs):
        epoch_progress = new_epoch / total_epochs

        if epoch_progress < 0.5:
            self.loss_state = 0
        else:
            self.loss_state = 1



# ---------------------
# > Criterion Loading <
# ---------------------
def get_criterion(criterion_name, class_weights=None, vae_criterion_beta=1.0, vae_criterion_use_smooth_beta=True, vae_criterion_use_smooth_beta_start_value=0.2):
    """
    Load a criterion (loss function) based on the provided criterion name.

    Reduction="none" is used to return the loss for each sample, allowing for custom aggregation later.

    Args:
        criterion_name (str): Name of the criterion to load.
        class_weights (torch.Tensor, optional): Weights for each class.

    Returns:
        torch.nn.Module: The loaded criterion.
    """
    if criterion_name.lower() == "cross_entropy":
        return torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1, reduction='none')
    
    # Focal Loss ist der Goldstandard bei starkem Ungleichgewicht
    elif criterion_name.lower() == "focal_loss":
        return FocalLoss(gamma=2.0, alpha=class_weights)

    elif criterion_name.lower() == "vae_loss":
        return MixedVAEClassificationLoss(beta=vae_criterion_beta, beta_growing=vae_criterion_use_smooth_beta, beta_start_epoch=vae_criterion_use_smooth_beta_start_value)
        # return VAELoss(beta=vae_criterion_beta, beta_growing=vae_criterion_use_smooth_beta, beta_start_epoch=vae_criterion_use_smooth_beta_start_value)
    
    else:
        raise ValueError(f"Criterion {criterion_name} not supported.")
    
    # if criterion_name.lower() == "cross_entropy":
    #     return torch.nn.CrossEntropyLoss()
    # elif criterion_name.lower() == "mse":
    #     return torch.nn.MSELoss()
    # else:
    #     raise ValueError(f"Criterion {criterion_name} not supported.")
    









