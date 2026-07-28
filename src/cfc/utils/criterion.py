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
class AELoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.latest_reconstruction_loss = float("inf")

    def forward(self, inputs, targets):
        reconstructed_x, z = inputs[:2]
        return self.loss_(reconstructed_x, targets)

    def loss_(self, reconstructed_x, x):
        # batch_size = x.size(0)

        # comparison to the original image
        # reconstruction_loss = F.mse_loss(reconstructed_x, x, reduction='mean')
        reconstruction_loss = F.l1_loss(reconstructed_x, x, reduction='mean')
        self.latest_reconstruction_loss = reconstruction_loss.detach().cpu()

        return reconstruction_loss 


class CenterLoss(nn.Module):
    def __init__(self, num_classes, latent_dim, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        
        # learnable centroids for every class in latent space
        self.centers = nn.Parameter(torch.randn(num_classes, latent_dim))

    def forward(self, z, labels):
        # z: [Batch_size, latent_dim]
        # labels: [Batch_size]
        batch_size = z.size(0)
        
        # Get the center of every sample in current batch
        centers_batch = self.centers[labels] # [Batch_size, latent_dim]
        
        # Compute the mean squared distance to the center
        loss = torch.sum(torch.norm(z - centers_batch, dim=1)**2) / batch_size
        return loss


class DiversityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, labels):
        # z: [Batch_size, latent_dim]
        # Calc pairwise cosine similarity or euclidean distances in the batch
        # we want that the similiarites are as different as possible
        z_norm = F.normalize(z, p=2, dim=1)
        sim_matrix = torch.mm(z_norm, z_norm.t()) # Ähnlichkeitsmatrix [Batch, Batch]

        # Create mask for pairs with different classes
        # labels.unsqueeze(1) is [Batch, 1], labels.unsqueeze(0) is [1, Batch]
        # Creates a [Batch, Batch] matrix with true where the classes are different
        labels_1 = labels.unsqueeze(1)
        labels_2 = labels.unsqueeze(0)
        diff_class_mask = (labels_1 != labels_2)
        
        # We mask the diagonal (sample itself are of course max similiar)
        diag_mask = torch.eye(z.size(0), device=z.device).bool()
        valid_mask = diff_class_mask & (~diag_mask)
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=z.device)

        diff_similarities = sim_matrix[valid_mask]
        
        # this loss gets high if the latent representation of them are similiar
        return diff_similarities.mean()



class MixedAEClassificationLoss(nn.Module):
    def __init__(self, num_classes, latent_dim, lambda_center):
        super().__init__()
        self.ae_loss = AELoss()
        self.class_loss = FocalLoss()
        self.center_loss = DiversityLoss() # CenterLoss(num_classes, latent_dim)

        self.loss_state = 0
        self.lambda_rec = 1.0
        self.lambda_cls = 0.0
        self.lambda_center = lambda_center

        self.latest_cls_loss = float("inf")
        self.latest_center_loss = float("inf")

    def forward(self, inputs, targets):
        origin_x, class_target = targets

        # if self.loss_state == 0:
        #     return self.ae_loss(inputs, origin_x)
        # elif self.loss_state == 1:
        class_pred = inputs[-1]
        rec_loss = self.ae_loss(inputs, origin_x)
        cls_loss = self.class_loss(class_pred, class_target)
        z = inputs[1] 
        cent_loss = self.center_loss(z, class_target)

        self.latest_cls_loss = cls_loss.detach().cpu()
        self.latest_center_loss = cent_loss.detach().cpu()

        return rec_loss*self.lambda_rec + cls_loss*self.lambda_cls + cent_loss*self.lambda_center
        # else:
        #     raise ValueError(f"Have unknwon loss-state: {self.loss_state}")

    def epoch_update(self, new_epoch, total_epochs):
        epoch_progress = new_epoch / total_epochs

        if epoch_progress < 0.5:
            self.loss_state = 0
        else:
            self.loss_state = 1
            self.lambda_cls = epoch_progress *1.5

    

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
        reconstructed_x, mu, logvar = inputs[:3]
        return self.loss_(reconstructed_x, targets, mu, logvar)

    def loss_(self, reconstructed_x, x, mu, logvar):
        # batch_size = x.size(0)

        # comparison to the original image
        reconstruction_loss = F.mse_loss(reconstructed_x, x, reduction='mean')
        self.latest_reconstruction_loss = reconstruction_loss.detach().cpu()
    
        # kl-divergence -> force latent space to follow standard-normal-distribution
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()  # sum on latent dims
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
        self.lambda_ = 0.0

    def forward(self, inputs, targets):
        origin_x, class_target = targets

        if self.loss_state == 0:
            return self.vae_loss(inputs, origin_x)
        elif self.loss_state == 1:
            class_pred = inputs[-1]
            return self.vae_loss(inputs, origin_x) + self.lambda_ * self.class_loss(class_pred, class_target)
        else:
            raise ValueError(f"Have unknwon loss-state: {self.loss_state}")

    def epoch_update(self, new_epoch, total_epochs):
        epoch_progress = new_epoch / total_epochs

        if epoch_progress < 0.5:
            self.loss_state = 0
        else:
            self.loss_state = 1
            self.lambda_ = epoch_progress *1.5



# ---------------------
# > Criterion Loading <
# ---------------------
def get_criterion(criterion_name, class_weights=None, num_classes=10, latent_dim=128, lambda_center=0.01, vae_criterion_beta=1.0, vae_criterion_use_smooth_beta=True, vae_criterion_use_smooth_beta_start_value=0.2):
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
    
    elif criterion_name.lower() == "ae_loss":
            return MixedAEClassificationLoss(num_classes=num_classes, latent_dim=latent_dim, lambda_center=lambda_center)

    else:
        raise ValueError(f"Criterion {criterion_name} not supported.")
    
    # if criterion_name.lower() == "cross_entropy":
    #     return torch.nn.CrossEntropyLoss()
    # elif criterion_name.lower() == "mse":
    #     return torch.nn.MSELoss()
    # else:
    #     raise ValueError(f"Criterion {criterion_name} not supported.")
    









