# -----------
# > Imports <
# -----------
import time
import os
import shutil

import torch
# matplotlib background mode without tkinter, default is TkAgg
# Agg = Anti-Grain Geometry: A purely file-based backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from cfc.model.autoencoder import ConvAE



# ------------
# > Plotting <
# ------------
def plot_sample_images(input_img, pred_img, class_pred, class_pred_name, class_label, class_label_name, save_path=None, cur_epoch=0, tensorboard_writer=None):
    """
    Plot sample images for classication with NCA
    """
    plt.style.use('ggplot')
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())
    ax[0].imshow(input_img)
    ax[0].set_title("Input Image")
    ax[0].text(0.5, -0.1, f"Class: {class_label} (Label: {class_label_name})", ha='center', va='top', fontsize=10, transform=ax[0].transAxes)
    ax[0].axis("off")

    ax[1].imshow(pred_img)
    ax[1].set_title("Predicted Image")
    ax[1].text(0.5, -0.1, f"Class: {class_pred} (Label: {class_pred_name})", ha='center', va='top', fontsize=10, transform=ax[1].transAxes)
    ax[1].axis("off")

    # plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    if tensorboard_writer:
        tensorboard_writer.add_figure("SamplePlot", fig, global_step=cur_epoch)
    plt.close(fig)




# ----------------------
# > AutoEncoder Helper <
# ----------------------
def latent_cluster_analysis(latent_vectors, latent_labels, output_dir, plots, cur_epoch, tensorboard_writer):
    if len(latent_vectors) > 0:
        # Latent Space Cluster Analysis (PCA + Silhoutte Score)
        X_latent = torch.cat(latent_vectors, dim=0).numpy()
        y_labels = torch.cat(latent_labels, dim=0).numpy()

        # Compute Silhoutte Score
        if len(set(y_labels)) > 1:
            silhouette_score_ = silhouette_score(X_latent, y_labels)
            tensorboard_writer.add_scalar("Latent/Silhoutte_Score", silhouette_score_, cur_epoch)
            print(f"Latent Space Silhoutte Score: {silhouette_score_:.4f}")

        # PCA Scatter Plot from Latent Space
        pca_2d = PCA(n_components=2).fit_transform(X_latent)
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(pca_2d[:, 0], pca_2d[:, 1], c=y_labels, cmap='tab10', alpha=0.6, s=15)
        plt.colorbar(scatter, ax=ax, label="Class Labels")
        ax.set_title(f"Latent Space PCA (Epoch {cur_epoch})")

        save_path = f"{output_dir}/latent_space_pca_{cur_epoch:03}_{plots}.png" if output_dir else None
        if save_path:
            plt.savefig(save_path)

        tensorboard_writer.add_figure("LatentSpace/PCA_Scatter", fig, global_step=cur_epoch)
        plt.close(fig)



@torch.no_grad()
def check_active_dimensions(model, val_loader, device, threshold=0.01):
    model.eval()
    all_latent_values = []
    
    for imgs, _, _, _ in val_loader:
        imgs = imgs.to(device)
        latent_space = model.encoder(imgs)
        latent_space = torch.flatten(latent_space, start_dim=1)
        if isinstance(model, ConvAE):
            z = model.encoder_fc(latent_space)
            all_latent_values.append(z.cpu())
        else:
            mu = model.fc_mu(latent_space)
            all_latent_values.append(mu.cpu())

    # [N_samples, latent_dim]
    all_latent_values = torch.cat(all_latent_values, dim=0)

    # compute the variance of latent-dimension over all validation samples
    variances = torch.var(all_latent_values, dim=0)
    
    # a dimension is active if its variance is beyond a threhold
    active_dims = torch.sum(variances > threshold).item()
    total_dims = all_latent_values.shape[1]
    
    print(f"Active Latent-Dimensionen: {active_dims} / {total_dims}")
    if active_dims < 2:
        print("Posterior Collapse detected! Almost all dimensions are dead.")
        
    return active_dims 












