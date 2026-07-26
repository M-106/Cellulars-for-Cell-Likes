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

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from cfc.utils.config import save_config
from cfc.utils.metrics import calculate_isic_metrics, get_used_label_names
from cfc.utils.data import get_data
from cfc.model.model_loading import get_model
from cfc.utils.optimizer import get_optimizer
from cfc.utils.scheduler import get_scheduler
from cfc.utils.criterion import get_criterion
from cfc.utils.config import log_config_to_tensorboard, config_as_str
from cfc.model.neural_cellular_automata import measure_nca_stability
from cfc.utils.stepwise_train import save_cur_train_state, load_train_state, get_config_and_dir_from_last_train
from cfc.utils.train_utils import plot_sample_images, check_active_dimensions, latent_cluster_analysis



# --------------
# > Train Loop <
# --------------
def validate(model, model_name, criterion, data_loader, device, tensorboard_writer, cur_epoch, output_dir=None, vae_is_latent_training=True, vae_using_nca=False):
    model.eval()
    is_autoencoder = model_name.lower() in ["vae", "convvae", "ae", "convae"] 
    all_predictions, all_labels, weights = [], [], []
    latent_vectors, latent_labels = [], []
    loss = 0.0
    plots = 0
    saved_transition = False

    # AutoEncoder Check Active Latent Dimensions
    if is_autoencoder and vae_is_latent_training:
        active_dims = check_active_dimensions(model, data_loader, device, threshold=0.01)
        tensorboard_writer.add_scalar("ActiveLatentDims", active_dims, cur_epoch)

    with torch.no_grad():
        for imgs, labels, _, validation_weights in tqdm(data_loader, total=len(data_loader), desc="Validation Run"):
            imgs = imgs.to(device)
            if not is_autoencoder:
                labels = labels.to(device)
            else:
                labels_ = labels
                labels = imgs

            # NCA Transition Plot & Step Stability
            if saved_transition is False and (not is_autoencoder or vae_using_nca):
                l2_stability = measure_nca_stability(model, imgs[0:1])
                tensorboard_writer.add_scalar("Stability/L2_Change", l2_stability, cur_epoch)

                if hasattr(model, "save_transition_sequence"):
                    model.save_transition_sequence(x=imgs, save_path=os.path.join(output_dir, f"nca_transition_epoch_{cur_epoch:03}.png"))
                    saved_transition = True

            # forwarding & calc loss
            outputs = model(imgs)
            loss += criterion(outputs, labels).mean().item()  # mean to avoid gradient explosion

            # Sample Plotting
            if plots < 5:  # only plot a few samples
                if is_autoencoder:
                    reconstructed_img = outputs[0][0].detach().cpu().permute(1, 2, 0).numpy()
                    reconstructed_img = (reconstructed_img - reconstructed_img.min()) / (reconstructed_img.max() - reconstructed_img.min() + 1e-8) 
                    origin_img = imgs[0].detach().cpu().permute(1, 2, 0).numpy()
                    origin_img = (origin_img - origin_img.min()) / (origin_img.max() - origin_img.min() + 1e-8) 

                    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
                    ax[0].imshow(origin_img)
                    ax[0].set_title("Input Original")
                    ax[0].axis("off")
                    ax[1].imshow(reconstructed_img)
                    ax[1].set_title("AutoEncoder Reconstruction")
                    ax[1].axis("off")

                    save_path = f"{output_dir}/sample_plot_{cur_epoch:03}_{plots}.png" if output_dir else None
                    if save_path:
                        plt.savefig(save_path)

                    tensorboard_writer.add_figure("SamplePlot", fig, global_step=cur_epoch)
                    plt.close(fig)

                # NCA Sample Plot
                else:
                    last_state = model.get_last_state(imgs[0:1])  # [1, C, H, W]
                    pred_grid = last_state[0].detach().cpu().permute(1, 2, 0).numpy()
                    H, W, C = pred_grid.shape
                    # pred_img = pred_grid[:, :, :3]
                    pca = PCA(n_components=3)
                    pca_data = pca.fit_transform(pred_grid.reshape(H * W, C))
                    pred_img = pca_data.reshape(H, W, 3)
                    # min-max normalization, to [0, 1]
                    pred_img = (pred_img - pred_img.min()) / \
                               (pred_img.max() - pred_img.min()) 

                    pred_label = torch.argmax(outputs[0], dim=0).item()
                    gt_label = labels[0].item()  # labels[0].detach().cpu().numpy()

                    save_path = f"{output_dir}/sample_plot_{cur_epoch:03}_{plots}.png" if output_dir else None
                    plot_sample_images(
                        input_img=imgs[0].detach().cpu().permute(1, 2, 0).numpy(),
                        pred_img=pred_img,
                        class_pred=pred_label,
                        class_pred_name=data_loader.dataset.idx_to_class[pred_label],
                        class_label=gt_label,  
                        class_label_name=data_loader.dataset.idx_to_class[gt_label],
                        save_path=save_path,
                        cur_epoch=cur_epoch,
                        tensorboard_writer=tensorboard_writer
                    )

                plots += 1  


            if is_autoencoder:
                outputs_ = model(imgs, classify=True)
                _, preds = torch.max(outputs_, 1)
                all_predictions.extend(preds.detach().cpu().numpy())
                all_labels.extend(labels_.detach().cpu().numpy())
                weights.extend(validation_weights.detach().cpu().numpy())

                if vae_is_latent_training:
                    latent_vectors.append(outputs[1].detach().cpu())
                    latent_labels.append(labels_.detach().cpu())  # FIXME -> get from model
            else:
                _, preds = torch.max(outputs, 1)
                all_predictions.extend(preds.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())
                weights.extend(validation_weights.detach().cpu().numpy())


    # calc metrics
    used_label_names = get_used_label_names(all_labels, all_predictions, idx_to_class=data_loader.dataset.idx_to_class)
    metrics = calculate_isic_metrics(all_labels, all_predictions, used_label_names, None)  # , sample_weights=weights -> we have another validation split!
    print(f"Weighted Balanced Val Accuracy: {metrics['balanced_accuracy']:.4f}")

    # Silhoutte Score + PCA Scatter Plot
    if is_autoencoder and vae_is_latent_training:
        latent_cluster_analysis(latent_vectors, latent_labels, output_dir, plots, cur_epoch, tensorboard_writer)

    return metrics, loss / len(data_loader)



def get_vae_training_is_latent_training(model):
    if hasattr(model, "train_target_state"):
        # note: even if that is another model, True would be fine (/not matter)
        if model.train_target_state == 0:
            return True

    return False


def train(
        model_name, 
        model_kwargs,
        data_path, 
        num_epochs, 
        batch_size, 
        learning_rate, 
        weight_decay, 
        criterion_name,
        vae_criterion_beta,
        vae_criterion_use_smooth_beta,
        vae_criterion_use_smooth_beta_start_value,
        optimizer_name, 
        scheduler_name, 
        output_dir,
        exp_name,
        used_train_samples,
        used_val_samples,
        config,
        continue_training
    ):
    """
    Train a model on a specified dataset.

    Args:
        model_name (str): Name of the model to train.
        data_path (str): Path to the dataset.
        num_epochs (int): Number of epochs to train for.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        criterion_name (str): Loss/criterion to use.
        vae_criterion_beta (float): Beta value of VAE Loss for Controlling Normal Distribution Force.
        vae_criterion_use_smooth_beta (bool): Decides if using smoothing if using VAE Loss.
        vae_criterion_use_smooth_beta_start_value (float): Defines the startvalue of the smoothing of VAE Loss.
        optimizer_name (str): Optimizer to use.
        scheduler_name (str): Learning rate scheduler to use.
        output_dir (str): Directory to save the trained model, plots, and logs.
        exp_name (str): Name of the experiment for logging and saving purposes.
        used_train_samples (int): Decides how much samples getting used for training.
        used_val_samples (int): Decides how much samples getting used for validation.
        config (pydantic.BaseModel): Configuration object for user setting handling.
    """
    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.accelerator.current_accelerator()

    # get data
    train_data = get_data(
        data_path=data_path, 
        batch_size=batch_size, 
        partition="train",
        shuffle=True,
        used_samples=used_train_samples
    )
    val_data = get_data(
        data_path=data_path, 
        batch_size=batch_size, 
        partition="val",
        shuffle=False,
        used_samples=used_val_samples
    )

    # get model
    is_autoencoder = model_name.lower() in ["vae", "convvae", "ae", "convae"]
    vae_using_nca = model_kwargs["vae_using_nca"]
    model = get_model(model_name, num_classes=len(train_data.dataset.class_names), **model_kwargs)
    model.to(device)
    vae_is_latent_training = get_vae_training_is_latent_training(model)
    
    optimizer = get_optimizer(optimizer_name, model.parameters(), learning_rate, weight_decay)
    scheduler = get_scheduler(scheduler_name, optimizer, num_epochs)
    criterion = get_criterion(criterion_name, vae_criterion_beta=vae_criterion_beta, vae_criterion_use_smooth_beta=vae_criterion_use_smooth_beta, vae_criterion_use_smooth_beta_start_value=vae_criterion_use_smooth_beta_start_value)

    if continue_training:
        start_epoch, best_val_accuracy, best_model_epoch, latest_model_epoch, best_checkpoint_path = load_train_state(output_dir, model, optimizer, scheduler)

    # tensorboard_dir = f"{output_dir}/logs/{exp_name}"
    tensorboard_dir = f"{output_dir}/logs"
    tensorboard_writer = SummaryWriter(log_dir=tensorboard_dir)
    if not continue_training:
        # add all configs to tensorboard
        tensorboard_writer.add_text("Model-Name", str(model_name))
        tensorboard_writer.add_text("Data-Path", str(data_path))
        tensorboard_writer.add_text("Num-Epochs", str(num_epochs))
        tensorboard_writer.add_text("Batch-Size", str(batch_size))
        tensorboard_writer.add_text("Learning-Rate", str(learning_rate))
        tensorboard_writer.add_text("Weight-Decay", str(weight_decay))
        tensorboard_writer.add_text("Criterion", str(criterion_name))
        tensorboard_writer.add_text("Optimizer", str(optimizer_name))
        tensorboard_writer.add_text("Scheduler", str(scheduler_name))
        tensorboard_writer.add_text("Train-Data Len", str(len(train_data)))
        tensorboard_writer.add_text("Valid-Data Len", str(len(val_data)))

        log_config_to_tensorboard(writer=tensorboard_writer, tag="Config/Experiment", config=config)

        # if not is_autoencoder:
        if hasattr(model, "save_transition_sequence"):
            input_data = next(iter(val_data))[0].to(device)
            model.save_transition_sequence(x=input_data, save_path=os.path.join(output_dir, "nca_transition_epoch_-1.png"))
            del input_data

        best_val_accuracy = 0.0 if not is_autoencoder else float('inf')
        best_model_epoch = -1
        latest_model_epoch = -1
        best_checkpoint_path = None
        start_epoch = 0

    for epoch in range(start_epoch, num_epochs):    # tqdm(range(num_epochs), total=num_epochs, desc="Training Progress"):
        model.train()
        running_loss = 0.0

        if hasattr(criterion, "epoch_update"):
            criterion.epoch_update(new_epoch=epoch, total_epochs=num_epochs)
        if hasattr(criterion, "beta"):
            tensorboard_writer.add_scalar("CriterionBeta/Train", criterion.beta, epoch)

        if hasattr(model, "epoch_update"):
            model.epoch_update(epoch, num_epochs)
        if hasattr(model, "train_target_state"):
            tensorboard_writer.add_scalar("ModelState", model.train_target_state, epoch)
        vae_is_latent_training = get_vae_training_is_latent_training(model)

        # one train loop
        for imgs, labels, _, _ in tqdm(train_data, total=len(train_data), desc=f"NCA Training Epoch {epoch:03}"):
            imgs = imgs.to(device)
            if not is_autoencoder:
                labels = labels.to(device)
            else:
                labels = imgs

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels).mean()  # mean to avoid gradient explosion
            loss.backward()  

            # gradient clipping -> important with NCAs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        avg_loss = running_loss / len(train_data)
        tensorboard_writer.add_scalar("Loss/Train", avg_loss, epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Validate after each epoch
        metrics, val_loss = validate(model, model_name, criterion, val_data, device, tensorboard_writer, cur_epoch=epoch, output_dir=output_dir, vae_is_latent_training=vae_is_latent_training, vae_using_nca=vae_using_nca)
        tensorboard_writer.add_scalar("Loss/Validation", val_loss, epoch)
        tensorboard_writer.add_scalar("Balanced_Accuracy/Validation", metrics['balanced_accuracy'], epoch)
        if is_autoencoder and vae_is_latent_training:
            tensorboard_writer.add_scalar("Loss/Reconstruction", criterion.vae_loss.latest_reconstruction_loss.item(), epoch)
            tensorboard_writer.add_scalar("Loss/KL_Divergence", criterion.vae_loss.latest_kl_loss.item(), epoch)


        # Save Weights
        checkpoint_path = f"{output_dir}/latest_model.pth"
        torch.save(model.state_dict(), checkpoint_path)
        latest_model_epoch = epoch

        if not is_autoencoder or not vae_is_latent_training:
            cur_score = metrics['balanced_accuracy']
            cur_score_name = "Balanced Accuracy"
            cur_is_better = best_val_accuracy < cur_score
        else:
            cur_score = val_loss
            cur_score_name = "VAE-Loss"
            cur_is_better = best_val_accuracy > cur_score

        if cur_is_better:
            best_val_accuracy = cur_score
            best_checkpoint_path = f"{output_dir}/best_model.pth"
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"Best model saved with {cur_score_name}: {best_val_accuracy:.4f}")
            best_model_epoch = epoch

        
        # save train state as latest state so that we can continue here
        save_cur_train_state(
            output_dir=output_dir,
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            cur_epoch=epoch,
            best_val_accuracy=best_val_accuracy,
            best_model_epoch=best_model_epoch, 
            latest_model_epoch=latest_model_epoch,
            best_checkpoint_path=best_checkpoint_path
        )
    
    with open(f"{output_dir}/training_summary.txt", "w") as f:
        f.write(f"Training: {exp_name}\n")
        f.write(f"Output Directory: {output_dir}\n")
        f.write(f"Tensorboard Directory: {tensorboard_dir}\n")
        f.write(f"Best Model Epoch: {best_model_epoch}\n")
        f.write(f"Best Model Path: {best_checkpoint_path}\n")
        f.write(f"Best Validation Balanced Accuracy: {best_val_accuracy:.4f}\n")
        f.write(f"Latest Model Epoch: {latest_model_epoch}\n")
        f.write(f"Latest Model Path: {checkpoint_path}\n")
        f.write(f"\n\nConfig:\n{config_as_str(config)}\n")

    tensorboard_writer.close()
        



# -------------
# > Execution <
# -------------

def main(config):

    # check if should continue training
    continue_training = config.train.continue_training
    last_training_output_dir = config.train.last_training_output_dir
    if continue_training:
        config, output_dir = get_config_and_dir_from_last_train(output_dir=last_training_output_dir)

    # extract configs
    # to handle new added config attributes
    try_extract_optional_param = lambda attr, conf, default: getattr(conf, attr) if hasattr(conf, attr) else default

    model_name = config.model.name
    model_kwargs = config.model.kwargs
    data_path = config.data.path
    num_epochs = config.train.num_epochs
    batch_size = config.train.batch_size
    learning_rate = config.train.learning_rate
    weight_decay = config.train.weight_decay
    criterion = config.train.criterion
    vae_criterion_beta = try_extract_optional_param("vae_criterion_beta", config.train, 1.0)
    vae_criterion_use_smooth_beta = try_extract_optional_param("vae_criterion_use_smooth_beta", config.train, True)
    vae_criterion_use_smooth_beta_start_value = try_extract_optional_param("vae_criterion_use_smooth_beta_start_value", config.train, 0.2)
    optimizer = config.train.optimizer
    scheduler = config.train.scheduler
    output_dir = config.train.output_dir
    exp_name = config.train.exp_name
    used_train_samples = config.train.used_train_samples
    used_val_samples = config.train.used_val_samples


    # create exp output folder
    if not continue_training:
        output_dir = f"{output_dir}/{time.strftime('%Y-%m-%d_%H-%M-%S')}_{exp_name}"
        os.makedirs(output_dir, exist_ok=True)
        # shutil.rmtree(output_dir)
        # os.makedirs(output_dir, exist_ok=True)


        # track config for this run
        save_config(config, f"{output_dir}/config.yaml")
    else:
        output_dir = last_training_output_dir


    # start training
    train(
        model_name=model_name,
        model_kwargs=model_kwargs,
        data_path=data_path,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        criterion_name=criterion,
        vae_criterion_beta=vae_criterion_beta,
        vae_criterion_use_smooth_beta = vae_criterion_use_smooth_beta,
        vae_criterion_use_smooth_beta_start_value = vae_criterion_use_smooth_beta_start_value,
        optimizer_name=optimizer,
        scheduler_name=scheduler,
        output_dir=output_dir,
        exp_name=exp_name,
        used_train_samples=used_train_samples,
        used_val_samples=used_val_samples,
        config=config,
        continue_training=continue_training
    )







