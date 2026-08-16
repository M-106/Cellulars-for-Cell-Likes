# ------------
# > Imports <
# ------------
import os
import shutil
import inspect

import numpy as np
import pandas as pd

import torch

from tqdm import tqdm

from cfc.utils.metrics import calculate_isic_metrics, get_used_label_names
from cfc.utils.data import get_data
from cfc.model.model_loading import get_model, is_ae
from cfc.utils.config import load_config



# ----------
# > Helper <
# ----------
CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]

def save_isic_predictions(image_paths, preds, output_csv):

    # if pytorch tensor -> convert to numpy
    if hasattr(preds, "detach"):
        preds = preds.detach().cpu().numpy()
    elif isinstance(preds, list):
        preds = np.array(preds)

    # get only the image names
    image_names = [os.path.basename(str(p)) for img_batch in image_paths for p in img_batch]

    # Build DataFrame
    df = pd.DataFrame(preds, columns=CLASS_NAMES)
    df.insert(0, "image", image_names)

    # Save CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")


def reorder_batch_predictions(batch_preds, idx_to_class, target_classes):
    """
    Sorts a batch of predictions [batch_size, num_classes] 
    in the order of target_classes.
    """
    if hasattr(batch_preds, "detach"):
        batch_preds = batch_preds.detach().cpu().numpy()

    # Mpa index of target class to the modeloutput-idx
    class_to_idx = {v.lower(): k for k, v in idx_to_class.items()}
    mapping = [class_to_idx[c.lower()] for c in target_classes]

    # Re-order over col-indexing
    return batch_preds[:, mapping]



# -------------
# > Test Loop <
# -------------
def evaluate(model, data_loader, device, output_dir):
    model.eval()
    all_paths = []
    all_raw_predictions = []
    all_predictions = []
    all_labels = []
    weights = []

    idx_to_class = data_loader.dataset.idx_to_class

    with torch.no_grad():
        for inputs, labels, score_weight, validation_weight, img_path in tqdm(data_loader, total=len(data_loader), desc="Test Epoch"):
            inputs, labels = inputs.to(device), labels.to(device)

            if "classify" in inspect.signature(model.forward).parameters:
                outputs = model(inputs, classify=True)
            else:
                outputs = model(inputs)
            
            # softmax / logits
            raw_preds = torch.softmax(outputs, dim=1) if not torch.allclose(outputs.sum(dim=1), torch.tensor(1.0).to(device)) else outputs
            _, preds = torch.max(outputs, 1)

            # Re-ordering pro Batch
            ordered_raw_preds = reorder_batch_predictions(
                batch_preds=raw_preds, 
                idx_to_class=idx_to_class, 
                target_classes=CLASS_NAMES
            )

            all_paths.append(img_path)
            all_raw_predictions.append(ordered_raw_preds)
            all_predictions.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            weights.extend(score_weight)  # Assuming equal weights for each sample

    # stack all predictions together into [N, 9]
    all_raw_predictions = np.vstack(all_raw_predictions)
    out_path_root, out_exp_name = os.path.split(output_dir)
    save_isic_predictions(
        image_paths=all_paths, 
        preds=all_raw_predictions, 
        output_csv=os.path.join(out_path_root, "isic_2019_preds", f"{out_exp_name}.csv")
    )

    used_label_names = get_used_label_names(all_labels, all_predictions, idx_to_class=data_loader.dataset.idx_to_class)
    metrics = calculate_isic_metrics(all_predictions, all_labels, used_label_names, weights)
    return metrics



def test(model_name, model_kwargs, checkpoint_path, data_path, batch_size, output_dir):
    """
    Test a model on a specified dataset.

    Args:
        model_name (str): Name of the model to test.
        model_kwargs (dict): Arguments for the model.
        checkpoint_path (str): Path to the model checkpoint to use.
        data_path (str): Path to the dataset to use.
        batch_size (int): Batch size for testing.
        output_dir (str): Directory to save the tested model, plots, and logs.
    """
    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.accelerator.current_accelerator()

    # get data
    test_data = get_data(
        data_path=data_path, 
        batch_size=batch_size, 
        partition="test",
        shuffle=False, 
        used_samples=-1,
        return_also_img_path=True
    )

    # get model
    num_classes = len(test_data.dataset.class_names)
    is_autoencoder_or_vae = model_name.lower() in ["vae", "convvae", "ae", "convae"]
    vae_using_nca = model_kwargs["vae_using_nca"] if "vae_using_nca" in model_kwargs else False 
    model = get_model(model_name, num_classes=num_classes, checkpoint_path=checkpoint_path, **model_kwargs)
    model.to(device)
    vae_is_latent_training = False
    is_autoencoder_class_not_vae = is_ae(model)
    latent_dim = model_kwargs["latent_dim"] if "latent_dim" in model_kwargs else -1
    lambda_center = model_kwargs["lambda_center"] if "lambda_center" in model_kwargs else 0.0

    metrics = evaluate(model, test_data, device, output_dir)

    # Save metrics and loss to a text file
    with open(f"{output_dir}/test_summary.txt", "w") as f:
        f.write(f"Test Metrics:\n")
        # for key, value in metrics.items():
        #     f.write(f"  - {key}:\n{value}\n")  
        f.write(f"  - balanced_accuracy: {metrics["balanced_accuracy"]}\n")
        f.write(f"  - detailed_report:\n{metrics["detailed_report"]}\n")
        
        # f.write(f"Test Loss: {loss:.4f}\n")

    print(f"Saved test results under: '{f'{output_dir}/test_summary.txt'}'")









# -------------
# > Execution <
# -------------

def main(config):

    if config.model.name == "random" and (config.model.check_point_path is None or config.model.check_point_path == "None"):
        is_random_model = True
        origin_config = config
        checkpoint_path=None
    else:
        is_random_model = False
        checkpoint_path = os.path.join(config.model.check_point_path, "best_model.pth")
        origin_config_path = os.path.join(config.model.check_point_path, "config.yaml")
        origin_config = load_config(origin_config_path)

    # extract configs
    data_path = config.data.path
    batch_size = config.test.batch_size
    output_dir = config.test.output_dir
    model_name = origin_config.model.name
    model_kwargs = origin_config.model.kwargs
    exp_name = origin_config.train.exp_name

    # create exp output folder
    if is_random_model:
        exp_name = "./output/random"
    else:
        exp_name = os.path.split(os.path.dirname(checkpoint_path))[-1]  # FIXME: top dir ok? get exp name in this way?
    output_dir = f"{output_dir}/{exp_name}"
    os.makedirs(output_dir, exist_ok=True)
    # shutil.rmtree(output_dir)
    # os.makedirs(output_dir, exist_ok=True)
    
    test(
        model_name=model_name,
        model_kwargs=model_kwargs,
        checkpoint_path=checkpoint_path,
        data_path=data_path,
        batch_size=batch_size,
        output_dir=output_dir
    )
    







