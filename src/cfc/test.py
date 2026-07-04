# -------------
# > Imports <
# -------------
import os
import shutil

import torch

from cfc.utils.data import get_data
from cfc.model.model_loading import get_model
from cfc.utils.metrics import calculate_isic_metrics



# -------------
# > Test Loop <
# -------------
def evaluate(model, data_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    weights = []

    with torch.no_grad():
        for inputs, labels, score_weight, validation_weight in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            weights.extend(score_weight)  # Assuming equal weights for each sample

    metrics = calculate_isic_metrics(all_predictions, all_labels, weights)
    return metrics



def test(model_name, checkpoint_path, data_path, batch_size, output_dir):
    """
    Test a model on a specified dataset.

    Args:
        model_name (str): Name of the model to test.
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
        shuffle=False
    )

    # get model
    model = get_model(model_name, num_classes=len(test_data.dataset.class_names), checkpoint_path=checkpoint_path)
    model.to(device)

    metrics = evaluate(model, test_data, device)

    # Save metrics and loss to a text file
    with open(f"{output_dir}/test_summary.txt", "w") as f:
        f.write(f"Test Metrics:\n")
        for key, value in metrics.items():
            f.write(f"  - {key}: {value}\n")    
        
        # f.write(f"Test Loss: {loss:.4f}\n")









# -------------
# > Execution <
# -------------

def main(config):

    # extract configs
    model_name = config.model.name
    data_path = config.data.path
    batch_size = config.test.batch_size
    output_dir = config.test.output_dir
    checkpoint_path = config.model.check_point_path

    # create exp output folder
    exp_name = os.path.split(os.path.dirname(checkpoint_path))[-1]  # FIXME: top dir ok? get exp name in this way?
    output_dir = f"{output_dir}/test_{exp_name}"
    os.makedirs(output_dir, exist_ok=True)
    shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    test(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        data_path=data_path,
        batch_size=batch_size,
        output_dir=output_dir
    )
    







