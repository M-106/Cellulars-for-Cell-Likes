# -----------
# > Imports <
# -----------
import torch



# -----------------
# > Model Loading <
# -----------------
def get_model(model_name, num_classes, checkpoint_path=None, **kwargs):
    """
    Load a model based on the provided model name.

    Args:
        model_name (str): Name of the model to load.
        num_classes (int): Number of output classes for the model.

    Returns:
        torch.nn.Module: The loaded model.
    """
    # if model_name.lower() == "resnet18":
    #     from torchvision.models import resnet18
    #     model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    #     model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    # elif model_name.lower() == "resnet50":
    #     from torchvision.models import resnet50
    #     model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    #     model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    if model_name.lower() == "standard_nca":
        from cfc.model.neural_cellular_automata import NeuralCellularAutomata
        model = NeuralCellularAutomata(input_channels=3, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Model {model_name} not supported.")

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))

    return model






