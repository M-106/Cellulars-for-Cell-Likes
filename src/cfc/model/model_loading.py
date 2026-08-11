# -----------
# > Imports <
# -----------
import os

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
    width = height = 224
    # not 600 x 400

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
        from cfc.model.autoencoder import ConvAE

        nca_latent_path = kwargs["nca_latent_path"] if "nca_latent_path" in kwargs.keys() else None
        if nca_latent_path is not None and nca_latent_path not in ["None", "none"]:
            train_state = torch.load(os.path.join(nca_latent_path, "last_train_state.pth"), map_location="cpu")
            latent_kwargs = train_state["config"].model.kwargs

            latent_model = ConvAE(num_classes=num_classes, input_width=width, input_height=height, **latent_kwargs)  # also add if nca is used??
            latent_model.load_state_dict(torch.load(os.path.join(nca_latent_path, "best_model.pth"), map_location="cpu"))
            kwargs["latent_model"] = latent_model

        model = NeuralCellularAutomata(input_channels=3, num_classes=num_classes, **kwargs)

    elif model_name.lower() == "convvae":
        from cfc.model.varientional_autoencoder import ConvVAE
        model = ConvVAE(num_classes=num_classes, input_width=width, input_height=height, **kwargs)

    elif model_name.lower() == "convae":
            from cfc.model.autoencoder import ConvAE
            model = ConvAE(num_classes=num_classes, input_width=width, input_height=height, **kwargs)

    elif model_name.lower() == "alexnet":
            from cfc.model.alexnet import AlexNet
            model = AlexNet(input_channels=3, input_width=width, input_height=height, num_classes=num_classes, **kwargs)

    elif model_name.lower() == "mixed_alexnet_nca_feature_booster_net":
            from cfc.model.alexnet import AlexNetNCAFeatureBoosterNet
            model = AlexNetNCAFeatureBoosterNet(
                input_channels=3, 
                input_width=width, 
                input_height=height, 
                num_classes=num_classes, 
                **kwargs
            )

    elif model_name.lower() == "efficientnet":
        from cfc.model.efficientnet import EfficientNetClassifier
        model = EfficientNetClassifier(
            input_channels=3, 
            input_width=width, 
            input_height=height, 
            num_classes=num_classes, 
            **kwargs
        )

    elif model_name.lower() == "efficientnet_nca":
        from cfc.model.efficientnet import EfficientNetNCAClassifier
        model = EfficientNetNCAClassifier(
            input_channels=3, 
            input_width=width, 
            input_height=height, 
            num_classes=num_classes, 
            **kwargs
        )

    elif model_name.lower() == "random":
        from cfc.model.random import RandomClassifier
        model = RandomClassifier(num_classes=num_classes, **kwargs)
    
    else:
        raise ValueError(f"Model {model_name} not supported.")

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))

    return model




def is_ae(model):
     from cfc.model.autoencoder import ConvAE
     if isinstance(model, ConvAE):
          return True
     else:
          False



