# -----------
# > Imports <
# -----------
import argparse

from cfc.utils.config import load_config



# -------------
# > Execution <
# -------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, default="./configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    print("Configuration:")
    print(config)
    print("\n")
    

    if config.mode == "train":
        from cfc.train import main as train_main
        train_main(config)

    elif config.mode == "test":
        from cfc.test import main as test_main
        test_main(config)
        
    else:
        raise ValueError(f"'{config.mode}' is not an available mode for mcrlab.")
    







