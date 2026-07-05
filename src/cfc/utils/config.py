# ----------
# > Import <
# ----------
import json
from typing import Union
from pydantic import BaseModel
import yaml



# -----------------------------
# > Config Classes and Helper <
# -----------------------------
def config_as_str(config):
    # extract configs as json string
    config_dict = config.model_dump()
    json_str = json.dumps(config_dict, indent=4, default=str)
    return json_str

def log_config_to_tensorboard(writer, tag, config):
    # extract configs as json string
    json_str = config_as_str(config)
    
    # integrate as markdown text
    markdown_text = f"### Experiment Config\n```json\n{json_str}\n```"

    # log in writer
    writer.add_text(tag, markdown_text, global_step=0)

def load_config(path):
        # auto relative path conversion -> relative to mcr-lab top folder
        # if path.startswith("./"):
        #     path = path.replace("./", "../../../")
        # elif path.startswith("/"):
        #     path = "../../.." + path

        with open(path) as file_:
            data = yaml.safe_load(file_)
        return Config(**data)

def save_config(config, path):
    with open(path, "w") as file_:
        yaml.dump(config.dict(), file_)



class TrainConfig(BaseModel):
    num_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    criterion: str
    optimizer: str
    scheduler: Union[str, None]
    output_dir: str
    exp_name: str
    used_train_samples: int
    used_val_samples: int



class TestConfig(BaseModel):
    batch_size: int
    output_dir: str

# class ModelKwargs(BaseModel):
#     hidden_channels: int
#     steps: int
#     update_blocks: int
#     update_blocks_activation_kernel_size: int
#     update_blocks_activation_kernel_size_2: int
#     update_blocks_activation: str
#     final_update_block_activation_kernel_size: int
#     final_update_block_activation: str
#     perception_filter: str
#     dropout: float

class ModelConfig(BaseModel):
    name: str
    check_point_path: Union[str, None]
    kwargs: dict
    # kwargs: ModelKwargs


class DataConfig(BaseModel):
    path: str



class Config(BaseModel):
    mode: str
    train: TrainConfig
    test: TestConfig
    model: ModelConfig
    data: DataConfig
















