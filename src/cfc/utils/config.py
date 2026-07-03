# ----------
# > Import <
# ----------
from typing import Union
from pydantic import BaseModel
import yaml



# -----------------------------
# > Config Classes and Helper <
# -----------------------------
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



class ModelConfig(BaseModel):
    name: str
    check_point_path: Union[str, None]



class DataConfig(BaseModel):
    name: str
    path: str



class Config(BaseModel):
    mode: str
    train: TrainConfig
    model: ModelConfig
    data: DataConfig
















