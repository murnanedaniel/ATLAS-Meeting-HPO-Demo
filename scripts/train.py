import sys
import argparse
import yaml
import time

import torch
import numpy
import random
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

sys.path.append("../")
from lightning_modules.Models.interaction_gnn import InteractionGNN
import wandb

def main():
    print("Running main")
    print(time.ctime())
        
    default_config_path = "default_config.yaml"
    
    with open(default_config_path) as file:
        default_configs = yaml.load(file, Loader=yaml.FullLoader)
        
    wandb.init(config = default_configs, project=default_configs["project"])
    config = wandb.config
    
    print("Initialising model")
    print(time.ctime())
    model = InteractionGNN(dict(config))
    logger = WandbLogger(save_dir=default_configs["artifacts"])
    logger.watch(model, log="all")
    
    trainer = Trainer(gpus=1, max_epochs=default_configs["max_epochs"], logger=logger)
    trainer.fit(model)
    
if __name__ == "__main__":
    
    main()