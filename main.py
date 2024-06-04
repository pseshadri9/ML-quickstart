import yaml
import os
import json
import sys
import ast
import datetime

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.utilities.model_summary import ModelSummary

import torch

try:
    from pytorch_lightning.utilities.seed import seed_everything
except:
    from torch import manual_seed as seed_everything

from logging_debug import manifestHandler, CONFIG, EVAL, MODEL_PATH

CONFIG_PATH = 'config/base.yml'

def deepcopy(d):
    print(d)
    return ast.literal_eval(json.dumps(d))

def main(config):
    params = deepcopy(config)

    seed_everything(config["seed"])

    TEST_DIR = config['data_params'].pop('data_path')

    print('USING DIRECTORY: ', TEST_DIR)
    print('USING CONFIG: ', params)

    '''
    Load Dataset: 
    ex: Dataset.from_dirs(TEST_DIR, **config['data_params'])
    '''
    X = None 
    '''
    Load Dataloader: 
    ex: DataModule(X, **config['dataloader_params'])
    '''
    data = None

    '''
    Load Model: 
    ex: Model(**config['model_params'])
    '''
    model = None


    tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                            name=config['logging_params']['name'] + f': {exp_name}',)

    runner = Trainer(logger=tb_logger,
                    callbacks=[
                        LearningRateMonitor(),
                        ModelCheckpoint(save_top_k=1, 
                                        dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                        monitor= "val_loss")
                    ],
                    val_check_interval=0.2,
                    #strategy=DDPStrategy(find_unused_parameters=False),
                    **config['trainer_params'])
    runner.fit(model, data.train_dataloader(), data.val_dataloader())

    eval_dict = dict()
    m = runner.test(ckpt_path=runner.checkpoint_callback.best_model_path, dataloaders=data.test_dataloader())
    eval_dict = m[0]
    return {EVAL:eval_dict, CONFIG: params, MODEL_PATH: runner.checkpoint_callback.best_model_path}

if __name__ == '__main__':
    if len(sys.argv) > 1:
        CONFIG_PATH = sys.argv[1]

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    if config['dev']:
        print('RUNNING IN DEVELOPMENT MODE')
        exp_name = 'dev'
    else:
        print("Name of the current run (press ENTER for default):")
        exp_name = input()

    try:
        os.mkdir(config['logging_params']['manifest_path'])
    except FileExistsError:
        pass

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.cuda.empty_cache()
       
    start_time, start_datetime = datetime.datetime.now().strftime(format="%d_%m_%Y__%H_%M_%S"), datetime.datetime.now()
    print(f'RUN STARTED AT {start_time} with CONFIG:\n\n{config}\n')

    #run training instance
    results = main(config)

    #Create output manifest about training run
    manifest = manifestHandler(save_path=config['logging_params']['manifest_path'], name=exp_name,
                                **results)
    #Save to file
    manifest.save()

    print('TIME ELAPSED: ',round((datetime.datetime.now() - start_datetime).total_seconds() / 3600, 2), 'HOURS')


    
    
    